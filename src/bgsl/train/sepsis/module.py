"""
train/sepsis/module.py
----------------------
Sepsis-specific LightningModule subclass.
Adapts Sepsis metric tracking, batch keys, and threshold logic.
"""

from pathlib import Path
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

from bgsl.train.common.module import BaseBGSLLightningModule
from bgsl.core.sepsis.metrics import SepsisMetrics, PatientPrediction

__all__ = ["SepsisLightningModule"]


class SepsisLightningModule(BaseBGSLLightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        loss_fn: nn.Module,
        horizon_hours: int = 6,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        lr_scheduler: Literal["cosine", "none"] = "cosine",
        validation_threshold_target: float = 0.80,
        hypernetwork: Optional[nn.Module] = None,
        tau: float = 2.0,
        kappa: float = 1.0,
    ) -> None:
        super().__init__(
            backbone=backbone,
            loss_fn=loss_fn,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            lr_scheduler=lr_scheduler,
            validation_threshold_target=validation_threshold_target,
            hypernetwork=hypernetwork,
            horizon=float(horizon_hours),
            tau=tau,
            kappa=kappa,
        )
        self.save_hyperparameters(ignore=["backbone", "loss_fn", "hypernetwork"])

        # Sepsis-specific metrics
        self.validation_threshold_metrics = SepsisMetrics(threshold=float(self.selected_threshold.item()), sustained_k=1)
        self.trajectory_metrics = SepsisMetrics(threshold=float(self.selected_threshold.item()), sustained_k=1)

    def _update_trajectory_metrics(self, batch: Dict[str, torch.Tensor], probs: torch.Tensor, phase: str) -> None:
        tracker = self.validation_threshold_metrics if phase == "val" else self.trajectory_metrics
        
        lens = batch["seq_len"]
        hard_targets = batch["hard_targets"]
        soft_targets = batch.get("soft_targets")
        onset_hour = batch["onset_hour"]
        is_sepsis = batch["is_sepsis"]

        probs_cpu = probs.detach().cpu().numpy()

        for i in range(probs.shape[0]):
            T = int(lens[i].item())
            tracker.add(
                PatientPrediction(
                    id=batch["patient_id"][i],
                    probs=probs_cpu[i, :T],
                    hard_labels=hard_targets[i, :T].cpu().numpy(),
                    soft_targets=soft_targets[i, :T].cpu().numpy() if soft_targets is not None else None,
                    has_event=bool(is_sepsis[i].item()),
                    onset_time=float(onset_hour[i].item()),
                    horizon=float(self.hparams.horizon_hours),
                    seq_len=T,
                    time_units_per_step=1.0, # hours
                )
            )

    def _on_validation_epoch_end_trajectory(self) -> None:
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            self.validation_threshold_metrics.reset()
            return

        # Sepsis threshold logic (e.g. 80% sensitivity)
        if getattr(self.validation_threshold_metrics, "_predictions", None):
            self.selected_threshold.fill_(self.validation_threshold_metrics.select_threshold_fixed_sensitivity(
                target_sensitivity=self.validation_threshold_target
            ))
            self.trajectory_metrics.threshold = float(self.selected_threshold.item())
            self.validation_threshold_metrics.reset()

    def _on_test_epoch_end_trajectory(self) -> None:
        self.trajectory_metrics.threshold = float(self.selected_threshold.item())
        traj_results = self.trajectory_metrics.compute()
        traj_results["selected_threshold"] = float(self.selected_threshold.item())

        # Save predictions for visualization
        if self.trainer is not None:
            import pickle
            pred_dir = Path(self.trainer.default_root_dir) / "artifacts" / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            with open(pred_dir / "test_predictions.pkl", "wb") as f:
                pickle.dump(self.trajectory_metrics._predictions, f)

        # Log to Lightning
        self.log_dict({f"test_{k}": float(v) for k, v in traj_results.items()}, on_epoch=True)
        self.trajectory_metrics.reset()
