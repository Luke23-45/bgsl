"""
train/cmapss/module.py
----------------------
CMAPSS-specific LightningModule subclass.
Adapts CMAPSS metric tracking, cycle-based keys, and logic.
"""

from typing import Dict, Literal

import torch
import torch.nn as nn

from bgsl.train.common.module import BaseBGSLLightningModule
from bgsl.core.cmapss.metrics import CMAPSSMetrics, CyclePrediction

__all__ = ["CMAPSSLightningModule"]


class CMAPSSLightningModule(BaseBGSLLightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        loss_fn: nn.Module,
        horizon_cycles: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        lr_scheduler: Literal["cosine", "none"] = "cosine",
        validation_threshold_target: float = 0.80,
    ) -> None:
        super().__init__(
            backbone=backbone,
            loss_fn=loss_fn,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            lr_scheduler=lr_scheduler,
            validation_threshold_target=validation_threshold_target,
        )
        self.save_hyperparameters(ignore=["backbone", "loss_fn"])

        # CMAPSS-specific metrics
        self.validation_threshold_metrics = CMAPSSMetrics(threshold=self.selected_threshold, sustained_k=1)
        self.trajectory_metrics = CMAPSSMetrics(threshold=self.selected_threshold, sustained_k=1)

    def _update_trajectory_metrics(self, batch: Dict[str, torch.Tensor], probs: torch.Tensor, phase: str) -> None:
        tracker = self.validation_threshold_metrics if phase == "val" else self.trajectory_metrics
        
        lens = batch["seq_len"]
        hard_targets = batch["hard_targets"]
        soft_targets = batch.get("soft_targets")
        
        # In CMAPSS, we use failure_cycle instead of onset_hour
        failure_cycle = batch.get("failure_cycle", batch.get("onset_time"))
        is_failure = batch.get("is_failure", batch.get("has_event"))

        W = probs.shape[1]
        probs_cpu = probs.detach().cpu().numpy()

        for i in range(probs.shape[0]):
            T = int(lens[i].item())
            
            # CMAPSS uses left padding, so the valid sequence is at the END of the window
            probs_seq = probs_cpu[i, W - T : W]
            hard_seq = hard_targets[i, W - T : W].cpu().numpy()
            soft_seq = soft_targets[i, W - T : W].cpu().numpy() if soft_targets is not None else None

            tracker.add(
                CyclePrediction(
                    id=str(batch.get("unit_id", batch.get("patient_id", [""] * probs.shape[0]))[i]),
                    probs=probs_seq,
                    hard_labels=hard_seq,
                    soft_targets=soft_seq,
                    has_event=bool(is_failure[i].item()),
                    onset_time=float(failure_cycle[i].item()),
                    horizon=float(self.hparams.horizon_cycles),
                    seq_len=T,
                    time_units_per_step=1.0, # cycles
                )
            )

    def _on_validation_epoch_end_trajectory(self) -> None:
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            self.validation_threshold_metrics.reset()
            return

        if getattr(self.validation_threshold_metrics, "_predictions", None):
            # For CMAPSS, we might select threshold differently, but for now fallback to sensitivity
            self.selected_threshold = self.validation_threshold_metrics.select_threshold_fixed_sensitivity(
                target_sensitivity=self.validation_threshold_target
            )
            self.trajectory_metrics.threshold = self.selected_threshold
            self.validation_threshold_metrics.reset()

    def _on_test_epoch_end_trajectory(self) -> None:
        self.trajectory_metrics.threshold = self.selected_threshold
        traj_results = self.trajectory_metrics.compute()
        traj_results["selected_threshold"] = float(self.selected_threshold)
        
        self.log_dict({f"test_{k}": float(v) for k, v in traj_results.items()}, on_epoch=True)
        self.trajectory_metrics.reset()
