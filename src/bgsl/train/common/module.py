"""
train/common/module.py
----------------------
Base PyTorch Lightning Module for BGSL risk predictors.
Handles forward pass, standard metrics, loss computation, and optimization.
Domain-specific metrics are delegated to subclasses.
"""

from typing import Any, Dict, Optional, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from bgsl.core.common.losses import TLSLoss, BGSLLoss
from bgsl.core.common.targets import BaseSoftOnsetTarget

__all__ = ["BaseBGSLLightningModule"]


class BaseBGSLLightningModule(pl.LightningModule):
    """
    Base LightningModule for Biological Gradient Supervised Learning.
    Accepts 'backbone' and 'loss_fn' via dependency injection.
    """

    def __init__(
        self,
        backbone: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        lr_scheduler: Literal["cosine", "none"] = "cosine",
        validation_threshold_target: float = 0.80,
        hypernetwork: Optional[nn.Module] = None,
        horizon: Optional[float] = None,
        tau: float = 2.0,
        kappa: float = 1.0,
    ) -> None:
        super().__init__()
        # Ignore sub-modules in hparams to avoid deepcopy issues
        self.save_hyperparameters(ignore=["backbone", "loss_fn", "hypernetwork"])
        
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.hypernetwork = hypernetwork
        self.validation_threshold_target = validation_threshold_target
        self.register_buffer("selected_threshold", torch.tensor(0.5))

        # Target builder for online (hypernetwork) target construction
        # Infer horizon from loss_fn if available, fall back to passed value
        h = horizon
        if h is None and isinstance(loss_fn, BGSLLoss):
            h = loss_fn.H
        self._target_builder = (
            BaseSoftOnsetTarget(horizon=h or 6.0, tau=tau, kappa=kappa)
            if hypernetwork is not None else None
        )

        # Standard step-level metrics (computed on valid masked timesteps)
        metrics = MetricCollection({
            "auroc": BinaryAUROC(),
            "auprc": BinaryAveragePrecision(),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass returns raw logits [B, T]."""
        return self.backbone(x, mask=mask, seq_lens=seq_lens)

    def _shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, phase: str) -> torch.Tensor:
        """Handles forward pass, loss computation, and step metric updates."""
        # Generic batch keys
        x = batch["vitals"]
        m = batch.get("masks", None)
        lens = batch["seq_len"]

        # Forward pass
        logits = self(x, mask=m, seq_lens=lens)
        
        # Loss computation
        if isinstance(self.loss_fn, TLSLoss):
            # Build the correct TLS linear-ramp targets on-the-fly from the event time.
            # CMAPSS uses "failure_cycle"; Sepsis uses "onset_hour". Fall back gracefully.
            event_times = batch.get("failure_cycle", batch.get("onset_hour"))
            if event_times is None:
                raise KeyError(
                    "TLSLoss requires 'failure_cycle' (CMAPSS) or 'onset_hour' (Sepsis) in batch."
                )
            tls_targets = self.loss_fn.build_tls_targets(
                onset_hours=event_times,           # [B]  event time for each sample
                seq_lengths=batch["seq_len"],      # [B]  valid (non-padded) lengths
                device=logits.device,
            )
            loss_dict = self.loss_fn(logits, tls_targets, batch["valid_mask"])
        elif isinstance(self.loss_fn, BGSLLoss):
            # Map domain-specific keys to generic names used by the loss.
            # Sepsis → onset_hour / is_sepsis;  CMAPSS → failure_cycle / is_failure.
            onset_times = batch.get("onset_hour", batch.get("failure_cycle"))
            is_positive = batch.get("is_sepsis", batch.get("is_failure"))

            # Online target construction via HyperNetwork (EBGSL path)
            # Overrides pre-computed targets from the dataset.
            static = batch.get("static_covariates")
            if self.hypernetwork is not None and static is not None and onset_times is not None:
                delta_mu, sigma, kappa_shape = self.hypernetwork(static)
                rebuilt = self._target_builder(
                    onset_times=onset_times,
                    seq_lengths=batch["seq_len"],
                    device=logits.device,
                    delta_mu=delta_mu,
                    sigma=sigma,
                    kappa_shape=kappa_shape,
                )
                soft_tgts = rebuilt["soft_target"]
                vel_tgts = rebuilt["velocity_target"]
                acc_tgts = rebuilt["accel_target"]
                vmask = rebuilt["vel_mask"]
                amask = rebuilt["acc_mask"]
            else:
                soft_tgts = batch["soft_targets"]
                vel_tgts = batch.get("vel_targets")
                acc_tgts = batch.get("accel_targets")
                vmask = batch.get("vel_mask")
                amask = batch.get("acc_mask")

            loss_dict = self.loss_fn(
                logits,
                soft_tgts,
                batch["valid_mask"],
                vel_targets=vel_tgts,
                acc_targets=acc_tgts,
                vel_mask=vmask,
                acc_mask=amask,
                onset_times=onset_times,
                is_positive=is_positive,
                seq_lengths=batch.get("seq_len"),
            )
        else:
            # Baseline losses
            loss_dict = self.loss_fn(
                logits,
                batch["hard_targets"],
                batch["valid_mask"],
                vel_targets=batch.get("vel_targets"),
                acc_targets=batch.get("accel_targets"),
                vel_mask=batch.get("vel_mask"),
                acc_mask=batch.get("acc_mask"),
            )

        loss = loss_dict["loss"]
        
        # Logging losses
        batch_size = x.shape[0]
        for k, v in loss_dict.items():
            self.log(f"{phase}_{k}", v, on_step=(phase=="train"), on_epoch=True, batch_size=batch_size, prog_bar=(k=="loss"))

        # Update step metrics
        probs = torch.sigmoid(logits)
        valid_mask = batch["valid_mask"].bool()
        
        if valid_mask.any():
            preds_flat = probs[valid_mask]
            targets_flat = batch["hard_targets"][valid_mask]
            
            metrics = getattr(self, f"{phase}_metrics")
            metrics.update(preds_flat, targets_flat.long())

        # Delegate trajectory metrics to subclasses
        if phase in {"val", "test"}:
            self._update_trajectory_metrics(batch, probs, phase)

        return loss

    def _update_trajectory_metrics(self, batch: Dict[str, torch.Tensor], probs: torch.Tensor, phase: str) -> None:
        """
        To be implemented by domain-specific subclasses to update TrajectoryMetrics.
        """
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "test")

    def on_train_epoch_end(self) -> None:
        try:
            self.log_dict(self.train_metrics.compute(), on_epoch=True, prog_bar=True)
        except ValueError:
            pass  # No metric updates this epoch (all-same-class batches)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        try:
            self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)
        except ValueError:
            pass  # No metric updates (e.g. sanity check with mono-class batches)
        self.val_metrics.reset()
        self._on_validation_epoch_end_trajectory()

    def _on_validation_epoch_end_trajectory(self) -> None:
        """To be implemented by subclasses to log trajectory metrics and select threshold."""
        pass

    def on_test_epoch_end(self) -> None:
        try:
            self.log_dict(self.test_metrics.compute(), on_epoch=True)
        except ValueError:
            pass  # No metric updates this epoch
        self.test_metrics.reset()
        self._on_test_epoch_end_trajectory()

    def _on_test_epoch_end_trajectory(self) -> None:
        """To be implemented by subclasses to compute and log trajectory metrics."""
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        if self.hparams.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.epochs, eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        return optimizer
