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
    ) -> None:
        super().__init__()
        # Ignore sub-modules in hparams to avoid deepcopy issues
        self.save_hyperparameters(ignore=["backbone", "loss_fn"])
        
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.validation_threshold_target = validation_threshold_target
        self.selected_threshold = 0.5

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
            # TLSLoss expects domain-specific targets to be pre-built or built dynamically
            # Assuming the dataloader provides soft_targets or we fall back.
            # TLSLoss usually builds its own, but we'll assume it's pre-computed in batch["tls_targets"]
            # or we need the subclass to adapt it. For simplicity, if TLSLoss, use soft_targets.
            loss_dict = self.loss_fn(logits, batch["soft_targets"], batch["valid_mask"])
        elif isinstance(self.loss_fn, BGSLLoss):
            loss_dict = self.loss_fn(
                logits,
                batch["soft_targets"],
                batch["valid_mask"],     
                vel_targets=batch.get("vel_targets"),
                acc_targets=batch.get("accel_targets"),
                vel_mask=batch.get("vel_mask"),
                acc_mask=batch.get("acc_mask"),
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
            
            if len(torch.unique(targets_flat)) > 1:
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
        self.log_dict(self.train_metrics.compute(), on_epoch=True, prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)
        self.val_metrics.reset()
        self._on_validation_epoch_end_trajectory()

    def _on_validation_epoch_end_trajectory(self) -> None:
        """To be implemented by subclasses to log trajectory metrics and select threshold."""
        pass

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), on_epoch=True)
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
