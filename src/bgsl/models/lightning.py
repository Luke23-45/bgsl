"""
lightning.py
------------
PyTorch Lightning Module for BGSL risk predictors.
Updated for LightningCLI strict typing.
"""

from typing import Any, Dict, Optional, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from bgsl.core.losses import TLSLoss, BGSLLoss
from bgsl.core.metrics import BGSLMetrics, PatientPrediction

class BGSLLightningModule(pl.LightningModule):
    """
    LightningModule for Biological Gradient Supervised Learning.
    Utilizes LightningCLI dependency injection for 'model' and 'loss_fn'.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        horizon_hours: int = 6,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        lr_scheduler: Literal["cosine", "none"] = "cosine",
    ) -> None:
        super().__init__()
        # Ignore sub-modules in hparams to avoid deepcopy issues, 
        # LightningCLI handles their instantiation.
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        
        self.model = model
        self.loss_fn = loss_fn

        # Metrics (computed on valid masked timesteps)
        metrics = MetricCollection({
            "auroc": BinaryAUROC(),
            "auprc": BinaryAveragePrecision(),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Trajectory-level metrics for testing
        self.trajectory_metrics = BGSLMetrics(threshold=0.5, sustained_k=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, seq_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass returns raw logits [B, T]."""
        return self.model(x, mask=mask, seq_lens=seq_lens)

    def _shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, phase: str) -> torch.Tensor:
        """Handles forward pass, loss computation, and metric updates."""
        x = batch["vitals"]
        m = batch["masks"]
        lens = batch["seq_len"]

        # Forward pass
        logits = self(x, mask=m, seq_lens=lens)
        
        # Loss computation
        if isinstance(self.loss_fn, TLSLoss):
            targets = self.loss_fn.build_tls_targets(batch["onset_hour"], lens, self.device)
            loss_dict = self.loss_fn(logits, targets, batch["valid_mask"])
        elif isinstance(self.loss_fn, BGSLLoss):
            loss_dict = self.loss_fn(
                logits,
                batch["soft_targets"],   # BGSLLoss uses soft trajectory target
                batch["valid_mask"],     
                vel_targets=batch["vel_targets"],
                acc_targets=batch["accel_targets"],
                vel_mask=batch["vel_mask"],
                acc_mask=batch["acc_mask"],
            )
        else:
            loss_dict = self.loss_fn(
                logits,
                batch["hard_targets"],   # baseline losses use the standard hard early-warning label
                batch["valid_mask"],
                vel_targets=batch["vel_targets"],
                acc_targets=batch["accel_targets"],
                vel_mask=batch["vel_mask"],
                acc_mask=batch["acc_mask"],
            )

        loss = loss_dict["loss"]
        
        # Logging losses
        batch_size = x.shape[0]
        for k, v in loss_dict.items():
            self.log(f"{phase}_{k}", v, on_step=(phase=="train"), on_epoch=True, batch_size=batch_size, prog_bar=(k=="loss"))

        # Update metrics (flattened over sequence using valid mask)
        probs = torch.sigmoid(logits)
        valid_mask = batch["valid_mask"].bool()
        
        if valid_mask.any():
            preds_flat = probs[valid_mask]
            targets_flat = batch["hard_labels"][valid_mask]
            
            if len(torch.unique(targets_flat)) > 1:
                metrics = getattr(self, f"{phase}_metrics")
                metrics.update(preds_flat, targets_flat.long())

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "test")

        # Integrate BGSLMetrics for full trajectory evaluation
        x = batch["vitals"]
        m = batch["masks"]
        lens = batch["seq_len"]
        
        logits = self(x, mask=m, seq_lens=lens)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        for i in range(x.shape[0]):
            T = int(lens[i].item())
            p = PatientPrediction(
                patient_id=batch["patient_id"][i],
                probs=probs[i, :T],
                hard_labels=batch["hard_labels"][i, :T].cpu().numpy(),
                soft_targets=batch["soft_targets"][i, :T].cpu().numpy(),
                is_sepsis=bool(batch["is_sepsis"][i].item()),
                onset_hour=int(batch["onset_hour"][i].item()),
                horizon=self.hparams.horizon_hours,
                seq_len=T
            )
            self.trajectory_metrics.add(p)

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), on_epoch=True, prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), on_epoch=True)
        self.test_metrics.reset()
        
        # Compute trajectory-based metrics
        traj_results = self.trajectory_metrics.compute()
        
        # Log to Lightning
        self.log_dict({f"test_{k}": float(v) for k, v in traj_results.items()}, on_epoch=True)
        
        # Reset for next test run
        self.trajectory_metrics.reset()

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
