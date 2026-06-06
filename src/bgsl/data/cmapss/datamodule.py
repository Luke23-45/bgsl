"""
data/cmapss/datamodule.py
--------------------------
PyTorch Lightning DataModule for the CMAPSS turbofan degradation dataset.

Supports all four CMAPSS subsets (FD001–FD004) with a configurable
horizon and soft-target construction via CycleOnsetTarget.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from bgsl.data.cmapss.dataset import CMAPSSDataset

__all__ = ["CMAPSSDataModule"]


def collate_cmapss(batch: List[Dict]) -> Dict:
    """
    Collate a list of fixed-length CMAPSSDataset items into a batch dict.

    All items in a CMAPSS batch have the same window_size W, so no
    padding is required. The collator simply stacks tensors and
    gathers string unit_ids into a list.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    # Time-indexed tensor keys
    time_keys = [
        "vitals", "masks", "hard_targets", "soft_targets",
        "vel_targets", "accel_targets", "valid_mask", "vel_mask", "acc_mask",
    ]
    # Scalar tensor keys
    scalar_keys = ["failure_cycle", "is_failure", "seq_len"]

    out: Dict = {}
    for key in time_keys:
        out[key] = torch.stack([b[key] for b in batch])   # [B, W, ...]

    for key in scalar_keys:
        out[key] = torch.stack([b[key] for b in batch])   # [B]

    out["unit_id"] = [b["unit_id"] for b in batch]        # list[str]

    return out


class CMAPSSDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the CMAPSS turbofan degradation dataset.

    Expects preprocessed .pt split files at:
        {data_dir}/train/{subset}.pt
        {data_dir}/val/{subset}.pt
        {data_dir}/test/{subset}.pt

    Run ``python -m bgsl.data.cmapss.preprocessing --config ...`` first.

    Parameters
    ----------
    data_dir : str
        Root directory of processed CMAPSS files.
    subset : str
        One of "FD001", "FD002", "FD003", "FD004". Default "FD001".
    horizon_cycles : int
        Prediction horizon H in cycles. Default 30.
    tau : float
        Soft target temperature for CycleOnsetTarget. Default 2.0.
    batch_size : int
        DataLoader batch size. Default 64.
    num_workers : int
        DataLoader workers. Default os.cpu_count().
    pin_memory : bool
        DataLoader pin_memory. Default True if CUDA available.
    """

    def __init__(
        self,
        data_dir: str = "datasets/processed/cmapss",
        subset: str = "FD001",
        horizon_cycles: int = 30,
        tau: float = 2.0,
        batch_size: int = 64,
        num_workers: int = os.cpu_count() or 2,
        pin_memory: bool = torch.cuda.is_available(),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_ds: Optional[CMAPSSDataset] = None
        self.val_ds:   Optional[CMAPSSDataset] = None
        self.test_ds:  Optional[CMAPSSDataset] = None

    @property
    def feature_dim(self) -> int:
        if not hasattr(self, "_feature_dim"):
            path = Path(self.hparams.data_dir) / "train" / f"{self.hparams.subset}.pt"
            if not path.exists():
                return 14  # Fallback
            data = torch.load(path, map_location="cpu", weights_only=False)
            self._feature_dim = data["inputs"].shape[-1]
            del data
        return self._feature_dim

    def _make_dataset(self, split: str) -> CMAPSSDataset:
        path = Path(self.hparams.data_dir) / split / f"{self.hparams.subset}.pt"
        return CMAPSSDataset(
            data_path=str(path),
            horizon_cycles=self.hparams.horizon_cycles,
            tau=self.hparams.tau,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_ds = self._make_dataset("train")
            self.val_ds   = self._make_dataset("val")

        if stage in ("test", None):
            self.test_ds = self._make_dataset("test")

        if stage in ("predict", None):
            # Reuse test split for prediction
            if self.test_ds is None:
                self.test_ds = self._make_dataset("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_cmapss,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_cmapss,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_cmapss,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_cmapss,
            persistent_workers=self.hparams.num_workers > 0,
        )
