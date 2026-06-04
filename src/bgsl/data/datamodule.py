"""
datamodule.py
-------------
PyTorch Lightning DataModule for BGSL datasets.
Updated for strict typing with LightningCLI.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from bgsl.data.physionet2019 import PhysioNet2019Dataset, collate_trajectories, make_weighted_sampler
from bgsl.data.transforms import ClinicalNormalizer


class PhysioNetDataModule(pl.LightningDataModule):
    """
    LightningDataModule for PhysioNet 2019 Sepsis Challenge.
    Handles dataset splits, normalization calibration, and dataloaders.
    """

    def __init__(
        self,
        data_dir: str = "data/ready",
        horizon_hours: int = 6,
        tau: float = 2.0,
        min_length: int = 8,
        batch_size: int = 16,
        num_workers: int = os.cpu_count() or 2,
        pin_memory: bool = torch.cuda.is_available(),
        pos_weight_sampler: float = 1.0,
        enable_normalizer: bool = True,
        normalizer_epsilon: float = 1e-6,
        normalizer_safety_margin: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.normalizer = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Calibrate normalizer on train index stats
        if self.hparams.enable_normalizer:
            self.normalizer = ClinicalNormalizer(
                epsilon=self.hparams.normalizer_epsilon,
                safety_margin=self.hparams.normalizer_safety_margin,
            )
            idx_path = Path(self.hparams.data_dir) / "train_index.json"
            if idx_path.exists():
                self.normalizer.calibrate_from_stats(idx_path)
            
        kwargs = {
            "data_dir": self.hparams.data_dir,
            "horizon_hours": self.hparams.horizon_hours,
            "tau": self.hparams.tau,
            "normalizer": self.normalizer,
            "min_length": self.hparams.min_length,
        }

        if stage == "fit" or stage is None:
            self.train_ds = PhysioNet2019Dataset(split="train", **kwargs)
            self.val_ds = PhysioNet2019Dataset(split="val", **kwargs)
            
        if stage == "test" or stage is None:
            self.test_ds = PhysioNet2019Dataset(split="test", **kwargs)

    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        if self.hparams.pos_weight_sampler > 1.0:
            sampler = make_weighted_sampler(
                self.train_ds,
                pos_weight=self.hparams.pos_weight_sampler,
            )
            shuffle = False
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_trajectories,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_trajectories,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_trajectories,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
