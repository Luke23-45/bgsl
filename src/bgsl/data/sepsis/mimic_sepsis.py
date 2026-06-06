"""
datasets/mimic_sepsis.py
-------------------------
MIMIC-IV Sepsis cohort loader — interface stub for external validation (Month 4).

Architecture mirrors physionet2019.py exactly so experiments/train.py can
swap datasets without changing any model or loss code.

Status: Interface defined, implementation pending MIMIC-IV access.

Two supported routes:
    Route A — MIMIC-Sepsis benchmark pipeline
        Huang Y et al. "MIMIC-Sepsis: A Curated Benchmark for Modeling and
        Learning from Sepsis Trajectories in the ICU." IEEE BHI 2025 / arXiv:2510.24500.
        Pre-processed trajectories with standardized onset labels.

    Route B — Direct MIMIC-IV
        Build from scratch using the MIMIC-IV clinical tables.
        Requires physionet.org credentialed access.

Reference: plan.md §5.1 (Dataset 2), §18 (Month 4)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

__all__ = ["MIMICSepsisDataset"]

logger = logging.getLogger("bgsl.mimic_sepsis")


class MIMICSepsisDataset(Dataset):
    """
    MIMIC-IV Sepsis cohort dataset for BGSL external validation.

    Interface mirrors PhysioNet2019Dataset for drop-in compatibility.

    Parameters
    ----------
    data_dir : str
        Root directory containing pre-processed MIMIC-Sepsis LMDB data.
    split : str
        "train", "val", or "test".
    horizon_hours : int
        Prediction horizon H in hours. Default 6.
    tau : float
        Soft target temperature. Default 2.0.
    normalizer : ClinicalNormalizer or None
        If provided, normalization applied in __getitem__.

    Raises
    ------
    NotImplementedError
        Until the MIMIC-IV pipeline is implemented (Month 4).
    """

    def __init__(
        self,
        data_dir: str = "data/mimic_ready",
        split: str = "train",
        horizon_hours: int = 6,
        tau: float = 2.0,
        normalizer=None,
        min_length: int = 8,
    ) -> None:
        self.split = split
        self.horizon = horizon_hours
        self.tau = tau

        # Route A: MIMIC-Sepsis benchmark (preferred)
        self._route = self._detect_route(data_dir)

        if self._route == "stub":
            logger.warning(
                "MIMICSepsisDataset: no MIMIC data found at '%s'. "
                "This dataset is a stub — implement after Month 3 PhysioNet runs. "
                "See plan.md §5.1 for the two supported routes.",
                data_dir,
            )
            self._episodes: List[Dict] = []
        else:
            raise NotImplementedError(
                "MIMIC-Sepsis loading not yet implemented. "
                "Implement after PhysioNet external validation is complete (Month 4)."
            )

    @staticmethod
    def _detect_route(data_dir: str) -> str:
        from pathlib import Path
        p = Path(data_dir)
        if not p.exists():
            return "stub"
        # Check for MIMIC-Sepsis benchmark structure
        if (p / "train" / "data.lmdb").exists():
            return "mimic_sepsis_benchmark"
        # Check for raw MIMIC-IV
        if (p / "mimiciv_icu").exists():
            return "raw_mimic_iv"
        return "stub"

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("MIMICSepsisDataset not yet implemented.")

    @property
    def sepsis_rate(self) -> float:
        n_pos = sum(1 for e in self._episodes if e.get("is_sepsis", False))
        return n_pos / max(len(self._episodes), 1)
