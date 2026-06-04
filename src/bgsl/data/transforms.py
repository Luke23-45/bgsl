"""
datasets/transforms.py
-----------------------
Clinical normalizer for BGSL — a clean, self-contained implementation.

Isolated from the APEX-MoE project. No circular imports or project-specific
wrappers. Adapted from icu/datasets/normalizer.py (ClinicalNormalizer v8.0)
with the following differences:
    - No RevIN / per-patient mode (kept simple).
    - Single normalize() interface, no forward() alias confusion.
    - Static context handled via the same 28-channel array (no separate path).
    - Simplified __repr__ and no EMA sync code.

Pipeline
--------
1. NaN recovery (last resort — imputation should happen upstream)
2. Physics clamp  (biological hard decks)
3. Log1p for heavy-tailed channels (Lactate, Creatinine, etc.)
4. Robust quantile scaling [P01, P99] → [-1, 1]
5. Leaky clip at ±2.0 (preserves gradient for extreme values)

Reference: Kim et al. RevIN (ICLR 2022) for instance-norm context.
           PhysioNet 2019 data distribution for physics bounds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

__all__ = ["ClinicalNormalizer", "CANONICAL_COLUMNS", "PHYSICS_BOUNDS", "LOG_CHANNELS"]

logger = logging.getLogger("bgsl.normalizer")

# ---------------------------------------------------------------------------
# Canonical feature spec (Clinical 28 — PhysioNet 2019 ordering)
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS: List[str] = [
    # Group A: Hemodynamic (0–6)
    "HR", "O2Sat", "SBP", "DBP", "MAP", "Resp", "Temp",
    # Group B: Sepsis Drivers / Labs (7–17)
    "Lactate", "Creatinine", "Bilirubin", "Platelets", "WBC",
    "pH", "HCO3", "BUN", "Glucose", "Hgb", "Potassium",
    # Group C: Electrolytes & Support (18–21)
    "Magnesium", "Calcium", "Chloride", "FiO2",
    # Group D: Static Context (22–27)
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]

PHYSICS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HR": (30.0, 180.0),
    "O2Sat": (50.0, 100.0),
    "SBP": (50.0, 220.0),
    "DBP": (30.0, 120.0),
    "MAP": (40.0, 150.0),
    "Resp": (8.0, 45.0),
    "Temp": (32.0, 41.0),
    "Lactate": (0.2, 15.0),
    "Creatinine": (0.2, 10.0),
    "Bilirubin": (0.1, 8.0),
    "Platelets": (10.0, 1000.0),
    "WBC": (1.0, 50.0),
    "pH": (6.8, 7.8),
    "HCO3": (10.0, 50.0),
    "BUN": (2.0, 100.0),
    "Glucose": (20.0, 600.0),
    "Hgb": (5.0, 20.0),
    "Potassium": (2.0, 7.5),
    "Magnesium": (1.0, 5.0),
    "Calcium": (5.0, 15.0),
    "Chloride": (70.0, 130.0),
    "FiO2": (0.21, 1.0),
    "Age": (15.0, 100.0),
    "Gender": (0.0, 1.0),
    "Unit1": (0.0, 1.0),
    "Unit2": (0.0, 1.0),
    "HospAdmTime": (-1000.0, 0.0),
    "ICULOS": (0.0, 2000.0),
}

# Heavy-tailed labs requiring log1p transform
LOG_CHANNELS = {"Lactate", "Creatinine", "Bilirubin", "WBC", "BUN", "Glucose", "Platelets"}


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class ClinicalNormalizer(nn.Module):
    """
    Physics-aware clinical normalizer for ICU time-series (BGSL edition).

    Parameters
    ----------
    epsilon : float
        Numerical stability constant (FP16-safe default: 1e-6).
    safety_margin : float
        Fraction to expand [P01, P99] range for robustness.
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        safety_margin: float = 0.05,
    ) -> None:
        super().__init__()
        C = len(CANONICAL_COLUMNS)
        self.epsilon = epsilon
        self.safety_margin = safety_margin

        # Physics bounds [C]
        p_min = [PHYSICS_BOUNDS.get(col, (-1000.0, 1000.0))[0] for col in CANONICAL_COLUMNS]
        p_max = [PHYSICS_BOUNDS.get(col, (-1000.0, 1000.0))[1] for col in CANONICAL_COLUMNS]
        self.register_buffer("physics_min", torch.tensor(p_min, dtype=torch.float32))
        self.register_buffer("physics_max", torch.tensor(p_max, dtype=torch.float32))

        # Log channel mask [C]
        log_mask = [col in LOG_CHANNELS for col in CANONICAL_COLUMNS]
        self.register_buffer("log_mask", torch.tensor(log_mask, dtype=torch.bool))

        # Statistical bounds (calibrated from training data) [C]
        self.register_buffer("stat_min", torch.zeros(C))
        self.register_buffer("stat_max", torch.ones(C))
        self.register_buffer("is_calibrated", torch.tensor(False, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_from_stats(self, stats_path: Union[str, Path]) -> None:
        """
        Load P01/P99 quantiles from the index JSON produced by the build pipeline.

        Parameters
        ----------
        stats_path : str or Path
            Path to `{split}_index.json` produced by build_physionet_lmdb().
        """
        path = Path(stats_path)
        if not path.exists():
            raise FileNotFoundError(f"Stats file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        stats = meta.get("stats", {})
        if not stats:
            raise ValueError("No 'stats' block found in index JSON.")

        raw_min = stats.get("ts_p01") or stats.get("ts_min")
        raw_max = stats.get("ts_p99") or stats.get("ts_max")
        if raw_min is None or raw_max is None:
            raise ValueError("stats block missing ts_p01/ts_p99.")

        dev = self.physics_min.device
        t_min = torch.tensor(raw_min, dtype=torch.float32, device=dev)
        t_max = torch.tensor(raw_max, dtype=torch.float32, device=dev)

        # Clamp to physics
        t_min = torch.max(t_min, self.physics_min)
        t_max = torch.min(t_max, self.physics_max)

        # Log transform stats (alignment)
        safe_min = torch.log1p(torch.relu(t_min))
        safe_max = torch.log1p(torch.relu(t_max))
        t_min = torch.where(self.log_mask, safe_min, t_min)
        t_max = torch.where(self.log_mask, safe_max, t_max)

        # Safety margin
        rng = (t_max - t_min).clamp(min=self.epsilon)
        margin = rng * self.safety_margin
        self.stat_min.copy_(t_min - margin)
        self.stat_max.copy_(t_max + margin)

        self.is_calibrated.fill_(True)
        logger.info("ClinicalNormalizer calibrated from %s", path)

    def calibrate_from_array(
        self,
        data: np.ndarray,
        p01: float = 1.0,
        p99: float = 99.0,
    ) -> None:
        """
        Calibrate directly from a numpy array (useful for testing / MIMIC).

        Parameters
        ----------
        data : np.ndarray [N, C]
            Raw feature matrix.
        """
        assert data.shape[1] == len(CANONICAL_COLUMNS), "Column count mismatch."
        raw_min = np.percentile(data, p01, axis=0)
        raw_max = np.percentile(data, p99, axis=0)

        dev = self.physics_min.device
        t_min = torch.tensor(raw_min, dtype=torch.float32, device=dev)
        t_max = torch.tensor(raw_max, dtype=torch.float32, device=dev)

        t_min = torch.max(t_min, self.physics_min)
        t_max = torch.min(t_max, self.physics_max)

        safe_min = torch.log1p(torch.relu(t_min))
        safe_max = torch.log1p(torch.relu(t_max))
        t_min = torch.where(self.log_mask, safe_min, t_min)
        t_max = torch.where(self.log_mask, safe_max, t_max)

        rng = (t_max - t_min).clamp(min=self.epsilon)
        margin = rng * self.safety_margin
        self.stat_min.copy_(t_min - margin)
        self.stat_max.copy_(t_max + margin)
        self.is_calibrated.fill_(True)

    # ------------------------------------------------------------------
    # Forward (normalize)
    # ------------------------------------------------------------------

    def normalize(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Normalize a time-series tensor.

        Parameters
        ----------
        x : Tensor[..., C]  (2-D or 3-D: [B, T, C] or [T, C])
        mask : Tensor[..., C] optional — not used in normalization, passed through.

        Returns
        -------
        Tensor[..., C] in approximately [-1, 1] (leaky clip at ±2).
        """
        if not self.is_calibrated:
            return x  # pass-through before calibration

        # NaN guard
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Prepare broadcasting
        view = self._view(x)
        p_min = self.physics_min.to(x.device).view(view)
        p_max = self.physics_max.to(x.device).view(view)
        s_min = self.stat_min.to(x.device).view(view)
        s_max = self.stat_max.to(x.device).view(view)
        l_msk = self.log_mask.to(x.device).view(view)

        # Physics clamp
        x = torch.clamp(x, p_min, p_max)

        # Log1p for heavy-tailed channels
        x_log = torch.log1p(torch.relu(x))
        x = torch.where(l_msk, x_log, x)

        # Quantile scaling → [-1, 1]
        denom = (s_max - s_min).clamp(min=self.epsilon)
        x_01 = (x - s_min) / denom
        x_norm = x_01 * 2.0 - 1.0

        # Leaky clip (preserves gradient beyond bounds)
        x_norm = torch.where(x_norm > 1.0,  1.0 + (x_norm - 1.0) * 0.1, x_norm)
        x_norm = torch.where(x_norm < -1.0, -1.0 + (x_norm + 1.0) * 0.1, x_norm)
        x_norm = torch.clamp(x_norm, -2.0, 2.0)

        return x_norm

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.normalize(x, mask)

    # ------------------------------------------------------------------
    # Denormalize (for interpretability)
    # ------------------------------------------------------------------

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Invert normalize() — returns approximate clinical units."""
        if not self.is_calibrated:
            return x_norm

        view = self._view(x_norm)
        s_min = self.stat_min.to(x_norm.device).view(view)
        s_max = self.stat_max.to(x_norm.device).view(view)
        l_msk = self.log_mask.to(x_norm.device).view(view)

        x_01 = (x_norm + 1.0) / 2.0
        x_01 = torch.clamp(x_01, 0.0, 1.0)
        x_scaled = x_01 * (s_max - s_min) + s_min

        # Inverse log for log channels
        x_log = torch.expm1(torch.clamp(x_scaled, -11.0, 11.0))
        x_lin = torch.clamp(x_scaled, -5000.0, 5000.0)
        return torch.where(l_msk, x_log, x_lin)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _view(x: torch.Tensor) -> Tuple[int, ...]:
        """Return view shape for broadcasting against last dim."""
        if x.dim() == 2:
            return (1, -1)
        elif x.dim() == 3:
            return (1, 1, -1)
        else:
            return (-1,)

    def __repr__(self) -> str:
        status = "Calibrated" if self.is_calibrated.item() else "NOT calibrated"
        log_count = self.log_mask.sum().item()
        return (
            f"ClinicalNormalizer (BGSL edition)\n"
            f"  Status: {status}\n"
            f"  Channels: {len(CANONICAL_COLUMNS)} (log-transformed: {log_count})\n"
            f"  Safety margin: {self.safety_margin * 100:.0f}%\n"
            f"  Epsilon: {self.epsilon}"
        )
