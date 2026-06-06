"""
data/cmapss/dataset.py
-----------------------
CMAPSS PyTorch Dataset for BGSL training.

Design
------
The CMAPSS preprocessor outputs fixed-length sliding windows of shape
[N, window_size, F] stored in .pt files with keys 'inputs', 'rul', 'mask'.

For each window (a snapshot ending at cycle t for engine e):
  - RUL(t) = remaining cycles until failure
  - failure_cycle (within-window) = (window_size - 1) + RUL(t)
    i.e. the failure event occurs `RUL(t)` steps AFTER the last observed
    cycle in the window, which is at index (window_size - 1).

This failure_cycle is passed to CycleOnsetTarget to construct the
soft BGSL target trajectory over the window.

A sample is labelled is_failure=True when RUL < horizon_cycles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from bgsl.core.cmapss.targets import CycleOnsetTarget

__all__ = ["CMAPSSDataset"]


class CMAPSSDataset(Dataset):
    """
    BGSL-compatible CMAPSS dataset.

    Parameters
    ----------
    data_path : str
        Path to a preprocessed .pt split file (e.g. ``train/FD001.pt``).
    horizon_cycles : int
        Prediction horizon H in cycles. A unit is considered "at risk"
        when RUL < H. Default 30.
    tau : float
        Soft target temperature for CycleOnsetTarget. Default 2.0.
    """

    def __init__(
        self,
        data_path: str,
        horizon_cycles: int = 30,
        tau: float = 2.0,
    ) -> None:
        super().__init__()
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Processed CMAPSS data not found at {path}. "
                "Run data/cmapss/preprocessing.py first."
            )

        data = torch.load(str(path), weights_only=False)
        self.inputs: torch.Tensor = data["inputs"]   # [N, W, F]  float32
        self.rul:    torch.Tensor = data["rul"]      # [N]        float32
        self.mask:   torch.Tensor = data["mask"]     # [N, W]     float32

        self.window_size = self.inputs.shape[1]
        self.n_features  = self.inputs.shape[2]
        self.horizon_cycles = horizon_cycles

        self.target_fn = CycleOnsetTarget(horizon_cycles=horizon_cycles, tau=tau)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x    = self.inputs[idx]   # [W, F]
        rul  = self.rul[idx]      # scalar float32
        mask = self.mask[idx]     # [W]   (1 = valid cycle, 0 = left-pad)

        W = self.window_size
        rul_val = float(rul.item())

        # Within-window failure cycle:
        # The last timestep in the window is at index (W-1).
        # The true failure is `rul_val` cycles after that.
        failure_cycle = float(W - 1) + rul_val

        # is_failure: unit is within the warning window
        is_failure = rul_val < self.horizon_cycles

        # Sequence length: number of non-padded cycles
        seq_len = int(mask.sum().item())

        # Build BGSL soft targets (batch of 1, then squeeze)
        onset_t  = torch.tensor([failure_cycle], dtype=torch.float32)
        lengths  = torch.tensor([W], dtype=torch.long)
        tgt = self.target_fn(onset_t, lengths, device=x.device)

        g       = tgt["soft_target"].squeeze(0)      # [W]
        hard_g  = tgt["hard_target"].squeeze(0)      # [W]
        dg      = tgt["velocity_target"].squeeze(0)  # [W]
        d2g     = tgt["accel_target"].squeeze(0)     # [W]
        valid   = mask                                # [W]  use preproc mask
        vel_mask  = tgt["vel_mask"].squeeze(0)       # [W]
        acc_mask  = tgt["acc_mask"].squeeze(0)       # [W]

        # Reconcile valid_mask: both padding mask AND BGSL mask must be valid
        vel_mask = vel_mask * mask
        acc_mask = acc_mask * mask

        return {
            "vitals":        x,                                              # [W, F]
            "masks":         mask.unsqueeze(-1).expand_as(x),               # [W, F]  broadcast obs mask
            "hard_targets":  hard_g,                                         # [W]
            "soft_targets":  g,                                              # [W]
            "vel_targets":   dg,                                             # [W]
            "accel_targets": d2g,                                            # [W]
            "valid_mask":    valid,                                          # [W]
            "vel_mask":      vel_mask,                                       # [W]
            "acc_mask":      acc_mask,                                       # [W]
            "failure_cycle": torch.tensor(failure_cycle, dtype=torch.float32),
            "is_failure":    torch.tensor(is_failure, dtype=torch.bool),
            "seq_len":       torch.tensor(seq_len, dtype=torch.long),
            "unit_id":       str(idx),   # placeholder; subset+idx is unique per split
        }
