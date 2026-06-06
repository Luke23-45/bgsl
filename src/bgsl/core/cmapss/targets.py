"""
bgsl/core/cmapss/targets.py
---------------------------
CMAPSS-specific soft target construction.
Wraps the generic target constructor for cycle-based failure prediction.
"""

from __future__ import annotations

import torch
from typing import Dict

from bgsl.core.common.targets import BaseSoftOnsetTarget

__all__ = ["CycleOnsetTarget"]


class CycleOnsetTarget(BaseSoftOnsetTarget):
    """
    CMAPSS-specific wrapper for soft BGSL target trajectory.
    Uses 'cycles' terminology instead of hours.
    """

    def __init__(
        self,
        horizon_cycles: int = 30,
        tau: float = 2.0,
        post_onset_value: float = 1.0,
    ) -> None:
        super().__init__(
            horizon=float(horizon_cycles),
            tau=tau,
            post_onset_value=post_onset_value,
            time_step_duration=1.0,  # 1 cycle = 1 time step
        )

    def __call__(
        self,
        failure_cycles: torch.Tensor,
        seq_lengths: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        return super().__call__(failure_cycles, seq_lengths, device)

    def build_single(
        self,
        failure_cycle: int,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        return super().build_single(float(failure_cycle), seq_len, device)
