"""
bgsl/core/sepsis/targets.py
---------------------------
Sepsis-specific soft target construction.
Wraps the generic target constructor.
"""

from __future__ import annotations

import torch
from typing import Dict

from bgsl.core.common.targets import BaseSoftOnsetTarget

__all__ = ["SoftOnsetTarget"]


class SoftOnsetTarget(BaseSoftOnsetTarget):
    """
    Sepsis-specific wrapper for soft BGSL target trajectory.
    Uses 'hours' terminology.
    """

    def __init__(
        self,
        horizon_hours: int = 6,
        tau: float = 2.0,
        post_onset_value: float = 1.0,
        time_step_hours: float = 1.0,
    ) -> None:
        super().__init__(
            horizon=float(horizon_hours),
            tau=tau,
            post_onset_value=post_onset_value,
            time_step_duration=float(time_step_hours),
        )

    def __call__(
        self,
        onset_hours: torch.Tensor,
        seq_lengths: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        return super().__call__(onset_hours, seq_lengths, device)

    def build_single(
        self,
        onset_hour: int,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        return super().build_single(float(onset_hour), seq_len, device)
