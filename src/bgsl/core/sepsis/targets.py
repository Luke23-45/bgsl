"""
bgsl/core/sepsis/targets.py
---------------------------
Sepsis-specific soft target construction.
Wraps the generic target constructor.
"""

from __future__ import annotations

from typing import Optional, Dict

import torch

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
        kappa: float = 1.0,
        post_onset_value: float = 1.0,
        time_step_hours: float = 1.0,
    ) -> None:
        super().__init__(
            horizon=float(horizon_hours),
            tau=tau,
            kappa=kappa,
            post_onset_value=post_onset_value,
            time_step_duration=float(time_step_hours),
        )

    def __call__(
        self,
        onset_hours: torch.Tensor,
        seq_lengths: torch.Tensor,
        device: torch.device,
        delta_mu: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        kappa_shape: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Domain alias: 'onset_hours' → base class 'onset_times'."""
        return super().__call__(
            onset_hours, seq_lengths, device,
            delta_mu=delta_mu, sigma=sigma, kappa_shape=kappa_shape,
        )

    def build_single(
        self,
        onset_hour: float,
        seq_len: int,
        device: torch.device,
        delta_mu: Optional[float] = None,
        sigma: Optional[float] = None,
        kappa: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Domain alias: 'onset_hour' → base class 'onset_time'."""
        return super().build_single(
            float(onset_hour), seq_len, device,
            delta_mu=delta_mu, sigma=sigma, kappa=kappa,
        )

