"""
bgsl/core/sepsis/targets.py
---------------------------
Sepsis-specific soft target construction.
Wraps the generic target constructor.
"""

from __future__ import annotations



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

    # __call__ and build_single are intentionally inherited from
    # BaseSoftOnsetTarget unchanged — the base signatures already accept
    # the optional delta_mu / sigma / kappa_shape keyword arguments needed
    # by the EBGSL / HyperNetwork path.
