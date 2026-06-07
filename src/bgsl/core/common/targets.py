"""
bgsl/core/common/targets.py
----------------------------
Generic BGSL / EBGSL Soft Event-Onset Target Construction.
Agnostic to domain. Sepsis and CMAPSS modules can wrap this
if they need domain-specific argument names or masking logic.

Supports two modes:
1. Fixed-shape targets (BGSL baseline) — uses a global tau (temperature)
   and kappa parameter. This is the backward-compatible mode.
2. Patient-specific targets (EBGSL) — accepts per-batch (Δμ, σ, κ)
   produced by a HyperNetwork from static covariates, constructing the
   generalized logistic (Richards curve) and its analytic derivatives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional

__all__ = ["BaseSoftOnsetTarget"]


class BaseSoftOnsetTarget:
    """
    Constructs the soft BGSL target trajectory and its temporal derivatives.

    For event-positive patients (y=1), the soft target at time t is:

        u_{i,t} = (t - μ_i) / σ_i
        μ_i     = τ_i* - H/2 + Δμ_i
        g_{i,t} = σ(u_{i,t})^{κ_i}

    where σ(·) is the logistic sigmoid. This is the **generalized logistic**
    (Richards curve). When κ=1 it reduces to the standard logistic.

    For event-negative patients (y=0):
        g_{i,t} = 0,  g'_{i,t} = 0,  g''_{i,t} = 0

    Analytic derivatives (stable sigmoid formulation, s = σ(u)):

        g'  = (κ / σ_i) · s^{κ} · (1 - s)
        g'' = (κ / σ_i²) · s^{κ} · (1 - s) · (κ - (κ+1)·s)

    Parameters
    ----------
    horizon : float
        Prediction horizon H. The soft target midpoint is at τ* - H/2.
    tau : float
        Default scale parameter σ (temperature). Used when no patient-specific
        sigma is provided. Kept for backward compatibility. Default 2.0.
    kappa : float
        Default shape parameter κ for the generalized logistic.
        κ=1 gives the standard logistic sigmoid. Default 1.0.
    post_onset_value : float
        Kept for backward compatibility; unused in the generalized logistic
        (the Richards curve naturally saturates at 1 as t → ∞).
    time_step_duration : float
        Duration of each time step. Default 1.0.
    """

    def __init__(
        self,
        horizon: float = 6.0,
        tau: float = 2.0,
        kappa: float = 1.0,
        post_onset_value: float = 1.0,
        time_step_duration: float = 1.0,
    ) -> None:
        self.H = horizon
        self.sigma_init = tau       # default scale (backward-compat name)
        self.kappa_init = kappa     # default shape
        self.post_onset_value = post_onset_value
        self.dt = time_step_duration

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        onset_times: torch.Tensor,
        seq_lengths: torch.Tensor,
        device: torch.device,
        # Optional patient-specific shape params  (EBGSL / HyperNetwork path)
        delta_mu: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        kappa_shape: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build soft targets for a batch.

        Parameters
        ----------
        onset_times : Tensor[B]
            Event onset time τ*. Use -1 (or any negative) to indicate no event.
        seq_lengths : Tensor[B]
            Valid (non-padded) sequence length for each instance.
        device : torch.device
        delta_mu : Tensor[B] or None
            Patient-specific midpoint offset from hypernetwork.
        sigma : Tensor[B] or None
            Patient-specific scale (> 0) from hypernetwork.
        kappa_shape : Tensor[B] or None
            Patient-specific shape (> 0) from hypernetwork.

        Returns
        -------
        dict with keys:
            "soft_target"     — g_t     [B, T]
            "hard_target"     — hard alarm label [B, T]
            "velocity_target" — g'_t    [B, T]  analytic first derivative
            "accel_target"    — g''_t   [B, T]  analytic second derivative
            "valid_mask"      — length mask  [B, T]
            "vel_mask"        — velocity loss mask (t=0 excluded)
            "acc_mask"        — acceleration loss mask (t=0,1 excluded)
        """
        B = onset_times.shape[0]
        T_max = int(seq_lengths.max().item())

        t = torch.arange(T_max, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]

        # --- resolve patient-specific or default parameters ---
        onset = onset_times.float().to(device).unsqueeze(1)           # [B, 1]
        is_positive = (onset >= 0).float()                            # [B, 1]

        dmu = _zero_if_none(delta_mu, B, device)                     # [B, 1]
        sigma_vals = _default_if_none(sigma, self.sigma_init, B, device)    # [B, 1]
        kappa_vals = _default_if_none(kappa_shape, self.kappa_init, B, device)  # [B, 1]

        mu = onset - self.H / 2.0 + dmu                              # [B, 1]

        # --- generalized logistic via sigmoid for numerical stability ---
        u = (t - mu) / sigma_vals                                    # [B, T]
        s = torch.sigmoid(u)                                         # [B, T], s in (0, 1)
        s_kappa = s ** kappa_vals                                    # [B, T], g = s^κ

        # --- targets for positives, explicit zero for negatives ---
        g = s_kappa * is_positive                                    # [B, T]

        # Analytic first derivative: g' = (κ/σ) · s^κ · (1-s)
        dg = (kappa_vals / sigma_vals) * s_kappa * (1.0 - s)
        dg = dg * is_positive                                        # zero for negatives

        # Analytic second derivative: g'' = (κ/σ²) · s^κ · (1-s) · (κ - (κ+1)·s)
        inner = kappa_vals - (kappa_vals + 1.0) * s
        d2g = (kappa_vals / sigma_vals ** 2) * s_kappa * (1.0 - s) * inner
        d2g = d2g * is_positive                                      # zero for negatives

        # --- length mask ---
        length_mask = self._length_mask(seq_lengths, T_max, device)
        g = g * length_mask
        dg = dg * length_mask
        d2g = d2g * length_mask

        # --- hard target (unchanged) ---
        hard_g = self._build_hard_target(onset_times, seq_lengths, t, T_max, device)

        # --- derivative loss masks ---
        vel_mask = length_mask.clone()

        acc_mask = length_mask.clone()

        return {
            "soft_target":     g,
            "hard_target":     hard_g,
            "velocity_target": dg,
            "accel_target":    d2g,
            "valid_mask":      length_mask,
            "vel_mask":        vel_mask,
            "acc_mask":        acc_mask,
        }

    # ------------------------------------------------------------------
    # single-patient convenience
    # ------------------------------------------------------------------

    def build_single(
        self,
        onset_time: float,
        seq_len: int,
        device: torch.device,
        delta_mu: Optional[float] = None,
        sigma: Optional[float] = None,
        kappa: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Wrapper around __call__ for a single patient."""
        onset_t = torch.tensor([onset_time], dtype=torch.float32)
        lengths_t = torch.tensor([seq_len], dtype=torch.long)

        delta_mu_t = None if delta_mu is None else torch.tensor(
            [delta_mu], dtype=torch.float32
        )
        sigma_t = None if sigma is None else torch.tensor(
            [sigma], dtype=torch.float32
        )
        kappa_t = None if kappa is None else torch.tensor(
            [kappa], dtype=torch.float32
        )

        return self.__call__(
            onset_t, lengths_t, device,
            delta_mu=delta_mu_t, sigma=sigma_t, kappa_shape=kappa_t,
        )

    # ------------------------------------------------------------------
    # Hard label construction  (unchanged)
    # ------------------------------------------------------------------

    def _build_hard_target(
        self,
        onset_times: torch.Tensor,
        seq_lengths: torch.Tensor,
        t: torch.Tensor,
        T_max: int,
        device: torch.device,
    ) -> torch.Tensor:
        onset = onset_times.float().to(device).unsqueeze(1)
        is_positive = (onset >= 0).float()
        hard = (t >= (onset - self.H)).float()
        hard = is_positive * hard
        length_mask = self._length_mask(seq_lengths, T_max, device)
        hard = hard * length_mask
        return hard

    # ------------------------------------------------------------------
    # Length mask
    # ------------------------------------------------------------------

    @staticmethod
    def _length_mask(
        seq_lengths: torch.Tensor,
        T_max: int,
        device: torch.device,
    ) -> torch.Tensor:
        B = seq_lengths.shape[0]
        t_idx = torch.arange(T_max, device=device).unsqueeze(0).expand(B, -1)
        mask = (t_idx < seq_lengths.unsqueeze(1).to(device)).float()
        return mask

    # ------------------------------------------------------------------
    # Analytic derivative formulas (exposed for testing)
    # ------------------------------------------------------------------

    @staticmethod
    def first_derivative_from_sigmoid(
        s: torch.Tensor,
        sigma: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        """Compute g' from sigmoid output s = σ(u).
        g' = (κ/σ) · s^κ · (1-s)"""
        return (kappa / sigma) * (s ** kappa) * (1.0 - s)

    @staticmethod
    def second_derivative_from_sigmoid(
        s: torch.Tensor,
        sigma: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        """Compute g'' from sigmoid output s = σ(u).
        g'' = (κ/σ²) · s^κ · (1-s) · (κ - (κ+1)·s)"""
        return (
            (kappa / sigma ** 2)
            * (s ** kappa)
            * (1.0 - s)
            * (kappa - (kappa + 1.0) * s)
        )

    def __repr__(self) -> str:
        return (
            f"BaseSoftOnsetTarget("
            f"horizon={self.H}, "
            f"sigma_init={self.sigma_init}, "
            f"kappa_init={self.kappa_init}, "
            f"dt={self.dt})"
        )


# -----------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------


def _zero_if_none(
    t: Optional[torch.Tensor], B: int, device: torch.device
) -> torch.Tensor:
    """Return t or a zero tensor of shape [B, 1]."""
    if t is None:
        return torch.zeros(B, 1, dtype=torch.float32, device=device)
    return t.float().to(device).unsqueeze(1)


def _default_if_none(
    t: Optional[torch.Tensor], default: float, B: int, device: torch.device
) -> torch.Tensor:
    """Return t or a tensor filled with `default` of shape [B, 1]."""
    if t is None:
        return torch.full((B, 1), default, dtype=torch.float32, device=device)
    return t.float().to(device).unsqueeze(1)
