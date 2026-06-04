"""
bgsl/targets.py
---------------
BGSL Soft Event-Onset Target Construction.

Core idea
---------
Instead of training against a hard binary step label:
    y_t = 1[t* - H <= t <= t*]

BGSL uses a smooth sigmoid target centered at the midpoint of the warning horizon:
    g_t = sigmoid((t - c) / tau)    where c = t* - H/2

This ensures that the first and second temporal derivatives of g_t are
analytically smooth and clinically meaningful:

    Δg_t  = g_t - g_{t-1}          (velocity target)
    Δ²g_t = g_t - 2g_{t-1} + g_{t-2}  (acceleration target)

Masking rules
-------------
- Post-onset time steps for positive patients: g_t = 1 (saturated).
- All time steps for negative patients: g_t = 0.
- Derivative losses are NOT computed:
    (a) at t=0 (velocity) or t=0,1 (acceleration) — no prior time steps.
    (b) across padded/out-of-sequence boundaries.
    (c) if the patient trajectory is shorter than 2 steps.

Reference: BGSL plan.md §3.2, §3.3, §6.3
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional

__all__ = ["SoftOnsetTarget"]


class SoftOnsetTarget:
    """
    Constructs the soft BGSL target trajectory and its temporal derivatives.

    Parameters
    ----------
    horizon_hours : int
        Prediction horizon H in hours. The warning window is [t* - H, t*].
    tau : float
        Softness / temperature parameter for the sigmoid. Controls how
        quickly the target transitions from 0 to 1 near the midpoint.
        Larger tau → slower / softer transition.
        Recommended: tau ∈ {1.0, 2.0, 4.0}.
    post_onset_value : float
        Target value after the onset time (t > t*). Default 1.0.
    time_step_hours : float
        Duration of each time step in hours. Default 1.0 (hourly ICU data).
    """

    def __init__(
        self,
        horizon_hours: int = 6,
        tau: float = 2.0,
        post_onset_value: float = 1.0,
        time_step_hours: float = 1.0,
    ) -> None:
        self.H = horizon_hours
        self.tau = tau
        self.post_onset_value = post_onset_value
        self.dt = time_step_hours

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        onset_hours: torch.Tensor,
        seq_lengths: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Build soft targets for a batch.

        Parameters
        ----------
        onset_hours : Tensor[B]
            Event onset time in hours for each patient.
            Use -1 (or any negative) to indicate a *negative* patient
            (no sepsis event). The entire trajectory then has g_t = 0.
        seq_lengths : Tensor[B]
            Valid (non-padded) sequence length for each patient.
        device : torch.device

        Returns
        -------
        dict with keys:
            "soft_target"    : Tensor[B, T_max]   — g_t
            "velocity_target": Tensor[B, T_max]   — Δg_t
            "accel_target"   : Tensor[B, T_max]   — Δ²g_t
            "valid_mask"     : Tensor[B, T_max]   — 1 = compute loss here
            "vel_mask"       : Tensor[B, T_max]   — 1 = compute velocity here
            "acc_mask"       : Tensor[B, T_max]   — 1 = compute accel here
        """
        B = onset_hours.shape[0]
        T_max = int(seq_lengths.max().item())

        # Time axis: shape [1, T_max], in hours
        t = torch.arange(T_max, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]

        # --- Soft target g_t ---
        g = self._build_soft_target(onset_hours, seq_lengths, t, T_max, B, device)

        # --- Temporal derivatives of g ---
        dg = self._first_derivative(g)    # [B, T]
        d2g = self._second_derivative(g)  # [B, T]

        # --- Valid mask (within sequence length) ---
        length_mask = self._length_mask(seq_lengths, T_max, device)  # [B, T]

        # Velocity is undefined at t=0 (no t-1)
        vel_mask = length_mask.clone()
        vel_mask[:, 0] = 0.0

        # Acceleration is undefined at t=0,1 (no t-1, t-2)
        acc_mask = length_mask.clone()
        acc_mask[:, :2] = 0.0

        return {
            "soft_target":     g,
            "velocity_target": dg,
            "accel_target":    d2g,
            "valid_mask":      length_mask,
            "vel_mask":        vel_mask,
            "acc_mask":        acc_mask,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_soft_target(
        self,
        onset_hours: torch.Tensor,
        seq_lengths: torch.Tensor,
        t: torch.Tensor,          # [1, T_max]
        T_max: int,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Construct g_t for each patient in the batch.

        Positive patient (onset_hours[b] >= 0):
            c_b  = onset_hours[b] - H/2          (midpoint of warning window)
            g_bt = sigmoid((t - c_b) / tau)       for t <= onset_hours[b]
            g_bt = post_onset_value               for t >  onset_hours[b]

        Negative patient (onset_hours[b] < 0):
            g_bt = 0   for all t
        """
        onset = onset_hours.float().to(device).unsqueeze(1)  # [B, 1]
        is_positive = (onset >= 0).float()                    # [B, 1]

        # Midpoint of warning window
        c = onset - self.H / 2.0  # [B, 1]

        # Sigmoid part
        sigmoid_val = torch.sigmoid((t - c) / self.tau)  # [B, T]

        # Post-onset: clamp to post_onset_value
        # t > onset_hours[b] (broadcast)
        post_onset_mask = (t > onset).float()  # [B, T]
        g_positive = (
            sigmoid_val * (1.0 - post_onset_mask)
            + self.post_onset_value * post_onset_mask
        )

        # Negative patients: zero everywhere
        g = is_positive * g_positive  # [B, T]

        # Zero out beyond sequence length (padding)
        length_mask = self._length_mask(seq_lengths, T_max, device)
        g = g * length_mask

        return g  # [B, T]

    @staticmethod
    def _first_derivative(g: torch.Tensor) -> torch.Tensor:
        """
        Δg_t = g_t - g_{t-1}, with Δg_0 = 0.
        Shape: [B, T]
        """
        # Shift right by 1 (t-1), pad with zero at position 0
        g_prev = F.pad(g[:, :-1], (1, 0), value=0.0)  # [B, T]
        dg = g - g_prev
        dg[:, 0] = 0.0  # undefined at t=0
        return dg

    @staticmethod
    def _second_derivative(g: torch.Tensor) -> torch.Tensor:
        """
        Δ²g_t = g_t - 2·g_{t-1} + g_{t-2}, with Δ²g_0 = Δ²g_1 = 0.
        Shape: [B, T]
        """
        g_prev1 = F.pad(g[:, :-1], (1, 0), value=0.0)   # g_{t-1}
        g_prev2 = F.pad(g[:, :-2], (2, 0), value=0.0)   # g_{t-2}
        d2g = g - 2.0 * g_prev1 + g_prev2
        d2g[:, :2] = 0.0  # undefined at t=0,1
        return d2g

    @staticmethod
    def _length_mask(
        seq_lengths: torch.Tensor,
        T_max: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Binary mask: 1 for valid time steps, 0 for padding.
        Shape: [B, T_max]
        """
        B = seq_lengths.shape[0]
        t_idx = torch.arange(T_max, device=device).unsqueeze(0).expand(B, -1)
        mask = (t_idx < seq_lengths.unsqueeze(1).to(device)).float()
        return mask

    # ------------------------------------------------------------------
    # Utility: build targets from individual patient onset info
    # ------------------------------------------------------------------

    def build_single(
        self,
        onset_hour: int,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience wrapper for a single patient (batch of 1).

        Parameters
        ----------
        onset_hour : int
            Event onset time in hours. Pass -1 for negative patients.
        seq_len : int
            Sequence length.

        Returns
        -------
        Same dict as __call__, with batch dim B=1.
        """
        onset_t = torch.tensor([onset_hour], dtype=torch.float32)
        lengths_t = torch.tensor([seq_len], dtype=torch.long)
        return self.__call__(onset_t, lengths_t, device)

    def __repr__(self) -> str:
        return (
            f"SoftOnsetTarget("
            f"horizon={self.H}h, "
            f"tau={self.tau}, "
            f"post_onset_value={self.post_onset_value}, "
            f"dt={self.dt}h)"
        )
