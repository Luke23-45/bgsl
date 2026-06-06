"""
bgsl/core/common/targets.py
---------------------------
Generic BGSL Soft Event-Onset Target Construction.
Agnostic to domain. Sepsis and CMAPSS modules can wrap this
if they need domain-specific argument names or masking logic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict

__all__ = ["BaseSoftOnsetTarget"]


class BaseSoftOnsetTarget:
    """
    Constructs the soft BGSL target trajectory and its temporal derivatives.

    Parameters
    ----------
    horizon : float
        Prediction horizon H. The warning window is [t* - H, t*].
    tau : float
        Softness / temperature parameter for the sigmoid. Controls how
        quickly the target transitions from 0 to 1 near the midpoint.
    post_onset_value : float
        Target value after the onset time (t > t*). Default 1.0.
    time_step_duration : float
        Duration of each time step. Default 1.0.
    """

    def __init__(
        self,
        horizon: float = 6.0,
        tau: float = 2.0,
        post_onset_value: float = 1.0,
        time_step_duration: float = 1.0,
    ) -> None:
        self.H = horizon
        self.tau = tau
        self.post_onset_value = post_onset_value
        self.dt = time_step_duration

    def __call__(
        self,
        onset_times: torch.Tensor,
        seq_lengths: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Build soft targets for a batch.

        Parameters
        ----------
        onset_times : Tensor[B]
            Event onset time. Use -1 (or any negative) to indicate no event.
        seq_lengths : Tensor[B]
            Valid (non-padded) sequence length for each instance.
        device : torch.device

        Returns
        -------
        dict with keys:
            "soft_target"
            "hard_target"
            "velocity_target"
            "accel_target"
            "valid_mask"
            "vel_mask"
            "acc_mask"
        """
        B = onset_times.shape[0]
        T_max = int(seq_lengths.max().item())

        t = torch.arange(T_max, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]

        g = self._build_soft_target(onset_times, seq_lengths, t, T_max, B, device)
        hard_g = self._build_hard_target(onset_times, seq_lengths, t, T_max, device)

        dg = self._first_derivative(g)
        d2g = self._second_derivative(g)

        length_mask = self._length_mask(seq_lengths, T_max, device)

        vel_mask = length_mask.clone()
        if T_max > 0:
            vel_mask[:, 0] = 0.0

        acc_mask = length_mask.clone()
        if T_max > 1:
            acc_mask[:, :2] = 0.0

        return {
            "soft_target":     g,
            "hard_target":     hard_g,
            "velocity_target": dg,
            "accel_target":    d2g,
            "valid_mask":      length_mask,
            "vel_mask":        vel_mask,
            "acc_mask":        acc_mask,
        }

    def _build_soft_target(
        self,
        onset_times: torch.Tensor,
        seq_lengths: torch.Tensor,
        t: torch.Tensor,
        T_max: int,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        onset = onset_times.float().to(device).unsqueeze(1)
        is_positive = (onset >= 0).float()

        c = onset - self.H / 2.0
        sigmoid_val = torch.sigmoid((t - c) / self.tau)

        post_onset_mask = (t > onset).float()
        g_positive = (
            sigmoid_val * (1.0 - post_onset_mask)
            + self.post_onset_value * post_onset_mask
        )

        g = is_positive * g_positive
        length_mask = self._length_mask(seq_lengths, T_max, device)
        g = g * length_mask
        return g

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

    @staticmethod
    def _first_derivative(g: torch.Tensor) -> torch.Tensor:
        g_prev = F.pad(g[:, :-1], (1, 0), value=0.0)
        dg = g - g_prev
        if dg.shape[1] > 0:
            dg[:, 0] = 0.0
        return dg

    @staticmethod
    def _second_derivative(g: torch.Tensor) -> torch.Tensor:
        g_prev1 = F.pad(g[:, :-1], (1, 0), value=0.0)
        g_prev2 = F.pad(g[:, :-2], (2, 0), value=0.0)
        d2g = g - 2.0 * g_prev1 + g_prev2
        if d2g.shape[1] > 1:
            d2g[:, :2] = 0.0
        return d2g

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

    def build_single(
        self,
        onset_time: float,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        onset_t = torch.tensor([onset_time], dtype=torch.float32)
        lengths_t = torch.tensor([seq_len], dtype=torch.long)
        return self.__call__(onset_t, lengths_t, device)

    def __repr__(self) -> str:
        return (
            f"BaseSoftOnsetTarget("
            f"horizon={self.H}, "
            f"tau={self.tau}, "
            f"post_onset_value={self.post_onset_value}, "
            f"dt={self.dt})"
        )
