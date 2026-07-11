from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _var_activation(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x) + 1e-6


class CSDLoss(nn.Module):
    """Differentiable CSD regularization on latent state sequences.

    Computes a sliding-window lag-1 autocorrelation on each latent dimension
    and penalizes the MSE against a ramp target that increases from 0 to 1
    in the pre-bifurcation window [tau - W_pre, tau].

    This encourages the learned latent dynamics to exhibit critical slowing
    down (increasing autocorrelation) as the system approaches a tipping point.
    """

    def __init__(self, window_size: int = 30, pre_event_window: int = 50, csd_weight: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.pre_event_window = pre_event_window
        self.csd_weight = csd_weight
        self.register_buffer("_zero", torch.zeros(1))

    def _lag1_autocorr(self, z: torch.Tensor) -> torch.Tensor:
        """Sliding-window lag-1 autocorrelation per latent dimension.

        Uses causal window [t-W+1, t] at each time t.

        Args:
            z: [B, T, d] latent state sequence.
        Returns:
            rho: [B, T, d] autocorrelation per timestep per dimension.
        """
        B, T, d = z.shape
        W = min(self.window_size, T)

        z_perm = z.permute(0, 2, 1)
        z_padded = F.pad(z_perm, (W - 1, 0), mode="replicate")
        windows = z_padded.unfold(2, W, 1)

        means = windows.mean(dim=-1, keepdim=True)
        centered = windows - means

        num = (centered[:, :, :, :-1] * centered[:, :, :, 1:]).sum(dim=-1)
        denom = (centered ** 2).sum(dim=-1)
        denom = denom.clamp(min=1e-8)

        rho = num / denom
        return rho.permute(0, 2, 1)

    def forward(
        self,
        z: torch.Tensor,
        bifurcation_times: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute CSD loss.

        Args:
            z: [B, T, d] latent states from the observer.
            bifurcation_times: [B] known bifurcation times.
            seq_lengths: [B] valid sequence lengths.
        Returns:
            dict with "loss", "csd", "rho_mean", "rho_target_mean", "rho_gap".
        """
        B, T, d = z.shape
        device = z.device

        if T < self.window_size:
            return {
                "loss": torch.tensor(0.0, device=device),
                "csd": torch.tensor(0.0, device=device),
                "rho_mean": self._zero,
                "rho_target_mean": self._zero,
            }

        rho = self._lag1_autocorr(z)

        t_grid = torch.arange(T, device=device).float().unsqueeze(0).expand(B, -1)
        tau = bifurcation_times.unsqueeze(1).float()

        window_start = (tau - self.pre_event_window).long()
        window_end = tau.long()

        window_start_clipped = window_start.clamp(min=self.window_size)
        window_end_clipped = window_end.clamp(max=T)

        ramp_len = (window_end_clipped - window_start_clipped).float().clamp(min=1.0)
        local_t = t_grid - window_start_clipped.float()
        rho_target = (local_t / ramp_len).clamp(0.0, 1.0).unsqueeze(-1)

        in_window = (
            (t_grid >= window_start_clipped.float())
            & (t_grid < window_end_clipped.float())
            & (t_grid < seq_lengths.unsqueeze(1).float())
        ).float().unsqueeze(-1)

        per_step = ((rho - rho_target) ** 2) * in_window
        loss_csd = per_step.sum() / in_window.sum().clamp(min=1.0)

        rho_mean = (rho * in_window).sum() / in_window.sum().clamp(min=1.0)
        rho_target_mean = (rho_target * in_window).sum() / in_window.sum().clamp(min=1.0)

        total = self.csd_weight * loss_csd

        return {
            "loss": total,
            "csd": loss_csd,
            "rho_mean": rho_mean.detach(),
            "rho_target_mean": rho_target_mean.detach(),
        }


class SpectralRadiusLoss(nn.Module):
    """Penalizes the spectral radius of the observer's error dynamics.

    Constrains ρ((I - KC)A) < threshold, ensuring the observer remains
    stable while allowing near-instability (which enhances CSD).
    """

    def __init__(
        self,
        n_power_iter: int = 10,
        threshold: float = 0.99,
        weight: float = 0.1,
    ):
        super().__init__()
        self.n_power_iter = n_power_iter
        self.threshold = threshold
        self.weight = weight
        self.register_buffer("_zero", torch.zeros(1))

    @staticmethod
    def _power_iteration(M: torch.Tensor, n_iter: int) -> torch.Tensor:
        """Estimate spectral radius via differentiable power iteration.

        Args:
            M: [d, d] square matrix.
            n_iter: number of power iterations.
        Returns:
            rho_hat: scalar estimate of the spectral radius.
        """
        d = M.shape[0]
        v = torch.randn(d, device=M.device)
        v = v / (v.norm() + 1e-8)
        for _ in range(n_iter):
            v = M @ v
            v_norm = v.norm()
            if v_norm > 1e-8:
                v = v / v_norm
        rho_hat = v @ (M @ v)
        return rho_hat

    def forward(
        self,
        A: torch.Tensor,
        K: torch.Tensor,
        C: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute spectral radius penalty.

        Args:
            A: [d, d] transition matrix.
            K: [d, 1] observer gain.
            C: [1, d] observation matrix.
        Returns:
            dict with "loss" and "spectral_radius".
        """
        d = A.shape[0]
        eye = torch.eye(d, device=A.device, dtype=A.dtype)
        M = (eye - K @ C) @ A
        rho = self._power_iteration(M, self.n_power_iter)
        penalty = torch.clamp(rho - self.threshold, min=0.0)
        loss = self.weight * penalty
        return {
            "loss": loss,
            "spectral_radius": rho.detach(),
        }
