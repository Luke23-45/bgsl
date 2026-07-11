from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _var_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) + 1e-6


class CSDKalmanObserver(nn.Module):
    """Learned constant-gain Kalman observer with explicit A/C/K matrices.

    Designed for CSD (Critical Slowing Down) research: the latent state
    sequence is returned so CSD regularization can be applied, and the
    explicit A/K/C matrices enable spectral radius constraints.

    Architecture:
        obs_proj: input_dim → 1 (projects multi-channel input to scalar)
        Kalman:   z[t] = A @ z[t-1] + K @ (y[t] - C @ A @ z[t-1])
        Head:     LayerNorm → Linear → GELU → Linear → logit

    Parameters
    ----------
    input_dim : int
        Number of input feature channels.
    latent_dim : int
        Latent state dimension (default 4).
    """

    def __init__(self, input_dim: int, latent_dim: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.obs_proj = nn.Linear(input_dim, 1)

        A_init = torch.eye(latent_dim) * 0.95
        self.A = nn.Parameter(A_init)

        self.C = nn.Parameter(torch.randn(1, latent_dim).mul(0.1))
        self.K = nn.Parameter(torch.randn(latent_dim, 1).mul(0.1))

        self.log_Q = nn.Parameter(torch.full((latent_dim,), -4.6))

        h_dim = max(latent_dim // 2, 4)
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.obs_proj.weight)
        nn.init.zeros_(self.obs_proj.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, T, input_dim] feature sequences.
            mask: [B, T, input_dim] observation mask (1=observed, 0=imputed).
        Returns:
            logits: [B, T] per-timestep risk logits.
            zs: [B, T, d] latent state sequence (for CSD loss).
            A: [d, d] transition matrix.
            K: [d, 1] observer gain.
            C: [1, d] observation matrix (the learned C, not the input_dim).
        """
        B, T, _ = x.shape
        d = self.latent_dim
        device = x.device

        A = self.A
        C = self.C
        K = self.K

        z = torch.zeros(B, d, device=device)
        logits = []
        zs = []

        for t in range(T):
            x_t = x[:, t, :]
            if mask is not None:
                x_t = x_t * mask[:, t, :]

            obs = self.obs_proj(x_t)

            z_pred = z @ A.T
            y_pred = z_pred @ C.T
            residual = obs - y_pred
            z = z_pred + residual @ K.T

            logits.append(self.head(z))
            zs.append(z)

        logits = torch.stack(logits, dim=1).squeeze(-1)
        zs = torch.stack(zs, dim=1)

        return logits, zs, A, K, C

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CSDKalmanObserver(input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, params={self.count_parameters():,})"
        )
