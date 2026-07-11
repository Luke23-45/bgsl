from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CSDKalmanObserver(nn.Module):
    """Learned constant-gain Kalman observer with optional CSD feature augmentation.

    Architecture:
        obs_proj:   input_dim → 1
        Kalman:     z[t] = A @ z[t-1] + K @ (y[t] - C @ A @ z[t-1])
        ρ_ema[t]:   EMA of per-dim lag-1 autocorrelation of z
        Head input: z[t] (+ ρ_ema[t] + innov[t] if use_csd_features)
        Head:       LayerNorm → Linear → GELU → Linear → logit

    Parameters
    ----------
    input_dim : int
    latent_dim : int
        Latent state dimension (default 4).
    use_csd_features : bool
        If True, the head sees (z, ρ_ema, residual) instead of just z.
    rho_beta : float
        EMA decay rate for autocorrelation tracking (default 0.9).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        use_csd_features: bool = False,
        rho_beta: float = 0.9,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_csd_features = use_csd_features
        self.rho_beta = rho_beta

        self.obs_proj = nn.Linear(input_dim, 1)

        A_init = torch.eye(latent_dim) * 0.95
        self.A = nn.Parameter(A_init)
        self.C = nn.Parameter(torch.randn(1, latent_dim).mul(0.1))
        self.K = nn.Parameter(torch.randn(latent_dim, 1).mul(0.1))
        self.log_Q = nn.Parameter(torch.full((latent_dim,), -4.6))

        head_in = latent_dim
        if use_csd_features:
            head_in = latent_dim + latent_dim + 1  # z + rho + residual

        h_dim = max(head_in // 2, 4)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, h_dim),
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

        Returns
        -------
        logits : [B, T] risk logits.
        zs : [B, T, d] latent states.
        A, K, C : [d, d], [d, 1], [1, d] matrices.
        """
        B, T, _ = x.shape
        d = self.latent_dim
        device = x.device

        A = self.A
        C = self.C
        K = self.K
        beta = self.rho_beta

        z = torch.zeros(B, d, device=device)
        z_prev = torch.zeros(B, d, device=device)
        cov_ema = torch.zeros(B, d, device=device)
        var_ema = torch.ones(B, d, device=device) * 1e-2
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

            if t > 0:
                cov_ema = beta * cov_ema + (1 - beta) * z * z_prev
                var_ema = beta * var_ema + (1 - beta) * z ** 2
            rho = cov_ema / (var_ema + 1e-8)

            if self.use_csd_features:
                head_in = torch.cat([z, rho, residual], dim=-1)
            else:
                head_in = z

            logits.append(self.head(head_in))
            zs.append(z)
            z_prev = z.detach().clone()

        logits = torch.stack(logits, dim=1).squeeze(-1)
        zs = torch.stack(zs, dim=1)

        return logits, zs, A, K, C

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CSDKalmanObserver(input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"csd_features={self.use_csd_features}, "
            f"params={self.count_parameters():,})"
        )
