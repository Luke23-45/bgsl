from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CSDKalmanObserver(nn.Module):
    """Learned constant-gain Kalman observer with optional LSTM head.

    The Kalman filter produces a latent state z[t] from observations.
    Two head options:
      - MLP head: processes z[t] independently per timestep (default)
      - LSTM head: processes z[1:t] through a causal LSTM, enabling
        detection of temporal CSD patterns (rising autocorrelation,
        slowing recovery) that a per-step head cannot see.

    Parameters
    ----------
    input_dim : int
    latent_dim : int
        Latent state dimension (default 4).
    lstm_head : bool
        If True, use causal LSTM head instead of per-step MLP.
    lstm_dim : int
        LSTM hidden dimension (default 8, only used when lstm_head=True).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        lstm_head: bool = False,
        lstm_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lstm_head = lstm_head

        self.obs_proj = nn.Linear(input_dim, 1)

        A_init = torch.eye(latent_dim) * 0.95
        self.A = nn.Parameter(A_init)
        self.C = nn.Parameter(torch.randn(1, latent_dim).mul(0.1))
        self.K = nn.Parameter(torch.randn(latent_dim, 1).mul(0.1))

        if lstm_head:
            self.lstm = nn.LSTM(latent_dim, lstm_dim, batch_first=True)
            self.out_head = nn.Sequential(
                nn.LayerNorm(lstm_dim),
                nn.Linear(lstm_dim, lstm_dim // 2),
                nn.GELU(),
                nn.Linear(lstm_dim // 2, 1),
            )
        else:
            h_dim = max(latent_dim // 2, 2)
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
        if self.lstm_head:
            for n, p in self.lstm.named_parameters():
                if "weight" in n:
                    nn.init.orthogonal_(p)
            for m in self.out_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
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

        z = torch.zeros(B, d, device=device)
        lstm_dim = self.lstm.hidden_size if self.lstm_head else 0
        if self.lstm_head:
            hx = torch.zeros(1, B, lstm_dim, device=device)
            cx = torch.zeros(1, B, lstm_dim, device=device)
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

            if self.lstm_head:
                _, (hx, cx) = self.lstm(z.unsqueeze(1), (hx, cx))
                logits.append(self.out_head(hx.squeeze(0)))
            else:
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
            f"latent_dim={self.latent_dim}, "
            f"lstm_head={self.lstm_head}, "
            f"params={self.count_parameters():,})"
        )
