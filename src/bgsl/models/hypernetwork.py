"""
bgsl/models/hypernetwork.py
----------------------------
Hypernetwork for patient-specific target-shape parameters.

Given static covariates u_i (e.g., demographics), the hypernetwork
produces the triplet (Δμ_i, σ_i, κ_i) that defines the shape of the
soft onset target for patient i:

    μ_i = τ_i* - H/2 + Δμ_i       (midpoint offset)
    σ_i = exp(log σ_i) > 0        (scale / steepness)
    κ_i = exp(log κ_i) > 0        (generalized logistic shape)

When no hypernetwork is used, these default to Δμ=0, σ=τ (temperature),
and κ=1 (standard logistic).

Training minimizes the EBGSL objective jointly over backbone parameters
θ and hypernetwork parameters ψ.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

__all__ = ["HyperNetwork"]


class HyperNetwork(nn.Module):
    """
    Maps static covariates to patient-specific target-shape parameters.

    Parameters
    ----------
    static_dim : int
        Dimension of the static covariate vector u_i.
    hidden_dim : int
        Hidden layer dimension. Default 64.
    num_layers : int
        Number of hidden layers (>= 1). Default 2.
    """

    def __init__(
        self,
        static_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        if static_dim < 1:
            raise ValueError(
                f"static_dim must be >= 1, got {static_dim}. "
                "If there are no static covariates, do not use a HyperNetwork."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        layers = []
        in_dim = static_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 3))  # (Δμ, log σ, log κ)

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        u : Tensor[B, D_static]
            Static covariates for a batch of patients.

        Returns
        -------
        delta_mu : Tensor[B]
            Patient-specific midpoint offset (can be any real value).
        sigma : Tensor[B]
            Patient-specific scale parameter (> 0).
        kappa : Tensor[B]
            Patient-specific shape parameter (> 0).
        """
        raw = self.net(u)  # [B, 3]
        delta_mu = raw[:, 0]
        log_sigma = raw[:, 1]
        log_kappa = raw[:, 2]

        sigma = torch.exp(log_sigma)
        kappa = torch.exp(log_kappa)

        return delta_mu, sigma, kappa

    def __repr__(self) -> str:
        return (
            f"HyperNetwork("
            f"static_dim={self.net[0].in_features}, "
            f"hidden_dim={self.net[0].out_features}, "
            f"num_layers={(len(self.net) - 1) // 2})"
        )
