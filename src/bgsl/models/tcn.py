"""
models/tcn.py
-------------
Temporal Convolutional Network (TCN) risk predictor.

Architecture
------------
Dilated causal convolutions with exponentially increasing dilation:
    layer 0: dilation = 1   (receptive field: 2*kernel_size - 1)
    layer 1: dilation = 2
    layer 2: dilation = 4
    ...
    layer k: dilation = 2^k

Each layer is a residual block:
    Input → [CausalConv1d → BatchNorm → GELU → Dropout]×2 → + skip → Output

Total receptive field with K layers and kernel_size k:
    RF = (k - 1) * (2^K - 1) * 2 + 1

CAUSALITY: All convolutions use left-padding only (no right pad),
ensuring p_t depends only on x_1:t, not x_{t+1}:T.

Design notes (plan §8.1)
- TCN is the temporal convolution baseline.
- Receptive field should comfortably cover the prediction horizon H.
  With kernel=3, n_layers=6: RF = 127 time steps (>12h at hourly rate).
- Observation mask concatenated as extra input channels.
- Same output format as GRUPredictor: [B, T] logits.

Reference: Bai et al. "An Empirical Evaluation of Generic Convolutional
           and Recurrent Networks for Sequence Modeling." arXiv 2018.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TCNPredictor"]


# ---------------------------------------------------------------------------
# Causal convolution block
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """
    1-D causal convolution: applies left-padding to enforce causality.
    No future information leaks.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # left-only padding
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = F.pad(x, (self.padding, 0))  # pad left only
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """
    Temporal residual block with two causal dilated convolutions.

    Block:
        x → CausalConv → BN → GELU → Dropout
          → CausalConv → BN → GELU → Dropout
          → + skip_connection(x)
          → output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Skip connection (1×1 conv if channel dims differ)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return out + self.skip(residual)


# ---------------------------------------------------------------------------
# TCN Predictor
# ---------------------------------------------------------------------------

class TCNPredictor(nn.Module):
    """
    TCN risk predictor with dilated causal convolutions.

    Parameters
    ----------
    input_dim : int
        Number of input features. Default 28.
    n_filters : int
        Number of convolutional filters per layer. Default 128.
    kernel_size : int
        Convolution kernel size. Default 3.
    n_layers : int
        Number of residual layers. Default 6.
        With kernel=3 and n_layers=6: RF = 127 time steps.
    dropout : float
        Dropout rate. Default 0.2.
    use_mask_as_input : bool
        If True, concatenates observation mask to input. Default True.
    """

    def __init__(
        self,
        input_dim: int = 28,
        n_filters: int = 128,
        kernel_size: int = 3,
        n_layers: int = 6,
        dropout: float = 0.2,
        use_mask_as_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.use_mask_as_input = use_mask_as_input

        effective_input = input_dim * 2 if use_mask_as_input else input_dim

        # Input projection to filter dimension
        self.input_proj = nn.Conv1d(effective_input, n_filters, kernel_size=1)

        # Stack of dilated residual blocks
        blocks = []
        for i in range(n_layers):
            dilation = 2 ** i
            blocks.append(
                TCNResidualBlock(
                    in_channels=n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)

        # Output head: per-timestep logit
        self.head = nn.Sequential(
            nn.Conv1d(n_filters, n_filters // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_filters // 2, 1, kernel_size=1),
        )

        self._log_receptive_field(kernel_size, n_layers)
        self._init_weights()

    def _log_receptive_field(self, kernel_size: int, n_layers: int) -> None:
        import logging
        rf = 1
        for i in range(n_layers):
            dilation = 2 ** i
            rf += (kernel_size - 1) * dilation * 2
        logging.getLogger("bgsl.tcn").info(
            "TCNPredictor: kernel=%d, layers=%d → receptive field=%d time steps",
            kernel_size, n_layers, rf,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[B, T, C]
        mask : Tensor[B, T, C] or None
        seq_lens : Tensor[B] or None (unused in TCN — kept for API compatibility)

        Returns
        -------
        logits : Tensor[B, T]
        """
        if self.use_mask_as_input and mask is not None:
            x = torch.cat([x, mask], dim=-1)  # [B, T, 2C]

        # TCN operates on [B, C, T]
        x = x.permute(0, 2, 1)          # [B, 2C, T]
        x = self.input_proj(x)           # [B, n_filters, T]
        x = self.network(x)              # [B, n_filters, T]
        logits = self.head(x)            # [B, 1, T]
        logits = logits.squeeze(1)       # [B, T]

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"TCNPredictor("
            f"input_dim={self.input_dim}, "
            f"n_filters={self.n_filters}, "
            f"n_layers={self.n_layers}, "
            f"params={self.count_parameters():,})"
        )
