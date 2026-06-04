"""
models/gru.py
-------------
GRU-based risk predictor — primary reproducible baseline for BGSL.

Design
------
- Bidirectional=False (strictly causal: no future leakage).
- Variable-length trajectory input via pack_padded_sequence.
- Output: one logit per time step [B, T] representing risk of sepsis
  onset within the next H hours.
- Missingness mask is concatenated to input features as an additional
  signal channel (obs_mask), doubling the effective input dimensionality.

Architecture
------------
  Input: [B, T, C] + [B, T, C] mask → concat → [B, T, 2C]
  GRU: 2C → hidden_dim, num_layers layers, dropout
  Head: hidden_dim → 1 (linear projection, no sigmoid — raw logit)

The sigmoid is applied in the loss and metric functions, never in
the model forward pass, to preserve numerical stability.

Reference: plan.md §8.1 (Primary Architectures), §13.2 (API)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ["GRUPredictor"]


class GRUPredictor(nn.Module):
    """
    Causal GRU risk predictor outputting per-timestep logits.

    Parameters
    ----------
    input_dim : int
        Number of input features (default 28 for Clinical 28).
    hidden_dim : int
        GRU hidden state dimension.
    num_layers : int
        Number of GRU layers.
    dropout : float
        Dropout between GRU layers (0 for single-layer).
    use_mask_as_input : bool
        If True, concatenates the observation mask to input features.
        Doubles effective input dimension.
    """

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_mask_as_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_mask_as_input = use_mask_as_input

        gru_input_dim = input_dim * 2 if use_mask_as_input else input_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # CAUSAL — no bidirectional
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
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
            Normalized feature sequences.
        mask : Tensor[B, T, C] or None
            Observation mask (1=observed, 0=imputed). If provided and
            use_mask_as_input=True, concatenated to x.
        seq_lens : Tensor[B] or None
            Valid sequence lengths for pack_padded_sequence.
            If None, assumes all sequences are full length T.

        Returns
        -------
        logits : Tensor[B, T]
            Per-timestep raw risk logits (before sigmoid).
        """
        B, T, C = x.shape

        # Concatenate observation mask as additional feature channel
        if self.use_mask_as_input and mask is not None:
            # mask: [B, T, C] — use per-feature observation indicator
            inp = torch.cat([x, mask], dim=-1)  # [B, T, 2C]
        else:
            inp = x  # [B, T, C]

        # Pack sequences for efficiency (avoid computing on padding)
        if seq_lens is not None:
            seq_lens_cpu = seq_lens.clamp(max=T).cpu()
            packed = pack_padded_sequence(
                inp,
                seq_lens_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.gru(packed)
            gru_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            gru_out, _ = self.gru(inp)  # [B, T, hidden_dim]

        # Project each timestep to a scalar logit
        logits = self.head(gru_out).squeeze(-1)  # [B, T]

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"GRUPredictor("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"params={self.count_parameters():,})"
        )
