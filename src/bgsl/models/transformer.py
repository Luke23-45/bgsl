"""
models/transformer.py
----------------------
Causal Transformer encoder risk predictor.

Architecture
------------
- Causal self-attention (lower-triangular attention mask).
- Sinusoidal positional encoding (+ optional learned pos embedding).
- Pre-norm (LayerNorm before attention, not after) for training stability.
- Observation mask concatenated to features (same as GRU/TCN baselines).
- Output: per-timestep logit [B, T].

Causality enforcement
---------------------
The attention mask is a standard lower-triangular boolean mask:
    attn_mask[i, j] = True  iff j > i  (future positions are masked)
This is applied as `attn_mask` in nn.MultiheadAttention, which interprets
True as "ignore this position" (additive -inf masking).

Design rationale (plan §8.1)
- Transformer is the "modern sequence model" baseline.
- Architecture is kept deliberately simple (encoder only, no decoder,
  no cross-attention) to isolate the BGSL loss effect from model complexity.
- d_model and n_heads should be chosen so d_model / n_heads is an integer.

Reference: Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["TransformerPredictor"]


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPosEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).
    Accepts sequences up to max_len steps.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Pre-norm Transformer encoder layer
# ---------------------------------------------------------------------------

class PreNormEncoderLayer(nn.Module):
    """
    Single Pre-norm Transformer encoder layer.

    Pre-norm: LayerNorm applied BEFORE attention/FFN (not after).
    More stable training than Post-norm for smaller models.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + attn_out

        # Pre-norm FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Transformer Predictor
# ---------------------------------------------------------------------------

class TransformerPredictor(nn.Module):
    """
    Causal Transformer encoder risk predictor.

    Parameters
    ----------
    input_dim : int
        Number of input features. Default 40.
    d_model : int
        Transformer embedding dimension. Default 128.
        Must be divisible by n_heads.
    n_heads : int
        Number of attention heads. Default 4.
    n_layers : int
        Number of encoder layers. Default 4.
    dim_feedforward : int
        FFN inner dimension. Default 256.
    dropout : float
        Dropout rate. Default 0.1.
    max_seq_len : int
        Maximum sequence length for positional encoding. Default 512.
    use_mask_as_input : bool
        If True, concatenates observation mask to input. Default True.
    """

    def __init__(
        self,
        input_dim: int = 40,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_mask_as_input: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.input_dim = input_dim
        self.d_model = d_model
        self.use_mask_as_input = use_mask_as_input

        effective_input = input_dim * 2 if use_mask_as_input else input_dim

        # Input projection
        self.input_proj = nn.Linear(effective_input, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPosEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Encoder stack
        self.layers = nn.ModuleList([
            PreNormEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Per-timestep output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular boolean mask: True means "masked out" (future).
        Shape: [T, T]
        PyTorch MultiheadAttention interprets True as -inf (ignore).
        """
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def _key_padding_mask(
        self, seq_lens: torch.Tensor, T: int, device: torch.device
    ) -> torch.Tensor:
        """
        Key padding mask: True means "masked out" (padding).
        Shape: [B, T]
        """
        B = seq_lens.shape[0]
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        mask = t_idx >= seq_lens.unsqueeze(1).to(device)  # True for padding
        return mask

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
        seq_lens : Tensor[B] or None

        Returns
        -------
        logits : Tensor[B, T]
        """
        B, T, C = x.shape

        if self.use_mask_as_input and mask is not None:
            x = torch.cat([x, mask], dim=-1)  # [B, T, 2C]

        # Project to d_model
        x = self.input_proj(x)        # [B, T, d_model]
        x = self.pos_enc(x)           # [B, T, d_model]

        # Build masks
        causal_mask = self._causal_mask(T, x.device)   # [T, T]
        key_pad_mask = None
        if seq_lens is not None:
            key_pad_mask = self._key_padding_mask(seq_lens, T, x.device)  # [B, T]

        # Encoder stack
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask, key_padding_mask=key_pad_mask)

        x = self.final_norm(x)         # [B, T, d_model]
        logits = self.head(x).squeeze(-1)  # [B, T]

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"TransformerPredictor("
            f"input_dim={self.input_dim}, "
            f"d_model={self.d_model}, "
            f"n_layers={len(self.layers)}, "
            f"params={self.count_parameters():,})"
        )
