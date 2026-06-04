"""
bgsl/losses.py
--------------
BGSL Loss and all comparison baselines.

Loss hierarchy
--------------
BGSLLoss (full method):
    L_BGSL = L_state + λ_v · L_velocity + λ_a · L_acceleration

    where:
        L_state        = BCE(p_t, g_t) or focal / ASL / weighted-BCE
        L_velocity     = mean((Δp_t - Δg_t)²)       [masked]
        L_acceleration = mean((Δ²p_t - Δ²g_t)²)     [masked]

Baselines (all share the same call signature for fair comparison):
    WeightedBCELoss  — class-weighted binary cross-entropy
    FocalLoss        — focal loss (Lin et al. 2017)
    TLSLoss          — Temporal Label Smoothing (Yèche et al. ICML 2023)
    SmoothnessLoss   — BCE + mean((Δp_t)²)
    TotalVariationLoss — BCE + mean(|Δp_t|)

All losses accept:
    logits : Tensor[B, T]    — raw model outputs
    targets : Tensor[B, T]   — binary hard or soft labels
    mask   : Tensor[B, T]    — 1 = valid, 0 = padded / ignore

Key design choices
------------------
- Derivatives are computed on *probabilities* by default (sigmoid(logits)).
  Rationale: probability space is bounded [0, 1], so derivative magnitudes
  are clinically interpretable. Logit-space is a sensitivity analysis.
- Masking is applied BEFORE averaging, never after (avoids batch-size bias
  from variable-length sequences).
- lambda_v and lambda_a default to small values to avoid overpowering the
  classification signal under class imbalance.

Reference: BGSL plan.md §3.3, §9.1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Optional

__all__ = [
    "BGSLLoss",
    "WeightedBCELoss",
    "FocalLoss",
    "TLSLoss",
    "SmoothnessLoss",
    "TotalVariationLoss",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean of `values` over positions where `mask == 1`.
    Returns scalar 0 if mask is entirely zero (avoids NaN).
    """
    mask = mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _first_diff(p: torch.Tensor) -> torch.Tensor:
    """Δp_t = p_t - p_{t-1}. Δp_0 = 0. Shape: [B, T]"""
    p_prev = F.pad(p[:, :-1], (1, 0), value=0.0)
    dp = p - p_prev
    dp[:, 0] = 0.0
    return dp


def _second_diff(p: torch.Tensor) -> torch.Tensor:
    """Δ²p_t = p_t - 2p_{t-1} + p_{t-2}. Δ²p_0 = Δ²p_1 = 0. Shape: [B, T]"""
    p1 = F.pad(p[:, :-1], (1, 0), value=0.0)
    p2 = F.pad(p[:, :-2], (2, 0), value=0.0)
    d2p = p - 2.0 * p1 + p2
    d2p[:, :2] = 0.0
    return d2p


def _focal_weight(p: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Focal modulation factor: (1 - p_t)^γ for positives, p_t^γ for negatives."""
    p_t = torch.where(y == 1, p, 1.0 - p)
    return (1.0 - p_t) ** gamma


# ---------------------------------------------------------------------------
# Core: BGSL Loss
# ---------------------------------------------------------------------------

class BGSLLoss(nn.Module):
    """
    Biological Gradient Supervised Learning Loss.

    L_BGSL = L_state + λ_v · L_velocity + λ_a · L_acceleration

    Parameters
    ----------
    state_loss : str
        Base classification loss. One of: "bce", "focal", "asl", "weighted_bce".
    velocity_weight : float
        λ_v. Scales the first-derivative supervision term.
        Recommended search: {0.01, 0.05, 0.1, 0.25, 0.5}.
    acceleration_weight : float
        λ_a. Scales the second-derivative supervision term.
        Recommended search: {0.0, 0.01, 0.05, 0.1, 0.25}.
    derivative_space : str
        "probability" — derivatives computed on sigmoid(logits). (default)
        "logit"       — derivatives computed on raw logits.
    focal_gamma : float
        Gamma for focal loss (only used when state_loss="focal"). Default 2.0.
    pos_weight : float or None
        Positive-class weight for weighted_bce. If None, no weighting.
    reduction : str
        "mean" | "sum" | "none".
    """

    def __init__(
        self,
        state_loss: Literal["bce", "focal", "asl", "weighted_bce"] = "bce",
        velocity_weight: float = 0.1,
        acceleration_weight: float = 0.05,
        derivative_space: Literal["probability", "logit"] = "probability",
        focal_gamma: float = 2.0,
        pos_weight: Optional[float] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.state_loss_type = state_loss
        self.lambda_v = velocity_weight
        self.lambda_a = acceleration_weight
        self.derivative_space = derivative_space
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
        vel_targets: torch.Tensor,
        acc_targets: torch.Tensor,
        mask: torch.Tensor,
        vel_mask: Optional[torch.Tensor] = None,
        acc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        logits : Tensor[B, T]
            Raw model output (pre-sigmoid).
        soft_targets : Tensor[B, T]
            g_t from SoftOnsetTarget.
        vel_targets : Tensor[B, T]
            Δg_t from SoftOnsetTarget.
        acc_targets : Tensor[B, T]
            Δ²g_t from SoftOnsetTarget.
        mask : Tensor[B, T]
            1 = valid timestep, 0 = padding. Applied to state loss.
        vel_mask : Tensor[B, T] or None
            Mask for velocity loss (excludes t=0). Defaults to mask with t=0 zeroed.
        acc_mask : Tensor[B, T] or None
            Mask for acceleration loss (excludes t=0,1). Defaults to mask with t=0,1 zeroed.

        Returns
        -------
        dict:
            "loss"         : total scalar
            "state"        : state loss scalar
            "velocity"     : velocity loss scalar
            "acceleration" : acceleration loss scalar
        """
        mask = mask.float()
        B, T = logits.shape

        # Default derivative masks from state mask
        if vel_mask is None:
            vel_mask = mask.clone()
            if T > 0:
                vel_mask[:, 0] = 0.0
        if acc_mask is None:
            acc_mask = mask.clone()
            if T > 1:
                acc_mask[:, :2] = 0.0

        vel_mask = vel_mask.float()
        acc_mask = acc_mask.float()

        # Probabilities
        probs = torch.sigmoid(logits)  # [B, T]

        # ---------------------------------------------------------------
        # State loss
        # ---------------------------------------------------------------
        state_loss = self._compute_state_loss(logits, probs, soft_targets, mask)

        # ---------------------------------------------------------------
        # Derivative losses
        # ---------------------------------------------------------------
        # Choose space for predicted derivatives
        if self.derivative_space == "probability":
            pred_signal = probs
        else:
            pred_signal = logits

        dp = _first_diff(pred_signal)    # [B, T]
        d2p = _second_diff(pred_signal)  # [B, T]

        # Velocity loss: MSE between predicted and target first derivatives
        vel_loss = _masked_mean((dp - vel_targets) ** 2, vel_mask)

        # Acceleration loss: MSE between predicted and target second derivatives
        acc_loss = _masked_mean((d2p - acc_targets) ** 2, acc_mask)

        # ---------------------------------------------------------------
        # Total loss
        # ---------------------------------------------------------------
        total = state_loss + self.lambda_v * vel_loss + self.lambda_a * acc_loss

        return {
            "loss":         total,
            "state":        state_loss,
            "velocity":     vel_loss,
            "acceleration": acc_loss,
        }

    def _compute_state_loss(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.state_loss_type == "bce":
            per_step = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

        elif self.state_loss_type == "weighted_bce":
            pw = torch.ones_like(targets)
            if self.pos_weight is not None:
                pw = torch.where(targets > 0.5, torch.full_like(targets, self.pos_weight), pw)
            per_step = F.binary_cross_entropy_with_logits(
                logits, targets, weight=None, reduction="none"
            ) * pw

        elif self.state_loss_type == "focal":
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            focal_w = _focal_weight(probs, targets, self.focal_gamma)
            per_step = focal_w * bce

        elif self.state_loss_type == "asl":
            # Asymmetric Loss (Ridnik et al. 2021) — simplified version
            # Positive shift: normal BCE
            # Negative shift: shift probability down by margin
            margin = 0.05
            probs_neg = (probs - margin).clamp(min=0.0)
            bce_pos = F.binary_cross_entropy(probs, targets, reduction="none")
            bce_neg = F.binary_cross_entropy(probs_neg, targets.clamp(max=0.0), reduction="none")
            per_step = torch.where(targets > 0.5, bce_pos, bce_neg)

        else:
            raise ValueError(f"Unknown state_loss type: {self.state_loss_type!r}")

        return _masked_mean(per_step, mask)

    def __repr__(self) -> str:
        return (
            f"BGSLLoss("
            f"state={self.state_loss_type}, "
            f"λ_v={self.lambda_v}, "
            f"λ_a={self.lambda_a}, "
            f"space={self.derivative_space})"
        )


# ---------------------------------------------------------------------------
# Baseline: Weighted BCE
# ---------------------------------------------------------------------------

class WeightedBCELoss(nn.Module):
    """
    Class-weighted binary cross-entropy.

    Parameters
    ----------
    pos_weight : float
        Weight multiplier for positive-class samples.
    """

    def __init__(self, pos_weight: float = 10.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        pw = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.pos_weight),
            torch.ones_like(targets),
        )
        per_step = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ) * pw
        loss = _masked_mean(per_step, mask)
        return {"loss": loss, "state": loss, "velocity": torch.zeros(1), "acceleration": torch.zeros(1)}


# ---------------------------------------------------------------------------
# Baseline: Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for dense prediction (Lin et al. 2017).

    Parameters
    ----------
    gamma : float
        Focusing parameter. 0 → standard BCE.
    alpha : float or None
        Balance factor for positive class.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Modulating factor
        p_t = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal_w = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = torch.where(targets > 0.5,
                                  torch.full_like(targets, self.alpha),
                                  torch.full_like(targets, 1.0 - self.alpha))
            focal_w = alpha_t * focal_w

        per_step = focal_w * bce
        loss = _masked_mean(per_step, mask)
        return {"loss": loss, "state": loss, "velocity": torch.zeros(1), "acceleration": torch.zeros(1)}


# ---------------------------------------------------------------------------
# Baseline: Temporal Label Smoothing (TLS)
# Reference: Yèche et al. "Temporal Label Smoothing for Early Event Prediction"
#            ICML 2023.
# ---------------------------------------------------------------------------

class TLSLoss(nn.Module):
    """
    Temporal Label Smoothing for early-event prediction.

    Smoothes the binary alarm label backward in time:
        y_tls_t = min(1, max(0, 1 - (t* - t) / alpha))

    where alpha controls the ramp-up speed (in hours).

    This is the closest published direct baseline to BGSL.
    It modifies the *pointwise target* but does NOT supervise derivatives.

    Parameters
    ----------
    alpha : float
        Smoothing window in hours. Larger → gentler ramp.
    pos_weight : float or None
        Optional positive class weighting applied on top of TLS labels.
    """

    def __init__(self, alpha: float = 6.0, pos_weight: Optional[float] = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight

    def build_tls_targets(
        self,
        onset_hours: torch.Tensor,
        seq_lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Construct TLS soft targets.

        Shape: [B, T_max]
        """
        B = onset_hours.shape[0]
        T_max = int(seq_lengths.max().item())
        t = torch.arange(T_max, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]
        onset = onset_hours.float().to(device).unsqueeze(1)  # [B, 1]

        # Time before onset (negative means before onset)
        time_to_onset = onset - t  # [B, T], positive = still before onset

        # TLS target: 1 at onset, ramps linearly to 0 at t = t* - alpha
        tls = torch.clamp(1.0 - time_to_onset / self.alpha, 0.0, 1.0)

        # Negative patients: all zero
        is_positive = (onset >= 0).float()
        tls = tls * is_positive

        # Zero out padding
        t_idx = torch.arange(T_max, device=device).unsqueeze(0).expand(B, -1)
        pad_mask = (t_idx < seq_lengths.unsqueeze(1).to(device)).float()
        tls = tls * pad_mask

        return tls

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,  # TLS targets (pre-computed)
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.pos_weight is not None:
            pw = torch.where(
                targets > 0.5,
                torch.full_like(targets, self.pos_weight),
                torch.ones_like(targets),
            )
        else:
            pw = torch.ones_like(targets)

        per_step = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ) * pw
        loss = _masked_mean(per_step, mask)
        return {"loss": loss, "state": loss, "velocity": torch.zeros(1), "acceleration": torch.zeros(1)}


# ---------------------------------------------------------------------------
# Baseline: Smoothness penalty (BCE + mean((Δp)²))
# ---------------------------------------------------------------------------

class SmoothnessLoss(nn.Module):
    """
    BCE + mean squared first difference of predicted probability.

    Smoothness regularization says: do not change risk too much.
    BGSL says: change risk in the correct direction and at the correct time.

    This baseline isolates the stability-without-direction distinction.

    Parameters
    ----------
    smoothness_weight : float
        Weight on the smoothness penalty.
    pos_weight : float or None
        Optional positive class weighting for BCE.
    """

    def __init__(
        self,
        smoothness_weight: float = 0.1,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.lambda_s = smoothness_weight
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # State loss
        if self.pos_weight is not None:
            pw = torch.where(targets > 0.5,
                             torch.full_like(targets, self.pos_weight),
                             torch.ones_like(targets))
        else:
            pw = torch.ones_like(targets)

        per_step_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * pw
        state_loss = _masked_mean(per_step_bce, mask)

        # Smoothness penalty on probability
        probs = torch.sigmoid(logits)
        dp = _first_diff(probs)
        vel_mask = mask.clone().float()
        if logits.shape[1] > 0:
            vel_mask[:, 0] = 0.0
        smooth_penalty = _masked_mean(dp ** 2, vel_mask)

        total = state_loss + self.lambda_s * smooth_penalty
        return {
            "loss": total,
            "state": state_loss,
            "velocity": smooth_penalty,
            "acceleration": torch.zeros(1, device=logits.device),
        }


# ---------------------------------------------------------------------------
# Baseline: Total Variation penalty (BCE + mean(|Δp|))
# ---------------------------------------------------------------------------

class TotalVariationLoss(nn.Module):
    """
    BCE + mean absolute first difference of predicted probability.

    Total variation is a sparsity-promoting alternative to smoothness.
    It penalizes any changes in the risk trajectory.

    Parameters
    ----------
    tv_weight : float
        Weight on the TV penalty.
    pos_weight : float or None
        Optional positive class weighting for BCE.
    """

    def __init__(
        self,
        tv_weight: float = 0.1,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.lambda_tv = tv_weight
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.pos_weight is not None:
            pw = torch.where(targets > 0.5,
                             torch.full_like(targets, self.pos_weight),
                             torch.ones_like(targets))
        else:
            pw = torch.ones_like(targets)

        per_step_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * pw
        state_loss = _masked_mean(per_step_bce, mask)

        probs = torch.sigmoid(logits)
        dp = _first_diff(probs)
        vel_mask = mask.clone().float()
        if logits.shape[1] > 0:
            vel_mask[:, 0] = 0.0
        tv_penalty = _masked_mean(dp.abs(), vel_mask)

        total = state_loss + self.lambda_tv * tv_penalty
        return {
            "loss": total,
            "state": state_loss,
            "velocity": tv_penalty,
            "acceleration": torch.zeros(1, device=logits.device),
        }
