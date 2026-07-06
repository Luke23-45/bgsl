"""
bgsl/losses.py
--------------
BGSL Loss and all comparison baselines.

Loss hierarchy
--------------
BGSLLoss (full method):
    L_EBGSL = L_state + λ_v · L_velocity + λ_a · L_acceleration + L_mono

    where:
        L_state        = BCE(p_t, g_t) or focal / ASL / weighted-BCE
        L_velocity     = mean(w_t · (ṗ_t - g'_t)²)   [masked, importance-weighted]
        L_acceleration = mean(w_t · (p̈_t - g''_t)²)  [masked, importance-weighted]
        L_mono         = mean(r_t · [max(0, -ṗ_t)]²) [monotonicity penalty]

    Two derivative estimation methods are supported:
        1. "finite_diff"      — first / second order finite differences
        2. "savitzky_golay"   — Savitzky–Golay filter (scipy) with reflection padding

Baselines (all share the same call signature for fair comparison):
    WeightedBCELoss  — class-weighted binary cross-entropy
    FocalLoss        — focal loss (Lin et al. 2017)
    TLSLoss          — Temporal Label Smoothing (Yèche et al. ICML 2023)
    SmoothnessLoss   — BCE + mean((Δp_t)^2)
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
- Importance weights w_t concentrate derivative supervision near τ* - H.
- The monotonicity penalty enforces nondecreasing risk in the alarm window.

Reference: bgsl_math.md (formal definition)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from typing import Dict, Literal, Optional, Tuple

__all__ = [
    "BGSLLoss",
    "UtilityAwareBGSLLoss",
    "BCELoss",
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


def _focal_weight(p: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Focal modulation factor: (1 - p_t)^γ for positives, p_t^γ for negatives."""
    p_t = torch.where(y == 1, p, 1.0 - p)
    return (1.0 - p_t) ** gamma


# ---------------------------------------------------------------------------
# Finite-difference derivative estimators  (backward-compatible defaults)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Savitzky–Golay derivative estimators  (fully differentiable via conv1d)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_sg_kernel(
    window_length: int,
    polyorder: int,
    derivative: int,
) -> torch.Tensor:
    """
    Precompute Savitzky-Golay convolution kernel.

    Coefficients from scipy.signal.savgol_coeffs (dot-product order) are
    used directly as the conv1d kernel — no reversal needed, because both
    the SG formula and F.conv1d cross-correlation with reflect padding are
    identical at all interior points.

    Returns
    -------
    kernel : Tensor[1, 1, window_length]
    """
    from scipy.signal import savgol_coeffs

    coeffs = savgol_coeffs(window_length, polyorder, deriv=derivative, delta=1.0)
    # Reverse: F.conv1d cross-correlation maps kernel[k] → input[t+k-m],
    # while SG dot-order maps coeffs[k] → input[i-k+m] (reversed order).
    kernel = torch.from_numpy(coeffs[::-1].copy()).float()
    return kernel.view(1, 1, -1)


def _savgol_filter_1d(
    p: torch.Tensor,
    window_length: int,
    polyorder: int,
    derivative: int,
) -> torch.Tensor:
    """
    Fully differentiable 1-D Savitzky-Golay filter via convolution.

    Operates entirely in PyTorch: precomputed convolution coefficients are
    applied via ``F.conv1d`` with reflection padding, preserving the
    computational graph for gradient backpropagation.

    Parameters
    ----------
    p : Tensor[B, T]
        Input signals (e.g., predicted probabilities).
    window_length : int
        Filter window length (must be odd).
    polyorder : int
        Polynomial degree (must be < window_length).
    derivative : int
        Derivative order (1 or 2).

    Returns
    -------
    out : Tensor[B, T]
        Filtered derivative estimates with ``requires_grad`` preserved.
    """
    B, T = p.shape
    if T < window_length:
        if derivative == 1:
            return _first_diff(p)
        else:
            return _second_diff(p)

    kernel = _get_sg_kernel(window_length, polyorder, derivative)
    kernel = kernel.to(dtype=p.dtype, device=p.device)

    pad_len = window_length // 2
    p_padded = F.pad(p.unsqueeze(1), (pad_len, pad_len), mode="reflect")
    result = F.conv1d(p_padded, kernel)

    return result.squeeze(1)


def _savgol_first_derivative(
    p: torch.Tensor, window_length: int = 7, polyorder: int = 2
) -> torch.Tensor:
    """ṗ_t = S^{(1)}_{d,m,1}(p)_t.  Fully differentiable."""
    return _savgol_filter_1d(p, window_length, polyorder, derivative=1)


def _savgol_second_derivative(
    p: torch.Tensor, window_length: int = 7, polyorder: int = 2
) -> torch.Tensor:
    """p̈_t = S^{(2)}_{d,m,1}(p)_t.  Fully differentiable."""
    return _savgol_filter_1d(p, window_length, polyorder, derivative=2)


# ---------------------------------------------------------------------------
# Importance weights  (EBGSL)
# ---------------------------------------------------------------------------


def _compute_importance_weights(
    onset_times: torch.Tensor,        # [B]
    is_positive: torch.Tensor,         # [B] bool
    seq_lengths: torch.Tensor,         # [B]
    horizon: float,
    T_max: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Temporal importance weights for velocity / acceleration losses.

    For y_i = 1:
        w_{i,t} = exp( -(t - (τ_i* - H))² / (2σ_w²) ),   σ_w = H/3
    For y_i = 0:
        w_{i,t} = 1

    Returns
    -------
    w : Tensor[B, T]
        Importance weights, zero outside valid lengths.
    """
    B = onset_times.shape[0]
    t = torch.arange(T_max, device=device).float().unsqueeze(0)  # [1, T]

    sigma_w = horizon / 3.0
    onset = onset_times.float().to(device).unsqueeze(1)  # [B, 1]
    center = onset - horizon                               # [B, 1]

    w_pos = torch.exp(-((t - center) ** 2) / (2.0 * sigma_w ** 2))

    is_pos = is_positive.float().to(device).unsqueeze(1)  # [B, 1]
    w = torch.where(is_pos > 0.5, w_pos, torch.ones_like(w_pos))

    # Length mask
    t_idx = torch.arange(T_max, device=device).unsqueeze(0).expand(B, -1)
    length_mask = (t_idx < seq_lengths.unsqueeze(1).to(device)).float()
    w = w * length_mask

    return w


# ---------------------------------------------------------------------------
# Monotonicity penalty  (EBGSL)
# ---------------------------------------------------------------------------


def _compute_monotonicity_penalty(
    p_dot: torch.Tensor,               # [B, T]  prediction velocity
    onset_times: torch.Tensor,          # [B]
    is_positive: torch.Tensor,          # [B] bool
    seq_lengths: torch.Tensor,          # [B]
    horizon: float,
) -> torch.Tensor:
    """
    Monotonicity penalty.

        r_{i,t}^{mono} = m_{i,t} · 1[y_i=1] · 1[τ_i* - H ≤ t ≤ τ_i*]
        L_mono = mean( r^{mono} · [max(0, -ṗ)]² )

    Penalises decreasing risk (ṗ < 0) inside the alarm window for
    event-positive patients. The denominator is max(Σ r^{mono}, 1).
    """
    B, T = p_dot.shape
    device = p_dot.device

    t = torch.arange(T, device=device).float().unsqueeze(0)  # [1, T]

    # Length mask
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    length_mask = (t_idx < seq_lengths.unsqueeze(1).to(device)).float()

    # Positive mask
    pos_mask = is_positive.float().to(device).unsqueeze(1)  # [B, 1]

    # Alarm window: τ* - H ≤ t ≤ τ*
    onset = onset_times.float().to(device).unsqueeze(1)
    window_mask = ((t >= onset - horizon) & (t <= onset)).float()

    # Combined monotonicity region
    r_mono = length_mask * pos_mask * window_mask

    # Penalty: squared negative velocity
    neg_penalty = torch.clamp(-p_dot, min=0.0) ** 2

    weighted = r_mono * neg_penalty
    denom = r_mono.sum().clamp(min=1.0)
    return weighted.sum() / denom


# ---------------------------------------------------------------------------
# Core: EBGSL / BGSL Loss
# ---------------------------------------------------------------------------


class BGSLLoss(nn.Module):
    """
    Biological Gradient Supervised Learning Loss.

    Full EBGSL objective:
        L = L_state + λ_v · L_vel + λ_a · L_acc + λ_mono · L_mono

    Parameters
    ----------
    state_loss : str
        Base classification loss. One of: "bce", "focal", "asl", "weighted_bce".
    velocity_weight : float
        λ_v. Scales the first-derivative supervision term.
    acceleration_weight : float
        λ_a. Scales the second-derivative supervision term.
    monotonicity_weight : float
        λ_mono. Scales the monotonicity penalty. Default 0.0 (BGSL baseline).

    derivative_space : str
        "probability" — derivatives computed on sigmoid(logits). (default)
        "logit"       — derivatives computed on raw logits.
    derivative_method : str
        "finite_diff"    — standard finite differences (default, BGSL baseline).
        "savitzky_golay" — Savitzky–Golay filtered derivatives (EBGSL).
    savgol_window : int
        Window length L = 2m+1 for SG filter. Must be odd. Default 7.
    savgol_order : int
        Polynomial degree d < L for SG filter. Default 2.
    horizon : float
        Prediction horizon H. Used for importance weights and monotonicity
        region. Default 6.0.
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
        monotonicity_weight: float = 0.0,

        derivative_space: Literal["probability", "logit"] = "probability",
        derivative_method: Literal["finite_diff", "savitzky_golay"] = "finite_diff",
        savgol_window: int = 7,
        savgol_order: int = 2,
        horizon: float = 6.0,
        focal_gamma: float = 2.0,
        pos_weight: Optional[float] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        if derivative_method == "savitzky_golay":
            if savgol_window < 3 or savgol_window % 2 == 0:
                raise ValueError(
                    f"savgol_window must be an odd integer >= 3, got {savgol_window}"
                )
            if not (0 < savgol_order < savgol_window):
                raise ValueError(
                    f"savgol_order must satisfy 0 < d < window, "
                    f"got order={savgol_order}, window={savgol_window}"
                )
        self.state_loss_type = state_loss
        self.lambda_v = velocity_weight
        self.lambda_a = acceleration_weight
        self.lambda_mono = monotonicity_weight

        self.derivative_space = derivative_space
        self.derivative_method = derivative_method
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
        self.H = horizon
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        vel_targets: Optional[torch.Tensor] = None,
        acc_targets: Optional[torch.Tensor] = None,
        vel_mask: Optional[torch.Tensor] = None,
        acc_mask: Optional[torch.Tensor] = None,
        # EBGSL extra fields
        onset_times: Optional[torch.Tensor] = None,
        is_positive: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        logits : Tensor[B, T]
            Raw model output (pre-sigmoid).
        targets : Tensor[B, T]
            g_t from SoftOnsetTarget (soft targets for state loss).
        mask : Tensor[B, T]
            1 = valid timestep, 0 = padding. Applied to state loss.
        vel_targets : Tensor[B, T] or None
            g'_t from SoftOnsetTarget. Defaults to zeros if None.
        acc_targets : Tensor[B, T] or None
            g''_t from SoftOnsetTarget. Defaults to zeros if None.
        vel_mask : Tensor[B, T] or None
            Mask for velocity loss (excludes t=0). Defaults to mask with t=0 zeroed.
        acc_mask : Tensor[B, T] or None
            Mask for acceleration loss (excludes t=0,1). Defaults to mask with t=0,1 zeroed.
        onset_times : Tensor[B] or None
            Event onset times τ*. Required for importance weights and monotonicity.
        is_positive : Tensor[B] or None
            Boolean, whether event is observed. Required for importance weights and monotonicity.
        seq_lengths : Tensor[B] or None
            Valid sequence lengths. Required for importance weights and monotonicity.

        Returns
        -------
        dict:
            "loss"         : total scalar
            "state"        : state loss scalar
            "velocity"     : velocity loss scalar
            "acceleration" : acceleration loss scalar
            "monotonicity" : monotonicity penalty scalar (0 if disabled)
        """
        mask = mask.float()
        B, T = logits.shape

        # Default derivative masks from state mask
        if vel_mask is None:
            vel_mask = mask.clone()
        if acc_mask is None:
            acc_mask = mask.clone()

        vel_mask = vel_mask.float()
        acc_mask = acc_mask.float()

        # Finite differences produce invalid derivatives at boundary indices
        if self.derivative_method == "finite_diff":
            vel_mask = vel_mask.clone()
            acc_mask = acc_mask.clone()
            if T > 0:
                vel_mask[:, 0] = 0.0
            if T > 1:
                acc_mask[:, :2] = 0.0

        # Probabilities
        probs = torch.sigmoid(logits)  # [B, T]

        # ---------------------------------------------------------------
        # State loss
        # ---------------------------------------------------------------
        state_loss = self._compute_state_loss(logits, probs, targets, mask)

        # ---------------------------------------------------------------
        # Derivative losses
        # ---------------------------------------------------------------
        # Choose space for predicted derivatives
        if self.derivative_space == "probability":
            pred_signal = probs
        else:
            pred_signal = logits

        # Compute prediction derivatives
        dp = self._compute_first_derivative(pred_signal)
        d2p = self._compute_second_derivative(pred_signal)

        if self.training and pred_signal.requires_grad and not dp.requires_grad:
            raise RuntimeError(
                f"Computed derivatives do not require gradient "
                f"(derivative_method={self.derivative_method!r}). "
                f"Training cannot proceed — no gradient flows through derivative losses."
            )

        # Default derivative targets to zero tensors if not provided
        if vel_targets is None:
            vel_targets = torch.zeros_like(dp)
        if acc_targets is None:
            acc_targets = torch.zeros_like(d2p)

        # --- Importance weights ---
        use_weights = (
            onset_times is not None
            and is_positive is not None
            and seq_lengths is not None
        )
        if use_weights:
            w = _compute_importance_weights(
                onset_times, is_positive, seq_lengths,
                self.H, T, logits.device,
            )
            vel_weighted_mask = vel_mask * w
            acc_weighted_mask = acc_mask * w
        else:
            vel_weighted_mask = vel_mask
            acc_weighted_mask = acc_mask

        # Velocity loss: weighted MSE between predicted and target first derivatives
        vel_loss = _masked_mean((dp - vel_targets) ** 2, vel_weighted_mask)

        # Acceleration loss: weighted MSE between predicted and target second derivatives
        acc_loss = _masked_mean((d2p - acc_targets) ** 2, acc_weighted_mask)

        # ---------------------------------------------------------------
        # Monotonicity penalty
        # ---------------------------------------------------------------
        mono_penalty = torch.tensor(0.0, device=logits.device)
        if self.lambda_mono > 0.0 and use_weights:
            p_dot_mono = self._compute_first_derivative(probs)
            mono_penalty = _compute_monotonicity_penalty(
                p_dot_mono, onset_times, is_positive,
                seq_lengths, self.H,
            )

        # ---------------------------------------------------------------
        # Total loss
        # ---------------------------------------------------------------
        total = (
            state_loss
            + self.lambda_v * vel_loss
            + self.lambda_a * acc_loss
            + self.lambda_mono * mono_penalty
        )

        return {
            "loss":         total,
            "state":        state_loss,
            "velocity":     vel_loss,
            "acceleration": acc_loss,
            "monotonicity": mono_penalty,
        }

    # ------------------------------------------------------------------
    # Derivative computation dispatchers
    # ------------------------------------------------------------------

    def _compute_first_derivative(self, p: torch.Tensor) -> torch.Tensor:
        if self.derivative_method == "savitzky_golay":
            return _savgol_first_derivative(p, self.savgol_window, self.savgol_order)
        return _first_diff(p)

    def _compute_second_derivative(self, p: torch.Tensor) -> torch.Tensor:
        if self.derivative_method == "savitzky_golay":
            return _savgol_second_derivative(p, self.savgol_window, self.savgol_order)
        return _second_diff(p)

    # ------------------------------------------------------------------
    # State loss variants
    # ------------------------------------------------------------------

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
                pw = torch.where(
                    targets > 0.5,
                    torch.full_like(targets, self.pos_weight),
                    pw,
                )
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
            margin = 0.05
            probs_neg = (probs - margin).clamp(min=0.0)
            bce_pos = F.binary_cross_entropy(probs, targets, reduction="none")
            bce_neg = F.binary_cross_entropy(
                probs_neg, targets.clamp(max=0.0), reduction="none"
            )
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
            f"λ_mono={self.lambda_mono}, "
            f"space={self.derivative_space}, "
            f"deriv={self.derivative_method})"
        )


# ---------------------------------------------------------------------------
# Extension: Utility-Aware BGSL Loss
# ---------------------------------------------------------------------------


class UtilityAwareBGSLLoss(BGSLLoss):
    """
    BGSL + differentiable PhysioNet utility surrogate.

    Adds an auxiliary loss term that approximates the PhysioNet 2019
    clinical utility score using sigmoid-relaxed soft decisions and
    the official piecewise-linear reward weights.

    Total objective:
        L = L_state + λ_v · L_vel + λ_a · L_acc + λ_mono · L_mono
            + λ_u · L_utility

    Where L_utility = 1 - mean(normalized_soft_utility) across patients.

    This bridges the gap between ranking-oriented training (BCE / derivatives)
    and the clinical alerting value that matters for deployment. The gradient
    from L_utility directly penalises:
    - Missed detections in the optimal alarm window
    - False alarms in the quiescent period
    - Late predictions after the onset window closes

    Parameters
    ----------
    utility_weight : float
        λ_u. Scales the utility surrogate penalty. Default 0.5.
    utility_temperature : float
        Temperature for the soft decision sigmoid. Lower = sharper.
        Default 1.0.
    dt_early, dt_optimal, dt_late : float
        PhysioNet utility window offsets relative to t_sepsis. Defaults
        follow the official challenge specification.
    max_u_tp, min_u_fn, u_fp, u_tn : float
        PhysioNet utility reward/penalty values.
    **kwargs
        All remaining arguments forwarded to BGSLLoss.
    """

    def __init__(
        self,
        utility_weight: float = 0.5,
        utility_temperature: float = 1.0,
        dt_early: float = -12.0,
        dt_optimal: float = -6.0,
        dt_late: float = 3.0,
        max_u_tp: float = 1.0,
        min_u_fn: float = -2.0,
        u_fp: float = -0.05,
        u_tn: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lambda_u = utility_weight
        self.utility_temperature = utility_temperature
        # PhysioNet utility schedule constants
        self.dt_early = dt_early
        self.dt_optimal = dt_optimal
        self.dt_late = dt_late
        self.max_u_tp = max_u_tp
        self.min_u_fn = min_u_fn
        self.u_fp = u_fp
        self.u_tn = u_tn

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        vel_targets: Optional[torch.Tensor] = None,
        acc_targets: Optional[torch.Tensor] = None,
        vel_mask: Optional[torch.Tensor] = None,
        acc_mask: Optional[torch.Tensor] = None,
        onset_times: Optional[torch.Tensor] = None,
        is_positive: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Get base BGSL losses
        result = super().forward(
            logits, targets, mask,
            vel_targets=vel_targets,
            acc_targets=acc_targets,
            vel_mask=vel_mask,
            acc_mask=acc_mask,
            onset_times=onset_times,
            is_positive=is_positive,
            seq_lengths=seq_lengths,
            **kwargs,
        )

        # Compute utility surrogate if we have onset metadata
        utility_loss = torch.tensor(0.0, device=logits.device)
        if (
            onset_times is not None
            and is_positive is not None
            and seq_lengths is not None
            and self.lambda_u > 0.0
        ):
            utility_loss = self._compute_utility_surrogate(
                logits, onset_times, is_positive, seq_lengths,
            )

        result["utility"] = utility_loss
        result["loss"] = result["loss"] + self.lambda_u * utility_loss

        return result

    def _compute_utility_surrogate(
        self,
        logits: torch.Tensor,
        onset_times: torch.Tensor,
        is_positive: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable surrogate of the PhysioNet 2019 utility score.

        For each patient i and timestep t, computes the soft per-step
        utility using sigmoid-relaxed decisions:

            soft_pred = σ(logits / τ_decision)

        Then applies the official piecewise-linear reward structure:
            - For positive patients: time-dependent TP reward / FN penalty
            - For negative patients: FP penalty / TN reward

        Returns L_utility = 1 - mean(normalized_utility) so that
        minimizing L_utility maximizes the PhysioNet score.
        """
        B, T = logits.shape
        device = logits.device

        # Soft predictions (differentiable relaxation of hard threshold)
        soft_pred = torch.sigmoid(logits / max(self.utility_temperature, 1e-6))

        t = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)  # [1, T]

        # Length mask
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        length_mask = (t_idx < seq_lengths.unsqueeze(1).to(device)).float()  # [B, T]

        onset = onset_times.float().to(device).unsqueeze(1)  # [B, 1]
        is_pos = is_positive.float().to(device).unsqueeze(1)  # [B, 1]

        # Per-timestep utility weights (piecewise-linear schedule from PhysioNet)
        # dt = t - t_sepsis  (negative = before onset, positive = after)
        dt = t - onset  # [B, T]

        # --- True Positive reward (for positive patients, when model predicts 1) ---
        # Piecewise linear: rises from dt_early to dt_optimal, falls from dt_optimal to dt_late
        m1 = self.max_u_tp / (self.dt_optimal - self.dt_early)
        b1 = -m1 * self.dt_early
        m2 = -self.max_u_tp / (self.dt_late - self.dt_optimal)
        b2 = -m2 * self.dt_late

        # Rising ramp: [dt_early, dt_optimal]
        u_tp_rising = m1 * dt + b1
        # Falling ramp: (dt_optimal, dt_late]
        u_tp_falling = m2 * dt + b2

        u_tp = torch.where(
            dt <= self.dt_optimal,
            u_tp_rising,
            u_tp_falling,
        )
        # Clamp: u_tp should only apply in [dt_early, dt_late], minimum is u_fp
        in_window = ((dt >= self.dt_early) & (dt <= self.dt_late)).float()
        u_tp = torch.clamp(u_tp, min=self.u_fp)
        u_tp = u_tp * in_window + self.u_fp * (1.0 - in_window)
        # Outside the window but before onset: u_fp for prediction=1
        before_window = (dt < self.dt_early).float()
        u_tp = torch.where(
            before_window.bool(),
            torch.full_like(u_tp, self.u_fp),
            u_tp,
        )
        # After the window: 0 (no reward/penalty — observation has ended in PhysioNet logic)
        after_window = (dt > self.dt_late).float()
        u_tp = u_tp * (1.0 - after_window) + 0.0 * after_window

        # --- False Negative penalty (for positive patients, when model predicts 0) ---
        m3 = self.min_u_fn / (self.dt_late - self.dt_optimal)
        b3 = -m3 * self.dt_optimal
        u_fn_penalty = m3 * dt + b3

        # Only applies in (dt_optimal, dt_late], 0 otherwise
        in_fn_window = ((dt > self.dt_optimal) & (dt <= self.dt_late)).float()
        u_fn = u_fn_penalty * in_fn_window  # [B, T]

        # --- Combine for positive patients ---
        # U_pos(t) = soft_pred * u_tp + (1 - soft_pred) * u_fn
        u_positive = soft_pred * u_tp + (1.0 - soft_pred) * u_fn  # [B, T]

        # --- For negative patients ---
        # Predict 1 (false alarm) → u_fp
        # Predict 0 (correct)     → u_tn
        u_negative = soft_pred * self.u_fp + (1.0 - soft_pred) * self.u_tn  # [B, T]

        # --- Select per-patient utility ---
        u_per_step = is_pos * u_positive + (1.0 - is_pos) * u_negative  # [B, T]

        # Apply length mask
        u_per_step = u_per_step * length_mask

        # --- Sum per patient ---
        u_per_patient = u_per_step.sum(dim=1)  # [B]

        # --- Compute best possible and inaction utility for normalization ---
        # Best: predict 1 in [dt_early, dt_late] for positives, 0 for negatives
        best_pred = in_window * is_pos  # [B, T]
        u_best_step = is_pos * (best_pred * u_tp + (1.0 - best_pred) * u_fn)
        u_best_step = u_best_step + (1.0 - is_pos) * self.u_tn
        u_best_step = u_best_step * length_mask
        u_best_per_patient = u_best_step.sum(dim=1)

        # Inaction: predict 0 everywhere
        u_inaction_step = is_pos * u_fn + (1.0 - is_pos) * self.u_tn
        u_inaction_step = u_inaction_step * length_mask
        u_inaction_per_patient = u_inaction_step.sum(dim=1)

        # Normalized utility per patient: (U_obs - U_inaction) / (U_best - U_inaction)
        denom = (u_best_per_patient - u_inaction_per_patient).clamp(min=1e-8)
        normalized = (u_per_patient - u_inaction_per_patient) / denom

        # Only include patients where normalization is meaningful
        valid = (u_best_per_patient - u_inaction_per_patient).abs() > 1e-8
        if valid.any():
            mean_utility = normalized[valid].mean()
        else:
            mean_utility = torch.tensor(0.0, device=device)

        # L_utility = 1 - mean_utility (minimize this to maximize utility)
        return 1.0 - mean_utility

    def __repr__(self) -> str:
        return (
            f"UtilityAwareBGSLLoss("
            f"state={self.state_loss_type}, "
            f"λ_v={self.lambda_v}, "
            f"λ_a={self.lambda_a}, "
            f"λ_mono={self.lambda_mono}, "
            f"λ_u={self.lambda_u}, "
            f"space={self.derivative_space}, "
            f"deriv={self.derivative_method})"
        )





class BCELoss(nn.Module):
    """Plain binary cross-entropy with logits."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_zero", torch.zeros(1))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        per_step = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = _masked_mean(per_step, mask)
        return {"loss": loss, "state": loss, "velocity": self._zero, "acceleration": self._zero}


class WeightedBCELoss(nn.Module):
    """Class-weighted binary cross-entropy."""

    def __init__(self, pos_weight: float = 10.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.register_buffer("_zero", torch.zeros(1))

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
        return {
            "loss": loss,
            "state": loss,
            "velocity": self._zero,
            "acceleration": self._zero,
        }


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
        self.register_buffer("_zero", torch.zeros(1))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal_w = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = torch.where(
                targets > 0.5,
                torch.full_like(targets, self.alpha),
                torch.full_like(targets, 1.0 - self.alpha),
            )
            focal_w = alpha_t * focal_w

        per_step = focal_w * bce
        loss = _masked_mean(per_step, mask)
        return {
            "loss": loss,
            "state": loss,
            "velocity": self._zero,
            "acceleration": self._zero,
        }


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
        self.register_buffer("_zero", torch.zeros(1))

    def build_tls_targets(
        self,
        onset_hours: torch.Tensor,
        seq_lengths: torch.Tensor,
        device: torch.device,
        T_max: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Construct TLS soft targets.

        Parameters
        ----------
        T_max : int or None
            Time dimension override. If None, uses ``seq_lengths.max()``.
            Set this when the model output is padded to a longer dimension
            than the batch's own max sequence length.

        Shape: [B, T_max]
        """
        B = onset_hours.shape[0]
        if T_max is None:
            T_max = int(seq_lengths.max().item())
        t = torch.arange(T_max, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T]
        onset = onset_hours.float().to(device).unsqueeze(1)  # [B, 1]

        time_to_onset = onset - t  # [B, T], positive = still before onset

        tls = torch.clamp(1.0 - time_to_onset / self.alpha, 0.0, 1.0)

        is_positive = (onset >= 0).float()
        tls = tls * is_positive

        t_idx = torch.arange(T_max, device=device).unsqueeze(0).expand(B, -1)
        pad_mask = (t_idx < seq_lengths.unsqueeze(1).to(device)).float()
        tls = tls * pad_mask

        return tls

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
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
        return {
            "loss": loss,
            "state": loss,
            "velocity": self._zero,
            "acceleration": self._zero,
        }


# ---------------------------------------------------------------------------
# Baseline: Smoothness penalty (BCE + mean((Δp)²))
# ---------------------------------------------------------------------------


class SmoothnessLoss(nn.Module):
    """
    BCE + mean squared first difference of predicted probability.

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
        self.register_buffer("_zero", torch.zeros(1))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
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

        per_step_bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ) * pw
        state_loss = _masked_mean(per_step_bce, mask)

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
            "acceleration": self._zero,
        }


# ---------------------------------------------------------------------------
# Baseline: Total Variation penalty (BCE + mean(|Δp|))
# ---------------------------------------------------------------------------


class TotalVariationLoss(nn.Module):
    """
    BCE + mean absolute first difference of predicted probability.

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
        self.register_buffer("_zero", torch.zeros(1))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
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

        per_step_bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ) * pw
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
            "acceleration": self._zero,
        }
