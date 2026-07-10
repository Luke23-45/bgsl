"""Synthetic benchmark for evaluating soft-target early-event prediction losses.

The benchmark adds a pre-onset signal in the features starting at
``τ - 2H`` (12 hours before the event), ramping linearly before the
hard-label window.  This gives soft-target methods (TLS, BGSL) genuine
early signal to exploit that BCE's step-function target cannot capture.

The benchmark compares:
- BCE
- TLS (Temporal Label Smoothing, Yèche et al. ICML 2023)
- BGSL state-only
- BGSL with velocity supervision

Stress conditions:
- 5 seeds
- default velocity weight
- higher velocity weight sweep
- longer training check

An ``early_auroc`` metric measures patient-level discrimination on
the pre-warning-window timesteps where BCE has no positive label.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bgsl.core.common.losses import BCELoss, BGSLLoss, TLSLoss
from bgsl.core.common.targets import BaseSoftOnsetTarget
from bgsl.core.sepsis.metrics import PatientPrediction, SepsisMetrics
from bgsl.models.gru import GRUPredictor


METHODS = ("BCE", "TLS", "BGSL-state", "BGSL+vel", "Kalman")
SCENARIO_SEEDS = {"signal": 101, "null": 202}
STRESS_SEEDS = (101, 202, 303, 404, 505)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_indices(
    n: int,
    *,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_frac <= 0.0 or val_frac <= 0.0 or train_frac + val_frac >= 1.0:
        raise ValueError("Invalid split fractions.")

    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _ar1_noise(
    length: int,
    *,
    phi: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    eps = rng.normal(0.0, scale, size=length)
    out = np.zeros(length, dtype=np.float32)
    out[0] = eps[0]
    coeff = float(np.sqrt(max(1.0 - phi * phi, 0.0)))
    for t in range(1, length):
        out[t] = phi * out[t - 1] + coeff * eps[t]
    return out


class SyntheticICUDataset:
    """Synthetic ICU cohort with weak signal, nuisance channels, and missingness."""

    def __init__(
        self,
        *,
        mode: str,
        n_patients: int = 800,
        n_features: int = 16,
        max_length: int = 48,
        min_length: int = 18,
        horizon: float = 6.0,
        early_signal_horizon: float = 12.0,
        early_signal_amplitude: float = 1.0,
        pos_ratio: float = 0.28,
        informative_channels: int = 4,
        missing_rate: float = 0.18,
        signal_strength: float = 0.45,
        seed: int = 101,
    ) -> None:
        if mode not in {"signal", "null"}:
            raise ValueError("mode must be 'signal' or 'null'.")
        if not (0.0 < pos_ratio < 1.0):
            raise ValueError("pos_ratio must be in (0, 1).")
        if not (0.0 <= missing_rate < 1.0):
            raise ValueError("missing_rate must be in [0, 1).")
        if informative_channels < 1 or informative_channels > n_features:
            raise ValueError("informative_channels must be in [1, n_features].")
        if signal_strength <= 0.0:
            raise ValueError("signal_strength must be positive.")

        self.mode = mode
        self.n_patients = n_patients
        self.n_features = n_features
        self.max_length = max_length
        self.min_length = min_length
        self.horizon = horizon
        self.early_signal_horizon = early_signal_horizon
        self.early_signal_amplitude = early_signal_amplitude
        self.pos_ratio = pos_ratio
        self.informative_channels = informative_channels
        self.missing_rate = missing_rate
        self.signal_strength = signal_strength
        self.seed = seed

    def generate(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        n = self.n_patients
        t_max = self.max_length

        seq_lengths = rng.integers(self.min_length, t_max + 1, size=n, dtype=np.int64)
        is_positive = rng.random(n) < self.pos_ratio
        onset_times = np.full(n, -1.0, dtype=np.float32)

        for i in range(n):
            if not is_positive[i]:
                continue
            lower = int(np.ceil(self.horizon)) + 2
            upper = int(seq_lengths[i]) - 2
            if upper <= lower:
                is_positive[i] = False
                continue
            onset_times[i] = float(rng.integers(lower, upper + 1))

        target_builder = BaseSoftOnsetTarget(horizon=self.horizon, tau=2.5, kappa=1.0)
        target_batch = target_builder(
            torch.tensor(onset_times, dtype=torch.float32),
            torch.tensor(seq_lengths, dtype=torch.long),
            torch.device("cpu"),
        )
        soft_targets = target_batch["soft_target"].cpu().numpy().astype(np.float32)
        vel_targets = target_batch["velocity_target"].cpu().numpy().astype(np.float32)
        hard_targets = target_batch["hard_target"].cpu().numpy().astype(np.float32)
        valid_mask = target_batch["valid_mask"].cpu().numpy().astype(np.float32)

        latent = np.zeros((n, t_max), dtype=np.float32)
        for i in range(n):
            T_i = int(seq_lengths[i])
            if self.mode == "signal" and is_positive[i]:
                onset = onset_times[i]
                early_start = int(max(0, onset - self.early_signal_horizon))
                window_start = int(max(0, onset - self.horizon))
                # Phase 1 — early ramp (before hard-label window).
                # Linear from 0 up to `early_signal_amplitude`.  The soft
                # target takes over at window_start (it dips slightly below
                # the ramp endpoint, but is then overtaken within ~3 steps).
                if early_start < min(window_start, T_i):
                    ramp_len = max(window_start - early_start, 1)
                    for t in range(early_start, min(window_start, T_i)):
                        progress = (t - early_start) / ramp_len
                        latent[i, t] = progress * self.early_signal_amplitude
                # Phase 2 — full BGSL soft-target trajectory (warning window + post-onset).
                if window_start < T_i:
                    latent[i, window_start:T_i] = soft_targets[i, window_start:T_i]
            else:
                latent[i, :T_i] = 0.0

        patient_offset = rng.normal(0.0, 0.35, size=n).astype(np.float32)
        patient_gain = rng.uniform(0.6, 1.4, size=n).astype(np.float32)
        nuisance_state = rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)

        features = np.zeros((n, t_max, self.n_features), dtype=np.float32)
        masks = np.zeros((n, t_max, self.n_features), dtype=np.float32)

        for i in range(n):
            T_i = int(seq_lengths[i])
            state = latent[i, :T_i]
            state_delta = np.diff(np.concatenate([[state[0]], state])).astype(np.float32)
            noise_bank = {
                "slow": _ar1_noise(T_i, phi=0.85, scale=0.12, rng=rng),
                "mid": _ar1_noise(T_i, phi=0.45, scale=0.16, rng=rng),
                "fast": _ar1_noise(T_i, phi=0.15, scale=0.22, rng=rng),
            }

            features[i, :T_i, 0] = patient_offset[i] + patient_gain[i] * (
                self.signal_strength * (0.55 * state) + noise_bank["mid"]
            )
            features[i, :T_i, 1] = patient_offset[i] + patient_gain[i] * (
                self.signal_strength * (0.35 * state + 0.40 * state_delta) + noise_bank["slow"]
            )
            features[i, :T_i, 2] = patient_offset[i] + patient_gain[i] * (
                self.signal_strength * (0.20 * state) + noise_bank["fast"]
            )
            features[i, :T_i, 3] = patient_offset[i] + patient_gain[i] * (
                self.signal_strength * (0.25 * np.maximum(state, 0.0)) + noise_bank["mid"]
            )

            for j in range(4, self.n_features):
                block = j % 3
                if block == 0:
                    signal = nuisance_state[i, 0] + noise_bank["slow"]
                elif block == 1:
                    signal = nuisance_state[i, 1] + noise_bank["mid"]
                else:
                    signal = nuisance_state[i, 2] + noise_bank["fast"]
                features[i, :T_i, j] = signal

            for j in range(self.n_features):
                base_missing = rng.random(T_i) < self.missing_rate
                if j < self.informative_channels and is_positive[i]:
                    center = int(np.clip(round(onset_times[i] - self.horizon / 2.0), 0, max(T_i - 1, 0)))
                    local = np.abs(np.arange(T_i) - center) <= max(2, int(self.horizon // 2))
                    dropout = rng.random(T_i) < (0.35 * local + 0.10)
                    observed = ~(base_missing | dropout)
                else:
                    observed = ~base_missing
                masks[i, :T_i, j] = observed.astype(np.float32)
                features[i, :T_i, j] *= masks[i, :T_i, j]

        return {
            "features": features,
            "masks": masks,
            "soft_targets": soft_targets,
            "vel_targets": vel_targets,
            "hard_targets": hard_targets,
            "onset_times": onset_times.astype(np.float32),
            "is_positive": is_positive.astype(np.bool_),
            "seq_lengths": seq_lengths.astype(np.int64),
            "valid_mask": valid_mask,
            "horizon": np.full(n, self.horizon, dtype=np.float32),
        }


def _var_activation(x: torch.Tensor) -> torch.Tensor:
    """Softplus-like activation ensuring positive variances (matches RKN paper)."""
    return torch.nn.functional.softplus(x) + 1e-6


class RKNCell(torch.nn.Module):
    """Recurrent Kalman Network cell — factorized Kalman filter with locally linear dynamics.

    Based on: Becker et al., "Recurrent Kalman Networks", ICML 2019.

    Key differences from standard EKF:
      - Doubled latent state z = [p; l] (position-like + velocity-like), each of size lod
      - Diagonal/factorized covariance: 3 vectors (cov_u, cov_l, cov_s) instead of full [2*lod, 2*lod] matrix
      - Kalman update is element-wise (no matrix inversions)
      - Transition is locally linear: A(z) depends on current state
      - Encoder outputs both obs_mean and obs_var (per-step uncertainty estimate)

    Formal definition:
        State:    z = [p; l] in R^{2*lod}
        Cov:      Σ = [[diag(cov_u), diag(cov_s)], [diag(cov_s), diag(cov_l)]]

        Predict:  A(z) = Σ_i coeff_i(z) * basis_matrix_i    (locally linear)
                  z_pred = A(z) @ z
                  Σ_pred = A(z) @ Σ @ A(z)^T + Q             (factorized)

        Update:   k_u = Σ_u_pred / (Σ_u_pred + obs_var)      (element-wise gain)
                  k_l = Σ_s_pred / (Σ_u_pred + obs_var)
                  z = z_pred + [k_u ⊙ (z_obs - z_pred[:lod]), k_l ⊙ (z_obs - z_pred[:lod])]
                  Σ updated by Joseph form (factorized)

        Risk:     r = head(z[:lod])
    """

    def __init__(self, lod: int, num_basis: int = 15, bandwidth: int = 3) -> None:
        super().__init__()
        self.lod = lod  # latent observation dimension
        self.lsd = 2 * lod  # latent state dimension
        self.num_basis = num_basis

        # ---- banded matrix utilities ----
        self._build_band_util(lod, bandwidth)

        # ---- transition basis matrices (4 blocks, each banded) ----
        self.tm11_basis = torch.nn.Parameter(torch.zeros(num_basis, self._num_entries))
        self.tm12_basis = torch.nn.Parameter(torch.zeros(num_basis, self._num_entries))
        self.tm21_basis = torch.nn.Parameter(torch.zeros(num_basis, self._num_entries))
        self.tm22_basis = torch.nn.Parameter(torch.zeros(num_basis, self._num_entries))
        # Initialize off-diagonal blocks to small positive/negative values
        with torch.no_grad():
            self.tm12_basis[:, self._diag_idx] = 0.2
            self.tm21_basis[:, self._diag_idx] = -0.2

        # ---- coefficient network (state → basis weights) ----
        self.coeff_net = torch.nn.Sequential(
            torch.nn.Linear(self.lsd, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, num_basis),
            torch.nn.Softmax(dim=-1),
        )

        # ---- transition noise (diagonal, size 2*lod) ----
        self.log_trans_noise = torch.nn.Parameter(
            torch.log(torch.full((self.lsd,), 0.1))
        )

    def _build_band_util(self, lod: int, bw: int) -> None:
        """Precompute index masks for banded matrix operations."""
        mask = torch.zeros(lod, lod, dtype=torch.bool)
        for i in range(lod):
            for j in range(max(0, i - bw), min(lod, i + bw + 1)):
                mask[i, j] = True
        idx = torch.where(mask)
        self.register_buffer("_idx0", idx[0], persistent=False)
        self.register_buffer("_idx1", idx[1], persistent=False)
        self._num_entries = idx[0].shape[0]
        diag_idx = torch.where(idx[0] == idx[1])[0]
        self.register_buffer("_diag_idx", diag_idx, persistent=False)

    def _unflatten(self, flat: torch.Tensor) -> torch.Tensor:
        """Convert flat [B, num_entries] banded repr to full [B, lod, lod] matrix."""
        out = torch.zeros(flat.shape[0], self.lod, self.lod, device=flat.device)
        out[:, self._idx0, self._idx1] = flat
        return out

    def _bmv(self, mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Batched matrix-vector product: mat @ vec."""
        return torch.bmm(mat, vec.unsqueeze(-1)).squeeze(-1)

    def _dadat(self, a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Diagonal of A @ diag(d) @ A^T."""
        return self._bmv(a.square(), d)

    def _dadbt(self, a: torch.Tensor, d: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Diagonal of A @ diag(d) @ B^T."""
        return self._bmv(a * b, d)

    def _get_transition(self, z: torch.Tensor) -> tuple:
        """Compute state-dependent locally linear transition model."""
        coeffs = self.coeff_net(z).unsqueeze(-1)  # [B, num_basis, 1]
        blocks = []
        for basis in [self.tm11_basis, self.tm12_basis, self.tm21_basis, self.tm22_basis]:
            flat = (coeffs * basis).sum(dim=1)  # [B, num_entries]
            flat[:, self._diag_idx] += 1.0  # identity on diagonal
            blocks.append(self._unflatten(flat))
        trans_noise = _var_activation(self.log_trans_noise)  # [lsd]
        return blocks, trans_noise

    def _update(self, prior_mean: torch.Tensor,
                prior_cov: tuple,
                obs_mean: torch.Tensor,
                obs_var: torch.Tensor) -> tuple:
        """Element-wise Kalman update (no matrix inversion)."""
        cu, cl, cs = prior_cov  # each [B, lod]
        denom = cu + obs_var  # [B, lod]
        k_u = cu / denom  # [B, lod]
        k_l = cs / denom  # [B, lod]
        residual = obs_mean - prior_mean[:, :self.lod]  # [B, lod]
        new_mean = prior_mean + torch.cat([
            k_u * residual,
            k_l * residual,
        ], dim=-1)
        new_cu = (1.0 - k_u) * cu
        new_cl = cl - k_l * cs
        new_cs = (1.0 - k_u) * cs
        return new_mean, (new_cu, new_cl, new_cs)

    def _predict(self, post_mean: torch.Tensor,
                 post_cov: tuple) -> tuple:
        """Predict step with locally linear dynamics."""
        cu, cl, cs = post_cov  # each [B, lod]
        blocks, trans_noise = self._get_transition(post_mean)
        tm11, tm12, tm21, tm22 = blocks

        # Predict mean
        mu, ml = post_mean[:, :self.lod], post_mean[:, self.lod:]
        pred_mean = torch.cat([
            self._bmv(tm11, mu) + self._bmv(tm12, ml),
            self._bmv(tm21, mu) + self._bmv(tm22, ml),
        ], dim=-1)

        # Predict covariance (factorized)
        tn_u = trans_noise[None, :self.lod]
        tn_l = trans_noise[None, self.lod:]
        pred_cu = self._dadat(tm11, cu) + 2.0 * self._dadbt(tm11, cs, tm12) + self._dadat(tm12, cl) + tn_u
        pred_cl = self._dadat(tm21, cu) + 2.0 * self._dadbt(tm21, cs, tm22) + self._dadat(tm22, cl) + tn_l
        pred_cs = (self._dadbt(tm21, cu, tm11) + self._dadbt(tm22, cs, tm11)
                   + self._dadbt(tm21, cs, tm12) + self._dadbt(tm22, cl, tm12))
        return pred_mean, (pred_cu, pred_cl, pred_cs)

    def forward(self, prior_mean: torch.Tensor, prior_cov: tuple,
                obs_mean: torch.Tensor, obs_var: torch.Tensor) -> tuple:
        """One step: update then predict."""
        post_mean, post_cov = self._update(prior_mean, prior_cov, obs_mean, obs_var)
        next_mean, next_cov = self._predict(post_mean, post_cov)
        return post_mean, post_cov, next_mean, next_cov


class KalmanLikeObserver(torch.nn.Module):
    """RKN-style observer for early event prediction.

    Architecture:
        Encoder: MLP(input(16)→64→lod(28)) outputs obs_mean + log_obs_var
        RKNCell: factorized Kalman with doubled state (56-dim), element-wise updates,
                 locally linear transition (banded matrices, coefficient network)
        Head:    MLP(lod→lod//2→1) outputs risk logit from position half of state
    """

    def __init__(self, input_dim: int = 16, latent_dim: int = 28) -> None:
        super().__init__()
        self.lod = latent_dim
        self.input_dim = input_dim

        # Encoder: input → latent observation mean + log-variance
        self.enc_1 = torch.nn.Linear(input_dim, 64)
        self.enc_2a = torch.nn.Linear(64, latent_dim)  # mean
        self.enc_2b = torch.nn.Linear(64, latent_dim)  # log-variance

        # RKN cell
        self.cell = RKNCell(lod=latent_dim)

        # Risk prediction head (on position half of latent state)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(latent_dim),
            torch.nn.Linear(latent_dim, latent_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim // 2, 1),
        )

        # Initial state mean (learned)
        self.init_mean = torch.nn.Parameter(torch.zeros(2 * latent_dim))
        # Initial state covariance (log-scale, 3 vectors of size lod)
        self.log_init_cu = torch.nn.Parameter(torch.log(torch.full((latent_dim,), 1.0)))
        self.log_init_cl = torch.nn.Parameter(torch.log(torch.full((latent_dim,), 1.0)))
        self.log_init_cs = torch.nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        B, T, _ = x.shape
        dev = x.device
        lod = self.lod

        # Initial state
        z = self.init_mean.unsqueeze(0).expand(B, -1)  # [B, 2*lod]
        cu = _var_activation(self.log_init_cu).unsqueeze(0).expand(B, -1)
        cl = _var_activation(self.log_init_cl).unsqueeze(0).expand(B, -1)
        cs = self.log_init_cs.unsqueeze(0).expand(B, -1)
        post_cov = (cu, cl, cs)

        logits_list = []

        for t in range(T):
            x_t = x[:, t, :] * mask[:, t, :]

            # Encode observation
            h = torch.tanh(self.enc_1(x_t))
            obs_mean = self.enc_2a(h)  # [B, lod]
            obs_var = _var_activation(self.enc_2b(h))  # [B, lod]

            # One RKN step: update then predict
            post_mean, post_cov, z, prior_cov = self.cell(
                z, post_cov, obs_mean, obs_var
            )

            logits_list.append(self.head(post_mean[:, :lod]))

        logits = torch.stack(logits_list, dim=1).squeeze(-1)  # [B, T]
        return logits, obs_mean, post_mean[:, :lod]


def _make_loss(method: str, *, velocity_weight: float) -> torch.nn.Module:
    if method == "BCE":
        return BCELoss()
    if method == "TLS":
        return TLSLoss(alpha=6.0)
    if method == "BGSL-state":
        return BGSLLoss(velocity_weight=0.0, acceleration_weight=0.0, monotonicity_weight=0.0)
    if method == "BGSL+vel":
        return BGSLLoss(
            velocity_weight=velocity_weight,
            acceleration_weight=0.0,
            monotonicity_weight=0.0,
        )
    raise ValueError(f"Unknown method: {method!r}")


def _train_model(
    dataset: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    loss_fn: torch.nn.Module,
    seed: int,
    device: torch.device,
    hidden_dim: int = 28,
    num_layers: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 16,
    batch_size: int = 64,
    patience: int = 4,
) -> GRUPredictor:
    _seed_all(seed)

    x_train = torch.tensor(dataset["features"][train_idx], dtype=torch.float32, device=device)
    x_val = torch.tensor(dataset["features"][val_idx], dtype=torch.float32, device=device)
    m_train = torch.tensor(dataset["masks"][train_idx], dtype=torch.float32, device=device)
    m_val = torch.tensor(dataset["masks"][val_idx], dtype=torch.float32, device=device)
    soft_train = torch.tensor(dataset["soft_targets"][train_idx], dtype=torch.float32, device=device)
    soft_val = torch.tensor(dataset["soft_targets"][val_idx], dtype=torch.float32, device=device)
    hard_train = torch.tensor(dataset["hard_targets"][train_idx], dtype=torch.float32, device=device)
    hard_val = torch.tensor(dataset["hard_targets"][val_idx], dtype=torch.float32, device=device)
    vel_train = torch.tensor(dataset["vel_targets"][train_idx], dtype=torch.float32, device=device)
    vel_val = torch.tensor(dataset["vel_targets"][val_idx], dtype=torch.float32, device=device)
    mask_train = torch.tensor(dataset["valid_mask"][train_idx], dtype=torch.float32, device=device)
    mask_val = torch.tensor(dataset["valid_mask"][val_idx], dtype=torch.float32, device=device)
    onset_train = torch.tensor(dataset["onset_times"][train_idx], dtype=torch.float32, device=device)
    onset_val = torch.tensor(dataset["onset_times"][val_idx], dtype=torch.float32, device=device)
    is_pos_train = torch.tensor(dataset["is_positive"][train_idx], dtype=torch.bool, device=device)
    is_pos_val = torch.tensor(dataset["is_positive"][val_idx], dtype=torch.bool, device=device)
    lens_train = torch.tensor(dataset["seq_lengths"][train_idx], dtype=torch.long, device=device)
    lens_val = torch.tensor(dataset["seq_lengths"][val_idx], dtype=torch.long, device=device)

    model = GRUPredictor(
        input_dim=dataset["features"].shape[-1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        use_mask_as_input=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    stale_epochs = 0

    train_size = len(train_idx)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_size, device=device)
        for start in range(0, train_size, batch_size):
            batch_ids = order[start : start + batch_size]
            logits = model(x_train[batch_ids], mask=m_train[batch_ids], seq_lens=lens_train[batch_ids])

            if isinstance(loss_fn, TLSLoss):
                tls_targets = loss_fn.build_tls_targets(
                    onset_train[batch_ids], lens_train[batch_ids], device,
                    T_max=logits.shape[1],
                )
                loss_dict = loss_fn(logits, tls_targets, mask_train[batch_ids])
            elif isinstance(loss_fn, BGSLLoss):
                loss_dict = loss_fn(
                    logits,
                    soft_train[batch_ids],
                    mask_train[batch_ids],
                    vel_targets=vel_train[batch_ids],
                    onset_times=onset_train[batch_ids],
                    is_positive=is_pos_train[batch_ids],
                    seq_lengths=lens_train[batch_ids],
                )
            else:
                loss_dict = loss_fn(logits, hard_train[batch_ids], mask_train[batch_ids])

            loss = loss_dict["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_val, mask=m_val, seq_lens=lens_val)
            if isinstance(loss_fn, TLSLoss):
                tls_targets = loss_fn.build_tls_targets(onset_val, lens_val, device, T_max=logits.shape[1])
                val_loss = loss_fn(logits, tls_targets, mask_val)["loss"].item()
            elif isinstance(loss_fn, BGSLLoss):
                val_loss = loss_fn(
                    logits,
                    soft_val,
                    mask_val,
                    vel_targets=vel_val,
                    onset_times=onset_val,
                    is_positive=is_pos_val,
                    seq_lengths=lens_val,
                )["loss"].item()
            else:
                val_loss = loss_fn(logits, hard_val, mask_val)["loss"].item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _train_kalman(
    dataset: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    latent_dim: int = 28,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 16,
    batch_size: int = 64,
    patience: int = 4,
) -> KalmanLikeObserver:
    _seed_all(seed)

    n_features = dataset["features"].shape[-1]
    x_all = torch.tensor(dataset["features"], dtype=torch.float32, device=device)
    m_all = torch.tensor(dataset["masks"], dtype=torch.float32, device=device)
    soft_all = torch.tensor(dataset["soft_targets"], dtype=torch.float32, device=device)
    mask_all = torch.tensor(dataset["valid_mask"], dtype=torch.float32, device=device)

    x_train, x_val = x_all[train_idx], x_all[val_idx]
    m_train, m_val = m_all[train_idx], m_all[val_idx]
    s_train, s_val = soft_all[train_idx], soft_all[val_idx]
    v_train, v_val = mask_all[train_idx], mask_all[val_idx]

    model = KalmanLikeObserver(n_features, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    stale_epochs = 0
    train_size = len(train_idx)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_size, device=device)
        for start in range(0, train_size, batch_size):
            batch_ids = order[start : start + batch_size]
            logits, _, _ = model(x_train[batch_ids], m_train[batch_ids])
            valid = v_train[batch_ids]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, s_train[batch_ids], reduction="none"
            )
            loss = (bce * valid).sum() / valid.sum().clamp(min=1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(x_val, m_val)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, s_val, reduction="none"
            )
            val_loss = ((bce * v_val).sum() / v_val.sum().clamp(min=1.0)).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _build_predictions(
    model: torch.nn.Module,
    dataset: Dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
    method: str = "GRU",
) -> List[PatientPrediction]:
    x = torch.tensor(dataset["features"][indices], dtype=torch.float32, device=device)
    m = torch.tensor(dataset["masks"][indices], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        if method == "Kalman":
            logits, _, _ = model(x, m)
        else:
            lens = torch.tensor(dataset["seq_lengths"][indices], dtype=torch.long, device=device)
            logits = model(x, mask=m, seq_lens=lens)
        probs = torch.sigmoid(logits).cpu().numpy()

    out: List[PatientPrediction] = []
    for row, idx in enumerate(indices):
        T = int(dataset["seq_lengths"][idx])
        out.append(
            PatientPrediction(
                id=f"synthetic_{int(idx):04d}",
                probs=probs[row, :T].astype(np.float64),
                hard_labels=dataset["hard_targets"][idx, :T].astype(np.float64),
                soft_targets=dataset["soft_targets"][idx, :T].astype(np.float64),
                has_event=bool(dataset["is_positive"][idx]),
                onset_time=float(dataset["onset_times"][idx]),
                horizon=float(dataset["horizon"][idx]),
                seq_len=T,
                time_units_per_step=1.0,
            )
        )
    return out


def _select_threshold(predictions: Sequence[PatientPrediction], *, target_sensitivity: float = 0.80) -> float:
    tracker = SepsisMetrics(threshold=0.5, n_bootstrap=0)
    for p in predictions:
        tracker.add(p)
    return tracker.select_threshold_fixed_sensitivity(target_sensitivity=target_sensitivity)


def _evaluate_predictions(
    predictions: Sequence[PatientPrediction],
    *,
    threshold: float,
) -> Dict[str, float]:
    tracker = SepsisMetrics(threshold=threshold, n_bootstrap=0)
    for p in predictions:
        tracker.add(p)
    return tracker.compute()


def _compute_early_auroc(
    predictions: Sequence[PatientPrediction],
    *,
    early_start_delta: float = 12.0,
    early_end_delta: float = 6.0,
) -> float:
    """Patient-level AUROC on max probability in the early window.

    The early window is ``[onset - early_start_delta, onset - early_end_delta)``
    for positive patients, and all valid timesteps for negative patients.
    This measures whether a model can detect pre-onset signal *before* the
    hard label window fires (i.e. before BCE has any positive target).
    """
    from sklearn.metrics import roc_auc_score

    scores: List[float] = []
    labels: List[int] = []
    for p in predictions:
        onset = p.onset_time
        T = p.seq_len
        if p.has_event and onset > 0:
            t_start = max(0, int(onset - early_start_delta))
            t_end = max(0, int(onset - early_end_delta))
        else:
            t_start = 0
            t_end = T
        if t_end > t_start:
            scores.append(float(np.max(p.probs[t_start:t_end])))
            labels.append(1 if p.has_event else 0)

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


@dataclass(frozen=True)
class RunResult:
    method: str
    seed: int
    metrics: Dict[str, float]


@dataclass(frozen=True)
class StressResult:
    name: str
    description: str
    runs: List[RunResult]

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[Dict[str, float]]] = {m: [] for m in METHODS}
        for run in self.runs:
            grouped[run.method].append(run.metrics)

        out: Dict[str, Dict[str, float]] = {}
        for method, rows in grouped.items():
            if not rows:
                continue
            merged: Dict[str, float] = {}
            for key in rows[0]:
                vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
                merged[key] = float(np.mean(vals)) if vals else float("nan")
            out[method] = merged
        return out


def _run_method(
    dataset: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    method: str,
    seed: int,
    device: torch.device,
    velocity_weight: float,
    epochs: int,
) -> Dict[str, float]:
    if method == "Kalman":
        model = _train_kalman(
            dataset,
            train_idx,
            val_idx,
            seed=seed,
            device=device,
            epochs=max(epochs, 48),
        )
        method_key = "Kalman"
    else:
        loss_fn = _make_loss(method, velocity_weight=velocity_weight)
        model = _train_model(
            dataset,
            train_idx,
            val_idx,
            loss_fn=loss_fn,
            seed=seed,
            device=device,
            epochs=epochs,
        )
        method_key = "GRU"

    val_predictions = _build_predictions(model, dataset, val_idx, device=device, method=method_key)
    threshold = _select_threshold(val_predictions, target_sensitivity=0.80)
    test_predictions = _build_predictions(model, dataset, test_idx, device=device, method=method_key)
    metrics = _evaluate_predictions(test_predictions, threshold=threshold)
    metrics["early_auroc"] = _compute_early_auroc(
        test_predictions,
        early_start_delta=12.0,
        early_end_delta=6.0,
    )
    metrics["selected_threshold"] = float(threshold)
    return metrics


def _run_stress(
    *,
    name: str,
    description: str,
    dataset: Dict[str, np.ndarray],
    seed: int,
    device: torch.device,
    velocity_weight: float,
    epochs: int,
) -> StressResult:
    train_idx, val_idx, test_idx = _split_indices(len(dataset["features"]), seed=seed)
    runs: List[RunResult] = []
    for method in METHODS:
        _seed_all(seed)
        for run_seed in STRESS_SEEDS:
            metrics = _run_method(
                dataset,
                train_idx,
                val_idx,
                test_idx,
                method=method,
                seed=run_seed,
                device=device,
                velocity_weight=velocity_weight,
                epochs=epochs,
            )
            runs.append(RunResult(method=method, seed=run_seed, metrics=metrics))
    return StressResult(name=name, description=description, runs=runs)


def _build_dataset(mode: str, seed: int) -> Dict[str, np.ndarray]:
    return SyntheticICUDataset(
        mode=mode,
        n_patients=800,
        n_features=16,
        max_length=48,
        min_length=18,
        horizon=6.0,
        pos_ratio=0.28,
        informative_channels=4,
        missing_rate=0.18 if mode == "signal" else 0.15,
        signal_strength=0.45,
        seed=seed,
    ).generate()


def experiment_signal(device: torch.device, *, velocity_weight: float, epochs: int) -> StressResult:
    dataset = _build_dataset("signal", SCENARIO_SEEDS["signal"])
    return _run_stress(
        name="signal",
        description="Weak but real pre-onset trajectory.",
        dataset=dataset,
        seed=SCENARIO_SEEDS["signal"],
        device=device,
        velocity_weight=velocity_weight,
        epochs=epochs,
    )


def experiment_null(device: torch.device, *, velocity_weight: float, epochs: int) -> StressResult:
    dataset = _build_dataset("null", SCENARIO_SEEDS["null"])
    return _run_stress(
        name="null",
        description="Negative control: no trajectory signal in the features.",
        dataset=dataset,
        seed=SCENARIO_SEEDS["null"],
        device=device,
        velocity_weight=velocity_weight,
        epochs=epochs,
    )


def _mean_metric(result: Dict[str, Dict[str, float]], method: str, metric: str) -> float:
    return float(result.get(method, {}).get(metric, float("nan")))


def _summarize_aggregates(agg: Dict[str, Dict[str, float]]) -> None:
    metrics = ["auroc", "auprc", "physionet_utility", "median_lead_time", "asf", "tce", "early_auroc"]
    header = f"{'Method':<18s}" + "".join(f"{m:>18s}" for m in metrics)
    print(header)
    print("-" * len(header))
    for method in METHODS:
        values = agg.get(method, {})
        row = "".join(f"{values.get(m, float('nan')):>18.4f}" for m in metrics)
        print(f"{method:<18s}{row}")


def _verdict(signal: Dict[str, Dict[str, float]], null: Dict[str, Dict[str, float]]) -> Tuple[bool, str, str]:
    reasons: List[str] = []

    util_gain = _mean_metric(signal, "BGSL+vel", "physionet_utility") - _mean_metric(signal, "BCE", "physionet_utility")
    lead_gain = _mean_metric(signal, "BGSL+vel", "median_lead_time") - _mean_metric(signal, "BCE", "median_lead_time")
    state_delta = _mean_metric(signal, "BGSL+vel", "physionet_utility") - _mean_metric(signal, "BGSL-state", "physionet_utility")
    null_util_gap = abs(_mean_metric(null, "BGSL+vel", "physionet_utility") - _mean_metric(null, "BCE", "physionet_utility"))
    null_lead_gap = abs(_mean_metric(null, "BGSL+vel", "median_lead_time") - _mean_metric(null, "BCE", "median_lead_time"))
    kalman_util = _mean_metric(signal, "Kalman", "physionet_utility")
    kalman_lead = _mean_metric(signal, "Kalman", "median_lead_time")

    if util_gain >= 0.02 and lead_gain >= 0.5:
        reasons.append(f"SIGNAL PASS: utility gain={util_gain:+.4f}, lead gain={lead_gain:+.1f}")
    else:
        reasons.append(f"SIGNAL FAIL: utility gain={util_gain:+.4f}, lead gain={lead_gain:+.1f}")

    if state_delta >= -0.01:
        reasons.append(f"ABLATION PASS: BGSL+vel vs state-only delta={state_delta:+.4f}")
    else:
        reasons.append(f"ABLATION FAIL: BGSL+vel underperforms state-only by {state_delta:+.4f}")

    if null_util_gap <= 0.03 and null_lead_gap <= 1.0:
        reasons.append(f"NULL PASS: utility gap={null_util_gap:.4f}, lead gap={null_lead_gap:.1f}")
    else:
        reasons.append(f"NULL FAIL: utility gap={null_util_gap:.4f}, lead gap={null_lead_gap:.1f}")

    reasons.append(f"KALMAN (observer): utility={kalman_util:.4f}, lead={kalman_lead:.1f}")

    if util_gain >= 0.02 and lead_gain >= 0.5 and state_delta >= -0.01 and null_util_gap <= 0.03 and null_lead_gap <= 1.0:
        verdict = (
            "VERDICT: GO - the minimal BGSL hypothesis survives the stress test.\n"
            "Proceed to external validation."
        )
        passed = True
    else:
        verdict = (
            "VERDICT: NO-GO - the minimal BGSL hypothesis does not beat BCE on the stress test.\n"
            "Stop the project or reframe it as a negative result."
        )
        passed = False

    return passed, verdict, "\n".join(reasons)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print()

    settings = [
        ("Default", 0.05, 16),
        ("HighVel", 0.50, 16),
        ("LongTrain", 0.05, 32),
    ]

    overall_pass = True
    for label, velocity_weight, epochs in settings:
        print("=" * 72)
        print(f"Stress Test: {label}")
        print("=" * 72)
        started = time.time()
        signal = experiment_signal(device, velocity_weight=velocity_weight, epochs=epochs)
        null = experiment_null(device, velocity_weight=velocity_weight, epochs=epochs)
        elapsed = time.time() - started

        signal_agg = signal.aggregate()
        null_agg = null.aggregate()
        print(f"Velocity weight: {velocity_weight:.3f} | Epochs: {epochs}")
        print(f"Time: {elapsed:.1f}s")
        print()
        print("Signal")
        _summarize_aggregates(signal_agg)
        print()
        print("Null")
        _summarize_aggregates(null_agg)
        print()

        passed, verdict, reasoning = _verdict(signal_agg, null_agg)
        overall_pass = overall_pass and passed
        print(verdict)
        print()
        print(reasoning)
        print()

    print("=" * 72)
    if overall_pass:
        print("FINAL VERDICT: GO - the minimal BGSL hypothesis survived all stress settings.")
    else:
        print("FINAL VERDICT: NO-GO - the minimal BGSL hypothesis failed at least one stress setting.")


if __name__ == "__main__":
    main()
