"""Go/no-go synthetic benchmark for BGSL.

This script answers a narrow question:

    Does BGSL improve early-warning behavior on data where the temporal
    structure is fully under our control?

It is intentionally self-contained and deterministic. Run it manually.
It does not depend on PhysioNet or MIMIC.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bgsl.core.common.losses import BCELoss, BGSLLoss
from bgsl.core.common.targets import BaseSoftOnsetTarget
from bgsl.core.sepsis.metrics import PatientPrediction, SepsisMetrics
from bgsl.models.gru import GRUPredictor


RAMP_TYPES = ("logistic", "step", "linear", "delayed")
METHODS = ("BCE", "BGSL-state", "BGSL+vel", "BGSL+vel+acc", "BGSL+vel+acc+mono")
SHAPE_SEEDS = {
    "logistic": 101,
    "step": 102,
    "linear": 103,
    "delayed": 104,
}


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ar1_noise(
    length: int,
    *,
    phi: float = 0.35,
    scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    eps = rng.normal(0.0, scale, size=length)
    out = np.zeros(length, dtype=np.float64)
    out[0] = eps[0]
    for t in range(1, length):
        out[t] = phi * out[t - 1] + np.sqrt(max(1.0 - phi**2, 0.0)) * eps[t]
    return out


def _split_indices(
    n: int,
    *,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0):
        raise ValueError("train_frac and val_frac must be in (0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.")

    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


class SyntheticICUDataset:
    """Deterministic ICU-like synthetic cohort with controllable temporal structure."""

    def __init__(
        self,
        *,
        n_patients: int = 1200,
        n_features: int = 12,
        max_length: int = 48,
        min_length: int = 12,
        horizon: float = 6.0,
        tau: float = 2.0,
        kappa: float = 1.0,
        pos_ratio: float = 0.3,
        snr: float = 1.0,
        n_informative: Optional[int] = None,
        ramp_type: str = "logistic",
        label_noise_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        if ramp_type not in RAMP_TYPES:
            raise ValueError(f"ramp_type must be one of {RAMP_TYPES!r}")
        if not (0.0 < pos_ratio < 1.0):
            raise ValueError("pos_ratio must be in (0, 1).")
        if snr <= 0.0:
            raise ValueError("snr must be positive.")
        if min_length < 2 or max_length < min_length:
            raise ValueError("Invalid sequence length range.")

        self.n_patients = n_patients
        self.n_features = n_features
        self.max_length = max_length
        self.min_length = min_length
        self.horizon = horizon
        self.tau = tau
        self.kappa = kappa
        self.pos_ratio = pos_ratio
        self.snr = snr
        self.n_informative = n_informative or max(1, n_features // 3)
        self.ramp_type = ramp_type
        self.label_noise_std = label_noise_std
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

            lower = int(np.ceil(self.horizon)) + 1
            upper = int(seq_lengths[i]) - 1
            if upper <= lower:
                is_positive[i] = False
                continue

            onset_times[i] = float(rng.integers(lower, upper + 1))

        noisy_onset_times = onset_times.copy()
        if self.label_noise_std > 0.0:
            noise = rng.normal(0.0, self.label_noise_std, size=n)
            for i in range(n):
                if not is_positive[i]:
                    continue
                lower = float(np.ceil(self.horizon)) + 1.0
                upper = float(seq_lengths[i] - 1)
                noisy_onset_times[i] = float(np.clip(onset_times[i] + noise[i], lower, upper))

        target_builder = BaseSoftOnsetTarget(horizon=self.horizon, tau=self.tau, kappa=self.kappa)
        target_batch = target_builder(
            torch.tensor(noisy_onset_times, dtype=torch.float32),
            torch.tensor(seq_lengths, dtype=torch.long),
            torch.device("cpu"),
        )

        soft_targets = target_batch["soft_target"].cpu().numpy()
        vel_targets = target_batch["velocity_target"].cpu().numpy()
        acc_targets = target_batch["accel_target"].cpu().numpy()
        hard_targets = target_batch["hard_target"].cpu().numpy()
        valid_mask = target_batch["valid_mask"].cpu().numpy()

        if self.ramp_type == "logistic":
            latent_signal = soft_targets.copy()
        elif self.ramp_type == "step":
            latent_signal = hard_targets.copy()
        elif self.ramp_type == "linear":
            latent_signal = np.zeros_like(soft_targets, dtype=np.float32)
            for i in range(n):
                if not is_positive[i]:
                    continue
                onset = int(round(float(noisy_onset_times[i])))
                lo = max(0, onset - int(self.horizon))
                hi = min(int(seq_lengths[i]) - 1, onset)
                if hi <= lo:
                    continue
                latent_signal[i, :lo] = 0.0
                latent_signal[i, lo : hi + 1] = np.linspace(0.0, 1.0, hi - lo + 1)
        else:
            delayed_builder = BaseSoftOnsetTarget(
                horizon=self.horizon / 2.0,
                tau=max(self.tau / 2.0, 1e-3),
                kappa=self.kappa,
            )
            delayed = delayed_builder(
                torch.tensor(noisy_onset_times, dtype=torch.float32),
                torch.tensor(seq_lengths, dtype=torch.long),
                torch.device("cpu"),
            )
            latent_signal = delayed["soft_target"].cpu().numpy()

        loadings = np.zeros(self.n_features, dtype=np.float32)
        loadings[: self.n_informative] = rng.uniform(0.6, 1.4, size=self.n_informative)
        baselines = rng.normal(0.0, 0.1, size=self.n_features).astype(np.float32)

        features = np.zeros((n, t_max, self.n_features), dtype=np.float32)
        for j in range(self.n_features):
            for i in range(n):
                T_i = int(seq_lengths[i])
                noise = _ar1_noise(T_i, phi=0.35, scale=0.12, rng=rng).astype(np.float32)
                signal = latent_signal[i, :T_i] if j < self.n_informative else 0.0
                features[i, :T_i, j] = (
                    loadings[j] * signal * self.snr + baselines[j] + noise
                )

        return {
            "features": features,
            "soft_targets": soft_targets.astype(np.float32),
            "vel_targets": vel_targets.astype(np.float32),
            "acc_targets": acc_targets.astype(np.float32),
            "hard_targets": hard_targets.astype(np.float32),
            "onset_times": noisy_onset_times.astype(np.float32),
            "is_positive": is_positive.astype(np.bool_),
            "seq_lengths": seq_lengths.astype(np.int64),
            "valid_mask": valid_mask.astype(np.float32),
            "horizon": np.full(n, self.horizon, dtype=np.float32),
        }


def _build_batch_tensors(
    dataset: Dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "x": torch.tensor(dataset["features"][indices], dtype=torch.float32, device=device),
        "soft_targets": torch.tensor(dataset["soft_targets"][indices], dtype=torch.float32, device=device),
        "vel_targets": torch.tensor(dataset["vel_targets"][indices], dtype=torch.float32, device=device),
        "acc_targets": torch.tensor(dataset["acc_targets"][indices], dtype=torch.float32, device=device),
        "hard_targets": torch.tensor(dataset["hard_targets"][indices], dtype=torch.float32, device=device),
        "valid_mask": torch.tensor(dataset["valid_mask"][indices], dtype=torch.float32, device=device),
        "onset_times": torch.tensor(dataset["onset_times"][indices], dtype=torch.float32, device=device),
        "is_positive": torch.tensor(dataset["is_positive"][indices], dtype=torch.bool, device=device),
        "seq_lengths": torch.tensor(dataset["seq_lengths"][indices], dtype=torch.long, device=device),
        "horizon": torch.tensor(dataset["horizon"][indices], dtype=torch.float32, device=device),
    }


def _make_loss(name: str) -> Tuple[torch.nn.Module, bool]:
    if name == "BCE":
        return BCELoss(), True
    if name == "BGSL-state":
        return BGSLLoss(velocity_weight=0.0, acceleration_weight=0.0, monotonicity_weight=0.0), False
    if name == "BGSL+vel":
        return BGSLLoss(velocity_weight=0.1, acceleration_weight=0.0, monotonicity_weight=0.0), False
    if name == "BGSL+vel+acc":
        return BGSLLoss(velocity_weight=0.1, acceleration_weight=0.05, monotonicity_weight=0.0), False
    if name == "BGSL+vel+acc+mono":
        return BGSLLoss(velocity_weight=0.1, acceleration_weight=0.05, monotonicity_weight=0.01), False
    raise ValueError(f"Unknown method: {name!r}")


def _train_model(
    dataset: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    loss_fn: torch.nn.Module,
    use_hard_targets: bool,
    seed: int,
    device: torch.device,
    hidden_dim: int = 32,
    num_layers: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
) -> GRUPredictor:
    _seed_all(seed)

    x_train = torch.tensor(dataset["features"][train_idx], dtype=torch.float32, device=device)
    x_val = torch.tensor(dataset["features"][val_idx], dtype=torch.float32, device=device)
    soft_train = torch.tensor(dataset["soft_targets"][train_idx], dtype=torch.float32, device=device)
    soft_val = torch.tensor(dataset["soft_targets"][val_idx], dtype=torch.float32, device=device)
    hard_train = torch.tensor(dataset["hard_targets"][train_idx], dtype=torch.float32, device=device)
    hard_val = torch.tensor(dataset["hard_targets"][val_idx], dtype=torch.float32, device=device)
    vel_train = torch.tensor(dataset["vel_targets"][train_idx], dtype=torch.float32, device=device)
    vel_val = torch.tensor(dataset["vel_targets"][val_idx], dtype=torch.float32, device=device)
    acc_train = torch.tensor(dataset["acc_targets"][train_idx], dtype=torch.float32, device=device)
    acc_val = torch.tensor(dataset["acc_targets"][val_idx], dtype=torch.float32, device=device)
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
        use_mask_as_input=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    stale_epochs = 0

    train_target = hard_train if use_hard_targets else soft_train
    val_target = hard_val if use_hard_targets else soft_val
    train_size = len(train_idx)

    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(train_size, device=device)
        for start in range(0, train_size, batch_size):
            batch_ids = order[start : start + batch_size]
            logits = model(x_train[batch_ids], seq_lens=lens_train[batch_ids])
            if use_hard_targets:
                loss_dict = loss_fn(logits, train_target[batch_ids], mask_train[batch_ids])
            else:
                loss_dict = loss_fn(
                    logits,
                    train_target[batch_ids],
                    mask_train[batch_ids],
                    vel_targets=vel_train[batch_ids],
                    acc_targets=acc_train[batch_ids],
                    onset_times=onset_train[batch_ids],
                    is_positive=is_pos_train[batch_ids],
                    seq_lengths=lens_train[batch_ids],
                )

            loss = loss_dict["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_val, seq_lens=lens_val)
            if use_hard_targets:
                val_loss = loss_fn(logits, val_target, mask_val)["loss"].item()
            else:
                val_loss = loss_fn(
                    logits,
                    val_target,
                    mask_val,
                    vel_targets=vel_val,
                    acc_targets=acc_val,
                    onset_times=onset_val,
                    is_positive=is_pos_val,
                    seq_lengths=lens_val,
                )["loss"].item()

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
    model: GRUPredictor,
    dataset: Dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> List[PatientPrediction]:
    x = torch.tensor(dataset["features"][indices], dtype=torch.float32, device=device)
    lens = torch.tensor(dataset["seq_lengths"][indices], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x, seq_lens=lens)).cpu().numpy()

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


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    description: str
    results: Dict[str, Dict[str, float]]


def _run_method(
    dataset: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    method: str,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    loss_fn, use_hard = _make_loss(method)
    model = _train_model(
        dataset,
        train_idx,
        val_idx,
        loss_fn=loss_fn,
        use_hard_targets=use_hard,
        seed=seed,
        device=device,
    )

    val_predictions = _build_predictions(model, dataset, val_idx, device=device)
    threshold = _select_threshold(val_predictions, target_sensitivity=0.80)
    test_predictions = _build_predictions(model, dataset, test_idx, device=device)
    metrics = _evaluate_predictions(test_predictions, threshold=threshold)
    metrics["selected_threshold"] = float(threshold)
    return metrics


def _run_experiment(
    *,
    name: str,
    description: str,
    dataset: Dict[str, np.ndarray],
    seed: int,
    device: torch.device,
    n_runs: int = 2,
) -> ExperimentResult:
    train_idx, val_idx, test_idx = _split_indices(len(dataset["features"]), seed=seed)
    results: Dict[str, Dict[str, float]] = {}

    for method in METHODS:
        runs: List[Dict[str, float]] = []
        for run_idx in range(n_runs):
            run_seed = seed + 31 * run_idx
            runs.append(
                _run_method(
                    dataset,
                    train_idx,
                    val_idx,
                    test_idx,
                    method=method,
                    seed=run_seed,
                    device=device,
                )
            )

        averaged: Dict[str, float] = {}
        for key in runs[0]:
            values = [r[key] for r in runs if key in r and np.isfinite(r[key])]
            averaged[key] = float(np.mean(values)) if values else float("nan")
        results[method] = averaged

    return ExperimentResult(name=name, description=description, results=results)


def experiment_clean_ramp(device: torch.device, n_runs: int = 2) -> ExperimentResult:
    dataset = SyntheticICUDataset(
        n_patients=1200,
        n_features=12,
        max_length=48,
        min_length=12,
        horizon=6.0,
        tau=2.0,
        kappa=1.0,
        pos_ratio=0.3,
        snr=1.0,
        n_informative=3,
        ramp_type="logistic",
        label_noise_std=0.0,
        seed=42,
    ).generate()
    return _run_experiment(
        name="1_clean_ramp",
        description="Clean logistic ramp: BGSL should outperform BCE on utility and lead time.",
        dataset=dataset,
        seed=42,
        device=device,
        n_runs=n_runs,
    )


def experiment_no_ramp(device: torch.device, n_runs: int = 2) -> ExperimentResult:
    dataset = SyntheticICUDataset(
        n_patients=1200,
        n_features=12,
        max_length=48,
        min_length=12,
        horizon=6.0,
        tau=2.0,
        kappa=5.0,
        pos_ratio=0.3,
        snr=1.0,
        n_informative=3,
        ramp_type="step",
        label_noise_std=0.0,
        seed=43,
    ).generate()
    return _run_experiment(
        name="2_no_ramp",
        description="Step-like label process: BGSL should not show a large spurious gain.",
        dataset=dataset,
        seed=43,
        device=device,
        n_runs=n_runs,
    )


def experiment_noisy_labels(device: torch.device, n_runs: int = 2) -> ExperimentResult:
    dataset = SyntheticICUDataset(
        n_patients=1200,
        n_features=12,
        max_length=48,
        min_length=12,
        horizon=6.0,
        tau=2.0,
        kappa=1.0,
        pos_ratio=0.3,
        snr=1.0,
        n_informative=3,
        ramp_type="logistic",
        label_noise_std=3.0,
        seed=44,
    ).generate()
    return _run_experiment(
        name="3_noisy_labels",
        description="Noisy onset times: BGSL should not collapse relative to BCE.",
        dataset=dataset,
        seed=44,
        device=device,
        n_runs=n_runs,
    )


def experiment_shape_mismatch(device: torch.device, n_runs: int = 2) -> ExperimentResult:
    combined: Dict[str, Dict[str, float]] = {}
    for shape in ("logistic", "linear", "delayed"):
        dataset = SyntheticICUDataset(
            n_patients=900,
            n_features=12,
            max_length=48,
            min_length=12,
            horizon=6.0,
            tau=2.0,
            kappa=1.0,
            pos_ratio=0.3,
            snr=1.0,
            n_informative=3,
            ramp_type=shape,
            label_noise_std=0.0,
            seed=SHAPE_SEEDS[shape],
        ).generate()

        result = _run_experiment(
            name=f"shape_{shape}",
            description=f"Ramp shape: {shape}",
            dataset=dataset,
            seed=SHAPE_SEEDS[shape],
            device=device,
            n_runs=n_runs,
        )

        for method, metrics in result.results.items():
            combined[f"{shape}_{method}"] = metrics

    return ExperimentResult(
        name="4_shape_mismatch",
        description="Different ramp shapes: BGSL should remain competitive on most shapes.",
        results=combined,
    )


def _mean_metric(result: ExperimentResult, method: str, metric: str) -> float:
    return float(result.results.get(method, {}).get(metric, float("nan")))


def compute_verdict(results: Dict[str, ExperimentResult]) -> Tuple[str, str]:
    reasons: List[str] = []
    passed = 0
    total = 4

    clean = results.get("1_clean_ramp")
    if clean:
        bgsl = clean.results.get("BGSL+vel+acc+mono", {})
        bce = clean.results.get("BCE", {})
        util_gain = float(bgsl.get("physionet_utility", 0.0) - bce.get("physionet_utility", 0.0))
        lead_gain = float(bgsl.get("median_lead_time", 0.0) - bce.get("median_lead_time", 0.0))
        if util_gain >= 0.02 and lead_gain >= 1.0:
            passed += 1
            reasons.append(
                f"CLEAN RAMP PASS: utility gain={util_gain:+.4f}, lead gain={lead_gain:+.1f}"
            )
        else:
            reasons.append(
                f"CLEAN RAMP FAIL: utility gain={util_gain:+.4f}, lead gain={lead_gain:+.1f}"
            )

    no_ramp = results.get("2_no_ramp")
    if no_ramp:
        bgsl = no_ramp.results.get("BGSL+vel+acc+mono", {})
        bce = no_ramp.results.get("BCE", {})
        util_gap = abs(float(bgsl.get("physionet_utility", 0.0) - bce.get("physionet_utility", 0.0)))
        if util_gap <= 0.02:
            passed += 1
            reasons.append(f"NO-RAMP PASS: utility gap={util_gap:.4f} is small")
        else:
            reasons.append(f"NO-RAMP FAIL: utility gap={util_gap:.4f} is too large")

    noisy = results.get("3_noisy_labels")
    if noisy:
        bgsl = noisy.results.get("BGSL+vel+acc+mono", {})
        bce = noisy.results.get("BCE", {})
        util_diff = float(bgsl.get("physionet_utility", 0.0) - bce.get("physionet_utility", 0.0))
        if util_diff >= -0.02:
            passed += 1
            reasons.append(f"NOISY-LABELS PASS: utility diff={util_diff:+.4f}")
        else:
            reasons.append(f"NOISY-LABELS FAIL: utility diff={util_diff:+.4f}")

    shape = results.get("4_shape_mismatch")
    if shape:
        wins = 0
        total_shapes = 3
        for shape_name in ("logistic", "linear", "delayed"):
            bgsl = shape.results.get(f"{shape_name}_BGSL+vel+acc+mono", {})
            bce = shape.results.get(f"{shape_name}_BCE", {})
            util_gap = float(bgsl.get("physionet_utility", 0.0) - bce.get("physionet_utility", 0.0))
            if util_gap >= -0.02:
                wins += 1
        if wins >= 2:
            passed += 1
            reasons.append(f"SHAPE PASS: BGSL is competitive on {wins}/{total_shapes} shapes")
        else:
            reasons.append(f"SHAPE FAIL: BGSL is competitive on only {wins}/{total_shapes} shapes")

    clean_for_ablation = results.get("1_clean_ramp")
    ablation_pass = False
    if clean_for_ablation:
        order = list(METHODS)
        utility_values = [_mean_metric(clean_for_ablation, m, "physionet_utility") for m in order]
        valid = all(np.isfinite(v) for v in utility_values)
        if valid:
            deltas = np.diff(utility_values)
            if np.all(deltas >= -0.01):
                ablation_pass = True
                reasons.append("ABLATION PASS: utility is monotonic or flat across the ladder")
            else:
                reasons.append("ABLATION FAIL: utility is not monotonic across the ladder")
        else:
            reasons.append("ABLATION FAIL: missing utility values")

    if passed >= 3 and ablation_pass:
        verdict = (
            "VERDICT: STRONGLY POSITIVE - pursue BGSL.\n"
            "The synthetic benchmark supports the claim that trajectory supervision is useful."
        )
    elif passed < 2 or not ablation_pass:
        verdict = (
            "VERDICT: WEAK/UNSTABLE - do not proceed to MIMIC yet.\n"
            "The synthetic benchmark does not justify a stronger claim."
        )
    else:
        verdict = (
            "VERDICT: MIXED - proceed only after fixing the failing conditions above."
        )

    return verdict, "\n".join(reasons)


def _print_table(result: ExperimentResult) -> None:
    metrics = ["auroc", "auprc", "physionet_utility", "median_lead_time", "asf", "tce"]
    header = f"{'Method':<22s}" + "".join(f"{m:>18s}" for m in metrics)
    print(header)
    print("-" * len(header))
    for method in METHODS:
        values = result.results.get(method, {})
        row = "".join(f"{values.get(m, float('nan')):>18.4f}" for m in metrics)
        print(f"{method:<22s}{row}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print()

    n_runs = 2
    experiments = [
        ("Clean ramp", lambda: experiment_clean_ramp(device, n_runs)),
        ("No ramp", lambda: experiment_no_ramp(device, n_runs)),
        ("Noisy labels", lambda: experiment_noisy_labels(device, n_runs)),
        ("Shape mismatch", lambda: experiment_shape_mismatch(device, n_runs)),
    ]

    results: Dict[str, ExperimentResult] = {}
    for title, runner in experiments:
        print("=" * 72)
        print(title)
        print("=" * 72)
        start = time.time()
        exp = runner()
        elapsed = time.time() - start
        results[exp.name] = exp
        print(f"Time: {elapsed:.1f}s")
        _print_table(exp)
        print()

    verdict, reasoning = compute_verdict(results)
    print(verdict)
    print()
    print(reasoning)


if __name__ == "__main__":
    main()
