"""Minimal go/no-go synthetic benchmark for BGSL.

This is the final simplified decision gate:

- BCE baseline
- BGSL state-only
- BGSL state + velocity

No acceleration term.
No monotonicity term.
No hypernetwork.
No utility-aware loss.

The benchmark has only two scenarios:

1. Signal: a weak but real pre-onset trajectory exists in the features.
2. Null: no trajectory signal exists, but the event labels are still present.

If the minimal BGSL model does not beat BCE on Signal while staying stable on
Null, the project should stop.
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


METHODS = ("BCE", "BGSL-state", "BGSL+vel")
SCENARIO_SEEDS = {"signal": 101, "null": 202}


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
        acc_targets = target_batch["accel_target"].cpu().numpy().astype(np.float32)
        hard_targets = target_batch["hard_target"].cpu().numpy().astype(np.float32)
        valid_mask = target_batch["valid_mask"].cpu().numpy().astype(np.float32)

        latent = np.zeros((n, t_max), dtype=np.float32)
        for i in range(n):
            T_i = int(seq_lengths[i])
            if self.mode == "signal":
                latent[i, :T_i] = soft_targets[i, :T_i]
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

            # Informative channels.
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

            # Nuisance channels.
            for j in range(4, self.n_features):
                block = j % 3
                if block == 0:
                    signal = nuisance_state[i, 0] + noise_bank["slow"]
                elif block == 1:
                    signal = nuisance_state[i, 1] + noise_bank["mid"]
                else:
                    signal = nuisance_state[i, 2] + noise_bank["fast"]
                features[i, :T_i, j] = signal

            # Missingness, slightly worse near the warning window for positives.
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
            "acc_targets": acc_targets,
            "hard_targets": hard_targets,
            "onset_times": onset_times.astype(np.float32),
            "is_positive": is_positive.astype(np.bool_),
            "seq_lengths": seq_lengths.astype(np.int64),
            "valid_mask": valid_mask,
            "horizon": np.full(n, self.horizon, dtype=np.float32),
        }


def _make_loss(method: str) -> Tuple[torch.nn.Module, bool]:
    if method == "BCE":
        return BCELoss(), True
    if method == "BGSL-state":
        return BGSLLoss(velocity_weight=0.0, acceleration_weight=0.0, monotonicity_weight=0.0), False
    if method == "BGSL+vel":
        return BGSLLoss(velocity_weight=0.05, acceleration_weight=0.0, monotonicity_weight=0.0), False
    raise ValueError(f"Unknown method: {method!r}")


def _train_model(
    dataset: Dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    loss_fn: torch.nn.Module,
    use_hard_targets: bool,
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

    train_target = hard_train if use_hard_targets else soft_train
    val_target = hard_val if use_hard_targets else soft_val
    train_size = len(train_idx)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_size, device=device)
        for start in range(0, train_size, batch_size):
            batch_ids = order[start : start + batch_size]
            logits = model(x_train[batch_ids], mask=m_train[batch_ids], seq_lens=lens_train[batch_ids])
            if use_hard_targets:
                loss_dict = loss_fn(logits, train_target[batch_ids], mask_train[batch_ids])
            else:
                loss_dict = loss_fn(
                    logits,
                    train_target[batch_ids],
                    mask_train[batch_ids],
                    vel_targets=vel_train[batch_ids],
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
            logits = model(x_val, mask=m_val, seq_lens=lens_val)
            if use_hard_targets:
                val_loss = loss_fn(logits, val_target, mask_val)["loss"].item()
            else:
                val_loss = loss_fn(
                    logits,
                    val_target,
                    mask_val,
                    vel_targets=vel_val,
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
    m = torch.tensor(dataset["masks"][indices], dtype=torch.float32, device=device)
    lens = torch.tensor(dataset["seq_lengths"][indices], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x, mask=m, seq_lens=lens)).cpu().numpy()

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
            run_seed = seed + 17 * run_idx
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


def experiment_signal(device: torch.device, n_runs: int = 2) -> ExperimentResult:
    dataset = _build_dataset("signal", SCENARIO_SEEDS["signal"])
    return _run_experiment(
        name="signal",
        description="Weak but real pre-onset trajectory.",
        dataset=dataset,
        seed=SCENARIO_SEEDS["signal"],
        device=device,
        n_runs=n_runs,
    )


def experiment_null(device: torch.device, n_runs: int = 2) -> ExperimentResult:
    dataset = _build_dataset("null", SCENARIO_SEEDS["null"])
    return _run_experiment(
        name="null",
        description="Negative control: no trajectory signal in the features.",
        dataset=dataset,
        seed=SCENARIO_SEEDS["null"],
        device=device,
        n_runs=n_runs,
    )


def _mean_metric(result: ExperimentResult, method: str, metric: str) -> float:
    return float(result.results.get(method, {}).get(metric, float("nan")))


def compute_verdict(results: Dict[str, ExperimentResult]) -> Tuple[str, str]:
    reasons: List[str] = []

    signal = results.get("signal")
    null = results.get("null")

    signal_pass = False
    null_pass = False
    ablation_pass = False

    if signal:
        bce = signal.results.get("BCE", {})
        bgsl_state = signal.results.get("BGSL-state", {})
        bgsl_vel = signal.results.get("BGSL+vel", {})

        util_gain = float(bgsl_vel.get("physionet_utility", 0.0) - bce.get("physionet_utility", 0.0))
        lead_gain = float(bgsl_vel.get("median_lead_time", 0.0) - bce.get("median_lead_time", 0.0))
        state_gain = float(bgsl_vel.get("physionet_utility", 0.0) - bgsl_state.get("physionet_utility", 0.0))

        if util_gain >= 0.02 and lead_gain >= 0.5:
            signal_pass = True
            reasons.append(f"SIGNAL PASS: utility gain={util_gain:+.4f}, lead gain={lead_gain:+.1f}")
        else:
            reasons.append(f"SIGNAL FAIL: utility gain={util_gain:+.4f}, lead gain={lead_gain:+.1f}")

        if state_gain >= -0.01:
            ablation_pass = True
            reasons.append(f"ABLATION PASS: BGSL+vel vs state-only utility delta={state_gain:+.4f}")
        else:
            reasons.append(f"ABLATION FAIL: BGSL+vel underperforms state-only by {state_gain:+.4f}")

    if null:
        bce = null.results.get("BCE", {})
        bgsl_vel = null.results.get("BGSL+vel", {})
        util_gap = abs(float(bgsl_vel.get("physionet_utility", 0.0) - bce.get("physionet_utility", 0.0)))
        lead_gap = abs(float(bgsl_vel.get("median_lead_time", 0.0) - bce.get("median_lead_time", 0.0)))
        if util_gap <= 0.03 and lead_gap <= 1.0:
            null_pass = True
            reasons.append(f"NULL PASS: utility gap={util_gap:.4f}, lead gap={lead_gap:.1f}")
        else:
            reasons.append(f"NULL FAIL: utility gap={util_gap:.4f}, lead gap={lead_gap:.1f}")

    if signal_pass and null_pass and ablation_pass:
        verdict = (
            "VERDICT: GO - the minimal BGSL hypothesis survives the synthetic gate.\n"
            "Proceed to external validation."
        )
    elif not signal_pass or not ablation_pass:
        verdict = (
            "VERDICT: NO-GO - the minimal BGSL hypothesis does not beat BCE on the gate.\n"
            "Stop the project or reframe it as a negative result."
        )
    else:
        verdict = (
            "VERDICT: MIXED - the controls are acceptable, but the signal advantage is not strong enough.\n"
            "Do not proceed to MIMIC yet."
        )

    return verdict, "\n".join(reasons)


def _print_table(result: ExperimentResult) -> None:
    metrics = ["auroc", "auprc", "physionet_utility", "median_lead_time", "asf", "tce"]
    header = f"{'Method':<18s}" + "".join(f"{m:>18s}" for m in metrics)
    print(header)
    print("-" * len(header))
    for method in METHODS:
        values = result.results.get(method, {})
        row = "".join(f"{values.get(m, float('nan')):>18.4f}" for m in metrics)
        print(f"{method:<18s}{row}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print()

    n_runs = 2
    experiments = [
        ("Signal", lambda: experiment_signal(device, n_runs)),
        ("Null", lambda: experiment_null(device, n_runs)),
    ]

    results: Dict[str, ExperimentResult] = {}
    for title, runner in experiments:
        print("=" * 72)
        print(title)
        print("=" * 72)
        started = time.time()
        exp = runner()
        elapsed = time.time() - started
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
