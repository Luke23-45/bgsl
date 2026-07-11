"""CSD-Kalman Observer synthetic benchmark: go/no-go test for critical slowing down.

Hypothesis: Providing EMA-based autocorrelation (rho) and observation residual
as features to the risk head improves early detection of bifurcations.

Models compared:
    - Raw-CSD:           Classical CSD indicator on raw observations (non-learned)
    - Kalman-BCE:        Observer with head(z) only
    - Kalman-CSD-Aug:    Observer with head(z, rho, residual) — CSD as learnable features
    - Kalman-CSD-Aug-Spec: Same + spectral radius constraint on (I-KC)A

Data: Three bifurcation systems (fold, Hopf, logistic map) with known tipping times.

Verdict: GO if Kalman-CSD-Aug detects tipping >= 20 timesteps earlier than
Kalman-BCE on at least 2 of 3 systems.
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

from bgsl.core.common.csd_loss import SpectralRadiusLoss
from bgsl.data.synthetic.bifurcation import build_dataset
from bgsl.models.csd_observer import CSDKalmanObserver

SYSTEMS = ("fold", "hopf", "logistic")
METHODS = ("Raw-CSD", "Kalman-BCE", "Kalman-CSD-Aug", "Kalman-CSD-Aug-Spec")
STRESS_SEEDS = (101, 202, 303, 404, 505)

W_LABEL = 30


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_hard_targets(
    bifurcation_times: torch.Tensor,
    seq_lengths: torch.Tensor,
    T_max: int,
    device: torch.device,
) -> torch.Tensor:
    """Build hard BCE targets: 1 in [tau - W_label, tau], 0 otherwise."""
    B = bifurcation_times.shape[0]
    t = torch.arange(T_max, device=device).float().unsqueeze(0).expand(B, -1)
    tau = bifurcation_times.unsqueeze(1).float()
    in_window = (t >= tau - W_LABEL) & (t <= tau) & (t >= 0) & (tau >= 0)
    valid_len = t < seq_lengths.unsqueeze(1).float()
    labels = (in_window & valid_len).float()
    return labels


def _build_dataset_for_system(
    system: str,
    *,
    null: bool,
    n_trajectories: int,
    noise_scale: float,
    seed: int,
) -> Dict:
    kwargs = dict(
        n_trajectories=n_trajectories,
        max_length=200,
        noise_scale=noise_scale,
        seed=seed,
        null=null,
    )
    if system == "fold":
        return build_dataset("fold", **kwargs)
    elif system == "hopf":
        return build_dataset("hopf", **kwargs)
    else:
        return build_dataset("logistic", **kwargs)


# ---------------------------------------------------------------------------
# Training — all variants use BCE only; difference is model config
# ---------------------------------------------------------------------------


def _train_kalman(
    dataset: Dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    loss_type: str,
    seed: int,
    device: torch.device,
    latent_dim: int = 4,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 5,
    spec_weight: float = 0.1,
) -> CSDKalmanObserver:
    _seed_all(seed)

    use_csd_feat = loss_type in ("csd_aug", "csd_aug_spec")
    use_spec = loss_type == "csd_aug_spec"

    n_features = dataset["features"].shape[-1]

    x_all = torch.tensor(dataset["features"], dtype=torch.float32, device=device)
    m_all = torch.ones_like(x_all)
    seq_lens_all = torch.tensor(dataset["seq_lengths"], dtype=torch.long, device=device)
    bif_times_all = torch.tensor(dataset["bifurcation_times"], dtype=torch.float32, device=device)

    x_train, x_val = x_all[train_idx], x_all[val_idx]
    m_train, m_val = m_all[train_idx], m_all[val_idx]
    lens_train, lens_val = seq_lens_all[train_idx], seq_lens_all[val_idx]
    bif_train, bif_val = bif_times_all[train_idx], bif_times_all[val_idx]

    model = CSDKalmanObserver(
        input_dim=n_features,
        latent_dim=latent_dim,
        use_csd_features=use_csd_feat,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    spec_loss_fn = SpectralRadiusLoss(weight=spec_weight)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    stale_epochs = 0
    train_size = len(train_idx)
    T_max = x_all.shape[1]

    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_size, device=device)
        for start in range(0, train_size, batch_size):
            batch_ids = order[start : start + batch_size]
            logits, zs, A, K, C = model(x_train[batch_ids], m_train[batch_ids])

            targets = _make_hard_targets(
                bif_train[batch_ids], lens_train[batch_ids], T_max, device
            )
            valid_mask = (torch.arange(T_max, device=device).unsqueeze(0) <
                          lens_train[batch_ids].unsqueeze(1)).float()

            bce_per_step = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            loss = (bce_per_step * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

            if use_spec:
                loss = loss + spec_loss_fn(A, K, C)["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_val, zs_val, A_val, K_val, C_val = model(x_val, m_val)
            targets_val = _make_hard_targets(bif_val, lens_val, T_max, device)
            valid_mask_val = (torch.arange(T_max, device=device).unsqueeze(0) <
                              lens_val.unsqueeze(1)).float()

            bce_per_step_val = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_val, targets_val, reduction="none"
            )
            val_loss = (bce_per_step_val * valid_mask_val).sum() / valid_mask_val.sum().clamp(min=1.0)

            if use_spec:
                val_loss = val_loss + spec_loss_fn(A_val, K_val, C_val)["loss"]

            val_loss = val_loss.item()

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


# ---------------------------------------------------------------------------
# Raw CSD baseline (non-learned)
# ---------------------------------------------------------------------------


def _raw_csd_indicator(
    features: np.ndarray,
    window_size: int = 30,
) -> np.ndarray:
    """Classical CSD: sliding lag-1 autocorrelation on raw observations."""
    B, T, C = features.shape
    W = min(window_size, T)
    scores = np.zeros((B, T), dtype=np.float32)

    for b in range(B):
        for c in range(C):
            seq = features[b, :, c]
            for t in range(W, T):
                seg = seq[t - W : t]
                seg_c = seg - seg.mean()
                num = np.sum(seg_c[:-1] * seg_c[1:])
                denom = np.sum(seg_c ** 2) + 1e-8
                rho = num / denom
                scores[b, t] = max(scores[b, t], rho)

    return scores


def _evaluate_raw_csd(
    csd_scores_signal: np.ndarray,
    bif_times_signal: np.ndarray,
    is_pos_signal: np.ndarray,
    seq_lens_signal: np.ndarray,
    csd_scores_null: np.ndarray,
    seq_lens_null: np.ndarray,
    threshold: float,
    early_start_delta: float = 50.0,
    early_end_delta: float = 5.0,
) -> Dict[str, float]:
    """Evaluate classical CSD baseline (no training)."""
    detection_times = []
    early_probs = []
    early_labels = []

    for i in range(len(csd_scores_signal)):
        tau = bif_times_signal[i]
        T = int(seq_lens_signal[i])
        if is_pos_signal[i] and tau > 0:
            alert_idx = np.where(csd_scores_signal[i, :int(tau)] >= threshold)[0]
            if len(alert_idx) > 0:
                detection_times.append(tau - alert_idx[0])
            ep_start = max(0, int(tau - early_start_delta))
            ep_end = max(0, int(tau - early_end_delta))
            if ep_end > ep_start:
                early_probs.append(float(np.max(csd_scores_signal[i, ep_start:ep_end])))
                early_labels.append(1)

    for i in range(len(csd_scores_null)):
        T = int(seq_lens_null[i])
        if T > 0:
            early_probs.append(float(np.max(csd_scores_null[i, :T])))
            early_labels.append(0)

    metrics: Dict[str, float] = {}
    metrics["detection_time"] = float(np.mean(detection_times)) if detection_times else float("nan")
    metrics["csd_strength"] = float("nan")

    if len(set(early_labels)) >= 2:
        from sklearn.metrics import roc_auc_score
        metrics["ew_auc"] = float(roc_auc_score(early_labels, early_probs))
    else:
        metrics["ew_auc"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Prediction and threshold selection
# ---------------------------------------------------------------------------


def _build_probs(
    model: CSDKalmanObserver,
    dataset: Dict,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = torch.tensor(dataset["features"][indices], dtype=torch.float32, device=device)
    m = torch.ones_like(x)
    model.eval()
    with torch.no_grad():
        logits, _, _, _, _ = model(x, m)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def _select_threshold(
    val_probs: np.ndarray,
    val_bif_times: np.ndarray,
    val_is_positive: np.ndarray,
    val_seq_lengths: np.ndarray,
    *,
    target_sensitivity: float = 0.80,
) -> float:
    positives = val_is_positive & (val_bif_times > 0)
    if not positives.any():
        return 0.5

    scores = []
    labels = []
    for i in np.where(positives)[0]:
        tau = val_bif_times[i]
        T = int(val_seq_lengths[i])
        window = max(0, int(tau - W_LABEL))
        for t in range(window, min(T, int(tau))):
            scores.append(val_probs[i, t])
            labels.append(1)
        for t in range(0, max(window - 1, 0)):
            scores.append(val_probs[i, t])
            labels.append(0)

    if len(set(labels)) < 2:
        return 0.5

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = np.argmin(np.abs(tpr - target_sensitivity))
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_detection_time(
    probs: np.ndarray,
    bifurcation_times: np.ndarray,
    is_positive: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> float:
    times = []
    for i in range(len(probs)):
        if not is_positive[i]:
            continue
        tau = bifurcation_times[i]
        if tau <= 0:
            continue
        pre = probs[i, :int(tau)]
        alerts = np.where(pre >= threshold)[0]
        if len(alerts) > 0:
            times.append(tau - alerts[0])
    return float(np.mean(times)) if times else float("nan")


def _compute_early_warning_auc(
    probs_signal: np.ndarray,
    bif_times_signal: np.ndarray,
    is_pos_signal: np.ndarray,
    seq_lens_signal: np.ndarray,
    probs_null: np.ndarray,
    seq_lens_null: np.ndarray,
    *,
    early_start_delta: float = 50.0,
    early_end_delta: float = 5.0,
) -> float:
    """Patient-level AUROC: max prob in early window (pos) vs null max (neg)."""
    from sklearn.metrics import roc_auc_score

    scores = []
    labels = []

    for i in range(len(probs_signal)):
        tau = bif_times_signal[i]
        T = int(seq_lens_signal[i])
        if is_pos_signal[i] and tau > 0:
            t_start = max(0, int(tau - early_start_delta))
            t_end = max(0, int(tau - early_end_delta))
            window = probs_signal[i, t_start:t_end]
            if len(window) > 0:
                scores.append(float(np.max(window)))
                labels.append(1)

    for i in range(len(probs_null)):
        T = int(seq_lens_null[i])
        if T > 0:
            scores.append(float(np.max(probs_null[i, :T])))
            labels.append(0)

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _compute_false_positive_rate(
    probs: np.ndarray,
    seq_lengths: np.ndarray,
    threshold: float,
) -> float:
    total_steps = 0
    alert_steps = 0
    for i in range(len(probs)):
        T = int(seq_lengths[i])
        total_steps += T
        alert_steps += int((probs[i, :T] >= threshold).sum())
    return alert_steps / max(total_steps, 1)


def _compute_null_metrics(
    probs: np.ndarray,
    threshold: float,
    seq_lengths: np.ndarray,
) -> Dict[str, float]:
    return {
        "fpr": _compute_false_positive_rate(probs, seq_lengths, threshold),
    }


# ---------------------------------------------------------------------------
# Results data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    method: str
    seed: int
    metrics: Dict[str, float]


@dataclass(frozen=True)
class SystemResult:
    system: str
    runs: List[RunResult]

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[Dict[str, float]]] = {m: [] for m in METHODS}
        for run in self.runs:
            if run.method in grouped:
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


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _mean_metric(
    agg: Dict[str, Dict[str, float]], method: str, metric: str
) -> float:
    return float(agg.get(method, {}).get(metric, float("nan")))


def _verdict_system(
    signal_agg: Dict[str, Dict[str, float]],
    null_agg: Dict[str, Dict[str, float]],
) -> Tuple[bool, str]:
    """Per-system verdict: does CSD feature augmentation help?"""
    reasons = []

    dt_bce = _mean_metric(signal_agg, "Kalman-BCE", "detection_time")
    dt_aug = _mean_metric(signal_agg, "Kalman-CSD-Aug", "detection_time")
    dt_spec = _mean_metric(signal_agg, "Kalman-CSD-Aug-Spec", "detection_time")
    dt_raw = _mean_metric(signal_agg, "Raw-CSD", "detection_time")

    ewa_bce = _mean_metric(signal_agg, "Kalman-BCE", "ew_auc")
    ewa_aug = _mean_metric(signal_agg, "Kalman-CSD-Aug", "ew_auc")

    fpr_aug = _mean_metric(null_agg, "Kalman-CSD-Aug", "fpr")
    fpr_bce = _mean_metric(null_agg, "Kalman-BCE", "fpr")

    dt_gain = dt_aug - dt_bce if (np.isfinite(dt_aug) and np.isfinite(dt_bce)) else float("nan")
    ewa_gain = ewa_aug - ewa_bce if (np.isfinite(ewa_aug) and np.isfinite(ewa_bce)) else float("nan")

    reasons.append(
        f"  Raw-CSD detection time:              {dt_raw:.1f}" if np.isfinite(dt_raw)
        else "  Raw-CSD detection time:              nan"
    )
    reasons.append(
        f"  Kalman-BCE detection time:           {dt_bce:.1f}" if np.isfinite(dt_bce)
        else "  Kalman-BCE detection time:           nan"
    )
    reasons.append(
        f"  Kalman-CSD-Aug detection time:       {dt_aug:.1f}" if np.isfinite(dt_aug)
        else "  Kalman-CSD-Aug detection time:       nan"
    )
    reasons.append(
        f"  Kalman-CSD-Aug-Spec detection time:  {dt_spec:.1f}" if np.isfinite(dt_spec)
        else "  Kalman-CSD-Aug-Spec detection time:  nan"
    )
    reasons.append(
        f"  DT gain (CSD-Aug vs BCE):            {dt_gain:.1f}" if np.isfinite(dt_gain)
        else "  DT gain (CSD-Aug vs BCE):            nan"
    )
    reasons.append(
        f"  EW-AUC gain (CSD-Aug vs BCE):        {ewa_gain:.3f}" if np.isfinite(ewa_gain)
        else "  EW-AUC gain (CSD-Aug vs BCE):        nan"
    )
    reasons.append(
        f"  FPR ratio (CSD-Aug/BCE null):        {fpr_aug / max(fpr_bce, 1e-8):.3f}"
        if (np.isfinite(fpr_aug) and np.isfinite(fpr_bce))
        else "  FPR ratio (CSD-Aug/BCE null):        nan"
    )

    passed_dt = np.isfinite(dt_gain) and dt_gain >= 20.0
    passed_ewa = np.isfinite(ewa_gain) and ewa_gain >= 0.05
    passed_null = not (np.isfinite(fpr_aug) and np.isfinite(fpr_bce) and fpr_aug > 1.5 * fpr_bce + 0.05)

    if passed_dt:
        reasons.append(f"  DT PASS: gain={dt_gain:.1f} >= 20")
    else:
        reasons.append(f"  DT FAIL: gain={dt_gain:.1f}" if np.isfinite(dt_gain) else "  DT FAIL: nan")

    if passed_ewa:
        reasons.append(f"  EW-AUC PASS: gain={ewa_gain:.3f} >= 0.05")
    else:
        reasons.append(f"  EW-AUC FAIL: gain={ewa_gain:.3f}" if np.isfinite(ewa_gain) else "  EW-AUC FAIL: nan")

    if passed_null:
        reasons.append(f"  NULL PASS: FPR ratio={fpr_aug / max(fpr_bce, 1e-8):.3f}")
    else:
        reasons.append(f"  NULL FAIL: FPR ratio={fpr_aug / max(fpr_bce, 1e-8):.3f}")

    system_pass = passed_dt and passed_ewa and passed_null
    return system_pass, "\n".join(reasons)


def _overall_verdict(
    system_results: Dict[str, Tuple[bool, str]],
) -> Tuple[bool, str]:
    n_pass = sum(1 for passed, _ in system_results.values() if passed)
    n_total = len(system_results)

    if n_pass >= 2:
        verdict = (
            f"VERDICT: GO - CSD feature augmentation passes on "
            f"{n_pass}/{n_total} bifurcation systems.\n"
            f"Providing latent autocorrelation + innovation as features improves "
            f"early detection beyond the bare Kalman observer."
        )
        passed = True
    else:
        verdict = (
            f"VERDICT: NO-GO - CSD feature augmentation passes only "
            f"{n_pass}/{n_total} bifurcation systems.\n"
            f"CSD features do not reliably improve early detection over BCE-only observer."
        )
        passed = False

    details = "\n".join(reasons for _, reasons in system_results.values())
    return passed, f"{verdict}\n\nDetails:\n{details}"


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def _run_system_experiment(
    system: str,
    dataset_signal: Dict,
    dataset_null: Dict,
    *,
    seed: int,
    device: torch.device,
    spec_weight: float,
    epochs: int,
) -> SystemResult:
    train_idx_s, val_idx_s, test_idx_s = (
        dataset_signal["split_indices"]["train"],
        dataset_signal["split_indices"]["val"],
        dataset_signal["split_indices"]["test"],
    )
    test_idx_n = dataset_null["split_indices"]["test"]

    runs: List[RunResult] = []

    # --- Raw CSD baseline ---
    _seed_all(seed)
    csd_scores_test = _raw_csd_indicator(
        dataset_signal["features"][test_idx_s], window_size=30
    )
    csd_scores_null_test = _raw_csd_indicator(
        dataset_null["features"][test_idx_n], window_size=30
    )
    raw_metrics = _evaluate_raw_csd(
        csd_scores_test,
        dataset_signal["bifurcation_times"][test_idx_s],
        dataset_signal["is_positive"][test_idx_s],
        dataset_signal["seq_lengths"][test_idx_s],
        csd_scores_null_test,
        dataset_null["seq_lengths"][test_idx_n],
        threshold=0.6,
    )
    runs.append(RunResult(method="Raw-CSD", seed=seed, metrics=raw_metrics))

    # --- Kalman-BCE (head(z) only) ---
    model_bce = _train_kalman(
        dataset_signal, train_idx_s, val_idx_s,
        loss_type="bce",
        seed=seed, device=device, epochs=epochs,
    )
    probs_bce = _build_probs(model_bce, dataset_signal, test_idx_s, device)
    probs_bce_null = _build_probs(model_bce, dataset_null, test_idx_n, device)

    val_probs_bce = _build_probs(model_bce, dataset_signal, val_idx_s, device)
    thresh_bce = _select_threshold(
        val_probs_bce,
        dataset_signal["bifurcation_times"][val_idx_s],
        dataset_signal["is_positive"][val_idx_s],
        dataset_signal["seq_lengths"][val_idx_s],
    )
    dt_bce = _compute_detection_time(
        probs_bce,
        dataset_signal["bifurcation_times"][test_idx_s],
        dataset_signal["is_positive"][test_idx_s],
        dataset_signal["seq_lengths"][test_idx_s],
        thresh_bce,
    )
    ewa_bce = _compute_early_warning_auc(
        probs_bce,
        dataset_signal["bifurcation_times"][test_idx_s],
        dataset_signal["is_positive"][test_idx_s],
        dataset_signal["seq_lengths"][test_idx_s],
        probs_bce_null,
        dataset_null["seq_lengths"][test_idx_n],
    )
    null_bce = _compute_null_metrics(
        probs_bce_null, thresh_bce, dataset_null["seq_lengths"][test_idx_n]
    )
    runs.append(RunResult(method="Kalman-BCE", seed=seed, metrics={
        "detection_time": dt_bce, "ew_auc": ewa_bce, **null_bce,
    }))

    # --- Kalman-CSD-Aug (head(z, rho, residual)) ---
    model_aug = _train_kalman(
        dataset_signal, train_idx_s, val_idx_s,
        loss_type="csd_aug",
        seed=seed, device=device, epochs=epochs,
    )
    probs_aug = _build_probs(model_aug, dataset_signal, test_idx_s, device)
    probs_aug_null = _build_probs(model_aug, dataset_null, test_idx_n, device)

    val_probs_aug = _build_probs(model_aug, dataset_signal, val_idx_s, device)
    thresh_aug = _select_threshold(
        val_probs_aug,
        dataset_signal["bifurcation_times"][val_idx_s],
        dataset_signal["is_positive"][val_idx_s],
        dataset_signal["seq_lengths"][val_idx_s],
    )
    dt_aug = _compute_detection_time(
        probs_aug,
        dataset_signal["bifurcation_times"][test_idx_s],
        dataset_signal["is_positive"][test_idx_s],
        dataset_signal["seq_lengths"][test_idx_s],
        thresh_aug,
    )
    ewa_aug = _compute_early_warning_auc(
        probs_aug,
        dataset_signal["bifurcation_times"][test_idx_s],
        dataset_signal["is_positive"][test_idx_s],
        dataset_signal["seq_lengths"][test_idx_s],
        probs_aug_null,
        dataset_null["seq_lengths"][test_idx_n],
    )
    null_aug = _compute_null_metrics(
        probs_aug_null, thresh_aug, dataset_null["seq_lengths"][test_idx_n]
    )
    runs.append(RunResult(method="Kalman-CSD-Aug", seed=seed, metrics={
        "detection_time": dt_aug, "ew_auc": ewa_aug, **null_aug,
    }))

    # --- Kalman-CSD-Aug-Spec (same + spectral radius constraint) ---
    model_spec = _train_kalman(
        dataset_signal, train_idx_s, val_idx_s,
        loss_type="csd_aug_spec",
        seed=seed, device=device, epochs=epochs,
        spec_weight=spec_weight,
    )
    probs_spec = _build_probs(model_spec, dataset_signal, test_idx_s, device)
    probs_spec_null = _build_probs(model_spec, dataset_null, test_idx_n, device)

    val_probs_spec = _build_probs(model_spec, dataset_signal, val_idx_s, device)
    thresh_spec = _select_threshold(
        val_probs_spec,
        dataset_signal["bifurcation_times"][val_idx_s],
        dataset_signal["is_positive"][val_idx_s],
        dataset_signal["seq_lengths"][val_idx_s],
    )
    dt_spec = _compute_detection_time(
        probs_spec,
        dataset_signal["bifurcation_times"][test_idx_s],
        dataset_signal["is_positive"][test_idx_s],
        dataset_signal["seq_lengths"][test_idx_s],
        thresh_spec,
    )
    ewa_spec = _compute_early_warning_auc(
        probs_spec,
        dataset_signal["bifurcation_times"][test_idx_s],
        dataset_signal["is_positive"][test_idx_s],
        dataset_signal["seq_lengths"][test_idx_s],
        probs_spec_null,
        dataset_null["seq_lengths"][test_idx_n],
    )
    null_spec = _compute_null_metrics(
        probs_spec_null, thresh_spec, dataset_null["seq_lengths"][test_idx_n]
    )
    runs.append(RunResult(method="Kalman-CSD-Aug-Spec", seed=seed, metrics={
        "detection_time": dt_spec, "ew_auc": ewa_spec, **null_spec,
    }))

    return SystemResult(system=system, runs=runs)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _summarize_aggregates(agg: Dict[str, Dict[str, float]]) -> None:
    metrics = ["detection_time", "ew_auc", "fpr"]
    header = f"{'Method':<20s}" + "".join(f"{m:>18s}" for m in metrics)
    print(header)
    print("-" * len(header))
    for method in METHODS:
        values = agg.get(method, {})
        row = "".join(f"{values.get(m, float('nan')):>18.4f}" for m in metrics)
        print(f"{method:<20s}{row}")


def _summarize_system(signal_agg: Dict[str, Dict[str, float]],
                       null_agg: Dict[str, Dict[str, float]],
                       system: str) -> None:
    print(f"\nSystem: {system.capitalize()} Bifurcation")
    print("-" * 70)
    print("Signal:")
    _summarize_aggregates(signal_agg)
    print("\nNull:")
    _summarize_aggregates(null_agg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print()

    settings = [
        ("Default", 0.15, 500, 30),
        ("HighNoise", 0.30, 500, 30),
        ("LowData", 0.15, 200, 50),
    ]

    overall_pass = True

    for label, noise_scale, n_patients, epochs in settings:
        print("=" * 80)
        print(f"Stress Test: {label}")
        print(f"  noise_scale={noise_scale}, n_patients={n_patients}, "
              f"epochs={epochs}")
        print("=" * 80)
        started = time.time()

        system_results: Dict[str, Tuple[bool, str]] = {}

        for system in SYSTEMS:
            print(f"\n--- Generating {system} data ---")
            signal_data = _build_dataset_for_system(
                system, null=False,
                n_trajectories=n_patients, noise_scale=noise_scale,
                seed=101,
            )
            null_data = _build_dataset_for_system(
                system, null=True,
                n_trajectories=n_patients, noise_scale=noise_scale,
                seed=202,
            )

            print(f"  signal: {signal_data['features'].shape}, "
                  f"null: {null_data['features'].shape}")

            all_runs: List[RunResult] = []
            for run_seed in STRESS_SEEDS:
                _seed_all(run_seed)
                sys_res = _run_system_experiment(
                    system, signal_data, null_data,
                    seed=run_seed, device=device,
                    spec_weight=0.1,
                    epochs=epochs,
                )
                all_runs.extend(sys_res.runs)

            combined = SystemResult(system=system, runs=all_runs)
            signal_agg = combined.aggregate()
            null_agg_sys = combined.aggregate()
            _summarize_system(signal_agg, null_agg_sys, system)

            sys_pass, sys_reason = _verdict_system(signal_agg, null_agg_sys)
            system_results[system] = (sys_pass, sys_reason)
            overall_pass = overall_pass and sys_pass

        elapsed = time.time() - started
        print(f"\nTime: {elapsed:.1f}s")
        print("\nSystem Verdicts:")
        for sys_name, (passed, _) in system_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {sys_name.capitalize():>12s}: {status}")

        passed, verdict_str = _overall_verdict(system_results)
        print(f"\n{verdict_str}\n")

    print("=" * 80)
    if overall_pass:
        print("FINAL VERDICT: GO - CSD feature augmentation hypothesis "
              "survived all stress settings.")
    else:
        print("FINAL VERDICT: NO-GO - CSD feature augmentation hypothesis "
              "failed at least one stress setting.")


if __name__ == "__main__":
    main()
