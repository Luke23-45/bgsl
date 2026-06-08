from __future__ import annotations

"""
verify_bgsl_loss_significance.py
---------------------------------
Reproducible diagnostic for the BGSL trajectory terms.

The script answers a narrow question:

    Are the velocity / acceleration terms materially changing the objective,
    or are they numerically tiny compared to the state loss?

It evaluates two cohorts:
    1. A deterministic synthetic cohort generated from the same soft-onset
       target builder used by the implementation.
    2. Optionally, the processed PhysioNet sepsis cohort from this repository.

For each cohort, it reports:
    - state loss
    - velocity loss
    - acceleration loss
    - weighted trajectory contribution (lambda_v * vel + lambda_a * acc)
    - ratio of the trajectory contribution to the state loss
    - gradient norm ratios on logits for the individual terms

The script is intentionally self-contained and deterministic.
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from bgsl.core.common.losses import BGSLLoss
from bgsl.core.sepsis.targets import SoftOnsetTarget


try:
    from bgsl.data.sepsis.physionet2019 import PhysioNet2019Dataset, collate_trajectories
except Exception:  # pragma: no cover - optional real-data path
    PhysioNet2019Dataset = None
    collate_trajectories = None


@dataclass(frozen=True)
class CohortResult:
    name: str
    samples: int
    valid_steps: int
    state_loss: float
    velocity_loss: float
    acceleration_loss: float
    total_loss: float
    state_only_total: float
    trajectory_contrib: float
    trajectory_to_state: float
    delta_total: float
    state_grad_norm: float
    velocity_grad_norm: float
    acceleration_grad_norm: float
    trajectory_to_state_grad: float
    mean_abs_velocity_target: float
    mean_abs_acceleration_target: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pad_1d(values: Sequence[torch.Tensor], *, pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(int(v.shape[0]) for v in values)
    batch = len(values)
    out = torch.full((batch, max_len), pad_value, dtype=values[0].dtype)
    for i, value in enumerate(values):
        out[i, : value.shape[0]] = value
    return out


def _pad_2d(values: Sequence[torch.Tensor], *, pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(int(v.shape[0]) for v in values)
    feat_dim = int(values[0].shape[1])
    batch = len(values)
    out = torch.full((batch, max_len, feat_dim), pad_value, dtype=values[0].dtype)
    for i, value in enumerate(values):
        out[i, : value.shape[0]] = value
    return out


def _soft_target_onset(
    onset_hour: int,
    seq_len: int,
    device: torch.device,
    *,
    horizon_hours: float = 6.0,
    tau: float = 2.0,
) -> Dict[str, torch.Tensor]:
    builder = SoftOnsetTarget(horizon_hours=horizon_hours, tau=tau, kappa=1.0)
    return builder.build_single(onset_hour=onset_hour, seq_len=seq_len, device=device)


def build_synthetic_cohort(
    *,
    samples: int,
    seed: int,
    horizon_hours: float = 6.0,
    tau: float = 2.0,
    min_len: int = 12,
    max_len: int = 48,
    positive_rate: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Build a deterministic synthetic cohort that mirrors the repository batch schema.

    The synthetic predictor is not trained. The point is to isolate the numerical
    scale of the BGSL trajectory terms under a controlled, reproducible setup.
    """
    _seed_everything(seed)
    rng = np.random.default_rng(seed)

    vitals: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    hard_labels: List[torch.Tensor] = []
    soft_targets: List[torch.Tensor] = []
    hard_targets: List[torch.Tensor] = []
    vel_targets: List[torch.Tensor] = []
    acc_targets: List[torch.Tensor] = []
    valid_mask: List[torch.Tensor] = []
    vel_mask: List[torch.Tensor] = []
    acc_mask: List[torch.Tensor] = []
    onset_hour: List[torch.Tensor] = []
    is_sepsis: List[torch.Tensor] = []
    seq_len: List[torch.Tensor] = []
    patient_id: List[str] = []
    static_covariates: List[torch.Tensor] = []

    for idx in range(samples):
        T = int(rng.integers(min_len, max_len + 1))
        positive = bool(rng.random() < positive_rate)
        if positive:
            # Keep the event far enough inside the window to expose a non-trivial
            # pre-onset region while still being fully inside the sequence.
            lower = max(int(math.ceil(horizon_hours)) + 1, 2)
            upper = max(lower + 1, T - 1)
            onset = int(rng.integers(lower, upper))
            onset = min(onset, T - 1)
        else:
            onset = -1

        tgt = _soft_target_onset(
            onset,
            T,
            torch.device("cpu"),
            horizon_hours=horizon_hours,
            tau=tau,
        )

        soft = tgt["soft_target"].squeeze(0).float()
        hard = tgt["hard_target"].squeeze(0).float()
        vel = tgt["velocity_target"].squeeze(0).float()
        acc = tgt["accel_target"].squeeze(0).float()
        valid = tgt["valid_mask"].squeeze(0).float()
        vmask = tgt["vel_mask"].squeeze(0).float()
        amask = tgt["acc_mask"].squeeze(0).float()

        # The dynamic features are not used by the loss, but including them makes
        # the synthetic batch schema match the real dataset closely.
        vitals.append(torch.randn(T, 40))
        masks.append(torch.ones(T, 40))
        hard_labels.append(hard.clone())
        soft_targets.append(soft)
        hard_targets.append(hard)
        vel_targets.append(vel)
        acc_targets.append(acc)
        valid_mask.append(valid)
        vel_mask.append(vmask)
        acc_mask.append(amask)
        onset_hour.append(torch.tensor(onset, dtype=torch.long))
        is_sepsis.append(torch.tensor(positive, dtype=torch.bool))
        seq_len.append(torch.tensor(T, dtype=torch.long))
        patient_id.append(f"synthetic_{idx:04d}")
        static_covariates.append(torch.randn(5))

    return {
        "vitals": _pad_2d(vitals),
        "masks": _pad_2d(masks),
        "hard_labels": _pad_1d(hard_labels),
        "soft_targets": _pad_1d(soft_targets),
        "hard_targets": _pad_1d(hard_targets),
        "vel_targets": _pad_1d(vel_targets),
        "accel_targets": _pad_1d(acc_targets),
        "valid_mask": _pad_1d(valid_mask),
        "vel_mask": _pad_1d(vel_mask),
        "acc_mask": _pad_1d(acc_mask),
        "onset_hour": torch.stack(onset_hour),
        "is_sepsis": torch.stack(is_sepsis),
        "seq_len": torch.stack(seq_len),
        "patient_id": patient_id,
        "static_covariates": torch.stack(static_covariates),
    }


def load_real_cohort(
    *,
    data_dir: str,
    split: str,
    limit: Optional[int],
    seed: int,
) -> Dict[str, torch.Tensor]:
    if PhysioNet2019Dataset is None or collate_trajectories is None:
        raise RuntimeError("Real-data path is unavailable because the dataset module could not be imported.")

    dataset = PhysioNet2019Dataset(
        data_dir=data_dir,
        split=split,
        min_length=8,
        horizon_hours=6,
        tau=2.0,
        normalizer=None,
    )
    rng = np.random.default_rng(seed)
    if limit is None:
        indices = list(range(len(dataset)))
    else:
        limit = max(1, min(int(limit), len(dataset)))
        positives = [i for i, ep in enumerate(dataset.episodes) if ep["is_sepsis"]]
        negatives = [i for i, ep in enumerate(dataset.episodes) if not ep["is_sepsis"]]

        if positives and negatives:
            pos_count = min(len(positives), max(1, limit // 2))
            neg_count = min(len(negatives), limit - pos_count)
            if pos_count + neg_count < limit:
                pos_count = min(len(positives), limit - neg_count)
            if pos_count + neg_count < limit:
                neg_count = min(len(negatives), limit - pos_count)

            pos_pick = rng.choice(positives, size=pos_count, replace=False).tolist()
            neg_pick = rng.choice(negatives, size=neg_count, replace=False).tolist()
            indices = pos_pick + neg_pick
            rng.shuffle(indices)
        else:
            indices = rng.choice(len(dataset), size=limit, replace=False).tolist()

    samples = [dataset[i] for i in indices]
    batch = collate_trajectories(samples)
    dataset.close()
    return batch


def _proxy_logits(
    soft_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    seed: int,
    noise_scale: float,
    bias: float = 0.0,
) -> torch.Tensor:
    """
    Build a reproducible proxy predictor from the soft targets.

    This is intentionally conservative: we perturb the target trajectory only
    a little so we can measure whether the derivative supervision still matters
    when the classifier is already close to the target shape.
    """
    _seed_everything(seed)
    eps = 1e-4
    base = torch.logit(soft_targets.clamp(eps, 1.0 - eps))
    noise = torch.randn_like(base) * noise_scale
    logits = base + noise + bias
    return torch.where(valid_mask > 0, logits, torch.zeros_like(logits))


def _compute_grad_norm(
    loss_fn: BGSLLoss,
    batch: Dict[str, torch.Tensor],
    logits: torch.Tensor,
    component: str,
) -> float:
    probe_logits = logits.detach().clone().requires_grad_(True)
    out = loss_fn(
        probe_logits,
        batch["soft_targets"],
        batch["valid_mask"],
        vel_targets=batch["vel_targets"],
        acc_targets=batch["accel_targets"],
        vel_mask=batch["vel_mask"],
        acc_mask=batch["acc_mask"],
        onset_times=batch["onset_hour"],
        is_positive=batch["is_sepsis"],
        seq_lengths=batch["seq_len"],
    )

    if component == "state":
        scalar = out["state"]
    elif component == "velocity":
        scalar = loss_fn.lambda_v * out["velocity"]
    elif component == "acceleration":
        scalar = loss_fn.lambda_a * out["acceleration"]
    else:
        raise ValueError(f"Unknown gradient component: {component!r}")

    scalar.backward()
    grad = probe_logits.grad
    return float(grad.norm().item()) if grad is not None else 0.0


def evaluate_cohort(
    *,
    name: str,
    batch: Dict[str, torch.Tensor],
    seed: int,
    noise_scale: float,
    loss_fn: BGSLLoss,
) -> CohortResult:
    logits = _proxy_logits(
        batch["soft_targets"],
        batch["valid_mask"],
        seed=seed,
        noise_scale=noise_scale,
    )

    out = loss_fn(
        logits,
        batch["soft_targets"],
        batch["valid_mask"],
        vel_targets=batch["vel_targets"],
        acc_targets=batch["accel_targets"],
        vel_mask=batch["vel_mask"],
        acc_mask=batch["acc_mask"],
        onset_times=batch["onset_hour"],
        is_positive=batch["is_sepsis"],
        seq_lengths=batch["seq_len"],
    )

    state_loss = float(out["state"].item())
    velocity_loss = float(out["velocity"].item())
    acceleration_loss = float(out["acceleration"].item())
    total_loss = float(out["loss"].item())
    trajectory_contrib = float(loss_fn.lambda_v * out["velocity"].item() + loss_fn.lambda_a * out["acceleration"].item())
    state_only_total = state_loss
    delta_total = total_loss - state_only_total
    valid_steps = int(batch["valid_mask"].sum().item())
    trajectory_to_state = trajectory_contrib / max(state_loss, 1e-12)

    state_grad_norm = _compute_grad_norm(loss_fn, batch, logits, "state")
    velocity_grad_norm = _compute_grad_norm(loss_fn, batch, logits, "velocity")
    acceleration_grad_norm = _compute_grad_norm(loss_fn, batch, logits, "acceleration")
    trajectory_to_state_grad = (
        (velocity_grad_norm + acceleration_grad_norm) / max(state_grad_norm, 1e-12)
    )

    mean_abs_velocity_target = float(batch["vel_targets"].abs().sum().item() / max(valid_steps, 1))
    mean_abs_acceleration_target = float(batch["accel_targets"].abs().sum().item() / max(valid_steps, 1))

    return CohortResult(
        name=name,
        samples=int(batch["soft_targets"].shape[0]),
        valid_steps=valid_steps,
        state_loss=state_loss,
        velocity_loss=velocity_loss,
        acceleration_loss=acceleration_loss,
        total_loss=total_loss,
        state_only_total=state_only_total,
        trajectory_contrib=trajectory_contrib,
        trajectory_to_state=trajectory_to_state,
        delta_total=delta_total,
        state_grad_norm=state_grad_norm,
        velocity_grad_norm=velocity_grad_norm,
        acceleration_grad_norm=acceleration_grad_norm,
        trajectory_to_state_grad=trajectory_to_state_grad,
        mean_abs_velocity_target=mean_abs_velocity_target,
        mean_abs_acceleration_target=mean_abs_acceleration_target,
    )


def _format_result(result: CohortResult) -> str:
    lines = [
        f"[{result.name}] samples={result.samples} valid_steps={result.valid_steps}",
        f"  state_loss               = {result.state_loss:.8f}",
        f"  velocity_loss            = {result.velocity_loss:.8f}",
        f"  acceleration_loss        = {result.acceleration_loss:.8f}",
        f"  weighted_trajectory_term = {result.trajectory_contrib:.8f}",
        f"  total_loss               = {result.total_loss:.8f}",
        f"  state_only_total         = {result.state_only_total:.8f}",
        f"  total_minus_state        = {result.delta_total:.8f}",
        f"  trajectory/state         = {100.0 * result.trajectory_to_state:.4f}%",
        f"  state_grad_norm          = {result.state_grad_norm:.8f}",
        f"  velocity_grad_norm       = {result.velocity_grad_norm:.8f}",
        f"  acceleration_grad_norm   = {result.acceleration_grad_norm:.8f}",
        f"  grad_ratio(traj/state)   = {100.0 * result.trajectory_to_state_grad:.4f}%",
        f"  mean|vel_target|         = {result.mean_abs_velocity_target:.8f}",
        f"  mean|acc_target|         = {result.mean_abs_acceleration_target:.8f}",
    ]
    return "\n".join(lines)


def _result_to_dict(result: CohortResult) -> Dict[str, float]:
    return {
        "samples": result.samples,
        "valid_steps": result.valid_steps,
        "state_loss": result.state_loss,
        "velocity_loss": result.velocity_loss,
        "acceleration_loss": result.acceleration_loss,
        "total_loss": result.total_loss,
        "state_only_total": result.state_only_total,
        "trajectory_contrib": result.trajectory_contrib,
        "trajectory_to_state": result.trajectory_to_state,
        "delta_total": result.delta_total,
        "state_grad_norm": result.state_grad_norm,
        "velocity_grad_norm": result.velocity_grad_norm,
        "acceleration_grad_norm": result.acceleration_grad_norm,
        "trajectory_to_state_grad": result.trajectory_to_state_grad,
        "mean_abs_velocity_target": result.mean_abs_velocity_target,
        "mean_abs_acceleration_target": result.mean_abs_acceleration_target,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify whether the BGSL trajectory terms are numerically significant."
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real", "both"],
        default="both",
        help="Which cohort(s) to evaluate. Default: both.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Synthetic cohort size. Default: 128.",
    )
    parser.add_argument(
        "--real-samples",
        type=int,
        default=64,
        help="How many real samples to use when --mode includes real. Default: 64.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(REPO_ROOT / "datasets" / "processed" / "sepsis"),
        help="Processed sepsis dataset root for real-data verification.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Real-data split to evaluate. Default: train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for synthetic generation and proxy predictions. Default: 7.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.35,
        help="Noise scale applied to the proxy logits. Default: 0.35.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to write the summary as JSON.",
    )
    parser.add_argument(
        "--loss-ratio-threshold",
        type=float,
        default=0.01,
        help="Scalar trajectory/state ratio above which the trajectory terms are treated as non-negligible. Default: 0.01 (1%%).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    loss_fn = BGSLLoss(
        velocity_weight=0.1,
        acceleration_weight=0.05,
        monotonicity_weight=0.0,
        weighting_method="fixed",
        derivative_space="probability",
        derivative_method="finite_diff",
    )
    loss_fn.eval()

    results: List[CohortResult] = []

    if args.mode in {"synthetic", "both"}:
        synthetic = build_synthetic_cohort(
            samples=args.samples,
            seed=args.seed,
        )
        results.append(
            evaluate_cohort(
                name="synthetic",
                batch=synthetic,
                seed=args.seed,
                noise_scale=args.noise_scale,
                loss_fn=loss_fn,
            )
        )

    if args.mode in {"real", "both"}:
        try:
            real = load_real_cohort(
                data_dir=args.data_dir,
                split=args.split,
                limit=args.real_samples,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"[real] skipped: {exc}")
        else:
            results.append(
                evaluate_cohort(
                    name=f"real:{args.split}",
                    batch=real,
                    seed=args.seed,
                    noise_scale=args.noise_scale,
                    loss_fn=loss_fn,
                )
            )

    if not results:
        print("No cohorts were evaluated.")
        return 1

    payload = {
        "seed": args.seed,
        "noise_scale": args.noise_scale,
        "loss": {
            "velocity_weight": loss_fn.lambda_v,
            "acceleration_weight": loss_fn.lambda_a,
            "monotonicity_weight": loss_fn.lambda_mono,
            "derivative_method": loss_fn.derivative_method,
            "derivative_space": loss_fn.derivative_space,
        },
        "results": {result.name: _result_to_dict(result) for result in results},
    }

    for result in results:
        print(_format_result(result))
        print()

    if args.report_json is not None:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {report_path}")

    # The pass/fail criterion is based on the scalar objective contribution.
    # Gradient ratio is reported separately because it can remain meaningful
    # even when the weighted loss value itself is tiny.
    worst_ratio = max(result.trajectory_to_state for result in results)
    worst_grad_ratio = max(result.trajectory_to_state_grad for result in results)
    if worst_ratio > args.loss_ratio_threshold:
        print(
            "Trajectory terms are not negligible in the scalar objective: "
            f"loss_ratio={100.0 * worst_ratio:.2f}% "
            f"(threshold={100.0 * args.loss_ratio_threshold:.2f}%), "
            f"grad_ratio={100.0 * worst_grad_ratio:.2f}%"
        )
        return 2

    print(
        "Trajectory terms are small in the scalar objective: "
        f"loss_ratio={100.0 * worst_ratio:.2f}% "
        f"(threshold={100.0 * args.loss_ratio_threshold:.2f}%), "
        f"grad_ratio={100.0 * worst_grad_ratio:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
