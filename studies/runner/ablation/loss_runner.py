"""
studies/runner/ablation/loss_runner.py
--------------------------------------
Ablation study over BGSL loss components and direct baselines.

Dataset: select via DATASET constant below.
Model: GRU (fixed — isolates the loss component, not the architecture)
Seeds: 3
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from studies.runner.commons.cli_builder import build_run_spec
from studies.runner.commons.ablation_conditions import build_condition_overrides
from studies.runner.commons.executor import execute_runs
from studies.runner.commons.paths import (
    build_batch_paths,
    infer_experiment_id_from_file,
)

# ---------------------------------------------------------------------------
# Dataset selector
# ---------------------------------------------------------------------------
_DEFAULT_DATASET: str = "sepsis"

_BASE_CONFIGS: dict = {
    "sepsis": "src/bgsl/config/experiments/sepsis/gru.yaml",
    "cmapss": "src/bgsl/config/experiments/cmapss/gru.yaml",
}


def _get_base_config(dataset: str) -> str:
    if dataset not in _BASE_CONFIGS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            f"Valid options: {sorted(_BASE_CONFIGS)}"
        )
    return _BASE_CONFIGS[dataset]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loss component ablation runner."
    )
    parser.add_argument(
        "--dataset",
        choices=list(_BASE_CONFIGS.keys()),
        default=_DEFAULT_DATASET,
        help=f"Dataset to train on. Options: {sorted(_BASE_CONFIGS)}. Default: {_DEFAULT_DATASET!r}.",
    )
    parser.add_argument(
        "--mode",
        choices=["dry_run", "local_sequential", "slurm"],
        default="local_sequential",
        help="Execution mode. Default: local_sequential.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds. Default: 42 43 44.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    dataset = args.dataset
    seeds = args.seeds
    mode = args.mode

    experiment_name, experiment_type = infer_experiment_id_from_file(__file__)
    batch_paths = build_batch_paths(
        experiment_name=f"{experiment_name}/{dataset}",
        experiment_type=experiment_type,
    )

    base_config = _get_base_config(dataset)

    conditions = [
        # Baselines
        {"name": "baseline_bce",          "loss": {"class_path": "bgsl.core.common.losses.BCELoss"}},
        {"name": "baseline_weighted_bce", "loss": {"class_path": "bgsl.core.common.losses.WeightedBCELoss", "init_args": {"pos_weight": 10.0}}},
        {"name": "baseline_focal",        "loss": {"class_path": "bgsl.core.common.losses.FocalLoss",       "init_args": {"gamma": 2.0}}},
        {"name": "baseline_tls",          "loss": {"class_path": "bgsl.core.common.losses.TLSLoss",         "init_args": {"alpha": 6.0}}},
        {"name": "baseline_smoothness",   "loss": {"class_path": "bgsl.core.common.losses.SmoothnessLoss",  "init_args": {"smoothness_weight": 0.1}}},
        {"name": "baseline_tv",           "loss": {"class_path": "bgsl.core.common.losses.TotalVariationLoss", "init_args": {"tv_weight": 0.1}}},

        # BGSL family
        {"name": "bgsl_state_only",    "loss": {"class_path": "bgsl.core.common.losses.BGSLLoss", "init_args": {"velocity_weight": 0.0,  "acceleration_weight": 0.0}}},
        {"name": "bgsl_velocity_only", "loss": {"class_path": "bgsl.core.common.losses.BGSLLoss", "init_args": {"velocity_weight": 0.1,  "acceleration_weight": 0.0}}},
        {"name": "bgsl_accel_only",    "loss": {"class_path": "bgsl.core.common.losses.BGSLLoss", "init_args": {"velocity_weight": 0.0,  "acceleration_weight": 0.05}}},
        {"name": "bgsl_full",          "loss": {"class_path": "bgsl.core.common.losses.BGSLLoss", "init_args": {"velocity_weight": 0.1,  "acceleration_weight": 0.05}}},
    ]

    runs = []
    for cond in conditions:
        for seed in seeds:
            run_id = f"{cond['name']}_seed{seed}"
            run_paths = batch_paths.run_paths(run_id)
            runs.append(build_run_spec(
                base_config=base_config,
                run_paths=run_paths,
                run_id=run_id,
                cond_name=cond["name"],
                seed=seed,
                overrides=build_condition_overrides(cond),
            ))

    execute_runs(runs, batch_paths, mode=mode)


if __name__ == "__main__":
    main()
