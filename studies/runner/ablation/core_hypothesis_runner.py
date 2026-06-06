"""
studies/runner/ablation/core_hypothesis_runner.py
-------------------------------------------------
The decisive face-off for the BGSL paper.

This runner tests the central thesis:
"Does Biological Gradient Supervised Learning actually outperform a naive
smoothness penalty, or is BGSL just a complicated way to achieve the same thing?"

Conditions:
1. Naive BCE (Baseline)
2. TLS, the strongest direct temporal baseline
3. BCE + Smoothness (L2 penalty on dp)
4. BCE + Total Variation
5. BGSL State-Only
6. Full BGSL (Velocity + Acceleration targets on Soft Onset)

If BGSL cannot beat the direct temporal baseline and the stability penalties on
the 4-metric panel (AUPRC, ASF, Lead Time, POMS), the core hypothesis is weak.

Dataset: select via DATASET constant below.
Model: GRU (fixed — we isolate the loss, not the architecture)
Seeds: 3 (statistical significance baseline; bump for final paper)
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
# Default dataset used when no --dataset flag is provided.
# Override at runtime:
#   python core_hypothesis_runner.py --dataset cmapss

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
        description="Core hypothesis ablation runner (BGSL vs baselines)."
    )
    parser.add_argument(
        "--dataset",
        choices=list(_BASE_CONFIGS.keys()),
        default=_DEFAULT_DATASET,
        help=(
            f"Dataset to train on. "
            f"Options: {sorted(_BASE_CONFIGS)}. "
            f"Default: {_DEFAULT_DATASET!r}."
        ),
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
        help="Random seeds for each condition. Default: 42 43 44.",
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
        {
            "name": "01_BCE_Baseline",
            "loss": {"class_path": "bgsl.core.common.losses.BCELoss"},
        },
        {
            "name": "02_TLS",
            "loss": {
                "class_path": "bgsl.core.common.losses.TLSLoss",
                "init_args": {"alpha": 6.0},
            },
        },
        {
            "name": "03_BCE_Smoothness",
            "loss": {
                "class_path": "bgsl.core.common.losses.SmoothnessLoss",
                "init_args": {"smoothness_weight": 0.1},
            },
        },
        {
            "name": "04_BCE_TotalVariation",
            "loss": {
                "class_path": "bgsl.core.common.losses.TotalVariationLoss",
                "init_args": {"tv_weight": 0.1},
            },
        },
        {
            "name": "05_BGSL_StateOnly",
            "loss": {
                "class_path": "bgsl.core.common.losses.BGSLLoss",
                "init_args": {
                    "velocity_weight": 0.0,
                    "acceleration_weight": 0.0,
                },
            },
        },
        {
            "name": "06_Full_BGSL",
            "loss": {
                "class_path": "bgsl.core.common.losses.BGSLLoss",
                "init_args": {
                    "velocity_weight": 0.10,
                    "acceleration_weight": 0.05,
                },
            },
        },
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
