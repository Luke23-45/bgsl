"""
studies/runner/architecture/paired_runner.py
--------------------------------------------
Paired raw-vs-BGSL architecture study.

This runner answers a different question from the fixed-GRU loss ablation:

    "For a given architecture, does adding BGSL change the result relative
    to the same architecture trained with plain BCE?"

The experiment is intentionally paired:
    - GRU raw vs GRU + BGSL
    - TCN raw vs TCN + BGSL
    - Transformer raw vs Transformer + BGSL

Only one dataset is selected per invocation so the study remains easy to
interpret and the output tree stays clean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from studies.runner.commons.ablation_conditions import build_condition_overrides
from studies.runner.commons.cli_builder import RunSpec, build_run_spec
from studies.runner.commons.executor import execute_runs
from studies.runner.commons.paths import (
    BatchPaths,
    build_batch_paths,
    infer_experiment_id_from_file,
)
from studies.runner.aggregate import run_aggregate
from studies.runner.visualize import run_visualize

# ---------------------------------------------------------------------------
# Dataset selector
# ---------------------------------------------------------------------------

_DEFAULT_DATASET: str = "sepsis"

_BASE_CONFIGS: Dict[str, Dict[str, str]] = {
    "sepsis": {
        "gru": "src/bgsl/config/experiments/sepsis/gru.yaml",
        "tcn": "src/bgsl/config/experiments/sepsis/tcn.yaml",
        "transformer": "src/bgsl/config/experiments/sepsis/transformer.yaml",
    },
    "cmapss": {
        "gru": "src/bgsl/config/experiments/cmapss/gru.yaml",
        "tcn": "src/bgsl/config/experiments/cmapss/tcn.yaml",
        "transformer": "src/bgsl/config/experiments/cmapss/transformer.yaml",
    },
}

_ARCHITECTURE_ORDER: Tuple[str, ...] = ("gru", "tcn", "transformer")


def _get_base_configs(dataset: str) -> Dict[str, str]:
    if dataset not in _BASE_CONFIGS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Valid options: {sorted(_BASE_CONFIGS)}"
        )
    return _BASE_CONFIGS[dataset]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired architecture runner: raw BCE vs BGSL for each model family."
    )
    parser.add_argument(
        "--dataset",
        choices=list(_BASE_CONFIGS.keys()),
        default=_DEFAULT_DATASET,
        help=(
            f"Dataset to run. Options: {sorted(_BASE_CONFIGS)}. "
            f"Default: {_DEFAULT_DATASET!r}. Run the command twice if you want both datasets."
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
        help="Random seeds. Default: 42 43 44.",
    )
    return parser.parse_args()


def build_paired_conditions(dataset: str) -> List[Dict[str, object]]:
    """
    Build the six paired conditions for one dataset.

    Raw conditions explicitly swap in plain BCE.
    BGSL conditions intentionally keep the architecture-specific base config.
    """
    base_configs = _get_base_configs(dataset)

    conditions: List[Dict[str, object]] = []
    for arch in _ARCHITECTURE_ORDER:
        base_config = base_configs[arch]
        raw_name = f"{dataset}_{arch}_raw"
        bgsl_name = f"{dataset}_{arch}_bgsl"

        raw_overrides = build_condition_overrides(
            {
                "name": raw_name,
                "loss": {
                    "class_path": "bgsl.core.common.losses.BCELoss",
                },
            }
        )

        conditions.append(
            {
                "name": raw_name,
                "cond_name": f"{dataset}_{arch}_raw",
                "base_config": base_config,
                "overrides": raw_overrides,
                "architecture": arch,
                "loss_family": "raw",
            }
        )
        conditions.append(
            {
                "name": bgsl_name,
                "cond_name": f"{dataset}_{arch}_bgsl",
                "base_config": base_config,
                "overrides": {},
                "architecture": arch,
                "loss_family": "bgsl",
            }
        )

    return conditions


def build_run_specs(
    dataset: str,
    seeds: List[int],
) -> Tuple[BatchPaths, List[RunSpec], List[Dict[str, object]]]:
    """
    Build the batch paths, run specs, and the underlying condition metadata.
    """
    experiment_name, experiment_type = infer_experiment_id_from_file(__file__)
    batch_paths = build_batch_paths(
        experiment_name=experiment_name,
        experiment_type=experiment_type,
    )

    run_specs: List[RunSpec] = []
    conditions = build_paired_conditions(dataset)

    for cond in conditions:
        for seed in seeds:
            run_id = f"{cond['cond_name']}_seed{seed}"
            run_paths = batch_paths.run_paths(run_id)
            run_specs.append(
                build_run_spec(
                    base_config=str(cond["base_config"]),
                    run_paths=run_paths,
                    run_id=run_id,
                    cond_name=str(cond["cond_name"]),
                    seed=seed,
                    overrides=dict(cond["overrides"]),
                )
            )

    return batch_paths, run_specs, conditions


def main() -> None:
    args = _parse_args()
    batch_paths, run_specs, _ = build_run_specs(args.dataset, args.seeds)
    execute_runs(run_specs, batch_paths, mode=args.mode)

    # Post-processing: aggregate results and generate trajectory plots
    if args.mode == "local_sequential":
        print("\n" + "=" * 72)
        print("  POST-PROCESSING")
        print("=" * 72)

        run_aggregate(batch_paths.root)
        run_visualize(batch_paths.root)


if __name__ == "__main__":
    main()
