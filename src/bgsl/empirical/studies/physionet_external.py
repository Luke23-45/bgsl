"""
src/bgsl/empirical/studies/physionet_external.py
------------------------------------------------
PhysioNet 2019 external validation study.

Runs frozen methods (BCE, BGSL-state, BGSL+vel) on the real PhysioNet
2019 sepsis dataset using the existing acquisition and training pipeline.
No tuning, no new preprocessing, no architecture changes.

Grid
----
- methods:  BCE, BGSL-state, BGSL+vel
- seeds:    42, 43, 44

Total planned jobs: 3 × 3 = 9.

Usage
-----
    python -m bgsl.empirical.studies.physionet_external --mode dry_run
    python -m bgsl.empirical.studies.physionet_external --mode local_sequential
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root and src are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from studies.runner.commons.cli_builder import RunSpec, build_run_spec
from studies.runner.commons.executor import execute_runs
from studies.runner.commons.paths import build_batch_paths
from studies.runner.aggregate import run_aggregate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reuse the existing PhysioNet GRU experiment config (frozen — no changes).
BASE_CONFIG: str = "src/bgsl/config/experiments/sepsis/gru.yaml"

METHODS_ORDER: Tuple[str, ...] = ("BCE", "BGSL-state", "BGSL+vel")
DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44)

_METHOD_LOSS: Dict[str, Dict[str, Any]] = {
    "BCE": {
        "cond_name": "BCE",
        "loss_class": "bgsl.core.common.losses.BCELoss",
        "loss_args": {},
    },
    "BGSL-state": {
        "cond_name": "BGSL_state",
        "loss_class": "bgsl.core.common.losses.BGSLLoss",
        "loss_args": {
            "state_loss": "bce",
            "velocity_weight": 0.0,
            "acceleration_weight": 0.0,
            "monotonicity_weight": 0.0,
            "derivative_method": "finite_diff",
        },
    },
    "BGSL+vel": {
        "cond_name": "BGSL_vel",
        "loss_class": "bgsl.core.common.losses.BGSLLoss",
        "loss_args": {
            "state_loss": "bce",
            "velocity_weight": 0.05,
            "acceleration_weight": 0.0,
            "monotonicity_weight": 0.0,
            "derivative_method": "finite_diff",
        },
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhysioNet 2019 external validation for the BGSL hypothesis.",
    )
    parser.add_argument(
        "--mode",
        choices=["dry_run", "local_sequential", "slurm"],
        default="dry_run",
        help="Execution mode. Default: dry_run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS_ORDER),
        default=list(METHODS_ORDER),
        help=f"Method(s). Default: {' '.join(METHODS_ORDER)}.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Training seeds. Default: {' '.join(map(str, DEFAULT_SEEDS))}.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run-spec builder
# ---------------------------------------------------------------------------


def _build_method_overrides(method: str) -> Dict[str, Any]:
    """Construct the flat dot-path overrides dict for one method."""
    cfg = _METHOD_LOSS[method]
    overrides: Dict[str, Any] = {}

    overrides["model.init_args.loss_fn"] = cfg["loss_class"]
    for k, v in cfg["loss_args"].items():
        overrides[f"model.init_args.loss_fn.init_args.{k}"] = v

    return overrides


def build_run_specs(
    methods: List[str],
    seeds: List[int],
) -> Tuple[Any, List[RunSpec]]:
    """Build the batch paths and all run specs.

    Returns
    -------
    batch_paths : BatchPaths
    run_specs   : list of RunSpec
    """
    batch_paths = build_batch_paths(
        experiment_name="empirical",
        experiment_type="physionet_external",
    )

    run_specs: List[RunSpec] = []

    for method in methods:
        cond_name = _METHOD_LOSS[method]["cond_name"]
        overrides = _build_method_overrides(method)

        for seed in seeds:
            run_id = f"{cond_name}_seed{seed}"
            run_paths = batch_paths.run_paths(run_id)
            spec = build_run_spec(
                base_config=BASE_CONFIG,
                run_paths=run_paths,
                run_id=run_id,
                cond_name=cond_name,
                seed=seed,
                overrides=overrides,
            )
            run_specs.append(spec)

    return batch_paths, run_specs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    batch_paths, run_specs = build_run_specs(
        methods=args.methods,
        seeds=args.seeds,
    )

    print(f"Total planned jobs: {len(run_specs)}")
    print(f"  methods: {args.methods}")
    print(f"  seeds:   {args.seeds}")
    print()

    execute_runs(run_specs, batch_paths, mode=args.mode)

    # Post-processing for local runs
    if args.mode == "local_sequential":
        print("\n" + "=" * 72)
        print("  POST-PROCESSING")
        print("=" * 72)
        run_aggregate(batch_paths.root)


if __name__ == "__main__":
    main()
