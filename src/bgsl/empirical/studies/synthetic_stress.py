"""
src/bgsl/empirical/studies/synthetic_stress.py
----------------------------------------------
Synthetic stress-test study for the BGSL hypothesis.

Grid
----
- scenarios:  signal, null
- stress:     Default, HighVel, LongTrain
- methods:    BCE, BGSL-state, BGSL+vel
- seeds:      101, 202, 303, 404, 505

Total planned jobs: 3 × 2 × 3 × 5 = 90.

Usage
-----
    python -m bgsl.empirical.studies.synthetic_stress --mode dry_run
    python -m bgsl.empirical.studies.synthetic_stress --mode local_sequential
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

BASE_CONFIG: str = "src/bgsl/empirical/configs/synthetic_gru.yaml"

STRESS_SETTINGS: Dict[str, Dict[str, Any]] = {
    "Default":   {"velocity_weight": 0.05, "max_epochs": 16, "tau": 2.5},
    "HighVel":   {"velocity_weight": 0.50, "max_epochs": 16, "tau": 1.5},
    "LongTrain": {"velocity_weight": 0.05, "max_epochs": 32, "tau": 2.5},
}

DEFAULT_STRESS: Tuple[str, ...] = tuple(STRESS_SETTINGS.keys())
SCENARIOS: Tuple[str, ...] = ("signal", "null")
METHODS_ORDER: Tuple[str, ...] = ("BCE", "BGSL-state", "BGSL+vel", "TLS")
DEFAULT_SEEDS: Tuple[int, ...] = (101, 202, 303, 404, 505)

# Method-specific loss class paths and fixed init args
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
            "acceleration_weight": 0.0,
            "monotonicity_weight": 0.0,
            "derivative_method": "finite_diff",
        },
        # ``velocity_weight`` is injected from the stress setting at run time
    },
    "TLS": {
        "cond_name": "TLS",
        "loss_class": "bgsl.core.common.losses.TLSLoss",
        "loss_args": {
            "alpha": 6.0,
        },
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic stress-test study for the BGSL hypothesis.",
    )
    parser.add_argument(
        "--mode",
        choices=["dry_run", "local_sequential", "slurm"],
        default="dry_run",
        help="Execution mode. Default: dry_run.",
    )
    parser.add_argument(
        "--stress",
        nargs="+",
        choices=list(STRESS_SETTINGS.keys()),
        default=list(DEFAULT_STRESS),
        help=f"Stress setting(s). Default: {' '.join(DEFAULT_STRESS)}.",
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=list(SCENARIOS),
        default=list(SCENARIOS),
        help=f"Scenario(s). Default: {' '.join(SCENARIOS)}.",
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


def _build_method_overrides(method: str, stress_setting: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the flat dot-path overrides dict for one method×stress cell."""
    cfg = _METHOD_LOSS[method]
    overrides: Dict[str, Any] = {}

    # Loss class
    overrides["model.init_args.loss_fn"] = cfg["loss_class"]

    # Loss init args
    for k, v in cfg["loss_args"].items():
        overrides[f"model.init_args.loss_fn.init_args.{k}"] = v

    # Inject velocity weight from stress setting for BGSL+vel
    if method == "BGSL+vel":
        vw = stress_setting["velocity_weight"]
        overrides["model.init_args.loss_fn.init_args.velocity_weight"] = vw

    return overrides


def build_run_specs(
    scenarios: List[str],
    stress_names: List[str],
    methods: List[str],
    seeds: List[int],
) -> Tuple[Any, List[RunSpec]]:
    """Build the batch paths and all run specs for the requested grid.

    Returns
    -------
    batch_paths : BatchPaths
    run_specs   : list of RunSpec
    """
    batch_paths = build_batch_paths(
        experiment_name="empirical",
        experiment_type="synthetic_stress",
    )

    run_specs: List[RunSpec] = []

    for stress_name in stress_names:
        stress_setting = STRESS_SETTINGS[stress_name]

        for scenario in scenarios:
            for method in methods:
                method_cfg = _METHOD_LOSS[method]
                cond_name = f"{scenario}_{method_cfg['cond_name']}_stress{stress_name}"
                method_overrides = _build_method_overrides(method, stress_setting)

                for seed in seeds:
                    run_id = f"{cond_name}_seed{seed}"

                    # Merge overrides: method + scenario + stress
                    overrides = dict(method_overrides)
                    overrides["data.scenario"] = scenario
                    overrides["data.tau"] = stress_setting["tau"]
                    overrides["model.init_args.tau"] = stress_setting["tau"]
                    overrides["trainer.max_epochs"] = stress_setting["max_epochs"]
                    overrides["model.init_args.epochs"] = stress_setting["max_epochs"]

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
        scenarios=args.scenario,
        stress_names=args.stress,
        methods=args.methods,
        seeds=args.seeds,
    )

    print(f"Total planned jobs: {len(run_specs)}")
    print(f"  scenarios: {args.scenario}")
    print(f"  stress:    {args.stress}")
    print(f"  methods:   {args.methods}")
    print(f"  seeds:     {args.seeds}")
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
