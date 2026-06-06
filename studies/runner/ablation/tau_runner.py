"""
studies/runner/ablation/tau_runner.py
-------------------------------------
Ablation study over the soft target temperature parameter (tau).

tau controls the "softness" of the sigmoid target in SoftOnsetTarget.
This runner is Sepsis-only because CMAPSS uses horizon_cycles (not tau)
as the target sharpness parameter — a separate horizon_runner handles that.

Dataset: Sepsis only (tau is a Sepsis-domain parameter).
Model: GRU (fixed — isolates the tau parameter)
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from studies.runner.commons.cli_builder import build_run_spec
from studies.runner.commons.executor import execute_runs
from studies.runner.commons.paths import (
    build_batch_paths,
    infer_experiment_id_from_file,
)

# ---------------------------------------------------------------------------
# Dataset selector
# ---------------------------------------------------------------------------
# tau is a Sepsis-specific parameter (hours before onset). CMAPSS equivalently
# uses horizon_cycles. Both map to the datamodule's tau argument.
# If you extend this to CMAPSS, add "cmapss" here with the appropriate config.

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
        description="Tau (soft target temperature) ablation runner."
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    dataset = args.dataset
    mode = args.mode

    experiment_name, experiment_type = infer_experiment_id_from_file(__file__)
    batch_paths = build_batch_paths(
        experiment_name=f"{experiment_name}/{dataset}",
        experiment_type=experiment_type,
    )

    base_config = _get_base_config(dataset)
    # Usually fewer seeds needed for a parameter sweep.
    seeds = [42]

    # tau controls the "softness" of the sigmoid target.
    # Keep the sweep aligned with the locked config matrix.
    taus = [0.5, 1.0, 2.0, 4.0, 6.0]

    runs = []
    for tau in taus:
        for seed in seeds:
            cond_name = f"tau_{tau}"
            run_id = f"{cond_name}_seed{seed}"
            run_paths = batch_paths.run_paths(run_id)
            runs.append(build_run_spec(
                base_config=base_config,
                run_paths=run_paths,
                run_id=run_id,
                cond_name=cond_name,
                seed=seed,
                # data.init_args.tau — maps to PhysioNetDataModule.tau
                overrides={"data.init_args.tau": tau},
            ))

    execute_runs(runs, batch_paths, mode=mode)


if __name__ == "__main__":
    main()
