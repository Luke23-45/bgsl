"""
studies/runner/hyperparam/lr_runner.py
--------------------------------------
Grid search for optimal learning rates.

Dataset: select via DATASET constant below.
Model: GRU (fixed — isolates the learning rate, not the architecture)
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
        description="Learning rate grid search runner."
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
        default="dry_run",
        help="Execution mode. Default: dry_run.",
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
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    seeds = [42]

    runs = []
    for lr in learning_rates:
        for seed in seeds:
            cond_name = f"lr_{lr}"
            run_id = f"{cond_name}_seed{seed}"
            run_paths = batch_paths.run_paths(run_id)
            runs.append(build_run_spec(
                base_config=base_config,
                run_paths=run_paths,
                run_id=run_id,
                cond_name=cond_name,
                seed=seed,
                # LightningCLI subclass_mode_model=True: lr lives under model.init_args
                overrides={"model.init_args.lr": lr},
            ))

    execute_runs(runs, batch_paths, mode=mode)


if __name__ == "__main__":
    main()
