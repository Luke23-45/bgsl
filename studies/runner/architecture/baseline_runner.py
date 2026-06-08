"""
studies/runner/architecture/baseline_runner.py
----------------------------------------------
Sweeps over the three primary sequence architectures (GRU, TCN, Transformer)
across both datasets (Sepsis and CMAPSS).

Each run uses the dataset-specific experiment config which already encodes the
correct DataModule, LightningModule, input dimensions, and default
hyperparameters for that architecture × dataset combination.

This runner is an architecture sweep, not the paired raw-vs-BGSL comparison.
Use `paired_runner.py` for the controlled raw BCE vs BGSL study.

Dataset: select via DATASETS constant below. Defaults to both.
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
_ALL_DATASETS: list = ["sepsis", "cmapss"]

_ARCH_CONFIGS: dict = {
    "sepsis": {
        "gru":         "src/bgsl/config/experiments/sepsis/gru.yaml",
        "tcn":         "src/bgsl/config/experiments/sepsis/tcn.yaml",
        "transformer": "src/bgsl/config/experiments/sepsis/transformer.yaml",
    },
    "cmapss": {
        "gru":         "src/bgsl/config/experiments/cmapss/gru.yaml",
        "tcn":         "src/bgsl/config/experiments/cmapss/tcn.yaml",
        "transformer": "src/bgsl/config/experiments/cmapss/transformer.yaml",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Architecture sweep runner (GRU, TCN, Transformer × datasets)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=_ALL_DATASETS,
        default=_ALL_DATASETS,
        help=(
            f"One or more datasets to sweep. Options: {_ALL_DATASETS}. "
            f"Default: all datasets."
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    datasets = args.datasets
    mode = args.mode
    seeds = args.seeds

    experiment_name, experiment_type = infer_experiment_id_from_file(__file__)
    batch_paths = build_batch_paths(experiment_name, experiment_type)

    runs = []
    for dataset in datasets:
        if dataset not in _ARCH_CONFIGS:
            raise ValueError(
                f"Unknown dataset {dataset!r}. "
                f"Valid options: {sorted(_ARCH_CONFIGS)}"
            )
        for arch_name, config_path in _ARCH_CONFIGS[dataset].items():
            for seed in seeds:
                # run_id encodes both dataset and architecture for clarity
                run_id = f"{dataset}_{arch_name}_seed{seed}"
                run_paths = batch_paths.run_paths(run_id)
                runs.append(build_run_spec(
                    base_config=config_path,
                    run_paths=run_paths,
                    run_id=run_id,
                    cond_name=f"{dataset}_{arch_name}",
                    seed=seed,
                ))

    execute_runs(runs, batch_paths, mode=mode)


if __name__ == "__main__":
    main()
