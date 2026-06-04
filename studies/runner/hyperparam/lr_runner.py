"""
studies/runner/hyperparam/lr_runner.py
--------------------------------------
Grid search for optimal learning rates.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from studies.runner.commons.cli_builder import build_run_spec
from studies.runner.commons.executor import execute_runs
from studies.runner.commons.paths import (
    build_batch_paths,
    infer_experiment_id_from_file,
)


def main():
    experiment_name, experiment_type = infer_experiment_id_from_file(__file__)
    batch_paths = build_batch_paths(experiment_name, experiment_type)

    base_config = "experiments/configs/physionet_gru.yaml"
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
                overrides={"model.lr": lr},
            ))

    execute_runs(runs, batch_paths, mode="dry_run")


if __name__ == "__main__":
    main()
