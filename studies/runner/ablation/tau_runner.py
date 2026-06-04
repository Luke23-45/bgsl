"""
studies/runner/ablation/tau_runner.py
-------------------------------------
Ablation study over the soft target temperature parameter (tau).
"""

import sys
from pathlib import Path

# Add project root to path so we can import commons.
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
    seeds = [42]  # Usually fewer seeds needed for a parameter sweep.

    # tau controls the "softness" of the sigmoid target.
    # We sweep from very sharp (0.5) to very soft (4.0).
    taus = [0.5, 1.0, 2.0, 4.0]

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
                overrides={"data.tau": tau},
            ))

    execute_runs(runs, batch_paths, mode="dry_run")


if __name__ == "__main__":
    main()
