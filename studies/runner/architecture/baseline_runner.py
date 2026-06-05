"""
studies/runner/architecture/baseline_runner.py
----------------------------------------------
Sweeps over the three primary sequence architectures: GRU, TCN, and Transformer.
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

    # The architecture sweep is intended to execute directly under the locked
    # comparison protocol, so keep the runner on the real sequential mode.
    configs = {
        "gru": "experiments/configs/physionet_gru.yaml",
        "tcn": "experiments/configs/physionet_tcn.yaml",
        "transformer": "experiments/configs/physionet_transformer.yaml",
    }
    seeds = [42, 43, 44]

    runs = []
    for arch_name, config_path in configs.items():
        for seed in seeds:
            run_id = f"arch_{arch_name}_seed{seed}"
            run_paths = batch_paths.run_paths(run_id)
            runs.append(build_run_spec(
                base_config=config_path,
                run_paths=run_paths,
                run_id=run_id,
                cond_name=f"arch_{arch_name}",
                seed=seed,
            ))

    execute_runs(runs, batch_paths, mode="local_sequential")


if __name__ == "__main__":
    main()
