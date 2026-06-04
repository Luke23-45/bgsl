"""
studies/runner/ablation/core_hypothesis_runner.py
-------------------------------------------------
The decisive face-off for the BGSL paper.

This runner tests the central thesis:
"Does Biological Gradient Supervised Learning actually outperform a naive
smoothness penalty, or is BGSL just a complicated way to achieve the same thing?"

Conditions:
1. Naive BCE (Baseline)
2. BCE + Smoothness (L2 penalty on dp)
3. Full BGSL (Velocity + Acceleration targets on Soft Onset)

If BGSL cannot beat Smoothness on the 4-metric panel (AUPRC, ASF, Lead Time, POMS),
the core hypothesis is falsified.

Model: GRU
Dataset: PhysioNet 2019
Seeds: 3 (statistical significance baseline; bump for final paper)
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
    seeds = [42, 43, 44]

    conditions = [
        {
            "name": "01_BCE_Baseline",
            "overrides": {
                "model.loss_fn": "bgsl.core.losses.WeightedBCELoss",
                "model.loss_fn.init_args.pos_weight": 1.0,
            },
        },
        {
            "name": "02_BCE_Smoothness",
            "overrides": {
                "model.loss_fn": "bgsl.core.losses.SmoothnessLoss",
                "model.loss_fn.init_args.smoothness_weight": 0.1,
            },
        },
        {
            "name": "03_Full_BGSL",
            "overrides": {
                "model.loss_fn": "bgsl.core.losses.BGSLLoss",
                "model.loss_fn.init_args.velocity_weight": 0.10,
                "model.loss_fn.init_args.acceleration_weight": 0.05,
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
                overrides=cond["overrides"],
            ))

    # Change mode to "slurm" or "local_sequential" to actually run.
    execute_runs(runs, batch_paths, mode="local_sequential")


if __name__ == "__main__":
    main()
