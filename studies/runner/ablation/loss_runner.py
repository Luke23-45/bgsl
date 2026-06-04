"""
studies/runner/ablation/loss_runner.py
--------------------------------------
Ablation study over BGSL loss components and direct baselines.
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
    seeds = [42, 43, 44]

    conditions = [
        # Baselines
        {"name": "baseline_bce", "overrides": {"model.init_args.loss_fn.class_path": "bgsl.core.losses.WeightedBCELoss", "model.init_args.loss_fn.init_args.pos_weight": 1.0}},
        {"name": "baseline_weighted_bce", "overrides": {"model.init_args.loss_fn.class_path": "bgsl.core.losses.WeightedBCELoss", "model.init_args.loss_fn.init_args.pos_weight": 5.0}},
        {"name": "baseline_focal", "overrides": {"model.init_args.loss_fn.class_path": "bgsl.core.losses.FocalLoss", "model.init_args.loss_fn.init_args.gamma": 2.0}},

        # BGSL Velocity Ablation (fixing accel=0)
        {"name": "bgsl_v0.01_a0.0", "overrides": {"model.init_args.loss_fn.init_args.velocity_weight": 0.01, "model.init_args.loss_fn.init_args.acceleration_weight": 0.0}},
        {"name": "bgsl_v0.10_a0.0", "overrides": {"model.init_args.loss_fn.init_args.velocity_weight": 0.10, "model.init_args.loss_fn.init_args.acceleration_weight": 0.0}},
        {"name": "bgsl_v0.50_a0.0", "overrides": {"model.init_args.loss_fn.init_args.velocity_weight": 0.50, "model.init_args.loss_fn.init_args.acceleration_weight": 0.0}},

        # BGSL Full (Velocity + Acceleration)
        {"name": "bgsl_v0.10_a0.05", "overrides": {"model.init_args.loss_fn.init_args.velocity_weight": 0.10, "model.init_args.loss_fn.init_args.acceleration_weight": 0.05}},
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
    execute_runs(runs, batch_paths, mode="dry_run")


if __name__ == "__main__":
    main()
