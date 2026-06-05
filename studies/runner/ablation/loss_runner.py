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
from studies.runner.commons.ablation_conditions import build_condition_overrides
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
        {"name": "baseline_bce", "loss": {"class_path": "bgsl.core.losses.BCELoss"}},
        {"name": "baseline_weighted_bce", "loss": {"class_path": "bgsl.core.losses.WeightedBCELoss", "init_args": {"pos_weight": 10.0}}},
        {"name": "baseline_focal", "loss": {"class_path": "bgsl.core.losses.FocalLoss", "init_args": {"gamma": 2.0}}},
        {"name": "baseline_tls", "loss": {"class_path": "bgsl.core.losses.TLSLoss", "init_args": {"alpha": 6.0}}},
        {"name": "baseline_smoothness", "loss": {"class_path": "bgsl.core.losses.SmoothnessLoss", "init_args": {"smoothness_weight": 0.1}}},
        {"name": "baseline_tv", "loss": {"class_path": "bgsl.core.losses.TotalVariationLoss", "init_args": {"tv_weight": 0.1}}},

        # BGSL family
        {"name": "bgsl_state_only", "loss": {"class_path": "bgsl.core.losses.BGSLLoss", "init_args": {"state_loss": "bce", "derivative_space": "probability", "velocity_weight": 0.0, "acceleration_weight": 0.0}}},
        {"name": "bgsl_velocity_only", "loss": {"class_path": "bgsl.core.losses.BGSLLoss", "init_args": {"state_loss": "bce", "derivative_space": "probability", "velocity_weight": 0.1, "acceleration_weight": 0.0}}},
        {"name": "bgsl_accel_only", "loss": {"class_path": "bgsl.core.losses.BGSLLoss", "init_args": {"state_loss": "bce", "derivative_space": "probability", "velocity_weight": 0.0, "acceleration_weight": 0.05}}},
        {"name": "bgsl_full", "loss": {"class_path": "bgsl.core.losses.BGSLLoss", "init_args": {"state_loss": "bce", "derivative_space": "probability", "velocity_weight": 0.1, "acceleration_weight": 0.05}}},
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
                overrides=build_condition_overrides(cond),
            ))

    execute_runs(runs, batch_paths, mode="local_sequential")


if __name__ == "__main__":
    main()
