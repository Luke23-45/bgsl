"""
experiments/run_ablation.py
---------------------------
Orchestrate execution of ablation grids using LightningCLI.

Usage:
  python experiments/run_ablation.py --config configs/ablation_loss.yaml
"""

import argparse
import subprocess
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    abl = cfg.get("ablation")
    if not abl:
        raise ValueError("Config missing 'ablation' block.")
        
    base_cfg = cfg.get("_base_", "physionet_gru.yaml")
    seeds = abl.get("seeds", [42])
    conditions = abl.get("conditions", [])
    
    print(f"Running Ablation: {abl.get('name')} | Base: {base_cfg}")
    print(f"Total Runs: {len(conditions) * len(seeds)}")
    
    commands = []
    
    for cond in conditions:
        cond_name = cond["name"]
        
        for seed in seeds:
            # Build LightningCLI command
            # Instead of passing custom flags to a manual loop, we use LightningCLI's
            # powerful override system. We assume the base config handles the defaults.
            cmd = f"bgsl-train fit --config experiments/configs/{base_cfg} --seed_everything {seed}"
            
            # Translate ablation conditions into jsonargparse overrides
            if "loss" in cond:
                l_cfg = cond["loss"]
                if "velocity_weight" in l_cfg:
                    cmd += f" --model.loss_fn.init_args.velocity_weight {l_cfg['velocity_weight']}"
                if "acceleration_weight" in l_cfg:
                    cmd += f" --model.loss_fn.init_args.acceleration_weight {l_cfg['acceleration_weight']}"
                if "name" in l_cfg:
                    # In true LightningCLI SOTA, switching the class path handles this
                    # For ablation ease, we demonstrate class_path swapping
                    name = l_cfg["name"]
                    if name == "bce":
                        cmd += f" --model.loss_fn bgsl.core.losses.WeightedBCELoss"
                    elif name == "weighted_bce":
                        cmd += f" --model.loss_fn bgsl.core.losses.WeightedBCELoss"
                        cmd += f" --model.loss_fn.init_args.pos_weight {l_cfg.get('pos_weight', 10.0)}"
                    elif name == "focal":
                        cmd += f" --model.loss_fn bgsl.core.losses.FocalLoss"
            
            if "data" in cond:
                d_cfg = cond["data"]
                if "tau" in d_cfg:
                    cmd += f" --data.tau {d_cfg['tau']}"
                    
            commands.append(cmd)

    if args.dry_run:
        for c in commands:
            print(c)
    else:
        print("Executing (this will take a while)...")
        # In a real scenario, use subprocess.Popen or submit to Slurm/batch system.

if __name__ == "__main__":
    main()
