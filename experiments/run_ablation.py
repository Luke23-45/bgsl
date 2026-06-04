"""
experiments/run_ablation.py
---------------------------
Orchestrate execution of ablation grids using LightningCLI.

Usage:
  python experiments/run_ablation.py --config configs/ablation_loss.yaml
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from studies.runner.commons.ablation_conditions import build_condition_overrides
from studies.runner.commons.paths import project_root


def _resolve_config_path(config_value: str, config_dir: Path) -> Path:
    path = Path(config_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = config_dir / path
    return candidate.resolve()


def _build_command(base_config: Path, seed: int, overrides: dict[str, object]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "bgsl.cli",
        "fit",
        "--config",
        str(base_config),
        "--seed_everything",
        str(seed),
    ]
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    abl = cfg.get("ablation")
    if not abl:
        raise ValueError("Config missing 'ablation' block.")

    base_cfg_value = cfg.get("_base_", "physionet_gru.yaml")
    base_cfg = _resolve_config_path(base_cfg_value, config_path.parent)
    seeds = abl.get("seeds", [42])
    conditions = abl.get("conditions", [])

    print(f"Running Ablation: {abl.get('name')} | Base: {base_cfg}")
    print(f"Total Runs: {len(conditions) * len(seeds)}")

    commands: list[list[str]] = []
    for cond in conditions:
        overrides = build_condition_overrides(cond)
        for seed in seeds:
            commands.append(_build_command(base_cfg, seed, overrides))

    if args.dry_run:
        for cmd in commands:
            print(" ".join(shlex.quote(part) for part in cmd))
        return

    print("Executing runs sequentially...")
    for cmd in commands:
        print(" ".join(shlex.quote(part) for part in cmd))
        subprocess.run(cmd, cwd=project_root(), check=True)


if __name__ == "__main__":
    main()
