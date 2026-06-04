"""
studies/runner/commons/override_writer.py
-----------------------------------------
Writes a per-run override YAML that pins LightningCLI's output paths to a
specific RunPaths location.

Usage::

    override_path = write_run_override(run_paths)
    cmd = f"bgsl-train fit --config {base_yaml} --config {override_path}"

Important caveats
-----------------
* LightningCLI / jsonargparse merges configs by key, but for **list** values
  (notably ``trainer.callbacks``) the later config **replaces** the earlier
  one rather than appending to it. The override therefore emits the *full*
  callback list (ModelCheckpoint + EarlyStopping + LearningRateMonitor), so
  the resulting trainer has the same callback set as the base configs.

* All paths are written using ``Path.as_posix()`` to avoid YAML-escaping
  issues with Windows backslashes.

* The override filename is ``_override.yaml`` so it does not collide with
  LightningCLI's own ``save_config`` output (which defaults to
  ``config.yaml`` next to the active logger's output directory).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from studies.runner.commons.paths import RunPaths


def _build_override_dict(run_paths: RunPaths) -> Dict[str, Any]:
    """Construct the override config dictionary for one run."""
    return {
        "trainer": {
            "default_root_dir": run_paths.root.as_posix(),
            "callbacks": [
                {
                    "class_path": "bgsl.utils.progress.CloudProgressBar",
                },
                {
                    "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                    "init_args": {
                        "dirpath": run_paths.checkpoints_dir.as_posix(),
                        "filename": "{epoch:02d}-{val_loss:.4f}",
                        "monitor": "val_loss",
                        "mode": "min",
                        "save_top_k": 1,
                        "save_last": True,
                    },
                },
                {
                    "class_path": "pytorch_lightning.callbacks.EarlyStopping",
                    "init_args": {
                        "monitor": "val_loss",
                        "patience": 10,
                        "mode": "min",
                    },
                },
                {
                    "class_path": "pytorch_lightning.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "step",
                    },
                },
            ],
            "logger": {
                "class_path": "pytorch_lightning.loggers.CSVLogger",
                "init_args": {
                    "save_dir": run_paths.logs_dir.as_posix(),
                    "name": "",
                    "version": "",
                },
            },
        }
    }


def write_run_override(run_paths: RunPaths) -> Path:
    """
    Write the per-run override YAML and return its path.

    Ensures the run directory tree exists, then dumps the override mapping
    to ``run_paths.override_yaml``.
    """
    run_paths.create_dirs()
    payload = _build_override_dict(run_paths)
    with open(run_paths.override_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return run_paths.override_yaml
