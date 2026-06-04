"""
tests/test_paths.py
-------------------
Unit tests for studies.runner.commons.paths.
"""

from datetime import datetime
from pathlib import Path

import pytest

from studies.runner.commons.paths import (
    BatchPaths,
    RunPaths,
    build_batch_paths,
    infer_experiment_id_from_file,
    project_root,
)


def test_build_batch_paths_creates_structure(tmp_path):
    fixed = datetime(2026, 6, 4, 14, 30, 22)
    batch = build_batch_paths("ablation", "loss", outputs_root=tmp_path, now=fixed)

    assert isinstance(batch, BatchPaths)
    assert batch.experiment_name == "ablation"
    assert batch.experiment_type == "loss"
    assert batch.batch_id.startswith("20260604-143022__")
    assert batch.root.exists()
    assert batch.runs_dir.exists()
    assert batch.root.is_absolute()
    assert batch.root.parent.parent.parent == tmp_path.resolve()


def test_build_batch_paths_rejects_empty_ids(tmp_path):
    with pytest.raises(ValueError):
        build_batch_paths("", "loss", outputs_root=tmp_path)
    with pytest.raises(ValueError):
        build_batch_paths("ablation", "", outputs_root=tmp_path)


def test_batch_paths_run_paths_layout(tmp_path):
    batch = build_batch_paths("ablation", "loss", outputs_root=tmp_path)
    rp = batch.run_paths("baseline_bce_seed42")

    assert isinstance(rp, RunPaths)
    assert rp.run_id == "baseline_bce_seed42"
    assert rp.root == (batch.runs_dir / "baseline_bce_seed42").resolve()
    assert rp.override_yaml == rp.root / "_override.yaml"
    assert rp.logs_dir == rp.root / "logs"
    assert rp.checkpoints_dir == rp.root / "checkpoints"
    assert rp.metrics_dir == rp.root / "metrics"
    assert rp.artifacts_dir == rp.root / "artifacts"
    assert rp.console_log == rp.logs_dir / "console.log"
    assert rp.status_json == rp.root / "status.json"
    assert rp.final_metrics_json == rp.metrics_dir / "final_test_metrics.json"


def test_run_paths_create_dirs_idempotent(tmp_path):
    batch = build_batch_paths("ablation", "loss", outputs_root=tmp_path)
    rp = batch.run_paths("r1")
    rp.create_dirs()
    rp.create_dirs()  # second call must not raise
    for d in (rp.root, rp.logs_dir, rp.checkpoints_dir, rp.metrics_dir, rp.artifacts_dir):
        assert d.exists() and d.is_dir()


def test_batch_id_includes_uuid_suffix(tmp_path):
    fixed = datetime(2026, 6, 4, 14, 30, 22)
    b1 = build_batch_paths("ablation", "loss", outputs_root=tmp_path, now=fixed)
    b2 = build_batch_paths("ablation", "loss", outputs_root=tmp_path, now=fixed)
    assert b1.batch_id != b2.batch_id, "uuid suffix must disambiguate same-timestamp batches"


def test_infer_experiment_id_from_file():
    name, typ = infer_experiment_id_from_file(
        "/whatever/studies/runner/ablation/loss_runner.py"
    )
    assert name == "ablation"
    assert typ == "loss"

    name, typ = infer_experiment_id_from_file(
        "/whatever/studies/runner/hyperparam/lr_runner.py"
    )
    assert name == "hyperparam"
    assert typ == "lr"

    # Files without the _runner suffix still produce a non-empty stem.
    name, typ = infer_experiment_id_from_file("/x/y/architecture/baseline.py")
    assert name == "architecture"
    assert typ == "baseline"


def test_project_root_points_to_repo_root():
    root = project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "bgsl").exists()
