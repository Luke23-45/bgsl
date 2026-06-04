"""
tests/test_override_writer.py
-----------------------------
Unit tests for studies.runner.commons.override_writer.
"""

from pathlib import Path

import yaml

from studies.runner.commons.override_writer import write_run_override
from studies.runner.commons.paths import build_batch_paths


def _load_override(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_write_override_yaml_creates_file(tmp_path):
    batch = build_batch_paths("ablation", "loss", outputs_root=tmp_path)
    rp = batch.run_paths("baseline_bce_seed42")
    out = write_run_override(rp)

    assert out == rp.override_yaml
    assert out.exists()
    assert out.is_file()


def test_override_pins_default_root_dir_to_absolute_run_root(tmp_path):
    batch = build_batch_paths("ablation", "loss", outputs_root=tmp_path)
    rp = batch.run_paths("r1")
    write_run_override(rp)
    payload = _load_override(rp.override_yaml)

    drd = Path(payload["trainer"]["default_root_dir"])
    assert drd.is_absolute()
    assert drd == rp.root


def test_override_callbacks_include_modelcheckpoint_to_run_checkpoints(tmp_path):
    batch = build_batch_paths("ablation", "loss", outputs_root=tmp_path)
    rp = batch.run_paths("r1")
    write_run_override(rp)
    payload = _load_override(rp.override_yaml)

    callbacks = payload["trainer"]["callbacks"]
    assert isinstance(callbacks, list)
    class_paths = [cb["class_path"] for cb in callbacks]
    assert "pytorch_lightning.callbacks.ModelCheckpoint" in class_paths
    assert "pytorch_lightning.callbacks.EarlyStopping" in class_paths
    assert "pytorch_lightning.callbacks.LearningRateMonitor" in class_paths

    ckpt_cb = next(cb for cb in callbacks if cb["class_path"].endswith("ModelCheckpoint"))
    dirpath = Path(ckpt_cb["init_args"]["dirpath"])
    assert dirpath.is_absolute()
    assert dirpath == rp.checkpoints_dir
    assert ckpt_cb["init_args"]["monitor"] == "val_auprc"
    assert ckpt_cb["init_args"]["mode"] == "max"
    assert ckpt_cb["init_args"]["save_last"] is True


def test_override_logger_is_csvlogger_under_logs(tmp_path):
    batch = build_batch_paths("ablation", "loss", outputs_root=tmp_path)
    rp = batch.run_paths("r1")
    write_run_override(rp)
    payload = _load_override(rp.override_yaml)

    logger_cfg = payload["trainer"]["logger"]
    assert logger_cfg["class_path"] == "pytorch_lightning.loggers.CSVLogger"
    save_dir = Path(logger_cfg["init_args"]["save_dir"])
    assert save_dir.is_absolute()
    assert save_dir == rp.logs_dir


def test_override_paths_use_forward_slashes(tmp_path):
    batch = build_batch_paths("ablation", "loss", outputs_root=tmp_path)
    rp = batch.run_paths("r1")
    write_run_override(rp)

    raw_text = rp.override_yaml.read_text(encoding="utf-8")
    # Even on Windows the YAML must use forward slashes to avoid escape issues.
    assert "\\" not in raw_text, f"YAML contains backslashes: {raw_text!r}"
