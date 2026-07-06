from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bgsl.empirical.data.synthetic import (
    SyntheticICUConfig,
    SyntheticICUDataModule,
    SyntheticICUGenerator,
)
from bgsl.empirical.studies import physionet_external, synthetic_stress
from studies.runner.commons.paths import build_batch_paths


def test_synthetic_generator_is_deterministic() -> None:
    config = SyntheticICUConfig(
        mode="signal",
        seed=101,
        n_patients=24,
        n_features=8,
        max_length=18,
        min_length=8,
        signal_strength=0.45,
        missing_rate=0.18,
    )

    out1 = SyntheticICUGenerator(config).generate()
    out2 = SyntheticICUGenerator(config).generate()

    for key in ("features", "masks", "soft_targets", "vel_targets", "hard_targets"):
        assert np.array_equal(out1[key], out2[key])

    assert np.array_equal(out1["onset_times"], out2["onset_times"])
    assert np.array_equal(out1["is_positive"], out2["is_positive"])
    assert np.array_equal(out1["seq_lengths"], out2["seq_lengths"])


def test_synthetic_cache_refreshes_when_config_changes(tmp_path: Path) -> None:
    data_root = tmp_path / "synthetic"

    first = SyntheticICUDataModule(
        data_dir=str(data_root),
        scenario="signal",
        n_patients=12,
        n_features=8,
        signal_strength=0.45,
        batch_size=4,
    )
    first_dir = first._ensure_data()
    with open(first_dir / "manifest.json", "r", encoding="utf-8") as f:
        first_manifest = json.load(f)

    second = SyntheticICUDataModule(
        data_dir=str(data_root),
        scenario="signal",
        n_patients=12,
        n_features=8,
        signal_strength=0.70,
        batch_size=4,
    )
    second_dir = second._ensure_data()
    with open(second_dir / "manifest.json", "r", encoding="utf-8") as f:
        second_manifest = json.load(f)

    assert first_dir == second_dir
    assert first_manifest["generator_config"]["signal_strength"] == 0.45
    assert second_manifest["generator_config"]["signal_strength"] == 0.70
    assert first_manifest["config_fingerprint"] != second_manifest["config_fingerprint"]


def test_synthetic_stress_grid_uses_frozen_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        synthetic_stress,
        "build_batch_paths",
        lambda experiment_name, experiment_type: build_batch_paths(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            outputs_root=tmp_path,
        ),
    )

    batch_paths, run_specs = synthetic_stress.build_run_specs(
        scenarios=["signal"],
        stress_names=["Default", "HighVel", "LongTrain"],
        methods=["BGSL+vel"],
        seeds=[101],
    )

    assert batch_paths.root.parent.parent == tmp_path / "empirical"
    assert len(run_specs) == 3

    weights = {}
    epochs = {}
    taus = {}
    for spec in run_specs:
        stress = spec.run_id.split("_stress", 1)[1].rsplit("_seed", 1)[0]
        weights[stress] = _override_value(spec, "model.init_args.loss_fn.init_args.velocity_weight")
        epochs[stress] = _override_value(spec, "trainer.max_epochs")
        taus[stress] = _override_value(spec, "data.tau")

    assert weights == {"Default": "0.05", "HighVel": "0.5", "LongTrain": "0.05"}
    assert epochs == {"Default": "16", "HighVel": "16", "LongTrain": "32"}
    assert taus == {"Default": "2.5", "HighVel": "1.5", "LongTrain": "2.5"}


def test_physionet_external_uses_frozen_bgsl_velocity_weight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        physionet_external,
        "build_batch_paths",
        lambda experiment_name, experiment_type: build_batch_paths(
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            outputs_root=tmp_path,
        ),
    )

    batch_paths, run_specs = physionet_external.build_run_specs(
        methods=["BCE", "BGSL-state", "BGSL+vel"],
        seeds=[42],
    )

    assert batch_paths.root.parent.parent == tmp_path / "empirical"
    assert len(run_specs) == 3
    assert _override_value(
        next(spec for spec in run_specs if spec.cond_name == "BGSL_vel"),
        "model.init_args.loss_fn.init_args.velocity_weight",
    ) == "0.05"


def _override_value(spec, key: str) -> str:
    for override_key, override_value in spec.overrides:
        if override_key == key:
            return override_value
    raise AssertionError(f"Missing override: {key}")
