from __future__ import annotations

from datetime import datetime

from studies.runner.architecture.paired_runner import (
    build_paired_conditions,
    build_run_specs,
)


def test_build_paired_conditions_is_exact_for_sepsis():
    conditions = build_paired_conditions("sepsis")

    assert len(conditions) == 6
    assert [cond["name"] for cond in conditions] == [
        "sepsis_gru_raw",
        "sepsis_gru_bgsl",
        "sepsis_tcn_raw",
        "sepsis_tcn_bgsl",
        "sepsis_transformer_raw",
        "sepsis_transformer_bgsl",
    ]

    assert conditions[0]["base_config"] == "src/bgsl/config/experiments/sepsis/gru.yaml"
    assert conditions[2]["base_config"] == "src/bgsl/config/experiments/sepsis/tcn.yaml"
    assert conditions[4]["base_config"] == "src/bgsl/config/experiments/sepsis/transformer.yaml"

    assert conditions[0]["overrides"] == {
        "model.init_args.loss_fn": "bgsl.core.common.losses.BCELoss",
    }
    assert conditions[1]["overrides"] == {}
    assert conditions[3]["overrides"] == {}
    assert conditions[5]["overrides"] == {}


def test_build_run_specs_emits_six_runs_for_one_seed(tmp_path, monkeypatch):
    fixed_now = datetime(2026, 6, 8, 12, 0, 0)

    def _fake_build_batch_paths(experiment_name: str, experiment_type: str):
        from studies.runner.commons.paths import build_batch_paths as _real_build_batch_paths

        return _real_build_batch_paths(
            experiment_name,
            experiment_type,
            outputs_root=tmp_path,
            now=fixed_now,
        )

    monkeypatch.setattr(
        "studies.runner.architecture.paired_runner.build_batch_paths",
        _fake_build_batch_paths,
    )

    batch_paths, run_specs, conditions = build_run_specs("sepsis", [42])

    assert batch_paths.root.parent.parent.parent == tmp_path.resolve()
    assert batch_paths.root.name.startswith("20260608-120000__")
    assert len(conditions) == 6
    assert len(run_specs) == 6
    assert run_specs[0].run_id == "sepsis_gru_raw_seed42"
    assert run_specs[0].base_config == "src/bgsl/config/experiments/sepsis/gru.yaml"
    assert run_specs[0].overrides == (("model.init_args.loss_fn", "bgsl.core.common.losses.BCELoss"),)
    assert run_specs[1].run_id == "sepsis_gru_bgsl_seed42"
    assert run_specs[1].overrides == ()
