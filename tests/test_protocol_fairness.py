from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from bgsl.data.sepsis.datamodule import PhysioNetDataModule
from bgsl.train.sepsis.module import SepsisLightningModule as BGSLLightningModule
from bgsl.core.sepsis.metrics import PatientPrediction
from studies.runner.commons.ablation_conditions import build_condition_overrides


class _DummyModel(torch.nn.Module):
    def forward(self, x, mask=None, seq_lens=None):
        return torch.zeros(x.shape[0], x.shape[1], device=x.device)


class _DummyLoss(torch.nn.Module):
    def forward(self, logits, targets, mask, **kwargs):
        zero = torch.tensor(0.0, device=logits.device)
        return {"loss": zero, "state": zero, "velocity": zero, "acceleration": zero}


class _NullMetrics(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def compute(self):
        return {}

    def reset(self):
        return None


class _DummyDataset(list):
    def __init__(self, n: int = 4) -> None:
        super().__init__(range(n))
        self.episodes = [{"is_sepsis": False} for _ in range(n)]


@pytest.mark.parametrize(
    "condition, expected",
    [
        (
            {"name": "bce", "loss": {"class_path": "bgsl.core.common.losses.BCELoss"}},
            {"model.loss_fn": "bgsl.core.common.losses.BCELoss"},
        ),
        (
            {
                "name": "weighted_bce",
                "loss": {
                    "class_path": "bgsl.core.common.losses.WeightedBCELoss",
                    "init_args": {"pos_weight": 10.0},
                },
            },
            {
                "model.loss_fn": "bgsl.core.common.losses.WeightedBCELoss",
                "model.loss_fn.init_args.pos_weight": 10.0,
            },
        ),
        (
            {
                "name": "focal",
                "loss": {
                    "class_path": "bgsl.core.common.losses.FocalLoss",
                    "init_args": {"gamma": 2.0},
                },
            },
            {
                "model.loss_fn": "bgsl.core.common.losses.FocalLoss",
                "model.loss_fn.init_args.gamma": 2.0,
            },
        ),
        (
            {
                "name": "tls",
                "loss": {
                    "class_path": "bgsl.core.common.losses.TLSLoss",
                    "init_args": {"alpha": 6.0},
                },
            },
            {
                "model.loss_fn": "bgsl.core.common.losses.TLSLoss",
                "model.loss_fn.init_args.alpha": 6.0,
            },
        ),
        (
            {
                "name": "smoothness",
                "loss": {
                    "class_path": "bgsl.core.common.losses.SmoothnessLoss",
                    "init_args": {"smoothness_weight": 0.1},
                },
            },
            {
                "model.loss_fn": "bgsl.core.common.losses.SmoothnessLoss",
                "model.loss_fn.init_args.smoothness_weight": 0.1,
            },
        ),
        (
            {
                "name": "tv",
                "loss": {
                    "class_path": "bgsl.core.common.losses.TotalVariationLoss",
                    "init_args": {"tv_weight": 0.1},
                },
            },
            {
                "model.loss_fn": "bgsl.core.common.losses.TotalVariationLoss",
                "model.loss_fn.init_args.tv_weight": 0.1,
            },
        ),
        (
            {
                "name": "bgsl_full",
                "loss": {
                    "class_path": "bgsl.core.common.losses.BGSLLoss",
                    "init_args": {
                        "state_loss": "bce",
                        "derivative_space": "probability",
                        "velocity_weight": 0.1,
                        "acceleration_weight": 0.05,
                    },
                },
            },
            {
                "model.loss_fn": "bgsl.core.common.losses.BGSLLoss",
                "model.loss_fn.init_args.state_loss": "bce",
                "model.loss_fn.init_args.derivative_space": "probability",
                "model.loss_fn.init_args.velocity_weight": 0.1,
                "model.loss_fn.init_args.acceleration_weight": 0.05,
            },
        ),
        (
            {
                "name": "bgsl_state_only",
                "loss": {
                    "class_path": "bgsl.core.common.losses.BGSLLoss",
                    "init_args": {
                        "state_loss": "bce",
                        "derivative_space": "probability",
                        "velocity_weight": 0.0,
                        "acceleration_weight": 0.0,
                    },
                },
            },
            {
                "model.loss_fn": "bgsl.core.common.losses.BGSLLoss",
                "model.loss_fn.init_args.state_loss": "bce",
                "model.loss_fn.init_args.derivative_space": "probability",
                "model.loss_fn.init_args.velocity_weight": 0.0,
                "model.loss_fn.init_args.acceleration_weight": 0.0,
            },
        ),
        (
            {
                "name": "bgsl_velocity_only",
                "loss": {
                    "class_path": "bgsl.core.common.losses.BGSLLoss",
                    "init_args": {
                        "state_loss": "bce",
                        "derivative_space": "probability",
                        "velocity_weight": 0.1,
                        "acceleration_weight": 0.0,
                    },
                },
            },
            {
                "model.loss_fn": "bgsl.core.common.losses.BGSLLoss",
                "model.loss_fn.init_args.state_loss": "bce",
                "model.loss_fn.init_args.derivative_space": "probability",
                "model.loss_fn.init_args.velocity_weight": 0.1,
                "model.loss_fn.init_args.acceleration_weight": 0.0,
            },
        ),
        (
            {
                "name": "bgsl_accel_only",
                "loss": {
                    "class_path": "bgsl.core.common.losses.BGSLLoss",
                    "init_args": {
                        "state_loss": "bce",
                        "derivative_space": "probability",
                        "velocity_weight": 0.0,
                        "acceleration_weight": 0.05,
                    },
                },
            },
            {
                "model.loss_fn": "bgsl.core.common.losses.BGSLLoss",
                "model.loss_fn.init_args.state_loss": "bce",
                "model.loss_fn.init_args.derivative_space": "probability",
                "model.loss_fn.init_args.velocity_weight": 0.0,
                "model.loss_fn.init_args.acceleration_weight": 0.05,
            },
        ),
    ],
)
def test_build_condition_overrides_maps_exactly(condition, expected):
    assert build_condition_overrides(condition) == expected


def test_build_condition_overrides_rejects_unknown_keys():
    with pytest.raises(ValueError):
        build_condition_overrides(
            {
                "name": "broken",
                "loss": {"class_path": "bgsl.core.common.losses.BCELoss", "foo": 1},
            }
        )


def _lookup(mapping: dict, dotted: str):
    cur = mapping
    for part in dotted.split("."):
        cur = cur[part]
    return cur


@pytest.mark.parametrize(
    "config_path, expected_model",
    [
        (
            "experiments/configs/base.yaml",
            {"class_path": "bgsl.models.gru.GRUPredictor", "init_args.hidden_dim": 64, "init_args.num_layers": 1, "init_args.dropout": 0.35},
        ),
        (
            "experiments/configs/physionet_gru.yaml",
            {"class_path": "bgsl.models.gru.GRUPredictor", "init_args.hidden_dim": 64, "init_args.num_layers": 1, "init_args.dropout": 0.35},
        ),
        (
            "experiments/configs/physionet_tcn.yaml",
            {"class_path": "bgsl.models.tcn.TCNPredictor", "init_args.n_filters": 64, "init_args.n_layers": 4, "init_args.dropout": 0.30},
        ),
        (
            "experiments/configs/physionet_transformer.yaml",
            {"class_path": "bgsl.models.transformer.TransformerPredictor", "init_args.d_model": 64, "init_args.n_layers": 2, "init_args.dim_feedforward": 128, "init_args.dropout": 0.20},
        ),
    ],
)
def test_main_configs_use_shared_budget_and_plain_bce(config_path, expected_model):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    shared_paths = [
        "trainer.accelerator",
        "trainer.devices",
        "trainer.enable_progress_bar",
        "trainer.max_epochs",
        "trainer.gradient_clip_val",
        "data.data_dir",
        "data.horizon_hours",
        "data.tau",
        "data.min_length",
        "data.batch_size",
        "data.num_workers",
        "data.pin_memory",
        "data.enable_normalizer",
        "data.normalizer_epsilon",
        "data.normalizer_safety_margin",
        "model.horizon_hours",
        "model.lr",
        "model.weight_decay",
        "model.epochs",
        "model.lr_scheduler",
        "model.loss_fn.class_path",
        "model.loss_fn.init_args",
    ]
    shared_values = {
        path: _lookup(cfg, path) for path in shared_paths
    }

    assert shared_values["trainer.accelerator"] == "auto"
    assert shared_values["trainer.devices"] == 1
    assert shared_values["trainer.enable_progress_bar"] is True
    assert shared_values["trainer.max_epochs"] == 30
    assert shared_values["trainer.gradient_clip_val"] == 1.0
    assert shared_values["data.data_dir"] == "sepsis_clinical_40/physionet_2019_processed"
    assert shared_values["data.horizon_hours"] == 6
    assert shared_values["data.tau"] == 2.0
    assert shared_values["data.min_length"] == 8
    assert shared_values["data.batch_size"] == 16
    assert shared_values["data.num_workers"] == 2
    assert shared_values["data.pin_memory"] is False
    assert shared_values["data.enable_normalizer"] is True
    assert shared_values["model.horizon_hours"] == 6
    assert shared_values["model.lr"] == 5.0e-4
    assert shared_values["model.weight_decay"] == 1.0e-4
    assert shared_values["model.epochs"] == 30
    assert shared_values["model.lr_scheduler"] == "cosine"
    assert shared_values["model.loss_fn.class_path"] == "bgsl.core.common.losses.BCELoss"
    assert shared_values["model.loss_fn.init_args"] == {}

    callbacks = cfg["trainer"]["callbacks"]
    ckpt = next(cb for cb in callbacks if cb["class_path"].endswith("ModelCheckpoint"))
    early = next(cb for cb in callbacks if cb["class_path"].endswith("EarlyStopping"))

    assert ckpt["init_args"]["monitor"] == "val_loss"
    assert ckpt["init_args"]["mode"] == "min"
    assert "val_loss" in ckpt["init_args"]["filename"]
    assert early["init_args"]["monitor"] == "val_loss"
    assert early["init_args"]["mode"] == "min"
    assert early["init_args"]["patience"] == 6
    assert "pos_weight_sampler" not in cfg.get("data", {})
    assert _lookup(cfg, "model.model.class_path") == expected_model["class_path"]

    for key, expected in expected_model.items():
        if key == "class_path":
            continue
        assert _lookup(cfg, f"model.model.{key}") == expected


def test_datamodule_defaults_to_plain_sampling_and_can_opt_in():
    plain = PhysioNetDataModule(batch_size=2, num_workers=0)
    plain.train_ds = _DummyDataset()
    plain_loader = plain.train_dataloader()
    assert plain_loader.sampler.__class__.__name__ == "RandomSampler"

    weighted = PhysioNetDataModule(batch_size=2, num_workers=0, pos_weight_sampler=5.0)
    weighted.train_ds = _DummyDataset()
    weighted_loader = weighted.train_dataloader()
    assert weighted_loader.sampler.__class__.__name__ == "WeightedRandomSampler"


def test_validation_threshold_is_selected_on_validation_and_frozen_for_test():
    module = BGSLLightningModule(
        model=_DummyModel(),
        loss_fn=_DummyLoss(),
        validation_threshold_target=0.80,
    )
    module.log_dict = lambda *args, **kwargs: None
    module.val_metrics = _NullMetrics()

    module.validation_threshold_metrics.add(
        PatientPrediction(
            patient_id="p1",
            probs=torch.tensor([0.91, 0.91, 0.91, 0.91, 0.91]).numpy(),
            hard_labels=torch.tensor([1, 1, 1, 1, 1]).numpy(),
            soft_targets=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).numpy(),
            is_sepsis=True,
            onset_hour=4,
            horizon=6,
            seq_len=5,
        )
    )
    module.validation_threshold_metrics.add(
        PatientPrediction(
            patient_id="p2",
            probs=torch.tensor([0.09, 0.09, 0.09, 0.09, 0.09]).numpy(),
            hard_labels=torch.tensor([0, 0, 0, 0, 0]).numpy(),
            soft_targets=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).numpy(),
            is_sepsis=False,
            onset_hour=-1,
            horizon=6,
            seq_len=5,
        )
    )

    module.on_validation_epoch_end()

    assert module.selected_threshold == pytest.approx(0.91)
    assert module.trajectory_metrics.threshold == pytest.approx(0.91)


def test_loss_ablation_uses_the_locked_three_seed_matrix():
    with open("experiments/configs/ablation_loss.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["ablation"]["seeds"] == [42, 43, 44]
