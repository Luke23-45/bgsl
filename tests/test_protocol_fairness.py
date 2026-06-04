from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from bgsl.data.datamodule import PhysioNetDataModule
from bgsl.models.lightning import BGSLLightningModule
from bgsl.core.metrics import PatientPrediction
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
            {"name": "bce", "loss": {"class_path": "bgsl.core.losses.BCELoss"}},
            {"model.loss_fn": "bgsl.core.losses.BCELoss"},
        ),
        (
            {
                "name": "weighted_bce",
                "loss": {
                    "class_path": "bgsl.core.losses.WeightedBCELoss",
                    "init_args": {"pos_weight": 10.0},
                },
            },
            {
                "model.loss_fn": "bgsl.core.losses.WeightedBCELoss",
                "model.loss_fn.init_args.pos_weight": 10.0,
            },
        ),
        (
            {
                "name": "focal",
                "loss": {
                    "class_path": "bgsl.core.losses.FocalLoss",
                    "init_args": {"gamma": 2.0},
                },
            },
            {
                "model.loss_fn": "bgsl.core.losses.FocalLoss",
                "model.loss_fn.init_args.gamma": 2.0,
            },
        ),
        (
            {
                "name": "tls",
                "loss": {
                    "class_path": "bgsl.core.losses.TLSLoss",
                    "init_args": {"alpha": 6.0},
                },
            },
            {
                "model.loss_fn": "bgsl.core.losses.TLSLoss",
                "model.loss_fn.init_args.alpha": 6.0,
            },
        ),
        (
            {
                "name": "smoothness",
                "loss": {
                    "class_path": "bgsl.core.losses.SmoothnessLoss",
                    "init_args": {"smoothness_weight": 0.1},
                },
            },
            {
                "model.loss_fn": "bgsl.core.losses.SmoothnessLoss",
                "model.loss_fn.init_args.smoothness_weight": 0.1,
            },
        ),
        (
            {
                "name": "tv",
                "loss": {
                    "class_path": "bgsl.core.losses.TotalVariationLoss",
                    "init_args": {"tv_weight": 0.1},
                },
            },
            {
                "model.loss_fn": "bgsl.core.losses.TotalVariationLoss",
                "model.loss_fn.init_args.tv_weight": 0.1,
            },
        ),
        (
            {
                "name": "bgsl_full",
                "loss": {
                    "class_path": "bgsl.core.losses.BGSLLoss",
                    "init_args": {
                        "state_loss": "bce",
                        "derivative_space": "probability",
                        "velocity_weight": 0.1,
                        "acceleration_weight": 0.05,
                    },
                },
            },
            {
                "model.loss_fn": "bgsl.core.losses.BGSLLoss",
                "model.loss_fn.init_args.state_loss": "bce",
                "model.loss_fn.init_args.derivative_space": "probability",
                "model.loss_fn.init_args.velocity_weight": 0.1,
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
                "loss": {"class_path": "bgsl.core.losses.BCELoss", "foo": 1},
            }
        )


@pytest.mark.parametrize(
    "config_path",
    [
        "experiments/configs/base.yaml",
        "experiments/configs/physionet_gru.yaml",
        "experiments/configs/physionet_tcn.yaml",
        "experiments/configs/physionet_transformer.yaml",
    ],
)
def test_main_configs_use_val_loss_and_no_weighted_sampler(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    callbacks = cfg["trainer"]["callbacks"]
    ckpt = next(cb for cb in callbacks if cb["class_path"].endswith("ModelCheckpoint"))
    early = next(cb for cb in callbacks if cb["class_path"].endswith("EarlyStopping"))

    assert ckpt["init_args"]["monitor"] == "val_loss"
    assert ckpt["init_args"]["mode"] == "min"
    assert "val_loss" in ckpt["init_args"]["filename"]
    assert early["init_args"]["monitor"] == "val_loss"
    assert early["init_args"]["mode"] == "min"
    assert "pos_weight_sampler" not in cfg.get("data", {})


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
