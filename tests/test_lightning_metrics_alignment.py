from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bgsl.train.sepsis.module import SepsisLightningModule
from bgsl.train.common.module import BaseBGSLLightningModule


class _DummyModel(nn.Module):
    def forward(self, x, mask=None, seq_lens=None):
        return torch.zeros(x.shape[0], x.shape[1], device=x.device)


class _DummyLoss(nn.Module):
    def forward(self, logits, targets, mask, **kwargs):
        return {
            "loss": torch.tensor(0.0, device=logits.device),
            "state": torch.tensor(0.0, device=logits.device),
            "velocity": torch.tensor(0.0, device=logits.device),
            "acceleration": torch.tensor(0.0, device=logits.device),
        }


class _CaptureMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_preds = None
        self.last_targets = None

    last_preds: torch.Tensor | None = None
    last_targets: torch.Tensor | None = None

    def update(self, preds, targets):
        self.last_preds = preds.detach().clone()
        self.last_targets = targets.detach().clone()

    def compute(self):
        return {}

    def reset(self):
        self.last_preds = None
        self.last_targets = None


class _CaptureTrajectoryTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_prediction = None

    def add(self, prediction):
        self.last_prediction = prediction

    def select_threshold_fixed_sensitivity(self, *args, **kwargs):
        return 0.5

    def compute(self):
        return {}

    def reset(self):
        self.last_prediction = None


def test_shared_step_separates_clinical_and_target_metrics():
    module = BaseBGSLLightningModule(backbone=_DummyModel(), loss_fn=_DummyLoss())
    module.train_metrics = _CaptureMetrics()
    module.train_target_metrics = _CaptureMetrics()
    module.log = lambda *args, **kwargs: None

    batch = {
        "vitals": torch.zeros(1, 4, 40),
        "masks": torch.ones(1, 4, 40),
        "seq_len": torch.tensor([4]),
        "valid_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        "hard_labels": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "hard_targets": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        "soft_targets": torch.zeros(1, 4),
        "vel_targets": torch.zeros(1, 4),
        "accel_targets": torch.zeros(1, 4),
        "vel_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        "acc_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "onset_hour": torch.tensor([3]),
        "is_sepsis": torch.tensor([True]),
        "patient_id": ["p1"],
    }

    module._shared_step(batch, 0, "train")

    assert module.train_metrics.last_targets is not None
    assert module.train_metrics.last_targets.tolist() == [0, 0, 0, 1]
    assert module.train_target_metrics.last_targets is not None
    assert module.train_target_metrics.last_targets.tolist() == [0, 1, 1, 1]


def test_sepsis_trajectory_metrics_use_clinical_labels():
    module = SepsisLightningModule(
        backbone=_DummyModel(),
        loss_fn=_DummyLoss(),
    )
    module.validation_threshold_metrics = _CaptureTrajectoryTracker()
    module.trajectory_metrics = _CaptureTrajectoryTracker()

    batch = {
        "vitals": torch.zeros(1, 4, 40),
        "masks": torch.ones(1, 4, 40),
        "seq_len": torch.tensor([4]),
        "valid_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        "hard_labels": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "hard_targets": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        "soft_targets": torch.zeros(1, 4),
        "onset_hour": torch.tensor([3]),
        "is_sepsis": torch.tensor([True]),
        "patient_id": ["p1"],
    }
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.9]])

    module._update_trajectory_metrics(batch, probs, "val")
    assert module.validation_threshold_metrics.last_prediction is not None
    assert module.validation_threshold_metrics.last_prediction.hard_labels.tolist() == [0.0, 1.0, 1.0, 1.0]

    module._update_trajectory_metrics(batch, probs, "test")
    assert module.trajectory_metrics.last_prediction is not None
    assert module.trajectory_metrics.last_prediction.hard_labels.tolist() == [0.0, 0.0, 0.0, 1.0]
