from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bgsl.models.lightning import BGSLLightningModule


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


def test_shared_step_uses_hard_targets_for_metrics():
    module = BGSLLightningModule(model=_DummyModel(), loss_fn=_DummyLoss())
    module.train_metrics = _CaptureMetrics()
    module.log = lambda *args, **kwargs: None

    batch = {
        "vitals": torch.zeros(1, 4, 28),
        "masks": torch.ones(1, 4, 28),
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
    assert module.train_metrics.last_targets.tolist() == [0, 1, 1, 1]
