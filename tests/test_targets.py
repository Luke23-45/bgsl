import torch
import pytest
from bgsl.core.targets import SoftOnsetTarget

def test_negative_patient():
    target_fn = SoftOnsetTarget(horizon_hours=6, tau=2.0)
    res = target_fn.build_single(onset_hour=-1, seq_len=10, device=torch.device("cpu"))
    assert torch.all(res["soft_target"] == 0)
    assert torch.all(res["velocity_target"] == 0)
    assert torch.all(res["accel_target"] == 0)

def test_positive_patient_monotonic():
    target_fn = SoftOnsetTarget(horizon_hours=6, tau=2.0)
    res = target_fn.build_single(onset_hour=10, seq_len=15, device=torch.device("cpu"))
    g = res["soft_target"][0] # [T]
    assert (torch.diff(g) >= 0).all()

def test_padding_mask():
    target_fn = SoftOnsetTarget(horizon_hours=6, tau=2.0)
    res = target_fn(torch.tensor([10, -1]), torch.tensor([12, 5]), torch.device("cpu"))
    mask = res["valid_mask"]
    assert mask[0, 11] == 1
    assert mask[0, 12] == 0
    assert mask[1, 4] == 1
    assert mask[1, 5] == 0

def test_derivative_undefined_at_start():
    target_fn = SoftOnsetTarget(horizon_hours=6, tau=2.0)
    res = target_fn.build_single(onset_hour=5, seq_len=10, device=torch.device("cpu"))
    assert res["velocity_target"][0, 0] == 0
    assert res["accel_target"][0, 0] == 0
    assert res["accel_target"][0, 1] == 0
    assert res["vel_mask"][0, 0] == 0
    assert res["acc_mask"][0, 0] == 0
    assert res["acc_mask"][0, 1] == 0
