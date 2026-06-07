import torch
import pytest
from bgsl.core.common.targets import BaseSoftOnsetTarget as SoftOnsetTarget


def test_negative_patient():
    target_fn = SoftOnsetTarget(horizon=6, tau=2.0)
    res = target_fn.build_single(onset_time=-1, seq_len=10, device=torch.device("cpu"))
    assert torch.all(res["soft_target"] == 0)
    assert torch.all(res["velocity_target"] == 0)
    assert torch.all(res["accel_target"] == 0)


def test_positive_patient_monotonic():
    target_fn = SoftOnsetTarget(horizon=6, tau=2.0)
    res = target_fn.build_single(onset_time=10, seq_len=15, device=torch.device("cpu"))
    g = res["soft_target"][0]  # [T]
    assert (torch.diff(g) >= -1e-6).all()


def test_padding_mask():
    target_fn = SoftOnsetTarget(horizon=6, tau=2.0)
    res = target_fn(torch.tensor([10, -1]), torch.tensor([12, 5]), torch.device("cpu"))
    mask = res["valid_mask"]
    assert mask[0, 11] == 1
    assert mask[1, 4] == 1
    assert mask[1, 5] == 0  # padded beyond seq_length 5


def test_analytic_derivatives():
    """Analytic derivatives of the generalized logistic should be exact."""
    target_fn = SoftOnsetTarget(horizon=6, tau=2.0, kappa=1.5)
    res = target_fn.build_single(onset_time=8, seq_len=16, device=torch.device("cpu"))
    g = res["soft_target"][0]
    dg = res["velocity_target"][0]
    d2g = res["accel_target"][0]

    # Verify via finite differences (central, interior only)
    dg_fd = torch.zeros_like(dg)
    dg_fd[1:-1] = (g[2:] - g[:-2]) / 2.0
    dg_fd[0] = g[1] - g[0]
    dg_fd[-1] = g[-1] - g[-2]

    # First derivative should match finite diff well (interior)
    assert torch.allclose(dg[2:-2], dg_fd[2:-2], atol=5e-3, rtol=1e-1)

    # Second derivative via finite diff of analytic first derivative
    d2g_fd = torch.zeros_like(d2g)
    d2g_fd[2:-2] = (dg[3:-1] - dg[1:-3]) / 2.0
    assert torch.allclose(d2g[3:-3], d2g_fd[3:-3], atol=5e-2, rtol=2e-1)


def test_negative_patient_explicit_zero_derivatives():
    """For y=0, derivatives must be explicitly zero even at boundaries."""
    target_fn = SoftOnsetTarget(horizon=6, tau=2.0)
    res = target_fn.build_single(onset_time=-1, seq_len=10, device=torch.device("cpu"))
    assert torch.all(res["velocity_target"] == 0.0)
    assert torch.all(res["accel_target"] == 0.0)
    # Derivative masks are no longer zeroed at boundaries in targets.py
    # They match the length mask. Finite_diff zeroing happens in the loss function.
    assert res["vel_mask"][0, 0] == 1.0
    assert res["vel_mask"][0, 1] == 1.0
    assert res["acc_mask"][0, 0] == 1.0
    assert res["acc_mask"][0, 1] == 1.0


def test_kappa_one_equals_sigmoid():
    """When kappa=1, g should match the standard sigmoid."""
    target_fn = SoftOnsetTarget(horizon=6, tau=2.0, kappa=1.0)
    res = target_fn.build_single(onset_time=10, seq_len=15, device=torch.device("cpu"))
    g = res["soft_target"][0]
    # Manually compute sigmoid
    t = torch.arange(15, dtype=torch.float32)
    c = 10.0 - 6.0 / 2.0  # midpoint = onset - H/2
    u = (t - c) / 2.0  # tau=2
    expected = torch.sigmoid(u)
    assert torch.allclose(g, expected, atol=1e-6)


def test_kappa_not_one():
    """When kappa != 1, g should differ from standard sigmoid."""
    target_fn = SoftOnsetTarget(horizon=6, tau=2.0, kappa=2.0)
    res = target_fn.build_single(onset_time=10, seq_len=15, device=torch.device("cpu"))
    g_k2 = res["soft_target"][0]

    target_fn_1 = SoftOnsetTarget(horizon=6, tau=2.0, kappa=1.0)
    res_1 = target_fn_1.build_single(onset_time=10, seq_len=15, device=torch.device("cpu"))
    g_k1 = res_1["soft_target"][0]

    assert not torch.allclose(g_k2, g_k1, atol=1e-4)


def test_long_sequence_no_nan():
    """Very long sequences should not produce NaN derivatives."""
    target_fn = SoftOnsetTarget(horizon=6, tau=0.5, kappa=0.5)
    res = target_fn.build_single(onset_time=100, seq_len=200, device=torch.device("cpu"))
    assert not torch.isnan(res["soft_target"]).any()
    assert not torch.isnan(res["velocity_target"]).any()
    assert not torch.isnan(res["accel_target"]).any()
    assert not torch.isinf(res["soft_target"]).any()
    assert not torch.isinf(res["velocity_target"]).any()
    assert not torch.isinf(res["accel_target"]).any()
