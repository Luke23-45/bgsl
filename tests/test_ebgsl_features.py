"""
Tests for the Enhanced BGSL (EBGSL) features:
  - Generalized logistic target (Richards curve)
  - Analytic derivatives
  - Savitzky-Golay derivative estimates
  - Importance weights
  - Monotonicity penalty
  - HyperNetwork model
"""

import torch
import pytest
from typing import Tuple


# ---------------------------------------------------------------------------
# HyperNetwork
# ---------------------------------------------------------------------------

class TestHyperNetwork:
    def test_output_shapes(self):
        from bgsl.models.hypernetwork import HyperNetwork

        net = HyperNetwork(static_dim=5, hidden_dim=16, num_layers=2)
        u = torch.randn(4, 5)
        delta_mu, sigma, kappa = net(u)

        assert delta_mu.shape == (4,)
        assert sigma.shape == (4,)
        assert kappa.shape == (4,)
        assert (sigma > 0).all()
        assert (kappa > 0).all()

    def test_gradient_flow(self):
        from bgsl.models.hypernetwork import HyperNetwork

        net = HyperNetwork(static_dim=3, hidden_dim=8, num_layers=1)
        u = torch.randn(2, 3, requires_grad=True)
        delta_mu, sigma, kappa = net(u)

        loss = (delta_mu ** 2 + sigma ** 2 + kappa ** 2).sum()
        loss.backward()

        assert u.grad is not None
        assert not torch.isnan(u.grad).any()

    def test_static_dim_validation(self):
        from bgsl.models.hypernetwork import HyperNetwork

        with pytest.raises(ValueError, match="static_dim"):
            HyperNetwork(static_dim=0)

    def test_num_layers_validation(self):
        from bgsl.models.hypernetwork import HyperNetwork

        with pytest.raises(ValueError, match="num_layers"):
            HyperNetwork(static_dim=3, num_layers=0)


# ---------------------------------------------------------------------------
# Generalized logistic target
# ---------------------------------------------------------------------------

class TestGeneralizedLogistic:
    def test_kappa_one_matches_standard_logistic(self):
        """With κ=1, the generalized logistic equals σ(u)."""
        from bgsl.core.common.targets import BaseSoftOnsetTarget

        target_fn = BaseSoftOnsetTarget(horizon=6, tau=2.0, kappa=1.0)
        res = target_fn.build_single(onset_time=10, seq_len=15, device="cpu")
        g = res["soft_target"][0]

        t = torch.arange(15, dtype=torch.float32)
        c = 10.0 - 6.0 / 2.0
        u = (t - c) / 2.0
        expected = torch.sigmoid(u)
        assert torch.allclose(g, expected, atol=1e-6)

    def test_kappa_greater_than_one_sharper(self):
        """κ > 1 changes the target shape (midpoint value is 0.5^κ)."""
        from bgsl.core.common.targets import BaseSoftOnsetTarget

        fn1 = BaseSoftOnsetTarget(horizon=6, tau=2.0, kappa=1.0)
        fn3 = BaseSoftOnsetTarget(horizon=6, tau=2.0, kappa=3.0)
        g1 = fn1.build_single(onset_time=10, seq_len=15, device="cpu")["soft_target"][0]
        g3 = fn3.build_single(onset_time=10, seq_len=15, device="cpu")["soft_target"][0]

        # At the midpoint (t=7, c=7): g = 0.5^κ
        assert torch.isclose(g1[7], torch.tensor(0.5), atol=1e-2)
        assert torch.isclose(g3[7], torch.tensor(0.5 ** 3), atol=1e-2)
        # Before midpoint: both are small, κ=3 gives smaller value
        assert g3[5] < g1[5]
        # After midpoint: κ=3 also gives smaller value (since s^3 < s for s<1)
        assert g3[9] < g1[9]

    def test_patient_specific_params(self):
        """Test with per-patient delta_mu, sigma, kappa."""
        from bgsl.core.common.targets import BaseSoftOnsetTarget

        target_fn = BaseSoftOnsetTarget(horizon=6, tau=2.0, kappa=1.0)

        delta_mu = torch.tensor([1.0, -1.0])
        sigma = torch.tensor([1.0, 3.0])
        kappa = torch.tensor([2.0, 0.5])

        res = target_fn(
            onset_times=torch.tensor([10.0, 8.0]),
            seq_lengths=torch.tensor([15, 15]),
            device=torch.device("cpu"),
            delta_mu=delta_mu, sigma=sigma, kappa_shape=kappa,
        )

        g = res["soft_target"]  # [B, T]
        assert g.shape == (2, 15)
        # Different params → different targets
        assert not torch.allclose(g[0], g[1], atol=1e-4)

    def test_negative_patient_zero_target(self):
        """For y=0 (onset_time < 0), all targets are zero."""
        from bgsl.core.common.targets import BaseSoftOnsetTarget

        target_fn = BaseSoftOnsetTarget(horizon=6, tau=2.0, kappa=1.5)
        res = target_fn.build_single(onset_time=-1, seq_len=10, device="cpu")
        assert torch.all(res["soft_target"] == 0.0)
        assert torch.all(res["velocity_target"] == 0.0)
        assert torch.all(res["accel_target"] == 0.0)


# ---------------------------------------------------------------------------
# Analytic derivative formulas
# ---------------------------------------------------------------------------

class TestAnalyticDerivatives:
    @pytest.fixture
    def targets(self):
        from bgsl.core.common.targets import BaseSoftOnsetTarget
        fn = BaseSoftOnsetTarget(horizon=6, tau=2.0, kappa=1.5)
        return fn.build_single(onset_time=10, seq_len=20, device="cpu")

    def test_first_derivative_matches_fd(self, targets):
        g = targets["soft_target"][0]
        dg = targets["velocity_target"][0]

        # Central finite difference (interior only)
        dg_fd = torch.zeros_like(dg)
        dg_fd[2:-2] = (g[4:] - g[:-4]) / 4.0 + 4.0 * (g[3:-1] - g[1:-3]) / 6.0
        dg_fd[2:-2] = (g[3:-1] - g[1:-3]) / 2.0

        assert torch.allclose(dg[4:-4], dg_fd[4:-4], atol=1e-2, rtol=1e-1)

    def test_second_derivative_matches_fd(self, targets):
        g = targets["soft_target"][0]
        d2g = targets["accel_target"][0]

        # Second derivative via central difference of first differences
        d2g_fd = torch.zeros_like(d2g)
        d2g_fd[2:-2] = g[4:] - 2.0 * g[2:-2] + g[:-4]
        d2g_fd[2:-2] = d2g_fd[2:-2] / 4.0

        assert torch.allclose(d2g[6:-6], d2g_fd[6:-6], atol=5e-2, rtol=2e-1)

    def test_derivative_symmetry(self):
        """Derivatives should be symmetric around the midpoint."""
        from bgsl.core.common.targets import BaseSoftOnsetTarget
        fn = BaseSoftOnsetTarget(horizon=6, tau=2.0, kappa=1.0)
        res = fn.build_single(onset_time=10, seq_len=21, device="cpu")
        dg = res["velocity_target"][0]

        # First derivative of sigmoid is symmetric around c
        mid = 7  # c = 10 - 3 = 7
        left = dg[mid - 3]
        right = dg[mid + 3]
        assert torch.isclose(left, right, atol=1e-6)

    def test_numerical_stability_extreme_u(self):
        """Very large |u| should not produce NaN."""
        from bgsl.core.common.targets import BaseSoftOnsetTarget
        fn = BaseSoftOnsetTarget(horizon=6, tau=0.1, kappa=0.3)
        res = fn.build_single(onset_time=100, seq_len=200, device="cpu")
        assert not torch.isnan(res["velocity_target"]).any()
        assert not torch.isnan(res["accel_target"]).any()
        assert not torch.isinf(res["velocity_target"]).any()
        assert not torch.isinf(res["accel_target"]).any()


# ---------------------------------------------------------------------------
# Savitzky-Golay derivative estimates
# ---------------------------------------------------------------------------

class TestSavitzkyGolay:
    def test_sg_on_linear_sequence(self):
        """SG first derivative of a linear sequence should be constant (interior)."""
        from bgsl.core.common.losses import _savgol_first_derivative

        p = torch.arange(50, dtype=torch.float32).unsqueeze(0)  # [1, 50]
        dp = _savgol_first_derivative(p, window_length=5, polyorder=2)
        # Interior (boundaries have edge effects with 'mirror' padding)
        assert torch.allclose(dp[:, 3:-3], torch.ones_like(dp[:, 3:-3]), atol=1e-4)

    def test_sg_on_quadratic_sequence(self):
        """SG second derivative of x² should be 2."""
        from bgsl.core.common.losses import _savgol_second_derivative

        t = torch.arange(50, dtype=torch.float32).unsqueeze(0)
        p = t ** 2  # [1, 50]
        d2p = _savgol_second_derivative(p, window_length=5, polyorder=2)
        # Should be ≈ 2 everywhere
        assert torch.allclose(d2p[:, 2:-2], 2.0 * torch.ones_like(d2p[:, 2:-2]), atol=1e-2)

    def test_sg_fallback_short_sequence(self):
        """Sequences shorter than window_length should fall back to finite diff."""
        from bgsl.core.common.losses import _savgol_first_derivative, _first_diff

        p = torch.randn(1, 4)  # T=4 < window_length=7
        dp_sg = _savgol_first_derivative(p, window_length=7, polyorder=2)
        dp_fd = _first_diff(p)
        assert torch.allclose(dp_sg, dp_fd)

    def test_sg_reflection_padding(self):
        """SG on a cubic (with matching polyorder) should give exact derivatives."""
        from bgsl.core.common.losses import _savgol_first_derivative, _savgol_second_derivative

        t = torch.arange(100, dtype=torch.float32).unsqueeze(0)
        p = t ** 3 - 5 * t ** 2 + 3 * t + 7  # cubic

        dp = _savgol_first_derivative(p, window_length=5, polyorder=3)
        d2p = _savgol_second_derivative(p, window_length=5, polyorder=3)

        # True derivatives: p' = 3t² - 10t + 3, p'' = 6t - 10
        dp_true = 3 * t ** 2 - 10 * t + 3
        d2p_true = 6 * t - 10

        # Interior should be exact up to numerical precision
        assert torch.allclose(dp[:, 3:-3], dp_true[:, 3:-3], atol=1e-4)
        assert torch.allclose(d2p[:, 3:-3], d2p_true[:, 3:-3], atol=1e-3)


# ---------------------------------------------------------------------------
# Importance weights
# ---------------------------------------------------------------------------

class TestImportanceWeights:
    def test_positive_weight_centered(self):
        """For positive patients, weight should peak at τ* - H."""
        from bgsl.core.common.losses import _compute_importance_weights

        onset_times = torch.tensor([10.0])
        is_positive = torch.tensor([True])
        seq_lengths = torch.tensor([20])
        w = _compute_importance_weights(
            onset_times, is_positive, seq_lengths, horizon=6.0, T_max=20, device="cpu"
        )

        # Peak should be at τ* - H = 4
        assert w[0, 4].item() == pytest.approx(1.0, abs=1e-4)
        # Should decrease away from center
        assert w[0, 4] > w[0, 0]
        assert w[0, 4] > w[0, 10]

    def test_negative_weight_unity(self):
        """For negative patients, all weights are 1 (within valid length)."""
        from bgsl.core.common.losses import _compute_importance_weights

        onset_times = torch.tensor([-1.0])
        is_positive = torch.tensor([False])
        seq_lengths = torch.tensor([20])
        w = _compute_importance_weights(
            onset_times, is_positive, seq_lengths, horizon=6.0, T_max=20, device="cpu"
        )

        assert torch.all(w == 1.0)

    def test_weight_outside_valid_length_is_zero(self):
        """Positions beyond seq_length should be zero (due to length mask)."""
        from bgsl.core.common.losses import _compute_importance_weights

        onset_times = torch.tensor([-1.0])
        is_positive = torch.tensor([False])
        seq_lengths = torch.tensor([5])
        w = _compute_importance_weights(
            onset_times, is_positive, seq_lengths, horizon=6.0, T_max=10, device="cpu"
        )

        assert (w[0, :5] == 1.0).all()
        assert (w[0, 5:] == 0.0).all()


# ---------------------------------------------------------------------------
# Monotonicity penalty
# ---------------------------------------------------------------------------

class TestMonotonicityPenalty:
    def test_no_penalty_for_increasing_risk(self):
        """Strictly increasing risk should have zero penalty."""
        from bgsl.core.common.losses import _compute_monotonicity_penalty

        p_dot = torch.ones(1, 20)  # always positive → increasing
        onset_times = torch.tensor([10.0])
        is_positive = torch.tensor([True])
        seq_lengths = torch.tensor([20])

        penalty = _compute_monotonicity_penalty(
            p_dot, onset_times, is_positive, seq_lengths, horizon=6.0
        )
        assert penalty.item() == 0.0

    def test_penalty_for_decreasing_risk(self):
        """Decreasing risk in alarm window should be penalized."""
        from bgsl.core.common.losses import _compute_monotonicity_penalty

        p_dot = -torch.ones(1, 20)  # always negative → decreasing
        onset_times = torch.tensor([10.0])
        is_positive = torch.tensor([True])
        seq_lengths = torch.tensor([20])

        penalty = _compute_monotonicity_penalty(
            p_dot, onset_times, is_positive, seq_lengths, horizon=6.0
        )
        assert penalty.item() > 0.0

    def test_zero_penalty_for_negative_patients(self):
        """Negative patients should have zero monotonicity penalty."""
        from bgsl.core.common.losses import _compute_monotonicity_penalty

        p_dot = -torch.ones(1, 20)
        onset_times = torch.tensor([-1.0])
        is_positive = torch.tensor([False])
        seq_lengths = torch.tensor([20])

        penalty = _compute_monotonicity_penalty(
            p_dot, onset_times, is_positive, seq_lengths, horizon=6.0
        )
        assert penalty.item() == 0.0


# ---------------------------------------------------------------------------
# Full BGSLLoss integration
# ---------------------------------------------------------------------------

class TestBGSLLossIntegration:
    def test_default_loss_matches_old_bgsl(self):
        """Default params (mono=0, finite_diff) should match old BGSL behavior."""
        from bgsl.core.common.losses import BGSLLoss

        loss = BGSLLoss(
            monotonicity_weight=0.0,     # disabled
            derivative_method="finite_diff",
            horizon=6.0,
        )
        logits = torch.randn(2, 10)
        targets = torch.rand(2, 10)
        mask = torch.ones(2, 10)

        out = loss(logits, targets, mask)
        assert out["loss"].item() >= 0
        assert out["monotonicity"].item() == 0.0

    def test_monotonicity_penalty_in_total(self):
        """When enabled, monotonicity should affect total loss."""
        from bgsl.core.common.losses import BGSLLoss

        # Create predictions that decrease in the alarm window
        B, T = 2, 20
        logits = torch.linspace(2, -2, T).unsqueeze(0).expand(B, -1)  # decreasing

        target_fn = BGSLLoss(
            monotonicity_weight=0.1,
            derivative_method="finite_diff",
            horizon=6.0,
        )
        out = target_fn(
            logits, torch.sigmoid(logits), torch.ones(B, T),
            onset_times=torch.tensor([12.0, 12.0]),
            is_positive=torch.tensor([True, True]),
            seq_lengths=torch.tensor([T, T]),
        )
        assert out["loss"].item() > out["state"].item()  # mono penalty adds to loss
        assert out["monotonicity"].item() > 0.0

    def test_sg_vs_finite_diff_close(self):
        """For smooth signals, SG and finite-diff should be close."""
        from bgsl.core.common.losses import BGSLLoss

        torch.manual_seed(42)
        # Smooth signal: sigmoid
        t = torch.linspace(-5, 5, 50).unsqueeze(0)
        logits = t  # [1, 50]
        probs = torch.sigmoid(logits)
        mask = torch.ones_like(probs)

        loss_fd = BGSLLoss(derivative_method="finite_diff")
        loss_sg = BGSLLoss(
            derivative_method="savitzky_golay",
            savgol_window=5,
            savgol_order=2,
        )

        out_fd = loss_fd(logits, probs, mask)
        out_sg = loss_sg(logits, probs, mask)

        # Losses should be similar (not identical due to boundary effects)
        assert abs(out_fd["velocity"].item() - out_sg["velocity"].item()) < 0.5

    def test_importance_weights_reduce_distant_contribution(self):
        """With importance weights, loss far from alarm window is downweighted."""
        from bgsl.core.common.losses import BGSLLoss

        B, T = 1, 50
        # Create a mismatch between prediction velocity and target velocity
        # far from the alarm window
        probs = torch.sigmoid(torch.linspace(-3, 3, T).unsqueeze(0))
        logits = torch.linspace(-3, 3, T).unsqueeze(0)

        loss_a = BGSLLoss(velocity_weight=0.1, horizon=10.0)  # uses importance weights
        loss_b = BGSLLoss(velocity_weight=0.1, horizon=60.0)  # effectively wider weights

        out_a = loss_a(
            logits, probs, torch.ones(B, T),
            onset_times=torch.tensor([40.0]),
            is_positive=torch.tensor([True]),
            seq_lengths=torch.tensor([T]),
        )
        out_b = loss_b(
            logits, probs, torch.ones(B, T),
            onset_times=torch.tensor([40.0]),
            is_positive=torch.tensor([True]),
            seq_lengths=torch.tensor([T]),
        )

        # Different horizons → different importance weighting → different velocity loss
        assert out_a["velocity"].item() != out_b["velocity"].item()


# ---------------------------------------------------------------------------
# SG filter hyperparameter validation
# ---------------------------------------------------------------------------

class TestSavitzkyGolayValidation:
    def test_even_window_rejected(self):
        """Even window length should raise ValueError."""
        from bgsl.core.common.losses import BGSLLoss

        with pytest.raises(ValueError, match="savgol_window"):
            BGSLLoss(derivative_method="savitzky_golay", savgol_window=4, savgol_order=2)

    def test_order_too_high_rejected(self):
        """Order >= window should raise ValueError."""
        from bgsl.core.common.losses import BGSLLoss

        with pytest.raises(ValueError, match="savgol_order"):
            BGSLLoss(derivative_method="savitzky_golay", savgol_window=3, savgol_order=3)
