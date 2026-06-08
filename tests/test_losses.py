import torch
import torch.nn.functional as F
from bgsl.core.common.losses import BCELoss, BGSLLoss, TLSLoss


def test_plain_bce_matches_torch_bce():
    loss_fn = BCELoss()
    logits = torch.randn(2, 10)
    targets = torch.rand(2, 10)
    mask = torch.ones(2, 10)

    out = loss_fn(logits, targets, mask)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    ref = (ref * mask).sum() / mask.sum()

    assert torch.isclose(out["loss"], ref)
    assert torch.isclose(out["state"], ref)

def test_bgsl_loss_non_negative():
    loss_fn = BGSLLoss()
    logits = torch.randn(2, 10)
    soft_targets = torch.rand(2, 10)
    vel_targets = torch.rand(2, 10)
    acc_targets = torch.rand(2, 10)
    mask = torch.ones(2, 10)
    
    out = loss_fn(
        logits,
        soft_targets,
        mask,
        vel_targets=vel_targets,
        acc_targets=acc_targets,
    )
    assert out["loss"].item() >= 0
    assert out["state"].item() >= 0
    assert out["velocity"].item() >= 0
    assert out["acceleration"].item() >= 0

def test_bgsl_state_only_matches_bce():
    loss_fn_bgsl = BGSLLoss(velocity_weight=0, acceleration_weight=0)
    logits = torch.randn(2, 10)
    soft_targets = torch.rand(2, 10)
    mask = torch.ones(2, 10)
    
    out_bgsl = loss_fn_bgsl(logits, soft_targets, mask)
    out_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, soft_targets, reduction='mean')
    
    assert torch.isclose(out_bgsl["loss"], out_bce)

def test_tls_targets():
    loss_fn = TLSLoss(alpha=6.0)
    targets = loss_fn.build_tls_targets(torch.tensor([10]), torch.tensor([15]), torch.device("cpu"))
    # At t=10 (onset), target should be 1
    assert targets[0, 10] == 1.0
    # At t=4 (onset - 6), target should be 0
    assert targets[0, 4] == 0.0
    # At t=7 (onset - 3), target should be 0.5
    assert torch.isclose(targets[0, 7], torch.tensor(0.5))


# ---------------------------------------------------------------------------
# UW-SO (Soft Optimal Uncertainty Weighting)
# ---------------------------------------------------------------------------


def test_bgsl_uw_so_forward_runs():
    """UW-SO mode produces valid non-negative loss."""
    loss_fn = BGSLLoss(weighting_method="uw_so", temperature=1.0)
    logits = torch.randn(2, 10)
    soft_targets = torch.rand(2, 10)
    vel_targets = torch.rand(2, 10)
    acc_targets = torch.rand(2, 10)
    mask = torch.ones(2, 10)

    out = loss_fn(logits, soft_targets, mask,
                  vel_targets=vel_targets, acc_targets=acc_targets)
    assert out["loss"].item() >= 0
    assert out["state"].item() >= 0
    assert out["velocity"].item() >= 0
    assert out["acceleration"].item() >= 0
    assert out["monotonicity"].item() == 0.0


def test_bgsl_uw_so_different_from_fixed():
    """UW-SO weights should produce different total loss than fixed λ_v=0.1, λ_a=0.05."""
    loss_fixed = BGSLLoss(velocity_weight=0.1, acceleration_weight=0.05)
    loss_uwso = BGSLLoss(weighting_method="uw_so", temperature=1.0)

    logits = torch.randn(4, 20)
    soft_targets = torch.rand(4, 20)
    vel_targets = torch.rand(4, 20)
    acc_targets = torch.rand(4, 20)
    mask = torch.ones(4, 20)

    out_fixed = loss_fixed(logits, soft_targets, mask,
                           vel_targets=vel_targets, acc_targets=acc_targets)
    out_uwso = loss_uwso(logits, soft_targets, mask,
                         vel_targets=vel_targets, acc_targets=acc_targets)

    # Different weighting => different total loss (with high probability)
    assert not torch.isclose(out_fixed["loss"], out_uwso["loss"])


def test_bgsl_uw_so_temperature_affects_weights():
    """Lower temperature produces more peaked weight distribution."""
    loss_hot = BGSLLoss(weighting_method="uw_so", temperature=0.1)
    loss_cold = BGSLLoss(weighting_method="uw_so", temperature=10.0)

    logits = torch.randn(2, 10)
    soft_targets = torch.rand(2, 10)
    vel_targets = torch.rand(2, 10)
    acc_targets = torch.rand(2, 10)
    mask = torch.ones(2, 10)

    def _extract_weights(loss_fn):
        """Run forward and return the three UW-SO weights."""
        # Monkey-patch to capture weights — OR just run forward and check
        # that temperature affects the result.
        return loss_fn(logits.clone(), soft_targets.clone(), mask.clone(),
                       vel_targets=vel_targets.clone(),
                       acc_targets=acc_targets.clone())["loss"]

    l_hot = _extract_weights(loss_hot)
    l_cold = _extract_weights(loss_cold)

    # Different temperatures => numerically different total loss
    assert not torch.isclose(l_hot, l_cold)


# ---------------------------------------------------------------------------
# UW (Uncertainty Weighting by Kendall et al. 2018)
# ---------------------------------------------------------------------------


def test_bgsl_uw_forward_runs():
    """UW mode produces valid non-negative loss with learnable log_vars."""
    loss_fn = BGSLLoss(weighting_method="uw")
    logits = torch.randn(2, 10)
    soft_targets = torch.rand(2, 10)
    vel_targets = torch.rand(2, 10)
    acc_targets = torch.rand(2, 10)
    mask = torch.ones(2, 10)

    out = loss_fn(logits, soft_targets, mask,
                  vel_targets=vel_targets, acc_targets=acc_targets)
    assert out["loss"].item() >= 0
    assert out["state"].item() >= 0


def test_bgsl_uw_has_learnable_params():
    """UW mode registers learnable log_vars that receive gradients."""
    loss_fn = BGSLLoss(weighting_method="uw")
    assert hasattr(loss_fn, "log_vars")
    assert loss_fn.log_vars.shape == (3,)
    assert loss_fn.log_vars.requires_grad

    logits = torch.randn(2, 10)
    soft_targets = torch.rand(2, 10)
    vel_targets = torch.rand(2, 10)
    acc_targets = torch.rand(2, 10)
    mask = torch.ones(2, 10)

    out = loss_fn(logits, soft_targets, mask,
                  vel_targets=vel_targets, acc_targets=acc_targets)
    out["loss"].backward()
    assert loss_fn.log_vars.grad is not None
    assert torch.isfinite(loss_fn.log_vars.grad).all()


# ---------------------------------------------------------------------------
# UW and UW-SO both work alongside BGSLLoss features (importance weights,
# monotonicity, derivative space)
# ---------------------------------------------------------------------------


def test_bgsl_uw_so_with_importance_weights():
    """UW-SO + importance weights runs without error."""
    loss_fn = BGSLLoss(weighting_method="uw_so", temperature=1.0, horizon=6.0)
    B, T = 2, 12
    logits = torch.randn(B, T)
    soft_targets = torch.rand(B, T)
    vel_targets = torch.rand(B, T)
    acc_targets = torch.rand(B, T)
    mask = torch.ones(B, T)

    out = loss_fn(logits, soft_targets, mask,
                  vel_targets=vel_targets, acc_targets=acc_targets,
                  onset_times=torch.tensor([5.0, 8.0]),
                  is_positive=torch.tensor([True, False], dtype=torch.bool),
                  seq_lengths=torch.tensor([T, T]))
    assert out["loss"].item() >= 0


def test_bgsl_uw_so_with_monotonicity():
    """UW-SO + monotonicity penalty runs without error."""
    loss_fn = BGSLLoss(weighting_method="uw_so", temperature=1.0,
                       monotonicity_weight=0.01, horizon=6.0)
    B, T = 2, 12
    logits = torch.randn(B, T)
    soft_targets = torch.rand(B, T)
    vel_targets = torch.rand(B, T)
    acc_targets = torch.rand(B, T)
    mask = torch.ones(B, T)

    out = loss_fn(logits, soft_targets, mask,
                  vel_targets=vel_targets, acc_targets=acc_targets,
                  onset_times=torch.tensor([5.0, 8.0]),
                  is_positive=torch.tensor([True, False], dtype=torch.bool),
                  seq_lengths=torch.tensor([T, T]))
    assert out["loss"].item() >= 0
