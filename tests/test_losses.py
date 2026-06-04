import torch
from bgsl.core.losses import BGSLLoss, TLSLoss

def test_bgsl_loss_non_negative():
    loss_fn = BGSLLoss()
    logits = torch.randn(2, 10)
    soft_targets = torch.rand(2, 10)
    vel_targets = torch.rand(2, 10)
    acc_targets = torch.rand(2, 10)
    mask = torch.ones(2, 10)
    
    out = loss_fn(logits, soft_targets, vel_targets, acc_targets, mask)
    assert out["loss"].item() >= 0
    assert out["state"].item() >= 0
    assert out["velocity"].item() >= 0
    assert out["acceleration"].item() >= 0

def test_bgsl_state_only_matches_bce():
    loss_fn_bgsl = BGSLLoss(velocity_weight=0, acceleration_weight=0)
    logits = torch.randn(2, 10)
    soft_targets = torch.rand(2, 10)
    mask = torch.ones(2, 10)
    
    out_bgsl = loss_fn_bgsl(logits, soft_targets, torch.zeros_like(logits), torch.zeros_like(logits), mask)
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
