import numpy as np
from bgsl.core.sepsis.metrics import SepsisMetrics as BGSLMetrics, PatientPrediction

def test_asf_constant_trajectory():
    tracker = BGSLMetrics()
    p = PatientPrediction(
        patient_id="test",
        probs=np.zeros(24),
        hard_labels=np.zeros(24),
        soft_targets=np.zeros(24),
        is_sepsis=False,
        onset_hour=-1,
        horizon=6,
        seq_len=24
    )
    tracker.add(p)
    res = tracker.compute()
    assert res["asf"] == 0.0

def test_poms_perfect_monotonic():
    tracker = BGSLMetrics()
    probs = np.linspace(0.1, 0.9, 24) # perfectly rising
    p = PatientPrediction(
        patient_id="test",
        probs=probs,
        hard_labels=np.ones(24),
        soft_targets=probs,
        is_sepsis=True,
        onset_hour=23,
        horizon=6,
        seq_len=24
    )
    tracker.add(p)
    res = tracker.compute()
    assert res["poms"] == 1.0


def test_fixed_fappd_selects_highest_valid_threshold():
    tracker = BGSLMetrics(threshold=0.5, sustained_k=1)
    p = PatientPrediction(
        patient_id="neg",
        probs=np.full(24, 0.6),
        hard_labels=np.zeros(24),
        soft_targets=np.zeros(24),
        is_sepsis=False,
        onset_hour=-1,
        horizon=6,
        seq_len=24,
    )
    tracker.add(p)

    thresh = tracker.select_threshold_fixed_fappd(target_fappd=0.0)
    assert thresh == 1.0
    assert tracker.threshold == 1.0
