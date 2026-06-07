import numpy as np
from bgsl.core.sepsis.metrics import SepsisMetrics as BGSLMetrics, PatientPrediction

def test_asf_constant_trajectory():
    tracker = BGSLMetrics()
    p = PatientPrediction(
        id="test",
        probs=np.zeros(24),
        hard_labels=np.zeros(24),
        soft_targets=np.zeros(24),
        has_event=False,
        onset_time=-1,
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
        id="test",
        probs=probs,
        hard_labels=np.ones(24),
        soft_targets=probs,
        has_event=True,
        onset_time=23,
        horizon=6,
        seq_len=24
    )
    tracker.add(p)
    res = tracker.compute()
    assert res["poms"] == 1.0
