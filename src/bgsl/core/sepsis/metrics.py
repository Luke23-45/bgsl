"""
bgsl/core/sepsis/metrics.py
---------------------------
Sepsis-specific evaluation metrics for BGSL-trained early-warning models.
Extends the generic BGSL metrics to include the PhysioNet Utility Score.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from bgsl.core.common.metrics import BaseSequencePrediction, BaseTrajectoryMetrics

__all__ = ["PatientPrediction", "SepsisMetrics"]


class PatientPrediction(BaseSequencePrediction):
    """
    Sepsis-specific prediction wrapper.
    Adapts domain-specific fields to the generic BaseSequencePrediction interface.
    """
    @property
    def is_sepsis(self) -> bool:
        return self.has_event

    @property
    def onset_hour(self) -> float:
        return self.onset_time


class SepsisMetrics(BaseTrajectoryMetrics):
    """
    Computes BGSL evaluation metrics specifically for Sepsis, including
    the PhysioNet 2019 utility score.
    """

    def compute(self) -> Dict[str, float]:
        # Compute all generic metrics first
        results = super().compute()
        
        # Add Sepsis-specific PhysioNet utility
        results["physionet_utility"] = self._physionet_utility()
        
        # Sepsis-specific FAPPD (False Alerts Per Patient-Day)
        # Note: fa_rate from base class is already per time-unit.
        # Since Sepsis time-unit is 1 hour, rate per day is fa_rate * 24.
        if "fa_rate" in results:
            results["fappd"] = results["fa_rate"] * 24.0
            
        return results

    def _physionet_utility(self) -> float:
        """
        Approximation of the PhysioNet 2019 challenge utility score.
        """
        total_utility = 0.0
        for p in self._predictions:
            alert = self._sustained_alert(p.probs)
            util = self._patient_utility(alert, p.hard_labels, p.onset_time, p.horizon)
            total_utility += util

        n = len(self._predictions)
        return float(total_utility / max(n, 1))

    @staticmethod
    def _patient_utility(
        alert: np.ndarray,
        hard_labels: np.ndarray,
        onset_hour: float,
        horizon: float,
    ) -> float:
        T = len(hard_labels)

        if onset_hour < 0:
            n_fp = alert.sum()
            return float(-0.05 * n_fp)

        first_alert = int(np.argmax(alert)) if alert.any() else T
        lead = onset_hour - first_alert

        if alert.any() and lead >= 0:
            utility = max(0.0, lead) / max(horizon, 1.0)
            return float(utility)
        elif not alert.any():
            return -2.0
        else:
            return 0.0
