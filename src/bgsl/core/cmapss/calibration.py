"""
bgsl/core/cmapss/calibration.py
-------------------------------
CMAPSS-specific calibration wrappers for BGSL.
Provides the default CMAPSS cycle-to-failure buckets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from bgsl.core.common.calibration import BaseTrajectoryCalibration

__all__ = ["CMAPSSCalibration"]


class CMAPSSCalibration(BaseTrajectoryCalibration):
    """
    CMAPSS-specific calibration analysis providing cycle-based horizon buckets.
    """

    # Default buckets in cycles before failure
    DEFAULT_BUCKETS = {
        "0-10 cycles": (0, 10),
        "10-30 cycles": (10, 30),
        "30-50 cycles": (30, 50),
        "50+ cycles": (50, 9999),
    }

    def calibration_by_horizon(
        self,
        patient_probs: List[np.ndarray],
        patient_labels: List[np.ndarray],
        patient_onset_times: List[float],
        buckets: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Dict]:
        """
        Compute calibration within time-to-onset buckets for CMAPSS.
        Uses default cycle buckets if none are provided.
        """
        if buckets is None:
            buckets = self.DEFAULT_BUCKETS
        
        return super().calibration_by_horizon(
            patient_probs, patient_labels, patient_onset_times, buckets
        )

    def calibration_report(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        patient_probs: Optional[List[np.ndarray]] = None,
        patient_soft_targets: Optional[List[np.ndarray]] = None,
        patient_onset_times: Optional[List[float]] = None,
        patient_labels: Optional[List[np.ndarray]] = None,
        buckets: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, object]:
        if buckets is None:
            buckets = self.DEFAULT_BUCKETS
        
        return super().calibration_report(
            probs, labels, patient_probs, patient_soft_targets,
            patient_onset_times, patient_labels, buckets
        )
