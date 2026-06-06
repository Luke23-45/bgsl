"""
bgsl/core/sepsis/calibration.py
-------------------------------
Sepsis-specific calibration wrappers for BGSL.
Provides the default Sepsis time-to-onset buckets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from bgsl.core.common.calibration import BaseTrajectoryCalibration

__all__ = ["SepsisCalibration"]


class SepsisCalibration(BaseTrajectoryCalibration):
    """
    Sepsis-specific calibration analysis providing hour-based horizon buckets.
    """

    # Default buckets in hours before onset
    DEFAULT_BUCKETS = {
        "0-2h": (0, 2),
        "2-6h": (2, 6),
        "6-12h": (6, 12),
        "12h+": (12, 9999),
    }

    def calibration_by_horizon(
        self,
        patient_probs: List[np.ndarray],
        patient_labels: List[np.ndarray],
        patient_onset_times: List[float],
        buckets: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Dict]:
        """
        Compute calibration within time-to-onset buckets for Sepsis.
        Uses default hour buckets if none are provided.
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
