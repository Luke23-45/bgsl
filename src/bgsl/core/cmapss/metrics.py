"""
bgsl/core/cmapss/metrics.py
---------------------------
CMAPSS-specific evaluation metrics for BGSL-trained early-warning models.
Extends the generic BGSL metrics.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from bgsl.core.common.metrics import BaseSequencePrediction, BaseTrajectoryMetrics

__all__ = ["CyclePrediction", "CMAPSSMetrics"]


class CyclePrediction(BaseSequencePrediction):
    """
    CMAPSS-specific prediction wrapper.
    Adapts domain-specific fields to the generic BaseSequencePrediction interface.
    """
    @property
    def is_failure(self) -> bool:
        return self.has_event

    @property
    def failure_cycle(self) -> float:
        return self.onset_time


class CMAPSSMetrics(BaseTrajectoryMetrics):
    """
    Computes BGSL evaluation metrics specifically for CMAPSS.
    (e.g. Asymmetric scoring function could be implemented here).
    """

    def _compute_point_estimates(self) -> Dict[str, float]:
        results = super()._compute_point_estimates()
        # Add CMAPSS-specific metrics here if needed
        return results
