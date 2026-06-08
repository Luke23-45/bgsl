"""
bgsl/core/sepsis/metrics.py
---------------------------
Sepsis-specific evaluation metrics for BGSL-trained early-warning models.
Extends the generic BGSL metrics to include the PhysioNet Utility Score.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

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

    def _compute_point_estimates(self, indices: Optional[np.ndarray] = None) -> Dict[str, float]:
        # Compute all generic metrics first
        results = super()._compute_point_estimates(indices)
        
        # Add Sepsis-specific PhysioNet utility
        results["physionet_utility"] = self._physionet_utility(indices)
        
        # Sepsis-specific FAPPD (False Alerts Per Patient-Day)
        # Note: fa_rate from base class is already per time-unit.
        # Since Sepsis time-unit is 1 hour, rate per day is fa_rate * 24.
        if "fa_rate" in results:
            results["fappd"] = results["fa_rate"] * 24.0
            
        return results

    def _physionet_utility(self, indices: Optional[np.ndarray] = None) -> float:
        """
        Official PhysioNet 2019 challenge normalized utility score.
        Implements the per-timestep utility function from evaluate_sepsis_score.py
        and normalizes as (U_obs - U_inactive) / (U_opt - U_inactive).
        """
        dt_early = -12.0
        dt_optimal = -6.0
        dt_late = 3.0
        max_u_tp = 1.0
        min_u_fn = -2.0
        u_fp = -0.05
        u_tn = 0.0

        n = len(self._predictions)
        observed = np.zeros(n)
        best = np.zeros(n)
        inaction = np.zeros(n)

        eval_indices = np.unique(indices) if indices is not None else np.arange(n)

        for k in eval_indices:
            p = self._predictions[k]
            labels = p.hard_labels
            T = len(labels)
            predictions = (p.probs >= self.threshold).astype(float)

            t_sepsis = int(p.onset_time) if p.has_event else float('inf')

            # Best: predict 1 during [t_sepsis+dt_early, t_sepsis+dt_late], 0 elsewhere
            best_preds = np.zeros(T)
            if p.has_event:
                lo = max(0, t_sepsis + int(dt_early))
                hi = min(T, t_sepsis + int(dt_late) + 1)
                best_preds[lo:hi] = 1.0
            inaction_preds = np.zeros(T)

            observed[k] = self._compute_utility(labels, predictions, t_sepsis,
                dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
            best[k] = self._compute_utility(labels, best_preds, t_sepsis,
                dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
            inaction[k] = self._compute_utility(labels, inaction_preds, t_sepsis,
                dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

        if indices is not None:
            total_obs = observed[indices].sum()
            total_best = best[indices].sum()
            total_inaction = inaction[indices].sum()
        else:
            total_obs = observed.sum()
            total_best = best.sum()
            total_inaction = inaction.sum()

        denom = total_best - total_inaction
        if abs(denom) < 1e-9:
            return 0.0
        return float((total_obs - total_inaction) / denom)

    @staticmethod
    def _compute_utility(
        labels: np.ndarray,
        predictions: np.ndarray,
        t_sepsis: float,
        dt_early: float,
        dt_optimal: float,
        dt_late: float,
        max_u_tp: float,
        min_u_fn: float,
        u_fp: float,
        u_tn: float,
    ) -> float:
        """Per-patient utility following the official PhysioNet 2019 evaluate_sepsis_score.py."""
        n = len(labels)
        is_septic = np.isfinite(t_sepsis)

        m_1 = max_u_tp / (dt_optimal - dt_early)
        b_1 = -m_1 * dt_early
        m_2 = -max_u_tp / (dt_late - dt_optimal)
        b_2 = -m_2 * dt_late
        m_3 = min_u_fn / (dt_late - dt_optimal)
        b_3 = -m_3 * dt_optimal

        u = np.zeros(n)
        for t in range(n):
            if is_septic and t <= int(t_sepsis) + dt_late:
                if predictions[t]:
                    if t <= int(t_sepsis) + dt_optimal:
                        u[t] = max(m_1 * (t - int(t_sepsis)) + b_1, u_fp)
                    elif t <= int(t_sepsis) + dt_late:
                        u[t] = m_2 * (t - int(t_sepsis)) + b_2
                else:
                    if t <= int(t_sepsis) + dt_optimal:
                        u[t] = 0.0
                    elif t <= int(t_sepsis) + dt_late:
                        u[t] = m_3 * (t - int(t_sepsis)) + b_3
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

        return float(np.sum(u))
