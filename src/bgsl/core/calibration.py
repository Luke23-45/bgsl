"""
bgsl/calibration.py
--------------------
Calibration analysis for BGSL risk trajectories.

Modules
-------
TrajectoryCalibration
    - Reliability curves (ECE + diagram data)
    - Calibration by time-to-onset bucket (0–2h, 2–6h, 6–12h, >12h)
    - Trajectory Calibration Error (TCE): mean |p_t - g_t| in event window
    - Platt scaling (logistic recalibration)
    - Isotonic regression recalibration

The critical distinction for BGSL calibration
----------------------------------------------
Standard calibration: does p_t = P(event | input)?
Trajectory calibration: does p_t = g_t throughout the risk trajectory?

TCE specifically measures how well the model tracks the soft onset
target g_t near the event window — not just whether the final
probability is correct, but whether the shape of the risk curve
matches the expected temporal profile.

Reference: BGSL plan.md §10.2, §10.4
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = ["TrajectoryCalibration"]


class TrajectoryCalibration:
    """
    Calibration analysis for BGSL early-warning models.

    Parameters
    ----------
    n_bins : int
        Number of bins for reliability curves. Default 10.
    event_window_hours : int
        Half-width of event window for TCE (±event_window_hours around onset).
    """

    def __init__(
        self,
        n_bins: int = 10,
        event_window_hours: int = 3,
    ) -> None:
        self.n_bins = n_bins
        self.event_window = event_window_hours
        self._platt: Optional[LogisticRegression] = None
        self._isotonic: Optional[IsotonicRegression] = None

    # ------------------------------------------------------------------
    # Reliability curve
    # ------------------------------------------------------------------

    def reliability_curve(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute reliability diagram data.

        Parameters
        ----------
        probs : np.ndarray [N]
            Predicted probabilities (time-step level).
        labels : np.ndarray [N]
            Binary ground-truth labels.

        Returns
        -------
        dict:
            "fraction_of_positives" : [n_bins]
            "mean_predicted_value"  : [n_bins]
            "ece"                   : float
            "bin_counts"            : [n_bins]
        """
        # sklearn calibration_curve
        frac_pos, mean_pred = calibration_curve(
            labels, probs, n_bins=self.n_bins, strategy="uniform"
        )

        # ECE
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        bin_counts = []
        n = len(probs)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = (probs >= lo) & (probs < hi)
            count = in_bin.sum()
            bin_counts.append(int(count))
            if count == 0:
                continue
            conf = probs[in_bin].mean()
            acc = labels[in_bin].mean()
            ece += (count / n) * abs(conf - acc)

        return {
            "fraction_of_positives": frac_pos,
            "mean_predicted_value": mean_pred,
            "ece": float(ece),
            "bin_counts": np.array(bin_counts),
        }

    # ------------------------------------------------------------------
    # Calibration stratified by time-to-onset
    # ------------------------------------------------------------------

    def calibration_by_horizon(
        self,
        patient_probs: List[np.ndarray],
        patient_labels: List[np.ndarray],
        patient_onset_hours: List[int],
    ) -> Dict[str, Dict]:
        """
        Compute calibration within time-to-onset buckets.

        Buckets:
            [0, 2)h  — imminent (peri-onset)
            [2, 6)h  — near-term
            [6, 12)h — early
            ≥ 12h    — distal

        Parameters
        ----------
        patient_probs : list of [T_i] arrays
        patient_labels : list of [T_i] binary arrays
        patient_onset_hours : list of onset time in hours (-1 for negative)

        Returns
        -------
        dict mapping bucket_name → reliability_curve output
        """
        buckets = {
            "0-2h": (0, 2),
            "2-6h": (2, 6),
            "6-12h": (6, 12),
            "12h+": (12, 9999),
        }
        bucket_probs: Dict[str, List[float]] = {k: [] for k in buckets}
        bucket_labels: Dict[str, List[float]] = {k: [] for k in buckets}

        for probs, labels, onset in zip(
            patient_probs, patient_labels, patient_onset_hours
        ):
            if onset < 0:
                continue  # Negative patients excluded from horizon analysis

            T = len(probs)
            for t in range(T):
                time_to_onset = onset - t  # positive = still before onset
                for bname, (lo, hi) in buckets.items():
                    if lo <= time_to_onset < hi:
                        bucket_probs[bname].append(float(probs[t]))
                        bucket_labels[bname].append(float(labels[t]))
                        break

        results: Dict[str, Dict] = {}
        for bname in buckets:
            bp = np.array(bucket_probs[bname])
            bl = np.array(bucket_labels[bname])
            if len(bp) < 10 or bl.sum() == 0:
                results[bname] = {"ece": float("nan"), "n": len(bp)}
                continue
            results[bname] = self.reliability_curve(bp, bl)
            results[bname]["n"] = len(bp)

        return results

    # ------------------------------------------------------------------
    # Trajectory Calibration Error (TCE)
    # ------------------------------------------------------------------

    def trajectory_calibration_error(
        self,
        patient_probs: List[np.ndarray],
        patient_soft_targets: List[np.ndarray],
        patient_onset_hours: List[int],
    ) -> Dict[str, float]:
        """
        TCE = mean |p_t - g_t| in the event window [t* - W, t* + W].

        Unlike standard ECE (which measures pointwise accuracy),
        TCE measures how well the model tracks the soft onset target
        g_t in the critical window around onset.

        Parameters
        ----------
        patient_probs : list of [T_i] predicted probability arrays
        patient_soft_targets : list of [T_i] soft target g_t arrays
        patient_onset_hours : list of onset times (-1 for negative)

        Returns
        -------
        dict:
            "tce"              : mean TCE across patients (in event window)
            "tce_pre_onset"    : TCE in [t* - W, t*] (pre-onset)
            "tce_post_onset"   : TCE in [t*, t* + W] (post-onset)
            "n_patients"       : number of positive patients with valid window
        """
        tce_all, tce_pre, tce_post = [], [], []

        for probs, g_t, onset in zip(
            patient_probs, patient_soft_targets, patient_onset_hours
        ):
            if onset < 0:
                continue
            T = len(probs)
            W = self.event_window

            lo = max(0, onset - W)
            hi = min(T, onset + W + 1)
            mid = min(T, onset + 1)

            if hi > lo:
                tce_all.append(float(np.abs(probs[lo:hi] - g_t[lo:hi]).mean()))

            if mid > lo:
                tce_pre.append(float(np.abs(probs[lo:mid] - g_t[lo:mid]).mean()))

            if hi > mid:
                tce_post.append(float(np.abs(probs[mid:hi] - g_t[mid:hi]).mean()))

        return {
            "tce": float(np.mean(tce_all)) if tce_all else float("nan"),
            "tce_pre_onset": float(np.mean(tce_pre)) if tce_pre else float("nan"),
            "tce_post_onset": float(np.mean(tce_post)) if tce_post else float("nan"),
            "n_patients": len(tce_all),
        }

    # ------------------------------------------------------------------
    # Platt scaling recalibration
    # ------------------------------------------------------------------

    def fit_platt(
        self, probs_val: np.ndarray, labels_val: np.ndarray
    ) -> "TrajectoryCalibration":
        """
        Fit a logistic regression on the validation set for Platt scaling.
        Call `calibrate_platt()` after fitting.
        """
        self._platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._platt.fit(probs_val.reshape(-1, 1), labels_val)
        return self

    def calibrate_platt(self, probs: np.ndarray) -> np.ndarray:
        """Apply fitted Platt scaling to new predictions."""
        if self._platt is None:
            raise RuntimeError("Call fit_platt() before calibrate_platt().")
        return self._platt.predict_proba(probs.reshape(-1, 1))[:, 1]

    # ------------------------------------------------------------------
    # Isotonic regression recalibration
    # ------------------------------------------------------------------

    def fit_isotonic(
        self, probs_val: np.ndarray, labels_val: np.ndarray
    ) -> "TrajectoryCalibration":
        """
        Fit isotonic regression for non-parametric recalibration.
        """
        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(probs_val, labels_val)
        return self

    def calibrate_isotonic(self, probs: np.ndarray) -> np.ndarray:
        """Apply fitted isotonic regression to new predictions."""
        if self._isotonic is None:
            raise RuntimeError("Call fit_isotonic() before calibrate_isotonic().")
        return self._isotonic.predict(probs)

    # ------------------------------------------------------------------
    # Aggregated calibration report
    # ------------------------------------------------------------------

    def calibration_report(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        patient_probs: Optional[List[np.ndarray]] = None,
        patient_soft_targets: Optional[List[np.ndarray]] = None,
        patient_onset_hours: Optional[List[int]] = None,
        patient_labels: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, object]:
        """
        Full calibration report combining global ECE, horizon-stratified ECE,
        and TCE (if patient-level data is provided).
        """
        report: Dict[str, object] = {}

        # Global reliability
        rel = self.reliability_curve(probs, labels)
        report["global_ece"] = rel["ece"]
        report["global_reliability"] = rel

        # Horizon-stratified (optional)
        if (
            patient_probs is not None
            and patient_labels is not None
            and patient_onset_hours is not None
        ):
            report["horizon_calibration"] = self.calibration_by_horizon(
                patient_probs, patient_labels, patient_onset_hours
            )

        # TCE (optional)
        if (
            patient_probs is not None
            and patient_soft_targets is not None
            and patient_onset_hours is not None
        ):
            report["tce"] = self.trajectory_calibration_error(
                patient_probs, patient_soft_targets, patient_onset_hours
            )

        return report
