"""
bgsl/core/common/calibration.py
-------------------------------
Generic calibration analysis for BGSL risk trajectories.
Agnostic to domain (Sepsis, CMAPSS, etc). Domain-specific wrappers
can provide appropriate time-to-onset buckets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = ["BaseTrajectoryCalibration"]


class BaseTrajectoryCalibration:
    """
    Calibration analysis for BGSL early-warning models.
    """

    def __init__(
        self,
        n_bins: int = 10,
        event_window: float = 3.0,
    ) -> None:
        self.n_bins = n_bins
        self.event_window = event_window
        self._platt: Optional[LogisticRegression] = None
        self._isotonic: Optional[IsotonicRegression] = None

    def reliability_curve(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        frac_pos, mean_pred = calibration_curve(
            labels, probs, n_bins=self.n_bins, strategy="uniform"
        )

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

    def calibration_by_horizon(
        self,
        patient_probs: List[np.ndarray],
        patient_labels: List[np.ndarray],
        patient_onset_times: List[float],
        buckets: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Dict]:
        """
        Compute calibration within time-to-onset buckets.
        """
        bucket_probs: Dict[str, List[float]] = {k: [] for k in buckets}
        bucket_labels: Dict[str, List[float]] = {k: [] for k in buckets}

        for probs, labels, onset in zip(
            patient_probs, patient_labels, patient_onset_times
        ):
            if onset < 0:
                continue

            T = len(probs)
            for t in range(T):
                time_to_onset = onset - t
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

    def trajectory_calibration_error(
        self,
        patient_probs: List[np.ndarray],
        patient_soft_targets: List[np.ndarray],
        patient_onset_times: List[float],
    ) -> Dict[str, float]:
        tce_all, tce_pre, tce_post = [], [], []

        for probs, g_t, onset in zip(
            patient_probs, patient_soft_targets, patient_onset_times
        ):
            if onset < 0:
                continue
            T = len(probs)
            W = self.event_window

            lo = int(max(0, onset - W))
            hi = int(min(T, onset + W + 1))
            mid = int(min(T, onset + 1))

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

    def fit_platt(self, probs_val: np.ndarray, labels_val: np.ndarray) -> "BaseTrajectoryCalibration":
        self._platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self._platt.fit(probs_val.reshape(-1, 1), labels_val)
        return self

    def calibrate_platt(self, probs: np.ndarray) -> np.ndarray:
        if self._platt is None:
            raise RuntimeError("Call fit_platt() before calibrate_platt().")
        return self._platt.predict_proba(probs.reshape(-1, 1))[:, 1]

    def fit_isotonic(self, probs_val: np.ndarray, labels_val: np.ndarray) -> "BaseTrajectoryCalibration":
        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(probs_val, labels_val)
        return self

    def calibrate_isotonic(self, probs: np.ndarray) -> np.ndarray:
        if self._isotonic is None:
            raise RuntimeError("Call fit_isotonic() before calibrate_isotonic().")
        return self._isotonic.predict(probs)

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
        report: Dict[str, object] = {}
        rel = self.reliability_curve(probs, labels)
        report["global_ece"] = rel["ece"]
        report["global_reliability"] = rel

        if (patient_probs is not None and patient_labels is not None and 
            patient_onset_times is not None and buckets is not None):
            report["horizon_calibration"] = self.calibration_by_horizon(
                patient_probs, patient_labels, patient_onset_times, buckets
            )

        if (patient_probs is not None and patient_soft_targets is not None and 
            patient_onset_times is not None):
            report["tce"] = self.trajectory_calibration_error(
                patient_probs, patient_soft_targets, patient_onset_times
            )

        return report
