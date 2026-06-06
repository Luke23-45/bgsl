"""
bgsl/core/common/metrics.py
---------------------------
Generic evaluation metrics for BGSL-trained early-warning models.
Contains discrimination, calibration, and trajectory quality metrics
that are agnostic to the specific domain (Sepsis, CMAPSS, etc.).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

__all__ = ["BaseSequencePrediction", "BaseTrajectoryMetrics"]


@dataclass
class BaseSequencePrediction:
    """
    Generic data container for a single sequence prediction.
    Domain-specific wrappers (like PatientPrediction) can inherit from this.
    """
    id: str
    probs: np.ndarray
    hard_labels: np.ndarray
    soft_targets: Optional[np.ndarray]
    has_event: bool
    onset_time: float  # -1 if no event
    horizon: float     # warning window size H
    seq_len: int
    time_units_per_step: float = 1.0  # e.g., hours per step


class BaseTrajectoryMetrics:
    """
    Computes generic BGSL evaluation metrics over a cohort of predictions.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sustained_k: int = 1,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        event_window: float = 3.0,
    ) -> None:
        self.threshold = threshold
        self.sustained_k = sustained_k
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.event_window = event_window
        self._predictions: List[BaseSequencePrediction] = []

    def add(self, prediction: BaseSequencePrediction) -> None:
        self._predictions.append(prediction)

    def reset(self) -> None:
        self._predictions.clear()

    def select_threshold_fixed_sensitivity(
        self,
        target_sensitivity: float = 0.80,
        n_thresholds: int = 200,
    ) -> float:
        """
        Return the highest threshold that achieves at least *target_sensitivity*
        over all accumulated predictions.

        Sweeps *n_thresholds* evenly-spaced candidate thresholds from 0→1,
        evaluates sensitivity at each, and returns the largest threshold where
        sensitivity ≥ target_sensitivity.  Falls back to 0.5 if the target
        cannot be met (e.g. no positive labels collected yet).

        Parameters
        ----------
        target_sensitivity : float
            Minimum desired recall (TP / (TP+FN)). Default 0.80.
        n_thresholds : int
            Number of candidate thresholds to sweep. Default 200.
        """
        if not self._predictions:
            return 0.5

        all_probs  = np.concatenate([p.probs       for p in self._predictions])
        all_labels = np.concatenate([p.hard_labels for p in self._predictions])

        n_pos = (all_labels > 0.5).sum()
        if n_pos == 0:
            return 0.5  # no positives → threshold is meaningless

        best_threshold = 0.5
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        for thr in reversed(thresholds):      # high → low: keep highest that works
            preds = (all_probs >= thr).astype(int)
            tp = float(((preds == 1) & (all_labels > 0.5)).sum())
            fn = float(((preds == 0) & (all_labels > 0.5)).sum())
            sensitivity = tp / max(tp + fn, 1e-9)
            if sensitivity >= target_sensitivity:
                best_threshold = float(thr)
                break

        return best_threshold

    def compute(self) -> Dict[str, float]:
        if len(self._predictions) == 0:
            raise RuntimeError("No predictions added. Call .add() first.")

        results: Dict[str, float] = {}

        all_probs = np.concatenate([p.probs for p in self._predictions])
        all_labels = np.concatenate([p.hard_labels for p in self._predictions])

        results.update(self._discrimination(all_probs, all_labels))
        results["brier_score"] = float(brier_score_loss(all_labels, all_probs))
        results["ece"] = self._ece(all_probs, all_labels)
        
        # Lead times
        lead_times = self._lead_times()
        if lead_times:
            results["median_lead_time"] = float(np.median(lead_times))
            results["mean_lead_time"] = float(np.mean(lead_times))
            results["lead_time_iqr_low"] = float(np.percentile(lead_times, 25))
            results["lead_time_iqr_high"] = float(np.percentile(lead_times, 75))
        else:
            results["median_lead_time"] = float("nan")
            results["mean_lead_time"] = float("nan")

        results.update(self._trajectory_metrics())
        return results

    def _discrimination(self, probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if labels.max() == 0 or labels.min() == 1:
            warnings.warn("All labels are the same class; AUROC/AUPRC undefined.")
            out["auroc"] = float("nan")
            out["auprc"] = float("nan")
        else:
            out["auroc"] = float(roc_auc_score(labels, probs))
            out["auprc"] = float(average_precision_score(labels, probs))

        preds = (probs >= self.threshold).astype(int)
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        tn = float(((preds == 0) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())

        out["sensitivity"] = tp / max(tp + fn, 1e-9)
        out["specificity"] = tn / max(tn + fp, 1e-9)
        out["ppv"] = tp / max(tp + fp, 1e-9)
        out["npv"] = tn / max(tn + fn, 1e-9)
        out["f1"] = float(f1_score(labels, preds, zero_division=0))
        return out

    def _ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(probs)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = (probs >= lo) & (probs < hi)
            n_bin = in_bin.sum()
            if n_bin == 0: continue
            mean_conf = probs[in_bin].mean()
            mean_acc = labels[in_bin].mean()
            ece += (n_bin / n) * abs(mean_conf - mean_acc)
        return float(ece)

    def _lead_times(self) -> List[float]:
        lead_times = []
        for p in self._predictions:
            if not p.has_event:
                continue
            alert = self._sustained_alert(p.probs)
            if not alert.any():
                continue
            first_alert = int(np.argmax(alert))
            lead = p.onset_time - first_alert
            if lead >= 0:
                lead_times.append(float(lead * p.time_units_per_step))
        return lead_times

    def _trajectory_metrics(self) -> Dict[str, float]:
        asf_vals, rtv_vals, spj_vals, poms_vals, tce_vals = [], [], [], [], []
        fp_alert_events, neg_time_units = 0, 0.0

        for p in self._predictions:
            alert = self._sustained_alert(p.probs)
            probs = p.probs
            T = p.seq_len
            total_time = T * p.time_units_per_step

            # Alert Switching Frequency
            if T > 1:
                switches = int(np.abs(np.diff(alert.astype(int))).sum())
                asf_vals.append(switches / max(total_time, 1e-9))

            # Risk Total Variation
            if T > 1:
                rtv = float(np.abs(np.diff(probs)).sum())
                rtv_vals.append(rtv)

            # Stable-Period Jitter
            if T > 1:
                is_stable = (p.hard_labels < 0.5)
                if is_stable.sum() > 1:
                    diffs = np.abs(np.diff(probs))
                    stable_mask = is_stable[1:]
                    if stable_mask.sum() > 0:
                        spj_vals.append(float(diffs[stable_mask].mean()))

            # Pre-Onset Monotonicity Score
            if p.has_event and p.onset_time >= 0:
                window_start = int(max(0, p.onset_time - p.horizon))
                window_end = int(min(T - 1, p.onset_time))
                if window_end > window_start:
                    pre_onset = probs[window_start:window_end + 1]
                    non_decreasing = int((np.diff(pre_onset) >= 0).sum())
                    total_intervals = len(pre_onset) - 1
                    if total_intervals > 0:
                        poms_vals.append(non_decreasing / total_intervals)

            # Trajectory Calibration Error
            if p.soft_targets is not None and p.has_event and p.onset_time >= 0:
                lo = int(max(0, p.onset_time - self.event_window))
                hi = int(min(T, p.onset_time + self.event_window + 1))
                if hi > lo:
                    tce = float(np.abs(probs[lo:hi] - p.soft_targets[lo:hi]).mean())
                    tce_vals.append(tce)

            # False Alerts Per Time-Unit
            if not p.has_event:
                neg_time_units += total_time
                rising = np.diff(alert.astype(int), prepend=0)
                fp_alert_events += int((rising == 1).sum())

        return {
            "asf": float(np.mean(asf_vals)) if asf_vals else float("nan"),
            "rtv": float(np.mean(rtv_vals)) if rtv_vals else float("nan"),
            "spj": float(np.mean(spj_vals)) if spj_vals else float("nan"),
            "poms": float(np.mean(poms_vals)) if poms_vals else float("nan"),
            "tce": float(np.mean(tce_vals)) if tce_vals else float("nan"),
            "fa_rate": float(fp_alert_events / neg_time_units) if neg_time_units > 0 else float("nan")
        }

    def _sustained_alert(self, probs: np.ndarray) -> np.ndarray:
        T = len(probs)
        alert = np.zeros(T, dtype=bool)
        K = self.sustained_k
        if K <= 1:
            return probs >= self.threshold
        for t in range(T - K + 1):
            if np.all(probs[t : t + K] >= self.threshold):
                alert[t] = True
        return alert
