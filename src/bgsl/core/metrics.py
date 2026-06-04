"""
bgsl/metrics.py
---------------
All evaluation metrics for BGSL-trained early-warning models.

Metric groups
-------------
1. Standard discrimination:   AUROC, AUPRC, F1, Sensitivity, Specificity, PPV, NPV
2. Early-warning utility:     PhysioNet utility score, lead time, recall @ fixed FAR
3. Trajectory quality:        ASF, FAPPD, RTV, SPJ, POMS, TCE
4. Calibration:               Brier score, ECE

All patient-level metrics are computed at the patient (ICU stay) level, not the
time-step level, to avoid inflated statistics from within-patient correlations.

Bootstrap confidence intervals are patient-level (1000 resamples, 95% CI).

Definitions (from plan §10.4):
    ASF  = alert switching frequency  (binary state changes per patient-day)
    FAPPD = false alerts per patient-day (FP alert events on non-event patient-days)
    RTV  = risk total variation         (mean Σ_t |p_t - p_{t-1}|)
    SPJ  = stable-period jitter         (mean |Δp_t| during stable negative periods)
    POMS = pre-onset monotonicity score (fraction of pre-onset intervals where p non-decreasing)
    TCE  = trajectory calibration error (mean |p_t - g_t| in ±3h event window)

Threshold selection (plan §10.5):
    Thresholds selected on validation set; NOT on test set.
    Methods: fixed sensitivity (80%), fixed FAPPD, maximum utility.

Reference: BGSL plan.md §10
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

__all__ = ["BGSLMetrics", "PatientPrediction"]


# ---------------------------------------------------------------------------
# Data container for a single patient's predictions
# ---------------------------------------------------------------------------

@dataclass
class PatientPrediction:
    """
    Stores all prediction data for one patient (ICU stay).

    Attributes
    ----------
    patient_id : str
    probs : np.ndarray [T]
        Predicted risk probabilities at each hour.
    hard_labels : np.ndarray [T]
        Binary ground-truth labels (1 inside warning window, 0 outside).
    soft_targets : np.ndarray [T]
        Soft BGSL target g_t (for TCE).
    is_sepsis : bool
        Whether this patient develops sepsis.
    onset_hour : int
        Hour of sepsis onset (t*). -1 if negative patient.
    horizon : int
        Prediction horizon H in hours.
    seq_len : int
        Number of valid time steps.
    """
    patient_id: str
    probs: np.ndarray
    hard_labels: np.ndarray
    soft_targets: np.ndarray
    is_sepsis: bool
    onset_hour: int  # -1 for negative patients
    horizon: int     # H
    seq_len: int


# ---------------------------------------------------------------------------
# Main metrics class
# ---------------------------------------------------------------------------

class BGSLMetrics:
    """
    Computes all BGSL evaluation metrics over a cohort of patient predictions.

    Usage
    -----
    >>> metrics = BGSLMetrics(threshold=0.5, sustained_k=1)
    >>> for patient in predictions:
    ...     metrics.add(patient)
    >>> results = metrics.compute()
    >>> ci = metrics.bootstrap_ci(n_resamples=1000)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sustained_k: int = 1,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        event_window_hours: int = 3,  # for TCE: ±3h around onset
    ) -> None:
        """
        Parameters
        ----------
        threshold : float
            Alert threshold for binary metrics.
        sustained_k : int
            Number of consecutive hours above threshold before alert fires.
            K=1 means instant alert; K=2 requires two consecutive hours.
        n_bootstrap : int
            Number of bootstrap resamples for confidence intervals.
        ci_level : float
            Confidence interval level (default 95%).
        event_window_hours : int
            Hours around onset for TCE computation.
        """
        self.threshold = threshold
        self.sustained_k = sustained_k
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.event_window = event_window_hours
        self._patients: List[PatientPrediction] = []

    def add(self, patient: PatientPrediction) -> None:
        """Add a patient prediction to the cohort."""
        self._patients.append(patient)

    def reset(self) -> None:
        """Clear all stored predictions."""
        self._patients.clear()

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics over the current patient cohort.

        Returns
        -------
        dict mapping metric name → scalar value.
        """
        if len(self._patients) == 0:
            raise RuntimeError("No patients added. Call .add() first.")

        results: Dict[str, float] = {}

        # --- Build timestep-level arrays (for AUROC, AUPRC, Brier) ---
        all_probs = np.concatenate([p.probs for p in self._patients])
        all_labels = np.concatenate([p.hard_labels for p in self._patients])

        # --- Standard discrimination ---
        results.update(self._discrimination(all_probs, all_labels))

        # --- Calibration ---
        results["brier_score"] = float(brier_score_loss(all_labels, all_probs))
        results["ece"] = self._ece(all_probs, all_labels)

        # --- PhysioNet utility ---
        results["physionet_utility"] = self._physionet_utility()

        # --- Lead time ---
        lead_times = self._lead_times()
        if lead_times:
            results["median_lead_time_h"] = float(np.median(lead_times))
            results["mean_lead_time_h"] = float(np.mean(lead_times))
            results["lead_time_iqr_low"] = float(np.percentile(lead_times, 25))
            results["lead_time_iqr_high"] = float(np.percentile(lead_times, 75))
        else:
            results["median_lead_time_h"] = float("nan")
            results["mean_lead_time_h"] = float("nan")

        # --- Trajectory quality ---
        results.update(self._trajectory_metrics())

        return results

    # ------------------------------------------------------------------
    # Discrimination
    # ------------------------------------------------------------------

    def _discrimination(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}

        if labels.max() == 0 or labels.min() == 1:
            warnings.warn("All labels are the same class; AUROC/AUPRC undefined.")
            out["auroc"] = float("nan")
            out["auprc"] = float("nan")
        else:
            out["auroc"] = float(roc_auc_score(labels, probs))
            out["auprc"] = float(average_precision_score(labels, probs))

        # Binary metrics at threshold
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

    # ------------------------------------------------------------------
    # Expected Calibration Error
    # ------------------------------------------------------------------

    def _ece(
        self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> float:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(probs)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = (probs >= lo) & (probs < hi)
            n_bin = in_bin.sum()
            if n_bin == 0:
                continue
            mean_conf = probs[in_bin].mean()
            mean_acc = labels[in_bin].mean()
            ece += (n_bin / n) * abs(mean_conf - mean_acc)
        return float(ece)

    # ------------------------------------------------------------------
    # PhysioNet 2019 Utility Score
    # ------------------------------------------------------------------

    def _physionet_utility(self) -> float:
        """
        Approximation of the PhysioNet 2019 challenge utility score.

        U_optimal = Σ_i [U_tp_i - U_fn_i - U_fp_i]

        For a binary alert series per patient, the utility is computed
        according to the official scoring function:
          - True positive within horizon: +1 (decaying with lead time)
          - False positive: -0.05
          - False negative (missed alert): -2
          - True negative: 0

        Reference: Reyna et al., Critical Care Medicine 2020.
        """
        total_utility = 0.0
        for p in self._patients:
            alert = self._sustained_alert(p.probs)
            util = self._patient_utility(alert, p.hard_labels, p.onset_hour, p.horizon)
            total_utility += util

        n = len(self._patients)
        return float(total_utility / max(n, 1))

    @staticmethod
    def _patient_utility(
        alert: np.ndarray,
        hard_labels: np.ndarray,
        onset_hour: int,
        horizon: int,
    ) -> float:
        """
        Simplified per-patient PhysioNet utility.
        """
        T = len(hard_labels)

        if onset_hour < 0:
            # Negative patient: any alert is a false positive
            n_fp = alert.sum()
            return float(-0.05 * n_fp)

        # Positive patient
        first_alert = int(np.argmax(alert)) if alert.any() else T

        # Lead time in hours before onset
        lead = onset_hour - first_alert

        if alert.any() and lead >= 0:
            # True positive — utility depends on lead time
            # Official: normalized_utility = max(0, lead) / horizon
            utility = max(0.0, lead) / max(horizon, 1)
            return float(utility)
        elif not alert.any():
            # Missed — false negative penalty
            return -2.0
        else:
            # Alert after onset — still counts as TP but zero lead
            return 0.0

    # ------------------------------------------------------------------
    # Lead time
    # ------------------------------------------------------------------

    def _lead_times(self) -> List[float]:
        """
        Lead time = hours between first true alert and onset.
        Only computed for positive patients that received an alert.
        """
        lead_times = []
        for p in self._patients:
            if not p.is_sepsis:
                continue
            alert = self._sustained_alert(p.probs)
            if not alert.any():
                continue
            first_alert = int(np.argmax(alert))
            lead = p.onset_hour - first_alert
            if lead >= 0:
                lead_times.append(float(lead))
        return lead_times

    # ------------------------------------------------------------------
    # Trajectory quality metrics
    # ------------------------------------------------------------------

    def _trajectory_metrics(self) -> Dict[str, float]:
        asf_vals, rtv_vals, spj_vals, poms_vals, tce_vals = [], [], [], [], []
        fp_alert_events, neg_patient_days = 0, 0.0

        for p in self._patients:
            alert = self._sustained_alert(p.probs)
            probs = p.probs
            T = p.seq_len
            patient_days = T / 24.0

            # Alert Switching Frequency
            if T > 1:
                switches = int(np.abs(np.diff(alert.astype(int))).sum())
                asf_vals.append(switches / max(patient_days, 1e-9))

            # Risk Total Variation
            if T > 1:
                rtv = float(np.abs(np.diff(probs)).sum())
                rtv_vals.append(rtv)

            # Stable-Period Jitter (mean |Δp_t| during stable negative periods)
            # "Stable negative": hard_label == 0 and time > onset+H or entirely negative
            if T > 1:
                is_stable = (p.hard_labels < 0.5)
                if is_stable.sum() > 1:
                    diffs = np.abs(np.diff(probs))
                    stable_mask = is_stable[1:]  # align with diff
                    if stable_mask.sum() > 0:
                        spj_vals.append(float(diffs[stable_mask].mean()))

            # Pre-Onset Monotonicity Score
            if p.is_sepsis and p.onset_hour >= 0:
                window_start = max(0, p.onset_hour - p.horizon)
                window_end = min(T - 1, p.onset_hour)
                if window_end > window_start:
                    pre_onset = probs[window_start:window_end + 1]
                    non_decreasing = int((np.diff(pre_onset) >= 0).sum())
                    total_intervals = len(pre_onset) - 1
                    if total_intervals > 0:
                        poms_vals.append(non_decreasing / total_intervals)

            # Trajectory Calibration Error
            if p.soft_targets is not None and p.is_sepsis and p.onset_hour >= 0:
                lo = max(0, p.onset_hour - self.event_window)
                hi = min(T, p.onset_hour + self.event_window + 1)
                if hi > lo:
                    tce = float(np.abs(probs[lo:hi] - p.soft_targets[lo:hi]).mean())
                    tce_vals.append(tce)

            # False Alerts Per Patient-Day (for negative patients)
            if not p.is_sepsis:
                neg_patient_days += patient_days
                # Count alert events (rising edges)
                rising = np.diff(alert.astype(int), prepend=0)
                fp_alert_events += int((rising == 1).sum())

        results: Dict[str, float] = {}
        results["asf"] = float(np.mean(asf_vals)) if asf_vals else float("nan")
        results["rtv"] = float(np.mean(rtv_vals)) if rtv_vals else float("nan")
        results["spj"] = float(np.mean(spj_vals)) if spj_vals else float("nan")
        results["poms"] = float(np.mean(poms_vals)) if poms_vals else float("nan")
        results["tce"] = float(np.mean(tce_vals)) if tce_vals else float("nan")
        results["fappd"] = (
            float(fp_alert_events / neg_patient_days)
            if neg_patient_days > 0
            else float("nan")
        )

        return results

    # ------------------------------------------------------------------
    # Sustained alert helper
    # ------------------------------------------------------------------

    def _sustained_alert(self, probs: np.ndarray) -> np.ndarray:
        """
        Fire alert at time t if probs[t:t+K] are all >= threshold.
        K = self.sustained_k.

        Returns binary alert array of same length as probs.
        """
        T = len(probs)
        alert = np.zeros(T, dtype=bool)
        K = self.sustained_k

        if K <= 1:
            alert = probs >= self.threshold
            return alert

        for t in range(T - K + 1):
            if np.all(probs[t : t + K] >= self.threshold):
                alert[t] = True  # mark first crossing

        return alert

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self, n_resamples: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Patient-level bootstrap confidence intervals.

        Returns
        -------
        dict mapping metric_name → {"mean": ..., "lo": ..., "hi": ...}
        """
        n = n_resamples or self.n_bootstrap
        alpha = (1.0 - self.ci_level) / 2.0

        # Collect point estimates from each bootstrap resample
        all_results: Dict[str, List[float]] = {}

        rng = np.random.default_rng(seed=42)
        patients = self._patients

        for _ in range(n):
            idx = rng.integers(0, len(patients), size=len(patients))
            boot = BGSLMetrics(
                threshold=self.threshold,
                sustained_k=self.sustained_k,
                event_window_hours=self.event_window,
            )
            for i in idx:
                boot.add(patients[i])

            try:
                res = boot.compute()
            except Exception:
                continue

            for k, v in res.items():
                if not np.isnan(v):
                    all_results.setdefault(k, []).append(v)

        ci: Dict[str, Dict[str, float]] = {}
        for k, vals in all_results.items():
            arr = np.array(vals)
            ci[k] = {
                "mean": float(arr.mean()),
                "lo": float(np.percentile(arr, 100 * alpha)),
                "hi": float(np.percentile(arr, 100 * (1.0 - alpha))),
                "std": float(arr.std()),
            }
        return ci

    # ------------------------------------------------------------------
    # Threshold selection (validation set only)
    # ------------------------------------------------------------------

    def select_threshold_fixed_sensitivity(
        self, target_sensitivity: float = 0.80
    ) -> float:
        """
        Select the highest threshold that achieves at least `target_sensitivity`
        on the current cohort (should be validation set).
        """
        all_probs = np.concatenate([p.probs for p in self._patients])
        all_labels = np.concatenate([p.hard_labels for p in self._patients])
        thresholds = np.linspace(0.0, 1.0, 201)

        best_thresh = 0.5
        for t in sorted(thresholds, reverse=True):
            preds = (all_probs >= t).astype(int)
            tp = ((preds == 1) & (all_labels == 1)).sum()
            fn = ((preds == 0) & (all_labels == 1)).sum()
            sens = tp / max(tp + fn, 1e-9)
            if sens >= target_sensitivity:
                best_thresh = float(t)
                break

        return best_thresh

    def select_threshold_fixed_fappd(
        self, target_fappd: float = 1.0
    ) -> float:
        """
        Select threshold that keeps FAPPD <= target on the current cohort.
        Picks the highest threshold satisfying the constraint.
        """
        thresholds = np.linspace(0.0, 1.0, 201)
        best_thresh = 1.0

        for t in sorted(thresholds):
            self.threshold = t
            metrics = self._trajectory_metrics()
            fappd = metrics.get("fappd", float("inf"))
            if fappd <= target_fappd or np.isnan(fappd):
                best_thresh = float(t)
                break

        self.threshold = best_thresh
        return best_thresh

    def select_threshold_max_utility(self) -> float:
        """
        Select threshold that maximizes the PhysioNet utility score
        on the current cohort.
        """
        thresholds = np.linspace(0.0, 1.0, 201)
        best_thresh = 0.5
        best_util = -float("inf")

        for t in thresholds:
            self.threshold = t
            util = self._physionet_utility()
            if util > best_util:
                best_util = util
                best_thresh = float(t)

        self.threshold = best_thresh
        return best_thresh

    # ------------------------------------------------------------------
    # Recall at fixed false alarm rates
    # ------------------------------------------------------------------

    def recall_at_fixed_far(
        self, false_alarms_per_day: float = 0.5
    ) -> float:
        """
        Sensitivity at a fixed false alerts per patient-day operating point.
        Threshold is selected to match the target FAPPD on the current cohort.
        """
        t = self.select_threshold_fixed_fappd(target_fappd=false_alarms_per_day)
        self.threshold = t

        tp, fn = 0, 0
        for p in self._patients:
            if not p.is_sepsis:
                continue
            alert = self._sustained_alert(p.probs)
            # True positive: alert fires within the warning window
            window = p.hard_labels.astype(bool)
            if (alert & window).any():
                tp += 1
            else:
                fn += 1

        return float(tp / max(tp + fn, 1e-9))

    # ------------------------------------------------------------------
    # Summary string
    # ------------------------------------------------------------------

    def summary(self) -> str:
        res = self.compute()
        lines = [
            "=" * 60,
            "BGSL Evaluation Summary",
            "=" * 60,
            f"  Patients: {len(self._patients)}",
            f"  Threshold: {self.threshold:.3f} (K={self.sustained_k})",
            "",
            "  Discrimination:",
            f"    AUROC:  {res.get('auroc', float('nan')):.4f}",
            f"    AUPRC:  {res.get('auprc', float('nan')):.4f}",
            f"    F1:     {res.get('f1', float('nan')):.4f}",
            f"    Sens:   {res.get('sensitivity', float('nan')):.4f}",
            f"    Spec:   {res.get('specificity', float('nan')):.4f}",
            "",
            "  Calibration:",
            f"    Brier:  {res.get('brier_score', float('nan')):.4f}",
            f"    ECE:    {res.get('ece', float('nan')):.4f}",
            "",
            "  Utility & Lead Time:",
            f"    Utility:        {res.get('physionet_utility', float('nan')):.4f}",
            f"    Median Lead:    {res.get('median_lead_time_h', float('nan')):.1f}h",
            "",
            "  Trajectory Quality:",
            f"    ASF:   {res.get('asf', float('nan')):.4f}  (switches/pt-day)",
            f"    FAPPD: {res.get('fappd', float('nan')):.4f}  (false alerts/pt-day)",
            f"    RTV:   {res.get('rtv', float('nan')):.4f}  (total variation)",
            f"    SPJ:   {res.get('spj', float('nan')):.4f}  (stable jitter)",
            f"    POMS:  {res.get('poms', float('nan')):.4f}  (pre-onset monotonicity)",
            f"    TCE:   {res.get('tce', float('nan')):.4f}  (trajectory calib error)",
            "=" * 60,
        ]
        return "\n".join(lines)
