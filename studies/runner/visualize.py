"""
studies/runner/visualize.py
---------------------------
Loads saved test predictions from a completed batch and generates
trajectory plots for representative patients per condition.

Usage
-----
    python -m studies.runner.visualize <batch_dir> [--condition COND] [--seeds S1 S2 ...]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _find_prediction_files(batch_dir: Path, cond_name: Optional[str] = None,
                           seed_filter: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    manifest_file = batch_dir / "batch_manifest.json"
    if not manifest_file.exists():
        print(f"ERROR: No manifest at {manifest_file}")
        sys.exit(1)

    with open(manifest_file) as f:
        manifest = json.load(f)

    results: List[Dict[str, Any]] = []
    for run in manifest.get("runs", []):
        run_dir = Path(run["run_dir"])
        cname = run.get("cond_name", "unknown")
        seed = run.get("seed", -1)

        if cond_name is not None and cname != cond_name:
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue

        # Find prediction pickle under artifacts/
        pred_files = list(run_dir.glob("artifacts/predictions/test_predictions.pkl"))
        if not pred_files:
            continue

        results.append({
            "cond_name": cname,
            "seed": seed,
            "run_dir": run_dir,
            "predictions_pkl": pred_files[0],
        })

    return results


def _select_patients(predictions: list, threshold: float) -> Dict[str, list]:
    sep_patients = [p for p in predictions if p.has_event]
    non_sep_patients = [p for p in predictions if not p.has_event]

    if not sep_patients:
        sep_selected = []
    else:
        # Compute lead times
        lead_data = []
        for p in sep_patients:
            alert = p.probs >= threshold
            if alert.any():
                first_alert = int(np.argmax(alert))
                lead = p.onset_time - first_alert
                if lead >= 0:
                    lead_data.append((lead, p))
                else:
                    lead_data.append((float("-inf"), p))
            else:
                lead_data.append((float("-inf"), p))

        lead_data.sort(key=lambda x: x[0], reverse=True)
        n = len(lead_data)
        best = lead_data[0][1]
        median = lead_data[n // 2][1]
        worst = lead_data[-1][1]
        sep_selected = [best, median, worst]

    if not non_sep_patients:
        non_sep_selected = []
    else:
        # Sort non-sepsis by sustained alert frequency
        alert_counts = []
        for p in non_sep_patients:
            alert = p.probs >= threshold
            n_alerts = int(alert.sum())
            alert_counts.append((n_alerts, p))
        alert_counts.sort(key=lambda x: x[0])
        n = len(alert_counts)
        low_fp = alert_counts[0][1]
        high_fp = alert_counts[-1][1]
        flat = next((p for _, p in alert_counts if _ == 0), low_fp)
        non_sep_selected = [low_fp, high_fp, flat]

    return {"sepsis": sep_selected, "non_sepsis": non_sep_selected}


def _plot_patient(patient, threshold: float, cond_name: str, seed: int,
                  save_dir: Path) -> None:
    if not HAS_MPL:
        return

    T = len(patient.probs)
    time_axis = np.arange(T) * getattr(patient, "time_units_per_step", 1.0)

    sustained = np.zeros(T, dtype=bool)
    if T >= 1:
        sustained = patient.probs >= threshold

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, patient.probs, "b-", linewidth=1.5, label="Risk score $p_t$")
    if patient.soft_targets is not None:
        ax.plot(time_axis, patient.soft_targets[:T], "g--", linewidth=1.0,
                alpha=0.6, label="Soft target $g_t$")
    ax.axhline(y=threshold, color="r", linestyle=":", linewidth=0.8,
               label=f"Threshold = {threshold:.3f}")

    # Shade sustained alert regions
    if sustained.any():
        transitions = np.diff(np.concatenate([[0], sustained.astype(int), [0]]))
        ons = np.where(transitions == 1)[0]
        offs = np.where(transitions == -1)[0]
        for on, off in zip(ons, offs):
            ax.axvspan(on, off, alpha=0.12, color="red")

    # Sepsis onset
    if patient.has_event and getattr(patient, "onset_time", -1) >= 0:
        onset = getattr(patient, "onset_time", -1)
        ax.axvline(x=onset, color="red", linewidth=1.5, linestyle="--",
                   label=f"Onset t={onset:.0f}")
        # Compute lead time
        alert = patient.probs >= threshold
        if alert.any():
            first_alert = int(np.argmax(alert))
            lead = onset - first_alert
            ax.annotate(f"Lead={lead:.1f}", xy=(first_alert, 0.95),
                        fontsize=9, color="darkred",
                        arrowprops=dict(arrowstyle="->", color="darkred"))

    ax.set_xlabel("Time (hours)" if hasattr(patient, "time_units_per_step") else "Time step")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{cond_name} (seed={seed}) — Patient {getattr(patient, 'id', '?')} "
                 f"{'[SEPSIS]' if patient.has_event else '[NON-SEPSIS]'}")
    ax.grid(True, alpha=0.3)

    label = "sepsis" if patient.has_event else "non_sepsis"
    safe_id = str(getattr(patient, "id", "unknown")).replace("/", "_")
    out = save_dir / f"{cond_name}_seed{seed}_{label}_{safe_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trajectory plots from test predictions.")
    parser.add_argument("batch_dir", type=str, help="Batch output directory")
    parser.add_argument("--condition", type=str, default=None,
                        help="Only plot this condition (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Only plot these seeds (default: all)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: <batch_dir>/plots)")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib is required for visualization.")
        print("       Install with: pip install matplotlib")
        sys.exit(1)

    batch_dir = Path(args.batch_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else batch_dir / "plots"

    pred_sources = _find_prediction_files(batch_dir, args.condition, args.seeds)
    if not pred_sources:
        print("No prediction files found. Run the test pipeline first.")
        sys.exit(1)

    for src in pred_sources:
        print(f"[{src['cond_name']}] seed={src['seed']}: loading predictions...")
        with open(src["predictions_pkl"], "rb") as f:
            predictions = pickle.load(f)

        cond_run_dir = src["run_dir"]

        # Try to read threshold from final_test_metrics.json
        threshold = 0.5
        metrics_file = cond_run_dir / "metrics" / "final_test_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            threshold = metrics.get("test_selected_threshold",
                                    metrics.get("selected_threshold", 0.5))

        selected = _select_patients(predictions, threshold)
        all_selected = selected["sepsis"] + selected["non_sepsis"]

        save_dir = out_dir / src["cond_name"]
        for patient in all_selected:
            _plot_patient(patient, threshold, src["cond_name"], src["seed"], save_dir)

        print(f"  -> {len(all_selected)} plots saved to {save_dir}")

    print(f"\nAll plots saved under {out_dir}")


if __name__ == "__main__":
    main()
