"""
studies/runner/aggregate.py
---------------------------
Aggregates per-run ``final_test_metrics.json`` files across seeds within
each condition and produces a formatted comparison table.

Usage
-----
    python -m studies.runner.aggregate <batch_dir> [--csv <path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

KEY_METRICS = [
    "test_auroc",
    "test_auprc",
    "test_physionet_utility",
    "test_median_lead_time",
    "test_mean_lead_time",
    "test_asf",
    "test_rtv",
    "test_spj",
    "test_poms",
    "test_tce",
    "test_fa_rate",
    "test_fappd",
    "test_sensitivity",
    "test_specificity",
    "test_ppv",
    "test_npv",
    "test_f1",
    "test_brier_score",
    "test_ece",
    "test_stability_score",
    "test_flip_count",
    "test_selected_threshold",
]


def load_run_metrics(run_dir: Path) -> Dict[str, float]:
    metrics_file = run_dir / "metrics" / "final_test_metrics.json"
    if not metrics_file.exists():
        return {}
    with open(metrics_file) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-run test metrics across conditions.")
    parser.add_argument("batch_dir", type=str, help="Path to the batch output directory")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path (default: <batch_dir>/aggregated_results.csv)")
    args = parser.parse_args()

    batch_root = Path(args.batch_dir).resolve()
    manifest_file = batch_root / "batch_manifest.json"
    if not manifest_file.exists():
        print(f"ERROR: No manifest found at {manifest_file}")
        sys.exit(1)

    with open(manifest_file) as f:
        manifest = json.load(f)

    runs = manifest.get("runs", [])
    if not runs:
        print("No runs found in manifest.")
        return

    records: List[Dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        metrics = load_run_metrics(run_dir)
        if not metrics:
            continue
        records.append({
            "cond_name": run.get("cond_name", "unknown"),
            "seed": run.get("seed", -1),
            **metrics,
        })

    if not records:
        print("No metrics found for any run.")
        return

    df = pd.DataFrame(records)
    cond_groups = df.groupby("cond_name")

    all_metric_keys = sorted(set().union(*(r.keys() - {"cond_name", "seed"} for r in records)))

    summary_rows: List[Dict[str, Any]] = []
    for cond_name, group in cond_groups:
        row: Dict[str, Any] = {"condition": cond_name, "n_seeds": len(group)}
        for metric in all_metric_keys:
            values = group[metric].dropna()
            if len(values) > 0:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std()
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Print header
    print(f"\n{'=' * 72}")
    print(f"  AGGREGATED RESULTS — {batch_root.name}")
    print(f"  Conditions: {len(summary_rows)} | Total runs: {len(records)}")
    print(f"{'=' * 72}")

    for _, row in summary_df.iterrows():
        print(f"\n  --- {row['condition']} (n={row['n_seeds']}) ---")
        for metric in KEY_METRICS:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in row and pd.notna(row[mean_key]):
                mean_val = row[mean_key]
                std_val = row[std_key] if std_key in row and pd.notna(row[std_key]) else 0.0
                print(f"    {metric:42s}= {mean_val:.4f} ± {std_val:.4f}")

    out_path = Path(args.csv) if args.csv else batch_root / "aggregated_results.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n  Full results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
