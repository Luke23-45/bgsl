"""
studies/runner/aggregate.py
---------------------------
Aggregates per-run ``final_test_metrics.json`` files across seeds within
each condition and produces a formatted comparison table, including a
publication-ready LaTeX table with 95% Confidence Intervals.

Usage
-----
    python -m studies.runner.aggregate <batch_dir> [--csv <path>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

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
]

# Metrics where HIGHER is better
HIGHER_IS_BETTER = {
    "test_auroc", "test_auprc", "test_physionet_utility", 
    "test_median_lead_time", "test_mean_lead_time", "test_poms",
    "test_sensitivity", "test_specificity", "test_ppv", "test_npv", "test_f1"
}

# Metrics to include in the LaTeX table
LATEX_METRICS = {
    "test_auprc": "AUPRC",
    "test_physionet_utility": "Utility",
    "test_median_lead_time": "Lead Time",
    "test_asf": "ASF",
    "test_poms": "POMS"
}

def load_run_metrics(run_dir: Path) -> Dict[str, float]:
    metrics_file = run_dir / "metrics" / "final_test_metrics.json"
    if not metrics_file.exists():
        return {}
    with open(metrics_file) as f:
        return json.load(f)

def _compute_ci(values: pd.Series, confidence: float = 0.95) -> float:
    """Compute the margin of error for a given confidence interval."""
    n = len(values)
    if n < 2:
        return 0.0
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

def generate_latex_table(summary_df: pd.DataFrame, baseline_cond: str) -> str:
    """Generates a LaTeX table string highlighting the best results."""
    # Find best values for each metric
    best_vals = {}
    for metric in LATEX_METRICS.keys():
        mean_col = f"{metric}_mean"
        if mean_col not in summary_df.columns:
            continue
        
        # Ignore NaNs
        valid_vals = summary_df[mean_col].dropna()
        if len(valid_vals) == 0:
            continue
            
        if metric in HIGHER_IS_BETTER:
            best_vals[metric] = valid_vals.max()
        else:
            best_vals[metric] = valid_vals.min()

    # Build LaTeX string
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Aggregated Test Results (Mean $\\pm$ 95\\% CI)}")
    
    # Columns: Condition + the metrics
    cols = "l" + "c" * len(LATEX_METRICS)
    latex.append(f"\\begin{{tabular}}{{{cols}}}")
    latex.append("\\toprule")
    
    header = ["Condition"] + list(LATEX_METRICS.values())
    latex.append(" & ".join(header) + " \\\\")
    latex.append("\\midrule")

    for _, row in summary_df.iterrows():
        cond_clean = str(row['condition']).replace("_", "\\_")
        row_str = [cond_clean]
        
        for metric in LATEX_METRICS.keys():
            mean_col = f"{metric}_mean"
            ci_col = f"{metric}_ci95"
            pval_col = f"{metric}_pval"
            
            if mean_col in row and pd.notna(row[mean_col]):
                mean_val = row[mean_col]
                ci_val = row[ci_col] if ci_col in row and pd.notna(row[ci_col]) else 0.0
                pval = row[pval_col] if pval_col in row and pd.notna(row[pval_col]) else 1.0
                
                # Format
                cell = f"{mean_val:.3f} $\\pm$ {ci_val:.3f}"
                
                # Bold if best (within a tiny tolerance for floating point)
                if metric in best_vals and abs(mean_val - best_vals[metric]) < 1e-5:
                    cell = f"\\textbf{{{cell}}}"
                
                # Add asterisk if statistically significant against baseline
                if pval < 0.05 and str(row['condition']) != baseline_cond:
                    cell += "$^*$"
                
                row_str.append(cell)
            else:
                row_str.append("-")
                
        latex.append(" & ".join(row_str) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def run_aggregate(batch_dir: str | Path, output_csv: str | Path | None = None) -> None:
    batch_root = Path(batch_dir).resolve()
    manifest_file = batch_root / "batch_manifest.json"
    if not manifest_file.exists():
        print(f"ERROR: No manifest found at {manifest_file}")
        return

    with open(manifest_file) as f:
        manifest = json.load(f)

    runs = manifest.get("runs", [])
    if not runs:
        print("No runs found in manifest.")
        return

    records: List[Dict[str, Any]] = []
    for run in runs:
        run_dir = Path(run["run_dir"])
        if not run_dir.is_absolute():
             run_dir = batch_root / run_dir
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

    # Auto-detect baseline condition for statistical testing
    cond_names = sorted(cond_groups.groups.keys())
    baseline_cond = next((c for c in cond_names if "Baseline" in str(c) or "01" in str(c)), cond_names[0])
    baseline_group = cond_groups.get_group(baseline_cond)

    summary_rows: List[Dict[str, Any]] = []
    for cond_name, group in cond_groups:
        row: Dict[str, Any] = {"condition": cond_name, "n_seeds": len(group)}
        for metric in all_metric_keys:
            values = group[metric].dropna()
            if len(values) > 0:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std()
                row[f"{metric}_ci95"] = _compute_ci(values)
                
                # Statistical Significance (Welch's t-test against baseline)
                if cond_name != baseline_cond:
                    base_vals = baseline_group[metric].dropna()
                    if len(values) > 1 and len(base_vals) > 1:
                        # T-test for independent samples with unequal variance
                        _, p_val = stats.ttest_ind(values, base_vals, equal_var=False)
                        row[f"{metric}_pval"] = p_val
                    else:
                        row[f"{metric}_pval"] = None
                else:
                    row[f"{metric}_pval"] = 1.0  # baseline against itself
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_ci95"] = None
                row[f"{metric}_pval"] = None
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print(f"\n{'=' * 72}")
    print(f"  AGGREGATED RESULTS — {batch_root.name}")
    print(f"  Conditions: {len(summary_rows)} | Total runs: {len(records)}")
    print(f"  Baseline for t-tests: {baseline_cond}")
    print(f"{'=' * 72}")

    for _, row in summary_df.iterrows():
        print(f"\n  --- {row['condition']} (n={row['n_seeds']}) ---")
        for metric in KEY_METRICS:
            mean_key = f"{metric}_mean"
            ci_key = f"{metric}_ci95"
            if mean_key in row and pd.notna(row[mean_key]):
                mean_val = row[mean_key]
                ci_val = row[ci_key] if ci_key in row and pd.notna(row[ci_key]) else 0.0
                print(f"    {metric:42s}= {mean_val:.4f} ± {ci_val:.4f} (95% CI)")

    out_path = Path(output_csv) if output_csv else batch_root / "aggregated_results.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n  Full results saved to: {out_path}\n")

    print(f"{'=' * 72}")
    print("  LaTeX TABLE (95% CI)")
    print(f"{'=' * 72}\n")
    print(generate_latex_table(summary_df, baseline_cond))
    print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-run test metrics across conditions.")
    parser.add_argument("batch_dir", type=str, help="Path to the batch output directory")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path (default: <batch_dir>/aggregated_results.csv)")
    args = parser.parse_args()
    run_aggregate(args.batch_dir, args.csv)


if __name__ == "__main__":
    main()
