"""
src/bgsl/empirical/studies/report.py
-------------------------------------
Assembles the empirical summary report from completed synthetic and
PhysioNet batches.

Reads per-run ``final_test_metrics.json`` from the given batch output
directories and writes ``docs/results/empirical_summary.md`` with
seed-wise tables, mean ± uncertainty, and gap analysis.

Usage
-----
    python -m bgsl.empirical.studies.report \\
        --synthetic outputs/empirical/synthetic_stress/<batch_id> \\
        --physionet  outputs/empirical/physionet_external/<batch_id>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Metrics to report (keys from final_test_metrics.json)
REPORT_METRICS: Tuple[str, ...] = (
    "test_physionet_utility",
    "test_auroc",
    "test_auprc",
    "test_median_lead_time",
    "test_mean_lead_time",
    "test_asf",
    "test_poms",
    "test_selected_threshold",
)

METRIC_LABELS: Dict[str, str] = {
    "test_physionet_utility": "PhysioNet Utility",
    "test_auroc": "AUROC",
    "test_auprc": "AUPRC",
    "test_median_lead_time": "Median Lead Time (h)",
    "test_mean_lead_time": "Mean Lead Time (h)",
    "test_asf": "ASF",
    "test_poms": "POMS",
    "test_selected_threshold": "Threshold",
}

# Condition-name parsing for synthetic stress runs
# Expected pattern: {scenario}_{method}_stress{stressSetting}
_SYNTHETIC_COND_RE = re.compile(
    r"^(?P<scenario>signal|null)_"
    r"(?P<method>BCE|BGSL_state|BGSL_vel)_"
    r"stress(?P<stress>\w+)$"
)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
_OUTPUT_DIR: Path = _PROJECT_ROOT / "docs" / "results"
_OUTPUT_FILE: str = "empirical_summary.md"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_batch_metrics(batch_dir: Path) -> List[Dict[str, Any]]:
    """Load per-run ``final_test_metrics.json`` and manifest metadata.

    Returns a list of dicts with keys: cond_name, seed, and all metrics.
    """
    manifest_path = batch_dir / "batch_manifest.json"
    if not manifest_path.exists():
        print(f"[WARN] No manifest found at {manifest_path}")
        return []

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    records: List[Dict[str, Any]] = []
    for run in manifest.get("runs", []):
        run_dir = Path(run["run_dir"])
        if not run_dir.is_absolute():
            run_dir = (batch_dir / "runs" / run["run_id"]).resolve()
        metrics_file = run_dir / "metrics" / "final_test_metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        records.append({
            "cond_name": run.get("cond_name", "unknown"),
            "seed": run.get("seed", -1),
            **metrics,
        })

    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _group_by_condition(
    records: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by ``cond_name``."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        name = rec["cond_name"]
        groups.setdefault(name, []).append(rec)
    return groups


def _mean_std_str(values: List[float], fmt: str = ".4f") -> str:
    """Return ``mean ± std`` formatted string, or ``—`` if no data."""
    arr = np.array(values)
    finite = np.isfinite(arr)
    if not finite.any():
        return "—"
    m = float(arr[finite].mean())
    s = float(arr[finite].std()) if finite.sum() > 1 else 0.0
    return f"{m:{fmt}} ± {s:{fmt}}"


def _format_table(
    groups: Dict[str, List[Dict[str, Any]]],
    cond_order: List[str],
    metrics: Tuple[str, ...],
) -> str:
    """Build a markdown table from grouped metrics."""
    lines: List[str] = []

    # Header
    header = ["Condition"]
    for m in metrics:
        header.append(METRIC_LABELS.get(m, m))
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    # Rows
    for cond in cond_order:
        if cond not in groups:
            continue
        recs = groups[cond]
        row = [cond]
        for m in metrics:
            vals = [r.get(m) for r in recs if r.get(m) is not None and np.isfinite(r.get(m, float("nan")))]
            row.append(_mean_std_str(vals) if vals else "—")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _seed_wise_table(
    groups: Dict[str, List[Dict[str, Any]]],
    cond_order: List[str],
    metrics: Tuple[str, ...],
) -> str:
    """Build a seed-wise markdown table."""
    lines: List[str] = []

    # Collect all seeds
    all_seeds: set = set()
    for recs in groups.values():
        for r in recs:
            all_seeds.add(r["seed"])
    all_seeds = sorted(all_seeds)

    # Header
    header = ["Condition", "Seed"] + [METRIC_LABELS.get(m, m) for m in metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for cond in cond_order:
        if cond not in groups:
            continue
        # Sort runs by seed
        seed_map: Dict[int, Dict[str, Any]] = {}
        for r in groups[cond]:
            seed_map[r["seed"]] = r

        for seed in all_seeds:
            if seed not in seed_map:
                continue
            r = seed_map[seed]
            row = [cond, str(seed)]
            for m in metrics:
                val = r.get(m)
                if val is not None and np.isfinite(val):
                    row.append(f"{val:.4f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------


def _write_synthetic_section(
    groups: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Write the synthetic stress-test section of the report."""
    lines: List[str] = []
    lines.append("## Synthetic Benchmark\n")
    lines.append(
        "Primary paper evidence.  The synthetic ICU benchmark tests "
        "the BGSL hypothesis under controlled conditions with known "
        "ground-truth trajectory structure.\n"
    )

    # Organise by stress setting
    stress_order = ["Default", "HighVel", "LongTrain"]
    scenarios = ["signal", "null"]
    methods_pretty = ["BCE", "BGSL_state", "BGSL_vel"]

    for stress in stress_order:
        lines.append(f"### Stress: {stress}\n")

        for scenario in scenarios:
            label = "Signal" if scenario == "signal" else "Null"
            lines.append(f"#### {label} Scenario\n")

            conds = [f"{scenario}_{m}_stress{stress}" for m in methods_pretty]
            available = [c for c in conds if c in groups]

            if not available:
                lines.append("*No data available.*\n")
                continue

            lines.append("##### Seed-wise Results\n")
            lines.append(_seed_wise_table(groups, available, REPORT_METRICS))
            lines.append("")

            lines.append("##### Aggregated (Mean ± Std)\n")
            lines.append(_format_table(groups, available, REPORT_METRICS))
            lines.append("")

    return "\n".join(lines)


def _write_gap_analysis(
    syn_groups: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Write the gap-analysis section for synthetic data."""
    lines: List[str] = []
    lines.append("## Gap Analysis\n")
    lines.append(
        "Compares BGSL+vel against BCE for each scenario and stress "
        "setting.  Positive gaps favour BGSL.\n"
    )

    stress_order = ["Default", "HighVel", "LongTrain"]
    scenarios = ["signal", "null"]
    methods_pretty = ["BCE", "BGSL_state", "BGSL_vel"]

    report_metrics_subset = [
        "test_physionet_utility",
        "test_auroc",
        "test_auprc",
        "test_median_lead_time",
    ]

    # Gap table header
    header = ["Scenario", "Stress", "Metric", "BCE", "BGSL+vel", "Gap"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for scenario in scenarios:
        for stress in stress_order:
            bce_cond = f"{scenario}_BCE_stress{stress}"
            bgsl_cond = f"{scenario}_BGSL_vel_stress{stress}"

            bce_recs = syn_groups.get(bce_cond, [])
            bgsl_recs = syn_groups.get(bgsl_cond, [])

            if not bce_recs or not bgsl_recs:
                continue

            for mi, metric in enumerate(report_metrics_subset):
                bce_vals = [
                    r.get(metric) for r in bce_recs
                    if r.get(metric) is not None and np.isfinite(r.get(metric, float("nan")))
                ]
                bgsl_vals = [
                    r.get(metric) for r in bgsl_recs
                    if r.get(metric) is not None and np.isfinite(r.get(metric, float("nan")))
                ]

                if not bce_vals or not bgsl_vals:
                    continue

                bce_mean = np.mean(bce_vals)
                bgsl_mean = np.mean(bgsl_vals)
                gap = bgsl_mean - bce_mean

                scenario_label = "Signal" if scenario == "signal" else "Null"
                metric_label = METRIC_LABELS.get(metric, metric)

                row = [
                    scenario_label,
                    stress,
                    metric_label,
                    f"{bce_mean:.4f}",
                    f"{bgsl_mean:.4f}",
                    f"{gap:+.4f}",
                ]
                lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _write_physionet_section(
    groups: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Write the PhysioNet external validation section."""
    lines: List[str] = []
    lines.append("## PhysioNet 2019 External Validation\n")
    lines.append(
        "External validation on the real PhysioNet 2019 sepsis dataset.  "
        "All methods use the frozen GRU architecture with no tuning.\n"
    )

    cond_order = ["BCE", "BGSL_state", "BGSL_vel"]
    available = [c for c in cond_order if c in groups]

    if not available:
        lines.append("*No data available.*\n")
        return "\n".join(lines)

    lines.append("### Seed-wise Results\n")
    lines.append(_seed_wise_table(groups, available, REPORT_METRICS))
    lines.append("")

    lines.append("### Aggregated (Mean ± Std)\n")
    lines.append(_format_table(groups, available, REPORT_METRICS))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the empirical summary report.",
    )
    parser.add_argument(
        "--synthetic",
        type=str,
        default=None,
        help="Path to the synthetic stress batch output directory.",
    )
    parser.add_argument(
        "--physionet",
        type=str,
        default=None,
        help="Path to the PhysioNet external validation batch output directory.",
    )
    args = parser.parse_args()

    # Load data
    syn_records: List[Dict[str, Any]] = []
    physionet_records: List[Dict[str, Any]] = []

    if args.synthetic:
        syn_records = _load_batch_metrics(Path(args.synthetic))
        print(f"[report] Loaded {len(syn_records)} synthetic runs from {args.synthetic}")

    if args.physionet:
        physionet_records = _load_batch_metrics(Path(args.physionet))
        print(f"[report] Loaded {len(physionet_records)} PhysioNet runs from {args.physionet}")

    # Group
    syn_groups = _group_by_condition(syn_records)
    physionet_groups = _group_by_condition(physionet_records)

    # Build report
    sections: List[str] = []
    sections.append("# Empirical Summary Report\n")
    sections.append(
        "Automatically generated by ``src/bgsl/empirical/studies/report.py``.\n"
    )

    if args.synthetic:
        sections.append(_write_synthetic_section(syn_groups))
        sections.append(_write_gap_analysis(syn_groups))

    if args.physionet:
        sections.append(_write_physionet_section(physionet_groups))

    # Conclusion
    sections.append("## Conclusion\n")
    sections.append(
        "The tables above document whether the frozen BGSL formulation "
        "improves over BCE under each condition.  If the gap is "
        "consistently negative (utility, AUROC, AUPRC), the empirical "
        "evidence does not support the BGSL hypothesis.\n"
    )

    report_text = "\n".join(sections)

    # Write
    output_dir = _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _OUTPUT_FILE
    output_path.write_text(report_text, encoding="utf-8")
    print(f"[report] Written to {output_path}")


if __name__ == "__main__":
    main()
