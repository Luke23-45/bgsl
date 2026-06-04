"""
studies/runner/commons/manifest.py
----------------------------------
Batch manifest writer/updater and post-batch summary aggregator.

The manifest is a single JSON file at ``<batch_root>/batch_manifest.json``
that records the grid, execution mode, host metadata, and per-run statuses.
After a local batch finishes, ``aggregate_summary`` joins the per-run
``metrics.csv`` files (written by CSVLogger) with the manifest run metadata
to produce ``<batch_root>/summary.csv``.
"""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from studies.runner.commons.paths import BatchPaths, RunPaths, project_root

if TYPE_CHECKING:
    from studies.runner.commons.cli_builder import RunSpec


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_sha() -> str:
    """Return the current git SHA, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def _run_record(spec: "RunSpec") -> Dict[str, Any]:
    """Build the initial per-run record for the manifest."""
    return {
        "run_id": spec.run_id,
        "cond_name": spec.cond_name,
        "seed": spec.seed,
        "base_config": spec.base_config,
        "run_dir": spec.run_paths.root.as_posix(),
        "override_yaml": spec.run_paths.override_yaml.as_posix(),
        "cmd": spec.cmd,
        "status": "pending",
        "exit_code": None,
        "started_at": None,
        "ended_at": None,
        "duration_s": None,
    }


def write_manifest(
    batch_paths: BatchPaths,
    run_specs: Iterable["RunSpec"],
    exec_mode: str,
) -> Path:
    """
    Write the initial batch manifest. Overwrites any existing file (batch
    ids are unique per invocation, so this should not be a concern).
    """
    payload: Dict[str, Any] = {
        "experiment_name": batch_paths.experiment_name,
        "experiment_type": batch_paths.experiment_type,
        "batch_id": batch_paths.batch_id,
        "batch_root": batch_paths.root.as_posix(),
        "exec_mode": exec_mode,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "git_sha": _git_sha(),
        "started_at": _now_iso(),
        "ended_at": None,
        "submitted_at": None,
        "overall_status": None,
        "runs": [_run_record(s) for s in run_specs],
    }
    _atomic_write_json(batch_paths.manifest_json, payload)
    return batch_paths.manifest_json


# ---------------------------------------------------------------------------
# Manifest updates
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via a sibling temp file + rename for crash safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def _read_manifest(batch_paths: BatchPaths) -> Dict[str, Any]:
    with open(batch_paths.manifest_json, "r", encoding="utf-8") as f:
        return json.load(f)


def update_manifest(batch_paths: BatchPaths, **fields: Any) -> None:
    """Merge top-level fields into the manifest."""
    data = _read_manifest(batch_paths)
    data.update(fields)
    _atomic_write_json(batch_paths.manifest_json, data)


def update_run_in_manifest(
    batch_paths: BatchPaths,
    run_id: str,
    **fields: Any,
) -> None:
    """Update a single run's record (looked up by run_id) inside the manifest."""
    data = _read_manifest(batch_paths)
    for record in data.get("runs", []):
        if record.get("run_id") == run_id:
            record.update(fields)
            break
    else:
        raise KeyError(f"run_id {run_id!r} not found in manifest at {batch_paths.manifest_json}")
    _atomic_write_json(batch_paths.manifest_json, data)


# ---------------------------------------------------------------------------
# Per-run status file
# ---------------------------------------------------------------------------

def write_run_status(run_paths: RunPaths, status: Mapping[str, Any]) -> None:
    """Write the per-run status JSON (alongside the run's other artifacts)."""
    run_paths.root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_paths.status_json, dict(status))


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

_RUN_META_COLS = ("run_id", "cond_name", "seed", "status", "exit_code", "duration_s")


def _find_metrics_csv(run_dir: Path) -> Optional[Path]:
    """Locate the deepest metrics.csv under ``run_dir/logs/`` (CSVLogger may nest under ``version_*``)."""
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return None
    matches = sorted(logs_dir.rglob("metrics.csv"))
    return matches[-1] if matches else None


def _final_metric_row(csv_path: Path) -> Dict[str, float]:
    """
    Reduce a metrics.csv to a single dict of ``{metric_name: last_non_nan_value}``.

    CSVLogger writes one row per ``log_metrics`` call, with many columns
    typically NaN for any given row. Taking the last non-NaN per column
    yields the final value for every metric that was ever logged.
    """
    df = pd.read_csv(csv_path)
    drop_cols = {"epoch", "step"}
    out: Dict[str, float] = {}
    for col in df.columns:
        if col in drop_cols:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        value = series.iloc[-1]
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(fval):
            continue
        out[col] = fval
    return out


def aggregate_summary(batch_paths: BatchPaths) -> Path:
    """
    Build ``summary.csv`` by joining manifest run metadata with the last
    value of every metric logged in each run's ``metrics.csv``.

    Also writes ``<run>/metrics/final_test_metrics.json`` for each run that
    has a metrics.csv. Runs without a metrics.csv (e.g. crashed before any
    logging) are still listed in summary.csv with only their manifest fields
    populated.

    Returns the path to summary.csv.
    """
    manifest = _read_manifest(batch_paths)
    rows: List[Dict[str, Any]] = []
    metric_keys: set = set()

    for record in manifest.get("runs", []):
        run_id = record["run_id"]
        run_dir = Path(record["run_dir"])

        base_row: Dict[str, Any] = {
            "run_id": run_id,
            "cond_name": record.get("cond_name"),
            "seed": record.get("seed"),
            "status": record.get("status"),
            "exit_code": record.get("exit_code"),
            "duration_s": record.get("duration_s"),
        }

        csv_path = _find_metrics_csv(run_dir)
        if csv_path is not None:
            metrics = _final_metric_row(csv_path)
            base_row.update(metrics)
            metric_keys.update(metrics.keys())

            # Mirror the extracted metrics next to the run for convenience.
            final_path = Path(record["run_dir"]) / "metrics" / "final_test_metrics.json"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(final_path, metrics)

        rows.append(base_row)

    ordered_metric_cols = sorted(metric_keys)
    columns = list(_RUN_META_COLS) + ordered_metric_cols
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(batch_paths.summary_csv, index=False)
    return batch_paths.summary_csv
