"""
studies/runner/commons/paths.py
-------------------------------
Pure path layer for the persistent experiment-output directory scheme.

Directory layout produced by this module:

    outputs/
      <experiment_name>/                       # e.g. ablation | architecture | hyperparam
        <experiment_type>/                     # e.g. loss | tau | baseline | lr
          <YYYYMMDD-HHMMSS>__<uuid8>/          # one batch invocation
            batch_manifest.json
            summary.csv
            runs/
              <run_id>/                        # one (condition, seed) run
                _override.yaml
                logs/                          # console.log, metrics.csv, hparams.yaml, ...
                checkpoints/
                metrics/                       # final_test_metrics.json
                artifacts/
                status.json

The functions here do not write any content other than creating empty
directories. Content is written by override_writer / manifest / executor.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPaths:
    """All paths owned by a single (condition, seed) run."""
    run_id: str
    root: Path
    override_yaml: Path
    logs_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    artifacts_dir: Path
    console_log: Path
    status_json: Path
    final_metrics_json: Path

    def create_dirs(self) -> None:
        """Idempotently create every directory this run owns."""
        for d in (self.root, self.logs_dir, self.checkpoints_dir,
                  self.metrics_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BatchPaths:
    """All paths owned by a single batch invocation of a runner."""
    experiment_name: str
    experiment_type: str
    batch_id: str
    root: Path
    manifest_json: Path
    summary_csv: Path
    runs_dir: Path

    def create_dirs(self) -> None:
        """Idempotently create the batch root and runs/ subdirectory."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_paths(self, run_id: str) -> RunPaths:
        """Build a RunPaths under this batch for the given run identifier."""
        root = (self.runs_dir / run_id).resolve()
        logs_dir = root / "logs"
        checkpoints_dir = root / "checkpoints"
        metrics_dir = root / "metrics"
        artifacts_dir = root / "artifacts"
        return RunPaths(
            run_id=run_id,
            root=root,
            override_yaml=root / "_override.yaml",
            logs_dir=logs_dir,
            checkpoints_dir=checkpoints_dir,
            metrics_dir=metrics_dir,
            artifacts_dir=artifacts_dir,
            console_log=logs_dir / "console.log",
            status_json=root / "status.json",
            final_metrics_json=metrics_dir / "final_test_metrics.json",
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_batch_id(now: Optional[datetime] = None) -> str:
    """
    Build a batch identifier of the form ``YYYYMMDD-HHMMSS__<uuid8>``.

    The uuid suffix protects against same-second collisions if multiple
    batches are launched in quick succession.
    """
    ts = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}__{suffix}"


def build_batch_paths(
    experiment_name: str,
    experiment_type: str,
    outputs_root: str | Path = "outputs",
    now: Optional[datetime] = None,
) -> BatchPaths:
    """
    Build a BatchPaths anchored at ``<outputs_root>/<experiment_name>/<experiment_type>/<batch_id>/``
    and create the batch root + runs/ directory.

    Parameters
    ----------
    experiment_name : str
        Family label (e.g. 'ablation', 'architecture', 'hyperparam').
    experiment_type : str
        Specific runner label (e.g. 'loss', 'tau', 'baseline', 'lr').
    outputs_root : str | Path
        Root directory for all experiment outputs. Defaults to 'outputs' under
        the current working directory.
    now : datetime, optional
        Override for the timestamp component of the batch id. Used in tests.

    Returns
    -------
    BatchPaths
        Fully-populated, absolute-path BatchPaths whose root and runs/ subdir
        have been created on disk.
    """
    if not experiment_name or not experiment_type:
        raise ValueError("experiment_name and experiment_type must be non-empty.")

    batch_id = _make_batch_id(now)
    root = (Path(outputs_root) / experiment_name / experiment_type / batch_id).resolve()
    batch = BatchPaths(
        experiment_name=experiment_name,
        experiment_type=experiment_type,
        batch_id=batch_id,
        root=root,
        manifest_json=root / "batch_manifest.json",
        summary_csv=root / "summary.csv",
        runs_dir=root / "runs",
    )
    batch.create_dirs()
    return batch


# ---------------------------------------------------------------------------
# Helpers for runner scripts
# ---------------------------------------------------------------------------

_RUNNER_SUFFIX = "_runner"


def infer_experiment_id_from_file(file: str | Path) -> Tuple[str, str]:
    """
    Derive ``(experiment_name, experiment_type)`` from a runner file path.

    Expected layout: ``studies/runner/<experiment_name>/<experiment_type>_runner.py``.

    Examples
    --------
    >>> infer_experiment_id_from_file(
    ...     "studies/runner/ablation/loss_runner.py"
    ... )
    ('ablation', 'loss')
    """
    p = Path(file).resolve()
    name = p.parent.name
    stem = p.stem
    if stem.endswith(_RUNNER_SUFFIX):
        stem = stem[: -len(_RUNNER_SUFFIX)]
    if not name or not stem:
        raise ValueError(
            f"Could not infer experiment id from {file!r}; expected "
            f"<experiment_name>/<experiment_type>_runner.py layout."
        )
    return name, stem


def project_root() -> Path:
    """
    Return the BGSL project root directory.

    This module lives at ``<root>/studies/runner/commons/paths.py``, so the
    project root is four parents up.
    """
    return Path(__file__).resolve().parents[3]
