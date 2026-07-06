"""
studies/runner/commons/executor.py
----------------------------------
Executes a batch of :class:`RunSpec` objects under one of three modes.

Modes
-----
``dry_run``
    Print each run's command and target directory. No subprocesses, no
    aggregation. The manifest is still written so that ``--dry-run``
    invocations are inspectable.

``local_sequential``
    Run each spec in-process (one at a time) via ``subprocess.Popen``.
    Stdout + stderr are tee'd to both the parent terminal and the run's
    ``logs/console.log``. Per-run status is written to ``status.json`` and
    mirrored into the manifest. After all runs finish (or are interrupted),
    :func:`manifest.aggregate_summary` is called to produce ``summary.csv``.

``slurm``
    Generate a Bash script with one ``srun`` line per spec. Each job's
    stdout/stderr is directed at its own ``console.log``. The manifest is
    marked as ``submitted_at``; aggregation is the caller's responsibility
    on the cluster.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Literal, Optional

from studies.runner.commons.cli_builder import RunSpec
from studies.runner.commons.manifest import (
    aggregate_summary,
    update_manifest,
    update_run_in_manifest,
    write_manifest,
    write_run_status,
)
from studies.runner.commons.override_writer import write_run_override
from studies.runner.commons.paths import BatchPaths, project_root

logger = logging.getLogger("bgsl.runner.executor")

ExecMode = Literal["dry_run", "local_sequential", "slurm"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_runs(
    runs: List[RunSpec],
    batch_paths: BatchPaths,
    mode: ExecMode = "dry_run",
    slurm_script_path: Optional[str | Path] = None,
    fail_fast: bool = False,
) -> None:
    """
    Execute a batch of runs under the given mode.

    Parameters
    ----------
    runs : list of RunSpec
        Run specifications produced by ``cli_builder.build_run_spec``.
    batch_paths : BatchPaths
        Pre-built batch paths; the manifest will be written under
        ``batch_paths.manifest_json``.
    mode : str
        One of ``"dry_run"``, ``"local_sequential"``, ``"slurm"``.
    slurm_script_path : str or Path, optional
        Where to write the generated sbatch script (defaults to
        ``<batch_root>/slurm_submit.sh``). Only used by the ``"slurm"`` mode.
    fail_fast : bool
        If True, ``"local_sequential"`` raises on the first non-zero exit;
        otherwise it logs the failure and continues to the next run.
    """
    write_manifest(batch_paths, runs, exec_mode=mode)

    header = (
        f"--- Execution Mode: {mode.upper()} "
        f"| Total Jobs: {len(runs)} "
        f"| Batch: {batch_paths.batch_id} ---"
    )
    print(header)
    print(f"Outputs root: {batch_paths.root}")

    if mode == "dry_run":
        _run_dry(runs)
        return

    if mode == "local_sequential":
        _run_local_sequential(runs, batch_paths, fail_fast=fail_fast)
        return

    if mode == "slurm":
        script_path = (
            Path(slurm_script_path)
            if slurm_script_path is not None
            else batch_paths.root / "slurm_submit.sh"
        )
        _run_slurm(runs, batch_paths, script_path)
        return

    raise ValueError(f"Unknown execution mode: {mode!r}")


# ---------------------------------------------------------------------------
# Backwards-compat shim
# ---------------------------------------------------------------------------

def execute_commands(*args, **kwargs):
    """
    Deprecated stub left in place to flag callers of the old string-based
    API. Use :func:`execute_runs` with :class:`RunSpec` objects instead.
    """
    raise NotImplementedError(
        "execute_commands(List[str], ...) has been replaced by "
        "execute_runs(List[RunSpec], BatchPaths, ...). Update the runner to "
        "build RunSpec instances via cli_builder.build_run_spec."
    )


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def _run_dry(runs: Iterable[RunSpec]) -> None:
    runs = list(runs)
    for idx, spec in enumerate(runs):
        print(f"[{idx + 1}/{len(runs)}] run_id={spec.run_id}")
        print(f"    run_dir : {spec.run_paths.root}")
        print(f"    override: {spec.run_paths.override_yaml}")
        print(f"    cmd     : {spec.cmd}")


# ---------------------------------------------------------------------------
# Local sequential
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _classify_overall(exit_codes: List[Optional[int]]) -> str:
    if not exit_codes:
        return "empty"
    if all(c == 0 for c in exit_codes):
        return "success"
    if all(c is not None and c != 0 for c in exit_codes):
        return "failure"
    return "partial_failure"


def _run_local_sequential(
    runs: List[RunSpec],
    batch_paths: BatchPaths,
    fail_fast: bool,
) -> None:
    cwd = project_root()
    exit_codes: List[Optional[int]] = []
    interrupted = False

    try:
        for idx, spec in enumerate(runs):
            print(f"\n=== [{idx + 1}/{len(runs)}] {spec.run_id} ===")
            spec.run_paths.create_dirs()
            write_run_override(spec.run_paths)

            started_at = _now_iso()
            t0 = time.perf_counter()
            update_run_in_manifest(
                batch_paths, spec.run_id, status="running", started_at=started_at
            )

            exit_code: Optional[int] = None
            test_exit_code: Optional[int] = None
            status: str = "failed"

            try:
                exit_code = _spawn_and_tee(
                    argv=list(spec.argv),
                    cwd=cwd,
                    console_log=spec.run_paths.console_log,
                )
                status = "success" if exit_code == 0 else "failed"

                # Post-fit test evaluation on best checkpoint
                if status == "success":
                    test_exit_code = _run_test_after_fit(spec, cwd)
            except KeyboardInterrupt:
                status = "interrupted"
                interrupted = True
                raise
            finally:
                duration_s = time.perf_counter() - t0
                ended_at = _now_iso()
                status_payload = {
                    "run_id": spec.run_id,
                    "status": status,
                    "exit_code": exit_code,
                    "test_status": "success" if test_exit_code == 0 else "failed" if test_exit_code is not None else "skipped",
                    "test_exit_code": test_exit_code,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_s": duration_s,
                }
                write_run_status(spec.run_paths, status_payload)
                logger.info(
                    "[%s] finished (status=%s, exit_code=%s, test_status=%s)",
                    spec.run_id,
                    status,
                    exit_code,
                    status_payload["test_status"],
                )
                update_run_in_manifest(
                    batch_paths,
                    spec.run_id,
                    status=status,
                    exit_code=exit_code,
                    test_status=status_payload["test_status"],
                    test_exit_code=test_exit_code,
                    ended_at=ended_at,
                    duration_s=duration_s,
                )
                exit_codes.append(exit_code)

            if status != "success":
                print(f"[FAIL] {spec.run_id} exit_code={exit_code}")
                if fail_fast:
                    raise RuntimeError(
                        f"Run {spec.run_id} failed with exit_code={exit_code} "
                        f"(fail_fast=True)"
                    )

    finally:
        overall = "interrupted" if interrupted else _classify_overall(exit_codes)
        update_manifest(batch_paths, ended_at=_now_iso(), overall_status=overall)
        try:
            summary_path = aggregate_summary(batch_paths)
            print(f"\nSummary written: {summary_path}")
        except Exception as exc:  # noqa: BLE001 - we want any aggregation failure to be visible but non-fatal
            logger.warning("Summary aggregation failed: %s", exc)
        print(f"Manifest: {batch_paths.manifest_json}")
        print(f"Overall status: {overall}")


def _spawn_and_tee(argv: List[str], cwd: Path, console_log: Path) -> int:
    """
    Spawn ``argv`` (no shell), merge stderr into stdout, and tee every line
    to both the parent terminal and the run's console.log.

    Uses line-buffered reading (not char-by-char). The subprocess uses
    CloudProgressBar which emits plain ``print()`` calls with ``\\n``, so
    line-level iteration is both correct and dramatically faster than the
    previous character-level approach.

    Returns the subprocess exit code.
    """
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with open(console_log, "a", encoding="utf-8", errors="replace") as log_f:
        log_f.write(f"\n# === run started at {_now_iso()} ===\n")
        log_f.write(f"# cwd: {cwd}\n")
        log_f.write(f"# argv: {argv}\n\n")
        log_f.flush()

        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()

        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise

        exit_code = proc.wait()
        log_f.write(f"\n# === run ended at {_now_iso()} (exit_code={exit_code}) ===\n")
        return exit_code


# ---------------------------------------------------------------------------
# Slurm
# ---------------------------------------------------------------------------

def _run_test_after_fit(spec: "RunSpec", cwd: Path) -> Optional[int]:
    """
    Run ``bgsl.cli test`` using the best checkpoint from a completed fit run.

    Returns the subprocess exit code, or ``None`` if no suitable checkpoint
    was found.
    """
    ckpt_dir = spec.run_paths.checkpoints_dir
    ckpt_files = [
        f for f in ckpt_dir.glob("*.ckpt")
        if f.stem != "last"
    ]
    if not ckpt_files:
        logger.warning("No best checkpoint found for %s (skipping test)", spec.run_id)
        return None

    best_ckpt = str(ckpt_files[0])
    test_argv: List[str] = [
        sys.executable, "-m", "bgsl.cli",
        "test",
        "--config", spec.base_config,
        "--config", str(spec.run_paths.override_yaml),
        "--ckpt_path", best_ckpt,
    ]
    if spec.seed is not None:
        test_argv.extend(["--seed_everything", str(spec.seed)])

    for key, value in spec.overrides:
        test_argv.extend([f"--{key}", value])

    print(f"  [TEST] Running test on {best_ckpt}")
    return _spawn_and_tee(
        argv=test_argv,
        cwd=cwd,
        console_log=spec.run_paths.console_log,
    )


def _run_slurm(
    runs: List[RunSpec],
    batch_paths: BatchPaths,
    script_path: Path,
) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "#!/bin/bash",
        f"#SBATCH --job-name=bgsl_{batch_paths.experiment_name}_{batch_paths.experiment_type}",
        "#SBATCH --nodes=1",
        "#SBATCH --gpus=1",
        "#SBATCH --time=24:00:00",
        "",
        "set -euo pipefail",
        "",
        "source venv/bin/activate",
        "",
    ]

    for spec in runs:
        spec.run_paths.create_dirs()
        write_run_override(spec.run_paths)
        log = spec.run_paths.console_log.as_posix()
        lines.append(f"# --- {spec.run_id} ---")
        lines.append(f"srun --output={log} --error={log} {spec.cmd}")
        lines.append("")

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    update_manifest(batch_paths, submitted_at=_now_iso(), slurm_script=str(script_path))
    print(f"Slurm script generated at {script_path}.")
    print(f"Run `sbatch {script_path}` to submit.")
