"""
studies/runner/commons/cli_builder.py
-------------------------------------
Builds executable specifications for a single LightningCLI training run.

Each call to ``build_run_spec`` returns a :class:`RunSpec` that bundles:
- the run identity (id, condition, seed),
- the resolved :class:`RunPaths`,
- the argv list for local subprocess execution, and
- a POSIX-quoted command string suitable for Slurm scripts / display.

The override YAML that pins the run's output directories is written as a
side effect (via :func:`override_writer.write_run_override`).
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from studies.runner.commons.paths import RunPaths


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """Everything the executor needs to launch (or describe) one run."""
    run_id: str
    cond_name: str
    seed: Optional[int]
    base_config: str
    run_paths: RunPaths
    argv: Tuple[str, ...]
    cmd: str
    overrides: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _stringify(value: Any) -> str:
    """Convert override values to a stable string form for the CLI."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def build_run_spec(
    base_config: str,
    run_paths: RunPaths,
    run_id: str,
    cond_name: str,
    seed: Optional[int] = None,
    subcommand: str = "fit",
    overrides: Optional[Dict[str, Any]] = None,
) -> RunSpec:
    """
    Build a RunSpec for a single LightningCLI run.

    This function is pure: it does not write to disk. The executor is
    responsible for materializing the override YAML and creating the run
    directories before launching the subprocess.

    Parameters
    ----------
    base_config : str
        Path to the base YAML configuration (project-relative is fine; the
        subprocess is launched with ``cwd=project_root()``).
    run_paths : RunPaths
        Pre-built RunPaths for this run.
    run_id : str
        Stable identifier for the run; used in logging and the manifest.
    cond_name : str
        The ablation condition name (without the seed suffix).
    seed : int, optional
        Global seed; appended as ``--seed_everything <seed>`` when provided.
    subcommand : str
        LightningCLI subcommand (default: ``"fit"``).
    overrides : dict, optional
        Flat dot-path -> value overrides forwarded as ``--<dot.path> <value>``.

    Returns
    -------
    RunSpec
    """
    argv: List[str] = [
        sys.executable, "-m", "bgsl.cli",
        subcommand,
        "--config", str(base_config),
        "--config", str(run_paths.override_yaml),
    ]

    if seed is not None:
        argv.extend(["--seed_everything", str(int(seed))])

    flat_overrides: Tuple[Tuple[str, str], ...] = tuple(
        (k, _stringify(v)) for k, v in (overrides or {}).items()
    )
    for key, value in flat_overrides:
        argv.extend([f"--{key}", value])

    # POSIX-quoted command string for display, slurm scripts, and the manifest.
    # We deliberately render the python executable as a bare 'python' here so
    # the string remains portable to a different host (e.g. a Slurm cluster).
    display_argv = ["python", "-m", "bgsl.cli"] + argv[3:]
    cmd_str = " ".join(shlex.quote(a) for a in display_argv)

    return RunSpec(
        run_id=run_id,
        cond_name=cond_name,
        seed=seed,
        base_config=str(base_config),
        run_paths=run_paths,
        argv=tuple(argv),
        cmd=cmd_str,
        overrides=flat_overrides,
    )
