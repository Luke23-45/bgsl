"""
bgsl/data/acquisition.py
------------------------
Standalone, 3-tier data acquisition layer for BGSL.

This module is the SINGLE entry point for making a BGSL dataset available on
disk. It is **deliberately decoupled** from ``PhysioNetDataModule``,
``bgsl-train``, and the runner layer: invoke it once via the ``bgsl-acquire``
console script, then run ``bgsl-train`` as usual. The data module's existing
``FileNotFoundError`` path becomes the success indicator downstream.

Tier model
----------
1. **Local** (free) — If all 6 required artifacts exist in ``data_dir`` and an
   ``.bgsl_acquired.json`` marker is valid, short-circuit. No network, no
   subprocess.
2. **HuggingFace** — If tier 1 fails and an HF config is provided, call
   ``huggingface_hub.snapshot_download`` to materialize the pre-built LMDB
   archive into ``data_dir``. Verify SHA-256 against a repo-shipped
   ``manifest.json`` and re-emit the marker.
3. **Build from raw** — If tiers 1 + 2 fail, invoke the existing
   ``python -m bgsl.data.sepsis.physionet2019 --build`` subprocess (which handles
   kagglehub auto-download OR a user-supplied ``--raw-dir``).

Concurrency
-----------
A cross-platform file lock (``filelock.FileLock``) is held around any tier 2
or tier 3 work so multiple training jobs (or a Slurm prologue + Slurm epilogue
race) cannot clobber a half-built dataset.

Force / refresh
---------------
Set ``force=True`` (or the env var ``BGSL_FORCE_REACQUIRE=1``) to invalidate
the marker and re-run the tier chain from the top.

The module is pure stdlib + :mod:`yaml` at the import level. ``huggingface_hub``
and ``filelock`` are lazy-imported so missing extras surface a precise
``ModuleNotFoundError`` with a ``pip install`` instruction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple

import yaml

__all__ = [
    "AcquisitionConfig",
    "HfConfig",
    "BuildConfig",
    "AcquisitionResult",
    "ensure_dataset",
    "compute_manifest",
    "upload_to_hub",
    "REQUIRED_ARTIFACTS",
    "MANIFEST_FILENAME",
    "MARKER_FILENAME",
    "LOCK_FILENAME",
    "read_marker",
    "write_marker",
    "verify_manifest",
]

logger = logging.getLogger("bgsl.acquisition")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "manifest.json"
MARKER_FILENAME = ".bgsl_acquired.json"
LOCK_FILENAME = ".bgsl_acquired.lock"
BGSL_DATA_VERSION = "bgsl-1.1"

# The 6 artifacts that MUST exist for tier 1 to short-circuit. Paths are
# relative to ``data_dir`` and use forward slashes for cross-platform
# stability in the marker / manifest files.
REQUIRED_ARTIFACTS: Tuple[str, ...] = (
    "train/data.lmdb",
    "val/data.lmdb",
    "test/data.lmdb",
    "train_index.json",
    "val_index.json",
    "test_index.json",
)

# Default SHA-256 chunk size for manifest computation. 1 MiB strikes a balance
# between syscall overhead and peak memory.
_SHA_CHUNK = 1 << 20

# Environment variable that, when set to "1" / "true" / "yes", forces a
# re-acquisition even if the marker is present and valid.
FORCE_ENV_VAR = "BGSL_FORCE_REACQUIRE"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HfConfig:
    """HuggingFace Hub source configuration.

    Parameters
    ----------
    repo_id
        e.g. ``"your-org/bgsl-physionet2019"``.
    repo_type
        ``"dataset"`` (default) or ``"model"``.
    revision
        Branch, tag, or commit SHA. ``"main"`` is fine for day-to-day work;
        pin a commit SHA for reproducible experiments.
    filename_glob
        Glob pattern to filter the snapshot. ``"*"`` downloads everything.
    token_env
        Name of the environment variable carrying the HF token. ``None``
        means anonymous access (works for public repos only).
    """

    repo_id: str
    repo_type: Literal["dataset", "model"] = "dataset"
    revision: str = "main"
    filename_glob: str = "*"
    token_env: Optional[str] = "HF_TOKEN"


@dataclass
class BuildConfig:
    """Raw-data build configuration (tier 3).

    Parameters
    ----------
    raw_dir
        If set, ``python -m bgsl.data.sepsis.physionet2019 --build`` is invoked with
        ``--raw-dir <raw_dir>``. If ``None``, the build subprocess falls back
        to kagglehub auto-download.
    seed
        Patient-level split RNG seed.
    train_ratio, val_ratio
        Patient split fractions. Test = 1 - train - val.
    min_length
        Forwarded to the build subprocess for the ``min_length`` filter.
    horizon_hours
        Forwarded to the build subprocess (BGSL soft-target horizon).
    extra_args
        Any additional CLI args appended to the build subprocess verbatim.
    """

    raw_dir: Optional[str] = None
    seed: int = 2025
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    min_length: int = 8
    horizon_hours: int = 6
    extra_args: List[str] = field(default_factory=list)


@dataclass
class AcquisitionConfig:
    """Top-level acquisition configuration (mirrors the YAML schema).

    Parameters
    ----------
    data_dir
        Root directory that will hold ``{train,val,test}/data.lmdb`` plus
        ``{train,val,test}_index.json``.
    hf
        Tier-2 source. ``None`` disables the HF tier.
    build
        Tier-3 source. ``None`` disables the build tier (the script will then
        raise if tier 1 and 2 both fail).
    force
        If ``True``, bypass the marker cache and re-run the tier chain.
    lock_timeout_s
        How long to wait for a concurrent acquisition to release the lock.
    """

    data_dir: str
    hf: Optional[HfConfig] = None
    build: Optional[BuildConfig] = None
    force: bool = False
    lock_timeout_s: int = 600

    # ----- (de)serialization ---------------------------------------------

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AcquisitionConfig":
        """Build an :class:`AcquisitionConfig` from a plain dict (e.g. YAML)."""
        hf_raw = raw.get("hf")
        build_raw = raw.get("build")
        return cls(
            data_dir=raw["data_dir"],
            hf=HfConfig(**hf_raw) if hf_raw else None,
            build=BuildConfig(**build_raw) if build_raw else None,
            force=bool(raw.get("force", False)),
            lock_timeout_s=int(raw.get("lock_timeout_s", 600)),
        )

    @classmethod
    def from_yaml(cls, path: os.PathLike[str] | str) -> "AcquisitionConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if "data_dir" not in raw:
            raise ValueError(f"{path} is missing required key 'data_dir'.")
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "hf": dataclasses.asdict(self.hf) if self.hf else None,
            "build": dataclasses.asdict(self.build) if self.build else None,
            "force": self.force,
            "lock_timeout_s": self.lock_timeout_s,
        }


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AcquisitionResult:
    """Outcome of :func:`ensure_dataset`.

    ``ok`` is always set; ``source`` records which tier succeeded (or
    ``"none"`` if all failed). ``error`` carries a short description when
    ``ok is False``; on success it is ``None``.
    """

    ok: bool
    source: Literal["local", "hf", "build", "none"]
    data_dir: Path
    elapsed_s: float
    bytes_acquired: int
    tier_attempts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    marker: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Tier 1 — local check
# ---------------------------------------------------------------------------


def _is_built(data_dir: Path) -> bool:
    """Return ``True`` iff all 6 required artifacts exist with size > 0."""
    if not data_dir.is_dir():
        return False
    for rel in REQUIRED_ARTIFACTS:
        p = data_dir / rel
        if not p.is_file() or p.stat().st_size <= 0:
            return False
    return True


def _validate_marker_revision(
    marker: Dict[str, Any],
    expected_hf_revision: Optional[str],
) -> bool:
    """If the user pinned a specific HF revision, the marker must match.

    A marker written by tier 3 (build) does not record a revision, so this
    check is a no-op for non-HF acquisitions.
    """
    if expected_hf_revision is None:
        return marker.get("bgsl_data_version") == BGSL_DATA_VERSION
    if marker.get("bgsl_data_version") != BGSL_DATA_VERSION:
        return False
    if marker.get("source") != "hf":
        # Tier 1 with a tier-3 marker is acceptable: data is still on disk.
        return True
    return marker.get("hf_revision") == expected_hf_revision


# ---------------------------------------------------------------------------
# Marker file
# ---------------------------------------------------------------------------


def _marker_path(data_dir: Path) -> Path:
    return data_dir / MARKER_FILENAME


def read_marker(data_dir: Path) -> Optional[Dict[str, Any]]:
    """Read the marker file if present, else return ``None``."""
    p = _marker_path(data_dir)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Marker at %s is unreadable (%s); treating as absent.", p, exc)
        return None


def write_marker(
    data_dir: Path,
    source: str,
    hf_revision: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the acquisition marker atomically.

    A ``.tmp`` + ``os.replace`` pattern is used so that a crash mid-write
    never leaves a half-baked marker that would later be misinterpreted as
    success.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "source": source,
        "data_dir": str(data_dir.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bgsl_version": _bgsl_version(),
        "bgsl_data_version": BGSL_DATA_VERSION,
        "python_version": sys.version.split()[0],
    }
    if hf_revision is not None:
        payload["hf_revision"] = hf_revision
    if extras:
        payload.update(extras)

    final = _marker_path(data_dir)
    tmp = final.with_suffix(final.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, final)
    return payload


def _bgsl_version() -> str:
    try:
        from bgsl import __version__  # type: ignore[attr-defined]

        return str(__version__)
    except (ImportError, AttributeError):
        return "unknown"


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------


@contextmanager
def _acquire_lock(data_dir: Path, timeout_s: int) -> Iterator["filelock.FileLock"]:
    """Hold a process-safe lock for the duration of the ``with`` block.

    Imported lazily so users without ``filelock`` see a clear message and so
    the lock is only required when the script actually needs to write.
    """
    try:
        import filelock  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised in CI only
        raise ModuleNotFoundError(
            "filelock is required for concurrent-safe acquisition. "
            "Install it via `pip install filelock` or `pip install -e '.[data]'`."
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / LOCK_FILENAME
    lock = filelock.FileLock(str(lock_path), timeout=timeout_s)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_SHA_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest(
    data_dir: Path,
    *,
    include_globs: Tuple[str, ...] = ("*",),
) -> Dict[str, Any]:
    """Compute a manifest of {relative_path: {size, sha256}} over ``data_dir``.

    Skips the marker, the lock file, and the manifest itself. Excludes the
    temporary write-target ``.bgsl_acquired.json.tmp`` if present.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    skip_names = {MARKER_FILENAME, LOCK_FILENAME, MANIFEST_FILENAME,
                  f"{MARKER_FILENAME}.tmp"}
    files: List[Dict[str, Any]] = []
    total_bytes = 0
    for root, _dirs, fnames in os.walk(data_dir):
        for fn in fnames:
            if fn in skip_names:
                continue
            abs_path = Path(root) / fn
            rel = abs_path.relative_to(data_dir).as_posix()
            size = abs_path.stat().st_size
            sha = _sha256_of_file(abs_path)
            total_bytes += size
            files.append({"path": rel, "size": size, "sha256": sha})

    files.sort(key=lambda f: f["path"])
    return {
        "version": "bgsl-acquisition-manifest-1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(data_dir.resolve()),
        "total_bytes": total_bytes,
        "file_count": len(files),
        "files": files,
    }


def _load_repo_manifest(data_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the manifest shipped inside the downloaded snapshot, if any."""
    p = data_dir / MANIFEST_FILENAME
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_manifest(
    data_dir: Path,
    *,
    require_repo_manifest: bool = False,
) -> Dict[str, Any]:
    """Recompute the local manifest and assert it matches the repo manifest.

    Returns the freshly computed manifest on success. Raises :class:`RuntimeError`
    on the first mismatching file or missing repo manifest (when required).
    """
    if require_repo_manifest:
        repo_manifest = _load_repo_manifest(data_dir)
        if repo_manifest is None:
            raise RuntimeError(
                f"Repo shipped no {MANIFEST_FILENAME} at {data_dir}. "
                f"Re-upload with `bgsl-upload` to add one, or set "
                f"require_repo_manifest=False."
            )
    else:
        repo_manifest = _load_repo_manifest(data_dir)

    local = compute_manifest(data_dir)
    if repo_manifest is None:
        # No repo manifest to compare against. Write one for posterity.
        _write_local_manifest(data_dir, local)
        return local

    repo_by_path = {f["path"]: f for f in repo_manifest.get("files", [])}
    local_by_path = {f["path"]: f for f in local["files"]}
    missing_in_local = set(repo_by_path) - set(local_by_path)
    extra_in_local = set(local_by_path) - set(repo_by_path)
    if missing_in_local or extra_in_local:
        raise RuntimeError(
            f"Manifest mismatch between HF repo and local files: "
            f"missing={sorted(missing_in_local)[:5]} "
            f"extra={sorted(extra_in_local)[:5]}"
        )
    for path, info in repo_by_path.items():
        if info["sha256"] != local_by_path[path]["sha256"]:
            raise RuntimeError(
                f"sha256 mismatch for {path}: "
                f"repo={info['sha256']} local={local_by_path[path]['sha256']}"
            )
    return local


def _write_local_manifest(data_dir: Path, manifest: Dict[str, Any]) -> None:
    """Write a manifest alongside the data, atomically."""
    data_dir.mkdir(parents=True, exist_ok=True)
    final = data_dir / MANIFEST_FILENAME
    tmp = final.with_suffix(final.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(tmp, final)


# ---------------------------------------------------------------------------
# Tier 2 — HuggingFace download
# ---------------------------------------------------------------------------


def _hf_download(data_dir: Path, hf: HfConfig) -> int:
    """Download the configured repo snapshot into ``data_dir``.

    Returns the number of bytes written. Raises on failure.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "huggingface_hub is required for tier 2 (HF download). "
            "Install it via `pip install huggingface_hub` or "
            "`pip install -e '.[data]'`."
        ) from exc

    token: Optional[str] = None
    if hf.token_env:
        token = os.environ.get(hf.token_env)
        if not token:
            logger.info(
                "HF token env %r is unset; attempting anonymous download.", hf.token_env
            )

    allow_patterns: Optional[List[str]] = None
    if hf.filename_glob and hf.filename_glob != "*":
        allow_patterns = [hf.filename_glob]

    # snapshot_download writes directly into local_dir; we wipe first so a
    # half-finished previous attempt cannot leave stale files.
    if data_dir.exists():
        # Be careful: only wipe the contents, never the directory itself if
        # it's a symlink or has unusual perms; but here it's a normal
        # data_dir. Recreate empty dirs as needed.
        for entry in data_dir.iterdir():
            if entry.name in {MARKER_FILENAME, LOCK_FILENAME}:
                # Preserve any marker we wrote earlier in this run.
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=hf.repo_id,
        repo_type=hf.repo_type,
        revision=hf.revision,
        local_dir=str(data_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
        tqdm_class=None,
    )

    # Measure what landed.
    total = 0
    for root, _dirs, fnames in os.walk(data_dir):
        for fn in fnames:
            if fn in {MARKER_FILENAME, LOCK_FILENAME, MANIFEST_FILENAME}:
                continue
            total += (Path(root) / fn).stat().st_size
    return total


# ---------------------------------------------------------------------------
# Tier 3 — build from raw (delegates to physionet2019.py CLI)
# ---------------------------------------------------------------------------


def _run_build_subprocess(
    data_dir: Path,
    build: BuildConfig,
    log: Callable[[str], None] = logger.info,
) -> None:
    """Invoke ``python -m bgsl.data.sepsis.physionet2019 --build ...`` as a subprocess.

    Delegating to the existing CLI keeps the acquisition module free of
    pandas / lmdb imports and inherits the build pipeline's progress bars.
    """
    cmd: List[str] = [
        sys.executable,
        "-m",
        "bgsl.data.sepsis.physionet2019",
        "--build",
        "--config",
        "src/bgsl/config/dataset/sepsis.yaml",
        "--out-dir",
        str(data_dir),
        "--seed",
        str(build.seed),
    ]
    if build.raw_dir:
        cmd += ["--raw-dir", build.raw_dir]
    cmd += list(build.extra_args)

    log("Tier 3 build command: %s", " ".join(cmd))
    # ``text=True`` + capture_output for log forwarding; check=True so
    # non-zero exit propagates as a clear subprocess error.
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        for line in proc.stdout.splitlines():
            log("[build] %s", line)
    if proc.stderr:
        for line in proc.stderr.splitlines():
            log("[build:err] %s", line)
    if proc.returncode != 0:
        raise RuntimeError(
            f"build subprocess failed with exit code {proc.returncode}. "
            f"See the preceding [build:err] lines for details."
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ensure_dataset(
    cfg: AcquisitionConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> AcquisitionResult:
    """Run the 3-tier chain and return an :class:`AcquisitionResult`.

    Never raises for "expected" failures (HF unreachable, no raw data, etc.)
    — those are encoded in the returned ``result.ok / result.source``. Only
    programmer errors (e.g. bad config) and unexpected I/O errors propagate.
    """
    log_fn = log or logger.info
    t0 = time.monotonic()
    data_dir = Path(cfg.data_dir).expanduser().resolve()
    tier_attempts: List[str] = []
    bytes_acquired = 0

    # Honor the env-var override unless the user already explicitly set force.
    force = cfg.force or _env_truthy(FORCE_ENV_VAR)
    if force:
        log_fn("Force re-acquisition requested (force=%s, env=%s).",
               cfg.force, os.environ.get(FORCE_ENV_VAR))

    # ----- Tier 1 -----------------------------------------------------
    tier_attempts.append("local")
    if not force and _is_built(data_dir):
        marker = read_marker(data_dir)
        if marker is not None and _validate_marker_revision(
            marker, cfg.hf.revision if cfg.hf else None
        ):
            log_fn("[tier 1] Local data at %s verified via marker.", data_dir)
            return AcquisitionResult(
                ok=True,
                source="local",
                data_dir=data_dir,
                elapsed_s=time.monotonic() - t0,
                bytes_acquired=0,
                tier_attempts=tier_attempts,
                marker=marker,
            )
        if marker is None:
            log_fn("[tier 1] Required artifacts present at %s, but no marker found.",
                   data_dir)
        else:
            log_fn(
                "[tier 1] Marker revision mismatch (have %r, want %r).",
                marker.get("hf_revision"), cfg.hf.revision if cfg.hf else None,
            )

    # ----- Tier 2 -----------------------------------------------------
    if cfg.hf is not None:
        tier_attempts.append("hf")
        log_fn("[tier 2] Attempting HF download from %s @ %s …",
               cfg.hf.repo_id, cfg.hf.revision)
        try:
            with _acquire_lock(data_dir, cfg.lock_timeout_s):
                # If force, nuke any stale artifacts before download so the
                # snapshot is the only source of truth.
                if force and data_dir.is_dir():
                    _purge_data_artifacts(data_dir)
                bytes_acquired = _hf_download(data_dir, cfg.hf)
                # Verify against the shipped manifest (if any). On a fully
                # unmanaged repo this writes a local manifest for posterity.
                verify_manifest(data_dir, require_repo_manifest=False)
            marker = write_marker(
                data_dir,
                source="hf",
                hf_revision=cfg.hf.revision,
                extras={"hf_repo_id": cfg.hf.repo_id,
                        "hf_repo_type": cfg.hf.repo_type,
                        "bytes_acquired": bytes_acquired},
            )
            return AcquisitionResult(
                ok=True,
                source="hf",
                data_dir=data_dir,
                elapsed_s=time.monotonic() - t0,
                bytes_acquired=bytes_acquired,
                tier_attempts=tier_attempts,
                marker=marker,
            )
        except ModuleNotFoundError as exc:
            # Missing optional dep — surface as a tier failure, not a crash.
            log_fn("[tier 2] %s", exc)
        except Exception as exc:  # noqa: BLE001 - we want to catch all HF errors
            log_fn("[tier 2] HF download failed: %s", exc)
    else:
        log_fn("[tier 2] No HF config provided; skipping tier 2.")

    # ----- Tier 3 -----------------------------------------------------
    if cfg.build is not None:
        tier_attempts.append("build")
        log_fn("[tier 3] Building from raw (raw_dir=%s).",
               cfg.build.raw_dir or "<kagglehub auto>")
        try:
            with _acquire_lock(data_dir, cfg.lock_timeout_s):
                if force and data_dir.is_dir():
                    _purge_data_artifacts(data_dir)
                data_dir.mkdir(parents=True, exist_ok=True)
                _run_build_subprocess(data_dir, cfg.build, log=log_fn)
                # Sanity: build must produce all required artifacts.
                if not _is_built(data_dir):
                    raise RuntimeError(
                        "build subprocess completed but the 6 required "
                        "artifacts are not all present in the output dir."
                    )
                # Compute & persist a manifest so tier 2 can verify on later
                # re-acquisitions from a (future) HF upload of this build.
                local_manifest = compute_manifest(data_dir)
                _write_local_manifest(data_dir, local_manifest)
                bytes_acquired = local_manifest["total_bytes"]
            marker = write_marker(
                data_dir,
                source="build",
                extras={
                    "build_seed": cfg.build.seed,
                    "build_train_ratio": cfg.build.train_ratio,
                    "build_val_ratio": cfg.build.val_ratio,
                    "raw_dir": cfg.build.raw_dir,
                },
            )
            return AcquisitionResult(
                ok=True,
                source="build",
                data_dir=data_dir,
                elapsed_s=time.monotonic() - t0,
                bytes_acquired=bytes_acquired,
                tier_attempts=tier_attempts,
                marker=marker,
            )
        except Exception as exc:  # noqa: BLE001
            log_fn("[tier 3] build failed: %s", exc)
    else:
        log_fn("[tier 3] No build config provided; skipping tier 3.")

    return AcquisitionResult(
        ok=False,
        source="none",
        data_dir=data_dir,
        elapsed_s=time.monotonic() - t0,
        bytes_acquired=bytes_acquired,
        tier_attempts=tier_attempts,
        error="all tiers failed",
    )


# ---------------------------------------------------------------------------
# Upload (producer side)
# ---------------------------------------------------------------------------


def upload_to_hub(
    data_dir: os.PathLike[str] | str,
    hf: HfConfig,
    *,
    private: bool = False,
    commit_message: str = "Update BGSL PhysioNet2019 LMDB",
    dry_run: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Upload ``data_dir`` to the configured HF repo, with a dataset card.

    Steps
    -----
    1. Compute & persist a manifest in ``data_dir`` (if missing or stale).
    2. Build a README.md dataset card with provenance fields.
    3. ``HfApi.create_repo(..., exist_ok=True)`` then
       ``HfApi.upload_folder(..., commit_message=...)``.

    ``dry_run=True`` performs steps 1-2 locally and reports the would-be
    payload size; it never touches the network.
    """
    log_fn = log or logger.info
    data_dir = Path(data_dir).expanduser().resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    if not _is_built(data_dir):
        raise RuntimeError(
            f"data_dir at {data_dir} is missing one or more required artifacts. "
            f"Refusing to upload an incomplete dataset."
        )

    log_fn("Computing manifest for %s …", data_dir)
    manifest = compute_manifest(data_dir)
    _write_local_manifest(data_dir, manifest)
    log_fn("Manifest: %d files, %d bytes total.", manifest["file_count"], manifest["total_bytes"])

    log_fn("Rendering dataset card …")
    readme = _render_dataset_card(manifest, hf)
    _write_text_atomic(data_dir / "README.md", readme)

    if dry_run:
        log_fn("Dry run: not uploading. Repo would be: %s @ %s (private=%s)",
               hf.repo_id, hf.revision, private)
        return {
            "dry_run": True,
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "repo_id": hf.repo_id,
            "revision": hf.revision,
        }

    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "huggingface_hub is required for upload. "
            "Install it via `pip install huggingface_hub` or "
            "`pip install -e '.[data]'`."
        ) from exc

    token: Optional[str] = None
    if hf.token_env:
        token = os.environ.get(hf.token_env)
    if not token:
        raise RuntimeError(
            f"Environment variable {hf.token_env!r} is not set; cannot upload."
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=hf.repo_id,
        repo_type=hf.repo_type,
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=hf.repo_id,
        repo_type=hf.repo_type,
        revision=hf.revision,
        commit_message=commit_message,
    )
    log_fn("Upload complete: https://huggingface.co/datasets/%s/tree/%s",
           hf.repo_id, hf.revision)
    return {
        "dry_run": False,
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "repo_id": hf.repo_id,
        "revision": hf.revision,
        "url": f"https://huggingface.co/datasets/{hf.repo_id}/tree/{hf.revision}",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_truthy(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _purge_data_artifacts(data_dir: Path) -> None:
    """Remove all data artifacts and the marker, leaving the directory empty.

    The lock file is preserved so concurrent processes keep seeing the lock
    and waiting on whoever is mid-acquisition.
    """
    if not data_dir.is_dir():
        return
    for entry in data_dir.iterdir():
        if entry.name == LOCK_FILENAME:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp, path)


def _render_dataset_card(manifest: Dict[str, Any], hf: HfConfig) -> str:
    """Produce a HuggingFace dataset card (README.md) with provenance."""
    files_summary = "\n".join(
        f"- `{f['path']}` — {f['size']:,} bytes, sha256 `{f['sha256'][:12]}…`"
        for f in manifest["files"]
    )
    return f"""---
license: mit
tags:
- bgsl
- sepsis
- physionet
- clinical-time-series
- lmdb
---

# BGSL — PhysioNet/CinC 2019 Sepsis LMDB

Pre-built BGSL-formatted LMDB for the PhysioNet 2019 Sepsis Challenge,
produced by `bgsl.data.acquisition.upload_to_hub`.

## Provenance

- BGSL version: `{manifest.get('bgsl_version', 'unknown')}`
- Manifest version: `{manifest.get('version', 'unknown')}`
- Created: `{manifest.get('created_at', 'unknown')}`
- File count: **{manifest.get('file_count', 0)}**
- Total size: **{manifest.get('total_bytes', 0):,} bytes**
- HF repo: `{hf.repo_id}` (`{hf.repo_type}`) @ `{hf.revision}`

## Contents

{files_summary}

## Usage

```bash
bgsl-acquire --config data_acquisition.yaml
bgsl-train fit --config experiments/configs/physionet_gru.yaml
```

The `bgsl-acquire` script will detect this repo's marker and short-circuit
to tier 1.
"""


# ---------------------------------------------------------------------------
# Plan (dry-run) helper
# ---------------------------------------------------------------------------


def plan(cfg: AcquisitionConfig) -> List[Dict[str, Any]]:
    """Return the planned tier sequence without executing it."""
    plan_list: List[Dict[str, Any]] = [{
        "tier": 1,
        "name": "local",
        "action": "verify 6 required artifacts + marker",
    }]
    if cfg.hf is not None:
        plan_list.append({
            "tier": 2,
            "name": "hf",
            "action": f"snapshot_download({cfg.hf.repo_id} @ {cfg.hf.revision})",
        })
    if cfg.build is not None:
        plan_list.append({
            "tier": 3,
            "name": "build",
            "action": f"python -m bgsl.data.sepsis.physionet2019 --build "
                     f"(--raw-dir={cfg.build.raw_dir}, seed={cfg.build.seed})",
        })
    return plan_list
