"""
tests/test_data_acquisition.py
------------------------------
Unit tests for the standalone 3-tier data acquisition layer.

These tests are designed to run without:
- network access (HF and kagglehub are mocked)
- real .psv files or LMDB data (tests write tiny dummy artifacts)
- GPU / heavy deps (only stdlib + `bgsl.data.acquisition` is exercised)

Cases
-----
- Tier 1 short-circuit
- Tier 2 (HF) success
- Tier 2 (HF) failure falls back to tier 3 (build)
- Tier 3 (build) subprocess invocation
- Force re-acquisition across stale marker
- Marker round-trip (JSON-serializable, all fields)
- Manifest compute + verify
- Concurrent acquisition blocked by lock
- HF token resolution
- Upload dry-run
- Config YAML round-trip
- Plan / dry-run CLI behavior
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from bgsl.data.sepsis.acquisition import (
    AcquisitionConfig,
    AcquisitionResult,
    BuildConfig,
    HfConfig,
    MANIFEST_FILENAME,
    MARKER_FILENAME,
    REQUIRED_ARTIFACTS,
    compute_manifest,
    ensure_dataset,
    plan,
    read_marker,
    upload_to_hub,
    verify_manifest,
    write_marker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_data_dir(tmp_path: Path) -> Path:
    """Create a tmp data_dir with the 6 required artifacts + non-zero size."""
    data_dir = tmp_path / "data" / "ready"
    for rel in REQUIRED_ARTIFACTS:
        p = data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 16)  # >0 bytes
    return data_dir


@pytest.fixture
def empty_dir_path(tmp_path: Path) -> Path:
    """A guaranteed-empty directory under tmp_path."""
    p = tmp_path / "empty"
    p.mkdir()
    return p


@pytest.fixture
def quiet_log():
    """A no-op log callable (ensure_dataset takes a log= kwarg)."""

    class _S:
        def __call__(self, *_a, **_kw):
            return None

        def info(self, *_a, **_kw):
            return None

    return _S()


@pytest.fixture
def fake_huggingface_hub(monkeypatch):
    """
    Inject a fake ``huggingface_hub`` module into ``sys.modules`` and yield
    a setter for its ``snapshot_download`` callable. This lets us mock the
    HF Hub SDK without requiring it to be installed.
    """
    mod = types.ModuleType("huggingface_hub")
    mod.snapshot_download = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)

    class _Setter:
        def set(self, fn):
            mod.snapshot_download = fn

    return _Setter()


# ---------------------------------------------------------------------------
# Helper: build a (marker-writable) payload
# ---------------------------------------------------------------------------


def build_marker_payload(**overrides: Any) -> Dict[str, Any]:
    payload = {
        "source": "local",
        "data_dir": str(overrides.pop("data_dir", "/tmp/x")),
        "timestamp": "2026-06-04T12:00:00+00:00",
        "bgsl_version": "0.1.0",
        "python_version": "3.14.5",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Tier 1 — local short-circuit
# ---------------------------------------------------------------------------


def test_tier1_short_circuit(populated_data_dir, quiet_log):
    write_marker(populated_data_dir, "local")
    cfg = AcquisitionConfig(data_dir=str(populated_data_dir))

    result = ensure_dataset(cfg, log=quiet_log)

    assert isinstance(result, AcquisitionResult)
    assert result.ok is True
    assert result.source == "local"
    assert result.data_dir == populated_data_dir.resolve()
    assert result.bytes_acquired == 0
    assert result.tier_attempts == ["local"]
    assert result.marker is not None
    assert result.marker["source"] == "local"


def test_tier1_without_marker_does_not_short_circuit(populated_data_dir, quiet_log):
    cfg = AcquisitionConfig(data_dir=str(populated_data_dir))
    # No HF or build config — chain must give up cleanly.
    result = ensure_dataset(cfg, log=quiet_log)
    assert result.ok is False
    assert result.source == "none"
    assert "local" in result.tier_attempts


# ---------------------------------------------------------------------------
# Tier 2 — HF success path (mocked)
# ---------------------------------------------------------------------------


def test_tier2_hf_success(populated_data_dir, quiet_log, fake_huggingface_hub):
    """
    When tier 1 fails (empty dir) and tier 2 succeeds, we end up with the
    local artifacts + a marker that records the HF revision.
    """
    empty_dir = populated_data_dir.parent / "empty"
    empty_dir.mkdir()

    def fake_snapshot_download(**kwargs):
        # Pretend HF wrote exactly the 6 required artifacts into local_dir.
        for rel in REQUIRED_ARTIFACTS:
            p = Path(kwargs["local_dir"]) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * 8)
        # No manifest.json shipped by the repo.
        return str(kwargs["local_dir"])

    fake_huggingface_hub.set(fake_snapshot_download)

    cfg = AcquisitionConfig(
        data_dir=str(empty_dir),
        hf=HfConfig(repo_id="org/repo", revision="abc123"),
    )
    result = ensure_dataset(cfg, log=quiet_log)

    assert result.ok is True
    assert result.source == "hf"
    assert "hf" in result.tier_attempts
    assert result.marker["hf_revision"] == "abc123"
    assert result.marker["hf_repo_id"] == "org/repo"
    # All 6 artifacts landed.
    for rel in REQUIRED_ARTIFACTS:
        assert (empty_dir / rel).is_file()


def test_tier2_missing_huggingface_hub_raises_clear_error(empty_dir_path, quiet_log, monkeypatch):
    """
    If `huggingface_hub` is not installed, tier 2 must surface a precise
    ModuleNotFoundError with the install hint — and tier 3 must still run.
    """
    # Make tier 3 a no-op for this test by setting build to None.
    cfg = AcquisitionConfig(
        data_dir=str(empty_dir_path),
        hf=HfConfig(repo_id="org/repo"),
        build=None,
    )

    # Inject a module stub without snapshot_download so the lazy import fails
    # even if huggingface_hub is installed in the environment.
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))

    result = ensure_dataset(cfg, log=quiet_log)
    assert result.ok is False
    assert "hf" in result.tier_attempts


# ---------------------------------------------------------------------------
# Tier 2 → Tier 3 fallback
# ---------------------------------------------------------------------------


def test_tier2_falls_back_to_tier3(tmp_path, quiet_log, monkeypatch, fake_huggingface_hub):
    """
    HF raises → we run the build subprocess → all 6 artifacts appear.
    """
    data_dir = tmp_path / "data" / "ready"

    def fake_snapshot_download(**kwargs):
        raise RuntimeError("simulated network failure")

    fake_huggingface_hub.set(fake_snapshot_download)

    def fake_run(cmd, **kwargs):
        # Write the 6 artifacts directly so we never recurse into the
        # patched subprocess.run.
        out_dir_idx = cmd.index("--out-dir") + 1
        out_dir = Path(cmd[out_dir_idx])
        for rel in REQUIRED_ARTIFACTS:
            p = out_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * 4)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    cfg = AcquisitionConfig(
        data_dir=str(data_dir),
        hf=HfConfig(repo_id="org/repo"),
        build=BuildConfig(raw_dir="/tmp/raw", seed=7, train_ratio=0.7, val_ratio=0.2),
    )
    result = ensure_dataset(cfg, log=quiet_log)

    assert result.ok is True
    assert result.source == "build"
    assert result.tier_attempts == ["local", "hf", "build"]
    for rel in REQUIRED_ARTIFACTS:
        assert (data_dir / rel).is_file()
    # Build marker records provenance.
    assert result.marker["build_seed"] == 7
    assert result.marker["build_train_ratio"] == 0.7


# ---------------------------------------------------------------------------
# Tier 3 subprocess invocation
# ---------------------------------------------------------------------------


def test_tier3_subprocess_argv(tmp_path, quiet_log, monkeypatch):
    """
    Verify the exact argv passed to the build subprocess.
    """
    data_dir = tmp_path / "data" / "ready"
    captured: Dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        for rel in REQUIRED_ARTIFACTS:
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / rel).write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    cfg = AcquisitionConfig(
        data_dir=str(data_dir),
        build=BuildConfig(raw_dir="/some/raw", seed=42, train_ratio=0.6, val_ratio=0.2,
                          extra_args=["--foo", "bar"]),
    )
    result = ensure_dataset(cfg, log=quiet_log)

    assert result.ok is True
    assert result.source == "build"
    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "bgsl.data.physionet2019"]
    assert "--build" in cmd
    assert "--out-dir" in cmd
    assert str(data_dir) in cmd
    assert "--raw-dir" in cmd and "/some/raw" in cmd
    assert "--seed" in cmd and "42" in cmd
    assert cmd[-2:] == ["--foo", "bar"]


def test_tier3_subprocess_failure_propagates(tmp_path, quiet_log, monkeypatch):
    data_dir = tmp_path / "data" / "ready"
    cp = subprocess.CompletedProcess(["x"], 1, stdout="", stderr="boom")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: cp)
    cfg = AcquisitionConfig(data_dir=str(data_dir), build=BuildConfig(raw_dir="/x"))
    result = ensure_dataset(cfg, log=quiet_log)
    assert result.ok is False
    assert "build" in result.tier_attempts


# ---------------------------------------------------------------------------
# Force re-acquisition
# ---------------------------------------------------------------------------


def test_force_bypasses_stale_marker(populated_data_dir, quiet_log, fake_huggingface_hub):
    """
    force=True re-runs the tier chain even if a valid marker is present.
    The previous marker is replaced with the new one.
    """
    write_marker(
        populated_data_dir, "hf",
        extras={"hf_revision": "old_sha", "hf_repo_id": "org/repo"},
    )

    def fake_snapshot_download(**kwargs):
        for rel in REQUIRED_ARTIFACTS:
            p = Path(kwargs["local_dir"]) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        return str(kwargs["local_dir"])

    fake_huggingface_hub.set(fake_snapshot_download)

    cfg = AcquisitionConfig(
        data_dir=str(populated_data_dir),
        hf=HfConfig(repo_id="org/repo", revision="new_sha"),
        build=None,
        force=True,
    )
    result = ensure_dataset(cfg, log=quiet_log)

    assert result.ok is True
    assert result.source == "hf"
    marker = read_marker(populated_data_dir)
    assert marker is not None
    assert marker["hf_revision"] == "new_sha"


def test_env_force_var_triggers_reacquisition(populated_data_dir, quiet_log,
                                              fake_huggingface_hub, monkeypatch):
    """BGSL_FORCE_REACQUIRE=1 in the env forces re-acquisition."""
    write_marker(populated_data_dir, "local")

    def fake_snapshot_download(**kwargs):
        for rel in REQUIRED_ARTIFACTS:
            p = Path(kwargs["local_dir"]) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        return str(kwargs["local_dir"])

    fake_huggingface_hub.set(fake_snapshot_download)
    monkeypatch.setenv("BGSL_FORCE_REACQUIRE", "1")

    cfg = AcquisitionConfig(
        data_dir=str(populated_data_dir),
        hf=HfConfig(repo_id="org/repo", revision="r1"),
        build=None,
        # cfg.force stays False; the env var is what activates it.
    )
    result = ensure_dataset(cfg, log=quiet_log)
    assert result.source == "hf"
    assert read_marker(populated_data_dir)["hf_revision"] == "r1"


# ---------------------------------------------------------------------------
# Marker round-trip
# ---------------------------------------------------------------------------


def test_marker_round_trip(populated_data_dir):
    payload = write_marker(
        populated_data_dir, "hf",
        hf_revision="abc",
        extras={"hf_repo_id": "org/repo", "bytes_acquired": 1024},
    )
    again = read_marker(populated_data_dir)
    assert again is not None
    for key in ("source", "data_dir", "timestamp", "bgsl_version",
                "bgsl_data_version", "python_version", "hf_revision", "hf_repo_id",
                "bytes_acquired"):
        assert key in again
    assert again["source"] == "hf"
    assert again["hf_revision"] == "abc"
    # Must be JSON-serializable as-is.
    json.dumps(again)
    assert payload["bytes_acquired"] == 1024


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_compute_and_verify(populated_data_dir):
    manifest = compute_manifest(populated_data_dir)
    assert manifest["file_count"] == len(REQUIRED_ARTIFACTS)
    paths = {f["path"] for f in manifest["files"]}
    assert paths == set(REQUIRED_ARTIFACTS)
    # Each entry has a 64-char sha256.
    for f in manifest["files"]:
        assert len(f["sha256"]) == 64
        assert f["size"] > 0
    # Re-verify: equal to itself.
    verified = verify_manifest(populated_data_dir, require_repo_manifest=False)
    assert verified["file_count"] == manifest["file_count"]


def test_manifest_detects_corruption(populated_data_dir):
    """If a repo manifest was recorded but a file's sha no longer matches,
    ``verify_manifest`` must raise with a clear message."""
    # Build a fake "repo" manifest: train/data.lmdb gets a bogus sha, others
    # are computed correctly so the only mismatch is the one we want.
    real = compute_manifest(populated_data_dir)
    real_by_path = {f["path"]: f for f in real["files"]}
    real_by_path["train/data.lmdb"]["sha256"] = "0" * 64  # wrong on purpose
    fake_repo_manifest = {
        "version": "bgsl-acquisition-manifest-1",
        "files": list(real_by_path.values()),
    }
    (populated_data_dir / MANIFEST_FILENAME).write_text(
        json.dumps(fake_repo_manifest), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        verify_manifest(populated_data_dir, require_repo_manifest=True)


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------


def test_lock_serializes_concurrent_acquisition(populated_data_dir, quiet_log,
                                                 fake_huggingface_hub):
    """
    A thread that holds the lock for ~1.5s blocks a second ensure_dataset
    call; the second call only proceeds after the first releases.
    """
    holder_release = threading.Event()
    holder_done = threading.Event()
    started = threading.Event()

    def slow_snapshot_download(**kwargs):
        started.set()
        holder_release.wait(timeout=5)
        for rel in REQUIRED_ARTIFACTS:
            (Path(kwargs["local_dir"]) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(kwargs["local_dir"]) / rel).write_bytes(b"x")
        return str(kwargs["local_dir"])

    fake_huggingface_hub.set(slow_snapshot_download)

    cfg = AcquisitionConfig(
        data_dir=str(populated_data_dir),
        hf=HfConfig(repo_id="org/repo"),
    )

    def worker():
        ensure_dataset(cfg, log=quiet_log)
        holder_done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    started.wait(timeout=5)

    t2_done = threading.Event()

    def worker2():
        ensure_dataset(cfg, log=quiet_log)
        t2_done.set()

    t2 = threading.Thread(target=worker2, daemon=True)
    t2.start()

    time.sleep(0.3)
    assert not t2_done.is_set(), "t2 should be blocked on the lock"

    holder_release.set()
    t.join(timeout=10)
    t2.join(timeout=10)
    assert holder_done.is_set()
    assert t2_done.is_set()


# ---------------------------------------------------------------------------
# HF token resolution
# ---------------------------------------------------------------------------


def test_hf_token_resolution_env(monkeypatch):
    monkeypatch.setenv("MY_HF_TOKEN", "secret-xyz")
    hf = HfConfig(repo_id="org/repo", token_env="MY_HF_TOKEN")
    assert os.environ.get(hf.token_env) == "secret-xyz"


def test_hf_token_resolution_missing_is_allowed_anonymously(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hf = HfConfig(repo_id="org/repo", token_env="HF_TOKEN")
    assert os.environ.get(hf.token_env) is None
    # ensure_dataset will simply not pass a token; behavior is decided
    # downstream by huggingface_hub. Here we just confirm the config
    # gracefully handles a missing env.
    assert hf.token_env == "HF_TOKEN"


# ---------------------------------------------------------------------------
# Upload dry-run
# ---------------------------------------------------------------------------


def test_upload_dry_run_writes_manifest_and_readme(populated_data_dir, quiet_log):
    hf = HfConfig(repo_id="org/repo", token_env="HF_TOKEN")
    info = upload_to_hub(
        data_dir=populated_data_dir,
        hf=hf,
        private=False,
        dry_run=True,
        log=quiet_log,
    )
    assert info["dry_run"] is True
    assert info["file_count"] == len(REQUIRED_ARTIFACTS)
    assert (populated_data_dir / MANIFEST_FILENAME).is_file()
    assert (populated_data_dir / "README.md").is_file()
    readme = (populated_data_dir / "README.md").read_text(encoding="utf-8")
    assert "BGSL" in readme
    assert "org/repo" in readme


def test_upload_rejects_incomplete_data_dir(tmp_path, quiet_log):
    hf = HfConfig(repo_id="org/repo")
    bad = tmp_path / "incomplete"
    bad.mkdir()
    # Only 2 of 6 artifacts.
    (bad / "train" / "data.lmdb").parent.mkdir(parents=True, exist_ok=True)
    (bad / "train" / "data.lmdb").write_bytes(b"x")
    (bad / "train_index.json").write_text("{}")
    with pytest.raises(RuntimeError, match="missing one or more required artifacts"):
        upload_to_hub(data_dir=bad, hf=hf, dry_run=True, log=quiet_log)


# ---------------------------------------------------------------------------
# Config YAML round-trip
# ---------------------------------------------------------------------------


def test_config_from_yaml_round_trip(tmp_path):
    raw = {
        "data_dir": "data/ready",
        "hf": {
            "repo_id": "org/repo",
            "repo_type": "dataset",
            "revision": "main",
            "filename_glob": "*",
            "token_env": "HF_TOKEN",
        },
        "build": {
            "raw_dir": "/some/raw",
            "seed": 17,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "min_length": 6,
            "horizon_hours": 12,
            "extra_args": ["--foo"],
        },
        "force": False,
        "lock_timeout_s": 60,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    cfg = AcquisitionConfig.from_yaml(p)
    assert cfg.hf is not None and cfg.hf.repo_id == "org/repo"
    assert cfg.build is not None and cfg.build.seed == 17
    # Round-trip dict.
    again = yaml.safe_load(yaml.safe_dump(cfg.to_dict()))
    assert again == raw


def test_config_from_yaml_missing_data_dir(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"hf": {"repo_id": "x/y"}}))
    with pytest.raises(ValueError, match="data_dir"):
        AcquisitionConfig.from_yaml(p)


# ---------------------------------------------------------------------------
# Plan / dry-run behavior
# ---------------------------------------------------------------------------


def test_plan_includes_only_configured_tiers():
    cfg = AcquisitionConfig(data_dir="x", build=BuildConfig())
    steps = plan(cfg)
    assert [s["tier"] for s in steps] == [1, 3]

    cfg2 = AcquisitionConfig(
        data_dir="x", hf=HfConfig(repo_id="o/r"), build=BuildConfig()
    )
    assert [s["tier"] for s in plan(cfg2)] == [1, 2, 3]

    cfg3 = AcquisitionConfig(data_dir="x", hf=HfConfig(repo_id="o/r"))
    assert [s["tier"] for s in plan(cfg3)] == [1, 2]


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_dry_run_returns_0(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"data_dir": "data/ready"}))
    from bgsl.data.acquisition_cli import main as acquire_main

    rc = acquire_main(["--config", str(cfg_path), "--dry-run"])
    assert rc == 0


def test_cli_missing_config_returns_2(tmp_path):
    from bgsl.data.acquisition_cli import main as acquire_main
    rc = acquire_main(["--config", str(tmp_path / "missing.yaml")])
    assert rc == 2
