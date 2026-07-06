"""
bgsl/empirical/data/synthetic.py
---------------------------------
Synthetic ICU data generation, dataset, and Lightning DataModule for
controlled BGSL empirical studies.

Design
------
- SyntheticICUConfig   — frozen dataclass holding all generator parameters.
- SyntheticICUGenerator — stateless generator; deterministic for a given
  config + seed. Migrated from tests/test_synthetic_benchmark.py.
- SyntheticICUDataset  — PyTorch Dataset that loads cached arrays and
  computes BGSL soft targets on-the-fly, matching the PhysioNet2019Dataset
  batch contract exactly.
- SyntheticICUDataModule — LightningCLI-compatible DataModule that auto-
  generates and caches the cohort on first access.

Caching
-------
Generated cohorts are written to::

    datasets/synthetic/bgsl_v1/<scenario>_seed<seed>/
        manifest.json      — generator params, schema version, shapes
        features.pt        — [N, T_max, C] float32
        masks.pt           — [N, T_max, C] float32
        meta.pt            — dict with onset_times, is_positive,
                             seq_lengths, split_indices

Batch contract (all keys match PhysioNet2019Dataset.__getitem__)
---------------------------------------------------------------
vitals, masks, hard_labels, soft_targets, hard_targets,
vel_targets, accel_targets, valid_mask, vel_mask, acc_mask,
onset_hour, is_sepsis, seq_len, patient_id, static_covariates
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from bgsl.core.common.targets import BaseSoftOnsetTarget

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIO_SEEDS: Dict[str, int] = {"signal": 101, "null": 202}
DEFAULT_MISSING_RATES: Dict[str, float] = {"signal": 0.18, "null": 0.15}
SCHEMA_VERSION: str = "bgsl-synthetic-v2"
STATIC_COVARIATE_DIM: int = 1  # single constant placeholder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticICUConfig:
    """Frozen generator parameters for the synthetic ICU benchmark.

    All parameters are fixed at generation time and recorded in the
    cohort ``manifest.json`` for full reproducibility.
    """

    mode: str = "signal"
    seed: int = 101
    n_patients: int = 800
    n_features: int = 16
    max_length: int = 48
    min_length: int = 18
    horizon: float = 6.0
    pos_ratio: float = 0.28
    informative_channels: int = 4
    missing_rate: float = 0.18
    signal_strength: float = 0.45
    tau: float = 2.5
    kappa: float = 1.0
    train_frac: float = 0.6
    val_frac: float = 0.2

    def __post_init__(self) -> None:
        if self.mode not in ("signal", "null"):
            raise ValueError(f"mode must be 'signal' or 'null', got {self.mode!r}")
        if self.n_patients <= 0:
            raise ValueError(f"n_patients must be positive, got {self.n_patients}")
        if self.n_features <= 0:
            raise ValueError(f"n_features must be positive, got {self.n_features}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.min_length <= 0 or self.min_length > self.max_length:
            raise ValueError(
                f"min_length must satisfy 0 < min_length <= max_length, got "
                f"{self.min_length} and {self.max_length}"
            )
        if self.horizon <= 0.0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.tau <= 0.0:
            raise ValueError(f"tau must be positive, got {self.tau}")
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError(f"train_frac must be in (0, 1), got {self.train_frac}")
        if not (0.0 < self.val_frac < 1.0):
            raise ValueError(f"val_frac must be in (0, 1), got {self.val_frac}")
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError(
                f"train_frac + val_frac must be < 1, got "
                f"{self.train_frac + self.val_frac}"
            )
        if not (0.0 < self.pos_ratio < 1.0):
            raise ValueError(f"pos_ratio must be in (0, 1), got {self.pos_ratio}")
        if not (0.0 <= self.missing_rate < 1.0):
            raise ValueError(f"missing_rate must be in [0, 1), got {self.missing_rate}")
        if not (1 <= self.informative_channels <= self.n_features):
            raise ValueError(
                f"informative_channels must be in [1, n_features], "
                f"got {self.informative_channels}"
            )
        if self.signal_strength <= 0.0:
            raise ValueError(f"signal_strength must be positive, got {self.signal_strength}")


# ---------------------------------------------------------------------------
# AR(1) noise
# ---------------------------------------------------------------------------


def _ar1_noise(
    length: int,
    *,
    phi: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an AR(1) process of length ``length`` with coefficient ``phi``."""
    eps = rng.normal(0.0, scale, size=length)
    out = np.zeros(length, dtype=np.float32)
    out[0] = eps[0]
    coeff = float(np.sqrt(max(1.0 - phi * phi, 0.0)))
    for t in range(1, length):
        out[t] = phi * out[t - 1] + coeff * eps[t]
    return out


def _split_indices(
    n: int,
    *,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train / val / test split using numpy."""
    if train_frac <= 0.0 or val_frac <= 0.0 or train_frac + val_frac >= 1.0:
        raise ValueError("Invalid split fractions.")
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_idx = idx[:n_train]
    val_idx = idx[n_train: n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def _config_fingerprint(config: SyntheticICUConfig) -> str:
    """Stable fingerprint for cache validation and manifest comparison."""
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticICUGenerator:
    """Generates a synthetic ICU cohort with controllable signal structure.

    The generator is stateless: all randomness is driven by the explicit
    ``seed`` in the config.  Calling ``generate()`` twice with the same
    config produces identical output.
    """

    def __init__(self, config: SyntheticICUConfig) -> None:
        self.config = config

    def generate(self) -> Dict[str, np.ndarray]:
        """Return the full cohort as a dict of numpy arrays.

        Keys
        ----
        features      : float32 [N, T_max, C]  — raw vitals
        masks         : float32 [N, T_max, C]  — observation mask
        soft_targets  : float32 [N, T_max]     — g_t
        vel_targets   : float32 [N, T_max]     — g'_t
        hard_targets  : float32 [N, T_max]     — hard early-warning label
        valid_mask    : float32 [N, T_max]     — valid-timestep mask
        onset_times   : float32 [N]            — t* (-1 if no event)
        is_positive   : bool     [N]            — event flag
        seq_lengths   : int64    [N]            — valid length per patient
        split_indices : dict with keys 'train', 'val', 'test'
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n = cfg.n_patients
        t_max = cfg.max_length

        # ------------------------------------------------------------------
        # Sequence lengths and onset times
        # ------------------------------------------------------------------
        seq_lengths = rng.integers(cfg.min_length, t_max + 1, size=n, dtype=np.int64)
        is_positive = rng.random(n) < cfg.pos_ratio
        onset_times = np.full(n, -1.0, dtype=np.float32)

        for i in range(n):
            if not is_positive[i]:
                continue
            lower = int(np.ceil(cfg.horizon)) + 2
            upper = int(seq_lengths[i]) - 2
            if upper <= lower:
                is_positive[i] = False
                continue
            onset_times[i] = float(rng.integers(lower, upper + 1))

        # ------------------------------------------------------------------
        # Soft targets (BGSL targets computed once for efficiency)
        # ------------------------------------------------------------------
        target_builder = BaseSoftOnsetTarget(
            horizon=cfg.horizon, tau=cfg.tau, kappa=cfg.kappa,
        )
        target_batch = target_builder(
            torch.tensor(onset_times, dtype=torch.float32),
            torch.tensor(seq_lengths, dtype=torch.long),
            torch.device("cpu"),
        )
        soft_targets = target_batch["soft_target"].cpu().numpy().astype(np.float32)
        vel_targets = target_batch["velocity_target"].cpu().numpy().astype(np.float32)
        accel_targets = target_batch["accel_target"].cpu().numpy().astype(np.float32)
        hard_targets = target_batch["hard_target"].cpu().numpy().astype(np.float32)
        valid_mask = target_batch["valid_mask"].cpu().numpy().astype(np.float32)
        vel_mask = target_batch["vel_mask"].cpu().numpy().astype(np.float32)
        acc_mask = target_batch["acc_mask"].cpu().numpy().astype(np.float32)

        # ------------------------------------------------------------------
        # Feature construction
        # ------------------------------------------------------------------
        latent = np.zeros((n, t_max), dtype=np.float32)
        for i in range(n):
            T_i = int(seq_lengths[i])
            latent[i, :T_i] = soft_targets[i, :T_i] if cfg.mode == "signal" else 0.0

        patient_offset = rng.normal(0.0, 0.35, size=n).astype(np.float32)
        patient_gain = rng.uniform(0.6, 1.4, size=n).astype(np.float32)
        nuisance_state = rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)

        features = np.zeros((n, t_max, cfg.n_features), dtype=np.float32)
        masks = np.zeros((n, t_max, cfg.n_features), dtype=np.float32)

        for i in range(n):
            T_i = int(seq_lengths[i])
            state = latent[i, :T_i]
            state_delta = np.diff(
                np.concatenate([[state[0]], state])
            ).astype(np.float32)
            noise_bank = {
                "slow": _ar1_noise(T_i, phi=0.85, scale=0.12, rng=rng),
                "mid": _ar1_noise(T_i, phi=0.45, scale=0.16, rng=rng),
                "fast": _ar1_noise(T_i, phi=0.15, scale=0.22, rng=rng),
            }

            # Informative channels (0-3) carry signal + noise
            features[i, :T_i, 0] = patient_offset[i] + patient_gain[i] * (
                cfg.signal_strength * (0.55 * state) + noise_bank["mid"]
            )
            features[i, :T_i, 1] = patient_offset[i] + patient_gain[i] * (
                cfg.signal_strength * (0.35 * state + 0.40 * state_delta)
                + noise_bank["slow"]
            )
            features[i, :T_i, 2] = patient_offset[i] + patient_gain[i] * (
                cfg.signal_strength * (0.20 * state) + noise_bank["fast"]
            )
            features[i, :T_i, 3] = patient_offset[i] + patient_gain[i] * (
                cfg.signal_strength * (0.25 * np.maximum(state, 0.0))
                + noise_bank["mid"]
            )

            # Nuisance channels (4+)
            for j in range(4, cfg.n_features):
                block = j % 3
                if block == 0:
                    signal = nuisance_state[i, 0] + noise_bank["slow"]
                elif block == 1:
                    signal = nuisance_state[i, 1] + noise_bank["mid"]
                else:
                    signal = nuisance_state[i, 2] + noise_bank["fast"]
                features[i, :T_i, j] = signal

            # Missingness pattern
            for j in range(cfg.n_features):
                base_missing = rng.random(T_i) < cfg.missing_rate
                if j < cfg.informative_channels and is_positive[i]:
                    center = int(np.clip(
                        round(onset_times[i] - cfg.horizon / 2.0),
                        0, max(T_i - 1, 0),
                    ))
                    local = np.abs(np.arange(T_i) - center) <= max(
                        2, int(cfg.horizon // 2)
                    )
                    dropout = rng.random(T_i) < (0.35 * local + 0.10)
                    observed = ~(base_missing | dropout)
                else:
                    observed = ~base_missing
                masks[i, :T_i, j] = observed.astype(np.float32)
                features[i, :T_i, j] *= masks[i, :T_i, j]

        # ------------------------------------------------------------------
        # Split indices
        # ------------------------------------------------------------------
        split_seed = cfg.seed  # scenario seed used for splitting
        train_idx, val_idx, test_idx = _split_indices(
            n, seed=split_seed,
            train_frac=cfg.train_frac, val_frac=cfg.val_frac,
        )

        return {
            "features": features,
            "masks": masks,
            "soft_targets": soft_targets,
            "vel_targets": vel_targets,
            "accel_targets": accel_targets,
            "hard_targets": hard_targets,
            "valid_mask": valid_mask,
            "vel_mask": vel_mask,
            "acc_mask": acc_mask,
            "onset_times": onset_times.astype(np.float32),
            "is_positive": is_positive,
            "seq_lengths": seq_lengths,
            "split_indices": {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            },
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_SPLIT_KEY_MAP: Dict[str, str] = {
    "train": "train",
    "val": "val",
    "test": "test",
}


class SyntheticICUDataset(Dataset):
    """PyTorch Dataset over a cached synthetic ICU cohort.

    Loads pre-generated arrays (including BGSL target tensors) from disk.
    Uses cached targets directly — no on-the-fly target recomputation.
    """

    _TARGET_KEYS: tuple = (
        "soft_targets", "vel_targets", "accel_targets", "hard_targets",
        "valid_mask", "vel_mask", "acc_mask",
    )

    def __init__(
        self,
        cohort_dir: str,
        split: str,
        horizon_hours: int = 6,
        tau: float = 2.5,
    ) -> None:
        self.cohort_dir = Path(cohort_dir)
        self.split = split

        # Load cached arrays
        features = torch.load(self.cohort_dir / "features.pt", map_location="cpu")
        masks = torch.load(self.cohort_dir / "masks.pt", map_location="cpu")
        meta = torch.load(self.cohort_dir / "meta.pt", map_location="cpu")

        # Load pre-computed target tensors
        targets = {}
        for key in self._TARGET_KEYS:
            targets[key] = torch.load(self.cohort_dir / f"{key}.pt", map_location="cpu")

        # Select split
        split_key = _SPLIT_KEY_MAP[split]
        self.indices = meta["split_indices"][split_key].numpy()

        self.features = features[self.indices]  # [N_split, T_max, C]
        self.masks = masks[self.indices]
        self.targets = {k: v[self.indices] for k, v in targets.items()}
        self.onset_times = meta["onset_times"][self.indices]
        self.is_positive = meta["is_positive"][self.indices]
        self.seq_lengths = meta["seq_lengths"][self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        T = int(self.seq_lengths[idx].item())

        # Slice to actual length so feature and target dimensions match
        x = self.features[idx, :T]       # [T, C]
        m = self.masks[idx, :T]          # [T, C]
        onset_h = int(self.onset_times[idx].item())
        is_sep = bool(self.is_positive[idx].item())

        # Slice pre-computed targets
        tgt = {k: v[idx, :T] for k, v in self.targets.items()}

        return {
            "vitals":            x,                    # [T, C]
            "masks":             m,                    # [T, C]
            "hard_labels":       tgt["hard_targets"],  # [T]
            "soft_targets":      tgt["soft_targets"],  # [T]
            "hard_targets":      tgt["hard_targets"],  # [T]
            "vel_targets":       tgt["vel_targets"],   # [T]
            "accel_targets":     tgt["accel_targets"], # [T]
            "valid_mask":        tgt["valid_mask"],    # [T]
            "vel_mask":          tgt["vel_mask"],      # [T]
            "acc_mask":          tgt["acc_mask"],      # [T]
            "onset_hour":        torch.tensor(onset_h, dtype=torch.long),
            "is_sepsis":         torch.tensor(is_sep, dtype=torch.bool),
            "seq_len":           torch.tensor(T, dtype=torch.long),
            "patient_id":        f"synthetic_{int(self.indices[idx]):04d}",
            "static_covariates": torch.zeros(STATIC_COVARIATE_DIM),
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_synthetic(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length synthetic trajectories to the batch maximum.

    Mirrors ``collate_trajectories`` from the PhysioNet pipeline.  Since
    each patient's sequence has a different actual length, time-indexed
    tensors are right-padded to ``T_max = max(seq_len)`` within the batch.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    T_max = max(b["vitals"].shape[0] for b in batch)
    B = len(batch)

    # Time-indexed keys (need padding)
    time_keys = [
        "vitals", "masks", "hard_labels", "soft_targets", "hard_targets",
        "vel_targets", "accel_targets", "valid_mask", "vel_mask", "acc_mask",
    ]
    out: Dict[str, torch.Tensor] = {}

    for key in time_keys:
        sample = batch[0][key]
        extra_dims = sample.shape[1:]
        padded = torch.zeros(B, T_max, *extra_dims, dtype=sample.dtype)
        for i, b in enumerate(batch):
            t = b[key].shape[0]
            padded[i, :t] = b[key]
        out[key] = padded

    # Scalar keys
    for key in ["onset_hour", "is_sepsis", "seq_len"]:
        out[key] = torch.stack([b[key] for b in batch])

    # Vector keys
    out["static_covariates"] = torch.stack([b["static_covariates"] for b in batch])

    # String keys
    out["patient_id"] = [b["patient_id"] for b in batch]

    return out


# ---------------------------------------------------------------------------
# Data Module
# ---------------------------------------------------------------------------


def _cohort_dir(root: str, scenario: str, tau: float = 2.5) -> Path:
    """Resolve the full cohort cache path for a given scenario.
    
    ``tau`` is included in the path. Other configuration changes are
    guarded by the manifest fingerprint.
    """
    seed = SCENARIO_SEEDS[scenario]
    tau_str = f"tau{tau}".replace(".", "_")
    return Path(root) / f"{scenario}_seed{seed}_{tau_str}"


def _generate_manifest(config: SyntheticICUConfig, cohort: Dict[str, Any]) -> Dict[str, Any]:
    """Build the manifest dict for a generated cohort."""
    n_sepsis = int(cohort["is_positive"].sum())
    n_non = config.n_patients - n_sepsis
    return {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": _config_fingerprint(config),
        "scenario": config.mode,
        "scenario_seed": config.seed,
        "generator_class": "bgsl.empirical.data.synthetic.SyntheticICUGenerator",
        "generator_config": asdict(config),
        "shapes": {
            "features": list(cohort["features"].shape),
            "masks": list(cohort["masks"].shape),
        },
        "n_patients": config.n_patients,
        "n_sepsis": n_sepsis,
        "n_non_sepsis": n_non,
        "split_sizes": {
            "train": int(len(cohort["split_indices"]["train"])),
            "val": int(len(cohort["split_indices"]["val"])),
            "test": int(len(cohort["split_indices"]["test"])),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _manifest_matches_config(manifest_path: Path, config: SyntheticICUConfig) -> bool:
    """Return True when the cached manifest exactly matches ``config``."""
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("config_fingerprint") == _config_fingerprint(config)
    )


def _generate_and_cache(config: SyntheticICUConfig, dest: Path) -> None:
    """Generate a cohort and write cached files + manifest."""
    dest.mkdir(parents=True, exist_ok=True)

    generator = SyntheticICUGenerator(config)
    cohort = generator.generate()

    # Cache tensors
    torch.save(torch.from_numpy(cohort["features"]), dest / "features.pt")
    torch.save(torch.from_numpy(cohort["masks"]), dest / "masks.pt")

    # Cache pre-computed BGSL target tensors (avoids recomputation in Dataset)
    for key in ["soft_targets", "vel_targets", "accel_targets", "hard_targets",
                "valid_mask", "vel_mask", "acc_mask"]:
        torch.save(torch.from_numpy(cohort[key]), dest / f"{key}.pt")

    # Metadata (convert numpy arrays to tensors for torch.save compatibility)
    meta = {
        "onset_times": torch.from_numpy(cohort["onset_times"]),
        "is_positive": torch.from_numpy(cohort["is_positive"]),
        "seq_lengths": torch.from_numpy(cohort["seq_lengths"]),
        "split_indices": {
            k: torch.from_numpy(v)
            for k, v in cohort["split_indices"].items()
        },
    }
    torch.save(meta, dest / "meta.pt")

    # Manifest
    manifest = _generate_manifest(config, cohort)
    with open(dest / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[synthetic] Cached cohort to {dest}")
    print(f"            {config.n_patients} patients "
          f"({manifest['n_sepsis']} sepsis, {manifest['n_non_sepsis']} non-sepsis)")


class SyntheticICUDataModule(pl.LightningDataModule):
    """Lightning DataModule for the synthetic ICU benchmark.

    Auto-generates and caches the cohort on first ``setup()`` call.
    Compatible with LightningCLI subclass-mode data.
    """

    def __init__(
        self,
        data_dir: str = "datasets/synthetic/bgsl_v1",
        scenario: str = "signal",
        horizon_hours: int = 6,
        tau: float = 2.5,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        # Generation overrides (applied on first access only)
        n_patients: int = 800,
        n_features: int = 16,
        signal_strength: float = 0.45,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds: Optional[SyntheticICUDataset] = None
        self.val_ds: Optional[SyntheticICUDataset] = None
        self.test_ds: Optional[SyntheticICUDataset] = None

    @property
    def feature_dim(self) -> int:
        """Number of dynamic features (16 for synthetic benchmark)."""
        return self.hparams.n_features

    def _ensure_data(self) -> Path:
        """Generate and cache the cohort if it does not already exist.

        Returns the resolved cohort directory path.
        """
        h = self.hparams
        dest = _cohort_dir(h.data_dir, h.scenario, float(h.tau))
        missing_rate = DEFAULT_MISSING_RATES.get(h.scenario, 0.18)
        scenario_seed = SCENARIO_SEEDS[h.scenario]
        config = SyntheticICUConfig(
            mode=h.scenario,
            seed=scenario_seed,
            n_patients=h.n_patients,
            n_features=h.n_features,
            missing_rate=missing_rate,
            signal_strength=h.signal_strength,
            tau=h.tau,
            horizon=h.horizon_hours,
        )
        if not _manifest_matches_config(dest / "manifest.json", config):
            _generate_and_cache(config, dest)
        return dest

    def setup(self, stage: Optional[str] = None) -> None:
        cohort_dir = self._ensure_data()
        ds_kwargs = {
            "cohort_dir": str(cohort_dir),
            "horizon_hours": self.hparams.horizon_hours,
            "tau": self.hparams.tau,
        }
        if stage == "fit" or stage is None:
            self.train_ds = SyntheticICUDataset(split="train", **ds_kwargs)
            self.val_ds = SyntheticICUDataset(split="val", **ds_kwargs)
        if stage == "test" or stage is None:
            self.test_ds = SyntheticICUDataset(split="test", **ds_kwargs)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_synthetic,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_synthetic,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_synthetic,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
