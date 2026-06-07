"""
datasets/physionet2019.py
--------------------------
PhysioNet/CinC 2019 Sepsis Challenge — BGSL data pipeline.

Key differences from APEX-MoE icu/datasets/build_dataset.py
-------------------------------------------------------------
1. Stores FULL patient trajectories in LMDB (not sliding windows).
   BGSL needs the entire trajectory to compute trajectory metrics.
2. Pre-computes onset_hour (t*) per patient and stores in the index.
3. Test split: PhysioNet provides a separate test set (Set B).
4. Patient-level splits: reproducible based on patient filename.
5. BGSL soft targets (g_t, Δg_t, Δ²g_t) are computed on-the-fly in
   __getitem__ using SoftOnsetTarget, not stored in LMDB (saves space,
   supports multiple tau at eval without rebuilding).
6. Self-contained: tries kagglehub, falls back to local data/raw/ directory.
7. Causal imputation only: forward fill, then default fill for leading gaps.
   No backward fill.

Schema
------
LMDB keys per episode:
    {ep_id}_vitals : float32 [T, 40]  — all 40 Clinical features
    {ep_id}_masks  : float32 [T, 40]  — 1=observed, 0=imputed
    {ep_id}_labels : float32 [T]      — binary SepsisLabel

Index JSON per split:
    episodes: [{episode_id, patient_id, length, onset_hour, is_sepsis, modalities}]
    metadata: {version, ts_columns, stats}


Schema
------
LMDB keys per episode:
    {ep_id}_vitals : float32 [T, 40]  — all 40 Clinical features
    {ep_id}_masks  : float32 [T, 40]  — 1=observed, 0=imputed
    {ep_id}_labels : float32 [T]      — binary SepsisLabel

Index JSON per split:
    episodes: [{episode_id, patient_id, length, onset_hour, is_sepsis, modalities}]
    metadata: {version, ts_columns, stats}

Reference: Reyna et al. Critical Care Medicine 2020.
           plan.md §5.1, §6, §7, §18 (Month 1)
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from bgsl.core.sepsis.targets import SoftOnsetTarget

__all__ = ["PhysioNet2019Dataset", "build_physionet_lmdb", "collate_trajectories"]

logger = logging.getLogger("bgsl.physionet2019")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LMDB_MAP_SIZE = 10 * 1024 ** 3   # 10 GB
RESERVOIR_SIZE = 200_000
SEED = 2025

# Clinical 40 feature specification (PhysioNet 2019 official column order)
# Ref: https://physionet.org/content/challenge-2019/1.0.0/
CLINICAL_SPECS: Dict[int, Dict] = {
    # Vital signs (indices 0-7)
    0:  {"col": "HR",            "default": 75.0,  "name": "HR"},
    1:  {"col": "O2Sat",         "default": 98.0,  "name": "O2Sat"},
    2:  {"col": "Temp",          "default": 37.0,  "name": "Temp"},
    3:  {"col": "SBP",           "default": 120.0, "name": "SBP"},
    4:  {"col": "MAP",           "default": 93.0,  "name": "MAP"},
    5:  {"col": "DBP",           "default": 80.0,  "name": "DBP"},
    6:  {"col": "Resp",          "default": 16.0,  "name": "Resp"},
    7:  {"col": "EtCO2",         "default": 40.0,  "name": "EtCO2"},
    # Laboratory values (indices 8-33)
    8:  {"col": "BaseExcess",    "default": 0.0,   "name": "BaseExcess"},
    9:  {"col": "HCO3",          "default": 24.0,  "name": "HCO3"},
    10: {"col": "FiO2",          "default": 0.21,  "name": "FiO2"},
    11: {"col": "pH",            "default": 7.4,   "name": "pH"},
    12: {"col": "PaCO2",         "default": 40.0,  "name": "PaCO2"},
    13: {"col": "SaO2",          "default": 98.0,  "name": "SaO2"},
    14: {"col": "AST",           "default": 25.0,  "name": "AST"},
    15: {"col": "BUN",           "default": 15.0,  "name": "BUN"},
    16: {"col": "Alkalinephos",  "default": 80.0,  "name": "Alkalinephos"},
    17: {"col": "Calcium",       "default": 9.5,   "name": "Calcium"},
    18: {"col": "Chloride",      "default": 102.0, "name": "Chloride"},
    19: {"col": "Creatinine",    "default": 1.0,   "name": "Creatinine"},
    20: {"col": "Bilirubin_direct","default": 0.2, "name": "Bilirubin_direct"},
    21: {"col": "Glucose",       "default": 100.0, "name": "Glucose"},
    22: {"col": "Lactate",       "default": 1.0,   "name": "Lactate"},
    23: {"col": "Magnesium",     "default": 2.0,   "name": "Magnesium"},
    24: {"col": "Phosphate",     "default": 4.0,   "name": "Phosphate"},
    25: {"col": "Potassium",     "default": 4.0,   "name": "Potassium"},
    26: {"col": "Bilirubin_total","default": 0.6,  "name": "Bilirubin"},
    27: {"col": "TroponinI",     "default": 0.0,   "name": "TroponinI"},
    28: {"col": "Hct",           "default": 42.0,  "name": "Hct"},
    29: {"col": "Hgb",           "default": 14.0,  "name": "Hgb"},
    30: {"col": "PTT",           "default": 30.0,  "name": "PTT"},
    31: {"col": "WBC",           "default": 9.0,   "name": "WBC"},
    32: {"col": "Fibrinogen",    "default": 300.0, "name": "Fibrinogen"},
    33: {"col": "Platelets",     "default": 250.0, "name": "Platelets"},
    # Demographics (indices 34-39)
    34: {"col": "Age",           "default": 60.0,  "name": "Age"},
    35: {"col": "Gender",        "default": 1.0,   "name": "Gender"},
    36: {"col": "Unit1",         "default": 0.0,   "name": "Unit1"},
    37: {"col": "Unit2",         "default": 0.0,   "name": "Unit2"},
    38: {"col": "HospAdmTime",   "default": -10.0, "name": "HospAdmTime"},
    39: {"col": "ICULOS",        "default": 1.0,   "name": "ICULOS"},
}
FEATURE_NAMES = [CLINICAL_SPECS[i]["name"] for i in range(40)]

PHYSICS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HR": (20.0, 300.0), "O2Sat": (20.0, 100.0), "Temp": (24.0, 45.0),
    "SBP": (20.0, 300.0), "MAP": (20.0, 250.0), "DBP": (10.0, 200.0),
    "Resp": (4.0, 80.0), "EtCO2": (10.0, 80.0),
    "BaseExcess": (-20.0, 30.0), "HCO3": (5.0, 60.0), "FiO2": (0.21, 1.0),
    "pH": (6.5, 7.8), "PaCO2": (10.0, 150.0), "SaO2": (20.0, 100.0),
    "AST": (5.0, 10000.0), "BUN": (1.0, 250.0),
    "Alkalinephos": (10.0, 5000.0), "Calcium": (2.0, 20.0),
    "Chloride": (50.0, 150.0), "Creatinine": (0.1, 25.0),
    "Bilirubin_direct": (0.1, 30.0), "Glucose": (10.0, 1200.0),
    "Lactate": (0.1, 30.0), "Magnesium": (0.5, 10.0),
    "Phosphate": (0.5, 15.0), "Potassium": (1.0, 12.0),
    "Bilirubin": (0.1, 80.0), "TroponinI": (0.0, 100.0),
    "Hct": (10.0, 60.0), "Hgb": (2.0, 25.0), "PTT": (10.0, 200.0),
    "WBC": (0.1, 200.0), "Fibrinogen": (50.0, 1000.0),
    "Platelets": (1.0, 2000.0),
}

MIN_STAY_HOURS = 8    # Default minimum valid ICU stay length
MAX_STAY_HOURS = 336  # Default 14 days cap (removes ultra-long stays that are outliers)
DATASET_FORMAT_VERSION = "bgsl-2.0"
CAUSAL_IMPUTATION_STRATEGY = "forward_fill_then_default"

# Static covariate indices within the 40-channel vitals array.
# These are patient-level constants (not time-varying) and are used
# by the HyperNetwork for patient-specific target shape parameters.
STATIC_COVARIATE_INDICES = [34, 35, 36, 37, 38]  # Age, Gender, Unit1, Unit2, HospAdmTime


def _causal_impute_series(raw: np.ndarray, default: float) -> pd.Series:
    """Impute missing values without peeking into future time steps."""
    return pd.Series(raw).ffill().fillna(default)


def _validate_index_metadata(metadata: Dict, *, index_path: Path) -> None:
    """Reject indices built before the causal-imputation fix."""
    version = metadata.get("version")
    preprocessing = metadata.get("preprocessing", {})
    imputation = preprocessing.get("imputation")
    causal = preprocessing.get("causal")

    if (
        version != DATASET_FORMAT_VERSION
        or imputation != CAUSAL_IMPUTATION_STRATEGY
        or causal is not True
    ):
        raise RuntimeError(
            "Legacy PhysioNet index detected at "
            f"{index_path}. Expected metadata.version={DATASET_FORMAT_VERSION!r} "
            f"and preprocessing.imputation={CAUSAL_IMPUTATION_STRATEGY!r}, but got "
            f"version={version!r}, imputation={imputation!r}, causal={causal!r}. "
            "Rebuild the processed dataset before training."
        )


# ---------------------------------------------------------------------------
# Build pipeline (LMDB ingestor)
# ---------------------------------------------------------------------------

class _Ingestor:
    """
    Ingests raw PhysioNet .psv files and serializes to LMDB.

    Stores per-patient:
        - Full vitals matrix [T, 40]
        - Imputation mask [T, 40]
        - Binary SepsisLabel [T]
        - onset_hour in the JSON index

    Causal imputation:
        Forward fill only (no future-information leakage).
        Backward fill ONLY for the very first hour (patient had baseline values
        before ICU admission — this is clinically standard and used by PhysioNet).
        Population median fill for features never measured during the stay.
    """

    def __init__(
        self,
        out_dir: str,
        split: str,
        min_stay_hours: int = MIN_STAY_HOURS,
        max_stay_hours: int = MAX_STAY_HOURS,
        causal_imputation_strategy: str = CAUSAL_IMPUTATION_STRATEGY,
    ) -> None:
        self.split_dir = Path(out_dir) / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.lmdb_path = self.split_dir / "data.lmdb"
        self.env = lmdb.open(str(self.lmdb_path), map_size=LMDB_MAP_SIZE, subdir=False)
        self.min_stay_hours = min_stay_hours
        self.max_stay_hours = max_stay_hours
        self.causal_imputation_strategy = causal_imputation_strategy

        self.index: List[Dict] = []
        self.ep_cnt = 0
        self.reservoir: List[np.ndarray] = []

        # Running min/max for fallback
        self.global_min = np.full(40, np.inf)
        self.global_max = np.full(40, -np.inf)

    def process(self, files: List[str]) -> None:
        with self.env.begin(write=True) as txn:
            for fpath in files:
                try:
                    self._process_one(fpath, txn)
                except Exception as e:
                    logger.debug("Skipping %s: %s", Path(fpath).name, e)

        self.env.close()
        self._save_index()

    def _process_one(self, fpath: str, txn: lmdb.Transaction) -> None:
        df = pd.read_csv(fpath, sep="|")
        L = len(df)

        # Minimum stay filter
        if L < self.min_stay_hours:
            return

        # Cap extremely long stays
        if L > self.max_stay_hours:
            df = df.iloc[:self.max_stay_hours]
            L = self.max_stay_hours

        vitals = np.zeros((L, 40), dtype=np.float32)
        masks = np.zeros((L, 40), dtype=np.float32)

        for i in range(40):
            spec = CLINICAL_SPECS[i]
            col, default, name = spec["col"], spec["default"], spec["name"]

            raw = df[col].values.astype(float) if col in df.columns else np.full(L, np.nan)

            # Physics bounds: out-of-range → treat as sensor error → NaN
            if name in PHYSICS_BOUNDS:
                lo, hi = PHYSICS_BOUNDS[name]
                raw = np.where((raw < lo) | (raw > hi), np.nan, raw)

            # Missingness mask BEFORE imputation
            masks[:, i] = (~np.isnan(raw)).astype(np.float32)

            # Causal imputation: forward fill, then default only for leading gaps.
            series = _causal_impute_series(raw, default)
            vitals[:, i] = series.values

        # Sepsis label
        if "SepsisLabel" in df.columns:
            labels = df["SepsisLabel"].values.astype(np.float32)
        else:
            labels = np.zeros(L, dtype=np.float32)

        # Onset time: first time step with SepsisLabel=1
        sepsis_idx = np.where(labels > 0.5)[0]
        if len(sepsis_idx) > 0:
            onset_hour = int(sepsis_idx[0])
            is_sepsis = True

            # Exclusion: patients septic at or before hour 2 are excluded
            # from the primary early-warning task (plan §4.2, §6.2)
            if onset_hour < 2:
                return
        else:
            onset_hour = -1
            is_sepsis = False

        # Stats update
        self._update_stats(vitals)

        # Serialize
        ep_id = f"ep_{self.ep_cnt:06d}"
        patient_id = Path(fpath).stem

        txn.put(f"{ep_id}_vitals".encode(), vitals.tobytes())
        txn.put(f"{ep_id}_masks".encode(), masks.tobytes())
        txn.put(f"{ep_id}_labels".encode(), labels.tobytes())

        self.index.append({
            "episode_id": ep_id,
            "patient_id": patient_id,
            "length": L,
            "onset_hour": onset_hour,
            "is_sepsis": is_sepsis,
            "modalities": {
                "vitals": {"key": f"{ep_id}_vitals", "dtype": "float32", "shape": [L, 40]},
                "masks":  {"key": f"{ep_id}_masks",  "dtype": "float32", "shape": [L, 40]},
                "labels": {"key": f"{ep_id}_labels", "dtype": "float32", "shape": [L]},
            }
        })
        self.ep_cnt += 1

    def _update_stats(self, vitals: np.ndarray) -> None:
        self.global_min = np.minimum(self.global_min, vitals.min(axis=0))
        self.global_max = np.maximum(self.global_max, vitals.max(axis=0))

        # Reservoir sampling (5% of rows)
        n = len(vitals)
        take = max(1, n // 20)
        idx = np.random.choice(n, take, replace=False)
        self.reservoir.append(vitals[idx])

        # Memory guard: consolidate when reservoir grows too large
        if len(self.reservoir) > 5000:
            big = np.concatenate(self.reservoir)
            if len(big) > RESERVOIR_SIZE:
                keep = np.random.choice(len(big), RESERVOIR_SIZE, replace=False)
                self.reservoir = [big[keep]]
            else:
                self.reservoir = [big]

    def _save_index(self) -> None:
        if self.reservoir:
            pool = np.concatenate(self.reservoir)
            p01 = np.percentile(pool, 1, axis=0)
            p99 = np.percentile(pool, 99, axis=0)
        else:
            p01 = self.global_min
            p99 = self.global_max

        # Prevent degenerate ranges
        diff = p99 - p01
        p99 = np.where(diff < 1e-6, p99 + 1.0, p99)

        metadata = {
            "version": DATASET_FORMAT_VERSION,
            "preprocessing": {
                "imputation": self.causal_imputation_strategy,
                "causal": True,
            },
            "ts_columns": FEATURE_NAMES,
            "stats": {
                "ts_min": self.global_min.tolist(),
                "ts_max": self.global_max.tolist(),
                "ts_p01": p01.tolist(),
                "ts_p99": p99.tolist(),
            },
        }

        # PhysioNet has test set in same directory: save index to parent
        out_path = self.split_dir.parent / f"{self.split}_index.json"
        with open(out_path, "w") as f:
            json.dump({"episodes": self.index, "metadata": metadata}, f, indent=2)

        n_pos = sum(1 for e in self.index if e["is_sepsis"])
        logger.info(
            "Saved %s index with %d episodes (%d sepsis)",
            self.split, len(self.index), n_pos
        )

def build_physionet_lmdb(
    raw_dir: str,
    out_dir: str,
    seed: int = SEED,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    min_stay_hours: int = MIN_STAY_HOURS,
    max_stay_hours: int = MAX_STAY_HOURS,
    causal_imputation_strategy: str = CAUSAL_IMPUTATION_STRATEGY,
) -> None:
    """Builds LMDB databases from raw PhysioNet CSV files.

    If raw_dir is missing or contains no .psv files, automatically downloads
    the dataset from Kaggle via kagglehub and uses the cache path directly
    (no redundant copy — the download cache IS the raw_dir).
    """
    random.seed(seed)
    np.random.seed(seed)

    raw_path = Path(raw_dir) if raw_dir else Path("datasets/raw/physionet2019")

    # -------------------------------------------------------------------------
    # Auto-download if raw data is absent
    # -------------------------------------------------------------------------
    has_psv = raw_path.exists() and next(raw_path.rglob("*.psv"), None) is not None

    if not has_psv:
        logger.info(
            "Raw directory '%s' is missing or contains no .psv files. "
            "Auto-downloading 'farjanayesmin/the-physionet-challenge-2019-dataset' via kagglehub ...",
            raw_path,
        )
        try:
            import kagglehub  # type: ignore

            downloaded_path = Path(
                kagglehub.dataset_download("farjanayesmin/the-physionet-challenge-2019-dataset")
            )
            logger.info("kagglehub cache path: %s", downloaded_path)

            psv_in_cache = list(downloaded_path.rglob("*.psv"))
            if not psv_in_cache:
                logger.error(
                    "Downloaded path '%s' contains no .psv files. "
                    "Dataset layout may differ — download manually and pass --raw-dir.",
                    downloaded_path,
                )
                return

            # Use the kagglehub cache directly — no copy needed.
            raw_path = downloaded_path
            logger.info(
                "Using kagglehub cache as raw source: %s  (%d .psv files)",
                raw_path, len(psv_in_cache),
            )

        except ImportError:
            logger.error(
                "kagglehub is not installed. Run:  pip install kagglehub\n"
                "Or download the dataset manually and pass --raw-dir."
            )
            return
        except Exception as exc:
            logger.error("Auto-download failed: %s", exc)
            logger.error(
                "Check network/Kaggle credentials, or download manually and pass --raw-dir."
            )
            return

    # -------------------------------------------------------------------------
    # Gather .psv files and validate
    # -------------------------------------------------------------------------
    files = sorted(raw_path.rglob("*.psv"))  # sorted → deterministic before shuffle
    if not files:
        logger.error("No .psv files found in '%s' after setup. Aborting.", raw_path)
        return

    logger.info("Found %d .psv files in '%s'.", len(files), raw_path)

    # -------------------------------------------------------------------------
    # Stratified patient-level split
    # Pre-scan each file to determine sepsis status so the class ratio is
    # preserved exactly in train / val / test.
    # We only read the SepsisLabel column — fast single-pass per file.
    # -------------------------------------------------------------------------
    logger.info("Pre-scanning files for stratified split (SepsisLabel column only) ...")

    sepsis_files: List[Path] = []
    nonsepsis_files: List[Path] = []

    for fpath in files:
        try:
            # Read only the SepsisLabel column; bail on missing column
            df = pd.read_csv(fpath, sep="|", usecols=["SepsisLabel"])
            labels = df["SepsisLabel"].values
            sepsis_idx = np.where(labels > 0.5)[0]
            if len(sepsis_idx) > 0 and int(sepsis_idx[0]) >= 2:
                sepsis_files.append(fpath)
            else:
                nonsepsis_files.append(fpath)
        except Exception:
            # Missing column, corrupt file — treat as non-sepsis; ingestor will skip it
            nonsepsis_files.append(fpath)

    logger.info(
        "Pre-scan complete: %d sepsis  |  %d non-sepsis  (prevalence=%.1f%%)",
        len(sepsis_files), len(nonsepsis_files),
        100.0 * len(sepsis_files) / max(len(files), 1),
    )

    # Shuffle each pool independently (seeded above → reproducible)
    random.shuffle(sepsis_files)
    random.shuffle(nonsepsis_files)

    def _stratified_pool(pool: List[Path], ratio: float) -> List[Path]:
        return pool[: int(len(pool) * ratio)]

    # Compute counts
    n_sep  = len(sepsis_files)
    n_non  = len(nonsepsis_files)

    sep_train  = int(n_sep * train_ratio)
    sep_val    = int(n_sep * val_ratio)
    non_train  = int(n_non * train_ratio)
    non_val    = int(n_non * val_ratio)

    train_files = sepsis_files[:sep_train]                       + nonsepsis_files[:non_train]
    val_files   = sepsis_files[sep_train : sep_train + sep_val]  + nonsepsis_files[non_train : non_train + non_val]
    test_files  = sepsis_files[sep_train + sep_val :]            + nonsepsis_files[non_train + non_val :]

    # Shuffle each split so sepsis/non-sepsis aren't in blocks
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    logger.info(
        "Stratified split:\n"
        "  train : %d files  (%d sepsis = %.1f%%)\n"
        "  val   : %d files  (%d sepsis = %.1f%%)\n"
        "  test  : %d files  (%d sepsis = %.1f%%)",
        len(train_files), sep_train,  100.0 * sep_train  / max(len(train_files), 1),
        len(val_files),   sep_val,    100.0 * sep_val    / max(len(val_files),   1),
        len(test_files),  n_sep - sep_train - sep_val,
        100.0 * (n_sep - sep_train - sep_val) / max(len(test_files), 1),
    )

    # Ensure output directory exists before any LMDB is created
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Ingest each split
    # -------------------------------------------------------------------------
    for split_name, split_files in [
        ("train", train_files),
        ("val",   val_files),
        ("test",  test_files),
    ]:
        if not split_files:
            logger.warning("Split '%s' is empty — skipping.", split_name)
            continue
        logger.info("Building '%s' split (%d files) ...", split_name, len(split_files))
        ingestor = _Ingestor(
            out_dir=out_dir,
            split=split_name,
            min_stay_hours=min_stay_hours,
            max_stay_hours=max_stay_hours,
            causal_imputation_strategy=causal_imputation_strategy,
        )
        ingestor.process([str(f) for f in split_files])

    logger.info("Done. LMDB datasets written to '%s'.", out_dir)

class PhysioNet2019Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        min_length: int = MIN_STAY_HOURS,
        horizon_hours: int = 6,
        tau: float = 3.0,
        normalizer = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.lmdb_path = self.data_dir / split / "data.lmdb"
        self.index_path = self.data_dir / f"{split}_index.json"

        if not self.lmdb_path.exists():
            raise FileNotFoundError(
                f"LMDB not found at {self.lmdb_path}. "
                f"Run build_physionet_lmdb() first."
            )
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}.")

        with open(self.index_path, "r") as f:
            full_index = json.load(f)

        self.metadata = full_index["metadata"]
        _validate_index_metadata(self.metadata, index_path=self.index_path)
        all_episodes = full_index["episodes"]

        # Filter by minimum length
        self.episodes = [e for e in all_episodes if e["length"] >= min_length]
        logger.info(
            "[%s] Loaded %d/%d episodes (min_length=%d)",
            split.upper(), len(self.episodes), len(all_episodes), min_length,
        )

        self.horizon = horizon_hours
        self.target_fn = SoftOnsetTarget(horizon_hours=horizon_hours, tau=tau)
        self.normalizer = normalizer

        # Lazy LMDB
        self._env: Optional[lmdb.Environment] = None
        self._parent_pid = os.getpid()

    def __len__(self) -> int:
        return len(self.episodes)

    def _init_lmdb(self) -> None:
        curr_pid = os.getpid()
        if self._env is not None and curr_pid != self._parent_pid:
            self._env = None
            self._parent_pid = curr_pid
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=False,
            )

    def _fetch(self, key: str, dtype: str, shape: List[int]) -> np.ndarray:
        self._init_lmdb()
        with self._env.begin(write=False) as txn:
            raw = txn.get(key.encode("ascii"))
            if raw is None:
                raise KeyError(f"LMDB key not found: {key}")
        return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = self.episodes[idx]
        T = ep["length"]
        mod = ep["modalities"]

        # Fetch arrays
        vitals = self._fetch(mod["vitals"]["key"], "float32", [T, 40])  # [T, 40]
        masks = self._fetch(mod["masks"]["key"],  "float32", [T, 40])  # [T, 40]
        labels = self._fetch(mod["labels"]["key"], "float32", [T])     # [T]

        # Tensors
        x = torch.from_numpy(vitals)      # [T, 40]
        m = torch.from_numpy(masks)       # [T, 40]
        y = torch.from_numpy(labels)      # [T]

        # Normalize (if calibrated)
        if self.normalizer is not None:
            x = self.normalizer.normalize(x.unsqueeze(0)).squeeze(0)  # [T, 40]

        # Build BGSL soft targets (on-the-fly, batch of 1)
        onset_h = ep["onset_hour"]
        tgt = self.target_fn.build_single(
            onset_hour=onset_h,
            seq_len=T,
            device=x.device,
        )
        # Squeeze batch dim
        g = tgt["soft_target"].squeeze(0)          # [T]
        hard_g = tgt["hard_target"].squeeze(0)     # [T]
        dg = tgt["velocity_target"].squeeze(0)      # [T]
        d2g = tgt["accel_target"].squeeze(0)        # [T]
        valid = tgt["valid_mask"].squeeze(0)         # [T]
        vel_mask = tgt["vel_mask"].squeeze(0)        # [T]
        acc_mask = tgt["acc_mask"].squeeze(0)        # [T]

        # Extract static covariates from the first time step.
        # These are patient-level constants (Age, Gender, Unit1, Unit2, HospAdmTime)
        # and do not vary over the ICU stay.
        static_covariates = x[0, STATIC_COVARIATE_INDICES].clone()  # [D_static]

        return {
            "vitals":             x,            # [T, 40]  normalized features
            "masks":              m,            # [T, 40]  observation mask
            "hard_labels":        y,            # [T]      binary SepsisLabel
            "soft_targets":       g,            # [T]      g_t (BGSL state target)
            "hard_targets":       hard_g,       # [T]      baseline hard early-warning target
            "vel_targets":        dg,           # [T]      Δg_t
            "accel_targets":      d2g,          # [T]      Δ²g_t
            "valid_mask":         valid,        # [T]      1=compute loss
            "vel_mask":           vel_mask,     # [T]
            "acc_mask":           acc_mask,     # [T]
            "onset_hour":         torch.tensor(onset_h, dtype=torch.long),
            "is_sepsis":          torch.tensor(ep["is_sepsis"], dtype=torch.bool),
            "seq_len":            torch.tensor(T, dtype=torch.long),
            "patient_id":         ep["patient_id"],
            "static_covariates":  static_covariates,  # [D_static]
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    @property
    def sepsis_rate(self) -> float:
        n_pos = sum(1 for e in self.episodes if e["is_sepsis"])
        return n_pos / max(len(self.episodes), 1)


# ---------------------------------------------------------------------------
# Collator: handles variable-length trajectories
# ---------------------------------------------------------------------------

def collate_trajectories(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pad variable-length trajectories to the maximum length in the batch.

    All time-indexed tensors are padded on the RIGHT with zeros.
    The valid_mask / vel_mask / acc_mask track which positions are real.

    Returns a dict with all keys from __getitem__, with tensors batched.
    """
    # Filter None items (safety)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    T_max = max(b["vitals"].shape[0] for b in batch)
    B = len(batch)

    # Keys that are [T, ...] tensors (need padding)
    time_keys = [
        "vitals", "masks", "hard_labels", "soft_targets", "hard_targets",
        "vel_targets", "accel_targets", "valid_mask", "vel_mask", "acc_mask",
    ]
    # Keys that are scalar (0-D) tensors
    scalar_keys = ["onset_hour", "is_sepsis", "seq_len"]
    # Keys that are fixed-dimension vectors (not time-varying)
    vector_keys = ["static_covariates"]

    out: Dict = {}

    for key in time_keys:
        sample = batch[0][key]
        extra_dims = sample.shape[1:]  # dimensions after T
        padded = torch.zeros(B, T_max, *extra_dims, dtype=sample.dtype)
        for i, b in enumerate(batch):
            t = b[key].shape[0]
            padded[i, :t] = b[key]
        out[key] = padded

    for key in scalar_keys:
        out[key] = torch.stack([b[key] for b in batch])

    for key in vector_keys:
        out[key] = torch.stack([b[key] for b in batch])

    # patient_id is a list of strings
    out["patient_id"] = [b["patient_id"] for b in batch]

    return out


# ---------------------------------------------------------------------------
# Weighted sampler for class imbalance
# ---------------------------------------------------------------------------

def make_weighted_sampler(
    dataset: PhysioNet2019Dataset,
    pos_weight: float = 5.0,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Over-sample sepsis-positive patients to address class imbalance.

    Parameters
    ----------
    pos_weight : float
        Sampling weight for positive (sepsis) patients relative to negatives.
    """
    weights = [
        pos_weight if ep["is_sepsis"] else 1.0
        for ep in dataset.episodes
    ]
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Build PhysioNet 2019 LMDB dataset")
    parser.add_argument("--build", action="store_true", help="Run the build pipeline")
    parser.add_argument("--config", type=str, default="src/bgsl/config/dataset/sepsis.yaml", help="Path to config file")
    
    # These args override config if provided
    parser.add_argument("--raw-dir", type=str, default=None, help="Directory with .psv files")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.build:
        import yaml

        # Load config (fall back to empty dict if file doesn't exist)
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            logger.warning(
                "Config file '%s' not found — using built-in defaults. "
                "Pass --raw-dir / --out-dir to override paths.",
                cfg_path,
            )
            cfg = {}

        preproc = cfg.get("preprocessing", {})

        # CLI args take priority over config, config takes priority over hard-coded defaults.
        raw_dir = args.raw_dir or cfg.get("raw_data_path")  # None → auto-download
        out_dir = args.out_dir or cfg.get("processed_data_path", "datasets/processed/sepsis")
        seed         = args.seed if args.seed is not None else preproc.get("seed", SEED)
        train_ratio  = preproc.get("train_ratio", 0.80)
        val_ratio    = preproc.get("val_ratio", 0.10)
        # The YAML uses 'min_length'; also accept 'min_stay_hours' for compatibility.
        min_stay_hours = (
            preproc.get("min_stay_hours")
            or preproc.get("min_length")
            or MIN_STAY_HOURS
        )
        max_stay_hours = preproc.get("max_stay_hours", MAX_STAY_HOURS)
        causal_imputation_strategy = preproc.get(
            "causal_imputation_strategy", CAUSAL_IMPUTATION_STRATEGY
        )

        logger.info(
            "Resolved config:\n"
            "  raw_dir        = %s  (None → auto-download)\n"
            "  out_dir        = %s\n"
            "  seed           = %s\n"
            "  train/val/test = %.0f%% / %.0f%% / %.0f%%\n"
            "  min_stay_hours = %s\n"
            "  max_stay_hours = %s",
            raw_dir, out_dir, seed,
            train_ratio * 100, val_ratio * 100,
            (1 - train_ratio - val_ratio) * 100,
            min_stay_hours, max_stay_hours,
        )

        build_physionet_lmdb(
            raw_dir=raw_dir,
            out_dir=out_dir,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            min_stay_hours=min_stay_hours,
            max_stay_hours=max_stay_hours,
            causal_imputation_strategy=causal_imputation_strategy,
        )
    else:
        print(
            "PhysioNet 2019 LMDB builder\n"
            "\n"
            "Usage:\n"
            "  python -m bgsl.data.sepsis.physionet2019 --build\n"
            "\n"
            "Options:\n"
            "  --config   Path to YAML config (default: src/bgsl/config/dataset/sepsis.yaml)\n"
            "  --raw-dir  Override raw .psv directory (skips auto-download if set)\n"
            "  --out-dir  Override processed output directory\n"
            "  --seed     Random seed override\n"
            "\n"
            "If --raw-dir is not set and the configured raw_data_path is missing,\n"
            "the dataset is downloaded automatically from Kaggle via kagglehub.\n"
        )

