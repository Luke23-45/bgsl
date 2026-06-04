from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bgsl.data.physionet2019 import (
    CAUSAL_IMPUTATION_STRATEGY,
    DATASET_FORMAT_VERSION,
    CLINICAL_SPECS,
    _Ingestor,
    _causal_impute_series,
    _validate_index_metadata,
)


def test_causal_imputation_never_uses_future_values():
    raw = np.array([np.nan, np.nan, np.nan, 111.0], dtype=float)
    filled = _causal_impute_series(raw, default=75.0)

    assert filled.iloc[:3].tolist() == [75.0, 75.0, 75.0]
    assert filled.iloc[3] == 111.0


def test_validate_index_metadata_rejects_legacy_layout(tmp_path: Path):
    legacy = {
        "version": "bgsl-1.0",
        "ts_columns": [spec["name"] for spec in CLINICAL_SPECS.values()],
        "stats": {},
    }

    with pytest.raises(RuntimeError, match="Legacy PhysioNet index"):
        _validate_index_metadata(legacy, index_path=tmp_path / "train_index.json")


def test_ingestor_uses_causal_fill_and_writes_new_metadata(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "ready"
    out_dir.mkdir()

    columns = [spec["col"] for spec in CLINICAL_SPECS.values()] + ["SepsisLabel"]
    frame = pd.DataFrame(index=range(8), columns=columns, dtype=float)

    for spec in CLINICAL_SPECS.values():
        frame[spec["col"]] = spec["default"]

    frame["HR"] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 111.0]
    frame["SepsisLabel"] = 0.0

    psv_path = raw_dir / "p000001.psv"
    frame.to_csv(psv_path, sep="|", index=False)

    ingestor = _Ingestor(str(out_dir), "train")
    try:
        with ingestor.env.begin(write=True) as txn:
            ingestor._process_one(str(psv_path), txn)
        ingestor.env.close()
        ingestor._save_index()
    finally:
        try:
            ingestor.env.close()
        except Exception:
            pass

    index_path = out_dir / "train_index.json"
    index = json.loads(index_path.read_text())
    assert index["metadata"]["version"] == DATASET_FORMAT_VERSION
    assert index["metadata"]["preprocessing"]["imputation"] == CAUSAL_IMPUTATION_STRATEGY
    assert index["metadata"]["preprocessing"]["causal"] is True

    episode = index["episodes"][0]
    lmdb_path = out_dir / "train" / "data.lmdb"
    import lmdb

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
    with env.begin() as txn:
        vitals = np.frombuffer(
            txn.get(f'{episode["episode_id"]}_vitals'.encode()),
            dtype=np.float32,
        ).reshape(episode["length"], 28)
        masks = np.frombuffer(
            txn.get(f'{episode["episode_id"]}_masks'.encode()),
            dtype=np.float32,
        ).reshape(episode["length"], 28)

    hr = vitals[:, 0].tolist()
    hr_mask = masks[:, 0].tolist()
    assert hr[:7] == [75.0] * 7
    assert hr[7] == 111.0
    assert hr_mask[:7] == [0.0] * 7
    assert hr_mask[7] == 1.0
