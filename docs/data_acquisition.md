# Data acquisition

BGSL ships a **standalone, 3-tier data acquisition layer** for the PhysioNet
2019 Sepsis Challenge. It is invoked *before* training and is fully decoupled
from `bgsl-train`, the `PhysioNetDataModule`, and the runner layer.

## The 3-tier chain

| Tier | Source | When it runs |
| ---- | ------ | ------------ |
| 1    | **Local** | All 6 required artifacts (`{train,val,test}/data.lmdb` + `{train,val,test}_index.json`) exist in `data_dir` AND a valid `.bgsl_acquired.json` marker is present. |
| 2    | **HuggingFace Hub** | Tier 1 fails and an `hf:` config is provided. `huggingface_hub.snapshot_download` materializes the pre-built LMDB into `data_dir`, then we SHA-verify every file against the repo-shipped `manifest.json`. |
| 3    | **Build from raw** | Tiers 1 + 2 fail and a `build:` config is provided. Delegates to the existing `python -m bgsl.data.physionet2019 --build` subprocess (which handles kagglehub auto-download OR a user-supplied `--raw-dir`). |

The first tier to succeed wins; the rest are skipped.

## Why a standalone script

- `bgsl-train` is unchanged: the existing `FileNotFoundError` path in
  `PhysioNet2019Dataset.__init__` is the success indicator downstream.
- The script has zero Lightning / torch / GPU dependencies at import time
  (only stdlib + `yaml`); you can run it on a tiny VM.
- The script is idempotent: a marker file records the last successful
  acquisition, so subsequent calls cost microseconds.
- A `filelock.FileLock` is held during tier 2 / tier 3 work, so multiple
  training jobs (or a Slurm prologue + epilogue race) cannot clobber a
  half-built dataset.

## Quick start

```bash
# 1. Install the optional extras.
pip install -e ".[data]"

# 2. Edit the config to point at your HF repo (or use the kaggle-only variant).
$EDITOR experiments/configs/data_acquisition.yaml

# 3. Acquire the data.
bgsl-acquire --config experiments/configs/data_acquisition.yaml

# 4. Train as usual.
bgsl-train fit --config experiments/configs/physionet_gru.yaml
```

`bgsl-acquire` exits 0 on success and 1 on failure (with a clear error
on stderr). For machine consumption, pass `--json` to emit a single-line
JSON result on stdout.

## Publish a new build

After you (re)build locally, push the result to a HF repo so other
contributors / CI can `bgsl-acquire` from it:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
bgsl-upload \
    --data-dir data/ready \
    --repo-id your-org/bgsl-physionet2019 \
    --private \
    --message "Add seed=2025 build with 0.7/0.15/0.15 split"
```

`bgsl-upload` will:
1. Validate that all 6 required artifacts exist.
2. Compute a fresh `manifest.json` with SHA-256 + size for every file.
3. Render a `README.md` dataset card with provenance.
4. `HfApi.create_repo(..., exist_ok=True)` + `HfApi.upload_folder(...)`.

To preview without touching the network, pass `--dry-run`.

## Config schema

```yaml
data_dir: "data/ready"            # root for {train,val,test}/data.lmdb + index JSONs

hf:                               # omit to disable tier 2
  repo_id: "your-org/bgsl-physionet2019"
  repo_type: "dataset"            # "dataset" | "model"
  revision: "main"                # branch, tag, or commit SHA
  filename_glob: "*"              # restrict snapshot_download
  token_env: "HF_TOKEN"           # env var carrying the HF token

build:                            # omit to disable tier 3
  raw_dir: null                   # set to a local .psv directory to skip kagglehub
  seed: 2025
  train_ratio: 0.8
  val_ratio: 0.1
  min_length: 8
  horizon_hours: 6
  extra_args: []                  # any extra flags for the build subprocess

force: false                      # if true, bypass the marker cache
lock_timeout_s: 600               # max wait for a concurrent acquisition
```

A complete example lives at
[`experiments/configs/data_acquisition.yaml`](../experiments/configs/data_acquisition.yaml);
two pre-canned variants (`data_acquisition.hf_only.yaml`,
`data_acquisition.kaggle_only.yaml`) are also available.

## Force re-acquisition

Three ways to bypass the marker cache and re-run the tier chain:

```bash
bgsl-acquire --config <yaml> --force         # one-off flag
BGSL_FORCE_REACQUIRE=1 bgsl-acquire ...      # env var
# or manually:
rm data/ready/.bgsl_acquired.json            # crude
```

Use this after bumping the HF revision, recovering from suspected
corruption, or switching between kaggle and HF sources.

## Internal files written by the script

`bgsl-acquire` writes only two hidden files in `data_dir`:

| File                         | Purpose                                                              |
| ---------------------------- | -------------------------------------------------------------------- |
| `data_dir/manifest.json`     | SHA-256 + size for every data file. Re-written on every acquisition. |
| `data_dir/.bgsl_acquired.json` | Acquisition marker: source, timestamp, revision, provenance fields. |
| `data_dir/.bgsl_acquired.lock` | Cross-process lock; transient, only present during tier 2/3 work.   |

None of these are read by the `PhysioNet2019Dataset` class. They are purely
acquisition-layer bookkeeping.

## Why not Lightning's `prepare_data`?

We deliberately do not hook into `PhysioNetDataModule.prepare_data`:
- Keeps the `bgsl-train` invocation free of network / disk surprises.
- The acquisition can be tested, debugged, and rerun independently of any
  training job.
- Slurm / cloud users can run the acquisition once on a shared filesystem
  rather than once per training job.
- The data module stays a thin wrapper around `PhysioNet2019Dataset`.

The cost: you must remember to run `bgsl-acquire` before `bgsl-train`.
This is intentional: the failure mode (missing LMDB → `FileNotFoundError`)
is loud and obvious, and a Slurm prologue can call `bgsl-acquire` for you.
