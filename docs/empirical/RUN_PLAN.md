# Empirical Run Plan

This file is the execution checklist for the empirical study.
It is aligned with the current code in `src/bgsl/empirical/` and the shared runner utilities in `studies/runner/`.

The publication evidence has two parts:

1. Synthetic stress benchmark, which is the primary controlled result.
2. PhysioNet 2019 external validation, which is the real-data corroboration step.

Do not change the frozen grids, model architecture, or data generation parameters after starting the publication run unless you intend to rerun the affected experiment.

## 1. Frozen experiment definition

### 1.1 Synthetic benchmark

This is the main publication grid.

- Scenarios: `signal`, `null`
- Stress settings: `Default`, `HighVel`, `LongTrain`
- Methods: `BCE`, `BGSL-state`, `BGSL+vel`
- Seeds: `101`, `202`, `303`, `404`, `505`
- Total runs: `3 x 2 x 3 x 5 = 90`

Frozen cell settings:

| Stress | `tau` | `max_epochs` | `BGSL+vel` velocity weight |
|---|---:|---:|---:|
| `Default` | `2.5` | `16` | `0.05` |
| `HighVel` | `1.5` | `16` | `0.50` |
| `LongTrain` | `2.5` | `32` | `0.05` |

Method detail:

- `BCE` uses hard targets
- `BGSL-state` uses soft targets with zero auxiliary weights
- `BGSL+vel` uses soft targets with the frozen velocity weight above

### 1.2 PhysioNet 2019 external validation

This is the real-data validation grid.

- Methods: `BCE`, `BGSL-state`, `BGSL+vel`
- Seeds: `42`, `43`, `44`
- Total runs: `3 x 3 = 9`

Frozen method settings:

- `BCE`: plain baseline
- `BGSL-state`: `state_loss="bce"`, velocity/acceleration/monotonicity weights set to `0.0`
- `BGSL+vel`: `state_loss="bce"`, velocity weight `0.05`, acceleration/monotonicity weights `0.0`

## 2. Correct environment checks

Run these before the first batch:

```powershell
python --version
python -c "import bgsl; print('OK')"
python -c "from bgsl.empirical.data.synthetic import SyntheticICUDataModule; print('Synthetic OK')"
python -c "from bgsl.data.sepsis.datamodule import PhysioNetDataModule; print('PhysioNet OK')"
python -c "from bgsl.empirical.studies.synthetic_stress import main as synthetic_main; from bgsl.empirical.studies.physionet_external import main as physionet_main; print('Studies OK')"
```

If the PhysioNet import fails, skip the PhysioNet batch and publish the synthetic result with that limitation stated.

## 3. Smoke test

Run one synthetic job first. This is the minimal pipeline check.

```powershell
python -m bgsl.empirical.studies.synthetic_stress `
    --mode local_sequential `
    --stress Default `
    --scenario signal `
    --methods BCE `
    --seeds 101
```

Optional, only if PhysioNet data is already available:

```powershell
python -m bgsl.empirical.studies.physionet_external `
    --mode local_sequential `
    --methods BCE `
    --seeds 42
```

Smoke test pass conditions:

- command ends with `Overall status: success`
- `summary.csv` has one row
- `runs/<run_id>/status.json` has `exit_code = 0`
- `runs/<run_id>/metrics/final_test_metrics.json` exists

## 4. Full publication runs

### 4.1 Synthetic benchmark

Run the full controlled benchmark:

```powershell
python -m bgsl.empirical.studies.synthetic_stress --mode local_sequential
```

This produces:

- `outputs/empirical/synthetic_stress/<batch_id>/batch_manifest.json`
- `outputs/empirical/synthetic_stress/<batch_id>/summary.csv`
- `outputs/empirical/synthetic_stress/<batch_id>/aggregated_results.csv`
- per-run `metrics/final_test_metrics.json`

### 4.2 PhysioNet 2019 external validation

Run this only if the PhysioNet dataset is available and the import check passes:

```powershell
python -m bgsl.empirical.studies.physionet_external --mode local_sequential
```

This produces:

- `outputs/empirical/physionet_external/<batch_id>/batch_manifest.json`
- `outputs/empirical/physionet_external/<batch_id>/summary.csv`
- `outputs/empirical/physionet_external/<batch_id>/aggregated_results.csv`
- per-run `metrics/final_test_metrics.json`

## 5. Required verification

Do not use a batch in the paper until these checks pass.

### 5.1 Synthetic batch checks

- `batch_manifest.json` exists
- manifest run count is `90`
- `summary.csv` has `90` rows
- every run has `exit_code = 0`
- `aggregated_results.csv` exists
- there are `18` unique conditions in `aggregated_results.csv`
- every run has `metrics/final_test_metrics.json`
- every metrics file contains at least:
  - `test_auroc`
  - `test_auprc`
  - `test_physionet_utility`
  - `test_median_lead_time`
  - `test_asf`
  - `test_poms`

### 5.2 PhysioNet batch checks

- `batch_manifest.json` exists
- manifest run count is `9`
- `summary.csv` has `9` rows
- every run has `exit_code = 0`
- `aggregated_results.csv` exists
- there are `3` unique conditions in `aggregated_results.csv`

### 5.3 Useful PowerShell checks

```powershell
Import-Csv outputs/empirical/synthetic_stress/<batch_id>/summary.csv | Measure-Object
Import-Csv outputs/empirical/physionet_external/<batch_id>/summary.csv | Measure-Object
```

Check for failures:

```powershell
Import-Csv outputs/empirical/synthetic_stress/<batch_id>/summary.csv |
    Where-Object { $_.exit_code -ne 0 }

Import-Csv outputs/empirical/physionet_external/<batch_id>/summary.csv |
    Where-Object { $_.exit_code -ne 0 }
```

## 6. Report assembly

After the batches pass verification, generate the publication summary:

```powershell
python -m bgsl.empirical.studies.report `
    --synthetic outputs/empirical/synthetic_stress/<synthetic_batch_id> `
    --physionet outputs/empirical/physionet_external/<physionet_batch_id>
```

The report writes:

- `docs/results/empirical_summary.md`

That file should contain:

- `Synthetic Benchmark`
- `Gap Analysis`
- `PhysioNet 2019 External Validation`
- `Conclusion`

Important report detail:

- The synthetic section is organized by stress setting and scenario.
- The gap analysis compares `BCE` against `BGSL+vel`.
- The gap analysis is metric-wise, so each of the 6 condition cells contributes multiple rows.

## 7. Publication artifact map

Use these files for the paper tables and appendix.

| Publication item | Source |
|---|---|
| Synthetic main table | `outputs/empirical/synthetic_stress/<batch_id>/aggregated_results.csv` |
| Synthetic seed-wise appendix | `outputs/empirical/synthetic_stress/<batch_id>/summary.csv` and per-run `final_test_metrics.json` |
| PhysioNet main table | `outputs/empirical/physionet_external/<batch_id>/aggregated_results.csv` |
| Report summary and gap analysis | `docs/results/empirical_summary.md` |

If a table needs a metric that is missing from `aggregated_results.csv`, rebuild it from the per-run `final_test_metrics.json` files.

## 8. Failure recovery

### 8.1 One run fails

1. Identify the failed run in `summary.csv`.
2. Inspect `runs/<run_id>/logs/console.log`.
3. Re-run only the failed condition with the same seed.

Example:

```powershell
python -m bgsl.empirical.studies.synthetic_stress `
    --mode local_sequential `
    --stress HighVel `
    --scenario signal `
    --methods BCE `
    --seeds 202
```

### 8.2 Synthetic cache looks stale

The synthetic cache is guarded by a manifest fingerprint, but if you intentionally changed generator behavior, clear the cache before rerunning:

```powershell
Remove-Item -Recurse -Force datasets/synthetic/bgsl_v1
```

### 8.3 Restarting after interruption

- The batch manifest is written before execution.
- Interrupted jobs may remain with a failure or interrupted status.
- Re-run only the missing or failed conditions if you want to preserve completed results.

## 9. Final publication checklist

Before writing the paper tables, confirm:

1. Synthetic batch complete: `90 / 90` successful runs.
2. PhysioNet batch complete: `9 / 9` successful runs, if PhysioNet was included.
3. No missing `final_test_metrics.json` files.
4. `docs/results/empirical_summary.md` exists and is current.
5. The frozen settings in this file match the code in `src/bgsl/empirical/`.

