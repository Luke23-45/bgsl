# Empirical Study Plan

## Goal

Create a modular empirical-study layer that reuses the existing BGSL training, logging, aggregation, and dataset-acquisition code instead of reimplementing those pieces inside each study.

The empirical layer should answer one question:

> Does the frozen BGSL formulation remain weaker than BCE when evaluated on controlled synthetic data and on the real PhysioNet 2019 sepsis benchmark?

The answer to that question should be produced by a reproducible pipeline, not by ad hoc scripts.

## 1. Design principle

The repository already has the pieces needed for a clean empirical workflow:

- dataset acquisition and build logic
- PhysioNet 2019 ingestion
- shared runner utilities
- batch manifests
- run-level logs
- summary aggregation

The empirical layer should only define:

- what to run
- which conditions to compare
- which metrics to report
- how to package the outputs

It should not duplicate training, logging, or result collation.

## 2. Directory responsibilities

### Synthetic data directory

Use a dedicated synthetic data area for the generated benchmark artifacts.

Suggested responsibilities:

- store the generated synthetic cohort or tensor files
- store the synthetic dataset configuration
- store the seed and version used to generate the data
- store a manifest describing the exact generator parameters

Recommended rule:

- generation belongs in the data layer
- evaluation belongs in the study layer

The synthetic generator should not train models.

### Empirical study directory

Create a dedicated empirical study directory for experiment definitions and study-specific scripts.

Suggested responsibilities:

- stress-test scripts
- seed-stability scripts
- PhysioNet 2019 external-validation scripts
- calibration or sensitivity scripts if needed
- report assembly scripts

The study directory should import and reuse the shared runner code under `studies/runner/commons`.

## 3. Recommended structure

Keep the following separation:

- `datasets/` for generated or processed data artifacts
- `studies/` for experiment orchestration
- `outputs/` for run artifacts, manifests, logs, and summaries

Within that structure:

- synthetic data lives under a versioned synthetic dataset path
- PhysioNet 2019 lives under the existing processed sepsis dataset path
- empirical study outputs live under `outputs/empirical/...`

This keeps the paper pipeline reproducible and avoids mixing raw data, generated data, and run outputs.

## 4. What the empirical layer should contain

The empirical layer should define a small number of study types.

### 4.1 Synthetic stress study

Purpose:

- test the revised BGSL hypothesis under the frozen synthetic benchmark

Conditions:

- signal
- null
- high velocity weight
- long training
- seed stability

Output:

- one summary table
- one result folder per run
- one manifest per batch

### 4.2 External validation study

Purpose:

- check whether the direction seen in the synthetic benchmark appears on real sepsis trajectories

Data source:

- PhysioNet 2019, using the existing ingestion and acquisition pipeline

Output:

- frozen comparison of BCE vs BGSL on held-out metrics
- seed-wise or split-wise summary if available

### 4.3 Diagnostic study

Purpose:

- explain the failure mode without changing the method

Possible checks:

- calibration
- lead-time sensitivity
- threshold sensitivity
- training-length sensitivity

These diagnostics should be read-only with respect to the model family.

## 5. How to build the synthetic data

The synthetic generator should be deterministic and versioned.

### Required inputs

- random seed
- sequence length distribution
- positive/negative prevalence
- signal strength
- missingness pattern
- noise model
- horizon definition
- target-shape parameters

### Required outputs

- generated features
- masks
- hard targets
- soft targets
- event times
- sequence lengths
- a metadata file describing the generator version and parameters

### Required constraints

- no training inside the generator
- no evaluation inside the generator
- no hidden random state outside the explicit seed
- no untracked changes to the data schema

### Required validation

After generation, verify:

- schema matches the study code
- seeds reproduce the same data
- the manifest records the generator version
- the data can be consumed by the shared runner without custom glue code

## 6. How to organize study scripts

Each study script should be thin.

It should:

1. declare the study name
2. declare the condition grid
3. declare the seeds
4. reuse the shared run-spec builder
5. reuse the shared executor
6. let the shared manifest and summary code write outputs

The script should not:

- write its own logging system
- write its own manifest format
- invent its own output directory layout
- implement custom metric aggregation

The study script should only describe the experiment.

## 7. Shared output and logging contract

All empirical scripts must use the existing runner contract:

- `BatchPaths` for batch directories
- `RunPaths` for per-run directories
- manifest creation before execution
- status JSON per run
- `console.log` capture per run
- summary aggregation after execution

That contract should remain the single source of truth for:

- run identity
- seed
- condition name
- config override
- exit code
- final metrics

## 8. What to record for each run

Every empirical run should record:

- run id
- condition name
- seed
- base config
- command line
- status
- exit code
- duration
- final metrics
- output path

For the paper, the most important metrics are:

- PhysioNet utility
- AUROC
- AUPRC
- median lead time
- calibration or threshold behavior if relevant

## 9. End-to-end workflow

### Step 1: Prepare data

- generate synthetic data or acquire PhysioNet 2019
- freeze the data version
- write a manifest

### Step 2: Define study

- create a study script in the empirical layer
- define the methods and conditions
- define the seeds

### Step 3: Launch runs

- build run specs
- execute through the shared runner
- capture console output and status files

### Step 4: Aggregate results

- collect `summary.csv`
- collect final test metrics
- verify run completeness

### Step 5: Write the report

- summarize synthetic findings
- summarize PhysioNet 2019 findings if included
- write the gap analysis
- package figures and tables for the paper

## 10. Gap analysis to report

The empirical report should explicitly separate:

- synthetic gap
- real-data gap
- stress-test gap

Each gap should answer the same question:

> Does the frozen BGSL formulation improve over BCE under this condition?

If the answer is consistently no, that is the publication result.

If a real-data result disagrees with the synthetic result, the paper should state that disagreement plainly and not hide it behind averaging.

## 11. What the empirical layer must not do

Do not use the empirical layer to:

- reopen method tuning
- add new loss functions
- add a new architecture search
- blur synthetic and real-data claims
- invent a separate logging stack

The point of the layer is discipline, not flexibility.

## 12. Recommended implementation order

1. Freeze the synthetic generator schema.
2. Create the empirical study directory.
3. Move study definitions there, but keep the shared runner utilities untouched.
4. Add the synthetic stress study first.
5. Add the PhysioNet 2019 external validation study second.
6. Add only the diagnostics needed for the paper.
7. Write the report from the frozen outputs.

## 13. Final rule

If a new script cannot reuse the existing output/logging/manifest chain, it should not be added.

The empirical layer must be modular, reproducible, and aligned with the current project structure.

