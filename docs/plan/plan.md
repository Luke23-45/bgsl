# BGSL Publication Plan

## Objective

Publish the current work as a bounded methodological paper based on the revised hypothesis:

> Under the tested benchmark and implementation, the minimal BGSL formulation does not outperform BCE for early-event prediction.

The paper should be written as a controlled evaluation of the BGSL objective, not as an open-ended method-development project.

## 1. What the paper is about

The manuscript should answer one question:

- Does the tested BGSL objective improve early-event prediction over BCE when evaluated under a controlled synthetic benchmark with signal, null, and stress conditions?

The current evidence says no. The paper should present that result clearly and professionally.

## 2. What to freeze now

Before writing or adding more experiments, freeze the following:

- the current BGSL code path
- the final synthetic benchmark generator
- the current seed list
- the current loss definitions
- the current evaluation metrics
- the current stress settings
- the final benchmark outputs

Do not change the method after the benchmark has already answered the question. If the method changes, the paper becomes a different study.

## 3. What experimental setup is already sufficient

The current setup already covers the minimum needed for a publishable decision:

- a signal condition with a weak pre-onset trajectory
- a null condition with no longitudinal signal
- a velocity-weight stress test
- a longer-training stress test
- multiple random seeds

That is enough to justify the main claim.

## 3.1 Where PhysioNet 2019 fits

PhysioNet 2019 should **not** replace the synthetic benchmark or reopen the method search.

Use it only as a real-data corroboration step after the synthetic gate has already been frozen.

Recommended role:

- synthetic benchmark: primary decision gate
- PhysioNet 2019: external validation of the same frozen hypothesis and pipeline

Do not make PhysioNet 2019 a prerequisite for the paper unless the dataset is already available and the processing pipeline is fully reproducible. The paper can still stand on the synthetic stress test alone.

## 4. What to add before submission

Add only the experiments that improve credibility, not new direction.

### Required additions

1. Compute confidence intervals or bootstrap uncertainty for the final metrics across seeds.
2. Record the exact seed-wise results in a supplementary table.
3. Verify that the final reported numbers are identical when rerun from the frozen commit.
4. Save the final benchmark outputs in a reproducible appendix or supplementary note.
5. If PhysioNet 2019 is included, run it once using the frozen preprocessing and report only the final held-out metrics.

### Optional additions if time permits

1. Add a calibration summary if the final discussion emphasizes threshold behavior.
2. Add a precision-recall curve figure if it helps explain the utility tradeoff.
3. Add a short sensitivity note on the decision thresholds used in the go/no-go gate.

### Do not add

1. New model families.
2. New loss functions.
3. New real datasets.
4. Large hyperparameter sweeps that change the research question.
5. Any PhysioNet-driven tuning loop.

## 5. Publication package

The submission should include:

- a concise main paper
- a results table summarizing the signal, null, and stress settings
- a figure showing the benchmark design
- a figure or panel showing the stress-test outcome
- a supplement with seed-wise metrics and implementation details

## 6. Recommended venue path

### Primary target: ML4H Findings

This is the strongest fit for the current work.

Reasons:

- the topic is clinical ML
- the benchmark uses a health prediction metric
- the Findings format is suitable for a careful methodological evaluation
- the track is intended for work that can generate useful discussion even when the result is not a performance win

The Findings format is a good fit for a short, focused submission. Check the current ML4H instructions for the exact page limit, appendix rules, and release requirements at submission time.

### Secondary target: ICBINB-style workshop framing

If a broader methodology audience is useful, the same result can be reframed as:

- an explicit hypothesis
- a falsification benchmark
- a diagnostic explanation of why the hypothesis did not hold

Use this only as a secondary outlet. The clinical framing is stronger for this project.

## 7. Paper structure

### Title

Use a title that states the result directly and professionally.

Examples:

- "When Soft Trajectory Targets Do Not Improve Early Sepsis Prediction"
- "A Controlled Evaluation of a Minimal BGSL Objective for Early Warning"

### Abstract

The abstract should include:

- the clinical motivation
- the revised hypothesis
- the benchmark design
- the main result
- the practical implication

### Introduction

Explain:

- why trajectory-aware losses are attractive
- why early warning is a reasonable test case
- why a synthetic decision gate was used before large-scale data

### Methods

Describe:

- the BGSL objective being tested
- the BCE baseline
- the synthetic benchmark
- the signal and null conditions
- the stress tests
- the evaluation metrics
- the pre-defined pass/fail thresholds

### Results

Present:

- one main table
- one benchmark schematic
- one stress-test comparison panel

### Discussion

Explain:

- what the benchmark showed
- what the stress tests ruled out
- why the outcome matters
- what future work would need to change before revisiting trajectory-aware losses

### Limitations

State clearly:

- the benchmark is synthetic
- the conclusion is specific to the tested minimal BGSL formulation
- the study does not rule out every future trajectory-aware design

## 8. Claims to keep tight

The manuscript should stay within these claims:

- the tested BGSL setup did not outperform BCE here
- more training did not rescue it
- stronger velocity supervision did not rescue it
- the current evidence does not justify moving this exact formulation to MIMIC

Do not claim:

- BGSL is universally invalid
- all soft-target methods fail
- the idea can never be improved
- the result automatically generalizes to all EHR settings

## 9. Writing and review workflow

### Drafting order

1. Write the abstract.
2. Write the methods section.
3. Draft the results section from the frozen outputs.
4. Write the discussion and limitations.
5. Write the introduction last, once the paper's claim is fixed.

### Internal review order

1. Check that every claim is supported by a figure, table, or exact number.
2. Check that the scope stays tied to the minimal BGSL formulation.
3. Check that the benchmark description matches the code exactly.
4. Check that the venue format and page limits are satisfied.

### Final technical review

1. Confirm all numbers from the manuscript match the frozen outputs.
2. Confirm all figures reproduce from the frozen commit.
3. Confirm the supplement contains the seed-wise results.
4. Confirm the anonymized PDF contains no identifying information.

## 10. Submission checklist

Before submission, verify:

- the target venue policy for page limit and appendices
- the current dual-submission policy
- the anonymization requirement
- the file format requested by the venue
- whether code or artifact release is required
- whether the submission should be archival or non-archival

For ML4H Findings specifically, the current public guidance describes it as a short submission track with a main-text page limit and appendices that can contain supplementary detail. Use the venue template and current instructions at submission time.

## 11. End-to-end deliverables

The project is complete only when all of the following exist:

- frozen benchmark code
- frozen final outputs
- seed-wise supplement
- polished manuscript draft
- venue-compliant anonymized PDF
- submission package prepared for the target venue
- optional PhysioNet 2019 external-validation appendix, if used

## 11.1 PhysioNet 2019 data collection plan

If you include PhysioNet 2019, collect it using the existing repository pipeline:

1. Try the local cache first through the acquisition layer.
2. If needed, pull the processed snapshot from HuggingFace through the configured acquisition path.
3. If no processed snapshot exists, build from raw PhysioNet 2019 files using the existing `python -m bgsl.data.sepsis.physionet2019 --build` path.

Use the frozen preprocessing already implemented in the repo:

- full patient trajectories
- causal forward-fill imputation
- precomputed onset hour
- train/val/test patient-level split
- the same PhysioNet utility metric already implemented in the codebase

What PhysioNet 2019 is for:

- confirm that the synthetic conclusion is not a quirk of one toy generator
- show whether the direction of the effect is stable on real sepsis trajectories

What PhysioNet 2019 is not for:

- deciding whether to keep tuning BGSL
- changing the loss definition
- changing the benchmark thresholds
- changing the final paper claim after the fact

### Gap to report if PhysioNet 2019 is included

Report the gap as the difference between the frozen BGSL formulation and BCE on held-out PhysioNet 2019 metrics, with emphasis on:

- PhysioNet utility
- AUROC
- AUPRC
- lead-time behavior
- calibration or threshold behavior, if relevant

The gap section should answer one question:

> Does the same direction of effect observed in the synthetic stress test also appear on real sepsis trajectories?

Do not turn this into a new optimization problem.

## 12. Bottom line

Do not extend the method first.

Freeze the benchmark, add only publication-strengthening checks, write the paper around the revised hypothesis, and submit a bounded manuscript that is faithful to the tested evidence.
