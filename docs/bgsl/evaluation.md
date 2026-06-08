# Evaluation Framework for BGSL

## 1. Core Objectives

Biological Gradient Supervised Learning (BGSL) introduces a family of loss
functions that incorporate *trajectory-aware regularisation* — velocity and
acceleration penalties on risk-score dynamics — during training. The central
hypothesis is:

> **BGSL produces risk-score trajectories that are smoother, more stable, and
> fire earlier at the same level of global discrimination**, compared with
> standard binary cross-entropy (BCE), temporal baselines (TLS), and naive
> smoothness penalties.

This document defines the evaluation protocol used to test that hypothesis.

## 2. Evaluation Framework

The framework distinguishes between two classes of metrics.

### 2.1 Discrimination & Calibration
Standard point-wise metrics computed over all time-steps pooled across
patients:

| Metric | Description |
|---|---|
| **AUROC** | Area under the receiver operating characteristic curve |
| **AUPRC** | Area under the precision–recall curve |
| **Sensitivity** | True-positive rate at the selected threshold |
| **Specificity** | True-negative rate at the selected threshold |
| **F₁** | Harmonic mean of precision and recall |
| **Brier Score** | Mean squared error between probability and label |
| **ECE** | Expected calibration error (10-bin equal-width) |

These metrics verify the **null premise**: that replacing the loss function
does not materially degrade the model's ability to discriminate positive from
negative time-steps. If AUROC shifts significantly across conditions, the
comparison of trajectory metrics is confounded.

### 2.2 Trajectory-Quality Metrics
Patient-level temporal metrics computed only at **test time** after a
threshold is selected from the validation set (fixed-sensitivity at 80 %):

| Metric | Abbrev. | Definition | Target |
|---|---|---|---|
| **Alert Switching Frequency** | ASF | Number of on→off/off→on transitions per unit time | ↓ (fewer oscillations) |
| **Risk Total Variation** | RTV | Sum of absolute consecutive differences in risk score | ↓ (smoother trajectory) |
| **Stable-Period Jitter** | SPJ | Mean absolute deviation of risk score during non-event periods | ↓ (less noise) |
| **Pre-Onset Monotonicity Score** | POMS | Fraction of non-decreasing intervals in the pre-onset window | ↑ (cleaner escalation) |
| **Trajectory Calibration Error** | TCE | Mean absolute deviation between risk score and soft onset target near the event | ↓ (better alignment) |
| **Median Lead Time** | — | Hours from first sustained alert to sepsis onset (warning time) | ↑ (earlier detection) |
| **False-Alarm Rate** | FA Rate | False sustained alerts per unit time for non-event patients | ↓ (fewer nuisance alerts) |
| **False Alarms Per Patient-Day** | FAPPD | FA Rate × 24 (clinically interpretable) | ↓ |
| **PhysioNet Utility** | — | Official scoring function from the PhysioNet 2019 Challenge | ↑ (higher clinical value) |
| **Selected Threshold** | — | Probability threshold achieving 80 % sensitivity on the validation set | varies |

## 3. Experimental Design

### 3.1 Conditions

Six loss functions are compared on a **fixed architecture** — a 2-layer GRU
with 128 hidden units — with no hyperparameter variation between conditions:

| # | Condition | Loss | Key Parameters |
|---|---|---|---|
| 1 | BCE Baseline | `BCELoss` | — |
| 2 | Temporal Logistic Smoothing | `TLSLoss` | α = 6.0 |
| 3 | BCE + Smoothness | `SmoothnessLoss` | λ_s = 0.1 |
| 4 | BCE + Total Variation | `TotalVariationLoss` | λ_tv = 0.1 |
| 5 | BGSL (State Only) | `BGSLLoss` | w_v = 0.0, w_a = 0.0 |
| 6 | BGSL (Full) | `BGSLLoss` | w_v = 0.1, w_a = 0.05 |

Conditions 3 and 4 control for the intuitive alternative: "why not just add a
smoothness penalty to BCE?" If BGSL outperforms these, the benefit derives
from the gradient-target structure, not from smoothness alone.

### 3.2 Repetitions

Each condition is trained with **3 independent random seeds** (42, 43, 44) to
assess statistical variability. Results are reported as **mean ± standard
deviation** across seeds.

### 3.3 Statistical Inference

Bootstrap confidence intervals (1000 iterations, 95 % CI) are computed for
every per-run metric via patient-level resampling with replacement. The
confidence bounds characterise sampling uncertainty within a single run and
are reported as `{metric}_ci_low` / `{metric}_ci_high`.

## 4. Threshold Selection

All trajectory metrics depend on a probability threshold that defines when an
alert is considered "fired". Rather than tuning this threshold for optimal
test-set performance, we fix the policy:

1. During each validation epoch, accumulate per-patient predictions.
2. At the end of validation, sweep 200 candidate thresholds ∈ [0, 1].
3. Select the highest threshold that achieves ≥ 80 % sensitivity.
4. Persist the selected threshold in the checkpoint for test-time use.

This protocol mirrors clinical deployment: the sensitivity target is set by
clinical requirements, and the threshold is frozen before test-set evaluation.

## 5. Test-Time Evaluation Pipeline

After training, each run is evaluated on the held-out test set:

```
fit ──► save best checkpoint ──► test ──► compute metrics ──► save results
                                            │
                                            ├── discrimination
                                            ├── calibration
                                            └── trajectory quality
```

The evaluation pipeline is automated via the experiment runner
(`studies/runner/ablation/core_hypothesis_runner.py`). The runner:

1. Builds one override configuration per (condition, seed) pair.
2. Trains the model with `bgsl.cli fit`.
3. Loads the best checkpoint (lowest validation loss) and runs `bgsl.cli test`.
4. Extracts all `test_*` metrics from the CSV logger output.
5. Aggregates results across seeds into a condition-level summary table.

For the architecture-agnostic BGSL claim, use the separate paired runner
(`studies/runner/architecture/paired_runner.py`), which compares raw BCE
against BGSL within each architecture family.

## 6. Result Interpretation

The hypothesis makes four specific predictions:

| Prediction | Metric | Expected Direction | Interpretation |
|---|---|---|---|
| 1. Equivalent discrimination | AUROC, AUPRC | Flat across conditions | Loss function does not affect global ranking |
| 2. Smoother trajectories | ASF ↓, RTV ↓, SPJ ↓ | BGSL < BCE ≈ TLS | BGSL suppresses oscillations |
| 3. Earlier warnings | Lead time ↑ | BGSL > BCE ≈ TLS | BGSL fires earlier at same sensitivity |
| 4. Higher clinical utility | Utility ↑ | BGSL > BCE ≈ TLS | BGSL translates to better clinical outcomes |

If AUROC/AUPRC are stable across conditions (prediction 1) while BGSL shows
statistically significant improvements in predictions 2–4, the hypothesis is
supported. If AUROC shifts or trajectory metrics do not improve, the
hypothesis is weakened.

## 7. Data

Experiments are conducted on the **PhysioNet 2019 Early Sepsis Prediction**
dataset, processed using the pipeline in `src/bgsl/data/sepsis/`.

- Input: 40 clinical features (28 vitals + 12 lab values), resampled to hourly
  granularity, with missingness indicators.
- Target: Sepsis onset within the next 6 hours (binary) + soft onset target
  (sigmoidal transition over a 6-hour window).
- Split: Temporal — patients are partitioned by ICU admission time into
  training / validation / test sets.

## 8. Limitations

- **Single architecture.** All experiments use a GRU with fixed hidden
  dimension and depth. Results may not generalise to other architectures
  (TCN, Transformer, LSTM).
- The architecture-generalisation question is handled by the paired runner
  rather than this fixed-GRU loss study.
- **Single dataset.** The PhysioNet 2019 benchmark is a single-domain early-
  warning task. Broader claims require replication on additional datasets
  (e.g. CMAPSS for predictive maintenance, MIMIC for general critical care).
- **Sensitivity threshold.** The 80 % sensitivity target is arbitrary.
  Sensitivity analyses at 70 %, 90 % would test robustness.
- **Bootstrap CIs.** Per-run bootstrap confidence intervals reflect sampling
  variability of the test set, not between-seed variability. The latter is
  captured by the mean ± std across seeds.
