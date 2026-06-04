# Gradient-Supervised Risk Trajectory Learning

## Publication Plan for BGSL with Sepsis Early Warning as the Flagship Study

**Status:** Final implementation and publication plan  
**Date:** June 2026  
**Primary application:** Sepsis early warning  
**Method scope:** General timestamped event-risk prediction  

---

## 1. Executive Positioning

BGSL should not be presented as a sepsis-only method. It should be presented as a general trajectory-supervision framework for timestamped event prediction, with sepsis as the first and strongest proof-of-concept.

Recommended framing:

> Biological Gradient Supervised Learning (BGSL) is a model-agnostic training objective for early event prediction. It supervises not only risk level, but also risk velocity and risk acceleration. We validate BGSL in sepsis early warning, where event onset is timestamped, lead time is clinically valuable, and unstable alert trajectories directly contribute to alarm fatigue.

This framing is important. The method is general, but the paper remains focused and empirically defensible.

### Core Thesis

Standard early-warning models optimize pointwise classification at each time step. This ignores the temporal shape of the risk signal. BGSL trains the model to learn the trajectory of risk as an event approaches:

- What is the current risk?
- Is risk changing in the correct direction?
- Is risk changing at a clinically plausible rate?
- Is the risk curve stable during non-event periods?

### Method Identity

Use one of these titles:

1. **Gradient-Supervised Risk Trajectory Learning for Stable Early Warning: Application to Sepsis Prediction**
2. **Learning Risk Momentum for Early Event Prediction: A Sepsis Early-Warning Study**
3. **Biological Gradient Supervised Learning for Stable Sepsis Early Warning**

Best recommendation:

> Gradient-Supervised Risk Trajectory Learning for Stable Early Warning: Application to Sepsis Prediction

This title keeps the method general and the validation concrete.

---

## 2. Research Questions

### Primary Research Question

Can derivative-supervised risk trajectory learning improve clinically useful early sepsis warning compared with pointwise classification and existing temporal label-shaping baselines?

### Secondary Research Questions

1. Does BGSL reduce alert switching and false alert burden?
2. Does BGSL preserve or improve AUROC and AUPRC?
3. Does BGSL improve warning lead time before sepsis onset?
4. Does BGSL improve trajectory calibration near onset?
5. Does BGSL provide benefit across multiple model architectures?
6. Which component matters most: soft state target, velocity supervision, acceleration supervision, or their combination?
7. Does the method transfer to a second timestamped clinical deterioration task, such as AKI or circulatory failure?

---

## 3. Method: General BGSL Specification

### 3.1 Timestamped Event-Risk Prediction

Let a patient trajectory be:

```text
X_1:T = (x_1, x_2, ..., x_T)
```

where `x_t` is the multivariate observation at time `t`.

Let:

```text
t* = event onset time
H  = early-warning horizon
p_t = f_theta(X_1:t)
```

where `p_t` is the model-predicted probability that the event will occur within horizon `H`.

The standard hard early-warning label is:

```text
y_t = 1[t* - H <= t <= t*]
```

for positive cases, and `y_t = 0` for all time steps in negative cases.

### 3.2 Soft Event-Onset Target

Do not train BGSL directly against only hard step labels. Hard labels create discontinuous derivative targets and make the velocity/acceleration terms brittle.

Use a soft onset target:

```text
g_t = sigmoid((t - c) / tau)
```

where:

```text
c = t* - H / 2
tau = softness/temperature parameter
```

For negative patients:

```text
g_t = 0 for all t
```

Recommended initial values:

```text
H in {6h, 12h}
tau in {1h, 2h, 4h}
```

The final paper should report an ablation over `tau`, because the softness of the target is a major reviewer concern.

### 3.3 BGSL Objective

The model produces a risk trajectory:

```text
p = (p_1, p_2, ..., p_T)
```

The soft target trajectory is:

```text
g = (g_1, g_2, ..., g_T)
```

First temporal derivative:

```text
Delta p_t = p_t - p_{t-1}
Delta g_t = g_t - g_{t-1}
```

Second temporal derivative:

```text
Delta2 p_t = p_t - 2p_{t-1} + p_{t-2}
Delta2 g_t = g_t - 2g_{t-1} + g_{t-2}
```

Full objective:

```text
L_BGSL = L_state + lambda_v L_velocity + lambda_a L_acceleration
```

where:

```text
L_state        = BCE(p_t, g_t) or focal/ASL state loss
L_velocity     = mean((Delta p_t - Delta g_t)^2)
L_acceleration = mean((Delta2 p_t - Delta2 g_t)^2)
```

Recommended search grid:

```text
lambda_v in {0.01, 0.05, 0.1, 0.25, 0.5}
lambda_a in {0.0, 0.01, 0.05, 0.1, 0.25}
```

Do not start with `lambda_v = 1.0` and `lambda_a = 0.5` as defaults. Those may overpower the classification signal, especially under class imbalance.

### 3.4 Optional Stability Term

An optional alarm stability term can be evaluated, but it should not be part of the primary BGSL definition unless ablations prove it helps.

Candidate:

```text
L_stable = mean_stable_periods(max(0, |Delta p_t| - epsilon)^2)
```

This should be framed as an operational extension, not the core method.

### 3.5 Why BGSL Is Not Smoothness

This distinction must be explicit in the paper.

Smoothness regularization says:

```text
Do not change risk too much.
```

BGSL says:

```text
Change risk in the correct direction, at the correct time, and with the correct temporal profile.
```

Smoothness penalizes useful early warning rises. BGSL rewards those rises when the soft onset target indicates that risk should be increasing.

This is the clean conceptual difference from total variation, ordinary temporal smoothing, and low-pass filtering.

---

## 4. Primary Application: Sepsis Early Warning

### 4.1 Why Sepsis Is the Right Flagship Domain

Sepsis is the strongest first domain for BGSL because:

1. It has clinically meaningful timestamped onset definitions.
2. Public datasets exist.
3. Deterioration often evolves over hours.
4. Early intervention is clinically valuable.
5. False alarms and alert fatigue are real deployment barriers.
6. Existing deployed sepsis models have shown poor external validity.
7. PhysioNet 2019 already includes utility-based early-warning evaluation.

Key examples supporting the motivation:

- The Epic Sepsis Model was externally validated with weak discrimination and poor sensitivity in major studies.
- Temporal Label Smoothing (TLS) shows that temporal label structure matters, making it the closest direct baseline.
- MIMIC-Sepsis provides a modern MIMIC-IV-derived sepsis trajectory benchmark with standardized onset labels and treatment variables.

### 4.2 Precise Sepsis Task

Primary task:

```text
Predict sepsis onset H hours before onset using hourly ICU trajectories.
```

Recommended horizons:

```text
H = 6h primary
H = 12h secondary
```

Prediction output:

```text
p_t = risk of sepsis onset within the next H hours
```

Label target:

```text
g_t = soft event-onset trajectory derived from t*
```

Patients already septic at or before the first usable observation should be excluded from the primary early-warning task, or analyzed separately as a detection task.

---

## 5. Datasets

### 5.1 Required Datasets

#### Dataset 1: PhysioNet/CinC 2019 Sepsis Challenge

Use as the primary benchmark.

Reasons:

- Public and widely used.
- Designed for early sepsis prediction.
- Includes a challenge utility score.
- Contains multi-hospital data.

Use:

```text
Primary development and main result table.
```

#### Dataset 2: MIMIC-IV Sepsis Cohort

Use as external validation or second primary benchmark.

Preferred route:

```text
Use the MIMIC-Sepsis benchmark pipeline if available and reproducible.
```

Reasons:

- Modern MIMIC-IV-based ICU cohort.
- Standardized trajectory construction.
- Explicit sepsis trajectory modeling.
- Stronger reproducibility than a private ad hoc Sepsis-3 implementation.

### 5.2 Optional Generality Dataset

Use one optional non-sepsis or sepsis-adjacent task only if implementation time allows.

Best option:

```text
Acute kidney injury prediction on MIMIC-IV
```

Why:

- Timestamped onset is definable.
- Deterioration evolves over time.
- Public precedent exists.
- Shows BGSL is not sepsis-only.

Second option:

```text
Circulatory failure prediction on HiRID
```

Why:

- Close to TLS and HiRID benchmark literature.
- Strong for direct comparison with TLS.

Do not over-expand the first paper. The main evidence should remain sepsis.

---

## 6. Cohort and Label Construction

### 6.1 Inclusion Criteria

Use:

- Adult ICU stays.
- Minimum observation length: 8 to 12 hours.
- Hourly time grid.
- At least one usable pre-onset segment for positive patients.
- Patient-level split, not window-level split.

### 6.2 Exclusion Criteria

Exclude or separately analyze:

- Patients septic at admission or before first usable observation.
- Stays shorter than the input history window.
- Episodes with no reliable timestamped event definition.
- Duplicate admissions that could leak across train/test splits.

### 6.3 Label Construction

For each positive patient:

```text
t* = sepsis onset time
H  = prediction horizon
```

Hard label:

```text
y_t = 1 if t in [t* - H, t*], else 0
```

Soft BGSL target:

```text
g_t = sigmoid((t - (t* - H/2)) / tau)
```

For post-onset timesteps:

```text
g_t = 1
```

For negative patients:

```text
g_t = 0
```

Masking:

- Do not compute derivative loss across padded time steps.
- Do not compute derivative loss across discontinuities between windows.
- If using sliding windows, preserve patient trajectory continuity only within the same ICU stay.

### 6.4 Multi-Horizon Setup

Primary:

```text
6-hour onset prediction
```

Secondary:

```text
12-hour onset prediction
```

Optional:

```text
3-hour horizon for short-term warning sensitivity
```

Avoid mixing horizons inside the first method unless explicitly studying multi-horizon BGSL.

---

## 7. Preprocessing

Use identical preprocessing for all models and all loss functions.

### 7.1 Time Alignment

- Resample to hourly bins.
- Use last measurement in bin or clinically defensible aggregation.
- Keep missingness indicators.
- Keep time-since-last-measured features if possible.

### 7.2 Imputation

Baseline imputation:

- Forward fill within patient.
- Backward fill only for variables where the benchmark standard permits it; otherwise avoid backward fill to reduce leakage concerns.
- Population median or normal-value fill after forward fill.
- Missingness mask included as model input.

Important:

> No imputation step may use future information relative to prediction time unless the benchmark explicitly defines retrospective preprocessing that is also used by all baselines.

For publication safety, prefer causal forward fill plus train-set statistics.

### 7.3 Normalization

- Fit normalization on training set only.
- Apply to validation/test.
- Report exact variable list and scaling.
- Store preprocessing configuration for reproducibility.

### 7.4 Feature Set

Minimum feature set:

- HR
- O2Sat
- SBP
- DBP
- MAP
- Resp
- Temp
- Age
- Sex
- Unit type if available
- Missingness indicators

Extended feature set:

- Lactate
- Creatinine
- Bilirubin
- Platelets
- WBC
- pH
- BUN
- Glucose
- Hgb
- Potassium
- Other benchmark-standard labs

For the first paper, use the feature set defined by each benchmark. Do not add treatment variables in the primary experiment unless the benchmark baseline also uses them. Treatment variables can be a sensitivity analysis.

---

## 8. Models

BGSL must be shown to be loss-level and model-agnostic.

### 8.1 Primary Architectures

Use at least three:

1. GRU or LSTM.
2. TCN.
3. Transformer encoder.

Recommended:

```text
GRU = primary reproducible baseline
TCN = temporal convolution baseline
Transformer = modern sequence model baseline
```

### 8.2 Classical Baselines

Include these for context, not as the main BGSL comparison:

- Logistic regression.
- Random forest or gradient boosting.
- XGBoost or LightGBM.
- Clinical scores: qSOFA, SIRS, NEWS/MEWS if variables permit.

These models may not output full smooth trajectories as naturally as neural sequence models, but they establish clinical and ML reference points.

---

## 9. Baselines and Ablations

This is the most important reviewer-facing section.

### 9.1 Loss Baselines

For each neural architecture, compare:

1. BCE.
2. Class-weighted BCE.
3. Focal loss.
4. Temporal Label Smoothing (TLS).
5. Smoothness penalty: `BCE + mean((Delta p_t)^2)`.
6. Total variation penalty: `BCE + mean(|Delta p_t|)`.
7. BGSL state only: `BCE(g_t)`.
8. BGSL velocity only: `BCE(g_t) + velocity`.
9. BGSL acceleration only: `BCE(g_t) + acceleration`.
10. Full BGSL: `BCE(g_t) + velocity + acceleration`.

If compute is limited, the minimum acceptable set is:

```text
BCE
Class-weighted BCE
Focal
TLS
BCE + smoothness
BGSL velocity only
Full BGSL
```

### 9.2 Component Ablations

Required:

- Hard target vs soft onset target.
- Velocity only.
- Acceleration only.
- Velocity plus acceleration.
- Different `tau`.
- Different `lambda_v`.
- Different `lambda_a`.

### 9.3 Robustness Ablations

Recommended:

- 6h vs 12h horizon.
- With and without missingness indicators.
- Per-source hospital performance on PhysioNet.
- Positive-only pre-onset derivative loss vs all-time derivative loss.
- Derivative loss on probabilities vs logits.

Default recommendation:

```text
Compute derivative losses on probabilities.
```

Sensitivity analysis:

```text
Compute derivative losses on logits to test saturation effects.
```

---

## 10. Evaluation Metrics

The paper must not rely only on AUROC.

### 10.1 Standard Discrimination

Report:

- AUROC.
- AUPRC.
- F1 at selected threshold.
- Sensitivity.
- Specificity.
- PPV.
- NPV.

AUPRC should be emphasized because sepsis is imbalanced.

### 10.2 Calibration

Report:

- Brier score.
- Expected calibration error.
- Reliability curves.
- Calibration by horizon/time-to-onset.

### 10.3 Early-Warning Utility

Report:

- PhysioNet 2019 utility score.
- Median warning lead time.
- Recall at low false alarm rates.
- Sensitivity at fixed false alarms per patient-day.

### 10.4 Trajectory Quality

Required:

```text
Alert Switching Frequency (ASF)
False Alerts Per Patient-Day (FAPPD)
Risk Total Variation (RTV)
Stable-Period Jitter (SPJ)
Pre-Onset Monotonicity Score (POMS)
Trajectory Calibration Error (TCE)
```

Definitions:

```text
ASF = number of binary alert state changes per patient-day
FAPPD = false positive alert events per non-event patient-day
RTV = mean sum_t |p_t - p_{t-1}|
SPJ = mean |Delta p_t| during stable negative periods
POMS = fraction of pre-onset intervals where p_t is non-decreasing
TCE = mean |p_t - g_t| near the event window
```

Use sustained alerts to avoid threshold flicker:

```text
Alert if p_t >= threshold for at least K consecutive hours
```

Recommended:

```text
K = 1 and K = 2 reported separately
```

### 10.5 Threshold Selection

Do not choose test thresholds directly.

Select thresholds on validation using:

1. Fixed sensitivity target, e.g. 80%.
2. Fixed false alerts per patient-day.
3. Maximum PhysioNet utility.

Then evaluate once on test.

---

## 11. Statistical Testing

Use patient-level statistics.

### 11.1 Random Seeds

Minimum:

```text
5 random seeds
```

Better:

```text
10 random seeds for final tables if compute allows
```

### 11.2 Confidence Intervals

Use:

- Patient-level bootstrap.
- 95% confidence intervals.
- Report mean and standard deviation across seeds.

### 11.3 Significance Tests

Recommended:

- Paired bootstrap for AUPRC, utility, lead time, FAPPD, ASF.
- DeLong test for AUROC if needed.
- Avoid window-level tests because windows from the same patient are correlated.

Primary statistical unit:

```text
patient or ICU stay
```

not time step.

---

## 12. Acceptance Criteria Before Paper Writing

Do not write the paper until BGSL satisfies these minimum criteria on the primary dataset:

1. AUPRC is not worse than the best non-BGSL neural baseline by more than a small tolerance.
2. BGSL reduces alert switching frequency by at least 15%.
3. BGSL reduces false alerts per patient-day or preserves it while improving lead time.
4. BGSL improves or preserves calibration.
5. BGSL benefit appears in at least two of three architectures.
6. Full BGSL beats pure smoothness on trajectory metrics.
7. Full BGSL beats TLS on at least one clinically meaningful trajectory metric.
8. No evidence of clinically harmful delayed alerting.

Ideal success:

```text
Same or better AUPRC + lower ASF/FAPPD + better lead time.
```

If AUROC/AUPRC do not improve but trajectory metrics improve strongly, the paper can still work. Frame the contribution as clinical usability and alarm stability, not pure discrimination.

---

## 13. Implementation Plan

### 13.1 Standalone Package Structure

Create a clean standalone implementation separate from APEX-MoE:

```text
research/bgsl/
  README.md
  plan.md
  bgsl/
    __init__.py
    losses.py
    targets.py
    metrics.py
    calibration.py
  datasets/
    physionet2019.py
    mimic_sepsis.py
    transforms.py
  models/
    gru.py
    tcn.py
    transformer.py
  experiments/
    configs/
    train.py
    evaluate.py
    run_ablation.py
  tests/
    test_targets.py
    test_losses.py
    test_metrics.py
```

The BGSL module must not depend on:

- APEX-MoE.
- Diffusion models.
- Ghost Bank.
- RL critic code.
- Project-specific wrappers.

### 13.2 BGSL API

Target API:

```python
loss_fn = BGSLLoss(
    state_loss="bce",
    velocity_weight=0.1,
    acceleration_weight=0.05,
    derivative_space="probability",
    reduction="mean",
)

out = loss_fn(
    logits=logits,          # [B, T]
    soft_targets=g,         # [B, T]
    mask=mask,              # [B, T], 1 = valid
)
```

Return:

```python
{
    "loss": total,
    "state": state_loss,
    "velocity": velocity_loss,
    "acceleration": acceleration_loss,
}
```

### 13.3 Unit Tests

Required tests:

1. Negative patients produce zero derivative target.
2. Positive patients produce monotonic soft target.
3. Masked time steps are ignored.
4. Derivative loss is zero when `p_t == g_t`.
5. No derivative is computed across padding boundaries.
6. Short sequences do not crash.
7. All losses are finite under all-zero/all-one targets.
8. Probability-space and logit-space derivative modes both work.

---

## 14. Experiment Matrix

### 14.1 Minimal Paper-Grade Matrix

Dataset:

```text
PhysioNet 2019
MIMIC-IV / MIMIC-Sepsis
```

Models:

```text
GRU
TCN
Transformer
```

Losses:

```text
BCE
Weighted BCE
Focal
TLS
BCE + smoothness
BGSL velocity only
Full BGSL
```

Seeds:

```text
5
```

This is:

```text
2 datasets x 3 models x 7 losses x 5 seeds = 210 runs
```

If compute is limited, use GRU for full ablations, then verify best settings on TCN and Transformer.

### 14.2 Full Matrix

Add:

- Total variation baseline.
- ASL.
- Acceleration only.
- Multiple horizons.
- Multiple tau values.
- AKI or HiRID secondary task.

---

## 15. Paper Figures and Tables

### 15.1 Required Figures

Figure 1: BGSL method diagram.

- Input sequence.
- Risk trajectory.
- State, velocity, acceleration supervision.
- Sepsis as example instantiation.

Figure 2: Soft onset target construction.

- Hard label vs soft target.
- First derivative.
- Second derivative.

Figure 3: Example patient trajectories.

- BCE vs TLS vs smoothness vs BGSL.
- Positive sepsis patient.
- Negative stable patient.

Figure 4: Alarm burden plot.

- False alerts per patient-day at fixed sensitivity.

Figure 5: Lead-time distribution.

- First true alert time before onset.

Figure 6: Ablation plot.

- State only, velocity, acceleration, full BGSL.

### 15.2 Required Tables

Table 1: Dataset characteristics.

- Patients/stays.
- Sepsis prevalence.
- Median length of stay.
- Number of time steps.
- Missingness by feature group.

Table 2: Main PhysioNet results.

- AUROC.
- AUPRC.
- Utility.
- Lead time.
- FAPPD.
- ASF.
- Calibration.

Table 3: External validation on MIMIC-Sepsis.

Table 4: Loss ablation.

Table 5: Architecture generality.

Table 6: Sensitivity analysis over `tau`, `lambda_v`, and `lambda_a`.

---

## 16. Reviewer Risk Register

### Risk 1: "This is just smoothing."

Response:

BGSL is directional derivative supervision. Smoothness penalizes all changes. BGSL rewards clinically timed rises and penalizes untimed oscillation.

Required evidence:

- Include `BCE + smoothness` and `BCE + total variation` baselines.
- Show BGSL has better lead time or pre-onset monotonicity than smoothing.

### Risk 2: "TLS already solves this."

Response:

TLS changes the pointwise target. BGSL supervises first and second temporal derivatives. The two are complementary.

Required evidence:

- Include TLS baseline.
- Include `TLS + BGSL` optional experiment if compute allows.

### Risk 3: "Sepsis onset labels are noisy."

Response:

Yes. Use multiple onset operationalizations and sensitivity to target softness `tau`.

Required evidence:

- Report performance under PhysioNet labels and MIMIC-Sepsis/Sepsis-3 labels.
- Include onset-label sensitivity analysis if feasible.

### Risk 4: "Acceleration term amplifies noise."

Response:

This is why acceleration is ablated and down-weighted.

Required evidence:

- Velocity-only vs full BGSL.
- Hyperparameter analysis for `lambda_a`.

### Risk 5: "Better trajectory metrics may hide worse discrimination."

Response:

BGSL must preserve discrimination within a clinically acceptable tolerance.

Required evidence:

- AUROC/AUPRC tables.
- Utility and operating-point metrics.

### Risk 6: "Window-level leakage."

Response:

Use patient-level splits and patient-level bootstrapping.

Required evidence:

- Explicit split description.
- No windows from the same patient across splits.

---

## 17. Generalization Beyond Sepsis

BGSL applies to tasks satisfying:

1. Sequential observations exist.
2. Event onset time is known or estimable.
3. Risk should evolve over time.
4. Early warning matters.
5. False alert burden matters.

Examples:

- Acute kidney injury.
- Respiratory failure.
- Septic shock.
- Circulatory failure.
- Cardiac arrest.
- ICU transfer.
- Rapid response activation.
- Machine failure.
- Financial default.

BGSL is not appropriate when:

- The task is static.
- Event time is unknown.
- There is no meaningful pre-event trajectory.
- Immediate spike detection is more important than stable early warning.

Paper wording:

> Although this study validates BGSL in sepsis early warning, the framework is event-agnostic: it requires only a timestamped event and a prediction horizon. We therefore view sepsis as the flagship application, not the boundary of the method.

---

## 18. Timeline

### Month 1: Foundations

- Implement standalone BGSL targets and loss.
- Implement metrics.
- Build PhysioNet 2019 loader.
- Reproduce BCE GRU baseline.
- Unit-test target construction and masking.

### Month 2: Baselines

- Implement TLS.
- Implement smoothness and total variation baselines.
- Add focal/weighted BCE.
- Add TCN and Transformer.
- Establish validation thresholding protocol.

### Month 3: Main PhysioNet Runs

- Run full GRU ablation.
- Run selected TCN and Transformer ablations.
- Generate preliminary tables.
- Identify best BGSL hyperparameters.

### Month 4: External Validation

- Build or adopt MIMIC-Sepsis pipeline.
- Run final selected models.
- Perform calibration and trajectory analysis.

### Month 5: Robustness and Optional Generality

- Horizon sensitivity.
- Tau/lambda sensitivity.
- Optional AKI or HiRID experiment.
- Statistical testing.

### Month 6: Paper

- Final figures.
- Final tables.
- Write method and experiments.
- Internal review.
- Submit to MLHC, CHIL, JAMIA, or Nature Digital Medicine depending on result strength.

---

## 19. Target Venues

Best fit:

- MLHC.
- CHIL.
- JAMIA.
- npj Digital Medicine.

Higher-risk ML venues:

- NeurIPS Datasets and Benchmarks if evaluation protocol is emphasized.
- ICLR/ICML workshop first if results are methodologically strong but not broad enough.

Clinical digital medicine venues are likely more receptive if the main gains are alarm burden, lead time, and calibration rather than raw AUROC.

---

## 20. Final Contribution Statements

Use these in the final paper:

1. We propose BGSL, a model-agnostic trajectory-supervision objective for timestamped event prediction.
2. BGSL extends pointwise event prediction by explicitly supervising risk velocity and acceleration from soft event-onset targets.
3. We validate BGSL in sepsis early warning across public ICU datasets and multiple sequence architectures.
4. We evaluate clinical usefulness using trajectory-aware metrics, including lead time, alert switching, false alerts per patient-day, and calibration.
5. We show that BGSL is distinct from and complementary to temporal label smoothing and ordinary smoothness regularization.

---

## 21. Final Abstract Draft

Early-warning models are commonly trained with pointwise classification losses, treating each time step as an independent prediction problem. This ignores the fact that clinically meaningful risk evolves as a trajectory: deterioration has direction, velocity, and acceleration. We propose Biological Gradient Supervised Learning (BGSL), a model-agnostic objective for timestamped event-risk prediction that supervises risk level and its temporal derivatives using soft event-onset targets. Unlike smoothness regularization, BGSL does not merely suppress changes in risk; it encourages risk to change in the correct direction and at the correct time. We evaluate BGSL in sepsis early warning using public ICU datasets, where onset timing, lead time, and alarm burden are clinically critical. Across recurrent, convolutional, and transformer-based sequence models, we compare BGSL against binary cross-entropy, class-weighted losses, temporal label smoothing, and smoothness baselines. Evaluation includes AUROC and AUPRC as well as warning lead time, false alerts per patient-day, alert switching frequency, and calibration. This study tests whether derivative-aware supervision can improve the clinical usability of early-warning risk trajectories while preserving predictive discrimination.

---

## 22. Key References to Verify and Cite

Use current, verified citations during manuscript writing:

1. Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016.
2. Reyna MA, et al. Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine. 2020.
3. Yèche H, Pace A, Rätsch G, Kuznetsova R. Temporal Label Smoothing for Early Event Prediction. ICML. 2023.
4. Wong A, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model. JAMA Internal Medicine. 2021.
5. Ostermayer DG, et al. External validation of the Epic sepsis predictive model in two county emergency departments. JAMIA Open. 2024.
6. Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data. 2023.
7. Huang Y, Yang Z, Rahmani A. MIMIC-Sepsis: A Curated Benchmark for Modeling and Learning from Sepsis Trajectories in the ICU. arXiv:2510.24500 / IEEE BHI 2025.
8. Hyland SL, et al. Early prediction of circulatory failure in the intensive care unit using machine learning. Nature Medicine. 2020.
9. Yèche H, et al. HiRID-ICU-Benchmark. NeurIPS Datasets and Benchmarks. 2022.
10. Le Guen V, Thome N. Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models. NeurIPS. 2019.
11. Tomašev N, et al. A clinically applicable approach to continuous prediction of future acute kidney injury. Nature. 2019.
12. GBD 2021 Global Sepsis Collaborators. Global, regional, and national sepsis incidence and mortality, 1990-2021. Lancet Global Health. 2025.

---

## 23. Final Supervisor Decision

Proceed with BGSL as the first standalone research paper.

The strongest paper is not:

```text
We made a better sepsis classifier.
```

The strongest paper is:

```text
We introduce derivative-aware trajectory supervision for early warning systems and validate it rigorously in sepsis, the most clinically important and benchmark-ready timestamped deterioration task.
```

This preserves generality, keeps the empirical scope realistic, and directly addresses reviewer concerns.
