# Research Plan: Critical Slowing Down Regularization for Learned Kalman Observers

## A General-Purpose ML Method for Early Event Prediction in Dynamical Systems

---

## 1. Novelty Verification (What I Confirmed After Exhaustive Search)

**No published paper combines:**
1. A learned (differentiable) Kalman observer as the latent state estimator
2. + Critical Slowing Down (CSD) regularization on the latent dynamics
3. + Event prediction head

**What DOES exist and how we differ:**

| Existing Work | Their Approach | Our Difference |
|--------------|----------------|----------------|
| Bury et al. PNAS 2021: "Deep learning for early warning signals of tipping points" | CNN/LSTM classifier on raw time series to classify proximity to tipping point | We use a structured Kalman observer with explicit linear dynamics + CSD regularization on the LATENT state, not raw data |
| Feng et al. 2023: "Early warning indicators via latent stochastic dynamical systems" | Learns latent SDE via diffusion maps, computes CSD indicators on latent coordinates | We have a RECURRENT KALMAN OBSERVER with closed-form state updates + stability guarantees via spectral radius constraint |
| Revach et al. 2021: "KalmanNet" | RNN replaces Kalman gain computation for state estimation | We add CSD regularization on the latent trajectory and a prediction head — they only do state estimation |
| Chen et al. 2022: "Neural Observer with Lyapunov Stability Guarantee" | Neural observer with LMI-based stability for control systems | We combine stability guarantee + CSD regularization + event prediction |
| Chinellato & Marcuzzi 2025: "Deep Kalman Filter: Regularization strategies" | Regularization for interpretable parameter learning in DKF | Our regularization is PHYSICALLY MOTIVATED from dynamical systems theory (CSD) |
| Scheffer et al. 2009 Nature: "Early-warning signals for critical transitions" | Generic CSD indicators (autocorrelation, variance) on observed signals | Our CSD operates on LEARNED LATENT STATES that are denoised by the Kalman filter |
| Liu et al. 2024 Physical Review X: "Early predictor for critical transitions in networked systems" | GIN+GRU predictor for tipping points in networks | We provide explicit latent dynamics + stability guarantees + interpretable CSD signals |

**Key insight:** No existing work uses the Kalman observer structure as a **learnable latent dynamics model** and regularizes it with CSD from dynamical systems theory. This is a genuinely novel ML method.

## 2. Core Scientific Hypothesis (General ML Formulation)

**Hypothesis:** For any dynamical system approaching a critical transition, the optimal latent representation for early event prediction should exhibit Critical Slowing Down (increased autocorrelation, increased variance, spectral radius → 1) before the event. By explicitly regularizing a learned Kalman observer's latent state to display CSD, we can:
1. Learn latent states that capture the system's loss of resilience
2. Achieve earlier event detection with better calibrated uncertainty
3. Provide an interpretable early warning signal (the latent autocorrelation itself)
4. Maintain provable stability guarantees via spectral radius constraints

**Theoretical justification:** Near a bifurcation, the dominant eigenvalue of the linearized dynamics approaches zero (in continuous time) or one (in discrete time), causing perturbations to decay more slowly — this is critical slowing down. The Kalman observer's learned dynamics matrix A captures the linearized system dynamics. When ρ(A) → 1 (or more precisely, ρ((I-KC)A) → 1), the latent state exhibits CSD. Regularizing the latent trajectory's autocorrelation toward 1 in the pre-event window is equivalent to encouraging the learned dynamics to approach instability, which is precisely the signature of an impending critical transition.

## 3. Research Questions

1. **RQ1 (Empirical — Synthetic):** On dynamical systems with known bifurcation times (fold, Hopf, period-doubling), does the CSD-regularized Kalman observer detect the approaching transition earlier and more reliably than unregularized baselines?

2. **RQ2 (Cross-Domain):** Does the method generalize across qualitatively different domains — e.g., industrial degradation (CMAPSS turbofan), physiological monitoring (sepsis), and/or financial time series?

3. **RQ3 (Ablation):** What is the contribution of each component: CSD regularization vs. spectral radius constraint vs. reconstruction loss? Does CSD regularization on latent states outperform CSD on raw observations?

4. **RQ4 (Theory):** What is the relationship between the CSD regularization strength λ_CSD and the spectral radius ρ((I-KC)A)? Can we characterize the trade-off between prediction accuracy and early warning signal quality?

5. **RQ5 (Comparison):** How does the method compare against existing approaches — raw CSD indicators, KalmanNet, SDML, and standard deep learning models (GRU, TCN, Transformer)?

## 4. Methodology

### 4.1 Model Architecture (What to keep from current codebase)

```
Input: x_t (vitals), O_t (observation mask)
    │
    ▼
Kalman Observer (learned A, C, K)
    ├── Predict:  ẑ_{t|t-1} = A ẑ_{t-1|t-1}
    ├── Innovate: ỹ_t = O_t ⊙ (x_t - C ẑ_{t|t-1})
    └── Correct:  ẑ_{t|t} = ẑ_{t|t-1} + K ỹ_t
    │
    ├──► CSD Regularization (on ẑ_{t|t} sequence)
    │     ├── Compute ρ_k(t) = lag-1 autocorrelation per latent dim
    │     ├── L_CSD = ||ρ̄(t) - ρ_target(t)||²  where ρ_target(t) → 1 before event
    │     └── Spectral radius penalty: L_spec = max(0, ρ((I-KC)A) - 0.99)
    │
    ├──► Risk MLP: r_t = σ(MLP(ẑ_{t|t}))
    │
    └──► Reconstruction: x̂_t = C ẑ_{t|t}, L_recon = MSE on observed features
```

**Total loss:**
L = L_cls + λ₁L_recon + λ₂L_csd + λ₃L_spec

Unlike the old BGSL loss, L_csd is a PHYSICS-INSPIRED regularizer, not a gradient supervision. This is a fundamentally different approach.

### 4.2 What to REMOVE from current codebase
- BGSL loss (velocity, acceleration supervision) — these are the components that failed
- All 10 baseline loss functions (move to legacy)
- EBGSL hypernetwork (unnecessary complexity)
- CMAPSS dataset support (already legacy)

### 4.3 What to ADD
- **CSD regularization module**: Computes sliding-window lag-1 autocorrelation per latent dimension, differentiable
- **Spectral radius constraint**: Power iteration to estimate ρ((I-KC)A), with penalty if > 0.99
- **CSD evaluation metrics**: Pre-onset autocorrelation trend, CSD strength, detection time before threshold crossing
- **Synthetic CSD data generator**: Generate data from near-bifurcation dynamical systems where ground truth CSD is known (for validation)
- **Baseline CSD methods**: Standard CSD on raw vitals (for comparison)

### 4.4 Dataset Strategy (General ML Paper)

**Synthetic (Primary validation):**
- Fold bifurcation: dx/dt = r - x² + noise, r → 0
- Hopf bifurcation: approach to limit cycle oscillation
- Period-doubling: logistic map approach to chaos
- Each with known tipping time → ground truth for evaluation
- Purpose: Validate that CSD regularization works when ground truth is known

**Real-world (Cross-domain generalization, pick 2-3):**
1. **CMAPSS turbofan degradation** — industrial, run-to-failure, clear failure events, partially in codebase
2. **PhysioNet 2019 sepsis** — medical, 40k patients, pipeline exists in codebase
3. **SP500 / financial crash data** — financial, regime changes, freely available (Yahoo Finance)
4. **AMOC climate index** — climate tipping point, freely available (NOAA/CMIP data)
5. **Power grid frequency** — grid instability events, freely available (FNET/GridEye)

**Recommendation:** Synthetic + 3 real-world datasets (CMAPSS + sepsis + one more) gives the strongest "general method" paper.

## 5. Experimental Protocol

### Phase 1: Synthetic Validation (Do this FIRST — 2 weeks)
**Goal**: Verify that CSD regularization works when ground truth is known.

- Generate data from fold bifurcation system: dx/dt = r - x² + noise, where r → 0 (tipping point at r=0)
- Train Kalman observer with and without L_csd on this data
- Verify: (a) latent autocorrelation increases as r → 0, (b) observer with L_csd detects tipping earlier, (c) spectral radius approaches 1 at bifurcation

**Success criteria**: CSD-regularized observer detects tipping > 50 steps earlier than unregularized, with AUC > 0.9 for "is tipping imminent?" classification.

### Phase 2: Physiological Validation (4 weeks)
**Goal**: Apply to PhysioNet 2019 sepsis data, evaluate CSD emergence.

- Train 3 variants: (1) observer without L_csd, (2) observer with L_csd, (3) raw-vitals CSD baseline
- For each positive patient, compute latent-space autocorrelation trajectory in the 24h before sepsis onset
- Statistical test: Does mean autocorrelation in [T-6h, T-0h] > [T-24h, T-12h]? (paired t-test)

**Success criteria**: Significant (p < 0.01) autocorrelation increase in pre-event window for >60% of positive patients in the L_csd-trained model.

### Phase 3: Benchmarking (4 weeks)
**Goal**: Compare against baselines on clinically meaningful metrics.

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| AUPRC | Overall discrimination | Standard fairness metric |
| Lead time | How early before onset | Clinical utility |
| ASF (Alert Switching Freq) | Stability of risk scores | Trust in deployment |
| FAPPD | False alarms per patient-day | Alarm fatigue |
| CSD strength | ρ_pre - ρ_baseline | Does our regularization work? |

**Baselines:**
1. RAW-CSD: Standard CSD on raw vitals (no latent state)
2. KALMAN-BCE: Kalman observer + BCE (no CSD reg)
3. GRU: Standard GRU predictor (no observer)
4. GRU-CSD: GRU + CSD regularization on hidden states
5. SDML: Surrogate data-based ML (Bury et al. method adapted)

### Phase 4: Ablation & Sensitivity (2 weeks)
- λ₂ (CSD weight): 0.0, 0.001, 0.01, 0.1, 1.0
- λ₃ (spectral radius weight): 0.0, 0.001, 0.01, 0.1
- Latent dimension d: 2, 4, 8, 16
- Window length W for autocorrelation: 6, 12, 24, 48 hours

## 6. Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CSD does not appear in latent states for sepsis data | Medium | This is still a publishable NEGATIVE result — "CSD not observed in physiological latent states" is novel. Pivot to synthetic systems. |
| CSD regularization harms prediction accuracy | Low-Medium | If L_csd degrades AUPRC, report the trade-off: "CSD improves lead time and interpretability at small AUPRC cost" |
| Spectral radius constraint conflicts with CSD (CSD requires ρ→1, stability requires ρ<1) | Low | We can enforce ρ < 1-ε for small ε, the trade-off is exactly the interesting science |
| Overlapping with Feng et al. 2023 | Low | They learn latent SDE, we use structured Kalman observer with explicit gain and recurrence. Completely different architecture. |
| Sepsis is not a true critical transition (too gradual) | Medium | If true, this itself is important: "Sepsis is NOT a critical transition — evidence from latent dynamics". We pivot to: "Using CSD regularization to test whether physiological events are critical transitions." |

## 7. Contribution Statement (For Paper)

**Title idea:** *"Critical Slowing Down Regularization for Learned Kalman Observers: A General Framework for Early Event Prediction in Dynamical Systems"*

**Venue focus:** ICLR, NeurIPS, ICML, or JMLR/AISTATS

**Contributions:**
1. **New regularization method:** We propose CSD regularization — a physics-inspired loss that encourages a learned Kalman observer's latent dynamics to exhibit critical slowing down (increased autocorrelation, spectral radius → 1) before critical events. This is the first integration of CSD theory from dynamical systems into end-to-end learned observers.
2. **Stability guarantee:** We introduce a differentiable spectral radius constraint on the observer dynamics ρ((I-KC)A) < 1 that ensures provable stability while allowing near-instability dynamics that precede transitions — a principled trade-off.
3. **General framework:** The method is domain-agnostic — it applies to any dynamical system with critical transitions. We validate on synthetic bifurcation systems (ground truth known) and demonstrate generalization across [3] real-world domains: [industrial degradation / physiological monitoring / financial markets].
4. **Empirical gains:** Across all domains, CSD regularization improves lead time by [X%] and early warning AUC by [Y] over unregularized baselines, while maintaining comparable AUPRC. The latent autocorrelation provides an interpretable early warning signal.

## 8. Target Venues

| Venue | Match | Deadline (approx) |
|-------|-------|-------------------|
| **ICLR 2027** | Best fit: interdisciplinary ML + dynamical systems | ~Sept 2026 |
| **NeurIPS 2026** | Good: if strong empirical results on sepsis | ~May 2026 |
| **PNAS** | If the CSD-sepsis link is scientifically significant | Rolling |
| **Nature Computational Science** | If cross-domain validation (sepsis + climate + finance) | Rolling |
| **L4DC (Learning for Dynamics & Control)** | If theory-heavy (stability analysis) | ~March 2027 |
| **Scientific Reports** | Fallback — solid but less prestigious | Rolling |

## 9. Timeline

```
Month 1: Phase 1 (synthetic validation) + code refactoring
Month 2: Phase 2 (sepsis, CSD emergence analysis)
Month 3: Phase 3 (benchmarking against all baselines)
Month 4: Phase 4 (ablation) + first draft
Month 5: Revisions, additional experiments, submit
```

## 10. Immediate Next Steps (What to do NOW)

1. **Open `src/bgsl/core/common/losses.py`** and create a new `CSDLoss` class that:
   - Takes latent state sequence [B, T, d] as input
   - Computes sliding-window lag-1 autocorrelation per dimension (vectorized, differentiable)
   - Computes ρ_target(t) as a ramp from 0 → 1 in the [τ*-H, τ*] window (same as existing soft target but for autocorrelation)
   - Returns L_csd = MSE(ρ(t), ρ_target(t))

2. **Add spectral radius constraint** using power iteration:
   - Compute M = (I - KC) @ A
   - Use power iteration (differentiable) to estimate ρ(M)
   - Penalize ρ(M) > 0.99

3. **Modify training:** Add L_csd and L_spec to the loss, update the LightningModule

4. **Create synthetic data generator:** Data from known bifurcation models with known tipping times

5. **Run Phase 1 experiments**

## 11. What You Already Have That's Useful

Everything in `docs/kalman_observer/kalman_observer.md` is directly usable:
- Section 3: Observer equations (keep as-is)
- Section 5: Error dynamics + spectral radius condition (CRITICAL for stability constraint)
- Section 6: Observation mask (keep)
- Section 13: CSD regularization (REWRITE: current version is a post-hoc diagnostic; we need it as a training-time loss)

What to DELETE from that document:
- Sections 7-10 (BGSL loss, MLP, learning objective) — not wrong but no longer the core contribution
- Sections 11-12 (EKF, Particle Filter) — good to mention as extensions but not central

## 12. Key References to Cite

1. Scheffer et al. 2009 Nature — "Early-warning signals for critical transitions" (THE foundational CSD paper)
2. Bury et al. 2021 PNAS — "Deep learning for early warning signals of tipping points" (DL + CSD)
3. Revach et al. 2021 — "KalmanNet: Neural Network Aided Kalman Filtering" (learned Kalman filter)
4. Chen et al. 2022 — "Neural Observer with Lyapunov Stability Guarantee" (stability for learned observers)
5. Feng et al. 2023 — "Early warning indicators via latent stochastic dynamical systems" (closest competitor)
6. Chinellato & Marcuzzi 2025 — "Deep Kalman Filter: Regularization strategies"
7. Liu et al. 2024 Physical Review X — "Early predictor for critical transitions in networked systems"
8. Beckers & Hirche 2020 — "Stability of learned observers" (LMI approaches)
9. Scheffer 2009 — "Critical Transitions in Nature and Society" (book, CSD theory)
10. Dakos et al. 2012 — "Methods for detecting early warnings of critical transitions" (CSD methodology)

---

## Appendix: Quick-Start Tasks

### What to implement first (in order):

1. **`CSDLoss` class** in `src/bgsl/core/common/losses.py`:
   - Input: latent state sequence [B, T, d]
   - Compute sliding-window lag-1 autocorrelation per dimension (differentiable)
   - Target: ramp from 0 → 1 in pre-event window
   - Loss: MSE(ρ(t), ρ_target(t))

2. **Spectral radius constraint** via power iteration:
   - M = (I - KC) @ A
   - Differentiable power iteration to estimate ρ(M)
   - Penalty: max(0, ρ(M) - 0.99)

3. **Synthetic data generator** (`src/bgsl/data/synthetic/`):
   - Fold bifurcation system
   - Hopf bifurcation system
   - Logistic map (period-doubling route to chaos)
   - Each with configurable noise, tipping time, and pre-tipping length

4. **Modified LightningModule**: Add CSD loss + spectral penalty + CSD metrics

5. **CSD metrics module**: Pre-onset autocorrelation trend, CSD strength, detection time

6. **Run Phase 1** (synthetic validation) — this proves the concept
