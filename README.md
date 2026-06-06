# BGSL — Biological Gradient Supervised Learning

**Gradient-Supervised Risk Trajectory Learning for Stable Early Warning**  
*Application to Sepsis Prediction in ICU Time Series*

---

## Overview

BGSL is a **model-agnostic training objective** for timestamped event-risk prediction.  
It extends standard pointwise classification by supervising not only the risk level, but also its **first and second temporal derivatives** using soft event-onset targets.

> Standard early-warning models optimize pointwise classification at each time step.  
> BGSL trains the model to learn the *trajectory* of risk as an event approaches.

### Core Loss

```
L_BGSL = L_state + λ_v · L_velocity + λ_a · L_acceleration
```

Where:
- `L_state` — BCE (or focal/ASL) against soft onset target `g_t`
- `L_velocity` — MSE between predicted and target first derivatives: `(Δp_t − Δg_t)²`
- `L_acceleration` — MSE between predicted and target second derivatives: `(Δ²p_t − Δ²g_t)²`

### Why Not Smoothness?

Smoothness regularization says: *do not change risk too much*.  
BGSL says: *change risk in the correct direction, at the correct time, with the correct profile*.

Smoothness penalizes useful early-warning rises. BGSL **rewards** those rises when the soft onset target indicates that risk should be increasing.

---

## Project Structure

```
bgsl/
├── README.md
├── requirements.txt
├── setup.py
│
├── bgsl/                     # Core BGSL method (model-agnostic)
│   ├── __init__.py
│   ├── targets.py            # SoftOnsetTarget: g_t, Δg_t, Δ²g_t
│   ├── losses.py             # BGSLLoss + all baselines (TLS, smoothness, TV, focal)
│   ├── metrics.py            # ASF, FAPPD, RTV, SPJ, POMS, TCE, AUROC, AUPRC, etc.
│   └── calibration.py        # ECE, reliability curves, TCE, calibration by horizon
│
├── datasets/                 # Data pipeline (PhysioNet 2019, MIMIC-IV)
│   ├── physionet2019.py      # Full-trajectory LMDB dataset for PhysioNet
│   ├── mimic_sepsis.py       # MIMIC-IV Sepsis cohort loader
│   └── transforms.py         # ClinicalNormalizer (isolated, no APEX-MoE deps)
│
├── models/                   # Sequence model backends
│   ├── gru.py                # GRUPredictor
│   ├── tcn.py                # TCNPredictor (dilated causal convolutions)
│   └── transformer.py        # TransformerPredictor (causal encoder)
│
├── experiments/              # Training, evaluation, ablations
│   ├── configs/              # YAML experiment configurations
│   ├── train.py              # Main training loop
│   ├── evaluate.py           # Evaluation + figure generation
│   └── run_ablation.py       # Full ablation matrix orchestration
│
└── tests/                    # Unit tests
    ├── test_targets.py
    ├── test_losses.py
    └── test_metrics.py
```

---

## Installation

```bash
cd icu_research/bgsl
pip install -e .
```

Or without editable install:

```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1. Build the Dataset

```bash
# Auto-download from Kaggle (requires kagglehub credentials):
python datasets/physionet2019.py --build --out-dir data/ready

# Or point to local raw .psv files:
python datasets/physionet2019.py --build --raw-dir data/raw --out-dir data/ready
```

### 2. Train

```bash
# GRU + Full BGSL (primary result):
python experiments/train.py \
    --config experiments/configs/physionet_gru.yaml \
    --loss bgsl \
    --seed 42

# GRU + BCE baseline:
python experiments/train.py \
    --config experiments/configs/physionet_gru.yaml \
    --loss bce \
    --seed 42
```

### 3. Evaluate

```bash
python experiments/evaluate.py \
    --checkpoint checkpoints/physionet_gru_bgsl_seed42.pt \
    --config experiments/configs/physionet_gru.yaml \
    --out-dir results/
```

### 4. Run Full Ablation

```bash
python experiments/run_ablation.py \
    --config experiments/configs/ablation_loss.yaml \
    --out-dir results/ablation/
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Data acquisition

BGSL ships a **standalone 3-tier data acquisition layer** for PhysioNet 2019.
It is invoked *before* training and is fully decoupled from `bgsl-train`,
`PhysioNetDataModule`, and the runner layer.

The chain is: **local check** → **HuggingFace download** → **build from raw**.
The first tier to succeed wins. A marker file (`.bgsl_acquired.json`) records
the last successful acquisition, so subsequent calls cost microseconds.

```bash
# Install optional extras (kagglehub + huggingface_hub + filelock)
pip install -e ".[data]"

# Edit the config to point at your HF repo (or use the kaggle-only variant).
$EDITOR experiments/configs/data_acquisition.yaml

# Acquire the data.
bgsl-acquire --config experiments/configs/data_acquisition.yaml

# Train as usual.
bgsl-train fit --config experiments/configs/physionet_gru.yaml
```

To publish a new build to a HF repo (so others can `bgsl-acquire` from it):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
bgsl-upload \
    --data-dir data/ready \
    --repo-id your-org/bgsl-physionet2019 \
    --message "Add seed=2025 build"
```

See [`docs/data_acquisition.md`](docs/data_acquisition.md) for the full
schema, force-refresh, and concurrency semantics.

---

## BGSL API

```python
from bgsl.targets import SoftOnsetTarget
from bgsl.losses import BGSLLoss

# Build soft targets
target_fn = SoftOnsetTarget(horizon_hours=6, tau=2.0)
targets = target_fn(onset_times, seq_lengths, device)
# targets: {"soft_target": [B,T], "velocity_target": [B,T],
#           "accel_target": [B,T], "valid_mask": [B,T]}

# Compute BGSL loss
loss_fn = BGSLLoss(
    state_loss="bce",
    velocity_weight=0.1,
    acceleration_weight=0.05,
    derivative_space="probability",
)
out = loss_fn(
    logits=logits,          # [B, T]  — raw model output
    soft_targets=targets["soft_target"],    # [B, T]
    vel_targets=targets["velocity_target"], # [B, T]
    acc_targets=targets["accel_target"],    # [B, T]
    mask=targets["valid_mask"],             # [B, T]
)
# out: {"loss": scalar, "state": scalar, "velocity": scalar, "acceleration": scalar}
```

---

## Evaluation Metrics

### Standard Discrimination
AUROC, AUPRC, F1, Sensitivity, Specificity, PPV, NPV

### Early-Warning Utility
PhysioNet 2019 utility score, median warning lead time, recall at fixed false alarm rates

### Trajectory Quality
| Metric | Description |
|--------|-------------|
| ASF | Alert Switching Frequency (binary state changes per patient-day) |
| FAPPD | False Alerts Per Patient-Day |
| RTV | Risk Total Variation |
| SPJ | Stable-Period Jitter |
| POMS | Pre-Onset Monotonicity Score |
| TCE | Trajectory Calibration Error |

### Calibration
Brier score, ECE, reliability curves, calibration by time-to-onset

---

## Datasets

> [!WARNING]
> **The CMAPSS dataset is legacy and will no longer be used in this project.**

| Dataset | Role | Status |
|---------|------|--------|
| PhysioNet/CinC 2019 | Primary benchmark | ✅ Implemented |
| MIMIC-IV Sepsis | External validation | 🔧 Interface stub (Month 4) |
| HiRID / AKI | Optional generality | 🔧 Future |
| CMAPSS | Turbofan engine degradation | ⚠️ Legacy (No longer used) |

---

## Baselines

All baselines are implemented in `bgsl/losses.py`:

1. BCE
2. Class-weighted BCE
3. Focal loss
4. Temporal Label Smoothing (TLS)
5. BCE + Smoothness penalty
6. BCE + Total Variation penalty
7. BGSL state only
8. BGSL velocity only
9. BGSL acceleration only
10. **Full BGSL** (state + velocity + acceleration)

---

## Acceptance Criteria (Before Paper Writing)

Per plan §12, BGSL must satisfy on the primary dataset before writing:

1. AUPRC not worse than best non-BGSL neural baseline (within tolerance).
2. ASF reduced by ≥ 15%.
3. FAPPD preserved or improved while lead time improves.
4. Calibration preserved or improved.
5. Benefit in ≥ 2 of 3 architectures.
6. Full BGSL beats pure smoothness on trajectory metrics.
7. Full BGSL beats TLS on ≥ 1 clinically meaningful trajectory metric.

---

## Citation

*Manuscript in preparation — June 2026*

> Gradient-Supervised Risk Trajectory Learning for Stable Early Warning: Application to Sepsis Prediction.

---

## References

1. Singer M et al. Sepsis-3. JAMA. 2016.
2. Reyna MA et al. PhysioNet/CinC Challenge 2019. Crit Care Med. 2020.
3. Yèche H et al. Temporal Label Smoothing. ICML. 2023.
4. Wong A et al. External Validation of Epic Sepsis Model. JAMA Intern Med. 2021.
5. Johnson AEW et al. MIMIC-IV. Scientific Data. 2023.
6. Huang Y et al. MIMIC-Sepsis. IEEE BHI. 2025.
7. Hyland SL et al. HiRID Early Prediction. Nature Medicine. 2020.
8. Tomašev N et al. AKI Prediction. Nature. 2019.
