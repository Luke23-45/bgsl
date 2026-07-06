# Final Formal Definitions for BGSL

This document defines the BGSL formulation used for the final comparison against
baseline losses. It is intended to be the mathematical source of truth for the
paper and should remain aligned with the implemented code paths:

- `src/bgsl/core/common/targets.py`
- `src/bgsl/core/sepsis/targets.py`
- `src/bgsl/core/common/losses.py`
- `src/bgsl/models/gru.py`
- `src/bgsl/train/common/module.py`
- `src/bgsl/train/sepsis/module.py`

The definition below separates the frozen minimal BGSL variants from optional
extensions that exist in the repository but are not part of the main claim.

## 1. Core Idea

BGSL, or Biological Gradient Supervised Learning, trains an early-event risk
model against a smooth target trajectory instead of only a binary event label.
For an event-positive patient, the target risk rises before the event boundary.
For an event-negative patient, the target remains zero.

The core hypothesis is:

> A model trained to match both the soft risk state and, optionally, the temporal
> derivatives of that state may learn earlier warning trajectories than a model
> trained only with hard binary cross-entropy.

The final empirical comparison tests this hypothesis against BCE and related
temporal baselines under the same architecture, data split, masking, and metric
pipeline.

## 2. Data and Notation

Let the dataset be

$$
\mathcal{D} =
\{(\mathbf{x}_i, \mathbf{m}_i, \mathbf{u}_i, y_i, \tau_i, T_i)\}_{i=1}^N .
$$

For patient \(i\):

- \(\mathbf{x}_i = (\mathbf{x}_{i,0}, \ldots, \mathbf{x}_{i,T_i-1})\), with
  \(\mathbf{x}_{i,t} \in \mathbb{R}^C\), is the dynamic feature trajectory.
- \(\mathbf{m}_i = (\mathbf{m}_{i,0}, \ldots, \mathbf{m}_{i,T_i-1})\), with
  \(\mathbf{m}_{i,t} \in \{0,1\}^C\), is the observation mask.
- \(\mathbf{u}_i\) denotes static covariates when available.
- \(y_i \in \{0,1\}\) indicates whether the event is observed.
- \(\tau_i\) is the event or onset index used by the data pipeline, and
  \(\tau_i < 0\) denotes an event-negative patient.
- \(T_i\) is the valid sequence length.

Let

$$
T_{\max} = \max_i T_i,
\qquad
M_{i,t} = \mathbf{1}[t < T_i]
$$

for \(t = 0, \ldots, T_{\max}-1\).

For the sepsis task, the repository uses \(H=6\) hours and a one-hour time step.
The data pipeline distinguishes the early-warning target used for training from
the true clinical label used for test-time clinical discrimination.

## 3. Shared Architecture

All frozen loss comparisons use the same causal sequence predictor unless a
separate architecture study explicitly states otherwise.

The primary predictor is a unidirectional GRU:

$$
\tilde{\mathbf{x}}_{i,t}
= [\mathbf{x}_{i,t}; \mathbf{m}_{i,t}]
\in \mathbb{R}^{2C}
$$

when the observation mask is used as an input channel. The recurrent state is

$$
\mathbf{h}_{i,0:T_i-1}
= \operatorname{GRU}_{\theta}(\tilde{\mathbf{x}}_{i,0:T_i-1}),
$$

with padded positions excluded by sequence packing. Each valid timestep is then
projected through the same prediction head:

$$
z_{i,t}
= W_2 \, \operatorname{Dropout}
\left(
  \operatorname{GELU}
  \left(
    W_1 \operatorname{LayerNorm}(\mathbf{h}_{i,t}) + b_1
  \right)
\right)
+ b_2 ,
$$

where \(z_{i,t}\) is a raw logit. The risk probability is

$$
p_{i,t} = \sigma(z_{i,t}) = \frac{1}{1 + \exp(-z_{i,t})}.
$$

The model is causal: \(z_{i,t}\) depends only on observations through time \(t\),
not on future timesteps.

## 4. Hard Early-Warning Target

The hard early-warning target used by BCE-style baselines is

$$
b_{i,t}
=
\mathbf{1}[y_i = 1] \,
\mathbf{1}[t \ge \tau_i - H] \,
M_{i,t}.
$$

For negative patients, \(b_{i,t}=0\) for every valid timestep.

The BCE baseline minimizes masked binary cross-entropy against \(b_{i,t}\):

$$
\mathcal{L}_{\mathrm{BCE}}
=
\frac{
  \sum_{i,t}
  M_{i,t}
  \operatorname{BCEWithLogits}(z_{i,t}, b_{i,t})
}{
  \max\left(\sum_{i,t} M_{i,t}, 1\right)
}.
$$

## 5. BGSL Soft-Onset Target

BGSL replaces the hard training target with a smooth onset trajectory. For
positive patients, define the target midpoint

$$
\mu_i = \tau_i - \frac{H}{2} + \Delta\mu_i ,
$$

scale

$$
s_i > 0,
$$

and shape

$$
\kappa_i > 0.
$$

The fixed-shape minimal BGSL setting uses

$$
\Delta\mu_i = 0,\qquad s_i = \tau,\qquad \kappa_i = 1 .
$$

In the current sepsis configuration, the default target scale is \(\tau=2.0\).

Let

$$
u_{i,t} = \frac{t-\mu_i}{s_i},
\qquad
a_{i,t} = \sigma(u_{i,t}).
$$

The BGSL soft target is the generalized logistic trajectory

$$
g_{i,t}
=
\mathbf{1}[y_i = 1] \, a_{i,t}^{\kappa_i} \, M_{i,t}.
$$

For negative patients, \(g_{i,t}=0\) at every valid timestep.

## 6. Target Derivatives

The analytic first derivative of the target trajectory is

$$
g'_{i,t}
=
\mathbf{1}[y_i = 1] \,
\frac{\kappa_i}{s_i}
a_{i,t}^{\kappa_i}
(1-a_{i,t})
M_{i,t}.
$$

The analytic second derivative is

$$
g''_{i,t}
=
\mathbf{1}[y_i = 1] \,
\frac{\kappa_i}{s_i^2}
a_{i,t}^{\kappa_i}
(1-a_{i,t})
\left(\kappa_i - (\kappa_i+1)a_{i,t}\right)
M_{i,t}.
$$

For the fixed-shape logistic case \(\kappa_i=1\), these reduce to

$$
g'_{i,t}
=
\mathbf{1}[y_i = 1]
\frac{1}{s_i} g_{i,t}(1-g_{i,t})
M_{i,t},
$$

and

$$
g''_{i,t}
=
\mathbf{1}[y_i = 1]
\frac{1}{s_i^2} g_{i,t}(1-g_{i,t})(1-2g_{i,t})
M_{i,t}.
$$

## 7. Prediction Derivatives

BGSL computes temporal derivatives of the model output. In the frozen minimal
comparison, derivatives are computed in probability space:

$$
r_{i,t} = p_{i,t}.
$$

The implementation also supports logit-space derivatives with
\(r_{i,t}=z_{i,t}\), but probability space is the default and the relevant
setting for the main comparison.

The default finite-difference operators are

$$
\Delta r_{i,t}
=
\begin{cases}
0, & t=0,\\
r_{i,t} - r_{i,t-1}, & t>0,
\end{cases}
$$

and

$$
\Delta^2 r_{i,t}
=
\begin{cases}
0, & t \in \{0,1\},\\
r_{i,t} - 2r_{i,t-1} + r_{i,t-2}, & t>1.
\end{cases}
$$

For finite differences, the velocity mask excludes \(t=0\), and the acceleration
mask excludes \(t=0\) and \(t=1\).

The repository also supports a Savitzky-Golay derivative estimator with
reflection padding. That path is an extension and should be reported separately
if used.

## 8. Masked Mean

For non-negative weights \(Q_{i,t}\), define

$$
\operatorname{MaskedMean}(A;Q)
=
\frac{
  \sum_{i=1}^N \sum_{t=0}^{T_{\max}-1} Q_{i,t} A_{i,t}
}{
  \max\left(
    \sum_{i=1}^N \sum_{t=0}^{T_{\max}-1} Q_{i,t},
    1
  \right)
}.
$$

If the denominator would be zero, the denominator is clamped to one.

## 9. Derivative Importance Weights

When onset metadata is supplied to `BGSLLoss`, derivative losses are multiplied
by temporal importance weights. For positive patients:

$$
w_{i,t}
=
\exp\left(
  -
  \frac{
    \left(t-(\tau_i-H)\right)^2
  }{
    2\sigma_w^2
  }
\right)
M_{i,t},
\qquad
\sigma_w = \frac{H}{3}.
$$

For negative patients:

$$
w_{i,t} = M_{i,t}.
$$

This weighting is intentionally separate from the soft-target midpoint
\(\mu_i=\tau_i-H/2\). The implementation emphasizes derivative supervision near
the start of the warning horizon, \(\tau_i-H\), while the soft state target is
centered halfway through the horizon.

## 10. BGSL Loss Components

### 10.1 State Loss

The frozen minimal BGSL state loss is BCE with logits against the soft target:

$$
\mathcal{L}_{\mathrm{state}}
=
\operatorname{MaskedMean}
\left(
  \operatorname{BCEWithLogits}(z_{i,t}, g_{i,t});
  M_{i,t}
\right).
$$

The implementation also supports focal, weighted-BCE, and asymmetric state-loss
variants. Those are not part of the minimal BGSL claim unless a specific
experiment names them.

### 10.2 Velocity Loss

Let \(V_{i,t}\) be the velocity mask. For finite differences,
\(V_{i,0}=0\) and \(V_{i,t}=M_{i,t}\) for \(t>0\). The weighted mask is

$$
Q^{v}_{i,t} = V_{i,t} w_{i,t}.
$$

The velocity loss is

$$
\mathcal{L}_{\mathrm{vel}}
=
\operatorname{MaskedMean}
\left(
  (\Delta r_{i,t} - g'_{i,t})^2;
  Q^{v}_{i,t}
\right).
$$

### 10.3 Acceleration Loss

Let \(A_{i,t}\) be the acceleration mask. For finite differences,
\(A_{i,0}=A_{i,1}=0\) and \(A_{i,t}=M_{i,t}\) for \(t>1\). The weighted mask is

$$
Q^{a}_{i,t} = A_{i,t} w_{i,t}.
$$

The acceleration loss is

$$
\mathcal{L}_{\mathrm{acc}}
=
\operatorname{MaskedMean}
\left(
  (\Delta^2 r_{i,t} - g''_{i,t})^2;
  Q^{a}_{i,t}
\right).
$$

### 10.4 Monotonicity Penalty

The implementation includes an optional monotonicity penalty. It is disabled in
the minimal frozen comparisons when \(\lambda_m=0\).

For completeness, its mask is

$$
R^{m}_{i,t}
=
M_{i,t}
\mathbf{1}[y_i=1]
\mathbf{1}[\tau_i-H \le t \le \tau_i],
$$

and the penalty is

$$
\mathcal{L}_{\mathrm{mono}}
=
\operatorname{MaskedMean}
\left(
  \max(0,-\Delta p_{i,t})^2;
  R^{m}_{i,t}
\right).
$$

## 11. Total BGSL Objective

The implemented BGSL objective is

$$
\mathcal{L}_{\mathrm{BGSL}}
=
\mathcal{L}_{\mathrm{state}}
+ \lambda_v \mathcal{L}_{\mathrm{vel}}
+ \lambda_a \mathcal{L}_{\mathrm{acc}}
+ \lambda_m \mathcal{L}_{\mathrm{mono}}.
$$

The frozen comparison should name the exact variant:

| Variant | State target | \(\lambda_v\) | \(\lambda_a\) | \(\lambda_m\) | Derivative method |
| --- | --- | ---: | ---: | ---: | --- |
| BCE | hard \(b_{i,t}\) | 0 | 0 | 0 | none |
| TLS | linear ramp \(c_{i,t}\) | 0 | 0 | 0 | none |
| BGSL-state | soft \(g_{i,t}\) | 0 | 0 | 0 | finite difference unused |
| BGSL+vel | soft \(g_{i,t}\) | positive | 0 | 0 | finite difference |
| Full BGSL | soft \(g_{i,t}\) | positive | positive | 0 unless stated | finite difference |

For the core hypothesis runner, the named full BGSL setting is
\(\lambda_v=0.10\), \(\lambda_a=0.05\), \(\lambda_m=0\), and finite-difference
derivatives. For publication tables, use the exact values from the frozen run
manifest rather than rewriting them from memory.

## 12. Baselines Used for Comparison

All baselines receive the same logits \(z_{i,t}\), valid mask \(M_{i,t}\), and
architecture unless a separate architecture experiment explicitly changes the
backbone.

### 12.1 BCE

BCE trains against the hard early-warning target \(b_{i,t}\):

$$
\mathcal{L}_{\mathrm{BCE}}
=
\operatorname{MaskedMean}
\left(
  \operatorname{BCEWithLogits}(z_{i,t}, b_{i,t});
  M_{i,t}
\right).
$$

### 12.2 Weighted BCE

Weighted BCE applies a positive-class multiplier \(\alpha_+\):

$$
\mathcal{L}_{\mathrm{WBCE}}
=
\operatorname{MaskedMean}
\left(
  \omega_{i,t}
  \operatorname{BCEWithLogits}(z_{i,t}, b_{i,t});
  M_{i,t}
\right),
$$

where

$$
\omega_{i,t}
=
\begin{cases}
\alpha_+, & b_{i,t} > 0.5,\\
1, & b_{i,t} \le 0.5.
\end{cases}
$$

### 12.3 Focal Loss

Focal loss uses

$$
\mathcal{L}_{\mathrm{focal}}
=
\operatorname{MaskedMean}
\left(
  \alpha_t (1-p_t^*)^\gamma
  \operatorname{BCEWithLogits}(z_{i,t}, b_{i,t});
  M_{i,t}
\right),
$$

where

$$
p_t^*
=
\begin{cases}
p_{i,t}, & b_{i,t} > 0.5,\\
1-p_{i,t}, & b_{i,t} \le 0.5.
\end{cases}
$$

The implementation default is \(\gamma=2.0\) and \(\alpha=0.25\) when focal
loss is instantiated without overrides.

### 12.4 Temporal Label Smoothing

TLS constructs a linear ramp target from the event time:

$$
c_{i,t}
=
\mathbf{1}[y_i=1]
\operatorname{clip}
\left(
  1 - \frac{\tau_i - t}{\alpha},
  0,
  1
\right)
M_{i,t}.
$$

The TLS loss is BCE with logits against \(c_{i,t}\):

$$
\mathcal{L}_{\mathrm{TLS}}
=
\operatorname{MaskedMean}
\left(
  \operatorname{BCEWithLogits}(z_{i,t}, c_{i,t});
  M_{i,t}
\right).
$$

### 12.5 BCE With Smoothness

The smoothness baseline adds an unsupervised squared-difference penalty:

$$
\mathcal{L}_{\mathrm{smooth}}
=
\mathcal{L}_{\mathrm{BCE}}
+ \lambda_s
\operatorname{MaskedMean}
\left(
  (\Delta p_{i,t})^2;
  M_{i,t}\mathbf{1}[t>0]
\right).
$$

### 12.6 BCE With Total Variation

The total-variation baseline adds an absolute-difference penalty:

$$
\mathcal{L}_{\mathrm{TV}}
=
\mathcal{L}_{\mathrm{BCE}}
+ \lambda_{\mathrm{TV}}
\operatorname{MaskedMean}
\left(
  |\Delta p_{i,t}|;
  M_{i,t}\mathbf{1}[t>0]
\right).
$$

## 13. Evaluation Targets and Metrics

Training losses and test metrics do not use exactly the same target stream.
This distinction must be preserved in the manuscript.

- BGSL state training uses the soft target \(g_{i,t}\).
- BCE-style baseline training uses the hard early-warning target \(b_{i,t}\).
- Step-level clinical AUROC/AUPRC use the true clinical label stream when it is
  available as `hard_labels`.
- The warning-window target metrics use \(b_{i,t}\).
- Trajectory and utility metrics use patient-level probability sequences,
  selected thresholds, onset metadata, and the domain metric implementation.

Therefore, a comparison between BCE and BGSL is a comparison of training
objectives under a shared predictor and shared evaluation pipeline, not a claim
that both methods optimize the same target.

## 14. Scope of the Final Claim

The final manuscript should restrict its claim to the tested formulation:

- fixed-shape soft-onset target unless a run explicitly uses a hypernetwork;
- GRU predictor unless the table explicitly reports another architecture;
- probability-space derivatives unless stated otherwise;
- finite-difference derivatives for the frozen minimal comparison unless the
  frozen manifest states otherwise;
- \(\lambda_m=0\) unless monotonicity is explicitly enabled;
- no utility-aware auxiliary objective unless `UtilityAwareBGSLLoss` is named.

The correct claim is:

> Under the tested benchmark, architecture, target construction, and loss
> weights, the frozen minimal BGSL formulation did not outperform BCE for
> early-event prediction.

The definition does not imply that all soft-target methods, all derivative
supervision methods, or all future BGSL variants are invalid.

## 15. Known Corrections Relative to `bgsl_math.md`

This file updates the earlier formal draft in the following ways:

- The implemented target is the generalized logistic \(g=s^\kappa\), with the
  fixed minimal case recovered by \(\kappa=1\).
- The soft-target midpoint is \(\tau_i-H/2\), but derivative importance weights
  are centered at \(\tau_i-H\) in the implementation.
- The frozen minimal state loss is BCE with logits against soft targets, not a
  focal state loss unless explicitly configured.
- Finite differences are the default derivative method for the minimal BGSL
  baseline; Savitzky-Golay derivatives are an optional extension.
- Monotonicity and utility-aware terms exist in code but are not part of the
  minimal frozen BGSL comparison unless explicitly enabled.
- The GRU architecture is part of the comparison definition because the claim is
  about a loss objective under a shared causal backbone.

