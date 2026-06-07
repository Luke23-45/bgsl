# Biological Gradient Supervised Learning (BGSL) — Formal Mathematical Definition

---

## 1. Problem Setup

### 1.1 Notation

Let a dataset consist of $N$ patients indexed by $i = 1, \dots, N$. For each patient,
we observe a multivariate time series of vitals and laboratory measurements over
$T_i$ discrete time steps:

$$
\mathbf{x}_i = \{ \mathbf{x}_{i,t} \in \mathbb{R}^D \mid t = 0, 1, \dots, T_i - 1 \},
$$

where $D$ is the number of physiological channels, and each time step corresponds
to one hour (sepsis) or one operational cycle (CMAPSS).

Each patient also has associated metadata:

$$
\begin{aligned}
y_i &\in \{0, 1\} &&\text{(binary event label: sepsis onset / failure)}, \\
\tau_i^* &\in [0, T_i) \cup \{-1\} &&\text{(onset time; $-1$ if no event)}, \\
H &\in \mathbb{R}^+ &&\text{(prediction horizon, in time units)}.
\end{aligned}
$$

### 1.2 Hard Label Construction

The standard hard binary label for early event prediction is defined over a
prediction horizon $H$:

> **Definition 1 (Hard Target).** For patient $i$ with onset time $\tau_i^*$, the hard target for each
> time step $t$ is:
> $$
> g_{i,t}^{\text{hard}} = \mathbb{1}\{\tau_i^* \ge 0\} \cdot \mathbb{1}\{t \ge \tau_i^* - H\},
> $$
> where $\mathbb{1}\{\cdot\}$ is the indicator function. For negative patients
> ($\tau_i^* = -1$), $g_{i,t}^{\text{hard}} = 0$ for all $t$.

This trajectory is binary: it transitions abruptly from 0 to 1 at the
$\tau_i^* - H$ decision boundary. Training against this hard target with
standard BCE loss encourages predictions to be above threshold after onset
but provides no graded supervisory signal about **how** the risk score
should evolve as the event approaches.

### 1.3 Valid Mask

Because sequences have varying lengths, we define a length mask:

$$
m_{i,t} = \mathbb{1}\{t < T_i\}.
$$

This mask zeroes out padded positions in minibatch collation. Derivative masks
additionally exclude boundary positions where finite differences are undefined:

$$
\begin{aligned}
m^{\text{vel}}_{i,t} &= m_{i,t} \cdot \mathbb{1}\{t \ge 1\}, \\
m^{\text{acc}}_{i,t} &= m_{i,t} \cdot \mathbb{1}\{t \ge 2\}.
\end{aligned}
$$

---

## 2. Soft Target Trajectory

The core enabling representation of BGSL is a **soft onset target**
trajectory. Instead of the binary jump, we construct a smooth, differentiable
target that ramps from 0 to 1 as the event approaches, and we compute its
first and second temporal derivatives.

> **Definition 2 (Soft Onset Target).** For patient $i$ with onset time $\tau_i^* \ge 0$, define the sigmoid
> midpoint:
> $$
> c_i = \tau_i^* - \frac{H}{2}.
> $$
> The soft target at time step $t$ is piecewise:
> $$
> g_{i,t} =
> \begin{cases}
> 0, & t < \tau_i^* - H, \\[4pt]
> \sigma\!\left(\dfrac{t - c_i}{\tau}\right), & \tau_i^* - H \le t < \tau_i^*, \\[8pt]
> g_{\text{post}}, & t \ge \tau_i^*,
> \end{cases}
> $$
> where:
> - $\tau > 0$ is a temperature parameter controlling the ramp steepness,
> - $g_{\text{post}} \in [0,1]$ is the target value after onset (typically $1.0$),
> - $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid.
>
> For negative patients ($\tau_i^* = -1$), $g_{i,t} = 0$ for all $t$.
>
> Smaller $\tau$ yields a sharper transition centered at $c_i$; larger $\tau$
> yields a flatter transition around the midpoint. The sigmoid is evaluated
> only inside the ramp window $[\tau_i^* - H, \tau_i^*)$, so the post-onset
> plateau is constant at $g_{\text{post}}$ and the pre-ramp period is zero.

### 2.1 Derivative Targets

The velocity (first derivative) and acceleration (second derivative) of the
soft target are defined via finite differences:

$$
\begin{aligned}
\Delta g_{i,t} &=
\begin{cases}
0, & t = 0, \\
g_{i,t} - g_{i,t-1}, & t \ge 1,
\end{cases} \\[6pt]
\Delta^2 g_{i,t} &=
\begin{cases}
0, & t \le 1, \\
g_{i,t} - 2 g_{i,t-1} + g_{i,t-2}, & t \ge 2.
\end{cases}
\end{aligned}
$$

These provide the supervisory signal for the velocity and acceleration
components of the BGSL loss.

### 2.2 Relationship to Temporal Label Smoothing (TLS)

Temporal Label Smoothing [Yeche et al., ICML 2023] constructs targets using a
linear ramp:

$$
y^{\text{TLS}}_{i,t} = \max\!\big(0,\; \min\!\big(1,\; 1 - \tfrac{\tau_i^* - t}{\alpha}\big)\big),
$$

where $\alpha$ is the smoothing window. BGSL's sigmoid-based soft target
differs from the linear TLS ramp in two ways: (i) it uses a logistic
sigmoid rather than a piecewise linear function, providing smooth
derivatives everywhere within the ramp window; (ii) the temperature
parameter $\tau$ controls the steepness — smaller $\tau$ yields a sharper
transition, larger $\tau$ yields a flatter one. The sigmoid formulation
provides smooth derivatives everywhere, which is essential for derivative supervision.

---

## 3. Backbone Architecture

Let $\phi(\cdot; \theta)$ denote the backbone sequence model parameterized
by $\theta$. For the experiments in this work, we use a Gated Recurrent
Unit (GRU):

$$
\mathbf{h}_{i,t} = \text{GRU}(\mathbf{x}_{i,t}, \mathbf{h}_{i,t-1}; \theta), \qquad
z_{i,t} = \mathbf{w}^\top \mathbf{h}_{i,t} + b,
$$

where $\mathbf{h}_{i,t} \in \mathbb{R}^{d_h}$ is the hidden state and
$z_{i,t} \in \mathbb{R}$ is the scalar logit output. The predicted risk probability
is obtained via the logistic sigmoid:

$$
p_{i,t} = \sigma(z_{i,t}) = \frac{1}{1 + e^{-z_{i,t}}}.
$$

---

## 4. Loss Functions

### 4.1 BGSL Loss (Primary)

The Biological Gradient Supervised Learning loss is a composite objective with
three terms:

$$
\boxed{\;
\mathcal{L}_{\text{BGSL}} = \mathcal{L}_{\text{state}}(z, g, m)
+ \lambda_v \cdot \mathcal{L}_{\text{velocity}}(\Delta p, \Delta g, m^{\text{vel}})
+ \lambda_a \cdot \mathcal{L}_{\text{acceleration}}(\Delta^2 p, \Delta^2 g, m^{\text{acc}})
\;}
$$

#### State Loss (Level Supervision)

The state loss measures the discrepancy between the predicted probability
$p_t$ and the soft target $g_t$. Several variants are supported:

$$
\mathcal{L}_{\text{state}}(z, g, m) =
\begin{cases}
\mathcal{L}_{\text{BCE}}(z, g, m), & \text{state\_loss = ``bce''} \\
\mathcal{L}_{\text{WBCE}}(z, g, m), & \text{state\_loss = ``weighted\_bce''} \\
\mathcal{L}_{\text{focal}}(z, g, m), & \text{state\_loss = ``focal''} \\
\mathcal{L}_{\text{ASL}}(z, g, m), & \text{state\_loss = ``asl''}
\end{cases}
$$

**Binary Cross-Entropy (default)** — computed via logits for numerical stability:

$$
\mathcal{L}_{\text{BCE}}(z, g, m) =
\frac{ \sum_t m_t \cdot \ell_{\text{BCEWithLogits}}(z_t, g_t) }
{ \max(\sum_t m_t,\; 1) },
$$

where $\ell_{\text{BCEWithLogits}}(z, g) = \log(1 + e^{-z}) + (1-g)z$ for
$g \in [0,1]$, i.e.\ the logit-sigmoid form $\log(1 + e^{(1-2g)z}) - (1-g)z$,
which is numerically equivalent to $-g\log\sigma(z) - (1-g)\log(1-\sigma(z))$
but avoids intermediate exponentiation.

**Weighted BCE** — with continuous target-dependent weighting:

$$
\mathcal{L}_{\text{WBCE}} = \frac{ \sum_t m_t \cdot w_t \cdot \ell_{\text{BCEWithLogits}}(z_t, g_t) }
{ \max(\sum_t m_t,\; 1) },
$$

where $w_t = 1 + (w_{\text{pos}} - 1) \, g_t$. This interpolates linearly
between weight $1$ at $g_t = 0$ and weight $w_{\text{pos}}$ at $g_t = 1$,
preserving continuous gradient flow compatible with soft targets.

**Focal Loss**:

$$
\mathcal{L}_{\text{focal}} = \frac{ \sum_t m_t \cdot (1 - p_t^*)^\gamma \cdot \big[ -g_t \log p_t - (1-g_t) \log (1-p_t) \big] }
{ \max(\sum_t m_t,\; 1) },
$$

where $p_t^* = p_t \cdot g_t + (1-p_t) \cdot (1-g_t)$ is the probability
assigned to the correct class, and $\gamma \ge 0$.

**Asymmetric Loss (ASL)** (see \cite{ridnik2021asl}):

Let $p = \sigma(z)$, $p_m = \max(p - m, 0)$, with asymmetric focusing
parameters $\gamma_- > \gamma_+ \ge 0$ and margin $m \ge 0$:

$$
\mathcal{L}_{\text{ASL}} =
\frac{ \sum_t m_t \cdot \big[
g_t \; (1-p_t)^{\gamma_+} \log(p_t)
\;+\; (1-g_t) \; (p_{m,t})^{\gamma_-} \log(1 - p_{m,t})
\big] }
{ \max(\sum_t m_t,\; 1) }.
$$

The asymmetric focusing ($\gamma_- > \gamma_+$) down-weights easy negative
examples more aggressively than easy positives, reducing false positives
in multi-label settings. In BGSL, ASL is an experimental variant and
not the default state loss; the soft-target trajectory requires
adaptation of the standard binary ASL formulation.

#### Velocity Loss (First-Derivative Supervision)

The velocity loss penalizes deviation of the prediction's first finite
difference from the target's first finite difference:

$$
\mathcal{L}_{\text{velocity}}(\Delta p, \Delta g, m^{\text{vel}})
= \frac{ \sum_t m^{\text{vel}}_t \cdot (\Delta p_t - \Delta g_t)^2 }
{ \max(\sum_t m^{\text{vel}}_t,\; 1) }.
$$

This encourages the model to increase its risk score at the same rate as
the soft target ramp, rather than allowing arbitrary dynamics.

#### Acceleration Loss (Second-Derivative Supervision)

The acceleration loss penalizes deviation of the prediction's second finite
difference from the target's second finite difference:

$$
\mathcal{L}_{\text{acceleration}}(\Delta^2 p, \Delta^2 g, m^{\text{acc}})
= \frac{ \sum_t m^{\text{acc}}_t \cdot (\Delta^2 p_t - \Delta^2 g_t)^2 }
{ \max(\sum_t m^{\text{acc}}_t,\; 1) }.
$$

This encourages convexity/concavity consistency: the model should accelerate
its risk increase at the correct time (when the sigmoid enters its linear
region), not before or after.

#### Derivative Space

All derivative losses are computed in probability space:
$s_t = p_t = \sigma(z_t)$. The targets $g_t$, $\Delta g_t$, and
$\Delta^2 g_t$ are also formulated in probability space, ensuring
domain-consistent supervision.

#### Default Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Velocity weight | $\lambda_v$ | 0.10 | Scales first-derivative supervision |
| Acceleration weight | $\lambda_a$ | 0.05 | Scales second-derivative supervision |
| Temperature | $\tau$ | 2.0 | Soft target ramp steepness |
| Horizon | $H$ | 6.0 (sepsis) / 30 (CMAPSS) | Prediction horizon |
| State loss type | — | `"bce"` | Level supervision variant |
| Post-onset value | $g_{\text{post}}$ | 1.0 | Target value after onset |
| Derivative space | — | `"probability"` | Domain for derivative computation |

### 4.2 Baseline Loss: Temporal Label Smoothing (TLS)

Following [Yeche et al., ICML 2023], the TLS loss uses a linear ramp target
constructed on-the-fly from onset times:

$$
\mathcal{L}_{\text{TLS}} = \frac{ \sum_t m_t \cdot w_t \cdot \mathcal{L}_{\text{BCE}}(z_t, y^{\text{TLS}}_t) }
{ \max(\sum_t m_t,\; 1) },
$$

where $y^{\text{TLS}}_t = \max(0, \min(1, 1 - \frac{\tau^* - t}{\alpha}))$
and $\alpha = 6.0$ is the smoothing window.

### 4.3 Baseline Loss: BCE + Smoothness Penalty

$$
\mathcal{L}_{\text{Smooth}} = \mathcal{L}_{\text{BCE}}(z, g^{\text{hard}}, m)
+ \lambda_s \cdot \frac{ \sum_t m^{\text{vel}}_t \cdot (\Delta p_t)^2 }
{ \max(\sum_t m^{\text{vel}}_t,\; 1) },
$$

where $\lambda_s = 0.1$. This penalizes the L2 norm of the first difference,
encouraging the prediction to change slowly regardless of event timing.

### 4.4 Baseline Loss: BCE + Total Variation Penalty

$$
\mathcal{L}_{\text{TV}} = \mathcal{L}_{\text{BCE}}(z, g^{\text{hard}}, m)
+ \lambda_{\text{tv}} \cdot \frac{ \sum_t m^{\text{vel}}_t \cdot |\Delta p_t| }
{ \max(\sum_t m^{\text{vel}}_t,\; 1) },
$$

where $\lambda_{\text{tv}} = 0.1$. The L1 penalty on first differences
encourages piecewise-constant risk trajectories.

### 4.5 Conceptual Comparison

- **BCE**: No temporal structure; all pre-onset timesteps are class 0, all post-horizon timesteps are class 1.
- **Smoothness / TV**: Penalize prediction change magnitude, but do not distinguish "good" change (ramping up for approaching event) from "bad" change (jitter far from onset).
- **TLS**: Provides a graded target level but no derivative signal; the model must infer the correct rate of increase from the level target alone.
- **BGSL**: Provides full kinematic supervision — explicit targets for position (level), velocity (first derivative), and acceleration (second derivative) of the risk trajectory.

---

## 5. Training Procedure

### 5.1 Optimization

We optimize the objective with AdamW and a cosine annealing learning-rate
schedule:

$$
\theta^* = \argmin_\theta \; \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathcal{L}(\mathbf{x}_i, g_i, \Delta g_i, \Delta^2 g_i; \theta),
$$

with initial learning rate $\eta = 10^{-3}$, weight decay $\beta = 10^{-4}$,
and $E_{\max} = 50$ epochs.

### 5.2 Threshold Selection

After each validation epoch, select a fixed operating threshold $\theta_{\text{thr}}$
that achieves a target sensitivity on the validation set:

**Algorithm: Threshold Selection via Fixed Sensitivity**

1. $\Theta \gets \text{linspace}(0, 1, K_{\text{grid}})$ where $K_{\text{grid}} = 200$
2. $\theta_{\text{thr}} \gets 0.5$ (default fallback)
3. **For** each $\theta \in \Theta$:
   - Compute sensitivity $s = \frac{TP}{\max(TP + FN, \varepsilon)}$, $\varepsilon = 10^{-9}$
   - **If** $s \ge s^*$ ($s^* = 0.80$): $\theta_{\text{thr}} \gets \theta$
4. **Return** $\theta_{\text{thr}}$ (highest threshold meeting target)

The selected threshold $\theta_{\text{thr}}$ is persisted as a buffer in the
Lightning module and used during test evaluation.

### 5.3 Sustained Alert Condition

An alert is fired only after $K_{\text{alert}}$ consecutive timesteps above threshold:

$$
\text{alert}_t =
\mathbb{1}\!\left\{\;
\min_{j=0,\dots,K_{\text{alert}}-1} p_{t-j} \ge \theta_{\text{thr}}
\;\right\},
\qquad t \ge K_{\text{alert}} - 1,
$$

with $K_{\text{alert}} = 1$ (no sustain requirement) by default.

---

## 6. Evaluation Metrics

### 6.1 Standard Discrimination Metrics

Aggregated over all valid patient-timesteps (per-step evaluation):

$$
\begin{aligned}
\text{AUROC} &= \text{area under the ROC curve (over all valid timesteps)} \\
\text{AUPRC} &= \text{area under the precision-recall curve} \\
F_1 &= \frac{2 \cdot TP}{2 \cdot TP + FP + FN} \\
\text{Brier} &=
\frac{ \sum_{i,t} m_{i,t} \cdot (p_{i,t} - g^{\text{hard}}_{i,t})^2 }
{ \max(\sum_{i,t} m_{i,t},\; 1) } \\
\text{ECE} &= \sum_{b=1}^{B} \frac{n_b}{N_{\text{valid}}}
\big| \text{conf}_b - \text{acc}_b \big|, \quad B = 10
\end{aligned}
$$

where $N_{\text{valid}} = \sum_{i,t} m_{i,t}$ is the total number of valid
patient-timesteps. For ECE, predictions are partitioned into $B$ equally
spaced confidence bins over $[0,1]$ using only valid timesteps;
$\text{conf}_b$ is the mean predicted probability in bin $b$ and
$\text{acc}_b$ is the fraction of positive hard labels in that bin.

### 6.2 Trajectory (Patient-Level) Metrics

Computed per-patient, then aggregated across the cohort (mean $\pm$ std,
with bootstrap 95\% confidence intervals over $n_{\text{bootstrap}} = 1000$
resamples).

> **Definition 3 (Risk Trajectory Variation — RTV).**
> $$
> \text{RTV}_i = \sum_{t=1}^{T_i - 1} |p_{i,t} - p_{i,t-1}|.
> $$
> Lower is smoother. RTV measures the cumulative jitter in the risk signal.

> **Definition 4 (Stable-Period Jitter — SPJ).**
> $$
> \text{SPJ}_i = \frac{1}{|\mathcal{S}_i|} \sum_{t \in \mathcal{S}_i} |p_{i,t} - p_{i,t-1}|,
> $$
> where $\mathcal{S}_i = \{ t \mid g^{\text{hard}}_{i,t} < 0.5 \land t \ge 1 \}$.
> Lower is better: the model should be steady when the patient is stable.

> **Definition 5 (Alert Stability Factor — ASF).**
> $$
> \text{ASF}_i = \frac{ \sum_{t=1}^{T_i-1} |\text{alert}_{i,t+1} - \text{alert}_{i,t}| }
> { T_i }.
> $$
> Lower is better: clinicians cannot act on a signal that oscillates.

> **Definition 6 (Pre-Onset Monotonicity Score — POMS).**
> $$
> \text{POMS}_i = \frac{ \sum_{t = \tau_i^* - H}^{\tau_i^* - 1} \mathbb{1}\{ p_{i,t+1} \ge p_{i,t} \} }
> { H },
> $$
> for positive patients only. Higher is better: risk should only rise as the
> critical event approaches.

> **Definition 7 (Temporal Calibration Error — TCE).**
> $$
> \text{TCE}_i = \frac{1}{2W + 1} \sum_{t = \tau_i^* - W}^{\tau_i^* + W}
> |p_{i,t} - g_{i,t}|,
> $$
> where $W = 3$ (default). Lower is better: the prediction should match the
> idealized soft trajectory.

> **Definition 8 (Lead Time).**
> For each positive patient where an alert fires before onset:
> $$
> \text{lead\_time}_i = \tau_i^* - \text{first\_alert\_time}_i.
> $$
> Reported as mean, median, and interquartile range across the positive cohort.

### 6.3 Clinical Utility Metrics

#### PhysioNet Sepsis Utility Score

We evaluate clinical utility following the official PhysioNet/CinC Challenge
2019 scoring function \cite{reyna2020sepsis}. The score is defined per
patient-hour (time window) and aggregated across all patient-hours, then
normalized so that a no-predictions baseline scores 0 and the optimal
classifier scores 1.

For each patient $s$ and each hourly time window $t \in T(s)$, let
$\text{TP}(s,t)$ indicate a true positive alert, $\text{FP}(s,t)$ a false
positive, $\text{FN}(s,t)$ a false negative, and $\text{TN}(s,t)$ a true
negative. The per-window utility $U(s,t)$ is:

$$
U(s,t) =
\begin{cases}
+3, & \text{TP at } t \text{ where } \tau_s^* - t \ge 12 \text{ hours}
\text{ (very early)}, \\
+2, & \text{TP at } t \text{ where } 4 \le \tau_s^* - t < 12 \text{ hours}
\text{ (early)}, \\
+1, & \text{TP at } t \text{ where } 0 \le \tau_s^* - t < 4 \text{ hours}
\text{ (on time)}, \\
0, & \text{TN at } t, \\
-1, & \text{FP at } t, \\
-1, & \text{FN at } t \text{ (missed),} \\
-2, & \text{TP at } t \text{ where } \tau_s^* - t < 0 \text{ (late)}.
\end{cases}
$$

The total raw utility is:

$$
U_{\text{total}} = \sum_{s \in S} \sum_{t \in T(s)} U(s,t).
$$

The normalized utility is:

$$
U_{\text{norm}} = \frac{U_{\text{total}} - U_{\text{no predictions}}}
{U_{\text{optimal}} - U_{\text{no predictions}}},
$$

where $U_{\text{no predictions}}$ is the score achieved by a classifier that
never predicts sepsis (all $\text{FN}$ for positive windows, all $\text{TN}$
for negative windows), and $U_{\text{optimal}}$ is the score of a perfect
classifier (all $\text{TP}$ for positive windows before onset, all $\text{TN}$
otherwise).

This normalized score lies in $[0, 1]$, with higher values indicating greater
clinical utility. The official implementation is available from the
PhysioNet Challenge repository.

**Remark.** The per-patient formulation often used in the literature
($U_i = \max(0, \text{lead\_time}_i/H)$ for alerted sepsis cases with
linear false-alarm penalties) is a simplified proxy. In this work we use
the official per-window score for all reported utility values.
#### False Alarm Probability Per Day (FAPPD) — Auxiliary Metric

Our own patient-level false-alarm density measure:

$$
\text{FAPPD} = \frac{ \sum_{i: y_i = 0} \text{FP\_alert\_events}_i }
{ \sum_{i: y_i = 0} T_i } \times 24.
$$

The multiplication by 24 converts from per-hour (the sepsis data resolution)
to per-day.

#### False Alarm Rate — Auxiliary Metric

For negative patients only:

$$
\text{FA\_rate} = \frac{ \sum_i \text{FP\_alert\_events}_i }
{ \sum_i T_i }.
$$

### 6.4 Bootstrap Confidence Intervals

For every trajectory metric, compute 95\% confidence intervals using the
percentile bootstrap with $n_{\text{bootstrap}} = 1000$ resamples:

1. Compute point estimate $\hat{m} = \text{mean}(\{m_i\})$
2. For $b = 1, \dots, n_{\text{bootstrap}}$:
   - Resample $\{m_i^*\} \sim \{m_i\}$ with replacement (same size $N_P$)
   - Compute $\hat{m}^{(b)} = \text{mean}(\{m_i^*\})$
3. $CI_{\text{low}} = \text{percentile}(\{\hat{m}^{(b)}\}, \alpha/2)$
4. $CI_{\text{high}} = \text{percentile}(\{\hat{m}^{(b)}\}, 1 - \alpha/2)$

---

## 7. Data Preprocessing

### 7.1 Clinical Normalizer (Sepsis)

1. **NaN/Inf recovery**: NaN $\to$ 0, $+\infty \to 1$, $-\infty \to -1$.
2. **Physics clamp**: $x_c \gets \text{clamp}(x_c, \text{phys\_min}_c, \text{phys\_max}_c)$.
3. **Log1p transform** (for heavy-tailed channels):
   $$
   x_c \gets \log(1 + \max(x_c, 0)),
   $$
   applied to lactate, creatinine, bilirubin, WBC, BUN, glucose, platelets,
   AST, alkaline phosphatase, and other skewed variables.
4. **Robust quantile scaling**:
   $$
   x_c^{\text{norm}} = 2 \cdot \frac{x_c - \text{p01}_c}{\text{p99}_c - \text{p01}_c} - 1,
   $$
   where $\text{p01}_c$ and $\text{p99}_c$ are the 1st and 99th percentiles
   from the training set.
5. **Leaky clip**:
   $$
   x_c^{\text{norm}} \gets
   \begin{cases}
   1.0 + 0.1 \cdot (x_c^{\text{norm}} - 1.0), & x_c^{\text{norm}} > 1, \\
   -1.0 + 0.1 \cdot (x_c^{\text{norm}} + 1.0), & x_c^{\text{norm}} < -1, \\
   x_c^{\text{norm}}, & \text{otherwise}
   \end{cases}
   $$
   followed by $x_c^{\text{norm}} \gets \text{clamp}(x_c^{\text{norm}}, -2, 2)$.

### 7.2 CMAPSS Preprocessing

The C-MAPSS dataset \cite{saxena2008cmapss} provides 3 operational settings
and 21 sensor channels (26 columns in total). In our preprocessing, we retain
14 sensors following our feature-selection criterion (sensors with sufficient
monotonic trend for degradation modeling). The retained sensor set includes:
$s_2, s_3, s_4, s_7, s_8, s_9, s_{11}, s_{12}, s_{13}, s_{14}, s_{15},
s_{17}, s_{20}, s_{21}$.

Sensor streams are min-max normalized using training-set statistics:

$$
x^{\text{norm}}_{c} = \frac{x_c - \min_c}{\max_c - \min_c}.
$$

Remaining Useful Life (RUL) is capped piece-wise linearly:

$$
\text{RUL}(t) = \min\big(\text{max\_cycle} - \text{cycle}_t,\; \text{max\_rul}\big),
$$

with $\text{max\_rul} = 125$. Sequences are segmented into sliding windows
of length $W = 30$.

---

## 8. Experimental Conditions

| Condition | Loss Function | Key Hyperparameters |
|-----------|---------------|-------------------|
| 01\_BCE\_Baseline | $\mathcal{L}_{\text{BCE}}(z, g^{\text{hard}}, m)$ | — |
| 02\_TLS | $\mathcal{L}_{\text{TLS}}(z, y^{\text{TLS}}, m)$ | $\alpha = 6.0$ |
| 03\_BCE\_Smoothness | $\mathcal{L}_{\text{Smooth}}(z, g^{\text{hard}}, m)$ | $\lambda_s = 0.1$ |
| 04\_BCE\_TotalVariation | $\mathcal{L}_{\text{TV}}(z, g^{\text{hard}}, m)$ | $\lambda_{\text{tv}} = 0.1$ |
| 05\_BGSL\_StateOnly | $\mathcal{L}_{\text{BGSL}}(\lambda_v = 0, \lambda_a = 0)$ | state loss only |
| 06\_Full\_BGSL | $\mathcal{L}_{\text{BGSL}}(\lambda_v = 0.1, \lambda_a = 0.05)$ | full kinematic |

---

## 9. Hypothesis

The central hypothesis of BGSL:

> **Supervising the velocity and acceleration of the risk score trajectory
> (Full BGSL) yields clinically superior risk predictors compared to
> BCE, BCE + smoothness/TV penalties, Temporal Label Smoothing (level-only
> supervision), and BGSL state-only (soft targets without derivatives).**

Specific expectations:

1. **Discrimination**: We expect the main gains of Full BGSL to appear in trajectory smoothness, temporal calibration, lead time, and clinical utility, while standard discrimination metrics (AUROC, AUPRC) may remain similar across conditions.
2. **Smoother trajectories**: Full BGSL is expected to achieve lower RTV, SPJ, and ASF than BCE and Smoothness baselines.
3. **Earlier warnings**: BGSL is expected to achieve longer mean/median lead time (or equivalent lead time at higher recall) than all baselines.
4. **Higher clinical utility**: Full BGSL is expected to achieve the highest PhysioNet utility score by reducing false alarms in negative patients while maintaining timely alerts for positive patients.

---

## 10. Related Work

- **Temporal Label Smoothing** [Yeche et al., ICML 2023]: Proposes ramp-based soft targets for early event prediction. Establishes a theoretical connection between label smoothing and the survival hazard. Closest prior art and primary baseline.
- **Label Smoothing** [Muller et al., NeurIPS 2019]: Shows that mixing hard targets with a uniform distribution improves calibration. BGSL temporalizes and differentiates this.
- **Total Variation Regularization** [Rudin et al., Physica D, 1992]: L1 norm of gradient for denoising. BGSL's RTV metric adapts this to 1D risk trajectories.
- **PhysioNet/CinC Challenge 2019** [Reyna et al., Crit Care Med, 2020]: Established the sepsis early prediction benchmark with a per-window utility scoring function that rewards early true positives and penalizes false alarms and late/missed predictions.
- **Focal Loss** [Lin et al., ICCV 2017]: Down-weights well-classified examples. Available as an alternative state loss.
- **Asymmetric Loss** [Ridnik et al., ICCV 2021]: Asymmetric focusing ($\gamma_- > \gamma_+$) and probability shifting for multi-label classification. Included as an experimental state-loss variant in BGSL, subject to soft-target adaptation.

---

## References

1. Reyna MA, Josef CS, Jeter R, et al. "Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019." *Critical Care Medicine*, 48(2):210–217, 2020.
2. Yeche H, Pace A, Rätsch G, Kuznetsova R. "Temporal label smoothing for early event prediction." *ICML*, 2023.
3. Müller R, Kornblith S, Hinton GE. "When does label smoothing help?" *NeurIPS*, 2019.
4. Rudin LI, Osher S, Fatemi E. "Nonlinear total variation based noise removal algorithms." *Physica D*, 60(1–4):259–268, 1992.
5. Lin T-Y, Goyal P, Girshick R, et al. "Focal loss for dense object detection." *ICCV*, 2017.
6. Ridnik T, Ben-Baruch E, Zamir N, et al. "Asymmetric loss for multi-label classification." *ICCV*, 2021.
7. Loshchilov I, Hutter F. "Decoupled weight decay regularization." *ICLR*, 2019.
8. Chung J, Gulcehre C, Cho K, Bengio Y. "Empirical evaluation of gated recurrent neural networks on sequence modeling." *arXiv:1412.3555*, 2014.
9. Saxena A, Goebel K, Simon D, Eklund N. "Damage propagation modeling for aircraft engine run-to-failure simulation." *PHM*, 2008.
10. Davis J, Goadrich M. "The relationship between precision-recall and ROC curves." *ICML*, 2006.
11. Naeini MP, Cooper GF, Hauskrecht M. "Obtaining well calibrated probabilities using Bayesian binning." *AAAI*, 2015.
