# Formal Definition of the Kalman Observer for Early Event Prediction

## 1. Notation

Let $\mathbb{N}_0 = \{0, 1, 2, \dots\}$ be the set of natural numbers
including zero, and $\mathbb{R}$ the set of real numbers.

| Symbol | Domain | Description |
|--------|--------|-------------|
| $t$ | $\mathbb{N}_0$ | Discrete time index |
| $T$ | $\mathbb{N}$ | Sequence length |
| $m$ | $\mathbb{N}$ | Observation dimension |
| $d$ | $\mathbb{N}$ | Latent state dimension |
| $h$ | $\mathbb{N}$ | Hidden dimension of the risk MLP |
| $H$ | $\mathbb{N}$ | Prediction horizon |
| $B$ | $\mathbb{N}$ | Batch size (number of sequences) |
| $P$ | $\mathbb{N}$ | Number of particles (particle filter) |
| $\mathbf{z}_t$ | $\mathbb{R}^d$ | True latent physiological state at time $t$ |
| $\mathbf{x}_t$ | $\mathbb{R}^m$ | Observed vital signs at time $t$ |
| $\hat{\mathbf{z}}_{t \mid s}$ | $\mathbb{R}^d$ | Estimate of $\mathbf{z}_t$ given observations up to time $s$ |
| $\tilde{\mathbf{y}}_t$ | $\mathbb{R}^m$ | Innovation (prediction error) at time $t$ |
| $\mathbf{e}_t$ | $\mathbb{R}^d$ | Estimation error: $\mathbf{e}_t = \mathbf{z}_t - \hat{\mathbf{z}}_{t \mid t}$ |
| $\mathbf{P}_{t \mid s}$ | $\mathbb{R}^{d \times d}$ | Error covariance of $\hat{\mathbf{z}}_{t \mid s}$ |
| $r_t$ | $(0, 1)$ | Predicted event risk at time $t$ |
| $g_t$ | $[0, 1]$ | Soft target label at time $t$ |
| $M_t$ | $\{0, 1\}$ | Valid mask: $1$ if a prediction is made at $t$, $0$ otherwise |
| $O_{t,j}$ | $\{0, 1\}$ | Observation mask: $1$ if feature $j$ is observed at $t$, $0$ if imputed |
| $\mathbf{O}_t$ | $\{0, 1\}^m$ | Observation mask vector: $[\mathbf{O}_t]_j = O_{t,j}$ |
| $\mathbf{A}$ | $\mathbb{R}^{d \times d}$ | State transition matrix |
| $\mathbf{C}$ | $\mathbb{R}^{m \times d}$ | Observation (emission) matrix |
| $\mathbf{K}$ | $\mathbb{R}^{d \times m}$ | Learned constant observer gain |
| $\mathbf{K}_t$ | $\mathbb{R}^{d \times m}$ | Optimal Kalman gain at time $t$ (Riccati-derived) |
| $\mathbf{Q}$ | $\mathbb{R}^{d \times d}$ | Process noise covariance (positive semi-definite) |
| $\mathbf{R}$ | $\mathbb{R}^{m \times m}$ | Observation noise covariance (positive definite) |
| $\mathbf{I}_d$ | $\mathbb{R}^{d \times d}$ | $d \times d$ identity matrix |
| $\mathbf{w}_t$ | $\mathbb{R}^d$ | Process noise vector |
| $\mathbf{v}_t$ | $\mathbb{R}^m$ | Observation noise vector |
| $\sigma(\cdot)$ | $\mathbb{R} \to (0, 1)$ | Logistic sigmoid: $\sigma(a) = (1 + e^{-a})^{-1}$ |
| $\rho(\cdot)$ | $\mathbb{C}^{n \times n} \to \mathbb{R}_{\ge 0}$ | Spectral radius: $\rho(\mathbf{M}) = \max_i \lvert \lambda_i \rvert$ |
| $\tau^*$ | $\mathbb{N} \cup \{\infty\}$ | Event time ($\infty$ for negative patients) |
| $\lambda$ | $\mathbb{R}_+$ | Reconstruction loss weight |
| $\odot$ | — | Element-wise (Hadamard) product |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | — | Multivariate Gaussian with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ |
| $\mathbb{E}[\cdot]$ | — | Expectation operator |
| $\mathbf{W}_1$ | $\mathbb{R}^{d \times h}$ | Input weight matrix of risk MLP (eq 22) |
| $\mathbf{b}_1$ | $\mathbb{R}^{h}$ | Hidden bias vector of risk MLP (eq 22) |
| $\mathbf{w}_2$ | $\mathbb{R}^{h}$ | Output weight vector of risk MLP (eq 23) |
| $b_2$ | $\mathbb{R}$ | Output bias scalar of risk MLP (eq 23) |
| $s_t$ | $\mathbb{R}$ | Scalar logit before sigmoid (eq 23) |
| $\boldsymbol{\theta}$ | Set | MLP parameters $\{\mathbf{W}_1, \mathbf{b}_1, \mathbf{w}_2, b_2\}$ (eq 25) |
| $\boldsymbol{\Theta}$ | Set | All learnable parameters (eq 34) |
| $\eta$ | $\mathbb{R}_+$ | Learning rate (eq 36) |
| $f$ | $\mathbb{R}^d \to \mathbb{R}^d$ | Nonlinear state transition function (eq 37) |
| $h$ | $\mathbb{R}^d \to \mathbb{R}^m$ | Nonlinear observation function (eq 38) |
| $\tilde{w}_t^{(p)}$ | $\mathbb{R}_+$ | Unnormalized particle importance weight (eq 42) |
| $W$ | $\mathbb{N}$ | Sliding window length for autocorrelation (eq 45) |
| $\rho_{\text{target}}(t)$ | $[0, 1]$ | Target autocorrelation for CSD regularization (eq 47) |
| $\lambda_1, \lambda_2$ | $\mathbb{R}_+$ | CSD regularization weights (eq 48) |

---

## 2. Generative State-Space Model

The patient's true physiological state evolves according to a
linear-Gaussian dynamical system.

**State (transition) equation:**

$$
\mathbf{z}_t = \mathbf{A} \, \mathbf{z}_{t-1} + \mathbf{w}_t,
\qquad \mathbf{w}_t \overset{\text{i.i.d.}}{\sim}
\mathcal{N}(\mathbf{0}, \mathbf{Q})
\tag{1}
$$

**Observation (emission) equation:**

$$
\mathbf{x}_t = \mathbf{C} \, \mathbf{z}_t + \mathbf{v}_t,
\qquad \mathbf{v}_t \overset{\text{i.i.d.}}{\sim}
\mathcal{N}(\mathbf{0}, \mathbf{R})
\tag{2}
$$

**Noise assumptions:**

$$
\mathbb{E}[\mathbf{w}_t \mathbf{w}_s^\mathsf{T}] = \mathbf{0},
\quad t \neq s, \qquad
\mathbb{E}[\mathbf{v}_t \mathbf{v}_s^\mathsf{T}] = \mathbf{0},
\quad t \neq s, \qquad
\mathbb{E}[\mathbf{w}_t \mathbf{v}_s^\mathsf{T}] = \mathbf{0},
\quad \forall t, s.
\tag{3}
$$

**Initial state:**

$$
\mathbf{z}_0 = \mathbf{0},
\qquad
\mathbf{P}_0 = \mathbf{I}_d.
\tag{4}
$$

The generative model is specified by the tuple
$(\mathbf{A}, \mathbf{C}, \mathbf{Q}, \mathbf{R})$ and the initial
condition (4). The latent state $\mathbf{z}_t$ is never observed
directly; all inference is conditioned on the observation sequence
$\mathbf{x}_1, \dots, \mathbf{x}_T$. In the learned observer variant
(§4), the gain $\mathbf{K}$ is learned directly in place of the
Riccati computation that would require $\mathbf{Q}$ and $\mathbf{R}$.

---

## 3. Observer Equations (Kalman Filter)

The Kalman filter computes the minimum mean-squared error (MMSE)
estimate of $\mathbf{z}_t$ given the observations up to time $t$,
under the Gaussian noise assumptions of §2. The recursion alternates
between a prediction (time update) and a correction (measurement
update) at each time step.

### 3.1 Prediction Step

Before observing $\mathbf{x}_t$, the state is projected forward:

$$
\hat{\mathbf{z}}_{t \mid t-1} =
\mathbf{A} \, \hat{\mathbf{z}}_{t-1 \mid t-1}
\tag{5}
$$

The prediction error covariance is:

$$
\mathbf{P}_{t \mid t-1} =
\mathbf{A} \, \mathbf{P}_{t-1 \mid t-1} \, \mathbf{A}^\mathsf{T}
+ \mathbf{Q}
\tag{6}
$$

### 3.2 Innovation

After observing $\mathbf{x}_t$, the prediction error (innovation) is:

$$
\tilde{\mathbf{y}}_t =
\mathbf{x}_t - \mathbf{C} \, \hat{\mathbf{z}}_{t \mid t-1}
\tag{7}
$$

### 3.3 Kalman Gain

The optimal gain minimizes the posterior error covariance
$\mathbf{P}_{t \mid t}$:

$$
\mathbf{K}_t =
\mathbf{P}_{t \mid t-1} \, \mathbf{C}^\mathsf{T}
\left(
\mathbf{C} \, \mathbf{P}_{t \mid t-1} \, \mathbf{C}^\mathsf{T}
+ \mathbf{R}
\right)^{-1}
\tag{8}
$$

### 3.4 Correction Step

The predicted state is corrected by the innovation scaled by the gain:

$$
\hat{\mathbf{z}}_{t \mid t} =
\hat{\mathbf{z}}_{t \mid t-1} + \mathbf{K}_t \, \tilde{\mathbf{y}}_t
\tag{9}
$$

The corrected error covariance is:

$$
\mathbf{P}_{t \mid t} =
\left( \mathbf{I}_d - \mathbf{K}_t \, \mathbf{C} \right)
\mathbf{P}_{t \mid t-1}
\tag{10}
$$

Equations (5)–(10) define the discrete-time Kalman filter. When the
system is linear and the noise statistics $(\mathbf{Q}, \mathbf{R})$
are known, $\mathbf{K}_t$ from (8) is the Bayes-optimal gain.

---

## 4. Learned Constant-Gain Observer

In the learned variant, the Riccati-derived gain $\mathbf{K}_t$ is
replaced by a constant matrix $\mathbf{K} \in \mathbb{R}^{d \times m}$
that is learned directly by gradient descent. This removes the
dependence on $(\mathbf{Q}, \mathbf{R})$ and the per-step matrix
inversion in (8).

**Predict:**

$$
\hat{\mathbf{z}}_{t \mid t-1} =
\mathbf{A} \, \hat{\mathbf{z}}_{t-1 \mid t-1}
\tag{11}
$$

**Innovate:**

$$
\tilde{\mathbf{y}}_t =
\mathbf{x}_t - \mathbf{C} \, \hat{\mathbf{z}}_{t \mid t-1}
\tag{12}
$$

**Correct:**

$$
\hat{\mathbf{z}}_{t \mid t} =
\hat{\mathbf{z}}_{t \mid t-1} + \mathbf{K} \, \tilde{\mathbf{y}}_t
\tag{13}
$$

**Single-step closed form** (obtained by substituting (11) and (12)
into (13)):

$$
\hat{\mathbf{z}}_{t \mid t} =
(\mathbf{I}_d - \mathbf{K} \mathbf{C}) \mathbf{A} \,
\hat{\mathbf{z}}_{t-1 \mid t-1}
+ \mathbf{K} \, \mathbf{x}_t
\tag{14}
$$

The initial condition is $\hat{\mathbf{z}}_{0 \mid 0} = \mathbf{0}$.

---

## 5. Estimation Error Dynamics

Define the estimation error at the corrected step:

$$
\mathbf{e}_t = \mathbf{z}_t - \hat{\mathbf{z}}_{t \mid t}
\tag{15}
$$

Substituting the true dynamics (1) and the observer (14):

$$
\begin{aligned}
\mathbf{e}_t &=
\mathbf{A} \mathbf{z}_{t-1} + \mathbf{w}_t
- (\mathbf{I}_d - \mathbf{K} \mathbf{C}) \mathbf{A}
\hat{\mathbf{z}}_{t-1 \mid t-1}
- \mathbf{K} \mathbf{x}_t \\[4pt]
&= \mathbf{A} \mathbf{z}_{t-1} + \mathbf{w}_t
- (\mathbf{I}_d - \mathbf{K} \mathbf{C}) \mathbf{A}
\hat{\mathbf{z}}_{t-1 \mid t-1}
- \mathbf{K} (\mathbf{C} \mathbf{z}_t + \mathbf{v}_t) \\[4pt]
&= \mathbf{A} \mathbf{z}_{t-1} + \mathbf{w}_t
- \mathbf{A} \hat{\mathbf{z}}_{t-1 \mid t-1}
+ \mathbf{K} \mathbf{C} \mathbf{A} \hat{\mathbf{z}}_{t-1 \mid t-1}
- \mathbf{K} \mathbf{C} \mathbf{A} \mathbf{z}_{t-1}
- \mathbf{K} \mathbf{C} \mathbf{w}_t
- \mathbf{K} \mathbf{v}_t \\[4pt]
&= (\mathbf{I}_d - \mathbf{K} \mathbf{C})
\mathbf{A} \, \mathbf{e}_{t-1}
+ (\mathbf{I}_d - \mathbf{K} \mathbf{C}) \mathbf{w}_t
- \mathbf{K} \mathbf{v}_t
\end{aligned}
\tag{16}
$$

Equation (16) is a first-order vector autoregressive process driven
by the noise terms. In the absence of noise
($\mathbf{w}_t = \mathbf{v}_t = \mathbf{0}$), the error evolves as:

$$
\mathbf{e}_t = (\mathbf{I}_d - \mathbf{K} \mathbf{C})
\mathbf{A} \, \mathbf{e}_{t-1}
\tag{17}
$$

**Stability condition.** The estimation error converges to zero
exponentially if and only if the spectral radius of the error
transition matrix is strictly less than unity:

$$
\rho\!\left( (\mathbf{I}_d - \mathbf{K} \mathbf{C}) \mathbf{A}
\right) < 1
\tag{18}
$$

When (18) holds, the observer is asymptotically stable and the
effect of initial condition $\mathbf{e}_0$ decays geometrically.

---

## 6. Observation Mask Integration

In practice, not all features are observed at every time step. Let
$\mathbf{O}_t \in \{0, 1\}^m$ be the observation mask at time $t$,
where $O_{t,j} = 1$ if feature $j$ was measured and $0$ if it was
imputed. The masked innovation is:

$$
\tilde{\mathbf{y}}_t =
\mathbf{O}_t \odot \big( \mathbf{x}_t -
\mathbf{C} \, \hat{\mathbf{z}}_{t \mid t-1} \big)
\tag{19}
$$

Equivalently, component-wise:

$$
\tilde{y}_{t,j} = O_{t,j} \,
\big( x_{t,j} - [\mathbf{C} \hat{\mathbf{z}}_{t \mid t-1}]_j \big),
\qquad j = 1, \dots, m
\tag{20}
$$

The correction step then uses the masked innovation:

$$
\hat{\mathbf{z}}_{t \mid t} =
\hat{\mathbf{z}}_{t \mid t-1} + \mathbf{K} \, \tilde{\mathbf{y}}_t
\tag{21}
$$

When $O_{t,j} = 0$, the $j$-th feature contributes no correction,
and the observer relies solely on the dynamics model (11) for that
component of the state.

---

## 7. Risk Prediction

From the corrected latent state estimate, the event risk at time $t$
is computed by a two-layer multilayer perceptron (MLP) with a
logistic sigmoid output.

**Hidden layer:**

$$
\mathbf{h}_t = \tanh\!\big(
\mathbf{W}_1^\mathsf{T} \hat{\mathbf{z}}_{t \mid t} + \mathbf{b}_1
\big),
\qquad
\mathbf{W}_1 \in \mathbb{R}^{d \times h},
\; \mathbf{b}_1 \in \mathbb{R}^h
\tag{22}
$$

**Scalar logit:**

$$
s_t = \mathbf{w}_2^\mathsf{T} \mathbf{h}_t + b_2,
\qquad
\mathbf{w}_2 \in \mathbb{R}^{h},
\; b_2 \in \mathbb{R}
\tag{23}
$$

**Risk probability:**

$$
r_t = \sigma(s_t) = \frac{1}{1 + e^{-s_t}}
\tag{24}
$$

The risk $r_t \in (0, 1)$ is interpreted as the probability that
the event will occur within the prediction horizon $H$ from time
$t$.

Let $\boldsymbol{\theta} = \{\mathbf{W}_1, \mathbf{b}_1,
\mathbf{w}_2, b_2\}$ denote the MLP parameters. The full risk
function can be written compactly as:

$$
r_t = \sigma\!\big(
\text{MLP}(\hat{\mathbf{z}}_{t \mid t}; \boldsymbol{\theta})
\big)
\tag{25}
$$

---

## 8. Soft Target Definition

The target label $g_t$ is a smoothed ramp that provides a graded
supervisory signal in the alarm window $[\tau^* - H, \tau^*)$.

### 8.1 Positive Patients ($\tau^* < \infty$)

For $t < \tau^*$ (pre-event window):

$$
g_t = \text{clip}\!\left(
\frac{t - (\tau^* - H)}{H}, \; 0, \; 1
\right)
\tag{26}
$$

where $\text{clip}(a, 0, 1) = \max(0, \min(1, a))$.

For $t \ge \tau^*$ (during and after the event):

$$
g_t = 0
\qquad
M_t = 0 \quad \text{(loss excluded)}
\tag{27}
$$

### 8.2 Negative Patients ($\tau^* = \infty$)

For all $t$:

$$
g_t = 0,
\qquad
M_t = 1
\tag{28}
$$

### 8.3 Valid Mask

The valid mask $M_t \in \{0, 1\}$ controls which time steps
contribute to the classification loss:

$$
M_t =
\begin{cases}
1, & t < \tau^* \;\text{or}\; \tau^* = \infty, \\[2pt]
0, & t \ge \tau^* \;\text{and}\; \tau^* < \infty.
\end{cases}
\tag{29}
$$

---

## 9. Learning Objective

The model is trained end-to-end by minimizing a weighted sum of a
classification loss and a reconstruction loss.

### 9.1 Classification Loss (Masked Binary Cross-Entropy)

$$
\mathcal{L}_{\text{cls}} =
-\frac{1}{\sum_{t=0}^{T-1} M_t}
\sum_{t=0}^{T-1} M_t \,
\Big[ g_t \log r_t + (1 - g_t) \log(1 - r_t) \Big]
\tag{30}
$$

### 9.2 Reconstruction Loss (Masked Mean Squared Error)

The predicted observation at time $t$ is the projection of the
corrected latent state:

$$
\hat{\mathbf{x}}_t = \mathbf{C} \, \hat{\mathbf{z}}_{t \mid t}
\tag{31}
$$

The reconstruction loss is computed only on observed features:

$$
\mathcal{L}_{\text{recon}} =
\frac{1}{\sum_{t=0}^{T-1} \sum_{j=1}^{m} O_{t,j}}
\sum_{t=0}^{T-1} \sum_{j=1}^{m} O_{t,j} \,
\big( x_{t,j} - \hat{x}_{t,j} \big)^2
\tag{32}
$$

### 9.3 Combined Objective

$$
\mathcal{L} = \mathcal{L}_{\text{cls}}
+ \lambda \, \mathcal{L}_{\text{recon}}
\tag{33}
$$

where $\lambda \in \mathbb{R}_+$ controls the strength of the
self-supervised reconstruction regularizer.

---

## 10. Parameter Learning

### 10.1 Parameter Set

All parameters are learned jointly:

$$
\boldsymbol{\Theta} =
\{\, \mathbf{A},\; \mathbf{C},\; \mathbf{K},\;
\mathbf{W}_1,\; \mathbf{b}_1,\; \mathbf{w}_2,\; b_2 \,\}
\tag{34}
$$

### 10.2 Objective

The optimal parameters minimize the expected loss under the
data distribution $\mathcal{D}$:

$$
\boldsymbol{\Theta}^* =
\arg\min_{\boldsymbol{\Theta}}
\; \mathbb{E}_{(\mathbf{x}, \mathbf{g}) \sim \mathcal{D}}
\big[ \mathcal{L}(\mathbf{x}, \mathbf{g}; \boldsymbol{\Theta}) \big]
\tag{35}
$$

### 10.3 Optimization Algorithm

The expectation is approximated by mini-batch stochastic gradient
descent. At each iteration $k$, for a batch of $B$ sequences:

$$
\boldsymbol{\Theta}^{(k+1)} =
\boldsymbol{\Theta}^{(k)} - \eta \,
\nabla_{\boldsymbol{\Theta}} \,
\frac{1}{B} \sum_{i=1}^{B}
\mathcal{L}\big( \mathbf{x}^{(i)}, \mathbf{g}^{(i)};
\boldsymbol{\Theta}^{(k)} \big)
\tag{36}
$$

where $\eta$ is the learning rate. The AdamW optimizer is used,
which decouples weight decay from the adaptive gradient updates.

**Backpropagation through time.** The observer recurrence
(equations (11)–(13)) is unrolled for $T$ steps. Gradients are
computed by automatic differentiation through the full unrolled
computation graph.

---

## 11. Extension: Nonlinear Dynamics (Extended Kalman Filter)

When the state transition or observation model is nonlinear, the
linear observer is generalized to:

$$
\mathbf{z}_t = f(\mathbf{z}_{t-1}) + \mathbf{w}_t
\tag{37}
$$

$$
\mathbf{x}_t = h(\mathbf{z}_t) + \mathbf{v}_t
\tag{38}
$$

where $f: \mathbb{R}^d \to \mathbb{R}^d$ and
$h: \mathbb{R}^d \to \mathbb{R}^m$ are differentiable functions.

The Extended Kalman Filter (EKF) linearizes these functions at
each time step using Jacobian matrices:

$$
\mathbf{A}_t =
\left. \frac{\partial f}{\partial \mathbf{z}} \right|
_{\hat{\mathbf{z}}_{t-1 \mid t-1}}
\in \mathbb{R}^{d \times d}
\tag{39}
$$

$$
\mathbf{C}_t =
\left. \frac{\partial h}{\partial \mathbf{z}} \right|
_{\hat{\mathbf{z}}_{t \mid t-1}}
\in \mathbb{R}^{m \times d}
\tag{40}
$$

Equations (5)–(10) then apply with $\mathbf{A}_t$ and
$\mathbf{C}_t$ in place of the constant $\mathbf{A}$ and
$\mathbf{C}$. The learned gain $\mathbf{K}$ can be retained,
or replaced by a state-dependent gain network
$\mathbf{K}(\hat{\mathbf{z}}_{t \mid t-1})$.

---

## 12. Extension: Particle Filter

For non-Gaussian or multi-modal state distributions, a sequential
Monte Carlo (particle filter) replaces the analytic observer.

Let $\{ \mathbf{z}_t^{(p)}, w_t^{(p)} \}_{p=1}^{P}$
denote an ensemble of $P$ particles with normalized importance
weights $\sum_{p=1}^{P} w_t^{(p)} = 1$.

**Propagation:**

$$
\mathbf{z}_t^{(p)} \sim
p(\mathbf{z}_t \mid \mathbf{z}_{t-1}^{(p)})
= \mathcal{N}\!\big(
f(\mathbf{z}_{t-1}^{(p)}),\; \mathbf{Q}
\big)
\tag{41}
$$

**Weighting:**

$$
\tilde{w}_t^{(p)} = w_{t-1}^{(p)} \,
p(\mathbf{x}_t \mid \mathbf{z}_t^{(p)})
= w_{t-1}^{(p)} \,
\mathcal{N}\!\big(
\mathbf{x}_t; \; h(\mathbf{z}_t^{(p)}), \; \mathbf{R}
\big)
\tag{42}
$$

$$
w_t^{(p)} = \frac{\tilde{w}_t^{(p)}}
{\sum_{q=1}^{P} \tilde{w}_t^{(q)}}
\tag{43}
$$

**Resampling:** Draw $P$ particles with replacement from
$\{ \mathbf{z}_t^{(p)} \}$ with probabilities $\{ w_t^{(p)} \}$,
then reset $w_t^{(p)} = 1/P$.

**Risk prediction:**

$$
r_t =
\sum_{p=1}^{P} w_t^{(p)} \,
\sigma\!\big(
\text{MLP}(\mathbf{z}_t^{(p)}; \boldsymbol{\theta})
\big)
\tag{44}
$$

---

## 13. Extension: Critical Slowing Down Regularization

Critical slowing down (CSD) is a universal early-warning signal from
dynamical systems theory, characterized by increased autocorrelation
and variance before a critical transition. The observer's latent
state can be regularized to exhibit CSD properties before events.

Define the lag-1 autocorrelation of the $k$-th estimated latent
dimension over a sliding window of length $W$:

$$
\rho_k(t) =
\frac{
\sum_{\tau = t-W+1}^{t-1}
\big( \hat{z}_{\tau \mid \tau, k} - \hat{\bar{z}}_{k} \big)
\big( \hat{z}_{\tau+1 \mid \tau+1, k} - \hat{\bar{z}}_{k} \big)
}{
\sum_{\tau = t-W+1}^{t}
\big( \hat{z}_{\tau \mid \tau, k} - \hat{\bar{z}}_{k} \big)^2
},
\qquad
\hat{\bar{z}}_{k} = \frac{1}{W} \sum_{\tau = t-W+1}^{t}
\hat{z}_{\tau \mid \tau, k}
\tag{45}
$$

The average autocorrelation across dimensions is:

$$
\bar{\rho}(t) = \frac{1}{d} \sum_{k=1}^{d} \rho_k(t)
\tag{46}
$$

The CSD regularization loss penalizes deviation from a target
autocorrelation $\rho_{\text{target}}(t)$ that is high (near 1)
in the pre-event window:

$$
\mathcal{L}_{\text{CSD}} =
\big\| \bar{\rho}(t) - \rho_{\text{target}}(t) \big\|^2
\tag{47}
$$

**Augmented objective:**

$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{cls}}
+ \lambda_1 \mathcal{L}_{\text{recon}}
+ \lambda_2 \mathcal{L}_{\text{CSD}}
\tag{48}
$$

where $\lambda_1, \lambda_2 \in \mathbb{R}_+$ control the
regularization strength.