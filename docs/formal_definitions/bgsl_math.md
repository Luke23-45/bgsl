

# Biological Gradient Supervised Learning (EBGSL)

Let
[
\mathcal D={(\mathbf x_i,\mathbf u_i,y_i,\tau_i^*)}*{i=1}^N
]
be a patient cohort. For patient (i), the multivariate time series is
[
\mathbf x_i=(\mathbf x*{i,0},\mathbf x_{i,1},\dots,\mathbf x_{i,T_i-1}),\qquad \mathbf x_{i,t}\in\mathbb R^D,
]
with static covariates (\mathbf u_i\in\mathbb R^{D_{\text{static}}}). Let
[
T_{\max}=\max_i T_i,\qquad m_{i,t}=\mathbf 1[t<T_i],\qquad t=0,\dots,T_{\max}-1.
]

The label (y_i\in{0,1}) indicates whether the event is observed during the recorded interval. If (y_i=1), then (\tau_i^*\in[0,T_i)) is the observed event time. If (y_i=0), the event is not observed in the record, and (\tau_i^*) is not used anywhere in the model for that patient.

Fix a prediction horizon (H>0), measured in time steps. At each time (t), the task is to predict whether the event will occur within ([t,t+H]).

---

## 1. Backbone predictor

Let (f_\theta) be any causal sequence model with parameters (\theta). Define the logit and probability at each time as
[
s_{i,t}=f_\theta(\mathbf x_{i,\le t})\in\mathbb R,\qquad
p_{i,t}=\sigma(s_{i,t})=\frac{1}{1+e^{-s_{i,t}}}.
]
Here (p_{i,t}) is interpreted as the predicted probability that the event occurs within the horizon window ([t,t+H]).

---

## 2. Patient-specific soft target

Let a hypernetwork (h_\psi) with parameters (\psi) map the static covariates to target-shape parameters:
[
(\Delta\mu_i,\log\sigma_i,\log\kappa_i)=h_\psi(\mathbf u_i),
]
and define
[
\sigma_i=\exp(\log\sigma_i)>0,\qquad \kappa_i=\exp(\log\kappa_i)>0.
]

For event-positive patients ((y_i=1)), define
[
\mu_i=\tau_i^*-\frac{H}{2}+\Delta\mu_i,
\qquad
u_{i,t}=\frac{t-\mu_i}{\sigma_i}.
]
The soft onset target is
[
g_{i,t}=\left(1+e^{-u_{i,t}}\right)^{-\kappa_i}.
]

For event-negative patients ((y_i=0)), define
[
g_{i,t}=0,\qquad g'*{i,t}=0,\qquad g''*{i,t}=0
]
for all (t).

For (y_i=1), the analytic derivatives of the target are
[
g'_{i,t}
========

\frac{\kappa_i}{\sigma_i},e^{-u_{i,t}}\left(1+e^{-u_{i,t}}\right)^{-(\kappa_i+1)},
]
[
g''_{i,t}
=========

\frac{\kappa_i}{\sigma_i^2},e^{-u_{i,t}}\left(1+e^{-u_{i,t}}\right)^{-(\kappa_i+2)}
\left(\kappa_i e^{-u_{i,t}}-1\right).
]

These formulas are exact. The only notation change from the earlier draft is that the scaled time is written as (u_{i,t}), so it does not collide with the model logit (s_{i,t}).

---

## 3. Smoothed derivative estimates of the prediction

Choose an odd window length (L=2m+1) and a polynomial degree (d) satisfying (d<L). Assume unit time spacing:
[
\Delta=1.
]

Let (\mathcal S^{(r)}_{d,m,\Delta}) denote the Savitzky–Golay derivative operator of order (r\in{1,2}), defined by fitting a degree-(d) polynomial in each local window and evaluating its (r)-th derivative at the center. With (\Delta=1), this is the standard unit-spacing form.

Define the smoothed velocity and acceleration of the prediction sequence by
[
\dot p_{i,t}=\mathcal S^{(1)}*{d,m,1}(p_i)*t,\qquad
\ddot p*{i,t}=\mathcal S^{(2)}*{d,m,1}(p_i)_t.
]

When boundary values are needed, extend the sequence by a fixed padding rule such as reflection padding before applying the filter.

---

## 4. Temporal importance weights and monotonicity region

For event-positive patients, define the Gaussian importance weight
[
w_{i,t}
=======

\exp!\left(
-\frac{(t-(\tau_i^*-H))^2}{2\sigma_w^2}
\right),
\qquad
\sigma_w=\frac{H}{3}.
]
For event-negative patients, set
[
w_{i,t}=1.
]

To encourage nondecreasing risk inside the alarm window, define
[
r_{i,t}^{\mathrm{mono}}
=======================

m_{i,t},\mathbf 1[y_i=1],\mathbf 1[\tau_i^*-H\le t\le \tau_i^*].
]

---

## 5. Loss functions

Use the binary cross-entropy with soft target (g):
[
\ell_{\mathrm{BCE}}(p,g)=-g\log p-(1-g)\log(1-p).
]

The state-level loss is
[
\mathcal L_{\mathrm{state}}
===========================

\frac{\sum_{i=1}^N\sum_{t=0}^{T_{\max}-1} m_{i,t},\ell_{\mathrm{BCE}}(p_{i,t},g_{i,t})}
{\max!\left(\sum_{i=1}^N\sum_{t=0}^{T_{\max}-1} m_{i,t},,1\right)}.
]

The velocity and acceleration losses are weighted mean-squared errors:
[
\mathcal L_{\mathrm{vel}}
=========================

\frac{\sum_{i=1}^N\sum_{t=0}^{T_{\max}-1} m_{i,t},w_{i,t},(\dot p_{i,t}-g'*{i,t})^2}
{\max!\left(\sum*{i=1}^N\sum_{t=0}^{T_{\max}-1} m_{i,t},w_{i,t},,1\right)},
]
[
\mathcal L_{\mathrm{acc}}
=========================

\frac{\sum_{i=1}^N\sum_{t=0}^{T_{\max}-1} m_{i,t},w_{i,t},(\ddot p_{i,t}-g''*{i,t})^2}
{\max!\left(\sum*{i=1}^N\sum_{t=0}^{T_{\max}-1} m_{i,t},w_{i,t},,1\right)}.
]

The monotonicity penalty is
[
\mathcal L_{\mathrm{mono}}
==========================

\frac{\lambda_{\mathrm{mono}}}{\max!\left(\sum_{i=1}^N\sum_{t=0}^{T_{\max}-1} r_{i,t}^{\mathrm{mono}},,1\right)}
\sum_{i=1}^N\sum_{t=0}^{T_{\max}-1}
r_{i,t}^{\mathrm{mono}},[\max(0,-\dot p_{i,t})]^2.
]

The full EBGSL objective is
[
\mathcal L_{\mathrm{EBGSL}}
===========================

\mathcal L_{\mathrm{state}}
+
\lambda_v\mathcal L_{\mathrm{vel}}
+
\lambda_a\mathcal L_{\mathrm{acc}}
+
\mathcal L_{\mathrm{mono}}.
]

Training minimizes (\mathcal L_{\mathrm{EBGSL}}) jointly over the backbone parameters (\theta) and the hypernetwork parameters (\psi).

---

## 6. Hyperparameters

The fixed design hyperparameters are
[
H,\quad d,\quad m,\quad \lambda_v,\quad \lambda_a,\quad \lambda_{\mathrm{mono}},\quad \sigma_w.
]
A common default choice is
[
d=2,\qquad m=3,\qquad \lambda_v=0.10,\qquad \lambda_a=0.05,\qquad \lambda_{\mathrm{mono}}=0.01,\qquad \sigma_w=\frac{H}{3}.
]

The patient-specific parameters (\Delta\mu_i), (\sigma_i), and (\kappa_i) are produced by the hypernetwork.

---

## 7. Reduction to a simpler BGSL baseline

A simpler BGSL-style baseline is recovered by removing the hypernetwork, fixing the target-shape parameters, replacing the Savitzky–Golay derivatives with finite-difference estimates, setting (w_{i,t}=1), and taking
[
\lambda_{\mathrm{mono}}=0.
]

---


