\documentclass{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{mathtools}

\begin{document}

\section{Generic early-event objective}

\subsection{Data and model}
Let the dataset be
\[
\mathcal{D} = \{(\mathbf{x}_i, \mathbf{u}_i, y_i, \tau_i^*)\}_{i=1}^N,
\]
where $\mathbf{x}_i = (\mathbf{x}_{i,0},\dots,\mathbf{x}_{i,T_i-1})$ is a multivariate time series of length $T_i$, $\mathbf{u}_i$ is a vector of static covariates, $y_i\in\{0,1\}$ indicates whether the event occurs during the observation window, and $\tau_i^*$ is the event time when $y_i=1$.

Define the validity mask
\[
m_{i,t} = \mathbf{1}[t < T_i], \qquad T_{\max} = \max_i T_i,
\]
for $t=0,\dots,T_{\max}-1$.

A causal backbone $f_\theta$ produces a scalar logit at each time step:
\[
z_{i,t} = f_\theta(\mathbf{x}_{i,\le t}, \mathbf{u}_i), \qquad
p_{i,t} = \sigma(z_{i,t}) = \frac{1}{1+e^{-z_{i,t}}}.
\]

\subsection{Soft target generation}
Let $G:[0,T_{\max})\to[0,1]$ be a smooth, monotone non-decreasing target generator. Optionally, a small hypernetwork $h_\psi$ maps static covariates to patient-specific shape parameters
\[
\eta_i = h_\psi(\mathbf{u}_i).
\]

The soft target sequence is
\[
g_{i,t} =
\begin{cases}
G(t;\tau_i^*,H,\eta_i), & y_i=1,\\
0, & y_i=0,
\end{cases}
\]
where $H$ is the early-warning horizon.

If $G$ is differentiable, define the first and second derivatives analytically by
\[
g^{(1)}_{i,t} = \frac{\partial}{\partial t} g_{i,t}, \qquad
g^{(2)}_{i,t} = \frac{\partial^2}{\partial t^2} g_{i,t},
\]
and set $g^{(1)}_{i,t}=g^{(2)}_{i,t}=0$ for negative patients.

If $G$ is not used in closed form, the same discrete derivative operator applied to the predictions may also be applied to the target trajectory.

\subsection{Temporal importance weighting}
For positive patients, emphasize the region where the target rises by using a Gaussian weight centered near the middle of the alarm window:
\[
w_{i,t} =
\begin{cases}
1, & y_i=0,\\[4pt]
\exp\!\left(
-\dfrac{\bigl(t-(\tau_i^* - H/2)\bigr)^2}{2\sigma_w^2}
\right), & y_i=1,
\end{cases}
\]
with $\sigma_w>0$ a hyperparameter.

Define the combined per-step auxiliary weight
\[
q_{i,t} = m_{i,t} w_{i,t}.
\]

\subsection{Savitzky--Golay derivative operator}\label{sec:SG}
During training, the entire predicted sequence
\[
\mathbf{p}_i = (p_{i,0},\dots,p_{i,T_{\max}-1})
\]
is available. We apply a fixed Savitzky--Golay derivative filter of order $r\in\{1,2\}$ with odd window length $L$ and polynomial degree $d$ satisfying
\[
L \text{ odd}, \qquad 0 \le r \le d < L.
\]
With unit time spacing, define
\[
\dot{p}_{i,t} = \mathcal{S}^{(1)}_{L,d}(\mathbf{p}_i)_t,
\qquad
\ddot{p}_{i,t} = \mathcal{S}^{(2)}_{L,d}(\mathbf{p}_i)_t.
\]
Boundary values are handled by reflection padding before applying the filter, and losses are masked by the valid sequence length.

\subsection{Masked mean}
For any quantity $a_{i,t}$ and non-negative weights $q_{i,t}$,
\[
\operatorname{MaskedMean}(a_{i,t};\,q_{i,t}) =
\frac{\sum_{i=1}^{N}\sum_{t=0}^{T_{\max}-1} q_{i,t}\, a_{i,t}}
     {\max\!\left(\sum_{i=1}^{N}\sum_{t=0}^{T_{\max}-1} q_{i,t},\,1\right)}.
\]
If all weights are zero, the quantity evaluates to zero.

\subsection{Loss components}
\paragraph{State loss.}
We employ a soft-target focal binary cross-entropy loss with modulating factor $\gamma\ge 0$:
\[
\ell_{\mathrm{state}}(z,g) =
- g\,(1-\sigma(z))^\gamma \log \sigma(z)
- (1-g)\,\sigma(z)^\gamma \log(1-\sigma(z)).
\]
Setting $\gamma=0$ recovers standard binary cross-entropy.

The state loss is
\[
\mathcal{L}_{\mathrm{state}} =
\operatorname{MaskedMean}\!\Bigl(\ell_{\mathrm{state}}(z_{i,t},g_{i,t});\, m_{i,t}\Bigr).
\]

\paragraph{Velocity loss.}
\[
\mathcal{L}_{\mathrm{vel}} =
\operatorname{MaskedMean}\!\Bigl((\dot{p}_{i,t}-g^{(1)}_{i,t})^2;\, q_{i,t}\Bigr).
\]

\paragraph{Acceleration loss.}
\[
\mathcal{L}_{\mathrm{acc}} =
\operatorname{MaskedMean}\!\Bigl((\ddot{p}_{i,t}-g^{(2)}_{i,t})^2;\, q_{i,t}\Bigr).
\]

\paragraph{Monotonicity loss.}
Define the monotonicity mask for positive patients:
\[
r^{\mathrm{mono}}_{i,t} = m_{i,t}\,\mathbf{1}[y_i=1]\,
\mathbf{1}[\tau_i^*-H \le t \le \tau_i^*].
\]
The monotonicity penalty discourages negative first derivatives inside the early-warning window:
\[
\mathcal{L}_{\mathrm{mono}} =
\operatorname{MaskedMean}\!\Bigl(\bigl[\max(0,-\dot{p}_{i,t})\bigr]^2;\,
                               r^{\mathrm{mono}}_{i,t}\Bigr).
\]

\subsection{Total loss}
The full objective is
\[
\mathcal{L} =
\mathcal{L}_{\mathrm{state}}
+ \lambda_v \mathcal{L}_{\mathrm{vel}}
+ \lambda_a \mathcal{L}_{\mathrm{acc}}
+ \lambda_m \mathcal{L}_{\mathrm{mono}},
\]
with hyperparameters $\lambda_v,\lambda_a,\lambda_m \ge 0$.

\section{Sepsis early-warning instantiation}

\subsection{Horizon and event time}
For sepsis early warning, set the prediction horizon to
\[
H = 6 \text{ hours},
\]
or its aligned equivalent in discrete time steps. Here $\tau_i^*$ denotes the first time at which sepsis becomes clinically present in the record.

\subsection{Logistic ramp target}
A simple and robust choice for the positive-patient target is a logistic ramp:
\[
G_{\mathrm{sepsis}}(t;\tau_i^*,H,s) =
\sigma\!\left(\frac{t-\mu_i}{s}\right),
\qquad
\mu_i = \tau_i^* - \frac{H}{2},
\]
with a shared slope parameter $s>0$.

Thus,
\[
g_{i,t} =
\begin{cases}
\sigma\!\left(\dfrac{t-\mu_i}{s}\right), & y_i=1,\\[8pt]
0, & y_i=0.
\end{cases}
\]

\subsection{Analytic derivatives}
Because the ramp is differentiable, the target derivatives are available in closed form:
\[
g^{(1)}_{i,t} = \frac{1}{s}\,g_{i,t}\,(1-g_{i,t}),
\qquad
g^{(2)}_{i,t} = \frac{1}{s^2}\,g_{i,t}\,(1-g_{i,t})(1-2g_{i,t}),
\]
for $y_i=1$, and
\[
g^{(1)}_{i,t} = g^{(2)}_{i,t} = 0
\]
for $y_i=0$.

\subsection{Remarks}
The generic wrapper is unchanged; one simply instantiates it with the sepsis horizon $H$, the sepsis-specific target generator $G_{\mathrm{sepsis}}$, and its derivatives. The slope parameter $s$ is kept fixed here for robustness, but it could be made patient-dependent by replacing $s$ with a hypernetwork output if needed.

\end{document}