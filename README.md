# Optimization Algorithms: From Gradient Descent → Adam  

---

## Evolution Flow of First-Order Optimizers

Gradient Descent
↓
Stochastic Gradient Descent
↓
Mini-Batch Gradient Descent
↓
Momentum (Polyak Heavy Ball)
↓
Nesterov Accelerated Gradient
↓
AdaGrad
↓
AdaDelta
↓
RMSProp
↓
Adam



---

# Optimization Objective

We solve the empirical risk minimization problem:

$$
\min_{\theta \in \mathbb{R}^d} J(\theta)
$$

Where:

- $\theta$ = parameter vector  
- $J(\theta)$ = loss function  
- $g_t = \nabla J(\theta_t)$ = gradient at step $t$  

---

# Curvature, Hessian and Conditioning

Second-order structure of the loss is captured by the **Hessian matrix**:

$$
H = \nabla^2 J(\theta)
$$

Eigenvalues of $H$ determine curvature along different directions.

Condition number:

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

Large $\kappa$ ⇒ slow convergence and oscillatory dynamics.

---

# Momentum (Polyak Heavy Ball)

## Update Rule

$$
v_t = \beta v_{t-1} + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

---

## Intuition

Unrolling velocity:

$$
v_t = \sum_{k=0}^{t} \beta^{t-k} g_k
$$

Momentum performs **exponential smoothing of gradients**.

- consistent gradient directions accumulate  
- oscillating directions cancel  

Improves convergence rate:

$$
O(\kappa) \rightarrow O(\sqrt{\kappa})
$$

---

## Complexity

- Training compute per step: $O(d)$  
- Space complexity: $O(2d)$  
- Prediction compute: model dependent  
- Inference latency: model dependent  

---

## Failure Modes

- overshooting minima  
- sensitivity to $(\eta, \beta)$  
- no adaptive scaling  

---

# Nesterov Accelerated Gradient (NAG)

## Look-Ahead Gradient

$$
\tilde{\theta}_t = \theta_t - \eta \beta v_{t-1}
$$

$$
g_t = \nabla J(\tilde{\theta}_t)
$$

$$
v_t = \beta v_{t-1} + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

---

## Theory Insight

Provides optimal convergence rate for smooth convex objectives:

$$
J(\theta_t) - J^\* = O\left(\frac{1}{t^2}\right)
$$

---

## Failure

- unstable with noisy gradients  
- still uses global learning rate  

---

# AdaGrad

## Update Rule

$$
r_t = r_{t-1} + g_t \odot g_t
$$

$$
\theta_{t+1} =
\theta_t -
\frac{\eta}{\sqrt{r_t + \epsilon}} g_t
$$

---

## Interpretation

Per-coordinate adaptive scaling:

$$
\theta_{t+1} =
\theta_t -
\eta D_t^{-1/2} g_t
$$

Where $D_t = \text{diag}(r_t)$.

Rare features maintain larger step sizes.

---

## Failure

Monotonic accumulation:

$$
r_t \uparrow \Rightarrow \eta_{\text{eff}} \rightarrow 0
$$

Training stagnates.

---

# AdaDelta

## Update Rule

$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2
$$

$$
\Delta \theta_t =
-
\frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}
{\sqrt{E[g^2]_t + \epsilon}}
g_t
$$

---

## Key Insight

Effective learning rate becomes:

$$
\eta_{\text{eff}} =
\frac{\text{RMS(previous updates)}}
{\text{RMS(current gradients)}}
$$

Thus LR becomes **self-normalized**.

---

## Failure

- slower convergence  
- additional memory overhead  

---

# RMSProp

## Update Rule

$$
s_t = \beta s_{t-1} + (1-\beta) g_t^2
$$

$$
\theta_{t+1} =
\theta_t -
\frac{\eta}{\sqrt{s_t + \epsilon}} g_t
$$

---

## Interpretation

Finite-memory estimator of gradient variance.

Acts like **stochastic diagonal Newton step**.

---

## Failure

- noisy direction updates  
- lacks temporal gradient smoothing  

---

# Adam (Adaptive Moment Estimation)

## First Moment Estimate

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

## Second Moment Estimate

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

## Bias Correction

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

## Final Update

$$
\theta_{t+1} =
\theta_t -
\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}
\hat{m}_t
$$

---

## Deep Interpretation

Adam performs **momentum update in adaptively rescaled parameter space**:

$$
\theta_{t+1} =
\theta_t -
\eta D_t^{-1} \hat{m}_t
$$

Where:

$$
D_t = \text{diag}(\sqrt{\hat{v}_t})
$$

- $m_t$ → direction estimator  
- $v_t$ → curvature / noise scale estimator  

---

## Failure Modes

- convergence to sharp minima  
- theoretical divergence examples  
- higher memory usage  

---

# Benchmark Settings

| Setting | Model | Purpose |
|--------|------|---------|
| Synthetic Quadratic | parameter vector | trajectory visualization |
| California Housing | shallow MLP | regression convergence |
| MNIST | neural network | deep non-convex training |

---

# Massive Optimizer Comparison

| Optimizer | TC/Step | Space | Adaptive | Momentum | Convergence Speed | Generalization | Noise Stability | Failure Risk |
|-----------|--------|------|----------|----------|------------------|---------------|---------------|-------------|
| GD | $O(d)$ | $O(d)$ | ❌ | ❌ | Slow | Good | Low | Gets stuck |
| Momentum | $O(d)$ | $O(2d)$ | ❌ | ✅ | Faster | Very Good | Medium | Overshoot |
| NAG | $O(d)$ | $O(2d)$ | ❌ | ✅ | Very Fast | Good | Medium | Sensitive |
| AdaGrad | $O(d)$ | $O(2d)$ | ✅ | ❌ | Fast early | Medium | High | LR collapse |
| AdaDelta | $O(d)$ | $O(3d)$ | ✅ | ❌ | Medium | Medium | Medium | Slow |
| RMSProp | $O(d)$ | $O(2d)$ | ✅ | ❌ | Fast | Medium | Good | Unstable |
| Adam | $O(d)$ | $O(3d)$ | ✅ | ✅ | Very Fast | Sometimes worse | High | Sharp minima |

---

# Core Insight

Modern optimizers combine:

- gradient smoothing  
- variance normalization  
- diagonal curvature preconditioning  
- stochastic gradient estimation  

---


