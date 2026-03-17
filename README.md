# Optimization Algorithms in Machine Learning :

## From Gradient Descent → Adam : 

---

## Evolution Flow

Gradient Descent  
↓  
Stochastic Gradient Descent  
↓  
Mini-Batch Gradient Descent  
↓  
Momentum  
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

We want to solve :

$$
\min_{\theta \in \mathbb{R}^d} J(\theta)
$$

Where:

- $\theta$ → parameter vector  
- $J(\theta)$ → loss  
- $g_t = \nabla J(\theta_t)$ → gradient at step $t$  

---

# Curvature and Hessian : 

Second derivative matrix:

$$
H = \nabla^2 J(\theta)
$$

Eigenvalues $\lambda_i$ measure curvature.

Condition number:

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

Large $\kappa$ ⇒ slow convergence.

---

# Batch Gradient Descent : 

## Update

$$
\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)
$$

Gradient computed on full dataset.

---

## Mathematical Behaviour : 

If loss is quadratic:

$$
J(\theta)=\frac{1}{2}\theta^T H \theta
$$

Then:

$$
\theta_{t+1}=(I-\eta H)\theta_t
$$

Convergence speed depends on spectral radius:

$$
\rho = \max_i |1-\eta \lambda_i|
$$

---

## Complexity : 

- Compute per step: $O(nd)$  
- Space: $O(d)$  

---

## Failure : 

- extremely slow for large $n$  
- zig-zag in ill-conditioned valleys  

---

# Stochastic Gradient Descent : 

## Update

$$
\theta_{t+1} = \theta_t - \eta \nabla L_i(\theta_t)
$$

Single sample gradient.

---

## Mathematical Effect : 

Gradient estimator:

$$
\mathbb{E}[g_t] = \nabla J(\theta_t)
$$

Variance:

$$
\mathrm{Var}(g_t) > 0
$$

Noise helps escape saddle points.

---

## Complexity : 

- Per update: $O(d)$  
- Updates per epoch: $n$  

---

## Failure : 

- oscillatory convergence  
- needs LR decay  

---

# Mini-Batch Gradient Descent : 

## Update

$$
g_t = \frac{1}{b} \sum_{i=1}^{b} \nabla L_i(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

---

## Math Insight : 

Variance reduces:

$$
\mathrm{Var}(g_t) = \frac{\sigma^2}{b}
$$

Tradeoff:

- larger $b$ → stable  
- smaller $b$ → better exploration  

---

## Complexity : 

- Per update: $O(bd)$  
- Epoch cost: $O(nd)$  

---

# Momentum : 

## Update

$$
v_t = \beta v_{t-1} + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

---

## What Math Is Doing : 

Unroll:

$$
v_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2}+...
$$

This is **exponential moving average of gradients.**

In eigen-direction:

$$
z_{t+1}=(1+\beta-\eta \lambda_i)z_t-\beta z_{t-1}
$$

Second-order dynamics ⇒ acceleration.

---

## Complexity : 

- Compute: $O(d)$  
- Space: $O(2d)$  

---

## Failure : 

- overshoot  
- hyperparameter coupling  

---

# Nesterov Accelerated Gradient :

## Update

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

## What Math Means; 

Gradient evaluated at predicted future location.

Provides optimal convex rate:

$$
O(1/t^2)
$$

Earlier curvature feedback ⇒ smoother convergence.

---

# AdaGrad : 

## Update

$$
r_t = r_{t-1} + g_t^2
$$

$$
\theta_{t+1} =
\theta_t -
\frac{\eta}{\sqrt{r_t+\epsilon}} g_t
$$

---

## Mathematical Meaning : 

Scaling matrix:

$$
D_t = \mathrm{diag}(r_t)
$$

Thus step becomes:

$$
\Delta\theta = -\eta D_t^{-1/2} g_t
$$

Equivalent to diagonal second-order preconditioning.

---

## Failure : 

$$
r_t \uparrow \Rightarrow \eta_{eff} \downarrow \to 0
$$

Training stops.

---

# AdaDelta : 

## Update

$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2
$$

$$
\Delta\theta_t =
-
\frac{\sqrt{E[\Delta\theta^2]_{t-1}}}
{\sqrt{E[g^2]_t}} g_t
$$

---

## Math Meaning : 

Effective LR:

$$
\eta_{eff}=
\frac{\mathrm{RMS}(\Delta\theta)}
{\mathrm{RMS}(g)}
$$

Self-normalizing step size.

---

# RMSProp : 

## Update

$$
s_t=\beta s_{t-1}+(1-\beta)g_t^2
$$

$$
\theta_{t+1}=
\theta_t-
\frac{\eta}{\sqrt{s_t}}g_t
$$

---

## Math Meaning : 

Finite-window estimator:

$$
s_t \approx \mathbb{E}[g^2]
$$

Diagonal Newton approximation.

---

# Adam : 

## Updates

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$

Bias correction:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

Final step:

$$
\theta_{t+1}=
\theta_t-
\frac{\eta}{\sqrt{\hat v_t}} \hat m_t
$$

---

## Mathematical Meaning : 

Adam performs momentum in adaptively rescaled space:

$$
\Delta\theta=-\eta D_t^{-1}\hat m_t
$$

Where:

$$
D_t=\mathrm{diag}(\sqrt{\hat v_t})
$$

---

# Comparison :

# Massive Optimizer Comparison :

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


# Benchmark Settings : 

| Setting | Model | Purpose |
|--------|------|---------|
| Synthetic Quadratic | parameter vector | trajectory visualization |
| California Housing | shallow MLP | regression convergence |
| MNIST | neural network | deep non-convex training |

---
