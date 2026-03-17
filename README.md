# Optimization Algorithms : 

---

## Repository Objective : 

This repository studies the evolution of **first-order optimization algorithms used for training machine learning models**.

The goal is to understand optimizers from first principles:

- What optimization problem are we solving  
- Why vanilla gradient descent fails in practice  
- How conditioning, curvature and stochastic noise affect convergence  
- How each new optimizer improves a specific limitation of previous methods  
- How optimizers behave across different learning settings  

We benchmark optimizers across three problem settings :

| Setting | Model | Purpose |
|--------|------|---------|
| Synthetic quadratic surface | Parameter vector | Shows optimizer trajectory geometry and conditioning effects |
| California Housing regression | MLP | Shows convergence stability, gradient norms, training speed |
| MNIST classification | Neural network | Shows deep non-convex training behaviour and generalization |

---

## Optimization Problem

We solve:

$$
\min_{\theta \in \mathbb{R}^d} J(\theta)
$$

Where:

- $ \theta $ is the parameter vector  
- $ J(\theta) $ is the empirical loss  
- $ g_t = \nabla J(\theta_t) $ is the gradient at step $t$  

Curvature of the loss surface is described by the **Hessian** :

$$
H = \nabla^2 J(\theta)
$$

If eigenvalues of \(H\) vary significantly:

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \gg 1
$$

the problem is **ill-conditioned**, leading to slow zig-zag convergence.

---

## Evolution of Optimizers : 

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

# Gradient Descent (Batch) : 

### Intuition  
Uses exact gradient direction computed on full dataset.  
Stable but computationally expensive and sensitive to conditioning.

### Technical Insight  
For quadratic loss:

$$
J(\theta)=\frac{1}{2}\theta^T H \theta
$$

Update becomes:

$$
\theta_{t+1}=(I-\eta H)\theta_t
$$

Convergence rate depends on spectral radius:

$$
\rho = \max_i |1-\eta\lambda_i|
$$

### Time Complexity  
- Per step: \(O(nd)\)

### Shortcomings  
- Slow for large datasets  
- Severe oscillation in narrow valleys  
- Requires small learning rate  

---

# Stochastic Gradient Descent : 

### Intuition  
Uses gradient from a single sample.  
Cheap updates but introduces noise.

### Technical Insight  
Gradient estimator:

$$
\mathbb{E}[g_t]=\nabla J(\theta_t)
$$

Variance:

$$
\mathrm{Var}(g_t)=\sigma^2
$$

Noise helps escape saddle points.

### Time Complexity  
- Per update: \(O(d)\)

### Shortcomings  
- High variance updates  
- Oscillatory convergence  

---

# Mini-Batch Gradient Descent : 

### Intuition  
Balances stability and computational efficiency.

### Technical Insight  
Gradient estimate:

$$
g_t=\frac{1}{b}\sum_{i=1}^b \nabla L_i(\theta_t)
$$

Variance reduces:

$$
\mathrm{Var}(g_t)=\frac{\sigma^2}{b}
$$

### Time Complexity  
- Per step: \(O(bd)\)

### Shortcomings  
- Requires batch size tuning  

Improvement: reduced variance compared to SGD.

---

# Momentum (Polyak Heavy Ball) : 

### Intuition  
Accumulates gradient history to accelerate in consistent directions and damp oscillations.

### Technical Insight  
Velocity update:

$$
v_t=\beta v_{t-1}+g_t
$$

Parameter update:

$$
\theta_{t+1}=\theta_t-\eta v_t
$$

Unrolled:

$$
v_t=\sum_{k=0}^{t}\beta^{t-k} g_k
$$

Second-order dynamics:

$$
z_{t+1}=(1+\beta-\eta\lambda_i)z_t-\beta z_{t-1}
$$

Improves convergence from \(O(\kappa)\) to \(O(\sqrt{\kappa})\).

### Time Complexity  
- \(O(d)\)

### Shortcomings  
- Same learning rate for all parameters  
- Overshoot possible  

Improvement: solves conditioning and zig-zag problem.

---

# Nesterov Accelerated Gradient : 

### Intuition  
Computes gradient at predicted future position.

### Technical Insight  

$$
\tilde{\theta}_t=\theta_t-\eta\beta v_{t-1}
$$

$$
g_t=\nabla J(\tilde{\theta}_t)
$$

Provides optimal convex convergence:

$$
O(1/t^2)
$$

### Time Complexity  
- \(O(d)\)

### Shortcomings  
- Sensitive hyperparameters  
- Still global learning rate  

Improvement: earlier curvature feedback.

---

# AdaGrad : 

### Intuition  
Uses smaller learning rates for frequently updated parameters.

### Technical Insight  

$$
r_t=r_{t-1}+g_t^2
$$

$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{r_t+\epsilon}}g_t
$$

Equivalent to diagonal preconditioning:

$$
\Delta\theta=-\eta D_t^{-1/2}g_t
$$

### Time Complexity  
- \(O(d)\)

### Shortcomings  
- Learning rate decays to zero  

Improvement: per-parameter adaptive scaling.

---

# AdaDelta : 

### Intuition  
Removes monotonically shrinking learning rate.

### Technical Insight  

$$
E[g^2]_t=\beta E[g^2]_{t-1}+(1-\beta)g_t^2
$$

$$
\Delta\theta_t=
-\frac{\sqrt{E[\Delta\theta^2]_{t-1}}}{\sqrt{E[g^2]_t}}g_t
$$

Effective LR:

$$
\eta_{\text{eff}}=
\frac{\text{RMS}(\Delta\theta)}{\text{RMS}(g)}
$$

### Time Complexity  
- \(O(d)\)

### Shortcomings  
- Slower convergence  

Improvement: prevents AdaGrad decay.

---

# RMSProp : 

### Intuition  
Uses exponentially weighted variance estimate.

### Technical Insight  

$$
s_t=\beta s_{t-1}+(1-\beta)g_t^2
$$

$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{s_t}}g_t
$$

Acts as stochastic diagonal Newton step.

### Time Complexity  
- \(O(d)\)

### Shortcomings  
- No momentum smoothing  

Improvement: stable adaptive scaling.

---

# Adam : 

### Intuition  
Combines momentum and adaptive scaling.

### Technical Insight  

First moment:

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t
$$

Second moment:

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

Update:

$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat v_t}}\hat m_t
$$

Equivalent to momentum in adaptive coordinate space.

### Time Complexity  
- \(O(d)\)

### Shortcomings  
- Converges to sharp minima  
- Higher memory  

Improvement: combines direction smoothing + variance normalization.

---

## Visualization Objectives Across Experiments

| Experiment | Visualization |
|-----------|--------------|
| Synthetic surface | Trajectory curves showing zig-zag vs adaptive paths |
| Housing regression | Loss curves, gradient norms, convergence epochs |
| MNIST | Training speed vs final accuracy |

---

## Massive Comparison Table

| Optimizer | TC/Step | Space | Adaptive | Momentum | Convergence Speed | Generalization | Noise Stability | Failure Risk |
|-----------|--------|------|----------|----------|------------------|---------------|---------------|-------------|
| GD | O(d) | O(d) | No | No | Slow | Good | Low | Stuck |
| SGD | O(d) | O(d) | No | No | Medium | Good | Medium | Oscillation |
| MiniBatch | O(d) | O(d) | No | No | Fast | Good | Medium | Batch tuning |
| Momentum | O(d) | O(2d) | No | Yes | Faster | Very Good | Medium | Overshoot |
| NAG | O(d) | O(2d) | No | Yes | Very Fast | Good | Medium | Sensitive |
| AdaGrad | O(d) | O(2d) | Yes | No | Fast early | Medium | High | LR collapse |
| AdaDelta | O(d) | O(3d) | Yes | No | Medium | Medium | Medium | Slow |
| RMSProp | O(d) | O(2d) | Yes | No | Fast | Medium | Good | Instability |
| Adam | O(d) | O(3d) | Yes | Yes | Very Fast | Sometimes worse | High | Sharp minima |

---
