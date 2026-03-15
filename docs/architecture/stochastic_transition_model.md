# Architecture: Stochastic Transition Model (Layer D)

**Scientific layer:** D — Cell-state transition dynamics
**Package location:** `stagebridge/transition_model/`

## Role in the System

The transition model is the core scientific component. It learns how cells move between disease stages in dual-reference latent space, conditioned on local niche context and constrained by evolutionary compatibility.

**V1 uses Flow Matching (OT-CFM). Neural SDE is deferred to V2.**

## V1: Flow Matching (OT-CFM)

### Overview

Flow Matching learns a deterministic velocity field that transports cells from source to target distributions. With optimal transport coupling, it provides:
- Efficient training (simulation-free)
- Principled cell-to-cell pairing via Sinkhorn OT
- Continuous trajectories for interpretation

### Mathematical Formulation

The flow is defined by an ODE:
```
dx_t/dt = v_θ(x_t, t, c)
```

where:
- `v_θ` is the learned velocity field (neural network)
- `t ∈ [0, 1]` is the flow time
- `c` is the niche context vector from Layer C
- `x_0 ~ p_source`, `x_1 ~ p_target`

### OT Coupling (Sinkhorn)

Optimal transport provides principled pairing between source and target cells:

1. Compute cost matrix `C_ij = ||x_i^source - x_j^target||^2`
2. Sinkhorn iterations find entropic OT coupling `π*`
3. Sample pairs `(x_0, x_1) ~ π*` for training
4. Entropy regularization `ε` prevents degenerate matchings

Coupling is precomputed per disease edge and cached.

### Training Objective

Conditional Flow Matching (CFM) loss:

```
L_CFM = E_{t, (x_0,x_1)~π*} [ ||v_θ(x_t, t, c) - u_t(x_t | x_0, x_1)||^2 ]
```

where `u_t` is the conditional vector field:
```
x_t = (1-t) * x_0 + t * x_1
u_t = x_1 - x_0
```

### Velocity Network Architecture

MLP with context conditioning:
- Input: `[x_t, t_embed, c]` where `t_embed` is sinusoidal time embedding
- Hidden layers: 2-3 layers with GELU activation
- Context enters via concatenation or FiLM modulation
- Output: predicted velocity `v_θ(x_t, t, c)`

### Niche Conditioning

The context vector `c` from Layer C conditions the velocity field:
- Encodes local tissue microenvironment
- Allows niche-specific transition dynamics
- Ablation: compare conditioned vs unconditioned flow

### Inference

Euler integration from t=0 to t=1:
```
x_{t+dt} = x_t + v_θ(x_t, t, c) * dt
```

Higher-order integrators (RK4) available for smoother trajectories.

## V2: Neural SDE (Deferred)

Neural SDE extends flow matching with stochastic dynamics:

```
dx_t = f_θ(x_t, t, c) dt + σ(t) dW_t
```

This is **not required for V1** but provides:
- Uncertainty quantification via trajectory variance
- More expressive dynamics for multimodal transitions
- Score matching training objective

## Edge-Wise Design

Each disease edge has distinct dynamics:
- Normal→AAH, AAH→AIS, AIS→MIA, MIA→LUAD
- Edge embedding selects specialized behavior
- Shared parameters with edge-specific modulation

## WES Regularization

Auxiliary loss enforces evolutionary consistency:
- Penalizes when different WES profiles produce identical dynamics
- Effect: model learns evolutionary-state-aware transitions
- Ablation: compare with/without WES constraint

## Baseline Configurations

| Config | Velocity | Context | WES | OT Coupling |
|--------|----------|---------|-----|-------------|
| Linear | None (interpolation) | No | No | No |
| Uncoupled | Learned | No | No | Random pairs |
| OT-only | Learned | No | No | Yes |
| Conditioned | Learned | Layer C | No | Yes |
| Full V1 | Learned | Layer C | Regularizer | Yes |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Sinkhorn distance | OT distance between predicted and target distributions |
| MMD-RBF | Maximum mean discrepancy with RBF kernel |
| Trajectory smoothness | Mean velocity magnitude along paths |
| Niche sensitivity | Change in trajectories under context perturbation |

## Relationship to Other Layers

- **Upstream:** Layer A (dual-reference latent) defines the space; Layer B+C (niche encoder) provides context
- **Downstream:** Evaluation assesses transition quality; visualization renders trajectories
