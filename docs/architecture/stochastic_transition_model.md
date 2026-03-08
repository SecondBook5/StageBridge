# Architecture: Stochastic Transition Model

**Scientific layer:** 5 — Edge-wise stochastic transition modeling
**Package location:** `stagebridge/transition_model/`

## Role in the System

The transition model is the core scientific component. It learns how cells move from one disease stage to the next in HLCA latent space, conditioned on tissue microenvironment context and regularized by evolutionary state.

## Architecture

### Edge-Wise Design

Each disease edge (Normal→AAH, AAH→AIS, AIS→MIA, MIA→LUAD) has its own transition dynamics. The drift network takes a stage pair embedding so it can specialize per edge while sharing parameters.

### Drift-Diffusion SDE

The dynamics are:
```
dx_t = f(x_t, t, c, e) dt + sigma(t) dW_t
```

where:
- `f` is the learned drift (velocity field)
- `c` is the niche context vector from the context model
- `e` is the stage pair embedding
- `sigma(t)` is the diffusion coefficient (fixed schedule or learned)
- `dW_t` is Brownian noise

### Drift Network

MLP with FiLM conditioning:
- Sinusoidal time embedding modulates hidden layers
- Context vector c enters via concatenation or FiLM
- Stage pair embedding selects edge-specific behavior
- Output: predicted velocity at (x_t, t)

### Gaussian Schrodinger Bridge Initialization

Before learning, compute the closed-form Gaussian SB between source and target stage distributions:
- Fit multivariate Gaussians to source and target cells in HLCA latent space
- Compute the SB mean and covariance paths
- Use as initialization for the drift network (or as a baseline to beat)

### OT Coupling

Entropic optimal transport provides initial pairings:
- Sinkhorn iterations compute soft pairings between source and target cells
- Pairings define (x_0, x_1) training pairs for the flow
- Entropy regularization avoids degenerate matchings
- Precomputed per edge and cached

### Training (Schrodinger Bridge Objective)

1. Sample an OT pair (x_0, x_1)
2. Sample time t ~ Uniform(0, 1)
3. Compute bridge interpolant x_t between x_0 and x_1
4. Compute target velocity from the bridge
5. Predict velocity with drift network f(x_t, t, c, e)
6. Loss = ||predicted - target||^2

### WES Regularization

Auxiliary loss term:
- Compute per-donor transition statistics (e.g., average drift magnitude, trajectory spread)
- Penalize when donors with different WES profiles produce identical statistics
- Effect: the model produces evolutionary-state-aware dynamics

### Integration (Inference)

Euler-Maruyama integration from t=0 to t=1:
```
x_{t+dt} = x_t + f(x_t, t, c, e) * dt + sigma(t) * sqrt(dt) * z
```

Higher-order integrators available. Produces full trajectories, not just endpoints.

## Baseline Configurations

| Config | Drift | Context | WES | OT |
|--------|-------|---------|-----|-----|
| Linear | None (linear interp) | No | No | No |
| No-context | Learned | No | No | Yes |
| Gaussian-SB | Gaussian prior only | No | No | No |
| Set-only | Learned | Set Transformer | No | Yes |
| Full | Learned | Set + GoST | Yes | Yes |

## Relationship to Other Layers

- **Upstream:** Context model provides conditioning vector c; reference mapping defines the latent space; data ingestion provides cells
- **Downstream:** Evaluation layer assesses transition quality and biological meaning
