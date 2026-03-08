# 006 — Transition Model Contract

Transition learning code lives under `stagebridge/transition_model/`. The active v1 framing is edge-wise stochastic dynamics with WES regularization.

## What the Model Is

- **Edge-wise** — Each disease edge (e.g., AAH to AIS) has its own transition dynamics
- **Stochastic** — Captures both directed drift and stochastic variation
- **Drift-diffusion based** — Parameterized as an SDE with learned drift and diffusion
- **Gaussian Schrodinger Bridge informed** — Initialized from closed-form Gaussian SB prior
- **Optionally OT-coupled** — Entropic OT couplings provide warm-start pairings
- **WES-regularized** — Evolutionary features constrain admissible transport paths
- **Conditioned on typed niche context** — Drift network receives context from Set/Graph Transformer

## What the Model Is Not

Not merely latent interpolation. Does not linearly interpolate between stage centroids, average cell positions, treat transitions as deterministic point maps, or ignore stochastic biology. Latent interpolation produces geometrically plausible paths; the transition model learns dynamically plausible paths that respect biological structure.

## Components

### Disease Edges

| Edge | Source | Target |
|------|--------|--------|
| 0 | Normal | AAH |
| 1 | AAH | AIS |
| 2 | AIS | MIA |
| 3 | MIA | LUAD |

Each edge may have distinct dynamics.

### OT Couplings

Entropic optimal transport computes soft pairings between source and target distributions. Computed with Sinkhorn iterations, entropic regularization, precomputed per edge. Optional — model can also train with random or Gaussian bridge pairings.

### Gaussian Initialization

Fit Gaussians to source and target populations. Compute closed-form Schrodinger bridge between them. Use as initialization for learned drift-diffusion. Provides a natural baseline: if the learned model does not improve over Gaussian SB, spatial context is not helping.

### Drift Network

f(x_t, t, c) predicts velocity at point x_t, time t, context c. MLP with FiLM conditioning (sinusoidal time embedding modulates hidden layers). Input includes stage pair embedding.

### Diffusion Schedule

Controls stochastic noise at each time step. Can be fixed (linear, cosine) or learned. Governs balance between deterministic drift and stochastic exploration.

### Stochastic Dynamics Wrapper

Integrates drift-diffusion SDE from t=0 (source) to t=1 (target). Euler-Maruyama for training, higher-order integrators available for inference. Produces sampled trajectories.

### Schrodinger Bridge Objective

Given OT-coupled pairs (x_0, x_1), sample intermediate x_t via bridge interpolant. Train drift to predict conditional velocity at x_t. Loss: expected squared error between predicted and target velocity. Learns a stochastic process matching marginals at source/target while following minimum-entropy path.

### WES Regularizer

In v1, WES enters as regularization, not direct drift conditioning:
- Per-donor evolutionary state vector (mutation burden, drivers, CNV summary)
- Auxiliary loss penalizing transport paths inconsistent with genomic constraints
- Example: high-mutation-burden donors should not have identical dynamics to low-mutation donors
- Unrestricted WES conditioning deferred to v2 to avoid confounding

### Baseline Modes

| Mode | Description |
|------|-------------|
| Linear | Deterministic linear interpolation (sanity check) |
| No-context | Learned drift without niche conditioning |
| Gaussian-SB only | Gaussian bridge without learned correction |
| Set-only | Full model with Set Transformer, no GoST |
| Full | Set + GoST + WES regularization |
