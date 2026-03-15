# Tissue Dynamics Outputs

## Why Dynamical Interpretation Matters

A transition model that only predicts cell endpoints without revealing the dynamics of how cells get there is an expensive regression. The scientific value of StageBridge lies in what the learned dynamics reveal about tissue biology.

## Key Dynamical Outputs

### Trajectory Structure

The shape and organization of learned flow trajectories in latent space:

| Property | Description | Biological Meaning |
|----------|-------------|-------------------|
| **Convergence** | Do trajectories from different sources converge? | Common attractor states |
| **Divergence** | Do similar sources diverge based on context? | Niche-dependent fate decisions |
| **Smoothness** | How smooth are the velocity fields? | Continuous vs discontinuous transitions |
| **Edge specificity** | Does each transition have distinct geometry? | Stage-specific dynamics |

### Niche Regimes

Clusters of niche compositions that produce qualitatively different transition behavior:

| Regime Type | Description | Example |
|-------------|-------------|---------|
| **Permissive** | Transitions proceed readily | High proliferation signal |
| **Restrictive** | Transitions slowed or blocked | Immune surveillance |
| **Divergent** | Trajectories bifurcate | Stromal vs epithelial fate |

Identifying niche regimes is the primary output for testing the niche-gating hypothesis.

### Velocity Field Analysis

Properties of the learned velocity field `v_θ(x, t, c)`:

| Analysis | Method | Reveals |
|----------|--------|---------|
| **Fixed points** | Find x where v ≈ 0 | Stable/attractor states |
| **Divergence** | ∇·v at each point | Source/sink regions |
| **Context sensitivity** | ∂v/∂c | How much does niche affect dynamics? |

### Gene/Program Attribution

Which genes or programs contribute most to the velocity at key transitions:

| Transition | Expected Programs |
|------------|-------------------|
| Normal→AAH | Surfactant, early proliferation |
| AAH→AIS | Cell cycle, metabolic shift |
| AIS→MIA | EMT-related, invasion programs |
| MIA→LUAD | Immune evasion, angiogenesis |

Attribution should be validated against known LUAD biology.

### Transition Rate Variation

How transition dynamics vary across conditions:

| Comparison | Question |
|------------|----------|
| By niche | Do immune-rich niches accelerate or slow progression? |
| By evolution | Do high-TMB samples show different dynamics? |
| By edge | Is AAH→AIS faster or slower than AIS→MIA? |

## V1 Required Outputs

For publication, V1 must produce:

1. **Transition quality metrics** — Sinkhorn distance, MMD, trajectory smoothness
2. **Niche conditioning effect** — Comparison of conditioned vs unconditioned
3. **Niche regime identification** — At least preliminary clustering
4. **Context sensitivity analysis** — Quantify niche contribution to dynamics
5. **Biological validation** — Gene programs at key transitions

## V2 Extended Outputs

Deferred to V2:
- Full fixed point / attractor analysis
- Phase portrait visualization
- Cohort-level transport structure
- Detailed divergence/convergence analysis

## Why These Outputs Matter

A methods paper needs more than benchmark metrics. These outputs transform StageBridge from a technical contribution (new architecture, lower distance metrics) into a biological contribution (framework revealing how niche structure gates cancer initiation).

The claim "niche-gated transitions" requires evidence from these dynamical outputs, not just improved prediction accuracy.
