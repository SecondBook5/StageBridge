# Tissue Dynamics Outputs

## Why Dynamical Interpretation Matters

A transition model that predicts cell endpoints without revealing anything about the dynamics of how cells get there is an expensive regression. The scientific value of StageBridge lies in what the learned dynamics reveal about tissue biology.

## Key Dynamical Outputs

### Fixed Points

Points in latent space where the drift field is near zero — states that would not transition under the learned dynamics. Biologically, these may correspond to:

- Terminally differentiated cell states (e.g., mature alveolar cells that do not progress)
- Stem-like or progenitor states that are dynamically stable
- Barrier states that resist transition

Fixed points are stage-dependent: a cell state that is a fixed point in the Normal-to-AAH dynamics may not be a fixed point in the MIA-to-LUAD dynamics.

### Niche Regimes

Clusters of niche compositions that produce qualitatively different transition behavior. These answer the core biological question: which tissue neighborhoods gate progression?

Expected regime types:
- **Permissive niches** — Niche compositions where transitions proceed readily
- **Restrictive niches** — Compositions where transitions are slowed or redirected
- **Divergent niches** — Compositions where transition trajectories bifurcate into distinct outcomes

Identifying niche regimes is the primary output relevant to the niche-gating hypothesis.

### Trajectory Structure

The shape and organization of learned trajectories in latent space. Informative properties:

- **Convergence** — Do trajectories from different source states converge to common targets?
- **Divergence** — Do trajectories from similar sources diverge based on niche or evolutionary context?
- **Pseudotime ordering** — Does the learned dynamics produce a temporal ordering consistent with independent methods?
- **Edge-specific structure** — Does each disease edge have qualitatively distinct trajectory geometry?

### Gene/Program Attribution

Which genes or transcriptional programs contribute most to the velocity field at key transitions. This connects model dynamics to molecular biology:

- Surfactant programs in early stages (Normal to AAH)
- Proliferation programs at the hyperplasia boundary
- EMT-related programs at the invasion boundary (AIS to MIA)
- Immune evasion programs during progression

Attribution should be validated against known LUAD biology as a sanity check.

### Transition Rate Variation

How transition speed varies across:
- Niche composition — Do immune-rich niches accelerate or slow progression?
- Evolutionary state — Do high-mutation-burden donors show faster transitions?
- Disease edge — Is AAH-to-AIS faster or slower than AIS-to-MIA?

### Tissue-Level Summary

Aggregate dynamical outputs into tissue-level reports:
- Per-edge: dominant drift direction, typical trajectory duration, niche dependence strength
- Per-stage: which populations are most dynamic, which are stable
- Cross-edge: how dynamics change as disease progresses

## Why These Outputs Matter for the Paper

A Nature Methods submission needs more than benchmark metrics. Tissue dynamics outputs transform the model from a technical contribution (a new architecture that achieves lower Sinkhorn distance) into a biological contribution (a framework that reveals how niche structure gates cancer initiation). The evaluation contract (007) specifies how these outputs are computed; this document explains why they matter.
