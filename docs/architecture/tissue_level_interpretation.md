# Architecture: Tissue-Level Interpretation

**Scientific layer:** 6 — Tissue-level interpretation and evaluation
**Package location:** `stagebridge/evaluation/`

## Role in the System

This layer bridges model outputs (trajectories in latent space, velocity fields, loss values) and biological claims (niche gating, evolutionary constraint, tissue dynamics). It transforms the transition model from a machine learning benchmark into a scientific tool.

## Components

### Quantitative Evaluation

Standard metrics computed on donor-held-out test sets:
- Sinkhorn divergence between predicted and true target distributions
- MMD (kernel-based distribution comparison)
- Wasserstein distance
- Per-cell-type transport accuracy

These establish whether the model works at all. They do not, by themselves, constitute a scientific contribution.

### Ablation Framework

Systematic comparison of model configurations:
- RNA-only vs set-only (does spatial context help?)
- Set-only vs graph-of-sets (does graph attention help?)
- With vs without WES regularization (does evolutionary state help?)
- Learned vs Gaussian-SB (does learning improve over the prior?)

Each ablation uses identical data splits and evaluation protocols.

### Context Sensitivity

Niche shuffling test: permute niche compositions across patients within the same stage. Measure prediction change. A model that actually uses context will produce different trajectories after shuffling; one that ignores context will not.

### Dynamical Analysis

Computed from the learned drift field:
- **Fixed points** — Where drift is near zero. Mapped to biological states.
- **Niche regimes** — Clusters of niche compositions with distinct transition dynamics. This is the primary biological output.
- **Trajectory structure** — Convergence, divergence, bifurcation in learned paths.
- **Pseudotime correspondence** — Consistency with independent temporal ordering methods.

### Gene/Program Attribution

Gradient-based attribution from the drift network:
- Which genes contribute most to velocity at key transitions?
- Do attributed genes match known biology (surfactant, EMT, immune evasion)?
- Attribution as sanity check and hypothesis generator.

### Tissue-Level Reporting

Aggregate results into interpretable summaries:
- Per-edge: dominant drift direction, niche dependence strength, transition rate
- Per-stage: most dynamic vs most stable populations
- Cross-edge: how dynamics change through progression

## What Goes In

- Trained transition model with drift network
- Held-out test data with niche context
- WES features for regularization analysis

## What Comes Out

- Metrics tables (JSON)
- Ablation comparison tables
- Fixed point maps
- Niche regime characterizations
- Gene attribution rankings
- Tissue-level summary reports

## Key Design Principle

Evaluation code lives in `stagebridge/evaluation/`, not in the training pipeline or visualization layer. The training loop does not own tissue-level interpretation. The evaluation layer is not optional or deferred — it ships with the model.

## Relationship to Other Layers

- **Upstream:** Transition model provides the learned dynamics; context model provides niche representations
- **Downstream:** Results tracking records evaluation outputs; visualization renders them
