# Citation Map

Maps key papers to StageBridge method components. This is a reference for understanding which ideas inform each part of the system.

## Schrodinger Bridge and Transport

### Gaussian Schrodinger Bridge (Bunne et al.)
**"The Schrodinger Bridge between Gaussian Measures has a Closed Form"**

- **Component:** Gaussian initialization for the transition model
- **How it's used:** Closed-form Gaussian SB between source and target stage distributions provides the initial reference process. The learned drift-diffusion model starts from this prior and improves upon it.
- **Why it matters:** Principled initialization prevents poor starting conditions. The Gaussian SB is also a natural baseline — if the learned model does not beat it, the additional complexity is not justified.

### HM-OT (Huguet et al.)
**Latent transition structure comparator**

- **Component:** Evaluation / ablation baseline
- **How it's used:** Provides a comparison point for latent-space transition structure. Used to benchmark whether StageBridge's learned dynamics capture meaningful structure beyond what OT alone provides.

### CellOT (Bunne et al.)
**Neural OT for single-cell dynamics**

- **Component:** Future ablation / comparator
- **How it's used:** Neural OT approach to cell-level transport. Relevant as a baseline for comparing learned dynamics approaches. Direct comparison is a future ablation target.

## Spatial Methods

### Tangram (Biancalani et al.)
**"Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram"**

- **Component:** Spatial mapping layer (primary implementation)
- **How it's used:** Maps snRNA-seq profiles onto Visium spots to produce spot-level cell-type composition scores. These compositions become the niche tokens that define typed biological sets.
- **Why it matters:** Tangram is the bridge between single-cell resolution and spatial context. Without it, the context model has no spatial information.

### TACCO (Stickels et al.)
**"TACCO unifies annotation transfer and decomposition of cell identities for single-cell and spatial omics"**

- **Component:** Spatial mapping layer (alternative implementation)
- **How it's used:** Compositional transfer as an alternative to Tangram. Shares the common output contract (spot-level composition scores).

### DestVI (Lopez et al.)
**"DestVI identifies continuums of cell types in spatial transcriptomics data"**

- **Component:** Spatial mapping layer (alternative implementation)
- **How it's used:** Deep generative spatial decomposition. Provides richer output (continuous state within cell types) but is more complex to train. Alternative for future comparison.

## Dynamics and Tissue Interpretation

### OSDR / Tissue Dynamics (Nitzan et al.)
**"Temporal tissue dynamics from a spatial snapshot"**

- **Component:** Tissue-level interpretation, neighborhood-driven dynamics
- **How it's used:** Motivates the idea that spatial snapshots contain temporal information. The approach of inferring dynamics from spatial organization informs how StageBridge interprets learned drift fields in tissue context.
- **Why it matters:** Demonstrates that local tissue structure encodes dynamical information — the conceptual foundation of the niche-gating hypothesis.

### TLS Spatial Paper
**Spatial graph learning, pseudotime-like structure, pathology-linked interpretation**

- **Component:** Evaluation layer (trajectory analysis, pseudotime correspondence)
- **How it's used:** Spatial graph construction and pseudotime-like ordering provide independent validation targets for learned trajectories. Pathology-linked interpretation motivates tissue-level reporting.

### scDiffEq
**Stochastic drift-diffusion for single-cell dynamics**

- **Component:** Transition model (drift-diffusion SDE formulation)
- **How it's used:** Motivates the parameterization of cell dynamics as stochastic differential equations with learned drift and diffusion. StageBridge adapts this to edge-wise, niche-conditioned transitions.

## Set and Graph Architecture

### Set Transformer (Lee et al.)
**"Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"**

- **Component:** Context model (intra-set encoding)
- **How it's used:** ISAB, SAB, and PMA layers compress biological sets into fixed-size summary representations. Permutation invariance respects the unordered nature of cell populations.

### Deep Sets (Zaheer et al.)
**"Deep Sets"**

- **Component:** Theoretical foundation for set-based processing
- **How it's used:** Establishes the permutation-invariant function framework. The Set Transformer extends this with attention mechanisms.

## Reference Mapping

### HLCA (Sikkema et al.)
**Human Lung Cell Atlas**

- **Component:** Reference latent mapping
- **How it's used:** Provides the reference coordinate system. All cells are embedded in HLCA latent space. Cell-type labels from HLCA define the token types for the context model.

### scArches (Lotfollahi et al.)
**Architecture surgery for reference mapping**

- **Component:** Reference latent mapping (implementation)
- **How it's used:** Model surgery and query training to project new data into an existing reference latent space.

## Notes

- Citations are mapped to components, not to claims. StageBridge does not claim to replicate or extend these methods — it uses their ideas in a specific, integrated context.
- Additional papers may be relevant; this map covers the primary methodological inspirations.
