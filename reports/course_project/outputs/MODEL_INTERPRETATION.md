# Model Interpretation

## What the transformer is doing in StageBridge

StageBridge uses a **Set Transformer** to encode the local tissue niche context around each cell. Rather than treating cells as isolated transcriptomic profiles, the Set Transformer processes the typed spatial neighborhood as an unordered set of tokens, using attention mechanisms to learn which niche features are most informative for predicting stage transitions.

The key insight is that tumor progression is not just a cell-intrinsic process: the tissue microenvironment (immune infiltration, stromal remodeling, vascular programs) actively shapes how cells evolve from one stage to the next. The Set Transformer captures these relationships through:

1. **Induced Set Attention Block (ISAB)**: Compresses the variable-size token set through a learned set of inducing points, reducing O(n^2) attention to O(n*m) while preserving set-level information.
2. **Self-Attention Block (SAB)**: Refines token representations through full self-attention over the compressed set.
3. **Pooling by Multihead Attention (PMA)**: Produces a fixed-size context embedding from the variable-size set via learned seed vectors.

The resulting context vector conditions the downstream drift network, modulating the predicted velocity field for each cell based on its local niche composition.

## Why Set Transformer is the main active mechanism

1. **Permutation invariance**: Spatial neighborhoods have no natural ordering. The Set Transformer handles this natively, unlike sequence models.
2. **Attention-based weighting**: Different niche components (epithelial, stromal, immune, vascular) contribute differently to different transitions. Self-attention learns these relative importances.
3. **Empirical superiority over pooling**: On the AIS->MIA transition, Set Transformer (15.758) outperforms pooled context (15.909), confirming that attention-based encoding captures information that mean/std/max pooling misses.

## Why Graph Transformer is currently optional

The Graph-of-Sets Transformer adds tissue-level graph attention on top of local set encoding. In principle, this should capture broader tissue organization (e.g., spatial gradients in immune infiltration across the tissue). In practice:

- Graph-of-sets underperforms set_only on both tested edges
- The graph construction from spatial coordinates may not capture the right tissue-level patterns at current resolution
- The additional complexity does not yet earn its place empirically

The Graph Transformer remains in the codebase as an extension for future investigation with improved graph construction strategies.

## What the transition model does downstream

The context embedding from the Set Transformer conditions an **edge-conditioned drift network** that predicts velocity fields in the HLCA latent space. Specifically:

1. **Entropic OT coupling** (Sinkhorn) pairs source and target cells across stages
2. A **Schrodinger bridge interpolant** generates time-indexed training points between paired cells
3. The **drift network** predicts the velocity field v(x_t, t, c_s, e) conditioned on position x_t, time t, context c_s, and edge identity e
4. **Euler integration** from source to target produces predicted cell-state trajectories
5. Evaluation compares predicted target distributions against held-out observations via Sinkhorn divergence, classifier AUC, and calibration error
