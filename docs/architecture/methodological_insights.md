# Methodological Insights from Literature Review

This document captures key insights from methods papers reviewed during StageBridge development, with direct implications for our architecture.

---

## 1. Transformers in Single-Cell Omics (Szalata et al., Nature Methods 2024)

### Key Distinction: Gene-Centric vs Niche-Centric

| Standard SC Transformers | StageBridge |
|--------------------------|-------------|
| Token = gene | Token = spatial/reference element |
| Self-attention learns gene-gene relationships | Self-attention learns cell-context relationships |
| One cell = one forward pass | One neighborhood = one forward pass |
| Positional = gene rank or none | Positional = type embedding + distance |

### Input Representation Comparison

**Gene-centric models** (Geneformer, scGPT, UCE):
- Genes as tokens, expression as values
- MLM training predicts masked genes
- Learns gene regulatory relationships

**Spatial models** (SpaFormer, CellPLM):
- Multiple cells processed together
- Spatial coordinates as positional encodings
- Closer to StageBridge design

**StageBridge approach**:
- 9 tokens: receiver, ring1-4, HLCA, LuCA, pathway, stats
- MIL aggregation per ring before transformer
- Type embeddings distinguish token roles
- Receiver reconstruction as SSL objective

### Implications
- Our 9-token structure is justified by spatial biology, not convenience
- Type embeddings are critical (more than gene ordering used elsewhere)
- Not competing with gene-level foundation models - complementary approach

---

## 2. CellOT: Neural Optimal Transport for Perturbations (Bunne et al., Nature Methods 2023)

### Core Problem
Learning cell state transitions from *unpaired* distributions - never see the same cell before/after perturbation. This is exactly our situation with cross-sectional stage data.

### Why Optimal Transport Works
1. **Minimal effort principle**: Perturbations cause incremental changes, OT finds transformation minimizing total "transport cost"
2. **Captures heterogeneity**: Different cells respond differently - autoencoders (scGen) only learn mean shifts
3. **Preserves correlation structure**: Inter-feature correlations conserved between conditions

### Technical Approach
- Parameterize OT map via input convex neural networks (ICNNs)
- Learn dual potentials f and g, recover optimal map as gradient of g
- Convexity constraint provides stability (theory-motivated inductive bias)

### Critical Limitation
> "Performance drops when perturbations are too strong (cell distributions before and after are very different)... short-range developmental dynamics work better than long-range"

**Warning for StageBridge**: If Normal→AAH→AIS→LUAD involves dramatic population changes, OT/flow matching may struggle with large gaps.

### Direct Relevance

| CellOT | StageBridge |
|--------|-------------|
| Drug perturbation | Stage transition (natural "perturbation") |
| Control → Treated | Normal → AAH → AIS → LUAD |
| Unpaired cells | Cross-sectional snapshots |
| OT transport map | Flow matching transition head |

---

## 3. Optimal Transport for Single-Cell and Spatial Omics (Bunne et al., Nature Reviews Methods Primers 2024)

### Dynamic OT Formulation
The critical connection - OT can be reformulated dynamically:
- **Static OT**: find coupling P between distributions
- **Dynamic OT**: find time-varying vector field v(t,x) that evolves population along minimal path

The dynamic formulation minimizes kinetic energy while satisfying mass conservation - exactly what flow matching learns.

### Continuity Equation (Eq. 13)
```
∂μ_t/∂t + ∇·(μ_t v) = 0
```
Every curve μ_t can be interpreted as fluid flow along vector fields. Flow matching learns v(t,·) that satisfies this.

### OT for Microenvironment Modeling (Fig 4f)
> "model the ME [microenvironment] of each cell i by aggregating feature vectors of its spatial neighbours into a histogram... compute OT distance between all pairs of cellular MEs"

Validates our niche-centric approach - they use OT to compare microenvironments, we use attention to encode them.

### Gromov-Wasserstein for Heterogeneous Spaces
For aligning spaces with different metrics (like HLCA 30D vs LuCA 10D):
- Optimizes alignment based on *intra-space* distances
- Doesn't require spaces to be directly comparable
- Could be principled approach for dual-reference fusion

### Fate Transition Tables (Fig 5e)
OT plans compress into transition probability matrices between cell states. Example: iPS→Stromal (0.30), iPS→Neural (0.01). This is what we need for stage transitions - quantified probabilities, not just trajectories.

### Schrödinger Bridges
> "the entropy-regularized OT problem coincides with the famous Schrödinger bridges concept, which optimizes for the stochastic process that best describes the evolution of a population"

Connects OT to stochastic dynamics. Recent work uses Schrödinger bridges for differentiation with birth/death events.

### Flow Matching Connection (Outlook)
> "recent deep learning parameterizations of dynamic OT contain technologies known as diffusion generative models and flow matching methods"

**Confirms**: Flow matching is the neural network instantiation of dynamic OT.

### Key Implications

| OT Concept | StageBridge Application |
|------------|------------------------|
| Dynamic OT / velocity field | Flow matching transition head |
| OT distance for MEs | Could compare niche compositions |
| Gromov-Wasserstein | Principled HLCA/LuCA fusion |
| Unbalanced OT | Handle cell death/division |
| Fate transition tables | Stage transition probabilities |

---

## 4. scPhere: Hyperspherical and Hyperbolic Embeddings (Ding & Regev, Nature Communications 2021)

### The Cell Crowding Problem
Standard VAEs with Gaussian priors push all cells toward center of latent space:
- Normal prior encourages posterior means to cluster at origin
- Gets worse with longer training (posterior approximates prior)
- Cell types become indistinguishable in the center

### Solution: Non-Euclidean Geometry

| Geometry | Prior | Best For |
|----------|-------|----------|
| Hyperspherical | von Mises-Fisher (vMF) | Discrete cell types, no crowding |
| Hyperbolic (Poincaré disk) | Wrapped normal | Hierarchical/branching trajectories |

### Why Hypersphere Works
- Uniform distribution on hypersphere has no center
- Points not forced to cluster
- Cells of same type close on surface, but all cells visible
- Uses cosine distance naturally (L2-normalized vectors lie on unit hypersphere)

### Why Hyperbolic for Trajectories
- Exponential volume growth with radius
- Can embed trees with exponentially increasing nodes at depth
- **Distance from center of Poincaré disk = pseudotime**
- Branching developmental trajectories naturally represented

### Batch Correction via Conditioning
scPhere conditions on batch vectors as part of generative model:
- Learns batch-invariant latent z
- Can generate "what would this cell look like in a different batch"
- Handles *multilevel* batch effects (patient + disease + location)

### Component Collapse Problem
In Euclidean VAEs:
- "Component collapse" where decoder ignores some latent dimensions
- With 10D or 20D Euclidean, only 6-7 dimensions actually used
- Hyperspherical spaces don't suffer this - all dimensions contribute

### Key Results
- k-NN accuracy on hypersphere >> Euclidean, even in just 2D
- Preserves hierarchical global structure that t-SNE/UMAP destroy
- Rare cell types remain distinct (not crushed to center)

### Critical Insight for StageBridge

Our stage progression (Normal → AAH → AIS → LUAD) has natural hierarchical structure:
- Normal = "root" state
- Progression = moving away from healthy
- Branching possible (different LUAD subtypes)

**Hyperbolic geometry captures this naturally.** Distance from origin in Poincaré disk literally IS progression distance.

### Implications for StageBridge

| Current Design | scPhere Insight | Potential Improvement |
|----------------|-----------------|----------------------|
| Euclidean fused latent (40D) | Crowding + component collapse | Hyperspherical or hyperbolic |
| Stage as discrete label | Distance from center = pseudotime | Hyperbolic transition space |
| Concatenate HLCA/LuCA | Both references Euclidean | Project fused to hypersphere |
| Donor as batch | Condition decoder on donor | Donor-invariant encoder |

---

## 5. AMICI: Attention Mechanism Interpretation of Cell-cell Interactions (Hong et al., bioRxiv 2025)

**This is the foundational paper for StageBridge's receiver-centric design.**

### Core Problem AMICI Solves

Current spatial transcriptomics methods fail because:
1. Fixed radius/RBF kernels inadequate for varying interaction length scales
2. Broad cell-type labels obscure context-specific subpopulations
3. Existing attention methods yield dense, uninterpretable attention maps

### AMICI's Core Innovation: Receiver-Centered Attention

The framework that directly inspires StageBridge:
1. **Mask receiver's expression** and reconstruct from neighbors
2. Attention weights depend on BOTH neighbor phenotype AND distance
3. **Distance-dependent attention** with monotonically decreasing function
4. **Sparsity regularization** to isolate influential neighbors
5. **Multi-head design** captures multiple length scales simultaneously

### Architecture Details

**Distance-Modulated Attention (Equation 1)**:
```
α_h := Softmax(b_0 - b_1 * d_c || [C_Empty])
```
Where:
- `b_0` = standard QK attention (phenotype-based)
- `b_1` = learned distance coefficient (enforced positive via Softplus)
- `d_c` = vector of distances to all neighbors
- `C_Empty` = "empty neighbor" token allowing attention to drop to zero

**Critical constraint**: The `-b_1 * d_c` term ensures attention MONOTONICALLY DECREASES with distance. This is a hard architectural constraint, not learned freely.

**Query/Key/Value Computation**:
```
Q_c(h) := W_Q(h)^T LayerNorm(E_c)           # Query from receiver CELL-TYPE embedding
K_c'(h) := W_K(h)^T LayerNorm(f_A(X_c'))    # Key from neighbor expression  
V_c'(h) := W_V(h)^T LayerNorm(f_A(X_c'))    # Value from neighbor expression
```

**Key insight**: Query is cell-type embedding, NOT cell expression. AMICI learns "what does a cell of type T attend to?" not "what does this specific cell attend to?"

**Residual Prediction**:
```
Ŷ_c := Ȳ_t(c) + Δ̂(t(c), X_A(c), {d(c,c')}; θ)
```
Output = cell-type mean + attention-weighted neighbor contribution. Model predicts SHIFT from baseline caused by neighbors.

### Loss Function with Sparsity

```
L = ||Ŷ_c - Y_c||² + λ_α * Σ α_hj * log(1/α_hj) + λ_V * Σ|V_ij|
```

Three terms:
1. **MSE reconstruction** - primary objective
2. **Entropy penalty on attention** - encourages sparsity (few influential neighbors)
3. **L1 on value matrix** - encourages sparse influence patterns

### Interpretation Methods

1. **Neighbor Ablation**: Remove all cells of type t', measure prediction change
   ```
   Δ̂_c^{t'} := Ŷ_c^{\t'} - Ŷ_c
   ```

2. **Per-gene significance**: Wald test correlating ablation effect with observed expression

3. **Counterfactual attention**: Given learned Q, K, b_1, compute what attention WOULD be at any distance

4. **Length scale identification**: Distance where counterfactual attention drops below threshold (α_h,th = 0.1)

5. **Communication hubs**: Cluster cells by high-attention neighbor composition (not just neighbor identity)

### Semi-Synthetic Benchmark Design

Ground truth interactions with known length scales:
- Cell Type A → Cell Type B at 10μm
- Cell Type C → Cell Type A at 20μm
- Each cell type has "interacting" and "non-interacting" subclusters
- Interacting subcluster has distinct DE genes activated ONLY when sender is within range

AMICI recovers these ground truth length scales; competing methods (NCEM, NicheDE, GITIII, CGCom) cannot.

### Biological Results

**Mouse cortex (MERFISH)**:
- Oligodendrocytes → Astrocytes: upregulates Igfbp5, Gfap (known remyelination genes)
- Validates known astrocyte-oligodendrocyte signaling

**Breast cancer (Xenium)**:
- CD8+ T cells influence tumor ESR1 expression
- Tumors near T cells become MORE ER-dependent
- Explains paradox: TILs associate with worse survival in luminal breast cancer
- Tumor-immune interactions reinforce ER signaling despite checkpoint blockade

### Key Differences: AMICI vs StageBridge

| AMICI | StageBridge |
|-------|-------------|
| Query = cell-type embedding | Query = receiver cell embedding |
| Predicts gene expression | Predicts receiver latent reconstruction |
| M=50 neighbors, raw expression | 9 tokens, latent embeddings |
| Single attention layer | Set transformer with type embeddings |
| Distance as explicit coefficient | Distance encoded in ring structure |
| Gene-level interpretation | Latent-level + pathway interpretation |
| Static snapshot analysis | Stage transition modeling |

### Critical Insights for StageBridge

1. **Distance-dependent attention is essential**: Our ring structure should capture this - validate that attention decreases with ring number

2. **Sparsity regularization matters**: Consider entropy penalty on attention to prevent dense uninformative maps

3. **Residual from baseline**: AMICI predicts shift from cell-type mean. We do similar with receiver reconstruction from niche context.

4. **Ablation for interpretation**: We can ablate reference tokens (HLCA, LuCA) or ring tokens to understand their contribution - this is principled

5. **Semi-synthetic validation**: Our benchmark should test whether model recovers known interaction distances/rules

6. **Communication hubs**: Could define progression-associated niches by attention patterns, not just cell composition

### What StageBridge Adds Beyond AMICI

1. **Dual-reference context**: HLCA/LuCA tokens provide healthy vs disease reference that AMICI lacks
2. **Stage transition modeling**: Flow matching head for progression dynamics
3. **Latent rather than gene-level**: More compact, enables downstream pathway analysis
4. **Hierarchical aggregation**: MIL within rings before cross-ring attention

---

## Summary: Design Principles for StageBridge

### Validated by Literature
1. **Receiver-centric niche attention** - AMICI, OSDR, spatial models all support this
2. **Flow matching for transitions** - Neural instantiation of dynamic OT, well-suited for unpaired data
3. **Conditioning vs encoding** - LatentVelo, scPRISMA support conditioning on confounders

### Future Improvements to Consider
1. **Hyperbolic transition space** - Distance from origin = progression stage naturally
2. **Gromov-Wasserstein fusion** - Principled way to fuse HLCA/LuCA heterogeneous spaces
3. **Hyperspherical latent** - Avoid crowding and component collapse
4. **Unbalanced OT** - Handle cell death/proliferation during progression
5. **Donor conditioning in decoder** - Learn donor-invariant representations

### Limitations to Acknowledge
1. **Large distribution gaps** - OT/flow matching struggles when stages are very different
2. **Identifiability** - Cross-sectional data limits what transitions we can claim
3. **Correlation ≠ causation** - Attention weights are associative, not causal

---

## References

1. Szalata et al. (2024). Transformers in single-cell omics: a review and new perspectives. Nature Methods.
2. Bunne et al. (2023). Learning single-cell perturbation responses using neural optimal transport. Nature Methods 20, 1759-1768.
3. Bunne et al. (2024). Optimal transport for single-cell and spatial omics. Nature Reviews Methods Primers 4:58.
4. Ding & Regev (2021). Deep generative model embedding of single-cell RNA-Seq profiles on hyperspheres and hyperbolic spaces. Nature Communications 12:2554.
