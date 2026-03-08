# StageBridge: Transformer-Based Context-Aware Modeling of Tumor Stage Progression from Single-Cell and Spatial Transcriptomics

## Abstract

Lung adenocarcinoma (LUAD) evolves through a stereotyped histological progression from normal alveolar tissue through atypical adenomatous hyperplasia (AAH), adenocarcinoma in situ (AIS), and minimally invasive adenocarcinoma (MIA) to invasive LUAD. Standard computational approaches model this trajectory using cell-intrinsic gene expression alone, ignoring the local tissue microenvironment that shapes transition dynamics. We present **StageBridge**, a transformer-based framework that encodes typed local niche context from spatially resolved transcriptomics and uses the resulting context embeddings to condition an optimal-transport-based transition model between disease stages. A Set Transformer processes unordered sets of typed niche tokens (epithelial, stromal, immune, vascular) via induced set attention, self-attention, and pooling-by-multihead-attention to produce a fixed-size context vector. This context conditions a drift network that predicts velocity fields in a reference latent space, with cell pairings determined by entropic optimal transport (Sinkhorn) coupling. In donor-held-out evaluation on two clinically relevant transition edges, Set Transformer context outperforms pooled baselines on both edges and outperforms RNA-only baselines on the invasive AIS-to-MIA transition (Sinkhorn divergence 15.76 vs. 16.30), supporting the hypothesis that local niche composition carries information relevant to stage-transition modeling. An optional Graph Transformer extension for tissue-level context was evaluated but did not consistently improve over local set encoding. The framework is implemented as a modular Python package with reproducible configuration, evaluation, and visualization.

## 1. Introduction

Lung adenocarcinoma is the most common subtype of lung cancer and the leading cause of cancer mortality worldwide. It follows a well-characterized histological progression: normal alveolar epithelium transforms to AAH (atypical adenomatous hyperplasia), progresses through AIS (adenocarcinoma in situ) and MIA (minimally invasive adenocarcinoma), and culminates in fully invasive LUAD [14]. Understanding the molecular mechanisms driving each transition is critical for early detection and intervention.

Recent advances in single-cell RNA sequencing (snRNA-seq) and spatial transcriptomics have generated rich, multi-modal views of tumors at each stage. The GSE308103 dataset provides snRNA-seq profiles across all five stages, while GSE307534 provides matched 10x Visium spatial transcriptomics [14]. These paired datasets enable, for the first time, a unified computational framework that considers both cell-intrinsic transcriptomic states and spatially resolved tissue context when modeling stage transitions.

**The machine learning problem.** We formulate stage-to-stage tumor progression as a context-aware optimal transport problem: given cells at stage $s$ and cells at stage $s+1$, learn a velocity field that transports the source distribution to the target distribution, conditioned on the local tissue niche composition. This framing captures the biological insight that transitions are shaped not only by cell-autonomous programs but also by signals from the surrounding microenvironment.

**Why transformers.** The local tissue niche around each cell is naturally represented as an unordered set of typed tokens: epithelial cell proportions, stromal features, immune infiltration levels, and vascular programs. This structure demands a permutation-invariant encoder. The Set Transformer [1] provides exactly this: attention-based aggregation over variable-size sets with learned importance weighting, producing a compact context vector that captures the relative contributions of different niche components.

**Hypotheses.** This project tests three hypotheses:

1. **H1**: Transformer-based local niche encoding (Set Transformer) improves stage-transition modeling beyond RNA-only baselines that ignore context entirely.
2. **H2**: Set Transformer context outperforms non-transformer pooled context (mean/std/max aggregation), demonstrating that attention-based weighting captures niche-relevant information that simple statistics miss.
3. **H3**: Tissue-level graph context (Graph Transformer) may provide additional benefit for transitions involving broad microenvironment remodeling, but this must be empirically justified rather than assumed.

## 2. Related Work

### 2.1 Set Transformer

Lee et al. [1] introduced the Set Transformer for permutation-invariant processing of set-structured inputs. The architecture uses Induced Set Attention Blocks (ISAB) to reduce the quadratic attention cost to linear complexity via learned inducing points, followed by Self-Attention Blocks (SAB) and Pooling by Multihead Attention (PMA) to produce fixed-size set representations. Deep Sets [16] provides a simpler permutation-invariant baseline via element-wise transformations followed by sum/mean pooling. StageBridge uses both: the Set Transformer as the primary context encoder and Deep Sets / pooled variants as baselines.

### 2.2 Graph Transformers

Graph Transformers extend the transformer architecture to graph-structured data by incorporating edge information into the attention mechanism [2, 3]. Dwivedi and Bresson [2] proposed positional encodings based on graph Laplacian eigenvectors, while Ying et al. [3] demonstrated that with sufficient architectural modifications, transformers can match or exceed message-passing GNNs on graph benchmarks. In StageBridge, an optional Graph-of-Sets Transformer encodes tissue-level context by treating (patient, stage) cell sets as graph nodes with learned edge-type biases.

### 2.3 Spatial Deconvolution

Spatial transcriptomics measures gene expression at tissue locations (spots) rather than individual cells. Deconvolution methods estimate cell-type compositions per spot by integrating snRNA-seq references. Tangram [4] uses optimal transport to map single-cell profiles onto spatial coordinates. TACCO [18] uses transfer learning for annotation and composition estimation. DestVI [6] uses a deep generative model to identify continuous cell-type variation. StageBridge uses these methods as upstream spatial mapping providers, converting their spot-level composition estimates into typed niche tokens that feed the Set Transformer.

### 2.4 Trajectory and Transport Modeling

Several methods model cellular dynamics using optimal transport. Waddington-OT [11] uses unbalanced OT to reconstruct developmental trajectories from time-series scRNA-seq. TrajectoryNet [8] learns continuous normalizing flows conditioned on time. CellOT [9] learns perturbation responses via neural OT maps. Conditional flow matching (OT-CFM) [10] provides simulation-free training of velocity fields with OT-coupled pairs. The Schrodinger bridge formulation [12, 13] extends OT-CFM with entropic regularization, naturally handling biological noise and population heterogeneity.

StageBridge differs from these approaches in two key ways: (1) it conditions the velocity field on a learned Set Transformer context vector from the local tissue niche, rather than treating transport as context-free, and (2) it operates on discrete disease stages rather than continuous developmental time, using stage-specific edge conditioning.

## 3. Research Project Problem

### 3.1 Problem Statement

**Inputs:**
- snRNA-seq latent representations $\mathbf{x}_i \in \mathbb{R}^d$ for cells at each disease stage, embedded via HLCA reference mapping [15] into a 32-dimensional latent space
- Spatial mapping compositions $\mathbf{s}_j \in \mathbb{R}^K$ from Tangram [4], estimating cell-type proportions at each Visium spot
- Typed niche tokens $\{\mathbf{t}_1, \ldots, \mathbf{t}_N\}$, where each token aggregates spatial composition features into one of four biological categories: epithelial, stromal, immune, vascular

**Outputs:**
- A context vector $\mathbf{c}_s = f_\theta(\{\mathbf{t}_1, \ldots, \mathbf{t}_N\})$ produced by the Set Transformer
- A velocity field $\mathbf{v}_\phi(\mathbf{x}_t, t, \mathbf{c}_s, e)$ that transports cells from stage $s$ to stage $s+1$
- Predicted target-stage cell distributions for held-out donors

**Evaluation:**
- Sinkhorn divergence between predicted and observed target distributions (primary metric, lower = better)
- Classifier AUC (real vs. predicted discrimination)
- Calibration error (systematic shifts in predicted means)

### 3.2 Hypotheses

**H1 (Context vs. no context):** Conditioning the transition model on Set Transformer niche context reduces Sinkhorn divergence compared to RNA-only baselines on at least one transition edge.

**H2 (Attention vs. pooling):** Set Transformer context achieves lower Sinkhorn divergence than pooled context (mean/std/max aggregation) on both transition edges, demonstrating that learned attention weights capture niche-relevant patterns that simple statistics miss.

**H3 (Graph context):** Adding tissue-level Graph Transformer context on top of local Set Transformer context may improve transition modeling for edges involving significant microenvironment remodeling, but this benefit is not guaranteed and must be empirically validated.

### 3.3 Why a Transformer-Based Niche Encoder Is Central

The core architectural claim of StageBridge is that the Set Transformer is the right inductive bias for niche context encoding:

1. **Permutation invariance**: The niche around a cell has no natural ordering. Mean pooling is permutation-invariant but discards relational information. The Set Transformer preserves interactions between token types.
2. **Adaptive weighting**: Different transitions may depend on different niche features. Self-attention allows the model to learn edge-specific importance patterns (e.g., immune infiltration may matter more for AIS-to-MIA than for AAH-to-AIS).
3. **Scalability**: The ISAB reduces attention complexity from $O(N^2)$ to $O(N \cdot M)$ where $M$ is the number of inducing points, enabling efficient processing of large niche token sets.

## 4. Method

### 4.1 Data Representation

StageBridge operates on two GEO datasets from Hu et al. [14]:

- **GSE308103** (snRNA-seq): Single-nucleus RNA sequencing across five LUAD progression stages (Normal, AAH, AIS, MIA, LUAD) from multiple donors. Each cell is represented as a gene expression profile.
- **GSE307534** (10x Visium): Spatial transcriptomics providing tissue-level gene expression at ~55 $\mu$m resolution, with spatial coordinates for each measurement spot.

All cells are embedded into a shared 32-dimensional latent space via reference mapping to the Human Lung Cell Atlas (HLCA) [15, 17]. When the full HLCA reference (~20 GB) is unavailable, a PCA fallback provides an alternative latent representation.

### 4.2 Typed Niche Token Construction

Spatial mapping (via Tangram [4]) produces a cell-to-spot assignment matrix estimating cell-type composition at each Visium spot. These raw composition features are grouped into four biologically meaningful typed token categories:

- **Epithelial tokens**: AT2, basal, ciliated, secretory cell proportions
- **Stromal tokens**: Fibroblast, stromal cell proportions
- **Immune tokens**: Macrophage, mast cell, T cell proportions
- **Vascular tokens**: Endothelial and vascular program features

For each cell, the typed tokens from its assigned spatial neighborhood form an unordered set $\mathcal{T} = \{\mathbf{t}_1, \ldots, \mathbf{t}_N\}$ where $\mathbf{t}_i \in \mathbb{R}^{d_\text{token}}$.

### 4.3 Set Transformer Context Encoder

The Set Transformer processes the typed niche token set through three stages:

**Input projection.** Each token is linearly projected to a hidden dimension $d_h$:
$$\mathbf{h}_i = W_{\text{proj}} \mathbf{t}_i + \mathbf{b}_{\text{proj}}, \quad \mathbf{h}_i \in \mathbb{R}^{d_h}$$

**Induced Set Attention Block (ISAB).** A set of $M$ learned inducing points $\mathbf{I} \in \mathbb{R}^{M \times d_h}$ compresses the input set:
$$\mathbf{H} = \text{MHA}(\mathbf{I}, \mathbf{X}, \mathbf{X}) \quad \text{(inducing points attend to inputs)}$$
$$\mathbf{X}' = \text{MHA}(\mathbf{X}, \mathbf{H}, \mathbf{H}) \quad \text{(inputs attend to compressed representation)}$$

Both steps include layer normalization and feed-forward blocks with residual connections. An optional spatial relative position encoding (SpatialRPE) adds distance-based attention bias when spatial coordinates are available.

**Self-Attention Block (SAB).** Standard multi-head self-attention refines the token representations:
$$\mathbf{X}'' = \text{LayerNorm}(\mathbf{X}' + \text{MHA}(\mathbf{X}', \mathbf{X}', \mathbf{X}'))$$
$$\mathbf{X}''' = \text{LayerNorm}(\mathbf{X}'' + \text{FFN}(\mathbf{X}''))$$

**Pooling by Multihead Attention (PMA).** A learned seed vector $\mathbf{s} \in \mathbb{R}^{1 \times d_h}$ attends to the processed tokens to produce a single context vector:
$$\mathbf{c}_s = \text{MHA}(\mathbf{s}, \mathbf{X}''', \mathbf{X}''') \in \mathbb{R}^{d_h}$$

A final linear + GELU + LayerNorm head produces the output context embedding.

### 4.4 Optional Graph Transformer Context Encoder

An optional Graph-of-Sets Transformer adds tissue-level context by treating (patient, stage) cell sets as nodes in a graph. Each node's features are the PMA output from the local Set Transformer. The graph has three edge types:

- **Type 0**: Stage-adjacent edges (e.g., AAH-AIS)
- **Type 1**: Same-patient cross-stage edges
- **Type 2**: Same-stage cross-patient edges

Graph attention layers with per-edge-type learned biases propagate information across the tissue graph. The enriched node representations can replace or augment the local context vectors.

### 4.5 Transition Model

The transition model learns to transport cell distributions from stage $s$ to stage $s+1$.

**Entropic OT coupling.** Source cells $\{\mathbf{x}_i^{(s)}\}$ and target cells $\{\mathbf{x}_j^{(s+1)}\}$ are paired via Sinkhorn's algorithm [19] with entropic regularization $\varepsilon$:
$$\pi^* = \arg\min_{\pi \in U(\mathbf{a}, \mathbf{b})} \langle \pi, C \rangle - \varepsilon H(\pi)$$

where $C_{ij} = \|\mathbf{x}_i^{(s)} - \mathbf{x}_j^{(s+1)}\|^2$ is the squared Euclidean cost in latent space and $H(\pi)$ is the entropy of the coupling.

**Schrodinger bridge interpolant.** Training samples are generated along interpolation paths between coupled pairs:
$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1 + \sigma\sqrt{t(1-t)}\boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

The target drift velocity for training is:
$$\mathbf{u}_t = (\mathbf{x}_1 - \mathbf{x}_0) + \sigma^2 \frac{1 - 2t}{2t(1-t)} \sigma\sqrt{t(1-t)}\boldsymbol{\varepsilon}$$

For $\sigma \to 0$, this reduces to the deterministic OT-CFM target $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$ [10].

**Context-conditioned drift network.** The drift network predicts the velocity field conditioned on position, time, context, and edge identity:
$$\hat{\mathbf{v}} = f_\phi(\mathbf{x}_t, \text{SinEmb}(t), \mathbf{c}_s, \text{EdgeEmb}(e))$$

where $\text{SinEmb}$ is a sinusoidal time embedding and $\text{EdgeEmb}$ is a learned edge-type embedding. Context conditioning uses either concatenation or FiLM (Feature-wise Linear Modulation) [Perez et al. 2018]:
$$\text{FiLM}(\mathbf{x}, \mathbf{c}) = (1 + 0.1 \tanh(\gamma(\mathbf{c}))) \odot \mathbf{x} + \beta(\mathbf{c})$$

**Training objective.** The primary loss is drift matching:
$$\mathcal{L}_{\text{drift}} = \mathbb{E}_{t, (i,j) \sim \pi^*} \left[ \|\hat{\mathbf{v}}(\mathbf{x}_t, t, \mathbf{c}_s, e) - \mathbf{u}_t\|^2 \right]$$

An optional score matching term enables backward SDE inference:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{drift}} + \lambda_{\text{score}} \mathcal{L}_{\text{score}}$$

**Inference.** Euler integration propagates source cells to predicted target distributions:
$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \Delta t \cdot \hat{\mathbf{v}}(\mathbf{x}_t, t, \mathbf{c}_s, e)$$

### 4.6 Evaluation

Three complementary metrics evaluate transition quality on held-out donors:

1. **Sinkhorn divergence** (primary): Entropic OT distance between predicted and observed target distributions. Lower values indicate better distributional match.
2. **Classifier AUC**: A logistic regression classifier trained to distinguish real target cells from predicted target cells. AUC near 0.5 indicates the model produces indistinguishable distributions.
3. **Calibration error**: Mean absolute difference in per-dimension means between predicted and observed targets. Detects systematic directional biases.

Evaluation uses a donor-holdout split: all cells from held-out donors are excluded from training and used only for evaluation, preventing data leakage.

### 4.7 Baselines

Four context-encoding modes are compared under matched conditions:

| Mode | Transformer? | Context Source | Architecture |
|------|-------------|---------------|-------------|
| `rna_only` | No | None | Cell latent only |
| `pooled` | No | Typed tokens | Mean/std/max + MLP |
| `set_only` | Yes | Typed tokens | ISAB + SAB + PMA |
| `graph_of_sets` | Yes | Typed tokens + graph | Set Transformer + Graph Transformer |

All modes share the same transition model architecture and training procedure. Only the context encoder differs.

## 5. Experimental Plan (Draft)

### 5.1 Core Experiments

The primary comparison evaluates all four context modes on two transition edges:
- **AAH -> AIS**: Early preneoplastic transition
- **AIS -> MIA**: Transition to invasive adenocarcinoma (clinically most significant)

Each experiment uses donor-holdout evaluation with Tangram spatial mapping and no WES regularization or state-dependent diffusion.

### 5.2 Extension Experiments (Planned)

- WES regularization: constraining transport with whole-exome sequencing mutation features
- State-dependent diffusion: learning position-dependent noise schedules
- Full 5-stage chain evaluation
- Brain metastasis extension (GSE223499)

## 6. Preliminary Results (Draft)

### 6.1 Core Mode Comparison

Results from matched context-mode comparisons (see Figure 2, Table 1):

| Edge | RNA-only | Pooled | Set Transformer | Graph-of-Sets |
|------|----------|--------|----------------|--------------|
| AAH->AIS | **17.25** | 18.10 | 17.82 | 18.68 |
| AIS->MIA | 16.30 | 15.91 | **15.76** | 16.00 |

*Sinkhorn divergence (lower = better). Bold = best per edge.*

### 6.2 Key Findings

1. **H2 supported on both edges**: Set Transformer outperforms pooled context (17.82 < 18.10 on AAH->AIS; 15.76 < 15.91 on AIS->MIA), confirming that attention-based set encoding captures information beyond simple pooling statistics.

2. **H1 partially supported**: Set Transformer outperforms RNA-only on AIS->MIA (15.76 < 16.30) but not on AAH->AIS (17.82 > 17.25). This suggests local niche context is most valuable for transitions involving microenvironment remodeling.

3. **H3 not currently supported**: Graph-of-Sets does not outperform Set Transformer on either edge (18.68 > 17.82; 16.00 > 15.76). Tissue-level graph context does not yet add value beyond local niche encoding.

### 6.3 Extension Status

- **WES regularization**: Mixed evidence. Slight improvement on AAH->AIS, no improvement on AIS->MIA.
- **State-dependent diffusion**: Mixed evidence. No consistent improvement over drift-only.
- **Graph-of-Sets**: Does not earn flagship status. Retained as optional extension.

## 7. Planned Statistical Evaluation (Draft)

- Bootstrap confidence intervals on Sinkhorn divergence (1000 resamples)
- Wilcoxon signed-rank tests for paired mode comparisons across donors
- Replication across additional stage edges (Normal->AAH, MIA->LUAD)
- Sensitivity analysis: latent backend (HLCA vs. PCA), spatial provider (Tangram vs. TACCO vs. DestVI)

## 8. Conclusion (Draft)

StageBridge demonstrates that transformer-based encoding of local tissue niche context can improve stage-transition modeling in LUAD progression. The Set Transformer, applied to typed spatial tokens representing epithelial, stromal, immune, and vascular niche components, consistently outperforms non-transformer pooled baselines and achieves the best performance on the clinically significant AIS-to-MIA invasive transition.

The framework's modular design separates context encoding, spatial mapping, and transition modeling into independently testable components. Current limitations include the lack of consistent improvement over RNA-only baselines on all edges and the failure of Graph Transformer tissue-level context to earn flagship status. Future work will focus on larger-scale evaluation across the full stage chain, improved graph construction strategies, and validation on independent cohorts.

## References

[1] Lee, J. et al. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Input. ICML 2019.

[2] Dwivedi, V. P. & Bresson, X. (2020). A Generalization of Transformer Networks to Graphs. AAAI 2021 Workshop.

[3] Ying, C. et al. (2021). Do Transformers Really Perform Bad for Graph Representation? NeurIPS 2021.

[4] Biancalani, T. et al. (2021). Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram. Nature Methods, 18(11), 1352-1362.

[6] Lopez, R. et al. (2022). DestVI identifies continuums of cell types in spatial transcriptomics data. Nature Biotechnology, 40(9), 1360-1369.

[8] Tong, A. et al. (2020). TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics. ICML 2020.

[9] Bunne, C. et al. (2023). Learning single-cell perturbation responses using neural optimal transport. Nature Methods, 20(11), 1747-1756.

[10] Tong, A. et al. (2024). Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport. ICML 2023.

[11] Schiebinger, G. et al. (2019). Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming. Cell, 176(4), 928-943.

[12] De Bortoli, V. et al. (2021). Diffusion Schrodinger Bridge. NeurIPS 2021.

[13] Shi, Y. et al. (2024). Diffusion Schrodinger Bridge Matching. ICLR 2024.

[14] Hu, X. et al. (2023). Single-cell and spatial transcriptomics reveal the progression of lung adenocarcinoma. GEO Datasets GSE308103, GSE307534, GSE307529.

[15] Sikkema, L. et al. (2023). An integrated cell atlas of the lung in health and disease. Nature Medicine, 29(6), 1563-1577.

[16] Zaheer, M. et al. (2017). Deep Sets. NeurIPS 2017.

[17] Lotfollahi, M. et al. (2022). Mapping single-cell data to reference atlases by transfer learning. Nature Biotechnology, 40(1), 121-130.

[18] Lottaz, M. et al. (2023). TACCO: unified annotation transfer and decomposition of cell identities. Nature Biotechnology.

[19] Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS 2013.

[20] Vaswani, A. et al. (2017). Attention is all you need. NeurIPS 2017.
