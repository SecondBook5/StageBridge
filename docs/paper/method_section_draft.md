# Method

## Problem Formulation

We addressed the task of learning progression-aware cell representations from cross-sectional single-nucleus RNA sequencing (snRNA-seq) data. Let $\mathcal{D} = \{(x_i, s_i, \mathcal{N}_i)\}_{i=1}^{N}$ denote a dataset of $N$ cells, where $x_i \in \mathbb{R}^{G}$ represented the gene expression profile over $G$ genes, $s_i \in \mathcal{S} = \{\text{Normal}, \text{AAH}, \text{AIS}, \text{MIA}, \text{LUAD}\}$ denoted the histopathological stage, and $\mathcal{N}_i = \{x_j : j \in \text{neighbors}(i)\}$ represented the local spatial neighborhood derived from matched spatial transcriptomics.

The central hypothesis motivating our approach was that cross-sectional progression becomes more identifiable when conditioned on receiver-centered local niche context. That is, a cell's transcriptional state and its position along the disease trajectory were jointly determined by intrinsic properties and extrinsic signals from the tumor microenvironment. We operationalized this through a hierarchical Set Transformer architecture that explicitly modeled the flow of information from niche context to receiver cell state.

## Dual-Reference Geometry

To anchor query cells within a biologically interpretable coordinate system, we employed dual-reference mapping against two complementary single-cell atlases: the Human Lung Cell Atlas (HLCA) representing healthy lung tissue, and the Lung Cancer Atlas (LuCA) capturing malignant and stromal heterogeneity.

For each reference atlas $\mathcal{R} \in \{\text{HLCA}, \text{LuCA}\}$, we leveraged pretrained scANVI models and applied scArches surgical fine-tuning to project query cells into the reference latent space. Let $z_i^{\text{HLCA}} \in \mathbb{R}^{30}$ and $z_i^{\text{LuCA}} \in \mathbb{R}^{10}$ denote the resulting latent embeddings. Each embedding was L2-normalized independently to prevent scale dominance, then concatenated to form the fused representation:

$$z_i^{\text{fused}} = \left[ \frac{z_i^{\text{HLCA}}}{\|z_i^{\text{HLCA}}\|_2} \;\Big\|\; \frac{z_i^{\text{LuCA}}}{\|z_i^{\text{LuCA}}\|_2} \right] \in \mathbb{R}^{40} \tag{1}$$

This dual-reference embedding captured complementary biological signals: proximity to healthy cell states (HLCA) and similarity to known cancer phenotypes (LuCA). Calibrated confidence scores were additionally computed via percentile-rank normalization of k-nearest neighbor distances, enabling downstream uncertainty quantification.

## Spatial Mapping Backends

To construct spatial neighborhoods for snRNA-seq cells, spatial transcriptomics data was leveraged to assign tissue coordinates. Three established spatial mapping backends were evaluated: Tangram, DestVI, and TACCO. Tangram formulated the mapping problem as optimal transport, minimizing the Wasserstein distance between single-cell and spatial expression distributions while respecting cell type proportions. DestVI employed a deep generative model with variational inference to jointly learn cell type deconvolution and spatial assignment. TACCO used a probabilistic framework incorporating spatial autocorrelation priors to encourage spatially coherent assignments.

For each snRNA-seq cell $i$, the selected backend produced a probability distribution $p(l | i)$ over spatial locations $l$. The maximum probability location was assigned as the cell's spatial coordinate, and neighbors were defined as cells within specified radial distances from this coordinate. The backend selection was determined empirically through benchmarking on held-out spatial spots with known cell type composition.

## Receiver-Centered Niche Encoder

The core architectural contribution was a hierarchical Set Transformer that modeled the receiver cell as attending to its local microenvironment. Unlike conventional graph neural networks that propagate messages symmetrically, our formulation explicitly designated a receiver cell whose state reconstruction depended on information aggregated from sender cells in the spatial niche.

### Token Construction

For each receiver cell $i$, a sequence of tokens $\mathcal{T}_i = \{t_0, t_1, \ldots, t_K\}$ was constructed. The first token $t_0 \in \mathbb{R}^{d}$ served as the receiver token, containing the fused dual-reference embedding of cell $i$. The subsequent tokens $t_1, \ldots, t_R$ were spatial ring tokens that aggregated statistics of neighbors at increasing radial distances, capturing the concentric organization of the tumor microenvironment. The remaining tokens $t_{R+1}, \ldots, t_K$ were context tokens encoding auxiliary information including pathway activity scores, cell type composition summaries, and reference mapping confidence values.

Each spatial ring token $t_r$ summarized neighbors within radial shell $[d_{r-1}, d_r)$ via learned pooling over cell type proportions, mean expression signatures, and spatial density.

### Induced Set Attention

Processing variable-cardinality neighbor sets required permutation-invariant operations. Standard self-attention over a set $X \in \mathbb{R}^{n \times d}$ of $n$ elements computes pairwise interactions with $O(n^2)$ complexity, which becomes prohibitive for large neighborhoods. To address this, we adopted the Induced Set Attention Block (ISAB) from the Set Transformer architecture.

The core building block is the Multihead Attention Block (MAB). Given queries $Q$, keys $K$, and values $V$, the MAB applies multihead attention followed by a feedforward network with residual connections and layer normalization:

$$H = \text{LayerNorm}(Q + \text{MultiHead}(Q, K, V)) \tag{2}$$

$$\text{MAB}(Q, K, V) = \text{LayerNorm}(H + \text{FFN}(H)) \tag{3}$$

The MultiHead attention mechanism with $h$ heads projects the inputs into $h$ separate subspaces, computes scaled dot-product attention in each, and concatenates the results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, \ldots, head_h) W^O \tag{4}$$

$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \tag{5}$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \tag{6}$$

where $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_k}$ are learned projection matrices for head $i$, $W^O \in \mathbb{R}^{hd_k \times d}$ is the output projection, and $d_k = d/h$ is the per-head dimension.

The ISAB reduces complexity by introducing $m$ learned inducing points $I \in \mathbb{R}^{m \times d}$ that serve as a communication bottleneck between input elements:

$$H = \text{MAB}(I, X, X) \tag{7}$$

$$\text{ISAB}_m(X) = \text{MAB}(X, H, H) \tag{8}$$

The first MAB (Equation 6) compresses the input set $X$ into $m$ inducing point representations $H \in \mathbb{R}^{m \times d}$ by having the inducing points attend to all input elements. The second MAB (Equation 7) allows each input element to attend to this compressed representation. This two-stage process yields $O(nm)$ complexity instead of $O(n^2)$, where typically $m \ll n$.

### Hierarchical Architecture

The niche encoder operated in two stages. In the first stage (intra-ring aggregation), an ISAB was applied independently to the set of neighbor embeddings within each spatial ring $r$, producing a fixed-dimensional ring summary:

$$h_r = \text{ISAB}_m(\{z_j^{\text{fused}} : j \in \text{ring}_r(i)\}) \tag{9}$$

In the second stage (cross-ring attention), the receiver token attended to all ring summaries and context tokens via cross-attention, enabling information flow from the spatial neighborhood to the receiver:

$$\tilde{t}_0 = t_0 + \text{MultiHead}(Q=t_0, K=V=[h_1, \ldots, h_R, t_{R+1}, \ldots, t_K]) \tag{10}$$

This formulation enabled the model to learn which spatial scales and niche components were most informative for predicting receiver state, effectively implementing a learned receptive field that adapted to local tissue architecture.

### Pooling by Multihead Attention

Following the Set Transformer paradigm, a fixed-dimensional niche representation was extracted via Pooling by Multihead Attention (PMA). Unlike mean or max pooling which discard relational information, PMA uses learned seed vectors to query the encoded set:

$$c_i = \text{PMA}_k(\tilde{\mathcal{T}}_i) = \text{MAB}(S, \tilde{\mathcal{T}}_i, \tilde{\mathcal{T}}_i) \tag{11}$$

where $S \in \mathbb{R}^{k \times d}$ denotes $k$ learned seed vectors that act as queries and $\tilde{\mathcal{T}}_i$ is the encoded token sequence after hierarchical processing. Each seed vector attends over all tokens and produces a $d$-dimensional summary, yielding output $c_i \in \mathbb{R}^{k \times d}$. When $k=1$, this reduces to a single context vector. The seed vectors were learned end-to-end, allowing the model to discover task-relevant aggregation patterns rather than relying on fixed pooling operations.

## Training Objectives

A multi-task self-supervised learning framework was employed with the following objectives:

### Primary: Masked Receiver Reconstruction (70%)

The principal objective was reconstructing the receiver cell's expression profile from niche context alone. The receiver token was masked and the model was trained to predict $\hat{x}_i$ given only the spatial neighborhood:

$$\mathcal{L}_{\text{mask}} = \mathbb{E}_{i} \left[ \| x_i - \hat{x}_i \|_2^2 \right] \tag{12}$$

This forced the model to extract progression-relevant signals from the microenvironment, directly operationalizing our central hypothesis.

### Auxiliary Objectives (30%)

To regularize representations and encourage biologically meaningful structure, four auxiliary objectives were incorporated. A contrastive ranking loss (10% weight) enforced that positive pairs from the same donor and proximal disease stages embedded closer together than negative pairs from distant stages or different donors. A cross-view consistency loss (10% weight) encouraged augmented views of the same spatial neighborhood to yield similar representations, promoting robustness to minor perturbations. A spatial coherence loss (5% weight) penalized large representation shifts under small spatial coordinate perturbations, ensuring smooth variation across tissue space. Finally, a biological group structure loss (5% weight) encouraged cells sharing the same annotated cell type to cluster together in the learned embedding space. The total loss was a weighted combination:

$$\mathcal{L}_{\text{SSL}} = 0.70 \cdot \mathcal{L}_{\text{mask}} + 0.10 \cdot \mathcal{L}_{\text{rank}} + 0.10 \cdot \mathcal{L}_{\text{consist}} + 0.05 \cdot \mathcal{L}_{\text{spatial}} + 0.05 \cdot \mathcal{L}_{\text{group}} \tag{13}$$

## Continuous Flow Matching with Optimal Transport

Following self-supervised pretraining, a transition model was trained to learn continuous dynamics connecting cell states across disease stages. The continuous flow matching (CFM) framework was adopted, which learned a time-dependent vector field $v_\theta(x, t)$ that transported samples from a source distribution to a target distribution along straight paths in latent space.

For adjacent disease stages $s$ and $s'$, optimal transport couplings were computed between the corresponding cell populations using the Sinkhorn algorithm with entropic regularization:

$$\pi^* = \argmin_{\pi \in \Pi(p_s, p_{s'})} \langle C, \pi \rangle - \epsilon H(\pi) \tag{14}$$

where $C_{ij} = \|z_i - z_j\|_2^2$ was the pairwise cost matrix between source and target cell embeddings, $\Pi(p_s, p_{s'})$ denoted the set of valid transport plans with marginals $p_s$ and $p_{s'}$, and $H(\pi)$ was the entropy regularization term with coefficient $\epsilon = 0.05$.

The vector field was parameterized as a neural network conditioned on the niche context representation $c_i$ from the pretrained encoder:

$$v_\theta(x_t, t, c) = \text{MLP}([x_t \| \phi(t) \| c]) \tag{15}$$

where $\phi(t)$ was a sinusoidal time embedding and $[\cdot \| \cdot]$ denoted concatenation. Training minimized the flow matching objective:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, (x_0, x_1) \sim \pi^*} \left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|_2^2 \right] \tag{16}$$

where $x_t = (1-t)x_0 + tx_1$ was the linear interpolant between coupled source and target cells. This formulation enabled simulation of disease progression trajectories by integrating the learned vector field forward in time, providing a generative model of how cell states evolved through the LUAD progression sequence.

## Baselines

To validate the architectural contributions, StageBridge was compared against a hierarchy of baselines that progressively added structural inductive biases. The simplest baseline, PoolingMLP, mean-pooled cell embeddings within a neighborhood and passed the result through a multilayer perceptron:

$$f_{\text{pool}}(\mathcal{N}) = \text{MLP}\left(\frac{1}{|\mathcal{N}|}\sum_{j \in \mathcal{N}} z_j\right) \tag{17}$$

This baseline captured no permutation structure or spatial information, serving as a lower bound on performance.

The DeepSets architecture achieved permutation invariance through the decomposition theorem, which states that any permutation-invariant function on sets can be expressed as $\rho(\sum_{x \in X} \phi(x))$ for suitable functions $\phi$ and $\rho$. The DeepSets baseline implemented this as:

$$f_{\text{DS}}(\mathcal{N}) = \rho\left(\sum_{j \in \mathcal{N}} \phi(z_j)\right) \tag{18}$$

where $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^h$ was an element-wise encoder network and $\rho: \mathbb{R}^h \rightarrow \mathbb{R}^d$ was a decoder network. This tested whether permutation invariance alone sufficed without attention mechanisms.

The SetTransformer baseline employed full self-attention over the cell set using stacked Self-Attention Blocks (SAB), defined as:

$$\text{SAB}(X) = \text{MAB}(X, X, X) \tag{19}$$

followed by PMA for pooling. Unlike ISAB, SAB computed all pairwise interactions with $O(n^2)$ complexity. This isolated the contribution of attention mechanisms from spatial organization, as no ring structure was imposed.

The GraphSAGE baseline constructed a spatial graph $G = (V, E)$ where nodes represented cells and edges connected spatially proximal cells within a specified radius. Message passing aggregated neighbor features symmetrically across layers:

$$h_v^{(l+1)} = \sigma\left(W^{(l)} \cdot \text{CONCAT}\left(h_v^{(l)}, \text{AGG}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)\right) \tag{20}$$

where AGG denoted mean aggregation and $\sigma$ was a nonlinearity. Unlike the receiver-centered formulation, GraphSAGE propagated information bidirectionally between all connected nodes, testing whether symmetric message passing matched asymmetric receiver-centered attention.

The full StageBridge model combined receiver-centered niche encoding, dual-reference geometry, hierarchical Set Transformer architecture with ISAB and PMA, and the multi-task self-supervised objective described above.

## Ablation Studies

Systematic ablations were conducted to quantify the contribution of each architectural component. Reference ablations compared single-reference mapping (HLCA-only or LuCA-only) against the full dual-reference geometry to assess whether both healthy and disease anchors were necessary. Spatial ablations removed the ring structure entirely, applying flat attention over all neighbors regardless of distance, thereby isolating the contribution of explicit spatial organization. Receiver ablations replaced the asymmetric receiver-centered attention with symmetric bidirectional message passing to test whether the directional information flow was essential. Loss ablations systematically removed each auxiliary objective individually while maintaining the primary masked reconstruction loss, quantifying the regularization benefit of each term. Finally, scale ablations varied the number of spatial rings (2, 4, and 6) and their radii to determine optimal spatial resolution for capturing microenvironment structure.

## Evaluation Metrics

Model performance was assessed across multiple complementary dimensions. Reconstruction quality was measured via mean squared error (MSE) between predicted and ground-truth receiver expression profiles under masking. Representation quality was evaluated using silhouette score, adjusted Rand index (ARI), and normalized mutual information (NMI) to assess whether learned embeddings preserved cell type structure. Progression discrimination was quantified via area under the ROC curve (AUROC) for stage classification tasks using the learned representations as input features. Batch integration quality was assessed using the k-nearest neighbor batch entropy test (kBET) and local inverse Simpson's index (LISI) to ensure representations were not confounded by technical batch effects. Biological conservation was evaluated by measuring marker gene expression preservation and correlation of pathway activity scores between original and reconstructed profiles.

## Implementation Details

The model was implemented in PyTorch with the following specifications:

| Hyperparameter | Value |
|----------------|-------|
| Hidden dimension $d$ | 128 |
| Transformer layers | 2 |
| Attention heads | 8 |
| Inducing points $m$ | 16 |
| PMA seed vectors $k$ | 1 |
| Spatial rings $R$ | 4 |
| Ring radii (μm) | [0, 50), [50, 100), [100, 150), [150, 200) |
| Batch size | 32 neighborhoods |
| Optimizer | AdamW (lr=1e-3, weight decay=1e-4) |
| Training epochs | 150 with early stopping |

Training employed donor-held-out cross-validation to prevent leakage: all cells from a given patient appeared exclusively in train or validation splits. This rigorous evaluation protocol ensured that learned representations generalized across individuals rather than memorizing donor-specific patterns.

All experiments were conducted on NVIDIA H100 GPUs. Reference mapping via scArches surgery required approximately 4 hours for 787,709 query cells against both HLCA and LuCA atlases.
