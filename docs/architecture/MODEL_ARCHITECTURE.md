# StageBridge Model Architecture

**A Receiver-Centered Niche Encoding Model for Disease Stage Transitions**

This document provides a complete mathematical and architectural specification of StageBridge, a deep learning model that learns to predict how cells transition between disease stages by conditioning on their local spatial neighborhood (niche).

---

## Table of Contents

1. [Overview](#overview)
2. [Scientific Foundations](#scientific-foundations)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Input Representation](#input-representation)
5. [Architecture Components](#architecture-components)
   - [Receiver-Centered Niche Encoder](#component-1-receiver-centered-niche-encoder)
   - [Context Refiner (SAB)](#component-2-context-refiner-sab)
   - [Hierarchical Aggregator (ISAB + PMA)](#component-3-hierarchical-aggregator-isab--pma)
   - [Cross-Attention Drift Head](#component-4-cross-attention-drift-head)
   - [UDE Mode (Optional)](#component-5-ude-mode-optional)
   - [Evolution Branch (WES)](#component-6-evolution-branch-wes)
   - [Sample-Level Heads](#component-7-sample-level-heads)
   - [Auxiliary Biological Heads](#component-8-auxiliary-biological-heads)
6. [Training: OT-CFM Flow Matching](#training-ot-cfm-flow-matching)
7. [Inference](#inference)
8. [Hyperparameters](#hyperparameters)
9. [Ablation Studies](#ablation-studies)

---

## Overview

StageBridge learns to predict how cells transition between disease stages (e.g., AAH -> AIS -> MIA -> LUAD) by conditioning on the local spatial neighborhood (niche) around each cell. The key insight is **receiver-centering**: the focal cell receives signals from its neighbors, so it should be the query in attention, with neighbors as keys/values.

```
                              StageBridge Full Architecture
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│  INPUT: Receiver + 8 Neighbors + Distances + (optional) WES Features                   │
│  ────────────────────────────────────────────────────────────────────                   │
│                              │                                                          │
│                              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    DUAL-REFERENCE EMBEDDING (40d)                                │   │
│  │       HLCA (30d, healthy lung)  +  LuCA (10d, lung cancer)                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                          │
│                              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │              RECEIVER-CENTERED NICHE ENCODER                                     │   │
│  │                                                                                  │   │
│  │   Receiver ──Q──▶ ┌──────────────────────────┐                                  │   │
│  │                   │  Cross-Attention x L     │ ◀── Distance Bias (RBF)          │   │
│  │   Neighbors ─K,V─▶│  + Token Type Embeddings │                                  │   │
│  │                   └──────────────────────────┘                                  │   │
│  │                              │                                                   │   │
│  │                   Context + Context Tokens [B, 9, H]                            │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                          │
│                              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │              CONTEXT REFINER (SetTransformer SAB layers)                         │   │
│  │                                                                                  │   │
│  │   Context Tokens ──▶ [SAB x num_refiner_layers] ──▶ Refined Tokens              │   │
│  │                                                                                  │   │
│  │   Allows tokens to interact and share information before downstream use         │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                          │
│          ┌───────────────────┴───────────────────┐                                     │
│          │                                       │                                     │
│          ▼                                       ▼                                     │
│  ┌───────────────────────────┐     ┌─────────────────────────────────────────────┐    │
│  │  CELL-LEVEL PATH          │     │  SAMPLE-LEVEL PATH                          │    │
│  │                           │     │                                             │    │
│  │  ┌─────────────────────┐  │     │  ┌─────────────────────────────────────┐   │    │
│  │  │ CROSS-ATTENTION     │  │     │  │ HIERARCHICAL AGGREGATOR              │   │    │
│  │  │ DRIFT HEAD          │  │     │  │                                     │   │    │
│  │  │                     │  │     │  │ Niche Embeddings ──▶ [ISAB x L]     │   │    │
│  │  │ x_t + t ──Q──▶ MHA  │  │     │  │                       │             │   │    │
│  │  │ Ctx Tokens ─K,V─▶   │  │     │  │                       ▼             │   │    │
│  │  │        │            │  │     │  │                     [PMA]           │   │    │
│  │  │        ▼            │  │     │  │                       │             │   │    │
│  │  │  Gated Mixture      │  │     │  │            Sample Embedding [B, H]  │   │    │
│  │  │  g*ctx + (1-g)*lat  │  │     │  └─────────────────────────────────────┘   │    │
│  │  └─────────────────────┘  │     │                    │                        │    │
│  │           │               │     │                    ▼                        │    │
│  │           ▼               │     │  ┌─────────────────────────────────────┐   │    │
│  │  Drift Velocity v(x,t)    │     │  │ SAMPLE-LEVEL HEADS                  │   │    │
│  │  [B, 40]                  │     │  │                                     │   │    │
│  │                           │     │  │ Stage Logits [B, num_stages]        │   │    │
│  │  (Optional UDE mode:      │     │  │ Displacement [B, H]                 │   │    │
│  │   blend baseline+learned) │     │  └─────────────────────────────────────┘   │    │
│  └───────────────────────────┘     └─────────────────────────────────────────────┘    │
│                                                                                         │
│  (Optional) EVOLUTION BRANCH: WES features ──▶ Gated fusion with context               │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Scientific Foundations

### Central Hypothesis

**Cross-sectional progression becomes more identifiable when conditioned on receiver-centered local niche context.**

This means:
1. A cell's trajectory through disease stages depends on signals from its microenvironment
2. The direction of signaling matters: we model what the focal cell *receives*, not what it sends
3. By encoding the receiver-niche relationship, we can better predict and understand progression

### Biological Grounding

| Component | Biological Rationale |
|-----------|---------------------|
| **Receiver-centering** | Cells respond to signals from their environment; the focal cell integrates inputs |
| **Dual-reference embeddings** | Position cells relative to both healthy (HLCA) and disease (LuCA) states |
| **Distance-weighted attention** | Spatial proximity matters for paracrine signaling (cytokines, growth factors) |
| **Gated context/latent** | Some transitions are cell-autonomous, others depend on niche |
| **Token type embeddings** | Distinguish semantic roles (receiver vs. ring vs. reference token) |

### Target Biological Signals (Validation)

From Peng/Kadara literature:
- KAC/reactive pneumocyte-like alveolar progenitors as LUAD predecessors
- Epithelial-proinflammatory niches with IL1B-high macrophages
- IL1B-IL1R1 signaling axis
- These niches more common in AAH/AIS than LUAD (progression window)

---

## Mathematical Foundations

### Problem Setting

Let $\mathcal{X} \subset \mathbb{R}^D$ be the cell state space (where $D = 40$ for dual-reference embeddings). We observe cells from $S$ disease stages, denoted $\{s_1, s_2, \ldots, s_S\}$ (e.g., AAH, AIS, MIA, LUAD).

**Goal**: Learn a conditional velocity field $v_\theta: \mathcal{X} \times [0,1] \times \mathcal{C} \rightarrow \mathbb{R}^D$ that transports cells from stage $s_i$ to stage $s_j$, conditioned on local niche context $c \in \mathcal{C}$.

### Continuous Normalizing Flow Formulation

The stage transition is modeled as an ordinary differential equation (ODE):

$$\frac{dx_t}{dt} = v_\theta(x_t, t, c, e_{ij})$$

where:
- $x_t \in \mathbb{R}^D$ is the cell state at time $t \in [0, 1]$
- $c \in \mathbb{R}^H$ is the niche context encoding
- $e_{ij}$ is the stage transition embedding ($s_i \rightarrow s_j$)

### Conditional Flow Matching Objective

Following Lipman et al. (2023), we train by regressing the conditional velocity field:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t \sim U[0,1], (x_0, x_1) \sim \pi} \left[ \|v_\theta(x_t, t, c) - u_t(x_0, x_1)\|^2 \right]$$

where $x_t = (1 - t) x_0 + t x_1$ and $u_t = x_1 - x_0$.

### Optimal Transport Coupling

We use **entropic optimal transport** coupling:

$$\pi^* = \arg\min_{\pi \in \Pi(p_0, p_1)} \int \|x - y\|^2 \, d\pi(x, y) + \varepsilon H(\pi)$$

Solved via the Sinkhorn algorithm in log-space for numerical stability.

---

## Input Representation

### Dual-Reference Cell Embeddings

Each cell is embedded into a 40-dimensional space:

| Reference | Dimension | Description |
|-----------|-----------|-------------|
| HLCA | 30d | Healthy Lung Cell Atlas - positions cell relative to normal tissue |
| LuCA | 10d | Lung Cancer Atlas - positions cell relative to disease states |

### 9-Token Niche Structure

```
Token Index    Type              Description
──────────────────────────────────────────────────────────────
    0          Receiver          Focal cell (query source)
    1          Ring 1            Nearest neighbor
    2          Ring 2            2nd nearest
    3          Ring 3            3rd nearest
    4          Ring 4            4th nearest
    5          HLCA Token        Aggregated HLCA-mapped neighbors
    6          LuCA Token        Aggregated LuCA-mapped neighbors
    7          Pathway Token     Gene pathway activity summary
    8          Stats Token       Neighborhood statistics
```

---

## Architecture Components

### Component 1: Receiver-Centered Niche Encoder

The encoder transforms the 9-token neighborhood into a context representation. The key architectural choice is **receiver-centering**: the focal cell is always the attention query.

**Mathematical Formulation:**

$$Q = h_r W_Q \in \mathbb{R}^{1 \times H}$$
$$K = H_n W_K \in \mathbb{R}^{K \times H}$$
$$V = H_n W_V \in \mathbb{R}^{K \times H}$$

$$\text{Attention}(Q, K, V, d) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + b_{\text{dist}}(d)\right) V$$

The distance bias $b_{\text{dist}}$ encodes spatial relationships via RBF encoding:

$$\phi_{\text{RBF}}(d) = \left[ \exp\left(-\frac{(d - \mu_k)^2}{2\sigma^2}\right) \right]_{k=1}^{K}$$

**Output:**
- `context`: [B, H] - pooled niche representation
- `context_tokens`: [B, 9, H] - individual token representations
- `attention_weights`: [B, 8] - interpretable neighbor importance

---

### Component 2: Context Refiner (SAB)

Self-Attention Blocks (SAB) allow context tokens to interact and refine each other.

```python
class SetTransformerRefiner(nn.Module):
    """Refine context tokens via self-attention."""
    
    def __init__(self, dim, num_layers, num_heads, dropout):
        self.layers = nn.ModuleList([
            SAB(dim=dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, tokens, mask=None):
        h = tokens
        for layer in self.layers:
            h = layer(h, mask=mask)
        return h
```

**Scientific rationale:** After the receiver-centered encoding, tokens still represent individual entities. SAB layers let them share information (e.g., the HLCA token learns from the spatial ring tokens) before being used downstream.

---

### Component 3: Hierarchical Aggregator (ISAB + PMA)

For **sample-level** predictions (e.g., classifying a lesion rather than a cell), we aggregate multiple niche embeddings.

```
                    Hierarchical Aggregator
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Input: niche_embeddings [B, N, H]  (N niches per sample)       │
│                                                                   │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  ISAB Layer (Induced Set Attention Block)               │    │
│   │                                                         │    │
│   │  Input ──▶ [Cross-Attn to Inducing] ──▶ [Cross-Attn back]   │
│   │                                                         │    │
│   │  Complexity: O(N * M) instead of O(N^2)                │    │
│   │  (M = num_inducing_points, typically 16)               │    │
│   └─────────────────────────────────────────────────────────┘    │
│                              │                                    │
│                         (repeat L times)                          │
│                              │                                    │
│                              ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  PMA (Pooling by Multihead Attention)                   │    │
│   │                                                         │    │
│   │  Learned seed vector queries the set                   │    │
│   │  Aggregates to fixed-size output                       │    │
│   └─────────────────────────────────────────────────────────┘    │
│                              │                                    │
│                              ▼                                    │
│   Output: sample_embedding [B, H]                                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**ISAB Mathematical Definition:**

$$H = \text{softmax}\left(\frac{I W_Q (X W_K)^\top}{\sqrt{d_k}}\right) X W_V$$
$$\text{ISAB}(X) = \text{softmax}\left(\frac{X W_Q' H^\top}{\sqrt{d_k}}\right) H$$

where $I \in \mathbb{R}^{M \times H}$ are learnable inducing points.

---

### Component 4: Cross-Attention Drift Head

Predicts the velocity field $v_\theta(x_t, t)$ for OT-CFM flow matching.

```
                    Cross-Attention Drift Head
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  Inputs:                                                          │
│  ├── x_t: [B, 40]           Current state                        │
│  ├── t: [B]                 Time in [0, 1]                       │
│  ├── context_tokens: [B, 9, H]  From encoder/refiner             │
│  └── stage_pair_id: [B]     Stage transition                     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  CONTEXT PATH (Cross-Attention)                         │     │
│  │                                                         │     │
│  │  Q = Linear([x_t ; time_emb])                          │     │
│  │  KV = concat([context_tokens, stage_token])            │     │
│  │  h = MultiHeadAttention(Q, KV, KV)                     │     │
│  │  context_velocity = Linear(LayerNorm(h + FFN(h)))      │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  LATENT PATH (MLP baseline)                             │     │
│  │                                                         │     │
│  │  latent_velocity = MLP([x_t ; time_emb ; stage_emb])   │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  GATED MIXTURE                                          │     │
│  │                                                         │     │
│  │  gate = sigmoid(Linear([Q, h, stage_emb]))             │     │
│  │  v(x,t) = gate * context_velocity + (1-gate) * latent  │     │
│  │                                                         │     │
│  │  Interpretation:                                        │     │
│  │  - gate ≈ 1: niche strongly influences velocity        │     │
│  │  - gate ≈ 0: cell follows intrinsic dynamics           │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  Output: v(x_t, t) [B, 40]                                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

### Component 5: UDE Mode (Optional)

**Universal Differential Equation** mode blends a simple mechanistic baseline with the learned neural correction.

$$v_{\text{total}} = (1 - g_e) \cdot v_{\text{baseline}} + g_e \cdot v_{\text{learned}}$$

where:
- $v_{\text{baseline}} = \text{gate}(t) \cdot (\text{scale}_e \cdot x_t + \text{bias}_e)$ is a per-edge linear drift
- $g_e = \sigma(\text{logit}_e)$ is a learned per-edge gate

**Scientific rationale:** If some stage transitions follow simple average dynamics (e.g., proliferation), the baseline captures this. The neural component adds niche-specific corrections. The learned gate reveals which transitions rely on microenvironment.

**When to use:** Enable for interpretability about which transitions need context vs. follow average dynamics. Disable for maximum model flexibility.

---

### Component 6: Evolution Branch (WES)

Conditions on whole-exome sequencing (WES) features for genomic context.

```python
class EvolutionBranch(nn.Module):
    """Gated fusion of WES features with niche context."""
    
    def __init__(self, evolution_dim, model_dim, mode="gated"):
        self.proj = nn.Linear(evolution_dim, model_dim)
        if mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
                nn.Sigmoid(),
            )
    
    def forward(self, context, wes_features):
        wes_emb = self.proj(wes_features)
        gate = self.gate(torch.cat([context, wes_emb], dim=-1))
        return gate * wes_emb + (1 - gate) * context
```

**Scientific rationale:** Genomic mutations (e.g., TP53, KRAS) create fitness landscapes that interact with niche signals. WES features provide lesion-level genomic context.

---

### Component 7: Sample-Level Heads

For lesion/sample-level predictions after hierarchical aggregation.

```python
class SampleLevelHeads(nn.Module):
    """Predict stage and displacement from sample embedding."""
    
    def __init__(self, input_dim, num_stage_classes, dropout):
        self.stage_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_stage_classes),
        )
        self.displacement_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim),
        )
    
    def forward(self, sample_embedding):
        return {
            "stage_logits": self.stage_head(sample_embedding),
            "displacement": self.displacement_head(sample_embedding),
        }
```

---

### Component 8: Auxiliary Biological Heads

Auxiliary heads provide biological regularization during training by predicting known cellular phenotypes. These are **not** the model's primary outputs but encourage the learned representations to capture biologically meaningful structure.

**Design Principle**: We include only broad, generic biological signals (pathways, proliferation) that don't overlap with our scientific claims. Specifically:
- **Included**: PROGENy pathway activities, Ki67 proliferation
- **Excluded**: IL1B expression, KAC scores (these would create circular validation)

```python
class PathwayHead(AuxiliaryHead):
    """Predict 14 PROGENy pathway activity scores.
    
    Encourages pathway-aware latent structure. The 14 pathways are:
    Androgen, EGFR, Estrogen, Hypoxia, JAK-STAT, MAPK, NFkB,
    PI3K, TGFb, TNFa, Trail, VEGF, WNT, p53
    """
    
    def __init__(self, input_dim: int, n_pathways: int = 14):
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_pathways),
        )
    
    def forward(self, context):
        return self.head(context)  # [B, 14]


class ProliferationHead(AuxiliaryHead):
    """Predict Ki67 proliferation status.
    
    Anchors model to broad tissue dynamics signal.
    Inspired by OSDR's use of proliferation markers.
    """
    
    def __init__(self, input_dim: int):
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, context):
        return self.head(context)  # [B, 1] (logit)
```

**Training Loss Integration**:
```
L_total = L_flow_matching 
        + lambda_pathway * MSE(pathway_pred, pathway_target)
        + lambda_prolif * BCE(prolif_pred, prolif_target)
```

Default weights: `lambda_pathway = 0.1`, `lambda_prolif = 0.1`

---

## Training: OT-CFM Flow Matching

### Training Algorithm

```
Algorithm: OT-CFM Training Step
────────────────────────────────────────────────────────────────
Input: Batch of cells with niche context {(x, neighbors, distances, stage)}

1. Select source-target stage pair: (s_src, s_tgt)

2. Encode niche context:
   context, context_tokens = NicheEncoder(receiver, neighbors, distances)
   if use_context_refiner:
       context_tokens = ContextRefiner(context_tokens)

3. Get source/target populations from batch:
   X_0 = {x : stage(x) = s_src}
   X_1 = {x : stage(x) = s_tgt}

4. Compute OT coupling via Sinkhorn:
   C = ||X_0 - X_1||^2  (cost matrix)
   pi = Sinkhorn(C, epsilon)

5. Sample N pairs from coupling:
   {(i_k, j_k)}_{k=1}^N ~ pi

6. For each pair (i, j):
   - Sample time: t ~ Uniform(0, 1)
   - Interpolate: x_t = (1-t) * X_0[i] + t * X_1[j]
   - Target velocity: u_t = X_1[j] - X_0[i]

7. Predict velocity:
   v_t = DriftHead(x_t, t, context_tokens[i], stage_pair)

8. Compute loss:
   L = (1/N) * sum_k ||v_t[k] - u_t[k]||^2

9. Backpropagate and update
────────────────────────────────────────────────────────────────
```

### Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{FM}} + \lambda_{\text{entropy}} \mathcal{L}_{\text{entropy}}$$

---

## Inference

### Integration Methods

| Method | Formula | Error | Use Case |
|--------|---------|-------|----------|
| **Euler** | $x_{k+1} = x_k + \Delta t \cdot v(x_k, t_k)$ | $O(\Delta t)$ | Fast, default |
| **Euler-Maruyama** | $x_{k+1} = x_k + \Delta t \cdot v + \sigma\sqrt{\Delta t} \cdot \eta$ | Stochastic | Uncertainty |
| **RK4** | 4th-order Runge-Kutta | $O(\Delta t^4)$ | High accuracy |

### Trajectory Sampling

```python
def sample_trajectory(x0, context, stage_pair_id, num_steps=8, sigma=0.0):
    """Return full trajectory [B, num_steps+1, D]."""
    trajectory = [x0]
    x = x0
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = (k + 0.5) * dt
        v = forward_vector_field(x, t, context, stage_pair_id)
        x = x + dt * v
        if sigma > 0:
            x = x + sigma * sqrt(dt) * randn_like(x)
        trajectory.append(x)
    return stack(trajectory, dim=1)
```

---

## Hyperparameters

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 40 | Cell embedding dimension (30 HLCA + 10 LuCA) |
| `hidden_dim` | 256 | Internal representation dimension |
| `num_heads` | 8 | Attention heads |
| `num_encoder_layers` | 2 | Attention layers in niche encoder |
| `max_neighbors` | 8 | Maximum neighbors (9 tokens total) |
| `num_stages` | 3 | Disease stages |
| `time_dim` | 32 | Time embedding dimension |
| `stage_dim` | 32 | Stage embedding dimension |
| `dropout` | 0.1 | Dropout rate |
| `use_context_refiner` | true | Enable SAB refinement |
| `num_refiner_layers` | 2 | SAB layers |
| `use_hierarchical` | true | Enable ISAB+PMA aggregation |
| `use_cross_attn_drift` | true | Cross-attention drift head |
| `use_ude` | false | UDE baseline+correction mode |
| `use_evolution_branch` | false | WES feature conditioning |
| `use_sample_heads` | true | Sample-level predictions |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `warmup_epochs` | 5 | Linear warmup epochs |
| `ot_epsilon` | 0.05 | Sinkhorn regularization |
| `sinkhorn_iters` | 80 | Sinkhorn iterations |
| `num_ot_pairs` | 512 | Pairs sampled per batch |
| `sigma` | 0.0 | Brownian bridge noise |
| `gradient_clip` | 1.0 | Gradient norm clipping |

---

## Ablation Studies

StageBridge supports systematic ablation experiments to validate each component's contribution.

| Ablation | What's Removed | Tests |
|----------|----------------|-------|
| `full` | Nothing | Baseline comparison |
| `no_niche` | Zero context path | Does niche matter at all? |
| `no_distance` | Zero distance encoding | Does spatial structure matter? |
| `no_gate` | Fix gate=1 | Is adaptive gating useful? |
| `no_token_types` | Zero token type embeddings | Do semantic roles matter? |
| `random_niche` | Shuffle neighbor assignment | Is specific niche identity important? |
| `hlca_only` | Remove LuCA reference | Is disease reference needed? |
| `luca_only` | Remove HLCA reference | Is healthy reference needed? |
| `no_wes` | Disable evolution branch | Do genomic features help? |
| `with_wes` | Enable evolution branch | Does WES improve predictions? |

---

## Files

```
stagebridge/
├── models/
│   └── stagebridge.py          # Main StageBridge class with all components
├── context/
│   ├── encoder.py              # ReceiverCenteredNicheEncoder
│   ├── layers.py               # SAB, ISAB, PMA, SinusoidalTimeEmbedding
│   ├── aggregation.py          # HierarchicalAggregator, SampleLevelHeads
│   └── evolution.py            # EvolutionBranch (WES conditioning)
├── transition/
│   └── drift.py                # CrossAttentionDrift, UDEGate, BiologicalBaselineDrift
├── training/
│   └── trainer.py              # StageBridgeTrainer with OT-CFM
└── baselines/
    ├── pooling.py              # PoolingMLP baseline
    ├── deepsets.py             # DeepSets baseline
    ├── set_transformer.py      # SetTransformer baseline
    └── graph_sage.py           # GraphSAGE baseline
```

---

## Gradient Flow Analysis

Critical gradient paths that must be maintained for training:

```
                              Gradient Flow
    ┌───────────────────────────────────────────────────────────────────────┐
    │                                                                       │
    │   Loss (MSE)                                                          │
    │       │                                                               │
    │       ▼                                                               │
    │   drift_head (all layers)                                             │
    │       │                                                               │
    │       ├──────────────────────────┐                                    │
    │       ▼                          ▼                                    │
    │   context_tokens              stage_embedding                         │
    │       │                                                               │
    │       │ ◀── CRITICAL: must include processed receiver                 │
    │       │                                                               │
    │       ▼                                                               │
    │   context_refiner (SAB layers, if enabled)                            │
    │       │                                                               │
    │       ▼                                                               │
    │   output_proj                                                         │
    │       │                                                               │
    │       ▼                                                               │
    │   h_receiver (after L attention layers)                               │
    │       │                                                               │
    │       ├──────────────────────────┐                                    │
    │       ▼                          ▼                                    │
    │   attention_layers[L-1]     ffn_layers[L-1]                          │
    │       │                          │                                    │
    │       ▼                          ▼                                    │
    │   attention_layers[0]       ffn_layers[0]                            │
    │       │                          │                                    │
    │       ▼                          ▼                                    │
    │   receiver_proj             neighbor_proj                             │
    │                                                                       │
    │   CONTRACT: All attention layers must receive gradients.              │
    │   If context_tokens only contains neighbors (not processed receiver), │
    │   attention layers beyond layer 0 will have zero gradients.           │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘
```

### Gradient Flow Contract

```python
# CORRECT: Receiver flows through attention, included in context_tokens
context = output_proj(h_receiver)  # h_receiver updated by L attention layers
context_tokens = concat([context.unsqueeze(1), output_proj(h_neighbors)])

# WRONG: Only neighbors in context_tokens, attention layers get no gradient
context_tokens = output_proj(h_neighbors)  # h_receiver not included!
```

**Runtime Check:**
```python
assert context_tokens.shape[1] == neighbors.shape[1] + 1, (
    "context_tokens must include receiver for gradient flow"
)
```

### Minimum 85% Gradient Coverage Contract

After the first backward pass, verify:
```python
with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
total = sum(1 for _ in model.parameters())
assert with_grad / total >= 0.85
```

---

## Sinkhorn Algorithm Details

The entropic OT coupling is computed via Sinkhorn iterations in **log-space** for numerical stability.

**Input:** Source samples $\{x_0^{(i)}\}_{i=1}^n$, target samples $\{x_1^{(j)}\}_{j=1}^m$

**Step 1: Cost Matrix**

$$C_{ij} = \|x_0^{(i)} - x_1^{(j)}\|_2^2$$

**Step 2: Kernel Matrix (log-space)**

$$\log K_{ij} = -\frac{C_{ij}}{\varepsilon}$$

where $\varepsilon > 0$ is the entropic regularization (default: 0.05).

**Step 3: Initialize Marginals (uniform)**

$$\log a_i = -\log n, \quad \log b_j = -\log m$$

**Step 4: Sinkhorn Iterations (log-space)**

Initialize $\log u^{(0)} = \mathbf{0}$, $\log v^{(0)} = \mathbf{0}$.

For $k = 1, \ldots, K_{\text{iter}}$:

$$\log u^{(k)}_i = \log a_i - \text{logsumexp}_j(\log K_{ij} + \log v^{(k-1)}_j)$$

$$\log v^{(k)}_j = \log b_j - \text{logsumexp}_i(\log K_{ij} + \log u^{(k)}_i)$$

**Step 5: Coupling Matrix**

$$\pi_{ij} = \exp(\log u_i + \log K_{ij} + \log v_j)$$

**Properties:**
- Marginals: $\sum_j \pi_{ij} \approx 1/n$, $\sum_i \pi_{ij} \approx 1/m$
- As $\varepsilon \to 0$: $\pi \to$ exact OT coupling
- Typical: $K_{\text{iter}} = 80$, $\varepsilon = 0.05$

### Why OT Coupling?

Without OT, random pairing produces crossing trajectories:

```
     Random Pairing              OT Pairing
     
     x0_1 ───────────▶ x1_2      x0_1 ───────────▶ x1_1
           ╲    ╱                 
            ╲  ╱                  x0_2 ───────────▶ x1_2
             ╳                    
            ╱  ╲                  x0_3 ───────────▶ x1_3
           ╱    ╲                 
     x0_2 ───────────▶ x1_1      (No crossings)
     
     (Crossings cause averaging)
```

OT coupling finds the minimum-cost bijection, producing non-crossing trajectories.

---

## Attention Sparsity Mechanisms

Three sparsity mechanisms encourage interpretable attention distributions:

### 1. Entropy Regularization (Default)

Adds a penalty on attention entropy to encourage peaked distributions:

$$\mathcal{L}_{\text{entropy}} = -\frac{\lambda}{K \log K} \sum_{i=1}^{K} \alpha_i \log(\alpha_i + \epsilon)$$

where $\alpha_i$ are attention weights and $\lambda$ is the regularization strength.

### 2. Top-K Sparsity

Hard attention: only attend to the $k$ neighbors with highest scores:

$$\alpha_i = \begin{cases} 
\text{softmax}(s_i) & \text{if } s_i \in \text{top-}k(s) \\
0 & \text{otherwise}
\end{cases}$$

### 3. Sparsemax (Martins & Astudillo, 2016)

Euclidean projection of scores onto the probability simplex:

$$\text{sparsemax}(s) = \arg\min_{p \in \Delta^{K-1}} \|p - s\|^2$$

**Closed-form solution:**

$$\text{sparsemax}(s)_i = [s_i - \tau(s)]_+$$

where $\tau(s)$ is chosen so that the result sums to 1.

| Method | Properties | Gradient |
|--------|------------|----------|
| Entropy | Soft, differentiable, tunable | Dense |
| Top-K | Hard, interpretable | Sparse (STE) |
| Sparsemax | Exact zeros, differentiable | Sparse |

---

## Distance Encoding Details

Spatial distances $d \in \mathbb{R}_{\geq 0}$ (in micrometers) are encoded using **Radial Basis Functions (RBF)**:

**RBF Encoding Formula:**

$$\phi_{\text{RBF}}(d) = \left[ \exp\left(-\frac{(d - \mu_k)^2}{2\sigma^2}\right) \right]_{k=1}^{K}$$

where:
- $\mu_k = \frac{(k-1) \cdot d_{\max}}{K-1}$ are uniformly spaced centers
- $\sigma = \frac{d_{\max}}{K}$ is the bandwidth
- $d_{\max} = 100 \mu m$ is the maximum distance
- $K = 16$ is the number of RBF centers

**Distance Bias Computation:**

$$b_{\text{dist}} = W_d \cdot \phi_{\text{RBF}}(d) + b_d \in \mathbb{R}^{n_{\text{heads}}}$$

**Alternative Encodings:**

| Method | Formula | Properties |
|--------|---------|------------|
| RBF (default) | $\exp(-(d-\mu_k)^2 / 2\sigma^2)$ | Localized, smooth |
| Sinusoidal | $[\sin(d \cdot \omega_k), \cos(d \cdot \omega_k)]$ | Periodic, unbounded |
| MLP | $\text{MLP}(d)$ | Fully learnable |

---

## Parameter Count

For default configuration (`hidden_dim=256`, `num_heads=8`, `num_encoder_layers=2`):

| Component | Parameters |
|-----------|------------|
| **Niche Encoder** | |
| Receiver projection | 40 * 256 + 256 = 10,496 |
| Neighbor projection | 40 * 256 + 256 = 10,496 |
| Token type embeddings | 9 * 256 = 2,304 |
| Attention layers (x2) | 2 * (4 * 256^2) = 524,288 |
| FFN layers (x2) | 2 * (256 * 1024 + 1024 * 256) = 1,048,576 |
| Distance encoder | 16 * 8 + 8 = 136 |
| Output projection | 256 * 256 + 256 = 65,792 |
| **Context Refiner (SAB x2)** | ~1,580,000 |
| **Hierarchical Aggregator** | |
| ISAB layers (x2) | ~2,000,000 |
| PMA | ~400,000 |
| LayerNorm | 512 |
| **Drift Head** | |
| Query/KV projections | ~200,000 |
| Cross-attention | ~260,000 |
| FFN | ~260,000 |
| Gate network | ~130,000 |
| Latent MLP | ~130,000 |
| **Sample Heads** | |
| Stage head | ~33,000 |
| Displacement head | ~33,000 |
| **Evolution Branch (optional)** | ~50,000 |
| **Total** | **~8,500,000** |

---

## Baseline Comparison

StageBridge is validated against a baseline ladder:

| Model | Structure | Spatial | Niche | Description |
|-------|-----------|---------|-------|-------------|
| PoolingMLP | None | No | No | Mean pool all cells, MLP |
| DeepSets | Permutation | No | No | $\rho(\sum_i \phi(x_i))$ |
| SetTransformer | Attention | No | No | Self-attention over set |
| GraphSAGE | Graph | Yes | Symmetric | Neighborhood aggregation |
| **StageBridge** | **Attention** | **Yes** | **Receiver-centered** | Full model |

The key claim: **receiver-centered attention with spatial distance encoding outperforms symmetric alternatives**.

---

## Appendix: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $D$ | Cell embedding dimension (40) |
| $H$ | Hidden dimension (256) |
| $K$ | Number of neighbor tokens (8) |
| $L$ | Number of encoder layers (2) |
| $S$ | Number of disease stages |
| $T$ | Time embedding dimension (32) |
| $M$ | Number of inducing points (ISAB) |
| $x_t$ | Cell state at time $t$ |
| $v_\theta$ | Learned velocity field |
| $c$ | Niche context vector |
| $C_{\text{tokens}}$ | Context token matrix |
| $\alpha$ | Attention weights |
| $\pi$ | OT coupling matrix |
| $\varepsilon$ | Sinkhorn regularization |
| $\sigma$ | Brownian bridge noise |
| $g$ | Context gate value |
| $I$ | Inducing points (ISAB) |
| $e_{ij}$ | Stage transition embedding |

---

## References

1. Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling." ICLR.
2. Tong, A., et al. (2023). "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport." ICML.
3. Lee, J., et al. (2019). "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks." ICML.
4. Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NeurIPS.
5. Martins, A. & Astudillo, R. (2016). "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification." ICML.
6. Sikkema, L., et al. (2023). "An integrated cell atlas of the lung in health and disease." Nature Medicine.
