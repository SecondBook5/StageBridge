# StageBridge Model Architecture

**A Receiver-Centered Niche Encoding Model for Disease Stage Transitions**

This document provides a complete mathematical and architectural specification of StageBridge, a deep learning model that learns to predict how cells transition between disease stages by conditioning on their local spatial neighborhood (niche).

**Key References:**

- **AMICI** (Hong et al., bioRxiv 2025): Receiver-centered attention with monotonic distance decay. doi:10.1101/2025.09.22.677860. https://github.com/azizilab/amici. License: CC BY-NC-ND 4.0. Patent pending (U.S. Serial No. 63/884,704).
- **Set Transformer** (Lee et al., ICML 2019): ISAB and PMA for permutation-invariant set encoding.
- **OT-CFM** (Lipman et al., ICLR 2023): Optimal transport conditional flow matching.

---

## Table of Contents

1. [Overview](#overview)
2. [Scientific Foundations](#scientific-foundations)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Input Representation](#input-representation)
5. [Architecture Components](#architecture-components)
   - [AMICI Receiver-Centered Attention](#component-1-amici-receiver-centered-attention)
   - [Empty Neighbor Token](#component-2-empty-neighbor-token)
   - [Niche Encoder](#component-3-niche-encoder)
   - [Set Transformer Layers](#component-4-set-transformer-layers)
   - [Cross-Attention Drift Head](#component-5-cross-attention-drift-head)
   - [Learned GW Fusion](#component-6-learned-gw-fusion)
6. [Training: OT-CFM Flow Matching](#training-ot-cfm-flow-matching)
7. [Inference](#inference)
8. [Hyperparameters](#hyperparameters)
9. [Ablation Studies](#ablation-studies)
10. [Code Reference](#code-reference)

---

## Overview

StageBridge learns to predict how cells transition between disease stages (e.g., Normal -> Preinvasive -> Invasive) by conditioning on the local spatial neighborhood (niche) around each cell. The key insight is **receiver-centering**: the focal cell receives signals from its neighbors, so it should be the query in attention, with neighbors as keys/values.

```
                              StageBridge Architecture
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│  INPUT: Receiver cell + K nearest neighbors + distances                                 │
│  ────────────────────────────────────────────────────                                   │
│                              │                                                          │
│                              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    DUAL-REFERENCE EMBEDDING                                      │   │
│  │                                                                                  │   │
│  │   Raw expression ──▶ scArches HLCA (30d) ──┐                                    │   │
│  │                                            ├──▶ Concat or GW Fusion (40d)       │   │
│  │   Raw expression ──▶ scArches LuCA (10d) ──┘                                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                          │
│                              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │         AMICI RECEIVER-CENTERED ATTENTION (Hong et al., 2025)                    │   │
│  │                                                                                  │   │
│  │   Receiver ──Q──▶ ┌─────────────────────────────────────────────────────┐       │   │
│  │                   │  attn = softmax(QK^T/√d - b₁·distance || empty)     │       │   │
│  │   Neighbors ──K,V─▶│  b₁ = softplus(MLP(receiver))  [enforced positive] │       │   │
│  │                   │  context = attn · V                                  │       │   │
│  │   Distances ──────▶└─────────────────────────────────────────────────────┘       │   │
│  │                                                                                  │   │
│  │   Key: Attention MONOTONICALLY DECREASES with distance (architectural)          │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                          │
│                              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    SET TRANSFORMER REFINEMENT                                    │   │
│  │                                                                                  │   │
│  │   Context tokens ──▶ ISAB ──▶ SAB ──▶ PMA ──▶ Context vector [B, H]            │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                          │
│                              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    CROSS-ATTENTION DRIFT HEAD                                    │   │
│  │                                                                                  │   │
│  │   x_t + time_emb ──Q──▶ CrossAttn over context tokens ──▶ v_context            │   │
│  │                                                                                  │   │
│  │   x_t + time_emb + stage ──▶ MLP ──▶ v_latent                                  │   │
│  │                                                                                  │   │
│  │   gate = σ(MLP([q, h, stage]))                                                  │   │
│  │   v(x,t,c) = gate · v_context + (1-gate) · v_latent                            │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                          │
│                              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    OT-CFM FLOW MATCHING                                          │   │
│  │                                                                                  │   │
│  │   Loss = E[ ||v_θ(x_t, t, c) - (x₁ - x₀)||² ]                                  │   │
│  │   where (x₀, x₁) ~ π* (Sinkhorn OT coupling)                                   │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
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
| **Monotonic distance decay** | Paracrine signaling (cytokines, growth factors) decays with distance due to diffusion |
| **Empty neighbor token** | Some cells are niche-independent; the model learns when neighbors don't matter |
| **Dual-reference embeddings** | Position cells relative to both healthy (HLCA) and disease (LuCA) states |
| **Gated context/latent** | Some transitions are cell-autonomous, others depend on niche |

### Target Biological Signals (Validation)

From Peng/Kadara literature:
- KAC/reactive pneumocyte-like alveolar progenitors as LUAD predecessors
- Epithelial-proinflammatory niches with IL1B-high macrophages
- IL1B-IL1R1 signaling axis
- These niches more common in AAH/AIS than LUAD (progression window)

---

## Mathematical Foundations

### Problem Setting

Let X ⊂ ℝ^D be the cell state space (where D = 40 for dual-reference embeddings). We observe cells from S disease stages, denoted {s₁, s₂, ..., sₛ} (e.g., Normal, Preinvasive, Invasive).

**Goal**: Learn a conditional velocity field v_θ: X × [0,1] × C → ℝ^D that transports cells from stage sᵢ to stage sⱼ, conditioned on local niche context c ∈ C.

### Continuous Normalizing Flow Formulation

The stage transition is modeled as an ordinary differential equation (ODE):

```
dx_t/dt = v_θ(x_t, t, c, e_ij)
```

where:
- x_t ∈ ℝ^D is the cell state at time t ∈ [0, 1]
- c ∈ ℝ^H is the niche context encoding
- e_ij is the stage transition embedding (sᵢ → sⱼ)

### Conditional Flow Matching Objective

Following Lipman et al. (2023), we train by regressing the conditional velocity field:

```
L_CFM = E_{t∼U[0,1], (x₀,x₁)∼π*} [ ||v_θ(x_t, t, c) - u_t||² ]
```

where x_t = (1-t)x₀ + tx₁ and u_t = x₁ - x₀.

### Optimal Transport Coupling

We use **entropic optimal transport** coupling:

```
π* = argmin_{π ∈ Π(p₀,p₁)} ∫ ||x-y||² dπ(x,y) + ε·H(π)
```

Solved via the Sinkhorn algorithm in log-space for numerical stability.

---

## Input Representation

### Dual-Reference Cell Embeddings

Each cell is embedded into a 40-dimensional space via scArches:

| Reference | Dimension | Description |
|-----------|-----------|-------------|
| HLCA | 30d | Human Lung Cell Atlas - positions cell relative to healthy tissue |
| LuCA | 10d | Lung Cancer Atlas - positions cell relative to disease states |

**Fusion options:**
1. **Concatenation** (default): Simply concatenate [HLCA; LuCA] → 40d
2. **Learned GW**: Gromov-Wasserstein alignment in learned metric spaces

### Neighborhood Structure

For each receiver cell, we provide:
- K nearest neighbors (default K=20)
- Euclidean distances to each neighbor (in micrometers)
- Optional: neighbor mask for variable-size neighborhoods

---

## Architecture Components

### Component 1: AMICI Receiver-Centered Attention

**This is the core novelty adapted from Hong et al. (2025).**

The key architectural constraint is that attention **must monotonically decrease with distance**. This is enforced, not learned:

```
attn_logits = phenotype_score - distance_penalty

where:
  phenotype_score = (Q · K^T) / √d_k
  distance_penalty = b₁ · (dist / distance_scale)
  b₁ = softplus(MLP(receiver) + offset)   [ALWAYS POSITIVE]
```

**Code:** `stagebridge/context/encoder.py:149-282`

```python
class ReceiverCenteredAttention(nn.Module):
    def forward(self, receiver, neighbors, distances, neighbor_mask=None):
        # Project to multihead format
        q = self.q_proj(receiver).unsqueeze(1)  # [B, 1, D]
        k = self.k_proj(neighbors)               # [B, K, D]
        v = self.v_proj(neighbors)               # [B, K, D]
        
        # Reshape for multihead: [B, H, seq, d_h]
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Phenotype similarity (standard scaled dot-product)
        phenotype_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # CRITICAL: Distance decay coefficient MUST be positive
        distance_coef_raw = self.distance_coef_mlp(receiver) + self.distance_coef_offset
        distance_coef = F.softplus(distance_coef_raw)  # softplus > 0 always
        
        # Subtract penalty -> guarantees monotonic decay
        normalized_dist = distances / self.distance_scale
        distance_penalty = distance_coef.unsqueeze(-1) * normalized_dist.unsqueeze(1)
        attn_logits = phenotype_score - distance_penalty
        
        attn_weights = F.softmax(attn_logits, dim=-1)
        context = torch.matmul(attn_weights, v)
        return context
```

**Why this matters:**

The softplus function ensures b₁ > 0 regardless of what the MLP outputs:
- softplus(x) = log(1 + exp(x)) > 0 for all x
- Since we SUBTRACT b₁ · dist, larger distance → lower attention logit
- softmax preserves ordering → lower logit → lower weight

This is an **architectural guarantee**, not a soft constraint. The model cannot learn to increase attention with distance.

**Biological justification:** Paracrine signaling decays with distance. A cell 10μm away sends stronger signals than one 50μm away. The learned b₁ controls HOW FAST attention decays, but it always decays.

### Component 2: Empty Neighbor Token

The empty neighbor token allows attention to "escape" when no neighbor is informative:

```python
if self.use_empty_token:
    # Append fixed score for empty token
    empty_score = torch.full((B, H, 1, 1), self.empty_token_score, ...)
    attn_logits = torch.cat([attn_logits, empty_score], dim=-1)
    
    # Empty token contributes zero to output
    empty_v = torch.zeros((B, H, 1, d_h), ...)
    v = torch.cat([v, empty_v], dim=2)
```

**Interpretation:**
- High empty_attention (e.g., > 0.5) means "this cell's neighbors don't matter much"
- Low empty_attention means "neighbors strongly influence this cell"
- This is a diagnostic for niche-dependence

### Component 3: Niche Encoder

The full encoder wraps AMICI attention with input/output projections:

```python
class ReceiverCenteredNicheEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ...):
        self.receiver_proj = nn.Linear(input_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(input_dim, hidden_dim)
        
        # Token type embeddings for semantic differentiation
        self.token_type_embeddings = nn.Embedding(NUM_TOKEN_TYPES, hidden_dim)
        
        # AMICI attention layers
        self.attention_layers = nn.ModuleList([
            ReceiverCenteredAttention(hidden_dim, num_heads, ...)
            for _ in range(num_layers)
        ])
        
        # Optional reconstruction head for SSL
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)
```

**Code:** `stagebridge/context/encoder.py:285-500`

### Component 4: Set Transformer Layers

For hierarchical processing, we use Set Transformer components from Lee et al. (2019):

**Self-Attention Block (SAB):**
```
SAB(X) = LN(X + MHA(X, X, X))
```

**Induced Set Attention Block (ISAB):**
```
ISAB(X) = MAB(X, MAB(I, X))

where I ∈ ℝ^{m×d} are learnable inducing points
```

Complexity: O(n·m) instead of O(n²)

**Pooling by Multihead Attention (PMA):**
```
PMA(X) = MAB(S, X)

where S ∈ ℝ^{k×d} are learnable seed vectors
```

**Code:** `stagebridge/context/layers.py`

### Component 5: Cross-Attention Drift Head

The drift head predicts velocity v(x_t, t, c) with a gated mixture of context-informed and latent-only paths:

```
v(x_t, t, c) = g · v_context + (1-g) · v_latent

where:
  q = W_q · [x_t ; time_emb]
  v_context = CrossAttn(q, context_tokens ∪ stage_token)
  v_latent = MLP([x_t ; time_emb ; stage_emb])
  g = σ(MLP([q ; h ; stage_emb])) ∈ [0, 1]
```

**Code:** `stagebridge/transition/drift.py:15-106`

```python
class CrossAttentionDrift(nn.Module):
    def forward(self, x_t, time_emb, context_tokens, stage_emb):
        # Query from state + time
        q = self.query_proj(torch.cat([x_t, time_emb], dim=-1)).unsqueeze(1)
        
        # Keys/values from context + stage
        kv_ctx = self.kv_proj(context_tokens)
        stage_tok = self.stage_proj(stage_emb).unsqueeze(1)
        kv = torch.cat([kv_ctx, stage_tok], dim=1)
        
        # Cross-attention
        attn_out, _ = self.mha(query=q, key=kv, value=kv)
        h = self.ln2(self.ln1(q + attn_out) + self.ff(...))
        v_context = self.context_out_proj(h.squeeze(1))
        
        # Latent-only baseline
        v_latent = self.latent_only(torch.cat([x_t, time_emb, stage_emb], dim=-1))
        
        # Learned gate
        gate = self.context_gate(torch.cat([q.squeeze(1), h.squeeze(1), stage_emb], dim=-1))
        
        return gate * v_context + (1 - gate) * v_latent
```

**Interpretation:**
- gate → 1: Niche strongly influences velocity (niche-gated transition)
- gate → 0: Cell follows intrinsic dynamics (cell-autonomous transition)
- Mean gate value is a diagnostic for niche-dependence

### Component 6: Learned GW Fusion

Optional Gromov-Wasserstein alignment of HLCA and LuCA in learned metric spaces:

```
GW objective:
  min_P Σ_ijkl |C_X[i,k] - C_Y[j,l]|² · P[i,j] · P[k,l]

where:
  C_X = pairwise distances in learned HLCA metric
  C_Y = pairwise distances in learned LuCA metric
  P = coupling matrix (solved via Sinkhorn)
```

**Code:** `stagebridge/reference/learned_gw_fusion.py`

This is an HPO/ablation question: learned GW fusion may outperform simple concatenation, but the architecture supports both.

---

## Training: OT-CFM Flow Matching

### Two-Stage Training

1. **SSL Pretraining** (50 epochs): Learn niche-aware representations via masked receiver reconstruction
2. **Transition Learning** (100 epochs): Learn OT-CFM velocity field conditioned on frozen/fine-tuned encoder

### Flow Matching Algorithm

```
Algorithm: OT-CFM Training Step
────────────────────────────────────────────────────────────────
Input: Batch of cells with niche context {(x, neighbors, distances, stage)}

1. Select source-target stage pair: (s_src, s_tgt)

2. Encode niche context:
   context, context_tokens = NicheEncoder(receiver, neighbors, distances)

3. Get source/target populations from batch:
   X_0 = {x : stage(x) = s_src}
   X_1 = {x : stage(x) = s_tgt}

4. Compute OT coupling via Sinkhorn (log-space):
   C = ||X_0 - X_1||²  (cost matrix)
   π* = Sinkhorn(C, ε=0.05, iters=80)

5. Sample N pairs from coupling:
   {(i_k, j_k)} ~ π*

6. For each pair (i, j):
   - Sample time: t ~ Uniform(0, 1)
   - Interpolate: x_t = (1-t) · X_0[i] + t · X_1[j]
   - Optional noise: x_t += σ·√(t(1-t))·η where η~N(0,I)
   - Target velocity: u_t = X_1[j] - X_0[i]

7. Predict velocity:
   v_t = DriftHead(x_t, t, context_tokens[i], stage_pair)

8. Compute loss:
   L = (1/N) · Σ_k ||v_t[k] - u_t[k]||²
────────────────────────────────────────────────────────────────
```

**Code:** `stagebridge/training/trainer.py:958-1065`

### Sinkhorn Algorithm (Log-Space)

```python
def _sinkhorn_coupling(self, x_src, x_tgt):
    # Cost matrix in float64 for stability
    cost = torch.cdist(x_src.double(), x_tgt.double(), p=2).pow(2)
    
    # Uniform marginals in log space
    log_a = -torch.log(torch.tensor(n, dtype=torch.float64))
    log_b = -torch.log(torch.tensor(m, dtype=torch.float64))
    
    log_K = -cost / self.config.ot_epsilon
    log_u = torch.zeros(n, dtype=torch.float64)
    log_v = torch.zeros(m, dtype=torch.float64)
    
    for _ in range(self.config.sinkhorn_iters):  # 80 iterations
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)
    
    log_pi = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    return torch.exp(log_pi)
```

---

## Inference

### ODE Integration

At inference, we integrate the learned velocity field:

```
x₁ = x₀ + ∫₀¹ v_θ(x_t, t, c) dt
```

**Euler discretization (default, N=8-16 steps):**
```
x_{k+1} = x_k + Δt · v_θ(x_k, t_k, c)
```

**Euler-Maruyama (for uncertainty):**
```
x_{k+1} = x_k + Δt · v_θ(x_k, t_k, c) + σ·√Δt · η_k
```

The trajectories are nearly straight thanks to OT coupling, so few integration steps suffice.

---

## Hyperparameters

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 40 | Cell embedding dimension (30 HLCA + 10 LuCA) |
| `hidden_dim` | 128-256 | Internal representation dimension |
| `num_heads` | 4-8 | Attention heads |
| `num_encoder_layers` | 2 | AMICI attention layers |
| `max_neighbors` | 20 | K nearest neighbors |
| `distance_scale` | 50.0-100.0 | Distance normalization (μm) |
| `empty_token_score` | 3.0 | Fixed score for empty token |
| `dropout` | 0.1 | Dropout rate |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `ot_epsilon` | 0.05 | Sinkhorn entropic regularization |
| `sinkhorn_iters` | 80 | Sinkhorn iterations |
| `num_ot_pairs` | 256-512 | Pairs sampled per batch |
| `sigma` | 0.0 | Stochastic interpolant noise |

---

## Ablation Studies

### Core Ablations

| Ablation | What's Changed | Tests |
|----------|----------------|-------|
| `full` | Nothing | Full model baseline |
| `no_niche` | Zero context | Does niche matter at all? |
| `no_distance` | Remove distance penalty | Does monotonic decay matter? |
| `no_empty_token` | Disable empty token | Does escape matter? |
| `random_niche` | Shuffle neighbors | Is specific identity important? |
| `frozen_encoder` | Freeze encoder in stage 2 | Does SSL pretrain transfer? |

### Reference Ablations

| Ablation | Configuration | Tests |
|----------|---------------|-------|
| `hlca_only` | Zero LuCA embedding | Is disease reference needed? |
| `luca_only` | Zero HLCA embedding | Is healthy reference needed? |
| `gw_fusion` | Enable learned GW | Is GW better than concat? |

---

## Code Reference

```
stagebridge/
├── context/
│   ├── encoder.py              # ReceiverCenteredAttention, ReceiverCenteredNicheEncoder
│   ├── layers.py               # SAB, ISAB, PMA, SinusoidalTimeEmbedding
│   └── README.md               # Module documentation
├── transition/
│   ├── drift.py                # CrossAttentionDrift, FiLMConditioner
│   └── README.md               # Module documentation
├── reference/
│   ├── learned_gw_fusion.py    # LearnedGWFusion, gromov_wasserstein_differentiable
│   └── README.md               # Module documentation
├── training/
│   ├── trainer.py              # StageBridgeTrainer, _sinkhorn_coupling, _flow_matching_loss
│   └── README.md               # Module documentation
├── models/
│   ├── stagebridge.py          # Main StageBridge class
│   └── README.md               # Module documentation
├── interpretation/
│   └── README.md               # Interpretability tools
└── baselines/
    ├── pooling.py              # PoolingMLP
    ├── deepsets.py             # DeepSets
    ├── set_transformer.py      # SetTransformer
    └── graph_sage.py           # GraphSAGE
```

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| D | Cell embedding dimension (40) |
| H | Hidden dimension |
| K | Number of neighbors |
| d_k | Head dimension (H / num_heads) |
| b₁ | Distance decay coefficient (positive via softplus) |
| x_t | Cell state at time t |
| v_θ | Learned velocity field |
| c | Niche context vector |
| π* | OT coupling matrix |
| ε | Sinkhorn entropic regularization |
| g | Context gate value [0, 1] |

---

## References

1. Hong J, et al. (2025). "AMICI: Attention Mechanism Interpretation of Cell-cell Interactions." bioRxiv. doi:10.1101/2025.09.22.677860
2. Lipman Y, et al. (2023). "Flow Matching for Generative Modeling." ICLR.
3. Lee J, et al. (2019). "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks." ICML.
4. Cuturi M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NeurIPS.
5. Sikkema L, et al. (2023). "An integrated cell atlas of the lung in health and disease." Nature Medicine.
