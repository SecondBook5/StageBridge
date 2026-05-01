# StageBridge Architecture Figure

**File:** `stagebridge_ml.tex` / `stagebridge_ml.pdf`

**Title:** StageBridge Architecture - Receiver-centered niche encoding with OT-CFM for cell state transitions

---

## Overview

This figure illustrates the complete StageBridge model architecture in 5 panels, showing how cellular neighborhoods are tokenized, processed through attention mechanisms, fused with reference atlases, and used to predict cell state transitions via conditional flow matching.

---

## Panel Descriptions

### Panel a: Set Transformer Tokenization

**What it shows:** Converting variable-size cellular neighborhoods into fixed-length token sequences.

**Components:**
- **Input:** Variable number of neighbor cells (blue circles)
- **ISAB:** Induced Set Attention Block (m=4 inducing points) - compresses variable input
- **PMA:** Pooling by Multihead Attention (k=8) - produces fixed output

**Output tokens (9 total, dimension D each):**
| Token | Color | Description |
|-------|-------|-------------|
| recv | Red | Receiver cell (the cell being modeled) |
| ring_1-4 | Blue | Spatial ring tokens (concentric neighborhoods) |
| ref | Green | Reference atlas embeddings (HLCA + LuCA) |
| stats | Yellow | Biological statistics (cell cycle, etc.) |

**Key point:** "No matter how many neighbors a cell has, we get the same 9-token representation."

---

### Panel b: Receiver-Centered Attention

**What it shows:** How the receiver cell attends to its spatial context with distance-aware attention.

**Mechanism:**
- **Query:** Receiver token only
- **Keys/Values:** Spatial ring tokens
- **Distance decay:** Attention scores are penalized by physical distance

**Equation:**
```
attention = softmax(QK^T / sqrt(d) - beta * distance)
output = sum_j(alpha_j * V_j)
```

**Key point:** "Nearby cells have more influence than distant cells, controlled by learned parameter beta."

---

### Panel c: Gromov-Wasserstein Fusion

**What it shows:** Aligning and fusing two reference atlas embeddings (healthy vs cancer).

**Components:**
- **HLCA space:** Healthy Lung Cell Atlas embeddings (30D)
- **LuCA space:** Lung Cancer Atlas embeddings (10D)
- **GW Transport:** Gromov-Wasserstein optimal transport alignment

**Equation:**
```
min_T sum |d_H(i,j) - d_L(k,l)|^2 * T_ik * T_jl
```

**Output:** Fused embedding z_f in R^40

**Key point:** "We align healthy and cancer reference spaces by preserving pairwise distance structure, not just concatenating."

---

### Panel d: CrossAttentionDrift Network

**What it shows:** The velocity prediction network that drives the flow matching.

**Inputs:**
| Symbol | Name | Description |
|--------|------|-------------|
| x_t | state | Current cell state at time t |
| tau | time | Flow time embedding |
| c | context | Niche context vector from encoder |
| s | stage | Disease stage embedding |

**Two parallel paths:**
1. **Context-conditioned path:** Cross-attention between (state, time) query and context keys/values
2. **Latent-only path:** MLP processing state, time, and stage directly

**Gating mechanism:**
```
v_theta = g * v_ctx + (1-g) * v_lat
```

**Key point:** "The model can blend context-dependent and context-independent predictions, learning when niche matters."

---

### Panel e: Optimal Transport CFM

**What it shows:** The flow matching training objective with optimal transport coupling.

**Components:**
- **pi*:** Optimal coupling matrix (computed via Sinkhorn)
- **x_0:** Source cell state
- **x_1:** Target cell state
- **x_t:** Interpolated state at time t
- **v_theta:** Predicted velocity

**Loss function:**
```
L = E_{t, pi*}[||v_theta(x_t, t | c) - (x_1 - x_0)||^2]
```

**Key point:** "We train the model to predict straight-line velocities between optimally-coupled source-target pairs."

---

## Legend

| Symbol | Color | Meaning |
|--------|-------|---------|
| Red box | Salmon | Receiver token |
| Blue box | Sky blue | Spatial tokens |
| Green box/circle | Mint | Context/reference |
| Purple box | Lavender | Multi-head attention |
| Yellow box | Gold | Gate/stats |

---

## One-Sentence Summary

> StageBridge tokenizes variable-size cellular neighborhoods into fixed 9-token sequences, uses receiver-centered attention with distance decay to capture microenvironment influence, fuses healthy and cancer atlas references via Gromov-Wasserstein transport, then learns a gated conditional flow that predicts cell state transitions given niche context.

---

## Talking Points for Presentations

1. **Why tokenization?** "Single-cell neighborhoods vary in size - some cells have 5 neighbors, some have 50. The Set Transformer gives us a fixed representation regardless."

2. **Why receiver-centered?** "We're modeling how the microenvironment affects THIS cell specifically. The receiver is the query, the neighborhood is the context."

3. **Why GW fusion?** "HLCA tells us about healthy cell states, LuCA about cancer states. GW aligns them structurally rather than just concatenating."

4. **Why the gate?** "Sometimes niche context matters a lot (early progression), sometimes less (cell-autonomous changes). The gate learns this."

5. **Why OT-CFM?** "Optimal transport gives us biologically meaningful pairings between cell states. Flow matching learns smooth, reversible trajectories."
