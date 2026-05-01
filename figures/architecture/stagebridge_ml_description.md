# StageBridge Architecture Figure

**File:** `stagebridge_ml.tex` / `stagebridge_ml.pdf`

---

## One-Sentence Summary

> StageBridge tokenizes variable-size cellular neighborhoods into fixed 9-token sequences, uses receiver-centered attention with distance decay to capture microenvironment influence, fuses healthy and cancer atlas references via Gromov-Wasserstein transport, then learns a gated conditional flow that predicts cell state transitions given niche context.

---

## Panel-by-Panel Description

### Panel a: Set Transformer Tokenization

**What it does:** Converts a variable number of neighbor cells into a fixed 9-token sequence.

**Components:**
- **ISAB** (Induced Set Attention Block): Compresses variable-length input using m=4 inducing points
- **PMA** (Pooling by Multihead Attention): Outputs k=8 fixed tokens

**Output tokens:**
- 1 **receiver** (red) - the cell we're modeling
- 4 **spatial rings** (blue) - concentric neighborhood summaries
- 2 **reference** (green) - HLCA and LuCA atlas embeddings  
- 1 **stats** (yellow) - biological covariates (cell cycle, etc.)

**How to say it:**
> "We take a variable number of neighboring cells and compress them into a fixed 9-token representation using a Set Transformer. This gives us the receiver cell, four concentric spatial rings, two reference atlas embeddings, and a statistics token."

---

### Panel b: Receiver-Centered Attention

**What it does:** The receiver cell attends to spatial context with distance-aware weighting.

**Equation:**
```
softmax(QK^T / sqrt(d) - beta * d)
```

**How to say it:**
> "This is standard scaled dot-product attention, but we subtract a distance penalty. The term Q-K-transpose over root-d is the usual attention score. We then subtract beta times the physical distance d. Beta is learned and positive, so nearby cells get higher attention weights than distant cells. This is inspired by AMICI's distance decay mechanism."

**Output:**
```
sum_j (alpha_j * V_j)
```

> "The output is a weighted sum of value vectors, where the weights alpha come from the distance-penalized softmax."

---

### Panel c: Gromov-Wasserstein Fusion

**What it does:** Aligns HLCA (healthy, 30D) and LuCA (cancer, 10D) embeddings by matching their internal distance structures.

**Equation:**
```
min_T sum |d_H(i,j) - d_L(k,l)|^2 * T_ik * T_jl
```

**How to say it:**
> "We minimize over transport plans T. For each pair of points i,j in the HLCA space and k,l in the LuCA space, we compute how different their pairwise distances are - d_H of i,j versus d_L of k,l. We square that difference and weight it by the coupling entries T_ik and T_jl. The optimization finds a coupling that makes HLCA and LuCA distances as consistent as possible."

> "In plain terms: if two cells are close in healthy-reference space, they should also be close in cancer-reference space after alignment."

**Output:** Fused embedding z_f in R^40

---

### Panel d: CrossAttentionDrift Network

**What it does:** Predicts the velocity (direction of change) for a cell state, conditioned on niche context.

**Inputs:**
- x_t = current cell state at time t
- tau = time embedding
- c = niche context vector
- s = disease stage

**Two parallel paths:**
1. **Context-conditioned path:** Cross-attention where (state + time) queries attend to context
2. **Latent-only path:** MLP that ignores context

**Equation:**
```
v_theta = g * v_ctx + (1-g) * v_lat
```

**How to say it:**
> "The final velocity v-theta is a weighted blend of two predictions. v-ctx is the context-aware prediction from cross-attention. v-lat is the context-free prediction from the MLP. The gate g, which passes through a sigmoid, learns when to trust context versus when to rely on the cell's intrinsic state. When g is high, context matters; when g is low, the cell follows its own trajectory."

---

### Panel e: Optimal Transport CFM

**What it does:** Trains the model to predict straight-line velocities between optimally-coupled cell states.

**Components:**
- **pi*** = optimal coupling matrix (computed via Sinkhorn algorithm)
- **x_0** = source cell state
- **x_1** = target cell state  
- **x_t** = interpolated state at time t

**Equation:**
```
L = E_{t, pi*}[||v_theta(x_t, t | c) - (x_1 - x_0)||^2]
```

**How to say it:**
> "The loss L is an expectation over time t and over source-target pairs sampled from the optimal coupling pi-star. For each pair, we interpolate to get x_t, ask the model to predict v-theta, and penalize the squared difference from the true velocity x_1 minus x_0. We're training the model to predict straight-line paths between optimally matched cell states."

> "The Sinkhorn algorithm finds which source cells should map to which target cells - it's not random pairing, it's optimal transport pairing that minimizes total movement cost."

---

## Quick Reference: How to Pronounce Symbols

| Symbol | Say |
|--------|-----|
| x_t | "x sub t" or "x at time t" |
| v_theta | "v theta" (the learned velocity) |
| pi* | "pi star" (optimal coupling) |
| d_H, d_L | "d sub H", "d sub L" (distances in HLCA/LuCA space) |
| T_ik | "T i-k" (transport plan entry) |
| alpha_j | "alpha j" (attention weight) |
| beta | "beta" (distance decay parameter) |
| sigma(g) | "sigma of g" (sigmoid gate) |

---

## Talking Points for Presentations

1. **Why tokenization?** "Single-cell neighborhoods vary in size - some cells have 5 neighbors, some have 50. The Set Transformer gives us a fixed representation regardless."

2. **Why receiver-centered?** "We're modeling how the microenvironment affects THIS cell specifically. The receiver is the query, the neighborhood is the context."

3. **Why GW fusion?** "HLCA tells us about healthy cell states, LuCA about cancer states. GW aligns them structurally rather than just concatenating - it preserves the geometry of both spaces."

4. **Why the gate?** "Sometimes niche context matters a lot (early progression), sometimes less (cell-autonomous changes). The gate learns this automatically."

5. **Why OT-CFM?** "Optimal transport gives us biologically meaningful pairings between cell states. Flow matching learns smooth, reversible trajectories - we can run the model forward (progression) or backward (what was the precursor state?)."
