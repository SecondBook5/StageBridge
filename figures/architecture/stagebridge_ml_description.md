# StageBridge Architecture Figure

**File:** `stagebridge_ml.tex` / `stagebridge_ml.pdf`

---

## One-Sentence Summary

> StageBridge tokenizes variable-size cellular neighborhoods into fixed 9-token sequences, uses receiver-centered attention with distance decay to capture microenvironment influence, fuses healthy and cancer atlas references via Gromov-Wasserstein transport, then learns a gated conditional flow that predicts cell state transitions given niche context.

---

## Panel a: Set Transformer Tokenization

**Purpose:** Handle variable-size inputs (neighborhoods have different numbers of cells) while producing fixed-size outputs.

**Architecture:**
- **ISAB** (Induced Set Attention Block): Uses m=4 learnable inducing points to compress N input cells into m intermediate representations in O(Nm) instead of O(N²)
- **PMA** (Pooling by Multihead Attention): k=8 learnable seed vectors query the ISAB output to produce exactly k output tokens

**Output:** 9 tokens of dimension D each:
| Token | Description |
|-------|-------------|
| recv | Receiver cell embedding |
| ring₁₋₄ | Spatial ring aggregations (concentric shells) |
| ref₁₋₂ | HLCA and LuCA reference embeddings |
| stats | Biological covariates |

---

## Panel b: Receiver-Centered Attention

### Equation

$$\alpha = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} - \beta \cdot \mathbf{d}\right)$$

$$\text{output} = \sum_j \alpha_j V_j$$

### Mathematical Explanation

**Standard attention** computes compatibility scores between query Q and keys K:

$$a_{ij} = \frac{q_i^\top k_j}{\sqrt{d}}$$

where d is the key dimension. The √d scaling prevents dot products from growing too large with high dimensions (which would push softmax into saturated regions with near-zero gradients).

**Distance-penalized attention** modifies this by subtracting a term proportional to physical distance:

$$a_{ij} = \frac{q_i^\top k_j}{\sqrt{d}} - \beta \cdot d_{ij}$$

where:
- $d_{ij}$ = Euclidean distance between cell i and cell j in physical space
- $\beta > 0$ = learned scalar parameter

**Effect:** Since softmax is monotonic, subtracting $\beta d_{ij}$ reduces the attention weight for distant cells. Larger $\beta$ means sharper spatial decay. The model learns $\beta$ to find the optimal spatial scale for niche influence.

**Biological motivation:** Cell-cell signaling (cytokines, ligand-receptor) decays with distance. This inductive bias encodes that prior.

**Connection to AMICI:** This formulation follows Gabor et al.'s AMICI (Attentive MIL for Cell-Cell Interaction), which showed distance-aware attention improves cell communication modeling.

---

## Panel c: Gromov-Wasserstein Fusion

### Equation

$$\min_{T \in \Pi(\mu, \nu)} \sum_{i,j,k,l} |d_H(i,j) - d_L(k,l)|^2 \cdot T_{ik} \cdot T_{jl}$$

### Mathematical Explanation

**The problem:** We have two point clouds in different spaces:
- HLCA embeddings: $\{h_i\}_{i=1}^n \subset \mathbb{R}^{30}$ (healthy reference)
- LuCA embeddings: $\{l_k\}_{k=1}^m \subset \mathbb{R}^{10}$ (cancer reference)

Standard optimal transport requires a ground cost c(i,k) between points, but these live in incompatible spaces with different dimensions.

**Gromov-Wasserstein solution:** Instead of comparing points directly, compare *pairwise distances within each space*:
- $d_H(i,j) = \|h_i - h_j\|$ = distance between points i,j in HLCA space
- $d_L(k,l) = \|l_k - l_l\|$ = distance between points k,l in LuCA space

**The transport plan** $T \in \mathbb{R}^{n \times m}_+$ is a coupling matrix where:
- $T_{ik}$ = how much mass flows from HLCA point i to LuCA point k
- Row sums match source marginal: $\sum_k T_{ik} = \mu_i$
- Column sums match target marginal: $\sum_i T_{ik} = \nu_k$

**The objective** penalizes couplings that distort distance structure:
- If $d_H(i,j)$ is small (i,j close in HLCA) but $d_L(k,l)$ is large (their images k,l far in LuCA), the term $|d_H(i,j) - d_L(k,l)|^2$ is large
- Weighting by $T_{ik} T_{jl}$ means this penalty only matters when i→k and j→l have significant transport mass

**Intuition:** GW finds a "soft assignment" between spaces that preserves neighborhood structure. If two cells are similar in the healthy atlas, their cancer-atlas counterparts should also be similar.

**Output:** The fused embedding $z_f \in \mathbb{R}^{40}$ is formed by concatenating the aligned representations (or using the barycentric projection from the coupling).

---

## Panel d: CrossAttentionDrift Network

### Equation

$$v_\theta = g \cdot v_{\text{ctx}} + (1-g) \cdot v_{\text{lat}}$$

where $g = \sigma(f(x_t, \tau, s)) \in [0,1]$

### Mathematical Explanation

**Inputs:**
- $x_t \in \mathbb{R}^D$ = cell state at flow time t
- $\tau \in \mathbb{R}^{d_\tau}$ = sinusoidal time embedding
- $\mathbf{c} \in \mathbb{R}^D$ = niche context vector (from encoder)
- $s \in \mathbb{R}^{d_s}$ = disease stage embedding

**Two parallel pathways:**

1. **Context-conditioned path** (cross-attention):
   - Query: $Q = W_Q [x_t; \tau]$ (concatenation of state and time)
   - Key/Value: $K = W_K \mathbf{c}, \quad V = W_V \mathbf{c}$
   - Output: $v_{\text{ctx}} = \text{FFN}(\text{CrossAttn}(Q, K, V))$

2. **Latent-only path** (MLP):
   - $v_{\text{lat}} = \text{MLP}([x_t; \tau; s])$
   - No access to context $\mathbf{c}$

**Gating mechanism:**
- $g = \sigma(f(x_t, \tau, s))$ where $\sigma$ is sigmoid
- When $g \to 1$: output dominated by context-aware $v_{\text{ctx}}$
- When $g \to 0$: output dominated by context-free $v_{\text{lat}}$

**Why two paths?** 
- Some cell state changes are niche-dependent (e.g., immune evasion triggered by local TME)
- Some are cell-autonomous (e.g., cell cycle, intrinsic differentiation programs)
- The gate learns which regime applies at each point in state-time space

**Biological interpretation:** In early precursor stages, niche context (IL1B signaling, immune interactions) may strongly influence trajectory. In late-stage autonomous tumor cells, intrinsic programs dominate. The gate captures this transition.

---

## Panel e: Optimal Transport Conditional Flow Matching

### Equation

$$\mathcal{L} = \mathbb{E}_{t \sim U[0,1], \, (x_0, x_1) \sim \pi^*}\left[\|v_\theta(x_t, t \mid \mathbf{c}) - (x_1 - x_0)\|^2\right]$$

where $x_t = (1-t)x_0 + tx_1$ (linear interpolation)

### Mathematical Explanation

**Flow matching setup:**
- Source distribution: $p_0$ (e.g., healthy/early cell states)
- Target distribution: $p_1$ (e.g., cancer/late cell states)
- Goal: learn a vector field $v_\theta$ that transports $p_0$ to $p_1$

**Why optimal transport coupling?**
Standard flow matching samples $(x_0, x_1)$ independently from $p_0$ and $p_1$. This creates crossing trajectories and inefficient transport.

OT-CFM uses the **optimal coupling** $\pi^* \in \Pi(p_0, p_1)$:

$$\pi^* = \arg\min_{\pi} \mathbb{E}_{(x_0, x_1) \sim \pi}[\|x_1 - x_0\|^2]$$

This is the 2-Wasserstein optimal transport plan, computed via Sinkhorn algorithm (entropic regularization for efficiency).

**The conditional flow:**
Given a coupled pair $(x_0, x_1) \sim \pi^*$, the interpolant is:

$$x_t = (1-t)x_0 + tx_1$$

The **conditional velocity field** for this pair is constant:

$$u_t(x \mid x_0, x_1) = x_1 - x_0$$

**Training objective:**
The model $v_\theta$ learns to predict this conditional velocity:

$$\mathcal{L} = \mathbb{E}\left[\|v_\theta(x_t, t \mid \mathbf{c}) - (x_1 - x_0)\|^2\right]$$

**Why this works (Lipman et al., 2023):**
The marginal vector field $u_t(x) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$ generates the same flow as solving the continuity equation. Training on conditional paths recovers the marginal field.

**Conditioning on context:**
The context $\mathbf{c}$ makes this a *conditional* flow matching problem:
- Different niches → different velocity fields
- The model learns: "given THIS niche context, which direction should the cell move?"

**Inference:**
At test time, solve the ODE:

$$\frac{dx}{dt} = v_\theta(x, t \mid \mathbf{c}), \quad x(0) = x_0$$

to transport a source state to its predicted target state given niche context.

---

## Symbol Reference

| Symbol | Domain | Meaning |
|--------|--------|---------|
| $x_t$ | $\mathbb{R}^D$ | Cell state at flow time t |
| $v_\theta$ | $\mathbb{R}^D$ | Learned velocity field |
| $\pi^*$ | $\mathbb{R}^{n \times m}_+$ | Optimal transport coupling |
| $T$ | $\mathbb{R}^{n \times m}_+$ | GW transport plan |
| $d_H, d_L$ | $\mathbb{R}_+$ | Pairwise distances in HLCA/LuCA |
| $\alpha_j$ | $[0,1]$ | Attention weight (sums to 1) |
| $\beta$ | $\mathbb{R}_+$ | Learned distance decay |
| $g$ | $[0,1]$ | Sigmoid gate value |
| $\mathbf{c}$ | $\mathbb{R}^D$ | Niche context vector |

---

## Key References

- **Set Transformer:** Lee et al., 2019 - ISAB and PMA for set-input networks
- **AMICI:** Gabor et al., 2023 - Distance-aware attention for cell-cell interaction
- **Gromov-Wasserstein:** Mémoli, 2011; Peyré et al., 2016 - OT between incompatible spaces
- **Flow Matching:** Lipman et al., 2023 - Training continuous normalizing flows
- **OT-CFM:** Tong et al., 2023 - Optimal transport for flow matching
