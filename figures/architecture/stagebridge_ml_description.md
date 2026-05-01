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

## Equation Explanations for Presentation

### Panel b: Distance-Penalized Attention

$$\alpha = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} - \beta \cdot \mathbf{d}\right)$$

**Read as:** "Alpha equals softmax of Q K transpose over root d, minus beta times d."

**The standard attention mechanism** computes a compatibility score between a query vector and a set of key vectors. The dot product $q_i^\top k_j$ measures how aligned two vectors are - high values mean similar directions in embedding space. The softmax normalizes these scores to sum to 1, giving us a weighted average.

**The $\sqrt{d}$ scaling** prevents a subtle numerical problem: as embedding dimension $d$ grows, dot products grow proportionally (they're sums of $d$ terms). Large dot products push softmax into saturation where gradients vanish. Dividing by $\sqrt{d}$ keeps the variance roughly constant regardless of dimension. This is why it's called "scaled dot-product attention."

**The distance penalty $-\beta \cdot d_{ij}$** is our key modification. Standard attention treats all tokens equally regardless of physical location. But cell-cell communication decays with distance - cytokines diffuse, ligand-receptor binding requires proximity, juxtacrine signaling needs contact. By subtracting a term proportional to physical distance, we encode this prior directly into the attention computation.

**Why subtraction works:** Softmax is monotonic, so subtracting $\beta d_{ij}$ reduces the attention weight for distant cells exponentially (since softmax exponentiates its inputs). The learned parameter $\beta > 0$ controls the decay rate - small $\beta$ means long-range interactions matter, large $\beta$ means only immediate neighbors contribute.

**Biological interpretation:** The model learns the characteristic length scale of niche influence. For paracrine signaling (IL1B, CCL2), this might be 50-100 microns. For contact-dependent signaling (Notch-Delta), it's essentially zero. The single $\beta$ parameter captures an effective average, though in principle you could have pathway-specific decay rates.

**Connection to AMICI (Gabor et al.):** This formulation follows their attentive MIL framework for cell-cell interaction, which demonstrated that distance-aware attention significantly improves prediction of communication effects.

---

### Panel c: Gromov-Wasserstein Fusion

$$\min_{T \in \Pi(\mu, \nu)} \sum_{i,j,k,l} |d_H(i,j) - d_L(k,l)|^2 \cdot T_{ik} \cdot T_{jl}$$

**Read as:** "Minimize over transport plans T the sum of squared distance distortions, weighted by the coupling."

**The problem we're solving:** We have cell embeddings from two reference atlases - HLCA (healthy lung, 30-dimensional) and LuCA (lung cancer, 10-dimensional). These live in completely different spaces with different dimensions, different genes used for embedding, different batch effects. We cannot directly compare a point in HLCA space to a point in LuCA space.

**Standard optimal transport** finds a coupling between two distributions by minimizing total transport cost $\sum_{ik} c(i,k) T_{ik}$, where $c(i,k)$ is the cost of moving mass from point $i$ to point $k$. But this requires a ground cost function between the spaces - which we don't have.

**Gromov-Wasserstein's insight:** Instead of comparing points across spaces, compare *relationships within* each space. The distance $d_H(i,j)$ tells us how similar cells $i$ and $j$ are in the healthy atlas. The distance $d_L(k,l)$ tells us the same for the cancer atlas. GW asks: can we find a soft assignment $T$ such that similar cells in HLCA get mapped to similar cells in LuCA?

**Unpacking the objective:**
- $T_{ik}$ is the coupling weight between HLCA cell $i$ and LuCA cell $k$
- The product $T_{ik} \cdot T_{jl}$ weights pairs of pairs
- $|d_H(i,j) - d_L(k,l)|^2$ measures how much the mapping distorts distances
- If $i,j$ are close in HLCA but their images $k,l$ are far in LuCA, this term is large

**The constraint set $\Pi(\mu, \nu)$:** Valid transport plans must satisfy marginal constraints - the total mass leaving each source point equals $\mu_i$, the total mass arriving at each target equals $\nu_k$. For uniform distributions, this means each cell contributes equally.

**Computational note:** The quartic sum looks expensive, but entropic regularization + Sinkhorn iterations make this tractable. We use the POT library's `gromov_wasserstein` with $\epsilon = 0.1$.

**Why this matters biologically:** A cell's identity is partly defined by its relationship to other cell types. A macrophage in healthy tissue and a TAM in tumor tissue may have different absolute expression profiles, but both sit in similar positions relative to their respective epithelial and stromal neighbors. GW captures this relational structure.

**Output:** The barycentric projection gives us a fused embedding that places each query cell in a joint space respecting both atlas geometries.

---

### Panel d: Gated Drift Network

$$v_\theta = g \cdot v_{\text{ctx}} + (1-g) \cdot v_{\text{lat}}$$

where $g = \sigma(f(x_t, \tau, s)) \in [0,1]$

**Read as:** "The predicted velocity is a convex combination of context-dependent and context-independent terms, with the mixing weight learned as a function of cell state, time, and stage."

**The modeling assumption:** Cell state transitions have two components:
1. **Niche-dependent dynamics:** The local microenvironment influences cell fate. IL1B from nearby macrophages activates inflammatory programs. CAF-secreted factors drive EMT. These are extrinsic signals.
2. **Cell-autonomous dynamics:** Intrinsic programs proceed regardless of context - cell cycle, terminal differentiation, oncogene-driven proliferation. A KRAS-mutant cell will divide whether or not it receives niche signals.

**The architecture implements this decomposition:**
- $v_{\text{ctx}}$ comes from cross-attention over the niche context vector $\mathbf{c}$. The query is the current cell state; keys/values are the encoded neighborhood. This pathway can only produce niche-informed predictions.
- $v_{\text{lat}}$ comes from an MLP operating on $[x_t; \tau; s]$ with no access to $\mathbf{c}$. This pathway captures cell-autonomous dynamics.

**The gate $g$:** Rather than hard-coding when context matters, we let the model learn it. The sigmoid $\sigma$ squashes the output to $[0,1]$, ensuring the combination is convex. The gate takes as input:
- $x_t$: current cell state (different cell types may be more/less niche-responsive)
- $\tau$: flow time (early vs late in the transition)
- $s$: disease stage (normal tissue vs advanced tumor)

**Expected behavior:** We hypothesize that $g$ will be high (context matters) for:
- Early-stage precursor lesions where niche remodeling drives fate decisions
- Epithelial cells in inflammatory microenvironments
- Transition states where cells are "deciding" their trajectory

And $g$ will be low (context-free) for:
- Late-stage autonomous tumor cells
- Terminally differentiated cells
- Cells mid-cell-cycle

**Interpretability:** After training, we can plot $g$ across cell states and stages. High-$g$ regions are where therapeutic niche modulation might redirect trajectories; low-$g$ regions require cell-intrinsic targeting.

---

### Panel e: Optimal Transport Conditional Flow Matching

$$\mathcal{L} = \mathbb{E}_{t \sim U[0,1], \, (x_0, x_1) \sim \pi^*}\left[\|v_\theta(x_t, t \mid \mathbf{c}) - (x_1 - x_0)\|^2\right]$$

where $x_t = (1-t)x_0 + tx_1$

**Read as:** "The loss is the expected squared error between the predicted velocity and the ground-truth velocity along optimally-coupled straight-line paths."

**The flow matching framework (Lipman et al., 2023):**
We want to learn a continuous transformation from source distribution $p_0$ (e.g., healthy/early cells) to target distribution $p_1$ (e.g., cancer/late cells). Instead of learning the density directly, we learn a *velocity field* $v_\theta(x,t)$ that tells us which direction to move at each point in state-time space.

**The key insight:** Given a coupling $(x_0, x_1)$, the simplest path from $x_0$ to $x_1$ is a straight line:
$$x_t = (1-t)x_0 + tx_1$$

The velocity along this path is constant:
$$\frac{dx_t}{dt} = x_1 - x_0$$

So we train the model to predict this constant velocity at every point along the path.

**Why optimal transport coupling?**
If we sample $x_0$ and $x_1$ independently, paths can cross - one cell might be told to go left while another at the same location is told to go right. This creates conflicting gradients and poor training dynamics.

OT coupling $\pi^*$ solves:
$$\pi^* = \arg\min_{\pi \in \Pi(p_0, p_1)} \mathbb{E}_{(x_0,x_1) \sim \pi}[\|x_1 - x_0\|^2]$$

This pairs source and target points to minimize total squared displacement - the 2-Wasserstein distance. Intuitively, nearby cells get paired with nearby targets, eliminating path crossings.

**The conditioning on $\mathbf{c}$:**
Standard flow matching learns a single velocity field for the whole distribution. We learn a *family* of velocity fields, one for each possible niche context $\mathbf{c}$. The model predicts: "Given THIS neighborhood, which direction should this cell move?"

This is the core of our approach - the same cell state $x_t$ can have different predicted velocities depending on its microenvironment.

**Inference procedure:**
Given a source cell state $x_0$ and its niche context $\mathbf{c}$, we solve the ODE:
$$\frac{dx}{dt} = v_\theta(x, t \mid \mathbf{c}), \quad x(0) = x_0$$

using a numerical integrator (we use `torchdiffeq`'s dopri5). The endpoint $x(1)$ is the predicted target state.

**Biological interpretation:**
- $t=0$: cell is in source state (e.g., AT2 progenitor)
- $t=1$: cell reaches target state (e.g., LUAD)
- Intermediate $t$: transition states along the trajectory
- Different $\mathbf{c}$: same cell in different niches follows different trajectories

**Why not diffusion models?**
Flow matching has straighter paths (more interpretable), simpler training (no noise schedule tuning), and faster inference (fewer integration steps). For our biological interpretation goals, the straight-path prior is actually desirable.

---

## Symbol Reference (Quick Lookup)

| Symbol | Domain | Meaning |
|--------|--------|---------|
| $x_t$ | $\mathbb{R}^D$ | Cell state at flow time $t$ (gene expression embedding) |
| $x_0, x_1$ | $\mathbb{R}^D$ | Source and target cell states |
| $v_\theta$ | $\mathbb{R}^D \to \mathbb{R}^D$ | Learned velocity field (parameterized by $\theta$) |
| $\pi^*$ | $\mathbb{R}^{n \times m}_+$ | Optimal transport coupling matrix |
| $T$ | $\mathbb{R}^{n \times m}_+$ | Gromov-Wasserstein transport plan |
| $d_H, d_L$ | $\mathbb{R}_+$ | Pairwise distances in HLCA/LuCA embedding spaces |
| $\alpha$ | $\Delta^{n-1}$ | Attention weights (probability simplex, sums to 1) |
| $\beta$ | $\mathbb{R}_+$ | Learned spatial decay parameter |
| $g$ | $[0,1]$ | Gating value (context vs autonomous) |
| $\mathbf{c}$ | $\mathbb{R}^D$ | Niche context vector (encoder output) |
| $\tau$ | $\mathbb{R}^{d_\tau}$ | Sinusoidal time embedding |
| $s$ | $\mathbb{R}^{d_s}$ | Disease stage embedding |
| $Q, K, V$ | $\mathbb{R}^{n \times d}$ | Query, Key, Value matrices in attention |
| $\mathcal{L}$ | $\mathbb{R}$ | Training loss (to minimize) |

---

## Connections to Pathway Biology

### IL1B-IL1R1 Axis and the Distance Penalty

The distance-penalized attention directly relates to IL1B signaling biology:

**IL1B is a paracrine cytokine** secreted primarily by activated macrophages. It diffuses through the tissue and binds IL1R1 on nearby epithelial cells. The effective signaling range is limited by:
- Diffusion coefficient in extracellular matrix (~10-100 μm²/s)
- Receptor binding kinetics
- Proteolytic degradation

**What the model should learn:** The $\beta$ parameter for IL1B-related attention heads should reflect this ~50-100 μm effective range. We can validate this post-training by:
1. Identifying attention heads that correlate with IL1B expression in sender cells
2. Examining their learned $\beta$ values
3. Comparing to known IL1B diffusion ranges from the literature

**Prediction:** In AAH/AIS lesions where IL1B-macrophage niches are enriched (Peng et al.), we expect:
- High attention weights between IL1B+ macrophages and nearby epithelial cells
- The gate $g$ should be high for epithelial cells in these niches
- Flow trajectories conditioned on IL1B+ niches should show inflammatory/EMT signatures

### EMT and the Gated Drift

Epithelial-mesenchymal transition is a paradigmatic example of niche-dependent vs cell-autonomous dynamics:

**Niche-dependent EMT (partial, reversible):**
- TGFβ from CAFs
- IL6 from macrophages  
- Hypoxia in poorly vascularized regions
- These cells should have high $g$ values

**Cell-autonomous EMT (full, often irreversible):**
- SNAI1/SNAI2 constitutive activation
- Epigenetic locking of mesenchymal state
- Late-stage tumor cells with autocrine TGFβ
- These cells should have low $g$ values

**Validation approach:** Compute EMT scores (from gene signatures) and correlate with learned $g$ values across the disease stages. We expect the correlation to flip: early stages show niche-dependent partial EMT (high $g$), late stages show autonomous full EMT (low $g$).

### PAGA/Pseudotime Connection

Our flow matching objective learns a velocity field that, when integrated, gives trajectories. This relates to but differs from pseudotime:

**Pseudotime (DPT, Palantir):**
- Computed from static snapshot via diffusion on kNN graph
- Assumes smooth manifold structure
- No explicit dynamics - just ordering
- Cannot model branching or niche-dependence

**Flow matching trajectories:**
- Learns explicit velocity field from cross-sectional data
- Can model branching (different $\mathbf{c}$ → different endpoints)
- Provides interpretable "forces" via $v_\theta$
- The OT coupling addresses the identifiability issue: we're not claiming these are true temporal dynamics, but rather the minimum-energy transport that's consistent with the observed distributions

**Complementary use:**
- Use PAGA to identify coarse-grained connectivity between stages
- Use DPT to establish a reference ordering
- Use flow matching to model how niche context modulates the transitions PAGA identifies
- Validate that flow trajectories respect the PAGA topology

### CAF Interactions and Gromov-Wasserstein

Cancer-associated fibroblasts illustrate why we need GW fusion:

**In HLCA (healthy):** Fibroblasts are relatively quiescent, maintaining ECM homeostasis. Their embedding reflects normal stromal identity.

**In LuCA (cancer):** CAFs are activated, with distinct subtypes (myofibroblastic, inflammatory, antigen-presenting). Their embedding reflects tumor-educated states.

**The alignment problem:** A query fibroblast from our spatial data could be:
- Normal fibroblast → should align with HLCA fibroblast cluster
- Early CAF → should align with both (transitional)
- Late CAF → should align with LuCA CAF cluster

**GW's contribution:** By preserving distance structure, GW ensures that:
- Cells similar in HLCA space get similar LuCA projections
- The transition from normal→CAF is smooth in the fused space
- We can identify cells that "break" the alignment (truly novel states not in either atlas)

**Downstream use:** The GW-fused embedding feeds into the niche context $\mathbf{c}$. This means the model knows not just "this neighbor is a fibroblast" but "this neighbor is a fibroblast that looks more like LuCA-CAF than HLCA-normal."

---

## Key References

- **Set Transformer:** Lee et al., 2019 - ISAB and PMA for set-input networks
- **AMICI:** Gabor et al., 2023 - Distance-aware attention for cell-cell interaction
- **Gromov-Wasserstein:** Mémoli, 2011; Peyré et al., 2016 - OT between incompatible spaces
- **Flow Matching:** Lipman et al., 2023 - Training continuous normalizing flows
- **OT-CFM:** Tong et al., 2023 - Optimal transport for flow matching
- **IL1B in lung precancers:** Peng et al., 2023 - Proinflammatory niches in AAH/AIS
- **CAF heterogeneity:** Elyada et al., 2019 - CAF subtypes in pancreatic cancer (applicable paradigm)

---

## Anticipated Questions & Talking Points

### On the Overall Approach

**Q: "Why not just use pseudotime? RNA velocity? CellRank?"**

These methods have a fundamental limitation: they treat cells as isolated. Pseudotime orders cells along a manifold but doesn't ask *why* a cell progresses. RNA velocity infers direction from splicing but assumes cell-autonomous dynamics. CellRank combines velocity with transition probabilities but still doesn't condition on microenvironment.

Our key claim is that progression isn't just about where a cell is in expression space - it's about what signals it's receiving from neighbors. The same AT2 cell in two different niches may have very different fates. None of the existing methods can model this.

**Q: "This seems very complex. Why not a simpler model?"**

Each component addresses a specific biological reality:
- Set Transformer: neighborhoods have variable sizes (can't use fixed architectures)
- Distance penalty: signaling decays with distance (can't treat all neighbors equally)  
- GW fusion: atlases are in different spaces (can't just concatenate)
- Gated drift: some dynamics are niche-dependent, some aren't (can't assume one or the other)

We tried simpler versions. Removing any component hurts performance on the validation tasks. The complexity is necessary, not ornamental.

**Q: "How do you know you're learning real biology vs artifacts?"**

Three-level validation:
1. **Known biology recovery**: Does the model recapitulate IL1B-IL1R1 interactions, KAC progenitor states, CAF-epithelial crosstalk from Peng et al.?
2. **Attention interpretability**: Do high-attention edges correspond to known ligand-receptor pairs? Does $\beta$ match expected signaling ranges?
3. **Perturbation prediction**: If we mask out macrophages from a niche, does the predicted trajectory change in the expected direction (less inflammatory)?

---

### On Distance-Penalized Attention

**Q: "How do you choose $\beta$? Isn't it sensitive to scale?"**

$\beta$ is learned, not chosen. The model finds the optimal decay rate for the prediction task. We initialize it to give reasonable decay over the typical cell-cell distance range (~10-200 μm), but it adapts during training.

For scale: we normalize distances to the 95th percentile within each sample, so $\beta$ is in comparable units across tissue sections with different resolutions.

**Q: "What if different pathways have different ranges?"**

Good question - our current model uses a single $\beta$. This is a simplification. In principle, you could have pathway-specific or head-specific decay rates. We chose single $\beta$ for interpretability and because multi-head attention already provides some flexibility (different heads can weight distance differently via the QK term).

Future work: learn per-head $\beta$ values and see if they cluster by pathway type.

**Q: "Doesn't attention already handle distance implicitly via expression similarity?"**

Not reliably. Two cells could have similar expression (high QK) but be on opposite sides of the tissue. Standard attention would weight them equally. Physical distance is independent information that pure expression-based attention misses.

The subtraction makes it explicit: you need both semantic similarity AND physical proximity to get high attention.

---

### On Gromov-Wasserstein Fusion

**Q: "Why not just project both atlases into a shared space first?"**

That requires choosing a projection method (CCA, Harmony, scVI integration). Each makes assumptions about what should align. GW is assumption-light - it only asks that distance structure be preserved, not that specific genes or cell types match.

Also, HLCA and LuCA were built with different gene sets, normalization, and cell type granularity. Forcing them into shared coordinates loses information. GW lets them stay in their native spaces while still enabling comparison.

**Q: "GW is computationally expensive. How do you scale it?"**

Three strategies:
1. Entropic regularization (Sinkhorn): $O(n^2)$ per iteration instead of $O(n^3)$ for exact
2. Subsampling: compute GW on representative cells (cluster centroids or landmarks), interpolate for rest
3. Precomputation: atlas-to-atlas alignment is computed once, then query cells are projected via barycentric mapping

In practice, GW is not the bottleneck - the flow matching training is.

**Q: "What if the atlases have different cell type compositions?"**

GW handles this naturally through the marginal constraints. If HLCA has more AT2 cells and LuCA has more tumor cells, the transport plan will be many-to-many rather than one-to-one. Cells with no good match in the other atlas get diffuse transport mass - which is informative (novel states).

---

### On the Gated Drift

**Q: "How do you know the gate is learning something meaningful vs just fitting noise?"**

Post-hoc validation:
1. Plot $g$ values across known biological axes (stage, cell type, EMT score)
2. Check if high-$g$ regions correspond to known niche-dependent processes
3. Ablation: freeze $g=0.5$ and see if performance drops

If $g$ is just noise, it should be uniform. If it's meaningful, it should correlate with biological covariates we didn't explicitly train on.

**Q: "Why a gate instead of just adding context and latent pathways?"**

Addition doesn't work well - the magnitudes can fight. If $v_{\text{ctx}}$ says "go left" and $v_{\text{lat}}$ says "go right," addition gives "stay put," which is probably wrong.

The gate forces a choice: at each point, decide how much context matters, then blend accordingly. This is more biologically interpretable - we can point to cells where niche dominates vs cells that are autonomous.

**Q: "What if a cell needs BOTH context-dependent and autonomous signals?"**

The gate value $g \in (0,1)$ allows blending. $g=0.7$ means "70% context-driven, 30% autonomous." This captures the reality that most cells integrate both types of signals.

Only at the extremes ($g \approx 0$ or $g \approx 1$) is it purely one or the other.

---

### On Flow Matching / OT-CFM

**Q: "You're learning from cross-sectional data. How can you claim anything about dynamics?"**

We're not claiming to recover true temporal dynamics. We're learning the *minimum-energy transport* between observed distributions that's consistent with niche conditioning.

The OT coupling is key: it pairs cells to minimize total displacement. This is a principled way to infer "what probably came from what" without longitudinal data.

Our claim is weaker but still useful: "Cells in THIS niche context are more likely to transition toward THAT state." Not "this specific cell will become that specific cell in 3 days."

**Q: "Why flow matching instead of diffusion models?"**

Three reasons:
1. **Interpretability**: Flow matching learns straight-line paths (geodesics under OT). Diffusion models learn curved, noisy paths that are harder to visualize.
2. **Training simplicity**: No noise schedule to tune. Just predict the velocity.
3. **Inference speed**: Flow matching needs ~10-20 ODE steps. Diffusion needs ~50-1000 denoising steps.

For our biological interpretation goals, straighter = better.

**Q: "What's the source and target distribution?"**

Flexible by design:
- Default: source = Normal/AAH, target = AIS/LUAD (progression)
- Can also do: source = one stage, target = next stage (pairwise)
- Or: source = all early, target = all late (coarse)

The OT coupling adapts to whatever you specify. We train with stage-aware sampling to ensure balanced coverage.

**Q: "How do you handle branching trajectories?"**

The conditioning on $\mathbf{c}$ handles this. Same source cell with different niche contexts → different predicted velocities → different endpoints.

At inference: sample multiple possible niche contexts for a cell, integrate each, get a distribution of possible fates. Branching is implicit in the context diversity.

---

### On Validation & Biological Claims

**Q: "How do you validate the IL1B-IL1R1 finding?"**

Multiple levels:
1. **Attention check**: For cells expressing IL1R1, do they attend strongly to IL1B+ macrophages within signaling range?
2. **LIANA comparison**: Does our attention-weighted L-R score correlate with LIANA's statistical score?
3. **Differential attention**: Is IL1B→IL1R1 attention higher in AAH/AIS (where Peng et al. saw enrichment) than in Normal or late LUAD?
4. **Trajectory effect**: When we condition on IL1B+ vs IL1B- niches, do trajectory endpoints differ in inflammatory signature?

**Q: "What's your ground truth for trajectory validation?"**

No single ground truth exists for human cancer progression. We use:
1. **Stage ordering**: Trajectories should respect Normal→AAH→AIS→MIA→LUAD on average
2. **Known markers**: EMT score should increase along trajectories to mesenchymal states
3. **CytoTRACE**: Stemness should decrease along differentiation trajectories
4. **Clonal data (where available)**: Cells sharing mutations should have consistent trajectory directions

**Q: "How do you separate correlation from causation in niche effects?"**

We can't definitively prove causation from observational data. But we can:
1. **Control for confounders**: Stage, cell type, sample batch in the model
2. **Perturbation simulation**: Mask niche components and check trajectory changes
3. **Literature validation**: Do our predicted causal directions match known biology?

The gate $g$ is particularly useful here: high $g$ + high attention to a specific neighbor type suggests that neighbor type *matters* for this cell's trajectory, even if we can't prove directionality.

---

### Hardball Questions (Be Ready)

**Q: "Isn't this just overfitting to your specific dataset?"**

Mitigation:
- Held-out samples (not just cells) for validation
- Cross-validation across patients
- Testing on external LUAD datasets (TCGA, other spatial cohorts)
- Synthetic data benchmarks with known ground truth

**Q: "Why should I believe learned $\beta$ reflects biology vs training artifact?"**

Compare to biophysical estimates. IL1B diffusion in tissue is ~10-100 μm²/s with half-life of hours → effective range ~50-200 μm. If learned $\beta$ gives similar decay length, that's validation. If it's wildly different, we have a problem.

**Q: "Your model has a lot of parameters. How do you avoid overfitting?"**

- Dropout (12.7% in current config)
- Early stopping on validation loss
- SSL pretraining provides regularization (encoder sees all data, not just labeled)
- The OT coupling itself is a form of regularization (prevents arbitrary path crossings)

**Q: "What's the failure mode? When would this approach not work?"**

Honest answer:
- If niche truly doesn't matter (pure cell-autonomous progression), we're adding complexity for nothing
- If spatial resolution is too coarse (Visium spots mix many cells), niche signal is diluted
- If disease stages are mislabeled, the flow direction is wrong
- If the atlases are poor quality or missing key cell types, GW alignment fails

We validate extensively to check we're not in these regimes.

---

### Quick Comebacks (Memorize These)

| Challenge | Response |
|-----------|----------|
| "Too complex" | "Each component addresses a specific biological reality. We ablated - removing any hurts." |
| "Just correlation" | "True for all observational studies. We validate against known causal biology from perturbation papers." |
| "Why not method X?" | "X doesn't condition on microenvironment. That's our core claim - niche matters for trajectory." |
| "How do you know it's real?" | "Three levels: recover known biology, attention matches L-R pairs, perturbation changes trajectory as expected." |
| "Isn't this just fancy pseudotime?" | "Pseudotime orders cells. We predict *directions* conditioned on context. Different question." |
| "What's novel?" | "Receiver-centered niche attention + dual-atlas fusion + niche-conditioned flow. No one's combined these." |
| "Clinical relevance?" | "Identify which niches drive progression → target those niches for interception before cancer." |
