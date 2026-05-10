# StageBridge Key Equations

A breakdown of every equation in the codebase with derivations, explanations, and variable tables.

**References:**
- Hong J, Desai K, Nguyen TD, Nazaret A, Levy N, Ergen C, Plitas G, Azizi E. AMICI: Attention Mechanism Interpretation of Cell-cell Interactions. bioRxiv 2025. doi:10.1101/2025.09.22.677860. https://github.com/azizilab/amici. License: CC BY-NC-ND 4.0. Patent pending (U.S. Serial No. 63/884,704).
- Lee J, Lee Y, Kim J, Kosiorek A, Choi S, Teh YW. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. ICML 2019.
- Lipman Y, Chen RT, Ben-Hamu H, Nickel M, Le M. Flow Matching for Generative Modeling. ICLR 2023.
- Peyré G, Cuturi M. Computational Optimal Transport. 2019.

---

## Table of Contents

1. [Multihead Attention (Foundation)](#1-multihead-attention-the-foundation)
2. [Self-Attention Block (SAB)](#2-self-attention-block-sab)
3. [Induced Set Attention Block (ISAB)](#3-induced-set-attention-block-isab)
4. [Pooling by Multihead Attention (PMA)](#4-pooling-by-multihead-attention-pma)
5. [AMICI Receiver-Centered Attention](#5-amici-receiver-centered-attention)
6. [Empty Neighbor Token](#6-empty-neighbor-token)
7. [Sinusoidal Time Embedding](#7-sinusoidal-time-embedding)
8. [Optimal Transport Coupling](#8-optimal-transport-coupling)
9. [Entropic Regularization (Sinkhorn)](#9-entropic-regularization-sinkhorn)
10. [Flow Matching Loss (OT-CFM)](#10-flow-matching-loss-ot-cfm)
11. [Inference (ODE Integration)](#11-inference-ode-integration)
12. [CrossAttentionDrift (Gated Context)](#12-crossattentiondrift-gated-context)
13. [FiLM Conditioning](#13-film-conditioning)
14. [Learned Gromov-Wasserstein Fusion](#14-learned-gromov-wasserstein-fusion)
15. [Wasserstein Distance (Evaluation)](#15-wasserstein-distance-evaluation)
16. [Full Pipeline Summary](#16-full-pipeline-summary)
17. [Complexity Reference](#17-complexity-reference)
18. [Q&A Cheat Sheet](#18-qa-cheat-sheet)

---

## 1. Multihead Attention (the foundation)

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| Q | [n × d] | Query matrix | `q = self.q_proj(receiver)` |
| K | [m × d] | Key matrix | `k = self.k_proj(neighbors)` |
| V | [m × d] | Value matrix | `v = self.v_proj(neighbors)` |
| d_k | scalar | Key dimension (= d / num_heads) | `self.head_dim = dim // num_heads` |
| QKᵀ | [n × m] | Attention scores (dot product similarity) | `torch.matmul(q, k.transpose(-2, -1))` |
| √d_k | scalar | Scaling factor to prevent gradient explosion | `self.scale = head_dim ** -0.5` |
| softmax | row-wise | Normalizes scores to sum to 1 | `F.softmax(attn_logits, dim=-1)` |

**Derivation:** The dot product QKᵀ grows with dimension d, which pushes softmax into saturation (tiny gradients). Dividing by √d_k keeps variance ~1 regardless of d.

**How to explain:**
> "The query asks 'what's relevant to me?', the keys answer 'here's what I have', and the output is a weighted combination of values where the weights come from query-key similarity, normalized to sum to 1."

---

## 2. Self-Attention Block (SAB)

```
SAB(X) = LN(X + MHA(X, X, X))
       = LN(X + softmax(XW_Q · (XW_K)ᵀ / √d) · XW_V)
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| X | [n × d] | Input tokens | `x` parameter |
| W_Q, W_K, W_V | [d × d] | Learned projection matrices | Inside `nn.MultiheadAttention` |
| MHA | - | Multihead attention (Q=K=V=X) | `self.mha(query=x, key=x, value=x)` |
| LN | - | Layer normalization | `self.ln1`, `self.ln2` |
| X + ... | [n × d] | Residual connection | `x = self.ln1(x + attn_out)` |

**Code:** `stagebridge/context/layers.py:36-69`

```python
def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
    attn_out, _ = self.mha(query=x, key=x, value=x, ...)
    x = self.ln1(x + attn_out)
    out = self.ln2(x + self.ff(x))
    return out
```

**How to explain:**
> "In self-attention, every token queries every other token. Each token gets updated by attending to the full set. The residual connection means we're adding new information, not replacing."

**Complexity:** O(n²) - every token attends to every other token.

---

## 3. Induced Set Attention Block (ISAB)

```
ISAB_m(X) = MAB(X, MAB(I, X))

where:
  MAB(A, B) = LN(A + MHA(A, B, B))
  I ∈ ℝ^{m × d} = learnable inducing points
```

**Step-by-step expansion:**

```
Step 1: H = MAB(I, X)
        H = LN(I + softmax(IW_Q · (XW_K)ᵀ / √d) · XW_V)
        # H is [m × d] - inducing points compress the set

Step 2: ISAB(X) = MAB(X, H)
        = LN(X + softmax(XW_Q' · (HW_K')ᵀ / √d) · HW_V')
        # Output is [n × d] - same size as input
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| I | [1 × m × d] | Learnable inducing points | `self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, dim) * 0.02)` |
| m | scalar | Number of inducing points (default 16) | `num_inducing_points` parameter |
| H | [B × m × d] | Compressed set representation | Result of `self.mha_1(query=inducing, key=x, value=x)` |
| MAB | - | Multihead Attention Block | Two MHA calls with different Q/K/V |

**Code:** `stagebridge/context/layers.py:95-168`

```python
def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
    inducing = self.inducing_points.expand(batch_size, -1, -1)
    
    # Step 1: Inducing points query the input
    h, _ = self.mha_1(query=inducing, key=x, value=x, ...)
    h = self.ln_h1(inducing + h)
    h = self.ln_h2(h + self.ff_h(h))
    
    # Step 2: Input queries the compressed representation
    x_attn, _ = self.mha_2(query=x, key=h, value=h, ...)
    x = self.ln_x1(x + x_attn)
    out = self.ln_x2(x + self.ff_x(x))
    return out
```

**How to explain:**
> "Instead of n² attention, we route through m inducing points. First, the inducing points attend to all inputs - this compresses the set into m summary vectors. Then, each input attends to these summaries. Total cost: O(nm) instead of O(n²). The inducing points learn to capture the sufficient statistics of the set."

**Complexity:** O(n·m) where m << n.

---

## 4. Pooling by Multihead Attention (PMA)

```
PMA_k(X) = MAB(S, X)
         = LN(S + softmax(SW_Q · (XW_K)ᵀ / √d) · XW_V)

where:
  S ∈ ℝ^{k × d} = learnable seed vectors
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| S | [1 × k × d] | Learnable seed vectors | `self.seed_vectors = nn.Parameter(torch.randn(1, num_seed_vectors, dim) * 0.02)` |
| k | scalar | Number of output vectors (typically 1) | `num_seed_vectors` parameter |
| Output | [B × k × d] | Fixed-size aggregation | Return value |

**Code:** `stagebridge/context/layers.py:171-218`

```python
def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
    seeds = self.seed_vectors.expand(batch_size, -1, -1)
    pooled, _ = self.mha(query=seeds, key=x, value=x, ...)
    pooled = self.ln1(seeds + pooled)
    out = self.ln2(pooled + self.ff(pooled))
    return out
```

**How to explain:**
> "PMA extracts a fixed-size output from a variable-size set. The seed vector learns to query for the most relevant summary. It's like asking 'give me THE important thing about this set' and letting attention figure out what that is. Unlike mean pooling, this is adaptive."

---

## 5. AMICI Receiver-Centered Attention

**This is the core novelty adapted from Hong et al. (2025).**

```
attn_logits = phenotype_score - distance_penalty

where:
  phenotype_score = (QKᵀ) / √d_k
  distance_penalty = b₁ · (dist / distance_scale)
  b₁ = softplus(MLP(receiver) + offset)
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| receiver | [B × D] | Target cell embedding (the "receiver") | `receiver` parameter |
| neighbors | [B × K × D] | K nearest neighbor embeddings | `neighbors` parameter |
| distances | [B × K] | Euclidean distances to each neighbor (μm) | `distances` parameter |
| Q | [B × H × 1 × d_h] | Receiver projected to query space | `q = self.q_proj(receiver).unsqueeze(1)` |
| K | [B × H × K × d_h] | Neighbors projected to key space | `k = self.k_proj(neighbors)` |
| V | [B × H × K × d_h] | Neighbors projected to value space | `v = self.v_proj(neighbors)` |
| H | scalar | Number of attention heads | `self.num_heads` |
| d_h | scalar | Head dimension (= D / H) | `self.head_dim = dim // num_heads` |
| phenotype_score | [B × H × 1 × K] | Query-key similarity | `torch.matmul(q, k.transpose(-2, -1)) * self.scale` |
| b₁ | [B × H] | Distance coefficient (enforced positive) | `distance_coef = F.softplus(distance_coef_raw)` |
| distance_scale | scalar | Normalization constant (default 100.0 μm) | `self.distance_scale` |
| distance_penalty | [B × H × 1 × K] | Distance-based attention reduction | `distance_coef.unsqueeze(-1) * normalized_dist.unsqueeze(1)` |
| attn_logits | [B × H × 1 × K] | Final attention logits before softmax | `phenotype_score - distance_penalty` |

**Code:** `stagebridge/context/encoder.py:195-282`

```python
def forward(self, receiver: Tensor, neighbors: Tensor, distances: Tensor, ...) -> tuple:
    # Project to multihead format
    q = self.q_proj(receiver).unsqueeze(1)  # [B, 1, D]
    k = self.k_proj(neighbors)               # [B, K, D]
    v = self.v_proj(neighbors)               # [B, K, D]
    
    # Reshape for multihead: [B, H, seq, d_h]
    q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
    
    # Phenotype similarity (standard attention)
    phenotype_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    
    # AMICI distance decay (CRITICAL: softplus enforces monotonic decay)
    distance_coef_raw = self.distance_coef_mlp(receiver) + self.distance_coef_offset
    distance_coef = F.softplus(distance_coef_raw)  # Always positive!
    
    normalized_dist = distances / self.distance_scale
    distance_penalty = distance_coef.unsqueeze(-1) * normalized_dist.unsqueeze(1)
    distance_penalty = distance_penalty.unsqueeze(2)
    
    # Subtract penalty (monotonic decay with distance)
    attn_logits = phenotype_score - distance_penalty
    attn_weights = F.softmax(attn_logits, dim=-1)
```

**Derivation of monotonic decay:**

The key constraint is that attention must monotonically decrease with distance. This is enforced architecturally:

1. `b₁ = softplus(raw)` ensures b₁ > 0 (softplus(x) = log(1 + exp(x)) is always positive)
2. `attn = phenotype - b₁ · dist` means higher distance → lower attention logit
3. softmax preserves ordering: lower logit → lower weight

This isn't learned freely - the architecture **guarantees** monotonic decay regardless of what the MLP outputs.

**Biological justification:**
> "Paracrine signaling decays with distance. A cell 10μm away sends stronger signals than one 50μm away. The learned b₁ controls HOW FAST attention decays, but it always decays. This matches known biology: ligand-receptor binding probability decreases with distance due to diffusion."

---

## 6. Empty Neighbor Token

```
attn_logits_extended = [attn_logits ; empty_score]

where:
  empty_score = constant (default 3.0)
  empty_value = 0 ∈ ℝ^{d_h}
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| empty_score | scalar | Fixed attention score for empty token | `self.empty_token_score` (default 3.0) |
| empty_value | [B × H × 1 × d_h] | Zero vector (contributes nothing) | `torch.zeros((B, self.num_heads, 1, self.head_dim), ...)` |
| empty_attention | [B] | Attention weight to empty token | `attn_weights_mean[:, -1]` |

**Code:** `stagebridge/context/encoder.py:235-249`

```python
if self.use_empty_token:
    empty_score = torch.full(
        (B, self.num_heads, 1, 1),
        self.empty_token_score,
        device=attn_logits.device,
    )
    attn_logits = torch.cat([attn_logits, empty_score], dim=-1)
    
    empty_v = torch.zeros(
        (B, self.num_heads, 1, self.head_dim),
        device=v.device,
    )
    v = torch.cat([v, empty_v], dim=2)
```

**How to explain:**
> "The empty token lets attention 'escape' to nothing. If no neighbor is relevant, the receiver can attend to the empty token (which contributes zero to the output). High empty_attention means 'this cell's neighbors don't matter much.' This is interpretable: empty_attention → niche independence."

---

## 7. Sinusoidal Time Embedding

```
PE(t)₂ᵢ = sin(t · ω_i)
PE(t)₂ᵢ₊₁ = cos(t · ω_i)

where:
  ω_i = exp(-i · log(10000) / (d/2 - 1))
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| t | [B] | Time values in [0, 1] | `t` parameter |
| d | scalar | Embedding dimension | `self.dim` |
| ω_i | [d/2] | Frequency bands | `freq = torch.exp(torch.arange(half, ...) * (-math.log(10_000.0) / max(half - 1, 1)))` |
| PE(t) | [B × d] | Time embedding | Return value |

**Code:** `stagebridge/context/layers.py:280-309`

```python
def forward(self, t: Tensor) -> Tensor:
    half = self.dim // 2
    freq = torch.exp(
        torch.arange(half, device=device, dtype=dtype)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    phase = t[:, None] * freq[None, :]  # [B, d/2]
    emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    return emb
```

**Derivation:** From "Attention Is All You Need" (Vaswani et al., 2017). The frequencies span from 1 to 1/10000, giving the network access to both fine and coarse time information. Sin/cos pairs allow the network to learn linear functions of time via dot products.

**How to explain:**
> "We encode continuous time t∈[0,1] into a high-dimensional vector. Different frequencies capture different time scales. The network can learn to weight fast vs. slow time variations."

---

## 8. Optimal Transport Coupling

```
π* = argmin_{π ∈ Π(μ,ν)} ∫ c(x,y) dπ(x,y)
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| π | [n × m] | Coupling (joint distribution) | `coupling` variable |
| Π(μ,ν) | - | Set of couplings with marginals μ and ν | Constraint set |
| μ | [n] | Source distribution (cells at stage s) | Uniform: `1/n` |
| ν | [m] | Target distribution (cells at stage s') | Uniform: `1/m` |
| c(x,y) | scalar | Cost function: ‖x - y‖² | `torch.cdist(x_src_64, x_tgt_64, p=2).pow(2)` |
| π* | [n × m] | Optimal coupling (minimum cost) | Output of Sinkhorn |

**Discrete version (what we compute):**

```
π* = argmin_{π ∈ ℝ^{n×m}} Σᵢⱼ πᵢⱼ · cᵢⱼ

subject to:
  Σⱼ πᵢⱼ = aᵢ = 1/n    (row sums = source weights)
  Σᵢ πᵢⱼ = bⱼ = 1/m    (column sums = target weights)
  πᵢⱼ ≥ 0              (non-negative)
```

**How to explain:**
> "π is a transport plan - πᵢⱼ tells us how much mass to move from source cell i to target cell j. The constraints say we must empty each source and fill each target exactly. The optimization finds the plan with minimum total cost. For uniform distributions, a=1/n and b=1/m."

---

## 9. Entropic Regularization (Sinkhorn)

```
π* = argmin_{π ∈ Π(μ,ν)} Σᵢⱼ πᵢⱼ · cᵢⱼ + ε · H(π)

where:
  H(π) = -Σᵢⱼ πᵢⱼ · log(πᵢⱼ)  (entropy)
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| ε | scalar | Entropic regularization (default 0.05) | `self.config.ot_epsilon` |
| H(π) | scalar | Entropy of coupling | Regularization term |
| log_K | [n × m] | Log Gibbs kernel: -C/ε | `log_K = -cost / self.config.ot_epsilon` |
| log_u | [n] | Log row scaling variable | `log_u = torch.zeros(n, ...)` |
| log_v | [m] | Log column scaling variable | `log_v = torch.zeros(m, ...)` |
| sinkhorn_iters | scalar | Number of iterations (default 80) | `self.config.sinkhorn_iters` |

**Sinkhorn algorithm (log-space for numerical stability):**

```
Initialize: log_u = 0ₙ, log_v = 0ₘ
           log_K = -C / ε
           log_a = -log(n), log_b = -log(m)  (uniform marginals)

Repeat for k iterations:
  log_u ← log_a - logsumexp(log_K + log_v, axis=1)
  log_v ← log_b - logsumexp(log_Kᵀ + log_u, axis=1)

Output: log_π = log_u + log_K + log_vᵀ
        π = exp(log_π)
```

**Code:** `stagebridge/training/trainer.py:1036-1065`

```python
def _sinkhorn_coupling(self, x_src: Tensor, x_tgt: Tensor) -> Tensor:
    # Compute cost matrix in float64 for stability
    cost = torch.cdist(x_src_64, x_tgt_64, p=2).pow(2)
    
    # Uniform marginals in log space
    log_a = torch.full((n,), -torch.log(torch.tensor(n, ...)), ...)
    log_b = torch.full((m,), -torch.log(torch.tensor(m, ...)), ...)
    
    log_K = -cost / self.config.ot_epsilon
    
    log_u = torch.zeros(n, ...)
    log_v = torch.zeros(m, ...)
    
    for _ in range(self.config.sinkhorn_iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)
    
    log_pi = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    pi = torch.exp(log_pi).to(dtype)
    return pi
```

**Why entropy regularization?**
- Makes the problem **strictly convex** (unique solution)
- Enables fast **Sinkhorn iterations** (matrix scaling)
- Larger ε → smoother coupling (more spread out)
- ε → 0 → exact OT (but numerically unstable)

**How to explain:**
> "Sinkhorn alternates between normalizing rows and columns of the kernel matrix. It's like iteratively adjusting row and column scalings until both marginal constraints are satisfied. We run it in log-space to avoid numerical overflow. 80 iterations is typically enough for convergence."

---

## 10. Flow Matching Loss (OT-CFM)

```
L = 𝔼_{(x₀,x₁)∼π*, t∼U[0,1]} [ ‖v_θ(x_t, t, c) - u_t‖² ]

where:
  x_t = (1-t)·x₀ + t·x₁           (linear interpolation)
  u_t = x₁ - x₀                    (true velocity)
  
Optional noise (stochastic interpolant):
  x_t = (1-t)·x₀ + t·x₁ + σ·√(t(1-t))·η    where η ~ N(0, I)
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| (x₀, x₁) | [B × D] | Paired cells from OT coupling | `x_i = x0[src_idx]`, `y_j = x1[tgt_idx]` |
| π* | [n × m] | OT coupling matrix | Output of `_sinkhorn_coupling` |
| t | [B] | Time, sampled uniformly from [0,1] | `t = torch.rand(num_ot_pairs, ...)` |
| x_t | [B × D] | Interpolated point at time t | `x_t = (1 - t.unsqueeze(1)) * x_i + t.unsqueeze(1) * y_j` |
| v_θ | function | Neural network predicting velocity | `self.model.forward_vector_field(...)` |
| c | [B × C] | Niche context embedding | `ctx = context[src_idx]` |
| u_t | [B × D] | True velocity (constant along straight path) | `u_t = y_j - x_i` |
| σ | scalar | Noise scale (default 0.0) | `self.config.sigma` |
| num_ot_pairs | scalar | Number of pairs to sample (default 256) | `self.config.num_ot_pairs` |

**Code:** `stagebridge/training/trainer.py:958-1034`

```python
def _flow_matching_loss(self, x0, x1, context, ...) -> tuple[Tensor, Tensor]:
    # Get OT coupling
    coupling = self._sinkhorn_coupling(x0, x1)
    src_idx, tgt_idx = self._sample_from_coupling(coupling, self.config.num_ot_pairs)
    
    # Sample pairs
    x_i = x0[src_idx]
    y_j = x1[tgt_idx]
    ctx = context[src_idx]
    
    # Sample time
    t = torch.rand(self.config.num_ot_pairs, device=self.device)
    
    # Linear interpolation
    x_t = (1 - t.unsqueeze(1)) * x_i + t.unsqueeze(1) * y_j
    
    # Optional noise (stochastic interpolant)
    if self.config.sigma > 0:
        noise_scale = self.config.sigma * (t * (1 - t)).sqrt().unsqueeze(1)
        x_t = x_t + noise_scale * torch.randn_like(x_t)
    
    # True velocity
    u_t = y_j - x_i
    
    # Predicted velocity
    v_t = self.model.forward_vector_field(x_t=x_t, t=t, context=ctx, ...)
    
    # MSE loss
    loss = F.mse_loss(v_t, u_t)
    return loss, coupling
```

**Derivation:** From Lipman et al. (2023). The conditional velocity field u_t(x | x₀,x₁) = x₁ - x₀ is constant along the straight path from x₀ to x₁. By regressing v_θ against this over all OT pairs, we learn the marginal velocity field that transports μ to ν.

**Why OT coupling matters:**
> "Without OT, we'd randomly pair source and target cells. This creates crossing paths that cancel out. OT gives non-crossing paths - each source goes to its 'nearest' target. The learned velocity field can then be nearly constant along each trajectory, making the problem much easier."

---

## 11. Inference (ODE Integration)

```
dx/dt = v_θ(x, t, c)

x₁ = x₀ + ∫₀¹ v_θ(x_t, t, c) dt
```

**Euler discretization:**

```
x_{k+1} = x_k + Δt · v_θ(x_k, t_k, c)

where:
  Δt = 1/N           (step size)
  t_k = k · Δt       (time at step k)
  N = 8-16 steps     (typically sufficient due to OT coupling)
```

| Symbol | Dimension | Definition |
|--------|-----------|------------|
| Δt | scalar | Integration step size |
| N | scalar | Number of integration steps (8-16 typical) |
| x_k | [B × D] | State at step k |
| t_k | scalar | Time at step k |

**Euler-Maruyama (with stochasticity for uncertainty):**

```
x_{k+1} = x_k + Δt · v_θ(x_k, t_k, c) + σ·√Δt · η_k

where:
  η_k ~ N(0, I)       (Gaussian noise)
  σ = noise scale     (for uncertainty quantification)
```

**How to explain:**
> "At inference, we integrate the learned velocity field. Starting from a cell at stage s, we repeatedly step forward: position updates by velocity times step size. 8-16 steps is enough because the trajectories are nearly straight thanks to OT coupling. Adding Gaussian noise gives us stochastic trajectories for uncertainty quantification."

---

## 12. CrossAttentionDrift (Gated Context)

The drift network uses cross-attention over niche context tokens, with a learned gate to balance context-informed vs. latent-only predictions.

```
v_θ(x_t, t, c) = g · v_context + (1-g) · v_latent

where:
  q = W_q · [x_t ; t_emb]
  v_context = CrossAttn(q, context_tokens ∪ stage_token)
  v_latent = MLP([x_t ; t_emb ; stage_emb])
  g = σ(MLP([q ; h ; stage_emb])) ∈ [0, 1]
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| x_t | [B × D] | Current state at time t | `x_t` parameter |
| t_emb | [B × T] | Sinusoidal time embedding | `time_emb` parameter |
| context_tokens | [B × N × C] | Niche context tokens | `context_tokens` parameter |
| stage_emb | [B × S] | Stage pair embedding | `stage_emb` parameter |
| q | [B × 1 × C] | Query (state + time projected) | `self.query_proj(torch.cat([x_t, time_emb], dim=-1)).unsqueeze(1)` |
| h | [B × 1 × C] | Cross-attention output | After `self.mha(query=q, key=kv, value=kv)` |
| v_context | [B × D] | Context-informed velocity | `self.context_out_proj(h.squeeze(1))` |
| v_latent | [B × D] | Latent-only baseline velocity | `self.latent_only(torch.cat([x_t, time_emb, stage_emb], ...))` |
| g | [B × 1] | Context gate (0=latent only, 1=full context) | `self.context_gate(...)` with sigmoid |

**Code:** `stagebridge/transition/drift.py:15-106`

```python
def forward(self, x_t, time_emb, context_tokens, stage_emb) -> Tensor:
    # Query from state + time
    q = self.query_proj(torch.cat([x_t, time_emb], dim=-1)).unsqueeze(1)
    
    # Keys/values from context tokens + stage token
    kv_ctx = self.kv_proj(context_tokens)
    stage_tok = self.stage_proj(stage_emb).unsqueeze(1)
    kv = torch.cat([kv_ctx, stage_tok], dim=1)
    
    # Cross-attention
    attn_out, attn_weights = self.mha(query=q, key=kv, value=kv, ...)
    h = self.ln1(q + attn_out)
    h = self.ln2(h + self.ff(h))
    context_only = self.context_out_proj(h.squeeze(1))
    
    # Latent-only baseline
    latent_only = self.latent_only(torch.cat([x_t, time_emb, stage_emb], dim=-1))
    
    # Learned gate
    gate = self.context_gate(torch.cat([q.squeeze(1), h.squeeze(1), stage_emb], dim=-1))
    
    return gate * context_only + (1.0 - gate) * latent_only
```

**Why gating?**
> "The gate lets the model learn WHEN context matters. For some transitions, niche context is critical (g→1). For others, the transition is cell-autonomous (g→0). The mean gate value is a diagnostic: high g means niche-conditioned transitions dominate."

---

## 13. FiLM Conditioning

Feature-wise Linear Modulation for affine conditioning.

```
FiLM(x, c) = γ · x + β

where:
  [γ_raw, β] = W · c
  γ = 1 + 0.5 · tanh(γ_raw)    (centered around 1, range [0.5, 1.5])
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| x | [B × D] | Input features | `x` parameter |
| c | [B × C] | Conditioning input | `condition` parameter |
| γ | [B × D] | Learned scale | `gamma = 1.0 + 0.5 * torch.tanh(gamma)` |
| β | [B × D] | Learned shift | From `self.to_gamma_beta(condition)` |

**Code:** `stagebridge/transition/drift.py:109-124`

```python
class FiLMConditioner(nn.Module):
    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        gamma_beta = self.to_gamma_beta(condition)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = 1.0 + 0.5 * torch.tanh(gamma)  # Centered around 1
        return gamma * x + beta
```

**How to explain:**
> "FiLM applies an affine transformation where γ and β are predicted from the conditioning input. The tanh keeps γ centered around 1 (mostly scaling by ~1, with modulation ±0.5). This is a lightweight way to inject conditioning into intermediate representations."

---

## 14. Learned Gromov-Wasserstein Fusion

**Aligns HLCA and LuCA embeddings via learned metric spaces.**

```
GW objective:
  min_P Σ_ijkl |C_X[i,k] - C_Y[j,l]|² · P[i,j] · P[k,l]

where:
  C_X = pairwise distances in learned HLCA metric space
  C_Y = pairwise distances in learned LuCA metric space
  P = coupling matrix
```

| Symbol | Dimension | Definition | Code Reference |
|--------|-----------|------------|----------------|
| x_hlca | [B × 30] | HLCA embeddings | Input to fusion |
| x_luca | [B × 10] | LuCA embeddings | Input to fusion |
| metric_dim | scalar | Shared metric space dimension (default 32) | `config.metric_dim` |
| output_dim | scalar | Fused output dimension (default 40) | `config.output_dim` |
| C_X | [B × N × N] | Distance matrix in HLCA metric space | `torch.cdist(z_hlca, z_hlca)` |
| C_Y | [B × M × M] | Distance matrix in LuCA metric space | `torch.cdist(z_luca, z_luca)` |
| P | [B × N × M] | GW coupling | Output of `gromov_wasserstein_differentiable` |
| sinkhorn_reg | scalar | Entropic regularization (default 0.1) | `config.sinkhorn_reg` |
| gw_iters | scalar | Outer GW iterations (default 10) | `config.gw_iters` |
| sinkhorn_iters | scalar | Inner Sinkhorn iterations (default 20) | `config.sinkhorn_iters` |

**GW iterative algorithm:**

```
Initialize: P = a ⊗ b  (outer product of uniform marginals)

For each GW iteration:
  # Compute linear cost matrix for current P
  cost[i,j] = Σ_kl P[k,l] · (C_X[i,k] - C_Y[j,l])²
            = Σ_k C_X²[i,k] · Σ_l P[k,l]     # term 1
            + Σ_l C_Y²[j,l] · Σ_k P[k,l]     # term 2
            - 2 · Σ_kl C_X[i,k] · P[k,l] · C_Y[j,l]  # term 3
  
  # Sinkhorn step
  P = Sinkhorn(cost, reg=ε)

Output: P, gw_cost = Σ_ij P[i,j] · cost[i,j]
```

**Code:** `stagebridge/reference/learned_gw_fusion.py:97-157`

```python
def gromov_wasserstein_differentiable(C_X, C_Y, reg, num_gw_iters, num_sinkhorn_iters):
    # Initialize coupling as outer product of uniforms
    a = torch.ones(B, N, device=device) / N
    b = torch.ones(B, M, device=device) / M
    P = a.unsqueeze(2) * b.unsqueeze(1)
    
    C_X_sq = C_X ** 2
    C_Y_sq = C_Y ** 2
    
    for _ in range(num_gw_iters):
        # Term 1: [B, N, 1]
        term1 = torch.bmm(C_X_sq, P.sum(dim=2, keepdim=True))
        # Term 2: [B, 1, M]
        term2 = torch.bmm(P.sum(dim=1, keepdim=True), C_Y_sq)
        # Term 3: [B, N, M]
        term3 = torch.bmm(torch.bmm(C_X, P), C_Y)
        
        cost_matrix = term1 + term2.transpose(1, 2) - 2 * term3
        
        # Sinkhorn step
        P = sinkhorn_log_stabilized(cost_matrix, reg, num_sinkhorn_iters)
    
    gw_cost = (P * cost_matrix).sum(dim=(1, 2))
    return P, gw_cost
```

**Why learned GW?**
> "HLCA and LuCA have different dimensionalities (30d vs 10d) and capture different aspects of cell state (healthy vs. disease). GW finds structure-preserving alignment: cells that are neighbors in HLCA-space should map to neighbors in LuCA-space. Learning the metric projections lets the model discover which dimensions matter for alignment."

---

## 15. Wasserstein Distance (Evaluation)

```
W₂(μ, ν) = ( inf_{π ∈ Π(μ,ν)} ∫ ‖x - y‖² dπ(x,y) )^{1/2}
```

| Symbol | Definition |
|--------|------------|
| W₂ | Wasserstein-2 distance |
| μ | Predicted distribution |
| ν | True target distribution |
| ‖x - y‖² | Squared Euclidean cost |

**How to explain:**
> "The Wasserstein-2 distance is the minimum cost to transport one distribution to another, where cost is squared Euclidean distance. It's the 'earth mover's distance' - how much work to move the dirt from one pile to another. Lower is better."

---

## 16. Full Pipeline Summary

```
1. Embed cells:      z_hlca = scArches_HLCA(x) ∈ ℝ³⁰
                     z_luca = scArches_LuCA(x) ∈ ℝ¹⁰
                     z = GW_fusion(z_hlca, z_luca) ∈ ℝ⁴⁰  (or concat)

2. AMICI attention:  For each receiver r with neighbors N_r:
                     c_r = Σ_j∈N_r α_j · v_j
                     where α_j = softmax(q_r · k_j / √d - b₁ · dist_j)

3. Compute OT:       π* = Sinkhorn(C, ε)
                     where C_ij = ‖z_i - z_j‖²

4. Train velocity:   L = 𝔼_{(x₀,x₁)∼π*, t∼U[0,1]} [ ‖v_θ(x_t, t, c) - (x₁-x₀)‖² ]

5. Infer trajectory: x₁ = x₀ + ∫₀¹ v_θ(x_t, t, c) dt
```

---

## 17. Complexity Reference

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Self-attention (SAB) | O(n²) | Every token attends to every other |
| ISAB with m inducing | O(n·m) | Route through m bottleneck |
| AMICI attention | O(K) | K = number of neighbors (fixed) |
| Sinkhorn (k iterations) | O(k·n·m) | k ≈ 80 iterations |
| GW (outer × inner) | O(g·s·n·m) | g=10 GW, s=20 Sinkhorn |
| Flow matching step | O(d) | Just MLP forward pass |
| Euler integration | O(N·d) | N ≈ 8-16 steps |

---

## 18. Q&A Cheat Sheet

**"Why not just use RNA velocity?"**
> "RNA velocity estimates instantaneous direction from splicing ratios - it's a snapshot derivative. It doesn't give you the full trajectory or allow conditioning on niche. And it's noisy in spatial data where capture efficiency varies."

**"How do you handle the identifiability problem?"**
> "Weinreb et al. showed that without constraints, infinitely many velocity fields are consistent with two marginals. OT provides a unique coupling (minimum cost). The niche conditioning provides additional biological constraints. We're not claiming to recover true lineage - we're learning the most parsimonious niche-conditioned transport."

**"Why Sinkhorn over auction algorithms or network simplex?"**
> "Sinkhorn is GPU-friendly - it's just matrix operations. Network simplex is exact but sequential. For mini-batch OT in training, Sinkhorn's speed matters more than exactness, and entropic regularization actually helps generalization."

**"What's the relationship to Schrödinger bridges?"**
> "Schrödinger bridges add a diffusion term - they find the most likely stochastic process connecting two distributions. OT-CFM is the deterministic limit. We can add Brownian noise at inference (Euler-Maruyama) for uncertainty quantification, but training is deterministic."

**"Why Set Transformer over DeepSets?"**
> "DeepSets is φ(Σψ(xᵢ)) - element-wise transform, sum, output. No element interaction before pooling. Set Transformer lets tokens talk to each other first via attention. For niches, this means different neighbors can interact before we pool. Richer representations."

**"Why linear interpolation in flow matching?"**
> "OT coupling gives non-crossing paths, so linear interpolation is nearly optimal. Curved paths would require more integration steps and complicate training. The simplicity is a feature - simulation-free training."

**"What happens if ε is too large in Sinkhorn?"**
> "The coupling becomes more uniform - entropy regularization spreads mass out. You lose the sharp OT structure. Too small and you get numerical instability. ε=0.05 is a good default - sharp enough to be meaningful, stable enough to converge."

**"Why softplus for the distance coefficient?"**
> "softplus(x) = log(1 + exp(x)) is always positive. This architecturally guarantees monotonic attention decay with distance - the coefficient b₁ can never go negative. It's not a hyperparameter choice, it's a hard constraint that matches the biology of paracrine signaling."

**"What does empty_attention mean?"**
> "If a receiver has high attention to the empty token, it means 'none of my neighbors matter much for predicting my state.' This is interpretable: high empty_attention → cell is niche-independent. Low empty_attention → cell is strongly influenced by its microenvironment."

**"How does the context gate work?"**
> "The gate g ∈ [0,1] interpolates between v_context (niche-informed) and v_latent (cell-autonomous). The network learns when niche matters. You can inspect mean gate values: g→1 means niche dominates, g→0 means cell-intrinsic dominates."
