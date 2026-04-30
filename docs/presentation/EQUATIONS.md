# StageBridge Key Equations

A breakdown of every equation you need to understand and explain.

---

## 1. Multihead Attention (the foundation)

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

| Symbol | What it is | Intuition |
|--------|-----------|-----------|
| Q | Query matrix [n × d] | "What am I looking for?" |
| K | Key matrix [m × d] | "What do I have to offer?" |
| V | Value matrix [m × d] | "What information do I carry?" |
| d_k | Key dimension | Scaling factor to prevent dot products from exploding |
| QKᵀ | Attention scores [n × m] | How much each query matches each key |
| softmax | Row-wise normalization | Convert scores to weights that sum to 1 |
| · V | Weighted sum | Aggregate values by attention weights |

**How to say it:**
> "The query asks 'what's relevant to me?', the keys answer 'here's what I have', and the output is a weighted combination of values where the weights come from query-key similarity."

---

## 2. Self-Attention Block (SAB)

```
SAB(X) = LN(X + MHA(X, X, X))
       = LN(X + softmax(XW_Q · (XW_K)ᵀ / √d) · XW_V)
```

| Symbol | What it is |
|--------|-----------|
| X | Input tokens [n × d] |
| W_Q, W_K, W_V | Learned projection matrices |
| MHA | Multihead attention (Q=K=V=X) |
| LN | Layer normalization |
| X + ... | Residual connection |

**How to say it:**
> "In self-attention, every token queries every other token. Each token gets updated by attending to the full set. The residual connection means we're adding new information, not replacing."

**Complexity:** O(n²) - every token attends to every other token.

---

## 3. Induced Set Attention Block (ISAB)

```
ISAB_m(X) = MAB(X, MAB(I, X))

where:
  MAB(X, Y) = LN(X + MHA(X, Y, Y))
  I ∈ ℝ^{m × d} = learnable inducing points
```

**Step by step:**

```
Step 1: H = MAB(I, X)     # Inducing points query the input
        H = LN(I + softmax(IW_Q · (XW_K)ᵀ / √d) · XW_V)
        # H is [m × d] - compressed representation

Step 2: ISAB(X) = MAB(X, H)   # Input queries the compressed representation  
        = LN(X + softmax(XW_Q' · (HW_K')ᵀ / √d) · HW_V')
        # Output is [n × d] - same size as input
```

| Symbol | What it is |
|--------|-----------|
| I | m learnable inducing points [m × d] |
| m | Number of inducing points (typically 16-64) |
| H | Compressed set representation [m × d] |

**How to say it:**
> "Instead of n² attention, we route through m inducing points. First, the inducing points attend to all inputs - this compresses the set into m summary vectors. Then, each input attends to these summaries. Total cost: O(nm) instead of O(n²). The inducing points learn to capture the sufficient statistics of the set."

---

## 4. Pooling by Multihead Attention (PMA)

```
PMA_k(X) = MAB(S, X)
         = LN(S + softmax(SW_Q · (XW_K)ᵀ / √d) · XW_V)

where:
  S ∈ ℝ^{k × d} = learnable seed vectors
```

| Symbol | What it is |
|--------|-----------|
| S | k learnable seed vectors [k × d] |
| k | Number of output vectors (typically 1 for pooling) |
| Output | [k × d] - fixed size regardless of input size |

**How to say it:**
> "PMA extracts a fixed-size output from a variable-size set. The seed vector learns to query for the most relevant summary. It's like asking 'give me THE important thing about this set' and letting attention figure out what that is. Unlike mean pooling, this is adaptive."

---

## 5. Optimal Transport Coupling

```
π* = argmin_{π ∈ Π(μ,ν)} ∫ c(x,y) dπ(x,y)
```

| Symbol | What it is |
|--------|-----------|
| π | Coupling (joint distribution) |
| Π(μ,ν) | Set of all couplings with marginals μ and ν |
| μ | Source distribution (cells at stage s) |
| ν | Target distribution (cells at stage s') |
| c(x,y) | Cost function, typically ‖x - y‖² |
| π* | Optimal coupling (minimum cost) |

**Discrete version (what we actually compute):**

```
π* = argmin_{π ∈ ℝ^{n×m}} Σᵢⱼ πᵢⱼ · cᵢⱼ

subject to:
  Σⱼ πᵢⱼ = aᵢ    (row sums = source weights)
  Σᵢ πᵢⱼ = bⱼ    (column sums = target weights)
  πᵢⱼ ≥ 0        (non-negative)
```

**How to say it:**
> "π is a transport plan - πᵢⱼ tells us how much mass to move from source cell i to target cell j. The constraints say we must empty each source and fill each target exactly. The optimization finds the plan with minimum total cost. For uniform distributions, a=1/n and b=1/m."

---

## 6. Entropic Regularization (Sinkhorn)

```
π* = argmin_{π ∈ Π(μ,ν)} Σᵢⱼ πᵢⱼ · cᵢⱼ + ε · H(π)

where H(π) = -Σᵢⱼ πᵢⱼ · log(πᵢⱼ)  (entropy)
```

| Symbol | What it is |
|--------|-----------|
| ε | Regularization strength (typically 0.05) |
| H(π) | Entropy of coupling |

**Why entropy?**
- Makes the problem **strictly convex** (unique solution)
- Enables fast **Sinkhorn iterations**
- Larger ε → smoother, more spread-out coupling
- ε → 0 → exact OT

**Sinkhorn algorithm:**

```
Initialize: u = 1ₙ, v = 1ₘ
K = exp(-C / ε)        # Gibbs kernel

Repeat until convergence:
  u = a / (K · v)       # Row scaling
  v = b / (Kᵀ · u)      # Column scaling

Output: π* = diag(u) · K · diag(v)
```

**Log-space version (numerically stable):**

```
log u ← log a - logsumexp(log K + log v, axis=1)
log v ← log b - logsumexp(log Kᵀ + log u, axis=1)
```

**How to say it:**
> "Sinkhorn alternates between normalizing rows and columns of the kernel matrix. It's like iteratively adjusting row and column scalings until both marginal constraints are satisfied. We run it in log-space to avoid numerical overflow. 80 iterations is typically enough for convergence."

---

## 7. Flow Matching Loss (OT-CFM)

```
L = 𝔼_{(x₀,x₁)∼π*, t∼U[0,1]} [ ‖v_θ(x_t, t, c) - (x₁ - x₀)‖² ]

where:
  x_t = (1-t)·x₀ + t·x₁     (linear interpolation)
```

| Symbol | What it is |
|--------|-----------|
| (x₀, x₁) | Paired cells from OT coupling π* |
| t | Time, sampled uniformly from [0,1] |
| x_t | Interpolated point at time t |
| v_θ | Neural network predicting velocity |
| c | Niche context embedding |
| x₁ - x₀ | True velocity (target) |

**How to say it:**
> "We sample a paired source-target from the OT coupling. We pick a random time t and compute the interpolated point x_t. The network sees x_t, the time t, and the niche context c. It predicts the velocity. The true velocity is just x₁ - x₀ because we're interpolating linearly. The loss is MSE between predicted and true velocity."

**Why this works:**
> "The conditional velocity field u_t(x | x₀,x₁) = x₁ - x₀ is constant along the straight path from x₀ to x₁. By regressing against this over all OT pairs, we learn the marginal velocity field that transports the full distribution."

---

## 8. Inference (ODE Integration)

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
  N = 8-16 steps     (typically sufficient)
```

**Euler-Maruyama (with stochasticity):**

```
x_{k+1} = x_k + Δt · v_θ(x_k, t_k, c) + σ·√Δt · ηₖ

where:
  ηₖ ~ N(0, I)       (Gaussian noise)
  σ = noise scale    (for uncertainty)
```

**How to say it:**
> "At inference, we integrate the learned velocity field. Starting from a cell at stage s, we repeatedly step forward: position updates by velocity times step size. 8-16 steps is enough because the trajectories are nearly straight thanks to OT coupling. Adding Gaussian noise gives us stochastic trajectories for uncertainty quantification."

---

## 9. Distance Bias in Receiver-Centered Attention (AMICI-style)

```
Attention = softmax(QKᵀ/√d - b₁·dist)

where:
  b₁ = softplus(MLP(receiver))  # Enforced positive
```

| Symbol | What it is |
|--------|-----------|
| dist | Distance from receiver to each neighbor [1 × K] |
| b₁ | Learned distance coefficient (positive) |
| softplus | log(1 + exp(x)), ensures positivity |

**How to say it:**
> "We subtract a distance penalty from attention scores. The coefficient b₁ is forced positive via softplus, so attention **monotonically decreases** with distance. This isn't learned freely - it's architecturally constrained. Closer neighbors always get more attention than farther ones, all else equal. This matches the biology: paracrine signaling decays with distance."

---

## 10. RBF Distance Encoding

```
φ_RBF(d) = [ exp(-(d - μₖ)² / 2σ²) ]_{k=1}^{K}

where:
  μₖ = (k-1) · d_max / (K-1)    # Uniformly spaced centers
  σ = d_max / K                  # Bandwidth
  d_max = 100 μm                 # Maximum distance
  K = 16                         # Number of RBF centers
```

| Symbol | What it is |
|--------|-----------|
| d | Distance in micrometers |
| μₖ | Center of k-th RBF |
| σ | Width of each RBF |
| φ_RBF(d) | 16-dimensional encoding |

**How to say it:**
> "We encode continuous distance into a 16-dimensional vector using radial basis functions. Each RBF is a Gaussian centered at a different distance. A distance of 50μm activates the RBFs centered around 50μm most strongly. This gives the network a rich, smooth representation of distance rather than just a scalar."

---

## 11. Wasserstein Distance (for evaluation)

```
W₂(μ, ν) = ( inf_{π ∈ Π(μ,ν)} ∫ ‖x - y‖² dπ(x,y) )^{1/2}
```

**How to say it:**
> "The Wasserstein-2 distance is the minimum cost to transport one distribution to another, where cost is squared Euclidean distance. It's the 'earth mover's distance' - how much work to move the dirt from one pile to another. Lower is better: our model achieves W=0.680, no-niche baseline gets W=0.812."

---

## Summary: The Full Pipeline in Equations

```
1. Embed cells:        z = [HLCA(x) ‖ LuCA(x)] ∈ ℝ⁴⁰

2. Tokenize niche:     T = [receiver, ring₁, ring₂, ring₃, ring₄, hlca, luca, pathway, stats]

3. Encode niche:       c = PMA(SAB(ISAB(ISAB(T))))

4. Compute OT:         π* = Sinkhorn(C, ε)  where Cᵢⱼ = ‖zᵢ - zⱼ‖²

5. Train velocity:     L = 𝔼[ ‖v_θ(x_t, t, c) - (x₁-x₀)‖² ]

6. Infer trajectory:   x₁ = x₀ + ∫₀¹ v_θ(x_t, t, c) dt
```

---

## Quick Reference: Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Self-attention (SAB) | O(n²) | Every token attends to every other |
| ISAB with m inducing | O(nm) | Route through m bottleneck |
| Sinkhorn (k iterations) | O(k·n·m) | k ≈ 80 iterations |
| Flow matching step | O(d) | Just MLP forward pass |
| Euler integration | O(N·d) | N ≈ 8-16 steps |

---

## Q&A Cheat Sheet

**"Why not just use RNA velocity?"**
> "RNA velocity estimates instantaneous direction from splicing ratios - it's a snapshot derivative. It doesn't give you the full trajectory or allow conditioning on niche. And it's noisy in spatial data where capture efficiency varies."

**"How do you handle the identifiability problem?"**
> "Weinreb et al. showed that without constraints, infinitely many velocity fields are consistent with two marginals. OT provides a unique coupling (minimum cost). The niche conditioning provides additional biological constraints. We're not claiming to recover true lineage - we're learning the most parsimonious niche-conditioned transport."

**"Why Sinkhorn over auction algorithms or network simplex?"**
> "Sinkhorn is GPU-friendly - it's just matrix operations. Network simplex is exact but sequential. For mini-batch OT in training, Sinkhorn's speed matters more than exactness, and entropic regularization actually helps generalization."

**"What's the relationship to Schrödinger bridges?"**
> "Schrödinger bridges add a diffusion term - they find the most likely stochastic process connecting two distributions. OT-CFM is the deterministic limit. We can add Brownian noise at inference (Euler-Maruyama) for uncertainty quantification, but training is deterministic."

**"Why Set Transformer over DeepSets?"**
> "DeepSets is φ(Σψ(xᵢ)) - element-wise transform, sum, output. No element interaction before pooling. Set Transformer lets tokens talk to each other first via attention. For niches, this means the HLCA token can learn from spatial tokens before we pool. Richer representations."

**"Why linear interpolation in flow matching?"**
> "OT coupling gives non-crossing paths, so linear interpolation is nearly optimal. Curved paths would require more integration steps and complicate training. The simplicity is a feature - simulation-free training."

**"What happens if ε is too large in Sinkhorn?"**
> "The coupling becomes more uniform - entropy regularization spreads mass out. You lose the sharp OT structure. Too small and you get numerical instability. ε=0.05 is a good default - sharp enough to be meaningful, stable enough to converge."
