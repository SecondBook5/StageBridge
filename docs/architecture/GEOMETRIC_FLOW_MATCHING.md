# Geometric Flow Matching: From Euclidean to Spherical

This document captures the geometric considerations for OT-CFM and the path to a novel contribution.

---

## The Landscape

| Method | Embedding | Geometry | Dynamics | Niche Conditioning |
|--------|-----------|----------|----------|-------------------|
| moscot | scVI | Euclidean | Linear interpolation | None |
| CellRank | Any | Euclidean | Markov chain (discrete) | None |
| GeoBridge | Learned INN | Isometric (flat) | Linear in flat = geodesic in original | None |
| scPhere | vMF prior | Hyperspherical | N/A (embedding only) | None |
| StageBridge v1 | HLCA/LuCA | Euclidean | Learned velocity field | Receiver-centered |
| **StageBridge v2** | scPhere | Hyperspherical | Learned velocity field | Receiver-centered |

**The gap:** No one has done niche-conditioned flow matching on geometrically-principled embeddings.

---

## The Geometric Problem

### Why Geometry Matters

Standard VAEs (scVI, scArches) use Gaussian priors, which:
- Encourage cells to cluster at the origin ("crowding")
- Assume Euclidean distances are meaningful
- May distort biological relationships

scPhere (Ding & Regev, 2021) showed:
- von Mises-Fisher (vMF) prior places cells on hypersphere surface
- No crowding - uniform distribution has no center
- Preserves hierarchical structure better

### The Interpolation Problem

If embeddings live on a hypersphere (L2-normalized), linear interpolation is geometrically wrong:

```
lerp: x_t = (1-t)x₀ + tx₁
```

This cuts *through* the sphere - interpolated points have norm < 1 and aren't valid cell states.

```
        lerp (chord - wrong)
       x₀ -------- x₁
         \        /
          \      /    
           \    /     slerp (arc - correct)
            \  /
             ○  (sphere center)
```

---

## lerp vs slerp

### Linear Interpolation (lerp)

```python
def lerp(x0, x1, t):
    return (1 - t) * x0 + t * x1

# Velocity (target for OT-CFM)
v = x1 - x0  # constant along path
```

- Fast, simple
- Correct in Euclidean space
- **Wrong on hypersphere** - cuts through interior

### Spherical Linear Interpolation (slerp)

```python
def slerp(x0, x1, t):
    """Interpolate along great circle on unit sphere."""
    # Angle between points
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    
    # Handle small angles (numerical stability)
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < 1e-6
    
    # Slerp formula
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    
    # Fall back to lerp for small angles
    w0 = torch.where(small, 1 - t, w0)
    w1 = torch.where(small, t, w1)
    
    return w0 * x0 + w1 * x1

# Velocity (tangent to sphere at x_t)
def slerp_velocity(x0, x1, t):
    """Derivative of slerp - tangent vector at x_t."""
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # d/dt of slerp coefficients
    dw0 = -theta * torch.cos((1 - t) * theta) / sin_theta
    dw1 = theta * torch.cos(t * theta) / sin_theta
    
    return dw0 * x0 + dw1 * x1
```

- Follows geodesic (great circle) on sphere
- Interpolated points stay on surface (norm = 1)
- Velocity is tangent to sphere at each point

---

## Spherical OT-CFM

### Training

```python
def spherical_flow_matching_loss(model, x0, x1, context, t):
    """
    Loss for spherical OT-CFM.
    
    Args:
        model: velocity network v_θ(x_t, t, c)
        x0, x1: OT-coupled source/target cells (L2-normalized)
        context: niche context embedding
        t: time in [0, 1]
    """
    # Interpolate on sphere
    x_t = slerp(x0, x1, t)
    
    # True velocity: tangent to geodesic at x_t
    target_v = slerp_velocity(x0, x1, t)
    
    # Predicted velocity
    pred_v = model(x_t, t, context)
    
    # Project to tangent space (ensure output is tangent to sphere)
    pred_v_tangent = pred_v - (pred_v * x_t).sum(dim=-1, keepdim=True) * x_t
    
    return ((pred_v_tangent - target_v) ** 2).mean()
```

### Key Differences from Euclidean OT-CFM

| Aspect | Euclidean | Spherical |
|--------|-----------|-----------|
| Interpolation | `(1-t)x₀ + tx₁` | `slerp(x₀, x₁, t)` |
| Target velocity | `x₁ - x₀` (constant) | `d/dt slerp` (varies with t) |
| Output constraint | None | Must be tangent to sphere |
| Integration | Euler | Exponential map or Euler + renormalize |

### Inference

**Option 1: Euler + Renormalize**
```python
def integrate_spherical_euler(model, x0, context, num_steps=16):
    """Integrate on sphere via Euler + projection."""
    x = x0
    dt = 1.0 / num_steps
    
    for k in range(num_steps):
        t = k * dt
        v = model(x, t, context)
        
        # Project velocity to tangent space
        v_tangent = v - (v * x).sum(dim=-1, keepdim=True) * x
        
        # Euler step
        x = x + dt * v_tangent
        
        # Project back to sphere
        x = x / x.norm(dim=-1, keepdim=True)
    
    return x
```

**Option 2: Exponential Map (more principled)**
```python
def exp_map_sphere(x, v, dt):
    """Exponential map on unit sphere: geodesic step from x in direction v."""
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    theta = v_norm * dt
    return torch.cos(theta) * x + torch.sin(theta) * (v / v_norm)

def integrate_spherical_exp(model, x0, context, num_steps=16):
    """Integrate on sphere via exponential map."""
    x = x0
    dt = 1.0 / num_steps
    
    for k in range(num_steps):
        t = k * dt
        v = model(x, t, context)
        
        # Project velocity to tangent space
        v_tangent = v - (v * x).sum(dim=-1, keepdim=True) * x
        
        # Geodesic step via exponential map
        x = exp_map_sphere(x, v_tangent, dt)
    
    return x
```

---

## Implementation Path

### Option A: scPhere Embeddings (Recommended)

1. **Re-embed cells with scPhere** instead of scVI/scArches
   - Cells automatically on hypersphere
   - No HLCA/LuCA mapping (or need to retrain atlases with scPhere)

2. **Modify OT-CFM training**
   - Replace lerp with slerp
   - Replace constant velocity target with slerp derivative
   - Add tangent space projection to output

3. **Modify inference**
   - Use exponential map or Euler + renormalize

**Pros:** Principled end-to-end. scPhere is published and validated.
**Cons:** Lose atlas reference geometry. Need to re-embed everything.

### Option B: Normalize Existing Embeddings

1. **L2-normalize HLCA/LuCA embeddings**
   ```python
   z_sphere = z / z.norm(dim=-1, keepdim=True)
   ```

2. **Same modifications as Option A**

**Pros:** Quick experiment. Keep atlas structure.
**Cons:** Normalization may distort learned distances. Not principled.

### Option C: GeoBridge-style Isometry Learning

1. **Learn INN with isometry constraint**
   ```
   L_iso = |D_embed(x,y) - D_latent(f(x), f(y))|² for neighbors x, y
   ```

2. **Linear interpolation in flat space = geodesic in original**

3. **Niche-conditioned OT-CFM in flat space**

**Pros:** Works with any embedding. Geometrically principled.
**Cons:** Complex. INN training adds another moving part.

---

## Comparison to Related Work

### vs moscot (Klein et al., 2025)

moscot does OT + linear interpolation. No learned dynamics, no niche.

**StageBridge advantage:**
- Learned velocity field (more expressive than interpolation)
- Niche conditioning (microenvironment-aware)
- Spherical geometry (if implemented)

**Potential baseline:** Compare StageBridge predictions vs moscot interpolation on held-out cells.

### vs CellRank (Weiler et al., 2025)

CellRank builds transition probability matrices via kernels (velocity, pseudotime, connectivity). Markov chain, not continuous dynamics.

**StageBridge advantage:**
- Continuous trajectories, not discrete transitions
- Spatial niche conditioning
- Generative (can simulate new trajectories)

**Complementary:** StageBridge velocities could feed into CellRank as a custom kernel.

### vs GeoBridge (Zhu et al., 2026)

GeoBridge learns isometric mapping so linear interpolation = geodesic. No conditioning.

**StageBridge advantage:**
- Niche conditioning
- Don't need to learn INN (if using scPhere)

**What GeoBridge has:** Principled geometry without requiring hyperspherical embeddings.

### vs scPhere (Ding & Regev, 2021)

scPhere is an embedding method, not a dynamics method.

**Integration:** Use scPhere embeddings as input to StageBridge.

---

## The Novel Contribution

### Current Gap in Literature

| | No niche conditioning | Niche-conditioned |
|---|---|---|
| **Euclidean geometry** | moscot, CellRank | StageBridge v1 |
| **Geodesic-aware** | GeoBridge | **EMPTY** |

### StageBridge v2 Fills This Gap

> "First niche-conditioned flow matching on geometrically-principled cell embeddings"

**The claim to test:**
Niche-conditioned dynamics on hyperspherical embeddings outperform:
1. Context-free geodesic methods (GeoBridge)
2. Context-aware Euclidean methods (StageBridge v1)
3. Context-free Euclidean methods (moscot)

---

## Experimental Validation

### Metrics

1. **Wasserstein distance** to held-out target distribution
2. **Trajectory smoothness** (jerk, curvature)
3. **Biological plausibility** - marker gene dynamics along trajectory
4. **Niche effect recovery** - does attention highlight known L-R pairs?

### Ablations

| Config | Embedding | Geometry | Niche |
|--------|-----------|----------|-------|
| moscot baseline | scVI | Euclidean (lerp) | None |
| StageBridge v1 | HLCA/LuCA | Euclidean (lerp) | Yes |
| StageBridge v1-sphere | HLCA/LuCA (normalized) | Spherical (slerp) | Yes |
| StageBridge v2 | scPhere | Spherical (slerp) | Yes |
| GeoBridge baseline | Learned INN | Isometric | None |

### Expected Results

If niche matters AND geometry matters:
```
StageBridge v2 > StageBridge v1-sphere > StageBridge v1 > moscot
StageBridge v2 > GeoBridge (because of niche conditioning)
```

---

## References

- **scPhere:** Ding & Regev (2021). Deep generative model embedding of single-cell RNA-Seq profiles on hyperspheres and hyperbolic spaces. *Nat Commun.*
- **GeoBridge:** Zhu et al. (2026). Generating and navigating single cell dynamics via a geodesic bridge. *bioRxiv.*
- **moscot:** Klein et al. (2025). Mapping cells through time and space with moscot. *Nature.*
- **CellRank:** Weiler et al. (2025). CellRank: consistent and data view agnostic fate mapping. *Nat Protoc.*
- **Riemannian Flow Matching:** Chen & Lipman (2023). Riemannian Flow Matching on General Geometries. *arXiv.*
