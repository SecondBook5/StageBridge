# Reference Atlas Integration

Dual-reference geometry: cells embedded relative to HLCA (healthy) and LuCA (tumor) atlases.

## Key Modules

### `learned_gw_fusion.py` - LearnedGWFusion

Differentiable Gromov-Wasserstein alignment for HLCA-LuCA fusion.

**Why GW?** HLCA and LuCA have different latent spaces (30d vs 10d). GW finds structure-preserving alignment between heterogeneous spaces without requiring shared features.

```python
# Learn metric projections
h_metric = hlca_metric_head(hlca)  # [B, metric_dim]
l_metric = luca_metric_head(luca)  # [B, metric_dim]

# Compute distance matrices
C_h = pairwise_distance(h_metric)  # [B, B]
C_l = pairwise_distance(l_metric)  # [B, B]

# Solve GW for coupling
coupling = gromov_wasserstein(C_h, C_l, reg=0.1)

# Transport and fuse
aligned_luca = coupling @ luca
fused = fusion_head([hlca; alpha*aligned_luca + (1-alpha)*luca])
```

### `scarches_mapper.py` - scArches Reference Mapping

Maps query cells to reference atlases using scArches surgery.

### `gw_precompute.py` - PretrainedGWFusion

Uses precomputed GW coupling for inference (moscot-style).

## Fusion Options

| Type | Description | When to Use |
|------|-------------|-------------|
| `concat` | Simple [HLCA; LuCA] concatenation | Baseline |
| `learned_gw` | Differentiable GW fusion | Recommended |
| `precompute_gw` | Precomputed coupling | Large-scale inference |

## Architecture

```
HLCA (30d)     LuCA (10d)
    │              │
    ▼              ▼
┌───────┐     ┌───────┐
│Metric │     │Metric │
│ Head  │     │ Head  │
└───────┘     └───────┘
    │              │
    ▼              ▼
   C_h            C_l
    │              │
    └──────┬───────┘
           │
    ┌──────▼──────┐
    │  GW Solver  │
    │  (Sinkhorn) │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   Fusion    │
    │    Head     │
    └──────┬──────┘
           │
           ▼
      Fused (40d)
```
