---
name: gw_fusion_design
description: Critical design decision - GW fusion is PREPROCESSING not model component. Learned GW chosen over concat.
type: project
---

# GW Fusion Design Decision (2026-05-09)

## The Panic and Resolution

Nearly had a heart attack thinking the architecture was incoherent. It's NOT - the design makes sense once you understand:

1. **AMICI encoder**: Encodes spatial niche (receiver queries, neighbors keys/values)
2. **Set Transformer**: Permutation-invariant refinement over tokens
3. **Separate HLCA/LuCA tokens**: NOT redundant with fused embedding - different semantic roles
   - Fused 40d: "where am I in combined space" (used for spatial neighbor similarity)
   - HLCA token: "what healthy atlas says about this cell type" (reference context)
   - LuCA token: "what cancer atlas says about this cell type" (reference context)

## The Actual Problem

The 40d fused embedding is currently `[HLCA; LuCA]` concatenation. This assumes the two atlas spaces are aligned, which they're NOT.

## Decision: Learned GW Fusion

**Why not concat:** Assumes alignment that doesn't exist

**Why not precomputed GW:** Can't adapt to downstream task

**Why not ICNN:** Overkill for atlas alignment (designed for perturbation prediction)

**Chosen: Learned GW**
- Learn metric projections for HLCA and LuCA
- Compute GW coupling in learned spaces
- Coupling supervision helps metrics learn useful alignment
- Whole thing is differentiable

## Key Files

- `stagebridge/reference/ARCHITECTURE.md` - Full architecture documentation
- `stagebridge/reference/learned_gw_fusion.py` - Implementation
- `stagebridge/reference/gw_precompute.py` - Barycentric fallback (moscot-style)

## Two OT Components (DON'T CONFUSE)

1. **GW Atlas Fusion**: HLCA <-> LuCA alignment (heterogeneous spaces, 30d vs 10d)
2. **OT-CFM Flow Matching**: Stage transitions (same 40d space, Sinkhorn coupling)

These are SEPARATE operations.

## How to apply

- GW fusion is PREPROCESSING or early model component
- Creates the 40d embedding that becomes x_0 for OT-CFM
- The model then learns velocity field v(x, t, niche_context)
