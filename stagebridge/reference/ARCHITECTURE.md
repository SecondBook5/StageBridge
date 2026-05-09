# StageBridge Architecture: GW Fusion and Data Flow

## Overview

This document clarifies the StageBridge architecture, particularly how GW fusion fits into the pipeline. Written after near-panic when the design seemed incoherent - it's not, but the fusion strategy needs fixing.

## The Core Scientific Claim

**"Spatial niche context gates cell state transitions in lung cancer progression."**

## Data Flow (Current)

```
CELL DATA (per cell in neighborhoods.parquet):
├── receiver_z: [40d] = HLCA(30d) || LuCA(10d)  <-- PROBLEM: just concat
├── neighbor_cells: [K, 40d] = neighbor fused embeddings
├── neighbor_distances: [K] = distances in microns
├── hlca_z: [30d] = raw HLCA embedding
├── luca_z: [10d] = raw LuCA embedding
├── pathway_z: pathway features
└── stats_z: conditioning features (CAF, immune, cell cycle)
```

## Model Architecture

### 1. AMICI Encoder (ReceiverCenteredNicheEncoder)

```
Input:  receiver[40d], neighbors[K,40d], distances[K]
Output: niche_context[hidden_dim], attention_weights

Key features:
- Receiver is QUERY, neighbors are KEYS/VALUES
- Distance-modulated attention: score = phenotype - b*distance
- b enforced positive via Softplus → monotonic decay with distance
- Empty token allows "no neighbor is informative"
- L1 penalty on values for sparsity
```

### 2. Token Assembly

```
tokens = [niche_context, hlca_token, luca_token, pathway_token, stats_token]
         [from AMICI]   [30d→hidden] [10d→hidden] [features]   [conditioning]
```

**Why separate HLCA/LuCA tokens when fused embedding exists?**

They serve DIFFERENT semantic roles:
- **Fused 40d**: Used for spatial neighbor similarity (what's around me)
- **HLCA token**: Reference context - what healthy atlas says about this cell type
- **LUCA token**: Reference context - what cancer atlas says about this cell type

The fused embedding captures "where am I in combined space" while the separate tokens provide "what do the atlases say about cells like me". NOT redundant.

### 3. Set Transformer (HierarchicalSetTransformer)

```
Input:  tokens[5, hidden_dim]
Output: context[hidden_dim], context_tokens[5, hidden_dim]

Architecture: ISAB → ISAB(spatial_rpe) → SAB → PMA
- Permutation invariant over tokens
- PMA pools to single context vector
```

### 4. Conditioning Layers

```
context = EvolutionBranch(context, wes_features)      # WES mutations
context = StatsConditioner(context, stats)            # FiLM modulation  
context = PrototypeBottleneck(context)                # Soft archetypes
```

### 5. Drift Head (CrossAttentionDrift)

```
Input:  x_t[40d], t, context_tokens, stage_emb
Output: velocity v_t[40d]

Architecture:
- Query: [x_t; time_emb]
- Keys/Values: context_tokens + stage_token
- Gated output: v = gate*context_drift + (1-gate)*latent_drift

The gate is INTERPRETABLE: how much does niche context matter for this transition?
```

### 6. OT-CFM Training

```
1. Sinkhorn coupling between source/target stage cells
2. Sample (i,j) pairs from coupling
3. Interpolate: x_t = (1-t)*x_i + t*x_j + noise
4. Target: u_t = x_j - x_i
5. Loss: ||v_t - u_t||²
```

## Where GW Fusion Fits

### The Problem

The 40d fused embedding is currently `[HLCA; LuCA]` concatenation. This assumes the two atlas spaces are aligned, which they're NOT. HLCA and LuCA were trained independently on different cell populations.

### The Solution

**GW fusion is PREPROCESSING, not a model component.**

```
OFFLINE (run once on representative population):
  coupling = solve_gw(population_hlca, population_luca)
  # This finds structure-preserving alignment between spaces
  
DATA PREP (per cell):
  fused = align_and_fuse(cell.hlca, cell.luca, coupling)
  # This creates the 40d embedding using the precomputed alignment
```

The model receives pre-aligned fused embeddings. GW computation should NOT happen inside the model forward pass.

### Why Not Per-Batch GW?

The original code tried to compute GW per batch. This is wrong because:
1. GW needs population structure to find meaningful alignment
2. Single cells [B, 1, D] have no internal structure - 1x1 distance matrices
3. Coupling would change every batch (unstable)

### Fusion Options

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| Concat | [HLCA; LuCA] | Simple, fast | Assumes aligned (wrong) |
| Barycentric | k-NN + coupling weights | Principled OT, proven at scale | No learning |
| Learned GW | Learn metric/projection respecting GW | Can adapt during training | More complex |
| ICNN | Convex neural net, gradient = OT map | Guaranteed valid OT | Designed for perturbation prediction |

**Current decision: Need learned GW that adapts during training.**

## Two OT Components (Don't Confuse)

1. **GW Atlas Fusion**: HLCA ↔ LuCA alignment for same cells
   - Heterogeneous spaces (30d vs 10d)
   - Structure-preserving (GW, not Wasserstein)
   - Creates the fused embedding x

2. **OT-CFM Flow Matching**: Stage transitions
   - Same space (40d fused)
   - Sinkhorn coupling between stage populations
   - Learns velocity field v(x,t)

These are SEPARATE. Don't conflate them.

## Files

- `stagebridge/reference/gw_precompute.py` - Offline GW computation + BarycentricFusion
- `stagebridge/reference/gw_fusion.py` - Model-side fusion (being redesigned)
- `stagebridge/context/encoder.py` - AMICI encoder
- `stagebridge/context/tokenizer.py` - Token assembly (legacy ring version)
- `stagebridge/models/stagebridge.py` - Main model
- `stagebridge/transition/drift.py` - CrossAttentionDrift head
- `stagebridge/training/trainer.py` - OT-CFM training loop

## TODO

- [ ] Test all three fusion modes (concat, precompute_gw, learned_gw)
- [ ] Verify gradients flow correctly for learned_gw
- [ ] Wire fusion modes into HPO search space
- [ ] Remove dead code paths (per-batch GW, legacy ring tokenizer)
- [ ] **FUTURE: SDE/Schrödinger Bridge** - Current OT-CFM is ODE (deterministic velocity). Consider SDE where niche modulates stochasticity of transition, not just direction. Schrödinger Bridge is most principled for connecting distributions stochastically.
