# V1 Architecture Fix: AMICI Attention + Proper GW Fusion

**Status**: IMPLEMENTED  
**Date**: 2026-05-06

## Summary of Fixes

Three critical issues fixed:

1. **AMICI-style continuous attention** - Replaces ring binning with learned distance decay
2. **Proper GW fusion** - Precomputed on cell population, not per-batch on single cells
3. **Clean no_niche ablation** - Skips encoder entirely instead of zeroing inputs

---

## Fix 1: AMICI Continuous Attention

### Problem
Ring binning destroyed distance information. A cell at 5μm was treated the same as a cell at 45μm.

### Solution
Each attention head learns its own distance decay coefficient:
```
attn_logits = phenotype_score - distance_coef * distance
```

Where `distance_coef = softplus(MLP(receiver))` is learned and guaranteed positive.

### Interpretability
After training, plot `attention vs distance` per head to see learned interaction scales:
- High coefficient → steep decay → contact-dependent (~30μm)
- Medium coefficient → paracrine (~50-80μm)
- Low coefficient → diffusible factors (~100μm+)

### Usage
```python
# Data prep
PrepConfig(use_continuous_attention=True)

# Model
StageBridgeConfig(use_amici_attention=True)
```

---

## Fix 2: Proper Gromov-Wasserstein Fusion

### Problem
Per-batch GW computed alignment on 1x1 "distance matrices" - mathematically meaningless.
GW needs population structure to find structure-preserving alignment.

### Solution: Offline Precomputation

1. **Sample representative cells** (~5k, stratified by stage/celltype)
2. **Compute distance matrices** D_HLCA [N×N], D_LuCA [N×N]
3. **Solve GW** to find optimal coupling P* that preserves relative geometry
4. **Train neural transport map** supervised by the coupling
5. **Use neural map at inference** - no per-batch GW

### GW Objective (Eq. 12 from Bunne et al.)
```
min_P  Σ_ijkl P_ij P_kl (D_HLCA[i,k] - D_LuCA[j,l])²
```
Find coupling P such that if cells i,k are close in HLCA, their matched cells j,l are close in LuCA.

### Usage
```bash
# Step 1: Precompute alignment
python scripts/precompute_gw_alignment.py \
    --cells /path/to/cells.parquet \
    --output-dir /path/to/gw_alignment \
    --n-reference-cells 5000

# Step 2: Train with pretrained fusion
StageBridgeConfig(
    use_gw_fusion=True,
    gw_fusion_type="pretrained",
    gw_checkpoint_dir="/path/to/gw_alignment",
)
```

### Fusion Options
| Type | Description | Recommended |
|------|-------------|-------------|
| `pretrained` | Load precomputed GW alignment | Yes (proper) |
| `learned_projection` | Simple learned weighted projection | Fallback |
| `False` (concat) | Simple concatenation | Baseline |

**Note**: The broken `per_batch` implementation has been removed.

---

## Fix 3: Clean no_niche Ablation

### Problem
Old code zeroed inputs but still ran through learnable niche modules - leaky ablation.

### Solution
Skip encoder entirely when `use_niche_context=False`:
```python
if not self.config.use_niche_context:
    niche_context = torch.zeros(B, hidden_dim, device=device)
    # Don't run any niche modules
else:
    niche_context = self.amici_encoder(...)
```

---

## Files Changed

| File | Change |
|------|--------|
| `stagebridge/pipelines/prepare_data.py` | AMICI format with raw distances |
| `stagebridge/loaders/dataset.py` | `AMICIBatch`, `collate_amici_batch` |
| `stagebridge/models/stagebridge.py` | AMICI config, `encode_niche_amici()`, GW options, clean ablation |
| `stagebridge/reference/gw_precompute.py` | **NEW**: Offline GW computation + neural map |
| `stagebridge/reference/gw_fusion.py` | Added `PrecomputedGWFusion` |
| `scripts/precompute_gw_alignment.py` | **NEW**: CLI for GW precomputation |
| `tests/test_model.py` | `TestAMICIEncoder` |

---

## Validation

All 26 model tests pass:
- Original functionality preserved
- AMICI encoder works with distance decay
- Empty token handles masked neighbors
- no_niche ablation is clean
- GW fusion options all instantiate correctly

---

## Next Steps

1. Reprocess data with `use_continuous_attention=True`
2. Run `scripts/precompute_gw_alignment.py` on your cells.parquet
3. Train with AMICI + pretrained GW
4. Compare to ring-based baseline
5. Visualize learned distance decay patterns

---

## References

- Hong et al. (2025) "AMICI: Attention-based Multi-scale cell-cell Interaction"
- Bunne et al. (2024) "Optimal transport for single-cell and spatial omics" - Nature Reviews Methods Primers
