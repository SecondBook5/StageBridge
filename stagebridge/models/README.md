# Models

Core StageBridge model and prediction heads.

## StageBridge Model

The main model in `stagebridge.py` combines:

1. **Niche Encoding**: AMICI-style receiver-centered attention
2. **Context Refinement**: HierarchicalSetTransformer (ISAB -> SAB -> PMA)
3. **Drift Prediction**: CrossAttentionDrift for velocity fields
4. **Auxiliary Heads**: Pathway and proliferation prediction

### Data Formats

**AMICI Format** (recommended):
```python
# Continuous neighbors with distances
output = model.encode_niche_amici(
    receiver,      # [B, 40]
    neighbors,     # [B, K, 40]
    distances,     # [B, K]
    neighbor_mask, # [B, K]
    hlca, luca, pathway, stats,
)
```

**Ring Format** (legacy):
```python
# Discrete ring binning
output = model.encode_niche(
    receiver,    # [B, 40]
    ring_cells,  # List of 4 [B, max_cells, 40]
    ring_masks,  # List of 4 [B, max_cells]
    hlca, luca, pathway, stats,
)
```

### Forward Pass

```python
# 1. Encode niche
niche = model.encode_niche_amici(receiver, neighbors, distances, mask, ...)

# 2. Predict velocity at time t
v_t = model.forward_vector_field(x_t, t, niche.context, stage_pair_id)

# 3. Integrate trajectory
x_final = model.integrate_euler(x_start, niche.context, stage_pair_id, steps=8)
```

### Ablations

Controlled via StageBridgeConfig flags:

| Flag | Effect |
|------|--------|
| use_niche_context=False | no_niche ablation |
| use_hlca_reference=False | luca_only ablation |
| use_luca_reference=False | hlca_only ablation |
| use_gw_fusion=False | concat baseline |

## Auxiliary Heads (`heads.py`)

### PathwayHead

Predicts PROGENy pathway activities from niche context.

```python
pathway_logits = model.pathway_head(context)  # [B, 14]
```

### ProliferationHead

Predicts Ki67/proliferation state.

```python
prolif_logit = model.proliferation_head(context)  # [B, 1]
```

These provide biological regularization during training.
