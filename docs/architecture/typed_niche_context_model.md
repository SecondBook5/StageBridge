# Architecture: Local Niche Encoder (Layers B+C)

**Scientific layers:** B (Local Niche Encoding) + C (Hierarchical Aggregation)
**Package location:** `stagebridge/context_model/`

## Role in the System

Layers B and C encode local tissue microenvironments (niches) into representations that condition the transition model (Layer D). These layers are derived from the EA-MIST architecture but repurposed: the primary output is **niche context for conditioning transitions**, not lesion-level classification.

The EA-MIST lesion classification heads remain available as auxiliary losses but are not the central objective.

## Architecture Overview

```
Layer B: Local Niche Encoder
  - 9-token sequence per niche
  - Self-attention over tokens
  - Output: per-niche embedding

Layer C: Hierarchical Aggregation
  - Set transformer over niches
  - Optional prototype bottleneck
  - Output: aggregated context vector for Layer D
```

## Layer B: Local Niche Encoder

### Token Construction

Each niche is encoded as a **9-token sequence**:

| Token | ID | Source | Projection |
|-------|-----|--------|------------|
| Receiver | 0 | Cell expression + state | `Linear(D_r → dim) + StateEmb + TypeEmb` |
| Ring 1 | 1 | Composition at radius 1 | `Linear(D_s → dim) + RingEmb + TypeEmb` |
| Ring 2 | 1 | Composition at radius 2 | `Linear(D_s → dim) + RingEmb + TypeEmb` |
| Ring 3 | 1 | Composition at radius 3 | `Linear(D_s → dim) + RingEmb + TypeEmb` |
| Ring 4 | 1 | Composition at radius 4 | `Linear(D_s → dim) + RingEmb + TypeEmb` |
| HLCA | 2 | Healthy atlas similarity | `Linear(13 → dim) + TypeEmb` |
| LuCA | 3 | Tumor atlas similarity | `Linear(15 → dim) + TypeEmb` |
| Pathway | 4 | L-R activity summary | `Linear(D_lr → dim) + TypeEmb` |
| Stats | 5 | Density, entropy, etc. | `Linear(D_stats → dim) + TypeEmb` |

Optional 10th token (atlas contrast) when `use_atlas_contrast_token: true`.

### Self-Attention

Token sequence processed by SAB (Self-Attention Block) layers:
```
For each SAB: MultiHeadAttn(Q=X, K=X, V=X) → Residual → LayerNorm → FFN
```

### Pooling

PMA (Pooling by Multihead Attention) reduces to single niche embedding:
```
output = MultiHeadAttn(Q=seed, K=tokens, V=tokens) → LayerNorm
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_dim` | 128 | Token and output dimension |
| `num_heads` | 4 | Attention heads |
| `num_layers` | 2 | SAB layers |
| `num_rings` | 4 | Spatial distance rings |
| `dropout` | 0.1 | Dropout rate |

## Layer C: Hierarchical Aggregation

### Set Transformer Backbone

Aggregates variable-length set of niche embeddings:

| Block | Function |
|-------|----------|
| **ISAB** | Induced Set Attention with M inducing points (O(NM) complexity) |
| **SAB** | Full self-attention refinement |
| **PMA** | Pool to fixed-size output |

### Prototype Bottleneck (Optional)

Soft assignment to K learned prototypes:
- `assignment = softmax(embeddings @ prototypes.T / sqrt(d))`
- Encourages interpretable niche clustering
- Regularized by diversity and entropy losses

### Evolution Branch (Optional)

Conditions aggregated embedding on WES features:
- **Gated mode:** `fused = gate * z + (1-gate) * evo_proj`
- **FiLM mode:** `fused = z * (1 + γ) + β`

### Output

The output of Layer C is the **context vector** that conditions Layer D (transition model):
- Shape: `(batch, hidden_dim)`
- Contains niche-level information aggregated per sample
- Passed to velocity network as conditioning signal

## Auxiliary Outputs (Lesion Classification)

The EA-MIST multitask heads remain available for auxiliary supervision:

| Head | Output | Role in V1 |
|------|--------|------------|
| Stage | 5-way logits | Auxiliary loss (not primary) |
| Displacement | Scalar [0,1] | Auxiliary ordinal signal |
| Edge | Pairwise logits | Optional auxiliary |

These provide additional training signal but the model is evaluated on **transition quality**, not classification accuracy.

## Reference Feature Modes

Atlas features can be selectively ablated:

| Mode | HLCA | LuCA | Description |
|------|------|------|-------------|
| `no_atlas` | Zeroed | Zeroed | Spatial-only baseline |
| `hlca_only` | Active | Zeroed | Healthy atlas only |
| `luca_only` | Zeroed | Active | Cancer atlas only |
| `hlca_luca` | Active | Active | Both atlases (V1 default) |
| `hlca_luca_contrast` | Active | Active + contrast token | Cross-atlas modeling |

## Model Variants

| Variant | Layer B | Layer C | Use |
|---------|---------|---------|-----|
| `eamist` | Full encoder | Set transformer + prototypes | Primary |
| `eamist_no_prototypes` | Full encoder | Set transformer only | Ablation |
| `deep_sets` | Full encoder | DeepSets φ→ρ | Baseline |
| `pooled` | Full encoder | Mean pooling | Baseline |

## Relationship to Other Layers

- **Upstream:** Layer A (reference mapping) provides HLCA/LuCA embeddings; spatial mapping provides compositions
- **Downstream:** Layer D (transition model) receives context vector as conditioning input
