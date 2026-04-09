# StageBridge Dual-Reference Encoder Architecture

## Overview

StageBridge uses a **dual-reference geometry** where each cell is represented by its
coordinates in two reference atlas spaces:

- **HLCA** (Human Lung Cell Atlas): 30-dimensional scANVI latent - healthy reference
- **LuCA** (Lung Cancer Atlas): 10-dimensional scVI latent - disease-aware reference

## Data Flow

```
Reference Mapping (run_reference.py)
├── Query cells → HLCA scANVI surgery → z_hlca [N, 30]
├── Query cells → LuCA scVI surgery  → z_luca [N, 10]
└── Concatenation                    → z_fused [N, 40]

Data Preparation (complete_data_prep.py)
└── cells.parquet
    ├── z_hlca_0..29  (30 columns)
    ├── z_luca_0..9   (10 columns)
    └── z_fused_0..39 (40 columns)

Training (run_v1_ddp.py)
├── Load fused embeddings → [N, 40]
└── Create niche tokens   → [N, 9, 40]
```

## Encoder Architecture

### ReceiverCenteredNicheEncoder (V1 Choice)

Uses the fused embedding directly. The linear projection learns to weight HLCA vs LuCA.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ReceiverCenteredNicheEncoder                     │
├─────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                             │
│    receiver = niche_tokens[:, 0, :]  → [B, 40] = [HLCA | LuCA]     │
│    neighbors = niche_tokens[:, 1:K, :] → [B, K, 40]                │
│    distances = spatial distances     → [B, K]                       │
├─────────────────────────────────────────────────────────────────────┤
│  PROJECTION (Learned Fusion):                                       │
│    h_receiver = Linear(40 → 128)(receiver)                         │
│                 ↑                                                   │
│                 This learns: W @ [hlca; luca] + b                   │
│                 W[:, 0:30] weights HLCA features                    │
│                 W[:, 30:40] weights LuCA features                   │
│                                                                     │
│    h_neighbors = Linear(40 → 128)(neighbors)  → [B, K, 128]        │
├─────────────────────────────────────────────────────────────────────┤
│  ATTENTION LAYERS (×2):                                             │
│    For each layer:                                                  │
│      1. Q = W_q @ h_receiver         → [B, num_heads, 1, head_dim] │
│      2. K = W_k @ h_neighbors        → [B, num_heads, K, head_dim] │
│      3. V = W_v @ h_neighbors        → [B, num_heads, K, head_dim] │
│                                                                     │
│      4. attn_logits = Q @ K.T / sqrt(head_dim)                     │
│      5. attn_logits += distance_bias (RBF encoded)                 │
│      6. attn_weights = softmax(attn_logits)                        │
│      7. context = attn_weights @ V   → [B, 128]                    │
│                                                                     │
│      8. h_receiver = LayerNorm(h_receiver + context)               │
│      9. h_receiver = LayerNorm(h_receiver + FFN(h_receiver))       │
│                                                                     │
│    FFN = Linear(128→512) → GELU → Dropout → Linear(512→128)        │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT:                                                            │
│    context = Linear(128→128) → GELU → LayerNorm → [B, 128]         │
│    → Projected to context_dim (256) in StageBridgeV1Complete       │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Is Scientifically Correct

1. **No separate expression embedding exists**: The HLCA/LuCA coordinates ARE
   the cell representations. There's no raw expression embedding to use as "input".

2. **Learned fusion via projection**: The `Linear(40 → 128)` layer learns:
   ```
   h = W @ [z_hlca; z_luca] + b
   ```
   Where different rows of W can emphasize HLCA vs LuCA features differently.

3. **Attention learns context-dependent weighting**: The multi-head attention
   can learn to weight HLCA vs LuCA differently depending on:
   - Cell type context (via learned representations)
   - Spatial neighborhood (via distance-modulated attention)
   - Disease stage (via training signal)

4. **No redundancy**: Passing [fused | hlca | luca] = [[hlca|luca] | hlca | luca]
   wastes parameters and adds no information.

## Configuration

```python
# run_v1_ddp.py TrainingConfig
latent_dim: int = 40   # Fused embedding dimension (30 HLCA + 10 LuCA)
hlca_dim: int = 30     # For reference/logging
luca_dim: int = 10     # For reference/logging

# Model instantiation
ReceiverCenteredNicheEncoder(
    input_dim=40,        # Fused dimension
    hidden_dim=128,      # Internal representation
    num_heads=4,         # Multi-head attention
    num_layers=2,        # Attention depth
    dropout=0.1,
    use_reconstruction_head=True,  # For SSL
)
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fusion method | Learned projection | Linear layer learns optimal weighting |
| Separate HLCA/LuCA inputs | No | Redundant - already in fused |
| Encoder type | ReceiverCenteredNicheEncoder | Simpler, equally expressive |
| Distance encoding | RBF | Captures spatial locality |
| Sparsity | Entropy regularization | Encourages focused attention |

## Future Improvements (V2)

1. **Confidence-weighted fusion**: Weight HLCA/LuCA by mapping confidence
2. **Cross-reference attention**: Explicit attention between HLCA and LuCA spaces
3. **Gated fusion**: Learn gates for each reference based on cell type

## Files

- `stagebridge/context_model/receiver_niche_encoder.py` - Encoder implementation
- `stagebridge/pipelines/run_v1_complete.py` - Model definition
- `stagebridge/pipelines/run_v1_ddp.py` - Training pipeline
- `stagebridge/pipelines/complete_data_prep.py` - Data preparation
