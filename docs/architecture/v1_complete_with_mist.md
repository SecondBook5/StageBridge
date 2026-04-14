# StageBridge V1Complete with MIST Aggregation

## Overview

StageBridge V1Complete combines receiver-centered niche encoding with hierarchical
set aggregation from EA-MIST (Evolutionary-Aware Multiple Instance Set Transformer).
This enables both cell-level and sample-level predictions.

## Architecture Diagram

```
                        CELL-LEVEL PATH (existing)
                        ===========================
                        
    Niche Tokens [B, K=9, D=40]
           |
           v
    +---------------------------+
    | ReceiverCenteredNiche     |  Cross-attention: receiver as query,
    | Encoder                   |  neighbors as key/value
    +---------------------------+
           |
           v
    niche_embedding [B, 128]
           |
           +--------+--------+--------+
           |        |        |        |
           v        v        v        v
    SSL Heads   Transition  Counter-  IL1B
    (masked,    (OT-CFM)    factual   Head
    ranking)


                        SAMPLE-LEVEL PATH (new from EA-MIST)
                        ====================================
                        
    Niche Batch [B, N, K=9, D=40]     B = samples, N = niches per sample
           |
           v
    +---------------------------+
    | ReceiverCenteredNiche     |  Encode each niche independently
    | Encoder (batched)         |  (B*N forward passes)
    +---------------------------+
           |
           v
    niche_embeddings [B, N, 128]
           |
           v (optional)
    +---------------------------+
    | PrototypeBottleneck       |  K=16 learned niche "motifs"
    | (interpretability)        |  Soft assignment to prototypes
    +---------------------------+
           |
           v
    +---------------------------+
    | HierarchicalAggregator    |  ISAB layers + PMA pooling
    | (EA-MIST Layer C)         |  Permutation-invariant aggregation
    +---------------------------+
           |
           v
    sample_embedding [B, 128]
           |
           v (optional)
    +---------------------------+
    | EvolutionBranch           |  Gated fusion with WES features
    | (WES conditioning)        |  (TMB, smoking_sig, UV_sig)
    +---------------------------+
           |
           v
    +---------------------------+
    | SampleLevelHeads          |
    | - Stage classification    |  5-class: Normal, AAH, AIS, MIA, LUAD
    | - Displacement            |  Sample-level transition vector
    +---------------------------+
```

## Key Components

### 1. ReceiverCenteredNicheEncoder (Cell-Level)

Cross-attention architecture where the receiver cell queries its neighbors:
- **Query**: Receiver cell embedding
- **Key/Value**: Neighbor cell embeddings
- **Output**: Context-aware receiver representation

This is different from EA-MIST's LocalNicheTransformerEncoder which uses
self-attention over all 9 tokens equally.

### 2. HierarchicalAggregator (Sample-Level)

From EA-MIST Layer C. Aggregates multiple niches per sample using:
- **ISAB layers**: Induced Set Attention Blocks with learnable inducing points
- **PMA**: Pooling by Multihead Attention to single sample vector
- **Complexity**: O(N*M) where M = num_inducing_points (default 16)

### 3. PrototypeBottleneck (Optional)

Compresses niche embeddings to K learnable prototypes:
- Soft assignment via cosine similarity
- Enables interpretation: "What niche motifs dominate this lesion?"
- Output: prototype_composition [B, K] showing motif proportions

### 4. EvolutionBranch (Optional)

Gated fusion of WES features at sample level:
- **Gated mode**: `gate * lesion_emb + (1-gate) * wes_emb`
- **FiLM mode**: `lesion_emb * (1 + gamma) + beta`
- WES features: TMB, smoking signature, UV signature

## Usage

### Cell-Level (Existing API)

```python
model = StageBridgeV1Complete(use_hierarchical=False)

# Single niche encoding
niche_tokens = torch.randn(batch, 9, 40)  # [B, K, D]
context = model.encode_niche(niche_tokens)  # [B, 256]

# SSL forward
ssl_out = model.ssl_forward(niche_tokens, receiver_target)

# Transition forward
trans_out = model.transition_forward(z_source, z_target, context)
```

### Sample-Level (New API)

```python
model = StageBridgeV1Complete(
    use_hierarchical=True,
    use_prototypes=True,
    use_evolution_branch=True,
)

# Multiple niches per sample
niche_batch = torch.randn(batch, niches_per_sample, 9, 40)  # [B, N, K, D]
niche_mask = torch.ones(batch, niches_per_sample, dtype=torch.bool)
wes = torch.randn(batch, 3)  # WES features

output = model.sample_forward(niche_batch, niche_mask=niche_mask, wes_features=wes)

# Outputs
output["sample_embedding"]    # [B, 128] sample representation
output["stage_logits"]        # [B, 5] stage classification
output["displacement"]        # [B, 128] transition vector
output["prototype_output"]    # PrototypeBottleneckOutput
output["evolution_embedding"] # [B, 128] WES-conditioned embedding
```

## Constructor Parameters

```python
StageBridgeV1Complete(
    # Existing parameters
    latent_dim=40,              # Fused embedding dim (30 HLCA + 10 LuCA)
    niche_hidden_dim=128,       # Niche encoder hidden dim
    context_dim=256,            # Context projection dim
    dropout=0.1,
    
    # ABLATION: Niche encoder attention type
    niche_encoder_type="cross_attention",  # "cross_attention" or "self_attention"
    
    # Hierarchical aggregation (from EA-MIST)
    use_hierarchical=True,      # Enable sample-level path
    hierarchical_num_layers=2,  # Number of ISAB layers
    hierarchical_num_inducing=16,  # Inducing points per ISAB
    
    # Prototype bottleneck
    use_prototypes=False,       # Enable prototype compression
    num_prototypes=16,          # Number of learned motifs
    
    # Evolution conditioning
    use_evolution_branch=True,  # Gated WES fusion
    evolution_mode="gated",     # "gated" or "film"
    wes_feature_dim=3,          # TMB, smoking_sig, UV_sig
    
    # Task heads
    num_stage_classes=5,        # Normal, AAH, AIS, MIA, LUAD
)
```

## Ablations

### Niche Encoder: Cross-Attention vs Self-Attention

This ablation tests whether **architectural enforcement of receiver-centrality** helps,
or whether self-attention can learn to focus on the receiver naturally.

| Encoder Type | Attention Mechanism | Receiver Privileged? |
|--------------|---------------------|---------------------|
| `cross_attention` (default) | Receiver queries neighbors | Yes (architecturally) |
| `self_attention` (ablation) | All tokens attend to all | No (must learn) |

**Cross-Attention** (ReceiverCenteredNicheEncoder):
```
receiver = Query
neighbors = Key, Value
context = CrossAttention(receiver, neighbors, neighbors)
```

**Self-Attention** (SelfAttentionNicheEncoder):
```
tokens = [receiver, neighbor_1, ..., neighbor_K]
all_tokens = SelfAttention(tokens, tokens, tokens)
context = all_tokens[0]  # Extract receiver's updated representation
```

**Hypothesis**: Cross-attention should outperform self-attention because:
1. The biological question is receiver-centric ("How does niche affect THIS cell?")
2. Self-attention may waste capacity learning spurious neighbor-neighbor relations
3. Cross-attention provides stronger inductive bias

**Run ablation**:
```python
# Cross-attention (default)
model_cross = StageBridgeV1Complete(niche_encoder_type="cross_attention")

# Self-attention (ablation)
model_self = StageBridgeV1Complete(niche_encoder_type="self_attention")
```

## Validation Hypotheses Enabled

### H2: Methodological (Cell-Level)
> "Cross-sectional progression becomes more identifiable when conditioned on
> receiver-centered local niche context"

Uses cell-level path with `encode_niche()` and `ssl_forward()`.

### H3: Clonal Evolution (Sample-Level)
> "Transition probability should correlate with shared clones across stages"

Uses sample-level path with `sample_forward()` to aggregate all niches
from a sample/lesion for clone-based predictions.

## Comparison: V1Complete vs Original EA-MIST

| Feature | EA-MIST | V1Complete+MIST |
|---------|---------|-----------------|
| Niche encoder | LocalNicheTransformerEncoder (SAB) | ReceiverCenteredNicheEncoder (cross-attn) |
| Token structure | 9 tokens: receiver + 4 rings + HLCA + LuCA + pathway + stats | 9 tokens (same) |
| Hierarchical agg | LesionSetTransformerBackbone | HierarchicalAggregator (same ISAB/PMA) |
| Prototypes | Yes (default on) | Optional |
| Evolution branch | Yes | Optional |
| SSL pretraining | No | Yes (70% masked reconstruction) |
| OT-CFM transition | No | Yes |
| Counterfactual | No | Yes |
| IL1B head | No | Yes |

## Files

- **Model**: `stagebridge/pipelines/run_v1_complete.py`
- **Hierarchical components**: `stagebridge/context_model/set_encoder.py`
- **Prototype bottleneck**: `stagebridge/context_model/prototype_bottleneck.py`
- **Evolution branch**: `stagebridge/context_model/evolution_branch.py`
- **Original EA-MIST**: `stagebridge/context_model/lesion_set_transformer.py`
