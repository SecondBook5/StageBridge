# StageBridge Ablation Studies

This document describes the ablation experiments for the StageBridge paper.

## Implementation

All ablations use `run_v1_ablations.py`, which is a copy of `run_v1_ddp.py`
(production training) with ablation flags added. This ensures ablations use
the **same model** (StageBridgeV1Complete) as the full model results.

**IMPORTANT**: Do NOT use `run_v1_full.py` - it uses a different model class
(StageBridgeV1Full) and has been archived to `_archived/`.

## Ablation Table

| Ablation | CLI Flag | Scientific Question | Implementation |
|----------|----------|---------------------|----------------|
| **no_niche** | `--no_niche` | Does local neighborhood context improve prediction? | Zeros ring tokens (1-4), keeps receiver only |
| **no_wes** | `--no_wes` | Do WES/genomic features help evolutionary regularization? | Zeros WES feature tensor |
| **pooled_niche** | `--niche_encoder_type self_attention` | Does receiver-centered cross-attention help vs flat attention? | Uses self-attention over all tokens |
| **flat_hierarchy** | `--no_hierarchical` | Does hierarchical Set Transformer aggregation help? | Disables hierarchical aggregator |
| **hlca_only** | `--fusion_mode hlca_only` | Is the cancer reference (LuCA) necessary? | Zeros LuCA token (6) |
| **luca_only** | `--fusion_mode luca_only` | Is the healthy reference (HLCA) necessary? | Zeros HLCA token (5) |
| **deterministic** | `--deterministic` | Does flow matching help vs direct endpoint prediction? | Uses t=1 direct prediction instead of OT-CFM |
| **with_prototypes** | `--use_prototypes` | Do learned prototypes aid interpretability? | Enables prototype bottleneck |

## Token Structure

The model uses 9 tokens:
- Token 0: Receiver cell embedding (fused HLCA+LuCA)
- Tokens 1-4: Ring embeddings (spatial neighborhood at different radii)
- Token 5: HLCA reference embedding
- Token 6: LuCA reference embedding
- Token 7: Pathway/functional state
- Token 8: Neighborhood statistics

## Key Hypotheses Tested

### H1: Niche Context Matters (no_niche ablation)
The core hypothesis is that progression dynamics depend on local tissue microenvironment.
If no_niche performs similarly to full model, the niche hypothesis is not supported.

### H2: Dual Reference Geometry (hlca_only, luca_only ablations)
Tests whether both healthy (HLCA) and cancer (LuCA) reference atlases provide
complementary information. If single reference performs equally well, dual
reference is not necessary.

### H3: Flow Matching vs Direct Prediction (deterministic ablation)
Tests whether modeling the continuous flow field provides better transition
predictions than directly predicting endpoints. Key for Nature Methods novelty.

### H4: Hierarchical Structure (flat_hierarchy, pooled_niche ablations)
Tests whether the hierarchical Set Transformer architecture captures meaningful
structure beyond flat attention.

## Running Ablations

### Via Snakemake (recommended)
```bash
snakemake --profile workflow/slurm ablation_summary
```

### Manual single ablation
```bash
python -m stagebridge.pipelines.run_v1_ablations \
    --data_dir /path/to/canonical \
    --output_dir /path/to/output \
    --no_niche  # or other ablation flag
```

### Via orchestrator
```bash
python -m stagebridge.pipelines.run_ablations \
    --data_dir /path/to/canonical \
    --output_dir /path/to/ablations \
    --ablations no_niche no_wes hlca_only
```

## Expected Results

For the paper, we expect:
1. **no_niche** should show significant degradation (validates core hypothesis)
2. **hlca_only** and **luca_only** should both degrade (validates dual reference)
3. **deterministic** should show modest degradation (validates flow matching)
4. **with_prototypes** should maintain performance (interpretability without cost)

If ablations do NOT degrade performance, we need to revisit the corresponding
architectural claim.
