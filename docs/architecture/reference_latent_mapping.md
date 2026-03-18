# Architecture: Dual-Reference Latent Mapping (Layer A)

**Scientific layer:** A — Reference geometry
**Package location:** `stagebridge/reference/`

## Role in the System

Layer A produces the dual-reference latent space where cells are embedded relative to both healthy (HLCA) and tumor (LuCA) atlases. This geometry anchors the transition model — cells move through a space defined by their relationship to known biological references.

**V1 uses Euclidean geometry. Non-Euclidean (hyperbolic/spherical) is deferred to V2.**

## Dual-Reference Design

### Why Two Atlases?

Single-reference embedding loses information:
- HLCA alone cannot distinguish tumor subtypes
- LuCA alone lacks healthy baseline context

Dual-reference captures biological asymmetry:
- Early stages (Normal, AAH): high HLCA similarity, low LuCA similarity
- Late stages (LUAD): low HLCA similarity, high LuCA similarity
- The transition is movement in this dual space

### HLCA (Human Lung Cell Atlas)

The healthy lung reference (~584K cells):

| Property | Value |
|----------|-------|
| Source | CZI cellxgene via scvi-tools Hugging Face Hub |
| Repository | `scvi-tools/human-lung-cell-atlas-scanvi` |
| Latent key | `X_scanvi_emb` |
| Latent dimensions | 30 |
| Required obs columns | `ann_level_1`, `ann_level_2`, `ann_level_3` |

**Pipeline:**
1. **Atlas loading** — HLCA reference h5ad with pretrained scVI/scArches model
2. **Query alignment** — Gene set surgery to match reference (handles ENSG vs symbol via `feature_name`)
3. **Embedding** — Project query cells into HLCA latent manifold via k-NN in gene space
4. **Confidence** — Percentile-rank calibrated confidence (density-independent)

Output: `hlca_latent_0..29` (30-dim latent) + `hlca_confidence` (calibrated [0,1])

### LuCA (Lung Cancer Atlas)

The tumor reference for cancer-specific context:

| Property | Value |
|----------|-------|
| Source | Zenodo / LungCancerAtlas GitHub |
| Latent key | `X_scVI` |
| Latent dimensions | 10 |
| Required obs columns | `cell_type` |

**CRITICAL: LuCA Core vs Extended**

| Version | Cells | Latent Integrity | Recommendation |
|---------|-------|------------------|----------------|
| Core | ~790K | 100% valid | USE THIS |
| Extended | ~1.3M | 69% valid (31% NaN) | DO NOT USE |

Always verify latent integrity before mapping:

```bash
python -m stagebridge.reference.diagnose_reference /path/to/luca.h5ad \
    --latent-key X_scVI --diagnose-only
```

Output: `luca_latent_0..9` (10-dim latent) + `luca_confidence` (calibrated [0,1])

## V1: Euclidean Geometry

For V1, cells are embedded in Euclidean space:
- HLCA and LuCA similarities are concatenated or processed separately
- Distance metrics are standard L2
- Flow matching operates in this flat geometry

This is sufficient for the core scientific claims about niche-gated transitions.

## V2: Non-Euclidean Geometry (Deferred)

Non-Euclidean embeddings may better capture:
- Hierarchical cell-type relationships (hyperbolic)
- Cyclical/compositional structure (spherical)
- Mixed curvature for complex manifolds

These are **not required for V1** but provide future extension paths.

## Atlas Ablation

The evaluation framework tests each atlas configuration:

| Mode | HLCA | LuCA | Scientific Question |
|------|------|------|---------------------|
| `no_atlas` | Zeroed | Zeroed | Can spatial structure alone predict transitions? |
| `hlca_only` | Active | Zeroed | Does healthy reference suffice? |
| `luca_only` | Zeroed | Active | Does cancer reference suffice? |
| `hlca_luca` | Active | Active | Do both atlases together help? |
| `hlca_luca_contrast` | Active | Active + contrast | Does cross-atlas modeling add lift? |

## What Goes In

- snRNA-seq AnnData with raw counts
- HLCA reference h5ad with `X_scanvi_emb` latent (30 dims)
- LuCA Core reference h5ad with `X_scVI` latent (10 dims)

## What Comes Out

Output directory: `reference_geometry/`

| File | Contents |
|------|----------|
| `hlca_embedding.parquet` | cell_id, donor_id, sample_id, stage_id, hlca_latent_0..29 |
| `luca_embedding.parquet` | cell_id, donor_id, sample_id, stage_id, luca_latent_0..9 |
| `fused_embedding.parquet` | All metadata + hlca_latent + luca_latent + fused_latent |
| `reference_confidence.parquet` | cell_id, hlca_confidence, luca_confidence, *_method, reference_mode_used |
| `reference_manifest.json` | Run metadata, parameters, timestamps |
| `feature_overlap_report.json` | Gene overlap statistics for each reference |

## Quality Diagnostics

Integration quality must be verified:
- **Reference integrity**: `diagnose_reference.py` checks for NaN in latents
- **Gene overlap**: >30% overlap required (reported in `feature_overlap_report.json`)
- **Mapping collapse**: Variance check ensures cells don't map to single point
- **Confidence calibration**: Percentile-rank ensures cross-reference comparability

Tools:
```bash
# Check reference latent integrity
python -m stagebridge.reference.diagnose_reference /path/to/ref.h5ad --diagnose-only

# Full pipeline with validation
python -m stagebridge.pipelines.run_reference --data-root $DATA
```

## Relationship to Other Layers

- **Upstream:** Step 0 data pipeline provides merged AnnData
- **Downstream:** Layer B receives HLCA/LuCA features as tokens; Layer D operates in this latent space
