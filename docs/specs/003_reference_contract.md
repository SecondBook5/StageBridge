# 003 Reference Contract

Reference mapping code lives under `stagebridge/reference/`.
HLCA mapping is the active reference anchor for v1.

## Purpose

Projects all query cells into HLCA latent space via scArches surgery. Provides biologically anchored coordinates, cross-dataset comparability, and cell-type label transfer.

## Why HLCA

Most comprehensive human lung reference. Covers cell types relevant to LUAD initiation. HLCA latent space means the transition model operates in a biologically anchored coordinate system and new datasets can be added without recomputing.

## Implementation

1. Load HLCA reference model
2. Model surgery for query gene set
3. Train query model on StageBridge snRNA data
4. Extract latent embeddings and transferred labels

## Integration Is Not Assumed Perfect

Required diagnostics:
- Gene overlap report
- Latent space distribution checks
- Label transfer confidence distribution
- Stage-specific plausibility checks

Diagnostics are not optional. Must be produced and reviewed before downstream analysis.

## Graceful Fallback

If HLCA unavailable, fall back to PCA. Weaker, must be flagged in results, must not silently substitute.

## Expected Artifacts

| Artifact | Description |
|----------|-------------|
| `snrna_hlca_latent_{experiment}.h5ad` | HLCA latent embeddings in `.obsm` |
| `snrna_{experiment}_hlca_labels.parquet` | Label transfer table |
| `query_model_{experiment}/` | Trained scArches model |
| `hlca_mapping_report.json` | Runtime telemetry |
| `hlca_gene_id_report.json` | Gene overlap diagnostics |
| `hlca_validation_report.json` | Quality assessment |
