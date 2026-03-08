# Architecture: Reference Latent Mapping

**Scientific layer:** 2 — Reference latent mapping
**Package location:** `stagebridge/reference/`

## Role in the System

Reference latent mapping is the coordinate system of StageBridge. All downstream analysis — spatial mapping, context modeling, transition learning, evaluation — operates in the latent space produced by this layer.

## How It Works

1. **HLCA Atlas Loading** — The Human Lung Cell Atlas full reference (~20 GB h5ad) provides the reference latent space. It contains ~500K cells across human lung cell types.

2. **scArches Model Surgery** — The query dataset (StageBridge snRNA) has a different gene set than the reference. scArches performs model surgery to align the two, creating a query-compatible version of the reference model.

3. **Query Training** — The modified model is fine-tuned on the query data, learning to place StageBridge cells into the reference latent space while preserving reference structure.

4. **Embedding Extraction** — After training, each query cell gets a latent vector (typically 30D) and transferred cell-type labels from the reference.

## What Goes In

- snRNA-seq AnnData with raw counts and gene names
- HLCA reference atlas with pretrained model

## What Comes Out

- AnnData with HLCA latent embeddings in `.obsm['X_scvi']` or similar
- Cell-type label transfer table (parquet)
- Diagnostic reports (gene overlap, integration quality, label confidence)

## Key Design Decisions

- **HLCA, not custom PCA** — Using an external reference provides biological anchoring. Dataset-specific PCA is a fallback, not the default.
- **Diagnose, don't assume** — Integration quality diagnostics are mandatory. The system does not assume zero batch effects.
- **Labels feed context model** — Transferred cell-type labels become the token types for the context model (epithelial, stromal, immune, vascular).

## Relationship to Other Layers

- **Upstream:** Data ingestion provides the AnnData
- **Downstream:** Spatial mapping uses the embedded cells; context model uses the labels; transition model operates in the latent space
