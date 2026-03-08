# 004 — Spatial Mapping Contract

Spatial mapping code lives under `stagebridge/spatial_mapping/`. Tangram is the active implementation target. TACCO and DestVI interfaces exist as placeholders.

## Purpose

Spatial mapping bridges single-cell resolution (snRNA-seq) and spatial resolution (Visium). For each spatial spot, it estimates the likely cell-type composition. The outputs define the local tissue neighborhood that feeds the context model.

## Named Implementations

### Tangram (primary, first target)

Learns a mapping from snRNA-seq cells to spatial spots by optimizing gene expression alignment. Produces spot-level cell-type probability scores. Well-established, does not require a generative model, produces interpretable composition scores.

### TACCO (alternative)

Compositional transfer of annotations to spatial data. Estimates spot-level compositions using a different optimization framework.

### DestVI (alternative)

Deep generative model that decomposes spatial spots into cell-type proportions while also estimating continuous cell state. Richer output than pure composition but more complex to train.

## Common Output Contract

All implementations must produce the same downstream-compatible output:

| Output | Format | Description |
|--------|--------|-------------|
| Spot-level composition scores | DataFrame or `.obsm` matrix | Probability/proportion of each cell type per spot |
| Spatial AnnData | `.h5ad` | Spatial data augmented with composition scores |
| Composition parquet | `.parquet` | Spot x cell-type score table |
| Mapping report | `.json` | Runtime, convergence, and quality telemetry |

Swapping Tangram for TACCO should require changing one config flag, not rewriting downstream code.

## What Spatial Mapping Is Not

Spatial mapping is a preprocessing layer that converts raw spatial data into typed niche features. It is not the final model. Spatial mapping outputs feed into niche token construction, which feeds the context model.

## Niche Token Construction

From spatial mapping outputs, niche tokens are constructed per spot:

1. Cell-type composition vector (from Tangram/TACCO/DestVI)
2. Spatial neighborhood summary (k-nearest spots composition average)
3. Entropy and diversity features

These niche tokens become the typed tokens that populate biological sets. Token types correspond to broad lineages: epithelial, stromal, immune, vascular/program.
