# Architecture: Spatial Mapping Layer

**Scientific layer:** Input preprocessing
**Package location:** `stagebridge/spatial_mapping/`

## Role in the System

Spatial mapping connects single-cell identities to physical tissue locations. It answers: for each Visium spot, what cell types are present and in what proportions? These compositions define the typed niches that Layer B encodes.

**V1 requires benchmarking across multiple backends** to ensure robustness and justify the chosen method.

## Spatial Mapping Backends

### Tangram

Deep learning-based mapping that optimizes a cell-to-spot assignment matrix:
- Input: snRNA-seq AnnData (with cell-type labels), spatial AnnData
- Optimization: gradient descent (1000 epochs) maximizing cosine similarity
- Output: spot × cell-type probability matrix
- Runtime: ~1 hour per sample
- Early stopping: Not available (fixed iterations)

### TACCO

Optimal transport-based annotation transfer:
- Uses Sinkhorn OT to transfer annotations from reference to spatial data
- Probabilistic cell-type assignments per spot
- Single-pass optimization (no iterative training)
- Runtime: ~30 minutes per sample
- Early stopping: N/A (not iterative)

### DestVI

Variational inference deconvolution (scvi-tools):
- Two-stage training: CondSCVI (200 epochs) + DestVI (2500 epochs)
- Generative model for spot expression with cell-type proportions as latent
- Captures uncertainty via posterior sampling
- Supports gamma latent space for intra-cell-type variation
- Runtime: 2-4 hours per sample (with early stopping)
- Early stopping: Yes (patience=15/20, requires train_size=0.9 for validation set)

### Cell2location

Bayesian hierarchical model for absolute cell abundance:
- Two-stage: Reference signature model (250 epochs) + Spatial model (2500 epochs)
- Estimates absolute cell counts per location, not just proportions
- Accounts for detection efficiency and platform effects
- Runtime: 2-4 hours per sample
- Early stopping: Not available (no validation_step implementation)

## V1 Benchmark Design

The V1 publication includes a **4-backend × 2-label-source** spatial benchmark:

### Backends
All four backends (Tangram, DestVI, TACCO, Cell2location) run on each sample.

### Label Source Ablation
Each backend runs twice:
- **HLCA labels**: Cell types from Human Lung Cell Atlas (healthy reference)
- **LuCA labels**: Cell types from Lung Cancer Atlas (disease reference)

This tests whether disease-aware cell typing improves spatial deconvolution.

### Metrics

| Metric | Description |
|--------|-------------|
| Coverage | Fraction of spots with confident (>0.5) assignments |
| Mean entropy | Diversity of cell type predictions per spot |
| Sparsity | Fraction of near-zero proportions |
| Consistency | Agreement between backends on dominant cell types |

A robust result should be **backend-agnostic** — transition findings should hold across all four backends.

## From Spatial Scores to Niche Tokens

1. **Composition vector** — Per-spot probability distribution over cell types
2. **Neighborhood aggregation** — k-nearest spots' compositions averaged for local context
3. **Ring construction** — Compositions at increasing radii (Ring 1-4 tokens)
4. **Entropy features** — Shannon entropy captures niche diversity
5. **Token assignment** — Compositions grouped into the 9-token structure for Layer B

## What Goes In

- snRNA-seq AnnData with cell type labels (from HLCA or LuCA mapping)
- Spatial AnnData with spot coordinates and raw counts
- Sample manifest (for per-sample processing)

## What Comes Out

Per sample, per backend:
- `cell_type_proportions.parquet` — spot × cell-type probability matrix
- `upstream_metrics.json` — coverage, entropy, sparsity scores
- Backend-specific outputs:
  - Tangram: `tangram_mapper.npy`, `tangram_spatial_annotated.h5ad`
  - DestVI: `condscvi_model/`, `destvi_model/`, `destvi_gamma_*.csv`, training history CSVs
  - TACCO: `tacco_annotated_spatial.h5ad`
  - Cell2location: `cell_abundances.parquet`

Aggregated:
- `backend_comparison.json` — cross-backend metrics
- `canonical_backend.json` — selected backend + label source for downstream

## Key Design Decisions

- **Multiple backends** — Not locked to one method; benchmark determines choice
- **Common contract** — All methods produce the same output format
- **Preprocessing, not model** — Spatial mapping is feature extraction, not the scientific model
- **Quality diagnostics** — Mapping quality is monitored and reported

## Relationship to Other Layers

- **Upstream:** Layer A (reference mapping) provides cell-type labels for snRNA-seq
- **Downstream:** Layer B consumes niche tokens (ring compositions)
