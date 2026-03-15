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
- Optimization: gradient descent maximizing cosine similarity of mapped expression
- Output: spot × cell-type probability matrix

### TACCO

Optimal transport-based annotation transfer:
- Uses OT to transfer annotations from reference to spatial data
- Probabilistic cell-type assignments per spot
- Computationally efficient

### DestVI

Variational inference deconvolution:
- Generative model for spot expression
- Infers cell-type proportions as latent variables
- Captures uncertainty in assignments

## V1 Benchmark Requirement

The V1 publication **must** include a spatial backend benchmark:

| Metric | Description |
|--------|-------------|
| Reconstruction error | How well do inferred compositions explain spot expression? |
| Consistency | Do methods agree on dominant cell types? |
| Downstream impact | Does transition model performance vary by backend? |

A robust result should be **backend-agnostic** — transition findings should hold across Tangram, TACCO, and DestVI.

## From Spatial Scores to Niche Tokens

1. **Composition vector** — Per-spot probability distribution over cell types
2. **Neighborhood aggregation** — k-nearest spots' compositions averaged for local context
3. **Ring construction** — Compositions at increasing radii (Ring 1-4 tokens)
4. **Entropy features** — Shannon entropy captures niche diversity
5. **Token assignment** — Compositions grouped into the 9-token structure for Layer B

## What Goes In

- HLCA-labeled snRNA-seq AnnData
- Spatial AnnData with spot coordinates and expression
- Gene marker lists for mapping

## What Comes Out

- Spatial AnnData with composition scores in `.obsm`
- Niche token features (parquet or stored in AnnData)
- Mapping quality report (JSON)

## Key Design Decisions

- **Multiple backends** — Not locked to one method; benchmark determines choice
- **Common contract** — All methods produce the same output format
- **Preprocessing, not model** — Spatial mapping is feature extraction, not the scientific model
- **Quality diagnostics** — Mapping quality is monitored and reported

## Relationship to Other Layers

- **Upstream:** Layer A (reference mapping) provides cell-type labels for snRNA-seq
- **Downstream:** Layer B consumes niche tokens (ring compositions)
