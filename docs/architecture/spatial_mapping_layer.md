# Architecture: Spatial Mapping Layer

**Scientific layer:** 3 — Spatial mapping
**Package location:** `stagebridge/spatial_mapping/`

## Role in the System

Spatial mapping connects single-cell identities to physical tissue locations. It answers: for each Visium spot, what cell types are present and in what proportions? These compositions define the typed niches that the context model encodes.

## How It Works

### Tangram (Primary)

Tangram optimizes a mapping matrix M between N cells and S spots by maximizing the cosine similarity of mapped gene expression profiles.

- Input: snRNA-seq AnnData (with cell-type labels), spatial AnnData (with spot coordinates and expression)
- Optimization: gradient descent on mapping matrix, guided by marker genes
- Output: S x C matrix of cell-type probability scores per spot (C = number of cell types)

### TACCO / DestVI (Alternatives)

Same conceptual output (spot-level composition scores) via different methods. Share the common output contract so downstream code is agnostic to which method produced the scores.

## From Spatial Scores to Niche Tokens

1. **Composition vector** — Per-spot probability distribution over cell types
2. **Neighborhood aggregation** — k-nearest spatial neighbors' compositions are averaged to capture the local tissue context beyond a single spot
3. **Entropy features** — Shannon entropy of the composition captures niche diversity
4. **Typed token assignment** — Composition entries are grouped into broad lineages (epithelial, stromal, immune, vascular) to create the typed tokens consumed by the context model

## What Goes In

- HLCA-labeled snRNA-seq AnnData
- Spatial AnnData with spot coordinates

## What Comes Out

- Spatial AnnData with composition scores in `.obsm`
- Niche token features (parquet)
- Mapping report (JSON)

## Key Design Decisions

- **Tangram first** — Well-established, interpretable, no generative model required
- **Common contract** — All methods produce the same output format
- **Preprocessing, not model** — Spatial mapping is a feature extraction step, not the scientific model

## Relationship to Other Layers

- **Upstream:** Reference mapping provides cell-type labels; data ingestion provides spatial AnnData
- **Downstream:** Context model consumes niche tokens as typed biological sets
