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

The healthy lung reference (~500K cells):

1. **Atlas loading** — HLCA reference h5ad with pretrained scVI/scArches model
2. **Query alignment** — Gene set surgery to match reference
3. **Embedding** — Project query cells into HLCA latent manifold
4. **Similarity profile** — 13-dimensional cosine similarity vector against HLCA cell-type centroids

Output: `hlca_features (13,)` per cell/niche — similarity to healthy lung cell types.

### LuCA (Lung Cancer Atlas)

The tumor reference for cancer-specific context:

1. **Atlas loading** — LuCA reference covering tumor microenvironment
2. **Embedding** — Same scArches workflow as HLCA
3. **Similarity profile** — 15-dimensional cosine similarity vector against LuCA cell-type centroids

Output: `luca_features (15,)` per cell/niche — similarity to cancer-associated cell types.

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
- HLCA reference with pretrained model
- LuCA reference with pretrained model

## What Comes Out

- Per-cell cosine similarity vectors: `hlca_features (13,)`, `luca_features (15,)`
- Cell-type label transfer table
- Integration quality diagnostics
- Labels feed receiver state IDs in Layer B

## Quality Diagnostics

Integration quality must be verified:
- Gene overlap statistics
- UMAP visualization of query in reference space
- Label transfer confidence distribution
- Batch effect assessment

## Relationship to Other Layers

- **Upstream:** Step 0 data pipeline provides merged AnnData
- **Downstream:** Layer B receives HLCA/LuCA features as tokens; Layer D operates in this latent space
