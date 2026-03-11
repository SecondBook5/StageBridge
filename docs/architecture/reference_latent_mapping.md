# Architecture: Reference Latent Mapping

**Scientific layer:** 2 — Reference latent mapping
**Package location:** `stagebridge/reference/`

## Role in the System

Reference latent mapping produces the atlas-derived features that anchor the EA-MIST context model. Two independent atlases provide complementary perspectives on each cell's identity — one from healthy tissue, one from tumor — and the cosine similarity profiles against their reference cell types become the HLCA and LuCA feature vectors consumed by the local niche encoder.

## Atlases

### HLCA (Human Lung Cell Atlas)

The healthy lung reference (~500K cells across human lung cell types):

1. **Atlas loading** — Full HLCA reference h5ad provides the healthy latent space
2. **scArches model surgery** — Aligns the query gene set to the reference model
3. **Query training** — Fine-tunes on query data to embed cells into the reference manifold
4. **Cosine similarity profile** — Each query cell gets a **13-dimensional** vector of cosine similarities against HLCA reference cell-type centroids

Output: `hlca_features (13,)` per neighborhood — measures how similar each niche's cellular composition is to healthy lung cell types.

### LuCA (Lung Cancer Atlas)

The tumor reference provides cancer-specific cell-type context:

1. **Atlas loading** — LuCA reference covering lung tumor microenvironment cell types
2. **Embedding and label transfer** — Same scArches workflow as HLCA, targeting cancer cell types
3. **Cosine similarity profile** — Each query cell gets a **15-dimensional** vector of cosine similarities against LuCA reference cell-type centroids

Output: `luca_features (15,)` per neighborhood — measures how similar each niche's composition is to cancer-associated cell types.

### Design Rationale: Two Atlases

Using both HLCA and LuCA captures a critical biological asymmetry:

- **HLCA** anchors normal tissue identity (alveolar, stromal, immune populations)
- **LuCA** captures tumor-associated and transitional states (cancer epithelial, tumor-associated macrophages, cancer-associated fibroblasts)

A niche in the early stages (Normal, AAH) should have high HLCA similarity and low LuCA similarity; an invasive LUAD niche should show the reverse. The **atlas contrast token** (optional, enabled by `hlca_luca_contrast` mode) explicitly captures this divergence.

## Atlas Ablation

The evaluation framework systematically tests the contribution of each atlas:

| Mode | HLCA | LuCA | Contrast | Scientific question |
|------|------|------|----------|-------------------|
| `no_atlas` | Zeroed | Zeroed | No | Can spatial structure alone predict stage? |
| `hlca_only` | Active | Zeroed | No | Does healthy reference suffice? |
| `luca_only` | Zeroed | Active | No | Does cancer reference suffice? |
| `hlca_luca` | Active | Active | No | Do both atlases together help? |
| `hlca_luca_contrast` | Active | Active | Yes | Does explicit cross-atlas modeling add lift? |

Performance drop from `hlca_luca` to `no_atlas` quantifies how much atlas features contribute beyond raw spatial composition. The contrast mode tests whether the *relationship* between healthy and cancer features provides additional discriminative power.

## What Goes In

- snRNA-seq AnnData with raw counts and gene names
- HLCA reference atlas with pretrained model
- LuCA reference atlas with pretrained model

## What Comes Out

- Per-neighborhood cosine similarity vectors: `hlca_features (13,)`, `luca_features (15,)`
- Cell-type label transfer table (parquet)
- Diagnostic reports (gene overlap, integration quality, label confidence)
- Transferred labels feed receiver state IDs in the local niche encoder

## Key Design Decisions

- **Two atlases, not one** — Healthy and cancer references provide orthogonal biological information
- **Cosine similarity, not raw embeddings** — Compact interpretable profiles rather than high-dimensional latent vectors
- **Diagnose, don't assume** — Integration quality diagnostics are mandatory
- **Ablation-ready** — Atlas features can be zeroed at the model level to test contribution

## Relationship to Other Layers

- **Upstream:** Data ingestion provides the AnnData
- **Downstream:** Local niche encoder receives HLCA/LuCA features as typed tokens; atlas ablation grid tests each combination
