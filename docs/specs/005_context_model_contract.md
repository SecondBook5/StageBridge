# 005 — Context Model Contract

Context modeling code lives under `stagebridge/context_model/`. The active layer split is: token schema and building, cell-to-spot assignment, set encoding, graph building and graph-of-sets integration.

## Purpose

The context model encodes the local tissue microenvironment into a conditioning vector for the transition model. Given the cellular composition and spatial structure of the niche around a cell, what context should influence its transition dynamics?

## Typed Biological Sets

Each (patient, stage) combination defines a biological set. Within each set, cells are represented as typed tokens:

| Token Type | Biological Meaning |
|------------|-------------------|
| Epithelial | Alveolar and tumor epithelial cells undergoing progression |
| Stromal | Fibroblasts, myofibroblasts — structural microenvironment |
| Immune | T cells, macrophages, B cells, mast cells — immune microenvironment |
| Vascular/Program | Endothelial cells, pericytes, expression programs — vascular context |

Token types are derived from HLCA cell-type labels grouped into broad lineages.

## Set Transformer (Intra-Set Encoding)

Compresses each biological set into K summary tokens capturing cell-type composition, transcriptional state variation, and niche composition.

Architecture:
1. ISAB (Induced Set Attention Block) — Efficient attention via inducing points
2. SAB (Set Attention Block) — Full self-attention refinement
3. PMA (Pooling by Multihead Attention) — Compress to K summary tokens

Permutation-invariant. Fixed-size output regardless of set cardinality.

## Graph-of-Sets Transformer (Inter-Set Encoding)

Adds inter-set communication via sparse graph attention across nodes.

### Graph Structure

Nodes: Each (patient, stage) combination.

| Edge Type | Meaning |
|-----------|---------|
| 0 — Stage-adjacent | Consecutive stages (e.g., AAH to AIS) |
| 1 — Same-patient cross-stage | Same patient across different stages |
| 2 — Same-stage cross-patient | Different patients at same stage |

Each edge type has a learned attention bias.

### Architecture

- GraphAttentionLayer — Sparse multi-head attention with per-edge-type bias
- GraphTransformerBlock — Graph attention + FFN + residual + LayerNorm
- GraphOfSetsTransformer — L blocks (default L=2)

## Why Set-Only Is the First Serious Baseline

Set-only already encodes spatial niche information via typed tokens from Tangram. It captures within-set composition and state variation. It does not assume inter-set communication helps. It is simpler, faster, and easier to interpret.

## Why Graph-of-Sets Must Earn Its Place

GoST adds model complexity. Adopt only if ablation experiments show:
1. Inter-set attention meaningfully improves transition prediction on held-out data
2. Learned edge-type biases reveal interpretable structure
3. Improvement is not explained by simply having more parameters
