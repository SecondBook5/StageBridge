# Architecture: Typed Niche Context Model

**Scientific layer:** 4 — Typed niche context modeling
**Package location:** `stagebridge/context_model/`

## Role in the System

The context model transforms raw niche information (cell-type compositions, spatial features) into a conditioning vector that tells the transition model about the tissue microenvironment. It is the mechanism by which spatial structure influences transition dynamics.

## Architecture

### Biological Sets

Each (patient, stage) combination is a biological set: a collection of cells/spots from one patient at one disease stage. Within each set, elements are typed tokens:

- **Epithelial tokens** — Alveolar type II, tumor epithelial, club cells
- **Stromal tokens** — Fibroblasts, myofibroblasts, smooth muscle
- **Immune tokens** — T cells, macrophages, B cells, mast cells, dendritic cells
- **Vascular/Program tokens** — Endothelial cells, pericytes, transcriptional programs

Type assignment comes from HLCA labels grouped into broad lineages.

### Set Transformer (Intra-Set)

Processes each biological set independently.

```
Tokens → [ISAB] → [SAB] → [PMA] → K summary tokens
```

- **ISAB** uses M inducing points for O(NM) attention instead of O(N^2)
- **SAB** refines with full self-attention
- **PMA** pools to K fixed-size summary tokens via learned seed vectors
- Output is permutation-invariant and fixed-size regardless of set cardinality

### Graph-of-Sets Transformer (Inter-Set, Optional)

Processes relationships between sets.

```
Node summaries → [GraphAttn + FFN] x L → Context-enriched summaries
```

- Nodes: (patient, stage) sets
- Edge types: stage-adjacent (type 0), same-patient cross-stage (type 1), same-stage cross-patient (type 2)
- Per-edge-type learned bias in attention
- L=2 blocks by default

### Context Vector Extraction

Summary tokens from the query node (the node being transitioned) are pooled (mean or attention-weighted) into a single context vector c that conditions the drift network.

## Two Modes

| Mode | Set Transformer | Graph Transformer | Description |
|------|----------------|-------------------|-------------|
| Set-only | Yes | No | Default spatial baseline |
| Graph-of-Sets | Yes | Yes | Ablation candidate |

## What Goes In

- Typed token features for each (patient, stage) set
- Graph structure (edge_index, edge_type) for GoST mode

## What Comes Out

- Context vector c (conditioning input for transition model)

## Relationship to Other Layers

- **Upstream:** Spatial mapping provides niche tokens; reference mapping provides cell-type labels
- **Downstream:** Transition model receives context vector as conditioning input
