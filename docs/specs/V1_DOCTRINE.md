# StageBridge V1 Doctrine and Scope

This document defines the non-negotiable principles and scope boundaries for StageBridge V1.

---

## Part 1: Project Doctrine

### Core Identity

**StageBridge is a cell-level representation learning framework for modeling disease progression from cross-sectional spatial and single-cell transcriptomics data.**

It is NOT:
- A lesion classifier
- A bag-level model that happens to use cells
- A generic transformer architecture
- A communication inference framework

### The Scientific Hierarchy

```
CELLS          <- Primary scientific unit (learning happens here)
    |
LOCAL NICHES   <- Essential context (receiver-centered neighborhoods)
    |
BAGS/LESIONS   <- Computational containers (aggregation, not science)
    |
STAGE SAMPLES  <- Grouping for transition modeling
    |
TRANSITIONS    <- Downstream objective (Normal -> AAH -> AIS -> MIA -> IA)
```

### Non-Negotiable Principles

#### 1. Representation Learning First
The primary contribution is learning cell representations that:
- Capture disease-relevant variation
- Encode neighborhood context
- Support transition prediction
- Transfer across datasets

Classification accuracy is a downstream metric, not the goal.

#### 2. Cells Are the Scientific Unit
Every architectural choice must be justified in terms of cell-level learning:
- What does this teach us about cells?
- How does this improve cell representations?
- Does this preserve cell-level interpretability?

#### 3. Bags Are Containers, Not Science
Lesions/bags/stage samples exist for computational efficiency and hierarchical aggregation. They do NOT exist as the primary prediction target or the unit of scientific interpretation.

#### 4. Dual-Reference Geometry
Cell representations are anchored by:
- **HLCA** (Healthy Lung Cell Atlas) - normal reference
- **LuCA** (Lung Cancer Atlas) - disease reference

#### 5. Receiver-Centered Neighborhoods
The local niche encoder must be:
- **Receiver-centered**: model from the perspective of a focal cell
- **Distance-aware**: explicit spatial attention
- **Sparse**: regularized attention weights
- **Interpretable**: neighbor ablation possible

#### 6. Progression as Downstream Objective
The ultimate goal is modeling: Normal -> AAH -> AIS -> MIA -> Invasive Adenocarcinoma

---

## Part 2: V1 Scope Definition

### V1 Definition of Done

V1 is complete when:
1. [x] Notebook runs end-to-end on real LUAD-Evo data
2. [x] Baselines trained and evaluated - all 7 required baselines
3. [x] Full model beats baselines - statistically significant improvement
4. [x] Ablations justify components - each module contributes
5. [x] Biology validated - marker genes, pathways make sense
6. [ ] Figures publication-ready - camera-ready quality (in progress)
7. [x] Results reproducible - clean checkout -> same results

### V1 Technical Choices

| Decision | V1 Choice | Rationale |
|----------|-----------|-----------|
| Geometry | Euclidean | Simpler, sufficient for V1 |
| Transition model | Flow matching | Stable, well-understood |
| Spatial backends | Tangram, DestVI, TACCO, Cell2location | Established methods |
| Reference fusion | Concat + learned weights | Simple, effective |
| Niche encoder | Receiver-centered attention | Per doctrine |

### V1 Baselines (Required)

All 7 must be implemented and evaluated:
1. Mean Pool + MLP - weakest floor
2. Max Pool + MLP - extreme-feature baseline
3. DeepSets - set invariance only
4. Flat Set Transformer - attention without hierarchy
5. Hierarchical Set Transformer (no influence) - hierarchy without niche
6. GraphSAGE - graph aggregation baseline
7. GAT or Graph-of-Sets - attention-based graph

### V1 Exclusions (Explicit)

These are NOT in V1 scope:
- Non-Euclidean geometry (hyperbolic, spherical)
- Multi-dataset training
- Real-time inference API
- Phase portraits
- Hypergraph structures
- Cohort-level transport
- Destination conditioning

---

## Part 3: Scope Decisions

### Scope Decision Protocol

```
Is it on the critical path?
    |
    +-- YES -> Proceed
    |
    +-- NO -> Is V1 blocked without it?
                |
                +-- YES -> Proceed (note as expedient)
                |
                +-- NO -> Defer to V2
```

### Drift Detection

Work has drifted if it:
1. Optimizes lesion classification accuracy as the primary metric
2. Treats cells as interchangeable elements of a bag
3. Ignores the dual-reference structure
4. Implements neighborhoods without receiver-centering
5. Adds v2 features to v1 scope
6. Cannot be justified in representation learning terms

### V2 Parking Lot

Ideas deferred to V2 are tracked in `docs/V2_IDEAS.md`.

---

## Document Maintenance

This document is maintained by the `research-director` agent.

Last updated: 2026-03-28
