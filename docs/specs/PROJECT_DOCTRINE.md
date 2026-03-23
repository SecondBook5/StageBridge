# StageBridge Project Doctrine

This document defines the non-negotiable scientific and architectural principles of StageBridge. All agents and contributors must align with this doctrine.

## Core Identity

**StageBridge is a cell-level representation learning framework for modeling disease progression from cross-sectional spatial and single-cell transcriptomics data.**

It is NOT:
- A lesion classifier
- A bag-level model that happens to use cells
- A generic transformer architecture
- A communication inference framework

## The Scientific Hierarchy

```
CELLS          ← Primary scientific unit (learning happens here)
    ↓
LOCAL NICHES   ← Essential context (receiver-centered neighborhoods)
    ↓
BAGS/LESIONS   ← Computational containers (aggregation, not science)
    ↓
STAGE SAMPLES  ← Grouping for transition modeling
    ↓
TRANSITIONS    ← Downstream objective (Normal → AAH → AIS → MIA → IA)
```

## Non-Negotiable Principles

### 1. Representation Learning First

The primary contribution is learning cell representations that:
- Capture disease-relevant variation
- Encode neighborhood context
- Support transition prediction
- Transfer across datasets

Classification accuracy is a downstream metric, not the goal.

### 2. Cells Are the Scientific Unit

Every architectural choice must be justified in terms of cell-level learning:
- What does this teach us about cells?
- How does this improve cell representations?
- Does this preserve cell-level interpretability?

### 3. Bags Are Containers, Not Science

Lesions/bags/stage samples exist for:
- Computational efficiency (batching)
- Hierarchical aggregation (set pooling)
- Transition edge definition (source → target)

They do NOT exist as:
- The primary prediction target
- The unit of scientific interpretation
- A replacement for cell-level analysis

### 4. Dual-Reference Geometry

Cell representations are anchored by:
- **HLCA** (Healthy Lung Cell Atlas) - normal reference
- **LuCA** (Lung Cancer Atlas) - disease reference

This dual-reference structure:
- Provides biological grounding
- Enables interpretable embeddings
- Supports transfer learning

### 5. Receiver-Centered Neighborhoods

The local niche encoder must be:
- **Receiver-centered**: model from the perspective of a focal cell
- **Distance-aware**: explicit spatial attention
- **Sparse**: regularized attention weights
- **Interpretable**: neighbor ablation possible

NOT acceptable:
- Vague "context pooling"
- Symmetric message passing without receiver focus
- Dense attention without regularization

### 6. Progression as Downstream Objective

The ultimate goal is modeling:
```
Normal → AAH → AIS → MIA → Invasive Adenocarcinoma
```

This means:
- Learning transition dynamics, not static classification
- Capturing what changes between stages
- Predicting plausible next states

## Scope Boundaries

### V1 Scope (Current)

- Euclidean geometry
- Flow matching for transitions
- 7 baseline architectures
- Single dataset (LUAD-Evo)
- Publication-ready notebook

### V2 Scope (Deferred)

- Non-Euclidean geometry (hyperbolic/spherical)
- Additional stochastic backends
- Multi-dataset generalization
- Real-time inference API

### Out of Scope

- Phase portraits
- Hypergraph structures
- Cohort-level transport
- Destination conditioning (without explicit approval)

## Drift Detection

Work has drifted if it:
1. Optimizes lesion classification accuracy as the primary metric
2. Treats cells as interchangeable elements of a bag
3. Ignores the dual-reference structure
4. Implements neighborhoods without receiver-centering
5. Adds v2 features to v1 scope
6. Cannot be justified in representation learning terms

## Document Maintenance

This doctrine is maintained by the `research-director` agent.

Changes require:
1. Explicit discussion of what's changing and why
2. Assessment of impact on existing work
3. Update to all affected specification documents
