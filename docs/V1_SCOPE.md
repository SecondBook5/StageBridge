# StageBridge V1 Scope Definition

This document defines what is in and out of scope for V1. The goal is a **publishable paper with reproducible results**.

## V1 Definition of Done

V1 is complete when:

1. [x] **Notebook runs end-to-end** on real LUAD-Evo data
2. [x] **Baselines trained and evaluated** - all 7 required baselines
3. [x] **Full model beats baselines** - statistically significant improvement
4. [x] **Ablations justify components** - each module contributes
5. [x] **Biology validated** - marker genes, pathways make sense
6. [ ] **Figures publication-ready** - camera-ready quality (in progress)
7. [x] **Results reproducible** - clean checkout → same results

## V1 Critical Path

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Data Pipeline        │ Load, QC, export real data            │
├─────────────────────────────────────────────────────────────────┤
│ 2. Spatial Backend      │ Benchmark → select canonical          │
├─────────────────────────────────────────────────────────────────┤
│ 3. Reference Geometry   │ HLCA/LuCA dual-reference embeddings   │
├─────────────────────────────────────────────────────────────────┤
│ 4. Baselines            │ Train all 7, establish comparison     │
├─────────────────────────────────────────────────────────────────┤
│ 5. Full Model           │ Train complete StageBridge model      │
├─────────────────────────────────────────────────────────────────┤
│ 6. Ablations            │ Justify each component                │
├─────────────────────────────────────────────────────────────────┤
│ 7. Biology Validation   │ Scientific credibility                │
├─────────────────────────────────────────────────────────────────┤
│ 8. Figures              │ Publication-ready visualizations      │
├─────────────────────────────────────────────────────────────────┤
│ 9. Notebook Assembly    │ Reproducible artifact                 │
└─────────────────────────────────────────────────────────────────┘
```

## V1 Technical Choices

| Decision | V1 Choice | Rationale |
|----------|-----------|-----------|
| Geometry | Euclidean | Simpler, sufficient for V1 |
| Transition model | Flow matching | Stable, well-understood |
| Spatial backends | Tangram, DestVI, TACCO | Established methods |
| Reference fusion | Concat + learned weights | Simple, effective |
| Niche encoder | Receiver-centered attention | Per doctrine |

## V1 Baselines (Required)

All 7 must be implemented and evaluated:

1. **Mean Pool + MLP** - weakest floor
2. **Max Pool + MLP** - extreme-feature baseline
3. **DeepSets** - set invariance only
4. **Flat Set Transformer** - attention without hierarchy
5. **Hierarchical Set Transformer (no influence)** - hierarchy without niche
6. **GraphSAGE** - graph aggregation baseline
7. **GAT or Graph-of-Sets** - attention-based graph

## V1 Exclusions (Explicit)

These are NOT in V1 scope:

- Non-Euclidean geometry (hyperbolic, spherical)
- Additional spatial backends beyond the 3
- Multi-dataset training
- Real-time inference API
- Phase portraits
- Hypergraph structures
- Cohort-level transport
- Destination conditioning
- Additional baselines beyond the 7

## Scope Decision Protocol

When evaluating new work:

```
Is it on the critical path?
    │
    ├── YES → Proceed
    │
    └── NO → Is V1 blocked without it?
                │
                ├── YES → Proceed (note as expedient)
                │
                └── NO → Defer to V2
```

## V2 Parking Lot

Ideas deferred to V2 are tracked in `docs/V2_IDEAS.md`.

## Document Maintenance

This document is maintained by the `research-director` agent.

Last updated: 2026-03-22
