# 001 Repo Contract

The active Python package root is `stagebridge/`.
The active orchestration namespace is `stagebridge/pipelines/`.
The top-level notebook surface is `StageBridge.ipynb`.

## Principle

Organized by scientific concept. Every concept has one home.

## Package Layout

| Directory | Concept | Contains |
|-----------|---------|----------|
| `stagebridge/data/` | Data ingestion | GEO parsers, AnnData builders, stage normalization |
| `stagebridge/data/common/` | Shared data utilities | Schemas, stage ontology, validation |
| `stagebridge/data/luad_evo/` | LUAD evolution readers | GSE308103, GSE307534, GSE307529 |
| `stagebridge/data/brainmets/` | Brain mets readers | Reserved, not active in v1 |
| `stagebridge/reference/` | HLCA latent mapping | scArches surgery, label transfer, diagnostics |
| `stagebridge/spatial_mapping/` | Spatial deconvolution | Tangram (primary), TACCO, DestVI |
| `stagebridge/context_model/` | Niche context encoding | Set Transformer, Graph-of-Sets Transformer |
| `stagebridge/transition_model/` | Transition dynamics | Drift-diffusion, Schrodinger bridge, OT, WES regularization |
| `stagebridge/evaluation/` | Evaluation and interpretation | Metrics, ablations, context sensitivity, tissue reporting |
| `stagebridge/results/` | Run tracking | Result cards, milestone promotion, registry |
| `stagebridge/pipelines/` | Pipeline orchestration | Entry points called by notebook and CLI |
| `stagebridge/viz/` | Visualization | Spatial plots, embeddings, metrics curves |
| `stagebridge/utils/` | Shared utilities | Seeding, types, config helpers, logging |

## Context Model Substructure

- Token schema and token building
- Cell-to-spot assignment
- Set encoding (ISAB, SAB, PMA)
- Graph building and graph-of-sets integration

## One-Home Rule

Every concept has exactly one location. Do not create `misc/`, `helpers/`, `common/`, or `shared/` directories as dumping grounds outside the defined structure. If a function does not belong in one of the above locations, resolve the design question before writing it.
