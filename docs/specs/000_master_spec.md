# 000 Master Spec

StageBridge v1 asks:

**Which within-lung LUAD initiation stage transitions are niche-gated, and how is that gating modulated by evolutionary state?**

Active v1 ladder:

- Normal
- AAH
- AIS
- MIA
- LUAD

Active v1 layers:

1. data ingestion
2. reference latent mapping
3. spatial mapping
4. typed niche context modeling
5. edge-wise stochastic transition modeling
6. tissue-level interpretation and evaluation
7. results tracking

## Method Definition

StageBridge is a reference-anchored, spatially grounded, edge-wise stochastic transition framework that models within-lung LUAD initiation as a series of niche-gated drift-diffusion processes across histologically defined disease stages, conditioned on typed tissue microenvironment context and regularized by whole-exome evolutionary state.

## Active Modalities

| Modality | GEO Accession | Role |
|----------|---------------|------|
| snRNA-seq | GSE308103 | Cell-level transcriptomes, primary input |
| 10x Visium | GSE307534 | Spatial tissue architecture, niche definition |
| WES | GSE307529 | Evolutionary state, transition regularization |

## Active Exclusions

- No continuous Normal-to-BrainMets progression claim in v1
- No TCR conditioning
- No brain metastasis as part of the first complete system
- No claim of zero batch effects
- No assumption that graph-of-sets outperforms set-only
- No unrestricted learned genomics conditioning

## Disease Edges

| Edge | From | To |
|------|------|----|
| 0 | Normal | AAH |
| 1 | AAH | AIS |
| 2 | AIS | MIA |
| 3 | MIA | LUAD |

## Three Execution Modes

| Mode | Spatial | Graph | WES | Purpose |
|------|---------|-------|-----|---------|
| RNA-only | No | No | No | Minimal baseline |
| Set-only | Yes | No | No | First serious spatial baseline |
| Graph-of-Sets + WES | Yes | Yes | Yes | Full model, must earn place via ablation |

## Evaluation Philosophy

Evaluation is not post-hoc. The evaluation layer assesses whether transitions are calibrated, whether spatial context changes behavior, whether graph attention adds value, whether WES regularization constrains transport, and what the model reveals about tissue dynamics.

## Result-Tracking Philosophy

Every run saves resolved configuration, metrics, and a result card tied to a git commit. Milestones receive git tags. Git history is the archive.

## First Biological Focus

1. **AAH to AIS** — Spatial tissue reorganization expected to be most informative
2. **AIS to MIA** — Niche composition and invasive potential most directly testable
