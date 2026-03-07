# StageBridge TopoFlow — Master Specification

## Biological Hypothesis

Lung adenocarcinoma evolves through a stereotyped morphological progression:
**Normal alveolar epithelium → AAH → AIS → MIA → invasive LUAD**, with a
subset metastasising to brain and chest wall. Each transition involves
coordinated changes in gene expression, spatial tissue architecture, and
genomic alterations (somatic mutations, copy-number changes).

## Architecture Summary

StageBridge TopoFlow models this progression as **optimal transport flows in
HLCA latent space**, conditioned on tissue microenvironment context encoded by
a **Graph-of-Sets Transformer (GoST)**:

1. **HLCA Latent Alignment** — All single-cell and spatial data are projected
   into the Human Lung Cell Atlas latent space, eliminating batch effects and
   enabling cross-dataset comparison.

2. **Set Transformer (Intra-Set)** — Each (patient, stage) cell population is
   a *set*. ISAB + SAB + PMA compress it into K summary tokens capturing
   cell-type composition and transcriptional state.

3. **Graph Transformer (Inter-Set)** — Summary tokens from neighboring
   (patient, stage) nodes exchange information via sparse graph attention,
   producing context-enriched representations.

4. **Conditional Flow Matching** — A vector field network, conditioned on GoST
   context + stage embedding + optional genomic features, learns to transport
   cells from stage s to stage s+1 via entropic OT coupling.

5. **Schrodinger Bridge Extension** — For stochastic transitions with
   biological noise (e.g., metastasis), a Brownian bridge interpolant replaces
   the deterministic OT-CFM path.

## Two Cohorts

| Cohort | Stages | Modalities |
|--------|--------|------------|
| Peng (GSE308103/307534/307529) | Normal→AAH→AIS→MIA→LUAD | snRNA-seq, 10x Visium, WES |
| Rossi (GSE223499/501/502/500) | LUAD→BrainMet/ChestWallMet | snRNA-seq, Slide-seq V2, lpWGS, TCR-seq |

## Package Structure

- `stagebridge/` — Core models, training, IO (battle-tested, 96 tests)
- `src/stagebridge_topoflow/` — Pipeline orchestration, configs, artifacts
- `configs/topoflow/` — Plain YAML configuration
- `tests/` — All tests (original + topoflow)
