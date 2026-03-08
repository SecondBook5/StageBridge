# StageBridge

A reference-anchored, spatially grounded, edge-wise stochastic transition framework for modeling lung adenocarcinoma initiation.

## The v1 Problem

StageBridge models within-lung LUAD initiation as a sequence of niche-gated stochastic transitions across five histologically defined stages:

**Normal → AAH → AIS → MIA → LUAD**

The core biological question: **which stage transitions are gated by local tissue microenvironment composition, and how is that gating modulated by the evolutionary (genomic) state of the tumor?**

The first analytical focus is the AAH→AIS and AIS→MIA transitions, where spatial tissue architecture is most likely to change transition dynamics.

## Scope

### In scope (v1)

- Within-lung progression: Normal through invasive LUAD
- Three data modalities: snRNA-seq (GSE308103), 10x Visium spatial (GSE307534), WES (GSE307529)
- Reference latent alignment via HLCA (Human Lung Cell Atlas)
- Spatial mapping via Tangram (primary), with TACCO and DestVI as alternatives
- Typed niche context modeling via Set Transformer (baseline) and Graph-of-Sets Transformer (ablation candidate)
- Edge-wise stochastic transition modeling with drift-diffusion dynamics and Schrodinger bridge objective
- WES features as a regularizer on admissible transport
- Formal evaluation including ablations, calibration, context sensitivity, and tissue-level interpretation

### Out of scope (v1)

- Continuous Normal→BrainMets progression claim
- TCR conditioning
- Brain metastasis as part of the primary system (reserved for future extension)
- Claims of zero batch effects
- Assumption that graph-of-sets automatically outperforms set-only
- Unrestricted learned genomics conditioning

## Scientific Architecture

StageBridge is organized into seven scientific layers:

1. **Data Ingestion** — Parse GEO deposits into standardized AnnData with canonical stage labels, donor identity, and quality metadata.

2. **Reference Latent Mapping** — Project all cells into HLCA latent space via scArches surgery. This provides a shared coordinate system across datasets. Integration quality is diagnosed, not assumed.

3. **Spatial Mapping** — Map snRNA-seq profiles onto Visium spots using Tangram. Produces spot-level cell-type composition scores that define the local tissue neighborhood.

4. **Typed Niche Context Modeling** — Encode each (patient, stage) cell population as a biological set with typed tokens (epithelial, stromal, immune, vascular/program). The Set Transformer compresses these into summary representations. The Graph Transformer (optional) exchanges information across stage-adjacent and cross-patient sets.

5. **Edge-Wise Stochastic Transition Modeling** — Learn a conditional drift-diffusion process for each disease edge (e.g., AAH→AIS), initialized from Gaussian Schrodinger bridge priors and optionally coupled via entropic OT. The drift network is conditioned on niche context; WES features regularize admissible transport paths.

6. **Tissue-Level Interpretation and Evaluation** — Evaluate not just held-out metrics but also calibration, ablations, context sensitivity (niche shuffling), trajectory structure, fixed points, niche regimes, and gene/program attribution. Tissue-level dynamical interpretation is part of the scientific contribution.

7. **Results Tracking** — Every run produces a structured result card tied to a git commit. Important results are promoted to milestones with git tags.

## Repo Layout

```
StageBridge/
├── StageBridge.ipynb          # Single notebook front-end (orchestration only)
├── stagebridge/               # Python package
│   ├── data/                  # Data ingestion and loading
│   ├── reference/             # HLCA latent mapping
│   ├── spatial_mapping/       # Tangram / TACCO / DestVI
│   ├── context_model/         # Set Transformer + Graph-of-Sets Transformer
│   ├── transition_model/      # Drift-diffusion, Schrodinger bridge, OT coupling
│   ├── evaluation/            # Metrics, ablations, interpretation
│   ├── results/               # Run tracking, result cards, promotion
│   ├── pipelines/             # Pipeline entry points
│   ├── viz/                   # Visualization
│   └── utils/                 # Shared utilities
├── configs/                   # Hydra configuration tree
├── tests/                     # Test suite
├── docs/                      # Specs, decisions, biology, architecture
├── outputs/                   # Transient scratch artifacts only
└── results/                   # Formal results registry and run tree
```

Every concept has one home in the package. The notebook calls into `stagebridge/pipelines/`; it does not define model internals.

## Execution Modes

StageBridge supports three execution modes of increasing complexity:

### 1. RNA-only
Uses snRNA-seq data alone. Cells are embedded in HLCA latent space. The transition model operates without spatial context. This is the minimal viable configuration.

### 2. Set-only (spatial baseline)
Adds spatial mapping (Tangram). Each (patient, stage) population becomes a typed biological set with niche tokens derived from spatial composition. The Set Transformer encodes context. This is the first serious spatially-informed baseline.

### 3. Graph-of-Sets + WES
Adds inter-set graph attention and WES regularization. The Graph-of-Sets Transformer exchanges information across stage-adjacent and cross-patient nodes. WES features constrain which transport paths are genomically plausible. This mode must earn its place through ablation — it is not assumed to be superior.

## The Notebook

`StageBridge.ipynb` is the single user-facing front-end. It:
- Loads configuration
- Calls pipeline functions from `stagebridge/pipelines/`
- Displays diagnostics and outputs
- Writes structured results
- Promotes milestones

It does **not** define model architectures, training loops, or core logic inline.

## Results

Every run writes a structured result directory containing:
- Resolved configuration
- Metrics and evaluation outputs
- Ablation comparisons (when applicable)
- A result card summarizing the run

Milestone-worthy results are promoted with git tags and registry entries. Git history is the archive — there is no `legacy/` folder.

## Documentation

- [Master Spec](docs/specs/000_master_spec.md) — Central method specification
- [Repo Contract](docs/specs/001_repo_contract.md) — What belongs where
- [Data Contract](docs/specs/002_data_contract.md) — Dataset expectations
- Spec files `003`–`009` cover reference, spatial mapping, context model, transition model, evaluation, results, and notebook contracts
- [Architecture Decision Records](docs/decisions/) — Key design decisions
- [Biology docs](docs/biology/) — Scientific context and hypotheses
- [Architecture docs](docs/architecture/) — Layer-by-layer technical descriptions
- [Citation Map](docs/papers/citation_map.md) — Paper-to-component mapping

## Setup

```bash
micromamba create -f environment.yml
micromamba activate stagebridge
pip install -e .
```

Set the external data root:
```bash
export STAGEBRIDGE_DATA_ROOT=/mnt/e/StageBridge_data
```

Run tests:
```bash
pytest -q
```

## Data

External data lives outside the repo at `$STAGEBRIDGE_DATA_ROOT`. The three primary GEO deposits are:
- **GSE308103** — snRNA-seq (custom dense format)
- **GSE307534** — 10x Visium spatial transcriptomics
- **GSE307529** — Whole-exome sequencing

The HLCA full reference atlas (~20 GB) is required for latent alignment.
