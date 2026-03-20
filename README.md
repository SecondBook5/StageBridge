<p align="center">
  <h1 align="center">StageBridge</h1>
  <p align="center">
    <strong>Stochastic transition modeling for cell-state progression<br>in spatial and single-cell omics</strong>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11+"></a>
    <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch 2.2+"></a>
    <a href="https://github.com/SecondBook5/StageBridge/actions"><img src="https://img.shields.io/github/actions/workflow/status/SecondBook5/StageBridge/ci.yml?label=CI&logo=github" alt="CI"></a>
  </p>
</p>

---

## Overview

StageBridge is a **method for learning cell-state transitions under spatial and multimodal constraints**. The framework models progression at the **cell and niche level**, not as patient classification.

The primary application is lung adenocarcinoma (LUAD) progression:

```
Normal  ──>  AAH  ──>  AIS  ──>  MIA  ──>  LUAD
```

The framework integrates three data modalities—10x Visium spatial transcriptomics, snRNA-seq, and whole-exome sequencing—to learn how cells transition between states, conditioned on their local microenvironment (niche) and constrained by evolutionary compatibility.

### Core principles

- **Cell-level learning**: The scientific object is cell-state transition, not patient classification
- **Niche conditioning**: Transitions depend on local neighborhood context
- **Dual-reference geometry**: Cells are embedded relative to healthy (HLCA) and tumor (LuCA) atlases using model-based scArches surgery
- **Evolutionary constraints**: WES-derived features enforce biologically plausible transitions
- **Spatial backend agnostic**: Benchmarked across Tangram, TACCO, and DestVI

---

## Architecture

StageBridge uses a layered architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         StageBridge V1 Pipeline                             │
│                                                                             │
│  ┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐           │
│  │   Layer A   │   │     Layer B      │   │      Layer C       │           │
│  │  Dual-Ref   │──>│  Local Niche     │──>│  Set Transformer   │           │
│  │   Latent    │   │  Encoder (9-tok) │   │  (ISAB/SAB/PMA)    │           │
│  └─────────────┘   └──────────────────┘   └────────────────────┘           │
│        │                                            │                       │
│        v                                            v                       │
│  ┌─────────────┐                          ┌────────────────────┐           │
│  │ HLCA + LuCA │                          │     Layer D        │           │
│  │  Reference  │                          │  Flow Matching     │           │
│  │  Alignment  │                          │  (OT-CFM)          │           │
│  └─────────────┘                          └────────────────────┘           │
│                                                     │                       │
│                    WES Features ───────────────────>│                       │
│                    (Evolutionary Constraint)        v                       │
│                                           ┌────────────────────┐           │
│                                           │  Cell Transition   │           │
│                                           │  Trajectories      │           │
│                                           └────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Local niche encoding (Layer B)

Each spatial niche is encoded as a **9-token sequence**:

| Token | Source | Description |
|-------|--------|-------------|
| Receiver | Cell identity | Target cell expression + learned state embedding |
| Ring 1–4 | Spatial neighborhood | Cell-type composition at increasing radii |
| HLCA | Reference atlas | Similarity to healthy lung cell types |
| LuCA | Tumor atlas | Similarity to tumor-aware cell states |
| Pathway | Gene programs | Ligand-receptor and pathway activity summary |
| Stats | Neighborhood | Local density, entropy, and composition statistics |

### Stochastic transition model (Layer D)

V1 uses **Flow Matching** (OT-CFM) with Sinkhorn coupling:
- Learns continuous trajectories between cell states
- Optimal transport provides principled coupling
- Niche context conditions the flow field

---

## Project scope

### V1-Minimal (Current)

The first publication scope:

| Component | Status | Description |
|-----------|--------|-------------|
| Raw Data Pipeline | Complete | `stagebridge data-prep` orchestration |
| Spatial Backend Benchmark | In progress | Tangram/DestVI/TACCO comparison |
| Dual-Reference Latent | In progress | HLCA + LuCA alignment |
| Local Niche Encoder | Complete | 9-token transformer (from EA-MIST) |
| Set Transformer | Complete | ISAB/SAB/PMA hierarchy (from EA-MIST) |
| Flow Matching | In progress | OT-CFM with Sinkhorn coupling |
| Evolutionary Compatibility | Complete | WES-derived constraints |
| Donor-Held-Out Evaluation | Planned | With uncertainty quantification |

### V2/V3 Roadmap (Deferred)

- Non-Euclidean geometry (hyperbolic/spherical latents)
- Neural SDE backend
- Phase portrait / attractor decoder
- Cohort transport layer
- Destination-conditioned transitions (brain metastasis)

See [AGENTS.md](AGENTS.md) for detailed implementation plans.

---

## Data

StageBridge integrates multi-modal data from public GEO repositories:

| Dataset | Modality | GEO Accession | Role |
|---------|----------|---------------|------|
| Early LUAD snRNA-seq | Single-cell transcriptomics | [GSE308103](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308103) | Cell-level expression |
| 10x Visium | Spatial transcriptomics | [GSE307534](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307534) | Tissue architecture |
| Whole-exome sequencing | WES | [GSE307529](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307529) | Evolutionary features |

**Reference atlases:**
- [Human Lung Cell Atlas (HLCA)](https://doi.org/10.1038/s41591-023-02327-2) — healthy reference anchor
- [LuCA extended atlas](https://www.cell.com/cancer-cell/fulltext/S1535-6108(22)00499-8) — tumor-aware cell state reference

**Spatial mapping backends:**
- [Tangram](https://www.nature.com/articles/s41592-021-01264-7) — deep learning-based spatial mapping
- [TACCO](https://www.nature.com/articles/s41587-023-01657-3) — optimal transport-based annotation transfer
- [DestVI](https://www.nature.com/articles/s41587-022-01272-8) — variational inference deconvolution

---

## Installation

```bash
# Clone the repository
git clone https://github.com/SecondBook5/StageBridge.git
cd StageBridge

# Create conda environment
micromamba env create -f environment.yml
micromamba activate stagebridge

# Install in development mode
pip install -e ".[all]"

# Set data root (external data directory)
export STAGEBRIDGE_DATA_ROOT=/path/to/your/data
```

**Requirements:** Python 3.11+, PyTorch 2.2+, CUDA 12.4 (cu124 recommended for HPC compatibility)

---

## Quick start

### Step 0: Data preparation

Download raw data from GEO and run the data preparation pipeline:

```bash
# Set data root
export STAGEBRIDGE_DATA_ROOT=/path/to/your/data

# Run data preparation (extracts, merges, QC filters)
stagebridge data-prep
```

This creates:
- `processed/luad_evo/snrna_merged.h5ad` — merged snRNA-seq (798k cells × 18k genes)
- `processed/luad_evo/spatial_merged.h5ad` — merged Visium spatial
- `processed/luad_evo/wes_features.parquet` — WES-derived features
- `processed/luad_evo/data_prep_audit.json` — processing audit report

### Python API

```python
from stagebridge.notebook_api import compose_config, run_data_prep

# Data preparation
result = run_data_prep()

# Configure training (coming soon)
cfg = compose_config(overrides=["model=flow_matching"])
```

### Command line

```bash
# Data preparation
stagebridge data-prep --data-root /path/to/data

# With options
stagebridge data-prep --skip-qc --skip-normalization
```

---

## Repository structure

```
stagebridge/
├── context_model/          # Niche encoding and set transformers
│   ├── local_niche_encoder.py       # 9-token niche transformer (Layer B)
│   ├── set_encoder.py               # ISAB, SAB, PMA (Layer C)
│   ├── lesion_set_transformer.py    # Hierarchical aggregation
│   └── prototype_bottleneck.py      # Optional compression
├── transition_model/       # Stochastic dynamics (Layer D)
│   ├── flow_matching.py             # OT-CFM implementation
│   ├── stochastic_dynamics.py       # Neural SDE (V2)
│   └── schrodinger_bridge.py        # Sinkhorn coupling
├── data/                   # Data loading and preprocessing
│   └── luad_evo/                    # LUAD progression datasets
├── pipelines/              # End-to-end workflow orchestration
│   └── run_data_prep.py             # Step 0 data pipeline
├── reference/              # HLCA/LuCA atlas alignment
├── spatial_mapping/        # Tangram, TACCO, DestVI backends
├── evaluation/             # Metrics and ablations
└── viz/                    # Publication figures

configs/                    # Hydra YAML configuration
tests/                      # Test suite
docs/                       # Documentation
```

---

## Testing

```bash
# Full test suite
pytest tests/

# Data pipeline tests
pytest tests/test_data_prep.py

# Model tests
pytest tests/test_eamist_model.py
pytest tests/test_flow_matching.py
```

---

## Citation

If you use StageBridge in your research, please cite:

```bibtex
@software{book2026stagebridge,
  author = {Book, AJ},
  title = {StageBridge: Stochastic transition modeling for cell-state progression},
  year = {2026},
  url = {https://github.com/SecondBook5/StageBridge}
}
```

## License

[MIT](LICENSE)
