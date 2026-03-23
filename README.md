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
| Receiver | Cell identity | Target (focal) cell expression + learned state embedding |
| Ring 1-4 | Spatial neighborhood | Cell-type composition at increasing radii |
| HLCA | Reference atlas | Embedding similarity to healthy lung (HLCA) reference |
| LuCA | Tumor atlas | Embedding similarity to disease-aware (LuCA) reference |
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
| Spatial Backend Benchmark | Complete | Tangram/DestVI/TACCO/Cell2Location comparison |
| Dual-Reference Latent | Complete | HLCA + LuCA alignment via scArches surgery |
| Local Niche Encoder | Complete | Receiver-centered niche transformer |
| Set Transformer | Complete | ISAB/SAB/PMA hierarchy |
| Flow Matching | Complete | OT-CFM with Sinkhorn coupling |
| Evolutionary Compatibility | Complete | WES-derived constraints |
| Donor-Held-Out Evaluation | Complete | With uncertainty quantification |

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

# Configure training
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

## HPC Deployment (Snakemake)

StageBridge uses **Snakemake** for HPC orchestration. Do NOT use raw sbatch scripts.

### Quick Start

```bash
# Dry run (see what would execute)
snakemake -n --profile workflow/slurm

# Full run on HPC with SLURM
snakemake --profile workflow/slurm --jobs 20

# Generate DAG visualization
snakemake --dag | dot -Tpdf > dag.pdf
```

### Configuration

Edit `workflow/config.yaml` or override via command line:

```bash
snakemake --profile workflow/slurm --config data_root=/your/data/path
```

Default paths (configured for HPC):
```yaml
data_root: "/scratch/chaunzt1/stagebridge"
```

### Required Input Files

```
$DATA/
├── processed/luad_evo/
│   ├── snrna_qc_normalized_with_ensg.h5ad   # snRNA with ENSG IDs
│   ├── spatial_merged.h5ad                   # Merged Visium data
│   └── wes_features.parquet                  # WES features
└── references/
    ├── hlca/
    │   ├── hlca_reference.h5ad
    │   └── hub_cache/                        # scANVI model from HuggingFace
    └── luca/
        ├── luca_core_atlas.h5ad              # Use CORE, not Extended
        └── retrained_model/scanvi_model/
```

### Pipeline DAG

```
hlca_mapping ──┬──→ merge_cell_types ──→ validate_markers ──→ spatial_backend (4x)
               │                                                       │
               └──→ fuse_embeddings ←── luca_mapping                   │
                           │                                           │
                           └─────────────────────┬──────────────────────┘
                                                 ▼
                                        data_preparation
                                                 │
                                     ┌───────────┴───────────┐
                                     ▼                       ▼
                             semi_synthetic            validate_splits
                                     │                       │
                                     └───────────┬───────────┘
                                                 ▼
                                               hpo
                                                 │
                         ┌───────────────────────┼───────────────────────┐
                         ▼                       ▼                       ▼
             training (5×3=15)          baseline (4×5×3=60)          (wait)
                         │                       │                       │
                         ▼                       ▼                       │
                aggregate_cv_results      aggregate_baselines            │
                         │                       │                       │
                         └───────────┬───────────┴───────────────────────┘
                                     ▼
                         ┌───────────┴───────────┐
                         ▼                       ▼
                ablation (14x)         publication_figures
```

### Monitoring

```bash
# Check job status
squeue -u $USER

# Watch progress
watch -n 30 'squeue -u $USER'

# Check logs
tail -f $DATA/runs/logs/*.log
```

See `workflow/README.md` for detailed documentation.

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
@article{book2026stagebridge,
  author = {Book, AJ and others},
  title = {StageBridge: Receiver-Centered Niche Modeling for Cell-State Progression in Spatial and Single-Cell Omics},
  journal = {[Journal TBD]},
  year = {2026},
  note = {Manuscript in preparation}
}
```

**Note:** Author order and citation details will be finalized upon publication.

## License

[MIT](LICENSE)
