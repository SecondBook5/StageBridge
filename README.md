<p align="center">
  <h1 align="center">StageBridge</h1>
  <p align="center">
    <strong>Niche-conditioned cell state transition modeling<br>for spatial and single-cell transcriptomics</strong>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+"></a>
    <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch 2.0+"></a>
  </p>
</p>

---

## Overview

StageBridge learns **cell-state transitions conditioned on local microenvironment (niche) context**. The framework models progression at the cell and niche level, not as patient classification.

Primary application: lung adenocarcinoma (LUAD) progression

```
Normal  ──>  Preinvasive  ──>  Invasive
```

| Stage | Histological Types | Description |
|-------|-------------------|-------------|
| **Normal** | Normal alveolar | Healthy tissue reference |
| **Preinvasive** | AAH, AIS, MIA | Pre-malignant lesions |
| **Invasive** | LUAD | Invasive adenocarcinoma |

### Core Principles

- **Cell-level transitions**: Scientific object is cell-state change, not patient classification
- **Niche conditioning**: Transitions depend on local neighborhood context
- **Dual-reference geometry**: Cells embedded relative to healthy (HLCA) and tumor (LuCA) atlases
- **OT-CFM dynamics**: Optimal transport conditional flow matching for transition fields

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        StageBridge Pipeline                          │
│                                                                      │
│  ┌─────────────┐   ┌──────────────────┐   ┌────────────────────┐    │
│  │  Dual-Ref   │──>│  9-Token Niche   │──>│  Set Transformer   │    │
│  │   Latent    │   │    Encoder       │   │  (ISAB/SAB/PMA)    │    │
│  └─────────────┘   └──────────────────┘   └────────────────────┘    │
│        │                                            │                │
│        v                                            v                │
│  ┌─────────────┐                          ┌────────────────────┐    │
│  │ HLCA + LuCA │                          │    OT-CFM Flow     │    │
│  │  Reference  │                          │     Matching       │    │
│  │  Alignment  │                          └────────────────────┘    │
│  └─────────────┘                                    │                │
│                                                     v                │
│                                           ┌────────────────────┐    │
│                                           │  Cell Transition   │    │
│                                           │   Trajectories     │    │
│                                           └────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 9-Token Niche Representation

Each cell's neighborhood is encoded as a sequence of 9 tokens:

| Token | Source | Description |
|-------|--------|-------------|
| Receiver | Cell identity | Target cell embedding |
| Ring 1-4 | Spatial neighborhood | Neighbors at increasing radii (learned ISAB+PMA pooling) |
| HLCA | Healthy atlas | Embedding relative to HLCA reference |
| LuCA | Tumor atlas | Embedding relative to LuCA reference |
| Pathway | Gene programs | Pathway activity (14 PROGENy pathways) |
| Stats | Conditioning | CAF/immune fractions, diversity, cell cycle |

### Two-Stage Training

1. **SSL Pretraining**: Learn niche-aware representations via masked receiver reconstruction
2. **OT-CFM Transition**: Learn stage transitions conditioned on niche context

---

## Data

StageBridge integrates multi-modal data from the Peng/Kadara LUAD cohort:

| Modality | GEO Accession | Description |
|----------|---------------|-------------|
| snRNA-seq | [GSE308103](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308103) | 798k cells, single-cell expression |
| 10x Visium | [GSE307534](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307534) | 640k spots, spatial transcriptomics |
| WES | [GSE307529](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307529) | Whole-exome sequencing |

**Reference atlases:**
- [Human Lung Cell Atlas (HLCA)](https://doi.org/10.1038/s41591-023-02327-2) - healthy reference
- [LuCA](https://www.cell.com/cancer-cell/fulltext/S1535-6108(22)00499-8) - tumor-aware reference

**Cell type annotations** from LuCA DestVI deconvolution (33 cell types including malignant cells).

---

## Installation

```bash
git clone https://github.com/SecondBook5/StageBridge.git
cd StageBridge

# Create environment
conda create -n stagebridge python=3.11
conda activate stagebridge

# Install
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA 12.x

---

## Quick Start

### Python API (Recommended)

```python
import stagebridge as sb

# Load pretrained model
model = sb.StageBridge.from_pretrained("checkpoint.pt")

# Prepare neighborhoods from AnnData
sb.prepare_neighborhoods(adata, ring_radii=[50, 100, 150, 200])

# Get niche-aware embeddings
embeddings = model.embed_niches(adata.uns["X_neighborhoods"])

# Predict cell state transitions
predictions = model.predict(
    neighborhoods=adata.uns["X_neighborhoods"],
    source_stage="Normal",
    target_stage="Invasive",
)

# Visualize
sb.pl.embedding(embeddings.embeddings, stages=adata.obs["stage"])
sb.pl.flow_field(predictions.source_embeddings, 
                 predictions.predicted_embeddings - predictions.source_embeddings)
```

### Tutorials

See the [notebooks/](notebooks/) directory for detailed tutorials:

| Notebook | Description |
|----------|-------------|
| [01_quickstart.ipynb](notebooks/01_quickstart.ipynb) | Load model, run inference, visualize |
| [02_training.ipynb](notebooks/02_training.ipynb) | Train your own model |

### Command Line

```bash
# Demo with synthetic data
python run.py demo --epochs 5

# Train on real data (HPC with Snakemake)
snakemake --profile workflow/slurm --jobs 20

# Or directly
python -m stagebridge.training.train \
    --data-dir /path/to/data \
    --output-dir outputs/fold_0 \
    --fold 0
```

---

## HPC Deployment (Snakemake)

StageBridge uses **Snakemake** for HPC orchestration.

```bash
# Dry run
snakemake -n --profile workflow/slurm

# Full pipeline
snakemake --profile workflow/slurm --jobs 20

# Visualize DAG
snakemake --dag | dot -Tpdf > dag.pdf
```

### Configuration

Edit `workflow/config.yaml`:

```yaml
paths:
  data_dir: "/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
  output_dir: "/data1/chaunzt1/stagebridge/outputs/v1"
```

### Pipeline DAG

```
                        prepare_data
                             │
                        validate_data
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   train_full (5)    train_baseline (20)   run_ablation (45)
         │                   │                   │
         ▼                   │                   │
    infer_full               │                   │
         │                   │                   │
         ▼                   │                   │
   evaluate_full             │                   │
         │                   │                   │
         └───────────────────┴───────────────────┘
                             │
                             ▼
                    comparison_report
                             │
                             ▼
                     publication_figures
```

**Job counts (5-fold CV):**
- Full model: 5 jobs
- Baselines (4 types): 20 jobs
- Ablations (9 types): 45 jobs
- Figures: 9 jobs
- **Total: ~80 jobs**

---

## Project Structure

```
stagebridge/
├── baselines/       # DeepSets, SetTransformer, GraphSAGE
├── biology/         # L-R scoring, intervention targets
├── context/         # NicheTokenizer, HierarchicalSetTransformer
├── contracts.py     # Data schemas and validation
├── evaluation/      # Metrics, ablations
├── loaders/         # DataLoader, batching
├── models/          # StageBridge model
├── pipelines/       # prepare_data, infer
├── training/        # Two-stage trainer
└── transition/      # OT-CFM drift networks

workflow/
├── Snakefile        # Pipeline definition
├── config.yaml      # HPC paths and parameters
└── slurm/           # SLURM profile
```

---

## Baselines

```python
from stagebridge.baselines import get_baseline

model = get_baseline("pooling_mlp")      # Bag-of-cells (no structure)
model = get_baseline("deepsets")         # Permutation invariant
model = get_baseline("set_transformer")  # Flat attention
model = get_baseline("graphsage")        # Graph structure
```

---

## Ablations

| Ablation | Tests |
|----------|-------|
| `no_niche` | Remove all niche context |
| `no_distance` | Remove distance-based attention |
| `no_gate` | Remove biological baseline gate |
| `hlca_only` | Only healthy reference |
| `luca_only` | Only tumor reference |
| `no_ring_pooling` | Mean pooling vs learned ISAB+PMA |
| `no_context_refiner` | Remove hierarchical set transformer |
| `frozen_encoder` | Freeze encoder during transition training |

---

## Citation

```bibtex
@article{book2026stagebridge,
  author = {Book, AJ and others},
  title = {StageBridge: Niche-Conditioned Cell State Transition Modeling},
  journal = {[In preparation]},
  year = {2026}
}
```

## License

[MIT](LICENSE)
