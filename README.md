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

StageBridge learns **cell-state transitions conditioned on local microenvironment (niche) context**. The framework models lung adenocarcinoma (LUAD) progression at the cell and niche level.

```
Normal  ──>  Preinvasive  ──>  Invasive
```

| Stage | Histological Types | Description |
|-------|-------------------|-------------|
| **Normal** | Normal alveolar | Healthy tissue reference |
| **Preinvasive** | AAH, AIS, MIA | Pre-malignant lesions |
| **Invasive** | LUAD | Invasive adenocarcinoma |

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

### Python API

```python
import stagebridge as sb

# Load pretrained model
model = sb.StageBridge.from_pretrained("checkpoint.pt")

# Prepare AMICI neighborhoods from AnnData (K nearest neighbors with distances)
sb.prepare_neighborhoods(adata, k_neighbors=50)

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

**Job counts (5-fold x 3 seeds CV):**
- Full model: 15 jobs (5 folds x 3 seeds)
- Baselines (4 types): 60 jobs (4 x 5 x 3)
- Ablations: 45 jobs (3 key ablations x 5 x 3)
- Figures: 9 jobs
- **Total: ~130 jobs**

---

## Project Structure

```
stagebridge/
├── baselines/       # DeepSets, SetTransformer, GraphSAGE
├── biology/         # L-R scoring, intervention targets
├── context/         # AMICI encoder, Set Transformer layers
├── contracts.py     # Data schemas and validation
├── evaluation/      # Metrics, ablations
├── loaders/         # DataLoader, batching
├── models/          # StageBridge model
├── pipelines/       # prepare_data, infer, HPO
├── reference/       # HLCA/LuCA mapping, GW fusion
├── training/        # Two-stage trainer
└── transition/      # OT-CFM drift networks

scripts/
├── data/            # Data preparation pipeline
├── benchmarks/      # Benchmark and baseline scripts
├── analysis/        # Post-training analysis
├── figures/         # Publication figures
└── hpc/             # SLURM job scripts

workflow/
├── Snakefile        # Pipeline definition
├── config.yaml      # HPC paths and parameters
└── slurm/           # SLURM profile
```

**Documentation:**
- [scripts/](scripts/README.md) - Data prep, benchmarks, analysis, figures
- [stagebridge/context/](stagebridge/context/README.md) - AMICI attention, Set Transformer
- [stagebridge/reference/](stagebridge/reference/README.md) - Atlas fusion, learned GW
- [stagebridge/transition/](stagebridge/transition/README.md) - OT-CFM, drift networks
- [stagebridge/training/](stagebridge/training/README.md) - Two-stage training
- [stagebridge/models/](stagebridge/models/README.md) - Core model architecture
- [stagebridge/interpretation/](stagebridge/interpretation/README.md) - Interpretability tools

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
| `no_distance` | Remove distance-based attention decay |
| `hlca_only` | Only healthy reference |
| `luca_only` | Only tumor reference |
| `frozen_encoder` | Freeze encoder during transition training |

---

## Architecture

StageBridge uses **receiver-centered AMICI attention** where each cell attends to its K nearest neighbors with continuous distance-weighted attention (monotonic decay). Dual-reference embeddings (HLCA + LuCA) are fused via **learned Gromov-Wasserstein alignment**.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        StageBridge Pipeline                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   Preprocessing (scArches)                   │    │
│  │  ┌──────────┐     ┌──────────┐     ┌─────────────────────┐  │    │
│  │  │   HLCA   │     │   LuCA   │     │  Learned GW Fusion  │  │    │
│  │  │  (30d)   │────>│  (10d)   │────>│      (40d out)      │  │    │
│  │  └──────────┘     └──────────┘     └─────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              v                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    AMICI Niche Encoder                       │    │
│  │  ┌──────────┐     ┌───────────────┐     ┌───────────────┐   │    │
│  │  │ Receiver │────>│ K Neighbors   │────>│ Distance-     │   │    │
│  │  │  (40d)   │     │ + distances   │     │ weighted Attn │   │    │
│  │  └──────────┘     └───────────────┘     └───────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              v                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    OT-CFM Transition                         │    │
│  │         Stage transitions via optimal transport flow         │    │
│  │              Normal ──> Preinvasive ──> Invasive             │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

| Component | Description |
|-----------|-------------|
| Learned GW Fusion | Aligns HLCA (30d) + LuCA (10d) via Gromov-Wasserstein |
| Receiver | Target cell's fused embedding (40d) |
| Neighbors | K nearest spatial neighbors with continuous distances |
| Distance Attention | Monotonic decay weighting (AMICI-style) |
| OT-CFM | Optimal transport conditional flow matching |

**Two-Stage Training:**
1. **SSL Pretraining**: Learn niche-aware representations via masked receiver reconstruction
2. **OT-CFM Transition**: Learn stage transitions conditioned on niche context

**Technical Documentation:**
- [Model Architecture](docs/architecture/MODEL_ARCHITECTURE.md) - Complete architectural specification with code references
- [Key Equations](docs/presentation/EQUATIONS.md) - All equations with derivations, variable tables, and Q&A

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

**Key methodological references:**

Our receiver-centered attention mechanism is adapted from AMICI:

```bibtex
@article{Hong2025.09.22.677860,
  title = {AMICI: Attention Mechanism Interpretation of Cell-cell Interactions},
  author = {Hong, Justin and Desai, Khushi and Nguyen, Tu Duyen and Nazaret, Achille and Levy, Nathan and Ergen, Can and Plitas, George and Azizi, Elham},
  doi = {10.1101/2025.09.22.677860},
  journal = {bioRxiv},
  publisher = {Cold Spring Harbor Laboratory},
  year = {2025},
}
```

AMICI is available at https://github.com/azizilab/amici under CC BY-NC-ND 4.0 license. Patent pending (U.S. Serial No. 63/884,704).

## License

[MIT](LICENSE)
