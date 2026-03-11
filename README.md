<p align="center">
  <h1 align="center">StageBridge</h1>
  <p align="center">
    <strong>Transformer-based modeling of lung adenocarcinoma stage progression<br>from spatial transcriptomics, single-cell RNA-seq, and whole-exome sequencing</strong>
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

StageBridge models the full progression cascade of lung adenocarcinoma (LUAD) from pre-malignant lesions to invasive carcinoma:

```
Normal  ──>  AAH  ──>  AIS  ──>  MIA  ──>  LUAD
                                              ├──>  Brain Metastasis
                                              └──>  Chest Wall Metastasis
```

The framework integrates three data modalities -- 10x Visium spatial transcriptomics, snRNA-seq, and whole-exome sequencing -- into a unified transformer architecture that learns **lesion-level stage representations** from local tissue microenvironments (niches).

### Key contributions

- **EA-MIST** (Evolution-Aware Multiple-Instance Set Transformer) -- the primary benchmarked lesion-level model that encodes spatial niches as structured token sequences and aggregates them with a permutation-invariant Set Transformer
- **Benchmark model family** centered on EA-MIST variants (`eamist`, `eamist_no_prototypes`, `lesion_set_transformer`, `deep_sets`, `pooled`) under donor-held-out evaluation
- **Dual reference alignment** against the Human Lung Cell Atlas (HLCA) and LuCA tumor atlas for healthy-to-malignant context
- **Label repair system** with multi-evidence refinement (WES, CNA, clonal architecture, pathology) for rigorous stage annotation
- **Experimental research extensions** including Graph-of-Sets Transformer (GoST) and Schr&ouml;dinger bridge / OT transition modeling (not part of the default EA-MIST benchmark path)

---

## Architecture

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                    EA-MIST Pipeline                      │
                         │                                                         │
  Spatial Niche ────>    │   9-Token Local        Prototype        Set Transformer  │
  (receiver +            │   Niche Encoder   ──>  Bottleneck  ──>  (ISAB→SAB→PMA)  │
   4 rings +             │   (per niche)          (optional)       (per lesion)     │
   HLCA/LuCA +           │                                              │           │
   pathway + stats)      │                                              v           │
                         │                                    Evolution Branch      │
  WES Features ────────> │                                    (gated fusion)        │
                         │                                              │           │
                         │                                     ┌────────┴────────┐  │
                         │                                     │  Multitask Heads │  │
                         │                                     │  - Stage (5-way) │  │
                         │                                     │  - Displacement  │  │
                         │                                     │  - Edges (aux)   │  │
                         │                                     └─────────────────┘  │
                         └─────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  Experimental Research Extensions (not default EA-MIST benchmark path)         │
  │                                                                                 │
  │  Graph-of-Sets Transformer (GoST)          OT Transition Model                  │
  │  - Stage-adjacent edges                    - Sinkhorn OT coupling               │
  │  - Same-patient cross-stage edges          - FiLM-conditioned drift/diffusion    │
  │  - Same-stage cross-patient edges          - Euler trajectory integration        │
  │  - Scatter-softmax sparse attention        - Schrödinger bridge objective        │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

### Local niche encoding

Each spatial niche is encoded as a **9-token sequence**:

| Token | Source | Description |
|-------|--------|-------------|
| Receiver | Cell identity | Target cell expression + learned state embedding |
| Ring 1--4 | Spatial neighborhood | Cell-type composition at increasing radii |
| HLCA | Reference atlas | Similarity to healthy lung cell types |
| LuCA | Tumor atlas | Similarity to tumor-aware cell states |
| Pathway | Gene programs | Ligand-receptor and pathway activity summary |
| Stats | Neighborhood | Local density, entropy, and composition statistics |

### Model variants

| Model | Description | Use case |
|-------|-------------|----------|
| `eamist` | Full EA-MIST with prototypes + evolution branch | Primary benchmark |
| `eamist_no_prototypes` | EA-MIST without prototype bottleneck | Ablation |
| `lesion_set_transformer` | Set Transformer only (no local encoder) | Ablation |
| `deep_sets` | DeepSets baseline | Baseline |
| `pooled` | Mean-pooling baseline | Baseline |

### Experimental extensions

The repository also includes exploratory modules that are valuable for future work but are not part of the canonical V1 benchmark narrative:

- **Graph-of-Sets Transformer (GoST)** -- inter-lesion / inter-patient graph-context extension
- **Schr&ouml;dinger bridge / OT transition model** -- probabilistic trajectory modeling extension

These modules remain in-repo with configs and tests, but the default quick-start and benchmark workflow are centered on EA-MIST.

---

## Data

StageBridge integrates multi-modal data from public GEO repositories:

| Dataset | Modality | GEO Accession | Role |
|---------|----------|---------------|------|
| Early LUAD snRNA-seq | Single-cell transcriptomics | [GSE308103](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308103) | Cell-level expression |
| 10x Visium | Spatial transcriptomics | [GSE307534](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307534) | Tissue architecture |
| Whole-exome sequencing | WES | [GSE307529](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307529) | Evolutionary features |
| Brain metastasis snRNA-seq | Single-cell (extension) | [GSE223499](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223499) | Metastatic progression |

**Reference atlases:**
- [Human Lung Cell Atlas (HLCA)](https://doi.org/10.1038/s41591-023-02327-2) -- healthy reference anchor
- [LuCA extended atlas](https://www.cell.com/cancer-cell/fulltext/S1535-6108(22)00499-8) -- tumor-aware cell state reference

**Spatial mapping providers:**
- [Tangram](https://www.nature.com/articles/s41592-021-01264-7) -- deep learning-based spatial mapping of single-cell transcriptomes
- [TACCO](https://www.nature.com/articles/s41587-023-01657-3) -- transfer of annotations to cells and their combinations in spatial omics
- [DestVI](https://www.nature.com/articles/s41587-022-01272-8) -- multi-resolution deconvolution of spatial transcriptomics data

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

**Requirements:** Python 3.11+, PyTorch 2.2+, CUDA 12.x

---

## Quick start

The default workflow below is the canonical EA-MIST benchmark path.

### Python API

```python
from stagebridge.notebook_api import compose_config
from stagebridge.pipelines import (
    run_train_lesion,
    run_evaluate_lesion,
    run_eamist_reporting,
)

# Configure and train
cfg = compose_config(overrides=["context_model=eamist"])
results = run_train_lesion(cfg)

# Evaluate and generate publication figures
eval_results = run_evaluate_lesion(cfg)
report = run_eamist_reporting(cfg)
```

### Command line

```bash
# Train EA-MIST
python -m stagebridge.pipelines step train_lesion -o context_model=eamist

# Evaluate
python -m stagebridge.pipelines step evaluate_lesion -o context_model=eamist

# Generate figures and tables
python -m stagebridge.pipelines step eamist_report -o context_model=eamist
```

### Full pipeline (build bags, train, evaluate, report)

```bash
bash scripts/run_eamist_full.sh
```

---

## Evaluation

EA-MIST is evaluated under **donor-held-out cross-validation** on lesion-level prediction:

| Metric | Task |
|--------|------|
| Macro-F1 | 5-way stage classification |
| Balanced accuracy | Stage classification |
| Confusion matrix | Per-stage support analysis |
| MAE | Displacement regression |
| Spearman correlation | Displacement ordering |
| Monotonicity | Stage-wise displacement trend |

Additional evaluation modules:
- Sinkhorn distance, MMD-RBF, classifier AUC (transition-model extension)
- Context sensitivity analysis (real vs. shuffled context)
- Gene-context correlations and niche shift profiling
- Calibration error analysis

---

## Repository structure

```
stagebridge/
├── context_model/          # EA-MIST core + experimental context encoders (e.g., GoST)
│   ├── lesion_set_transformer.py    # EAMISTModel
│   ├── local_niche_encoder.py       # 9-token niche transformer
│   ├── set_encoder.py               # ISAB, SAB, PMA
│   ├── graph_of_sets.py             # Graph-of-Sets Transformer
│   └── prototype_bottleneck.py      # Prototype compression
├── transition_model/       # Experimental OT / Schrödinger bridge trajectory modules
│   ├── stochastic_dynamics.py       # StageBridgeModel
│   ├── schrodinger_bridge.py        # Sinkhorn OT coupling
│   └── drift_network.py            # FiLM-conditioned drift
├── data/                   # Data loading and preprocessing
│   ├── luad_evo/                    # LUAD progression datasets
│   └── brainmets/                   # Brain metastasis extension
├── evaluation/             # Metrics, calibration, ablations
├── pipelines/              # End-to-end workflow orchestration
├── reference/              # HLCA/LuCA atlas alignment
├── spatial_mapping/        # Tangram, TACCO, DestVI providers
├── labels/                 # Multi-evidence label refinement
├── viz/                    # Publication-quality figures
├── results/                # Run tracking and milestone management
└── utils/                  # Configuration, I/O, seeds, types

configs/                    # Hydra YAML configuration system
├── context_model/          # Model architecture configs
├── train/                  # Training profiles (full, medium, smoke)
├── evaluation/             # Evaluation and ablation configs
└── transition_model/       # Flow matching settings

tests/                      # 33 test files, ~4,400 lines
docs/                       # Architecture and biology documentation
```

---

## Testing

```bash
# Full test suite
pytest tests/

# EA-MIST model tests
pytest tests/test_eamist_model.py tests/test_eamist_pipelines.py

# Context model ablations
pytest tests/test_set_only_context.py tests/test_deep_sets_context.py

# Experimental Graph-of-Sets extension
pytest tests/test_graph_of_sets_context.py
```

---

## Configuration

StageBridge uses [Hydra](https://hydra.cc/) for composable YAML configuration:

```bash
# Train with specific model variant
python -m stagebridge.pipelines step train_lesion \
    -o context_model=eamist train=full_v1

# Run evaluation with ablation config
python -m stagebridge.pipelines step evaluate_lesion \
    -o context_model=eamist evaluation=ablation

# Smoke test (fast iteration)
python -m stagebridge.pipelines step train_lesion \
    -o context_model=eamist train=smoke
```

---

## Citation

If you use StageBridge in your research, please cite:

```bibtex
@software{book2026stagebridge,
  author = {Book, AJ},
  title = {StageBridge: Transformer-based modeling of lung adenocarcinoma stage progression},
  year = {2026},
  url = {https://github.com/SecondBook5/StageBridge}
}
```

## License

[MIT](LICENSE)
