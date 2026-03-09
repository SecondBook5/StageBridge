# StageBridge

**Niche-conditioned optimal transport for early lung adenocarcinoma progression**

StageBridge learns cell-state transition maps across the LUAD initiation ladder (Normal → AAH → AIS → MIA → LUAD) conditioned on local tissue microenvironment context from matched spatial transcriptomics. The framework benchmarks whether transformer-based niche encoding improves transition prediction over simpler aggregation baselines.

## Key results

| Task | Best method | Finding |
|------|------------|---------|
| AIS → MIA transition | Set Transformer context | Spatial niche context improves transition prediction |
| AAH → AIS transition | RNA-only baseline | Context does not help on this edge |
| Communication relay (clonal-proxy) | Pooled summary | Richer transformer architectures do not beat pooling with current supervision |

These task-dependent results define the boundary conditions for transformer-based spatial modeling in pre-malignant tissue.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Data Layer                                                 │
│  GSE308103 (snRNA-seq) + GSE307534 (Visium) + GSE307529    │
│  (WES)                                                      │
├─────────────────────────────────────────────────────────────┤
│  Reference Mapping         → HLCA-aligned latent space      │
│  Spatial Mapping           → Tangram / TACCO / DestVI       │
├─────────────────────────────────────────────────────────────┤
│  Context Encoding                                           │
│  Set Transformer · DeepSets · Graph-of-Sets · Hierarchical  │
│  Communication Relay (sender → LR → relay → receiver)       │
├─────────────────────────────────────────────────────────────┤
│  Transition Model                                           │
│  Schrödinger bridge interpolant · OT coupling (Sinkhorn)    │
│  Drift + diffusion networks · WES regularization            │
├─────────────────────────────────────────────────────────────┤
│  Evaluation                                                 │
│  Donor-held-out CV · Context ablation · Edge comparison     │
│  Gene attribution · Trajectory analysis                     │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone and create environment
git clone https://github.com/<user>/StageBridge.git
cd StageBridge
micromamba create -f environment.yml
micromamba activate stagebridge

# Install package in development mode
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+, CUDA-capable GPU (tested on NVIDIA RTX 4000 Ada, 20 GB VRAM).

## Quick start

```python
import stagebridge

# Compose a configuration and run the full pipeline
cfg = stagebridge.compose_config(
    overrides=["data=local", "context_model=set_only", "train=full_v1"]
)
outputs = stagebridge.run_pipeline(cfg)
```

Or run individual pipeline steps:

```python
from stagebridge.pipelines import run_transition_model, run_evaluation

transition = run_transition_model(cfg)
evaluation = run_evaluation(cfg, transition_output=transition)
```

## Data

The framework uses three matched GEO datasets from the Markov et al. early LUAD cohort:

| Accession | Modality | Description |
|-----------|----------|-------------|
| GSE308103 | snRNA-seq | Single-nucleus transcriptomes across 5 stages |
| GSE307534 | 10x Visium | Spatial transcriptomics (matched tissue sections) |
| GSE307529 | WES | Whole-exome sequencing (per-lesion mutation profiles) |

Data is stored externally and referenced via the `STAGEBRIDGE_DATA_ROOT` environment variable. See `configs/data/luad_evo.yaml` for the expected directory layout.

## Repository structure

```
stagebridge/
├── context_model/       # Niche context encoders (Set Transformer, GoST, etc.)
├── transition_model/    # OT coupling, Schrödinger bridge, drift/diffusion networks
├── evaluation/          # Metrics, ablations, gene attribution, trajectory analysis
├── reference/           # HLCA latent mapping and label transfer
├── spatial_mapping/     # Tangram, TACCO, DestVI spatial mapping backends
├── pipelines/           # Orchestration scripts for each pipeline stage
├── results/             # Result tracking and milestone system
├── data/                # Data loaders and stage ontology
├── viz/                 # Figure generation
└── utils/               # Configuration, seeds, type helpers
configs/                 # YAML configs (data, model, training, evaluation)
docs/
├── architecture/        # Module-level design docs
└── biology/             # Scientific rationale and hypotheses
reports/benchmarks/      # Reproducible benchmark tables
tests/                   # Test suite (87 tests)
```

## Configuration

StageBridge uses [OmegaConf](https://omegaconf.readthedocs.io/) with composable YAML configs:

```
configs/
├── default.yaml                    # Base configuration
├── context_model/
│   ├── set_only.yaml               # Set Transformer (default)
│   ├── deep_sets.yaml              # DeepSets baseline
│   ├── graph_of_sets.yaml          # Graph-of-Sets Transformer
│   └── typed_hierarchical_transformer.yaml
├── transition_model/
│   ├── schrodinger_bridge.yaml     # SB loss parameters
│   ├── stochastic_dynamics.yaml    # Drift/diffusion network config
│   └── wes_regularizer.yaml        # WES-conditioned OT regularization
├── train/
│   ├── full_v1.yaml                # Production training (150 epochs, GPU)
│   ├── medium.yaml                 # Medium runs (4 epochs, CPU)
│   └── smoke.yaml                  # CI smoke tests (2 epochs, CPU)
└── splits/
    └── donor_holdout.yaml          # Donor-held-out cross-validation
```

## Testing

```bash
pytest tests/ -v
```

The test suite validates OT coupling properties, model forward passes across all context encoder variants, Schrödinger bridge loss gradients, and end-to-end pipeline integration on real data subsets.

## Citation

If you use StageBridge in your research, please cite:

```bibtex
@software{stagebridge2026,
  title={StageBridge: Niche-Conditioned Optimal Transport for Early Lung Cancer Progression},
  author={Book, AJ},
  year={2026},
  url={https://github.com/<user>/StageBridge}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
