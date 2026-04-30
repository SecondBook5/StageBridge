# StageBridge

Receiver-centered niche modeling for lung cancer progression using spatial transcriptomics.

StageBridge learns cell state transitions conditioned on local microenvironment (niche) context, combining:
- **Dual-reference geometry**: HLCA (healthy) and LuCA (cancer) atlas embeddings
- **Receiver-centered attention**: spatial rings of neighbors centered on each cell
- **OT-CFM**: Optimal transport conditional flow matching for transition dynamics

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/StageBridge.git
cd StageBridge

# Install with pip (editable mode recommended for development)
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, PyTorch 2.0+

## Quick Start

### Demo with Synthetic Data

```bash
python run.py demo --epochs 5 --output results/
```

This will:
1. Generate synthetic neighborhood data (500 cells, 4 donors)
2. Train the model (SSL pretraining + transition training)
3. Run inference and report mean displacement

### Project Structure

```
stagebridge/
    baselines/       # Baseline models (DeepSets, SetTransformer, GraphSAGE)
    biology/         # Biological interpretation (L-R scoring, intervention targets)
    context/         # Niche tokenizer and hierarchical set transformer
    contracts.py     # Type contracts and constants
    evaluation/      # Evaluation metrics and ablations
    genomics/        # WES integration (clonality, variant annotation)
    loaders/         # Data loading and batching
    models/          # Core StageBridge model
    reference/       # HLCA/LuCA reference integration
    training/        # Two-stage trainer (SSL + transition)
    transition/      # OT-CFM drift networks and losses
```

## Data Format

StageBridge expects a `neighborhoods.parquet` file with columns:

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | str | Unique cell identifier |
| `donor_id` | str | Donor/sample identifier |
| `stage` | str | Disease stage (Normal, Preinvasive, Invasive) |
| `receiver_z` | list[float] | 40-dim latent embedding of receiver cell |
| `hlca_z` | list[float] | 40-dim HLCA reference embedding |
| `luca_z` | list[float] | 40-dim LuCA reference embedding |
| `pathway_z` | list[float] | 40-dim pathway activity embedding |
| `stats_z` | list[float] | 40-dim summary statistics embedding |
| `ring_1_cells` | list[list[float]] | Variable-length list of neighbor embeddings (ring 1) |
| `ring_2_cells` | list[list[float]] | Ring 2 neighbors |
| `ring_3_cells` | list[list[float]] | Ring 3 neighbors |
| `ring_4_cells` | list[list[float]] | Ring 4 neighbors |

Plus a `split_manifest.json` for cross-validation folds.

## Architecture

### 9-Token Niche Representation

Each cell's neighborhood is encoded as a sequence of 9 tokens:

```
[Receiver, Ring1, Ring2, Ring3, Ring4, HLCA, LuCA, Pathway, Stats]
```

- **Receiver**: The target cell being modeled
- **Ring 1-4**: Spatial neighbors at increasing distances, pooled via ISAB+PMA
- **HLCA/LuCA**: Reference atlas embeddings (healthy/cancer)
- **Pathway/Stats**: Functional annotations

### Two-Stage Training

1. **SSL Pretraining**: Learn niche-aware representations via masked token reconstruction
2. **OT-CFM Transition Training**: Learn stage transitions conditioned on niche context

### Key Components

- **NicheTokenizer**: Converts raw neighbor cells to 9-token structure using learned ISAB+PMA pooling
- **HierarchicalSetTransformer**: Refines tokens with spatial RPE (relative position encoding)
- **CrossAttentionDrift**: Predicts velocity field conditioned on context tokens

## Training

### Full Training Pipeline

```bash
# Using the trainer directly
python -c "
from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.training import StageBridgeTrainer, TrainerConfig
from stagebridge.loaders import create_dataloaders

train_loader, val_loader, _ = create_dataloaders('data/processed/', fold_idx=0)
model = StageBridge(StageBridgeConfig())
trainer = StageBridgeTrainer(model, TrainerConfig(ssl_epochs=50, transition_epochs=100))
trainer.train(train_loader, val_loader)
"
```

### Configuration

Key `StageBridgeConfig` parameters:

```python
StageBridgeConfig(
    input_dim=40,              # Latent dimension (matches scVI)
    hidden_dim=128,            # Transformer hidden dimension
    num_heads=4,               # Attention heads
    num_encoder_layers=2,      # Transformer depth
    num_stages=3,              # Number of disease stages
    use_learned_ring_pooling=True,   # ISAB+PMA per ring (recommended)
    use_context_refiner=True,        # Hierarchical set transformer
)
```

## Baselines

Compare against ablated architectures:

```python
from stagebridge.baselines import get_baseline

# Available baselines
model = get_baseline("pooling", input_dim=40, hidden_dim=64)      # Bag-of-cells
model = get_baseline("deepsets", input_dim=40, hidden_dim=64)     # Permutation invariant
model = get_baseline("set_transformer", input_dim=40, hidden_dim=64)  # Flat attention
model = get_baseline("graphsage", input_dim=40, hidden_dim=64)    # Graph structure
```

## WES Integration

StageBridge integrates whole-exome sequencing for clonality analysis:

```python
from stagebridge.genomics import (
    estimate_ccf,
    classify_clonality,
    annotate_germline,
    annotate_somatic,
)

# Estimate cancer cell fraction
ccf = estimate_ccf(vaf=0.4, purity=0.6, total_cn=2)

# Classify as clonal/subclonal
clonality = classify_clonality(vaf=0.4, ccf=ccf)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_gradient_contracts.py -v  # Gradient flow tests
pytest tests/test_clonality.py -v           # WES integration tests
```

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
paths:
  data_root: "/data1/chaunzt1/stagebridge"
  data_dir: "/data1/chaunzt1/stagebridge/processed/luad_evo"
  output_dir: "/data1/chaunzt1/stagebridge/outputs/v1"
```

### Pipeline DAG

```
                            validate_data
                                  │
           ┌──────────────────────┼──────────────────────┐
           ▼                      ▼                      ▼
    train_full (5×3)      train_baseline (4×5)    run_ablation (9×5)
           │                      │                      │
           ▼                      │                      │
       infer_full                 │                      │
           │                      │                      │
           ▼                      │                      │
      evaluate_full               │                      │
           │                      │                      │
           └──────────────────────┴──────────────────────┘
                                  │
                                  ▼
                         comparison_report
                                  │
                                  ▼
                    run_inference_for_figures
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
   fig1_architecture    fig4_embedding_flow   fig5_biological_validation
   fig2_training        fig6_phase_portrait   fig8_spatial_attention
   fig3_ablations       fig7_trajectories     fig9_novel_biology
```

**Job counts (5-fold CV, 3 seeds, 4 baselines, 9 ablations):**
- Full model training: 5 folds = **5 jobs** (seeds handled internally)
- Baselines: 4 baselines × 5 folds = **20 jobs**
- Ablations: 9 ablations × 5 folds = **45 jobs**
- Inference + evaluation: 5 folds = **10 jobs**
- Figures: **9 jobs**
- **Total: ~89 jobs**

### Available Rules

| Rule | Description |
|------|-------------|
| `all` | Run full pipeline (training + baselines + ablations + figures) |
| `validate_data` | Validate data directory contracts |
| `train_full` | Train StageBridge (2-stage: SSL + OT-CFM) on all folds |
| `train_baseline` | Train baseline models (pooling_mlp, deepsets, set_transformer, graphsage) |
| `run_ablation` | Run ablation experiments (no_niche, no_distance, hlca_only, etc.) |
| `infer_full` | Run inference with trained model |
| `evaluate_full` | Compute evaluation metrics |
| `comparison_report` | Aggregate results into comparison JSON |
| `figures` | Generate all 9 publication figures |
| `clean` | Remove all outputs |
| `clean_figures` | Remove only figures, keep training artifacts |
| `dry_run` | Show what would run without executing |

### Monitoring

```bash
# Check job status
squeue -u $USER

# Watch progress
watch -n 30 'squeue -u $USER'

# View Snakemake report
snakemake --report report.html
```

## Citation

If you use StageBridge in your research, please cite:

```bibtex
@software{stagebridge2026,
  title = {StageBridge: Receiver-centered niche modeling for cancer progression},
  year = {2026},
  url = {https://github.com/your-org/StageBridge}
}
```

## License

MIT License - see LICENSE file for details.
