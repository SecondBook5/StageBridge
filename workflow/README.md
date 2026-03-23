# StageBridge Reproducible Workflow

Snakemake workflow for reproducible execution of the complete StageBridge V1 pipeline.

## Quick Start

### Prerequisites

Ensure required input files exist:
```bash
DATA=/scratch/chaunzt1/stagebridge  # or your data root

# Required inputs
ls $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad
ls $DATA/references/hlca/hlca_reference.h5ad
ls $DATA/references/luca/luca_core_atlas.h5ad
ls $DATA/processed/luad_evo/spatial_merged.h5ad
ls $DATA/processed/luad_evo/wes_features.parquet
```

### Local Execution

```bash
cd /path/to/StageBridge

# Dry run (see what would execute)
snakemake -n -s workflow/Snakefile --configfile workflow/config.yaml

# Full execution (8 cores)
snakemake -s workflow/Snakefile --configfile workflow/config.yaml --cores 8
```

### HPC Execution (SLURM)

```bash
# With SLURM profile
snakemake --profile workflow/slurm -s workflow/Snakefile --jobs 20
```

## Pipeline DAG

```
  hlca_mapping ─────┬──► spatial_backend (4x parallel) ──┐
       │            │    tangram, destvi, tacco, c2l     │
       │            │                                     │
       └──► fuse_embeddings ◄── luca_mapping (parallel)  │
                   │                                      │
                   └────────────────┬─────────────────────┘
                                    ▼
                           data_preparation
                                    │
                                    ▼
                           training (4-GPU DDP)
                           - SSL Pretraining (100 epochs)
                           - Transition Model (50 epochs)
                                    │
                                    ▼
                           ablation (14x parallel)
                                    │
                                    ▼
                         publication_figures
```

**Key insight:** Spatial backends need cell types from HLCA, so they run AFTER hlca_mapping but IN PARALLEL with luca_mapping.

## Configuration

Edit `workflow/config.yaml`:

```yaml
data_root: "/scratch/chaunzt1/stagebridge"

# Reference mapping
hlca_latent_key: "X_scANVI"
luca_latent_key: "X_scANVI"
hlca_n_latent: 30
luca_n_latent: 10

# Training
ssl_epochs: 100
transition_epochs: 50
batch_size: 256
```

## Files

```
workflow/
├── Snakefile           # Main workflow definition
├── config.yaml         # Pipeline configuration
├── README.md           # This file
├── scripts/            # Helper scripts called by Snakemake rules
│   ├── aggregate_baselines.py        # Aggregate baseline comparison results
│   ├── aggregate_cv_results.py       # Aggregate cross-validation results
│   ├── compare_spatial_backends.py   # Compare spatial backend outputs
│   ├── fuse_embeddings.py            # Fuse HLCA + LuCA embeddings
│   ├── generate_publication_figures.py
│   ├── generate_semi_synthetic.py    # Generate semi-synthetic benchmark data
│   ├── merge_cell_types.py           # Merge cell type annotations
│   ├── plot_training_curves.py       # Plot training loss/metrics
│   ├── summarize_ablations.py        # Summarize ablation study results
│   ├── validate_markers.py           # Validate marker gene expression
│   ├── validate_splits.py            # Validate train/val/test splits
│   └── visualize_attention.py        # Visualize attention weights
└── slurm/              # SLURM cluster configuration
    ├── config.yaml     # Snakemake SLURM profile
    └── status.py       # Job status checker
```

## Outputs

After successful completion:

```
$DATA/
├── processed/luad_evo/
│   ├── reference_geometry/
│   │   ├── hlca_mapping/
│   │   ├── luca_mapping/
│   │   └── fused_embedding.parquet
│   └── canonical/
│       ├── cells.parquet
│       ├── neighborhoods.parquet
│       └── split_manifest.json
└── runs/
    ├── spatial_benchmark/
    ├── v1_complete/
    │   ├── checkpoints/
    │   │   ├── ssl_pretrained.pt
    │   │   ├── best_checkpoint.pt
    │   │   └── final_checkpoint.pt
    │   └── results.json
    ├── ablations/
    └── publication_figures/
```

## Troubleshooting

### Check job status
```bash
snakemake --profile workflow/slurm --jobs 20 --rerun-incomplete
```

### Force re-run of a specific rule
```bash
snakemake --profile workflow/slurm -R hlca_mapping
```

### Generate DAG visualization
```bash
snakemake --dag -s workflow/Snakefile | dot -Tpdf > dag.pdf
```
