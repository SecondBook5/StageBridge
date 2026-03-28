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
  hlca_mapping ─────┬──► add_cell_type_labels ──► validate_markers
       │            │              │
       │            │              ▼
       └──► fuse_embeddings   spatial_backend_sample (4 backends × 2 label sources × 56 samples)
                   │           tangram, destvi, tacco, cell2location
                   │           × hlca labels, luca labels
  luca_mapping ────┘                    │
                                        ▼
                              spatial_backend_aggregate (per backend)
                                        │
                                        ▼
                              spatial_comparison ──► canonical_backend.json
                                        │
                                        ▼
                              canonical_backend_sample (56 samples)
                                        │
                                        ▼
                                 data_preparation
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                  training (15×)   baselines (60×)   hpo
                  5 folds × 3 seeds  4 baselines
                         │              │
                         ▼              ▼
                  aggregate_cv    aggregate_baselines
                         │              │
                         └──────┬───────┘
                                ▼
                         ablation (14×)
                                │
                                ▼
                      publication_figures
```

**Key insights:**
- Spatial backends run per-sample (56 samples) for memory efficiency
- Each backend runs with BOTH HLCA and LuCA cell type labels (ablation)
- After benchmarking, the canonical backend runs on all samples
- Training uses 5-fold CV × 3 seeds = 15 runs for robust statistics

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
│   ├── add_cell_type_labels.py       # Add HLCA + LuCA cell type labels
│   ├── aggregate_baselines.py        # Aggregate baseline comparison results
│   ├── aggregate_cv_results.py       # Aggregate cross-validation results
│   ├── aggregate_spatial_samples.py  # Aggregate spatial backend per-sample outputs
│   ├── compare_spatial_backends.py   # Compare spatial backend outputs
│   ├── extract_spatial_samples.py    # Extract per-sample spatial data
│   ├── fuse_embeddings.py            # Fuse HLCA + LuCA embeddings
│   ├── generate_publication_figures.py
│   ├── generate_semi_synthetic.py    # Generate semi-synthetic benchmark data
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
