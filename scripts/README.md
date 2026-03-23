# Scripts Directory

Organized scripts for StageBridge pipeline execution, visualization, and development.

## Directory Structure

```
scripts/
├── hpc/                    # HPC/SLURM batch scripts
│   ├── hpc_step*.sbatch    # Pipeline steps (1-7)
│   ├── hpc_spatial_*.sbatch # Spatial backend runners
│   └── *.slurm             # Alternative SLURM scripts
├── dev/                    # Development utilities
│   ├── add_*_cells.py      # Notebook modification tools
│   ├── benchmark_*.py      # Performance benchmarking
│   └── audit_*.py          # Code auditing tools
├── viz/                    # Advanced visualization scripts
│   ├── atlas_umap_figure.py
│   └── generate_advanced_figures.py
└── *.py                    # Main pipeline scripts
```

## Main Scripts

| Script | Purpose |
|--------|---------|
| `add_ensembl_ids.py` | Add ENSEMBL gene IDs to AnnData |
| `generate_plots.py` | Unified plot generation (individual + multi-panel) |
| `generate_master_notebook.py` | Generate master Jupyter notebook |
| `label_pipeline.py` | Unified label repair pipeline (7 subcommands) |
| `run_baseline_evaluation.py` | Evaluate baseline models |
| `run_permutation_test.py` | Statistical permutation testing |
| `check_figure_completeness.py` | Verify all figures generated |
| `plot_transformer_architecture.py` | Generate architecture diagrams |

## HPC Scripts (`hpc/`)

Pipeline steps run in order:
1. `hpc_step1_hlca_mapping.sbatch` - HLCA reference mapping
2. `hpc_step2_luca_mapping.sbatch` - LuCA reference mapping
3. `hpc_step2b_fuse.sbatch` - Fuse dual embeddings
4. `hpc_step3_spatial_benchmark.sbatch` - Run spatial backends
5. `hpc_step4_complete_data_prep.sbatch` - Finalize data
6. `hpc_step5_training.sbatch` - Train model
7. `hpc_step6_ablations.sbatch` - Run ablation studies
8. `hpc_step7_figures.sbatch` - Generate publication figures

Spatial backend scripts (run in parallel):
- `hpc_spatial_tangram.sbatch`
- `hpc_spatial_destvi.sbatch`
- `hpc_spatial_tacco.sbatch`
- `hpc_spatial_cell2location.sbatch`

## Usage Examples

```bash
# Generate all plots
python scripts/generate_plots.py --mode both --data auto

# Run label pipeline
python scripts/label_pipeline.py all

# Check figure completeness
python scripts/check_figure_completeness.py

# Submit HPC pipeline
sbatch scripts/hpc/hpc_v1_master_pipeline.sbatch
```

## Consolidated Scripts

Several scripts have been consolidated for clarity:

- `generate_plots.py` replaces:
  - `extract_and_plot.py`
  - `generate_individual_plots.py`
  - `regenerate_publication_figures.py`

- `label_pipeline.py` replaces:
  - `build_cohort_manifest.py`
  - `generate_label_reports.py`
  - `evaluate_label_support.py`
  - `refine_labels.py`
  - `run_clonal_backend.py`
  - `run_cna_backend.py`
  - `run_phylogeny_backend.py`
