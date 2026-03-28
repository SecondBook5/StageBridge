# Scripts Directory

Utility scripts for StageBridge pipeline execution, visualization, and development.

**Note:** For HPC execution, use Snakemake (see `workflow/README.md`), not these scripts directly.

## Directory Structure

```
scripts/
├── dev/                    # Development utilities
│   ├── add_*_cells.py      # Notebook modification tools
│   ├── benchmark_*.py      # Performance benchmarking
│   └── audit_*.py          # Code auditing tools
├── viz/                    # Advanced visualization scripts
└── *.py                    # Main utility scripts
```

## Main Scripts

| Script | Purpose |
|--------|---------|
| `add_ensembl_ids.py` | Add ENSEMBL gene IDs to AnnData |
| `generate_plots.py` | Unified plot generation (individual + multi-panel) |
| `generate_master_notebook.py` | Generate master Jupyter notebook |
| `label_pipeline.py` | Unified label repair pipeline |
| `run_baseline_evaluation.py` | Evaluate baseline models |
| `run_permutation_test.py` | Statistical permutation testing |
| `check_figure_completeness.py` | Verify all figures generated |
| `plot_transformer_architecture.py` | Generate architecture diagrams |

## HPC Execution

**Use Snakemake** for HPC execution (not sbatch scripts):

```bash
# Dry run
snakemake -n --profile workflow/slurm

# Full execution
snakemake --profile workflow/slurm --jobs 20
```

See `workflow/README.md` for detailed HPC documentation.

## Usage Examples

```bash
# Generate all plots
python scripts/generate_plots.py --mode both --data auto

# Run label pipeline
python scripts/label_pipeline.py all

# Check figure completeness
python scripts/check_figure_completeness.py
```
