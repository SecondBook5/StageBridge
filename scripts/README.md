# Scripts

Organized scripts for StageBridge data preparation, benchmarking, analysis, and visualization.

## Directory Structure

```
scripts/
├── data/           # Data preparation and preprocessing
├── benchmarks/     # Benchmark creation and baseline comparisons
├── analysis/       # Post-training analysis and paper metrics
├── figures/        # Publication figure generation
├── hpc/            # HPC job submission scripts (sbatch/slurm)
└── _archive/       # Deprecated/one-time scripts
```

## data/ - Data Preparation

| Script | Description |
|--------|-------------|
| `prepare_training_data.py` | **Main entry point** - unified data prep pipeline (reference mapping, feature enrichment, cells.parquet, neighborhoods.parquet) |
| `create_split_manifest.py` | Generate donor-held-out CV splits |
| `precompute_gw_alignment.py` | Precompute Gromov-Wasserstein HLCA-LuCA alignment |
| `map_spatial_to_reference.py` | Map spatial/snRNA to HLCA/LuCA reference spaces |
| `add_luca_labels.py` | Transfer LuCA cell type labels and run LIANA |
| `run_full_data_prep.sh` | Shell wrapper for full pipeline |

**Typical workflow:**
```bash
# 1. Prepare all training data
python scripts/data/prepare_training_data.py \
    --snrna $DATA/snrna.h5ad \
    --spatial $DATA/spatial.h5ad \
    --output-dir $DATA/canonical

# 2. Create CV splits
python scripts/data/create_split_manifest.py \
    --cells $DATA/canonical/cells.parquet \
    --output $DATA/canonical/split_manifest.json
```

## benchmarks/ - Benchmarking

| Script | Description |
|--------|-------------|
| `create_ground_truth.py` | Apply AMICI-style ground truth labels to real data |
| `create_semisynthetic_benchmark.py` | Generate semi-synthetic benchmark with controlled spatial structure |
| `run_external_baseline.py` | Run external methods (moscot, CellRank) for comparison |
| `ablation_gw_vs_concat.py` | Controlled ablation: GW fusion vs concatenation |

## analysis/ - Post-Training Analysis

| Script | Description |
|--------|-------------|
| `run_interpretation.py` | Full interpretation pipeline (ablation, attention, networks, trajectories) |
| `compute_paper_numbers.py` | **Single source of truth** for all paper statistics |
| `compute_drift_alignment.py` | Drift alignment metrics (cosine similarity) |
| `compute_entropy.py` | Attention entropy metrics |
| `run_genomic_interpretation.py` | Genomic/WES interpretation analysis |

**After training:**
```bash
# Run full interpretation
python scripts/analysis/run_interpretation.py \
    --checkpoint results/best_checkpoint.pt \
    --data-dir $DATA/canonical \
    --output-dir results/interpretation \
    --all

# Compute paper numbers
python scripts/analysis/compute_paper_numbers.py \
    --data-dir $DATA/canonical \
    --output paper_numbers.json
```

## figures/ - Visualization

| Script | Description |
|--------|-------------|
| `generate_poster_panels.py` | Individual poster panels with proper scaling |
| `generate_advanced_figures.py` | Advanced visualizations (manifolds, trajectories) |
| `generate_eda_figures.py` | Exploratory data analysis plots |
| `generate_commot_figures.py` | COMMOT comparison figures |
| `generate_liana_figures.py` | LIANA comparison figures |
| `plot_alignment_umap.py` | HLCA/LuCA alignment UMAPs |

## hpc/ - HPC Job Scripts

SLURM/sbatch scripts for cluster execution. See individual scripts for resource requirements.

| Script | Description |
|--------|-------------|
| `run_hpo_*.sh` | HPO with Optuna |
| `run_training_*.sh` | Full training runs |
| `*.sbatch` | Individual job scripts |

## _archive/ - Deprecated

One-time fixes and superseded scripts. Kept for reference only.
