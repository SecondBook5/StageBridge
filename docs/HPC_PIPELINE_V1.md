# StageBridge V1 HPC Pipeline - Complete Guide

## Overview

This document describes the complete HPC pipeline for training StageBridge V1 on the LUAD evolutionary dataset. The pipeline is designed for Nature Methods publication quality.

## Quick Start

```bash
# Submit the entire pipeline with SLURM dependencies
sbatch scripts/hpc_v1_master_pipeline.sbatch
```

This will submit all 7 stages with proper dependencies, running them in the optimal order.

## Pipeline Architecture

```
                    PREPARATION (Prerequisites)
                    ===========================
                              |
         +--------------------+--------------------+
         |                                         |
         v                                         v
 +---------------+                         +---------------+
 | Stage 1: HLCA |                         | Stage 3:      |
 | Mapping       |                         | Spatial       |
 | (6h, 1 GPU)   |                         | Benchmark     |
 +-------+-------+                         | (24h, 1 GPU)  |
         |                                 +-------+-------+
         v                                         |
 +---------------+                                 |
 | Stage 2: LuCA |                                 |
 | Mapping       |                                 |
 | (6h, 1 GPU)   |                                 |
 +-------+-------+                                 |
         |                                         |
         +--------------------+--------------------+
                              |
                              v
                    +-------------------+
                    | Stage 4: Data     |
                    | Preparation       |
                    | (4h, 1 GPU)       |
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    | Stage 5: Training |
                    | SSL + Transition  |
                    | (24h, 4 GPUs)     |
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    | Stage 6: Ablation |
                    | 5-fold CV         |
                    | (48h, 4 GPUs)     |
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    | Stage 7: Figures  |
                    | Publication       |
                    | (4h, 1 GPU)       |
                    +-------------------+
```

## Prerequisites

Before running the pipeline, ensure these files exist:

```
$DATA/
├── processed/luad_evo/
│   ├── snrna_qc_normalized_with_ensg.h5ad  # From add_ensembl_ids.py
│   └── spatial_merged.h5ad                  # From run_data_prep.py
└── references/
    ├── hlca/
    │   ├── hlca_reference.h5ad             # From download_references.py
    │   └── hub_cache/                       # scANVI model from HuggingFace
    └── luca/
        ├── luca_core_atlas.h5ad            # CORE atlas (not Extended!)
        └── retrained_model/scanvi_model/    # From retrain_luca.sbatch
```

### Preparation Commands

```bash
# 1. Data preparation (if not done)
python -m stagebridge.pipelines.run_data_prep --data-root $DATA --spatial-merge-only

# 2. Add ENSG IDs (required for model-based mapping)
python scripts/add_ensembl_ids.py \
    --query $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
    --hlca $DATA/references/hlca/hlca_reference.h5ad \
    --output $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad

# 3. Download HLCA (if needed)
python -m stagebridge.pipelines.download_references --download_hlca --output_dir $DATA/references

# 4. Retrain LuCA model (if needed)
sbatch scripts/retrain_luca.sbatch
```

## Stage Details

### Stage 1: HLCA Reference Mapping

**Script:** `scripts/hpc_step1_hlca_mapping.sbatch`

**Purpose:** Map query cells to HLCA (healthy lung) reference space

**Method:** Model-based scArches surgery using pretrained scANVI model

**Outputs:**
- `hlca_embedding.parquet` - L2-normalized latents (30 dims)
- `hlca_mapping/hlca_labels.parquet` - Cell type predictions

**Resources:** 1 GPU, 256GB RAM, 6 hours

### Stage 2: LuCA Reference Mapping

**Script:** `scripts/hpc_step2_luca_mapping.sbatch`

**Purpose:** Map query cells to LuCA (lung cancer) reference space

**Method:** Model-based scArches surgery using retrained scANVI model

**Outputs:**
- `luca_embedding.parquet` - L2-normalized latents (10 dims)
- `fused_embedding.parquet` - Concatenated HLCA+LuCA (40 dims)
- `reference_confidence.parquet` - Calibrated confidence scores
- `cell_types.parquet` - HLCA + LuCA cell type predictions

**Resources:** 1 GPU, 256GB RAM, 6 hours

**Note:** May require pandas 1.5.x environment. If job fails with BlockPlacement error, create a compatible environment:
```bash
conda create -n luca_compat python=3.11
pip install pandas==1.5.3 scvi-tools torch --index-url https://download.pytorch.org/whl/cu124
```

### Stage 3: Spatial Backend Benchmark

**Script:** `scripts/hpc_step3_spatial_benchmark.sbatch`

**Purpose:** Compare spatial deconvolution backends, select best

**Backends:**
1. Tangram (scvi-tools integration)
2. DestVI (probabilistic)
3. TACCO (transfer learning)
4. Cell2location (reference-based)

**Outputs:**
- `backend_comparison.json` - Benchmark metrics
- `cell_type_proportions.parquet` - Selected backend results
- `{tangram,destvi,tacco,cell2location}/` - Per-backend results

**Resources:** 1 GPU, 256GB RAM, 24 hours

**Note:** Runs in PARALLEL with Stages 1-2

### Stage 4: Complete Data Preparation

**Script:** `scripts/hpc_step4_complete_data_prep.sbatch`

**Purpose:** Build canonical training format

**Dependencies:** Stages 2 AND 3 must complete first

**Outputs:**
- `cells.parquet` - All cells with latents, cell types, WES features
- `neighborhoods.parquet` - 9-token niche structure
- `stage_edges.parquet` - Valid progression transitions
- `split_manifest.json` - 5-fold donor-held-out CV
- `feature_spec.yaml` - Documentation

**Resources:** 1 GPU, 256GB RAM, 4 hours

### Stage 5: Full Model Training

**Script:** `scripts/hpc_step5_training.sbatch`

**Purpose:** Train StageBridge V1 (SSL + Transition)

**Training Stages:**
1. SSL Pretraining (50 epochs)
   - Masked receiver reconstruction (70% weight)
   - Ranking (10%), provider consistency (10%)
   - Coordinate corruption (5%), group relation (5%)
2. Transition Modeling (100 epochs)
   - Flow matching for progression dynamics

**Outputs:**
- `weights/final_model.pt` - Final checkpoint
- `weights/best_model_*.pt` - Per-dataset best
- `results.json` - Training metrics
- `figures/fig1-4_*.png` - Training figures

**Resources:** 4 GPUs, 512GB RAM, 24 hours

### Stage 6: Ablation Studies

**Script:** `scripts/hpc_step6_ablations.sbatch`

**Purpose:** Run comprehensive ablations (5-fold CV)

**Ablations:**
- Tier 1: Architecture (full, no_niche, no_wes, pooled_niche, hlca_only, luca_only, deterministic, flat_hierarchy)
- Tier 2: Fusion/SSL (learned_fusion, weighted_fusion, equal_loss_weights, no_auxiliary_losses)

**Baseline Ladder:**
1. MeanPoolMLP (weakest floor)
2. MaxPoolMLP
3. DeepSets
4. SetTransformer
5. HierarchicalSetTransformer
6. GraphSAGE
7. GAT
8. StageBridge (full)

**Outputs:**
- `table3_main_results.csv` - Nature Methods Table 3
- `table3_main_results.tex` - LaTeX version
- `figure7_ablation_heatmap.png` - Ablation heatmap
- `statistical_comparisons.csv` - Paired t-tests

**Resources:** 4 GPUs, 512GB RAM, 48 hours

### Stage 7: Publication Figures

**Script:** `scripts/hpc_step7_figures.sbatch`

**Purpose:** Generate all publication figures

**Main Figures:**
- Figure 1: Architecture (TikZ)
- Figure 2: Training curves
- Figure 3: Trajectory visualization
- Figure 4: Stage embeddings
- Figure 5: Ablation heatmap
- Figure 6: Baseline comparison
- Figure 7: Biology validation

**Supplementary:**
- S1: Reference mapping diagnostics
- S2: Spatial backend comparison
- S3: HPO optimization history
- S4: CV fold stability

**Resources:** 1 GPU, 128GB RAM, 4 hours

## Manual Execution

To run stages individually with explicit dependencies:

```bash
# Stage 1: HLCA Mapping
JOB1=$(sbatch --parsable scripts/hpc_step1_hlca_mapping.sbatch)
echo "HLCA job: $JOB1"

# Stage 2: LuCA Mapping (after HLCA)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/hpc_step2_luca_mapping.sbatch)
echo "LuCA job: $JOB2"

# Stage 3: Spatial Benchmark (parallel)
JOB3=$(sbatch --parsable scripts/hpc_step3_spatial_benchmark.sbatch)
echo "Spatial job: $JOB3"

# Stage 4: Data Prep (after ref + spatial)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3 scripts/hpc_step4_complete_data_prep.sbatch)
echo "Data prep job: $JOB4"

# Stage 5: Training (after data prep)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 scripts/hpc_step5_training.sbatch)
echo "Training job: $JOB5"

# Stage 6: Ablations (after training)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 scripts/hpc_step6_ablations.sbatch)
echo "Ablations job: $JOB6"

# Stage 7: Figures (after ablations)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB6 scripts/hpc_step7_figures.sbatch)
echo "Figures job: $JOB7"
```

## Monitoring

```bash
# Watch job queue
watch -n 30 'squeue -u $USER'

# Check specific job
squeue -j <JOBID>

# View log
tail -f /home/booka/StageBridge/logs/<jobname>_<jobid>.log

# Cancel all
scancel -u $USER
```

## Expected Outputs

After successful completion:

```
$DATA/
├── processed/luad_evo/
│   ├── reference_geometry/
│   │   ├── hlca_embedding.parquet
│   │   ├── luca_embedding.parquet
│   │   ├── fused_embedding.parquet
│   │   ├── reference_confidence.parquet
│   │   ├── cell_types.parquet
│   │   └── diagnostics_report.json
│   ├── spatial_benchmark/
│   │   ├── backend_comparison.json
│   │   ├── cell_type_proportions.parquet
│   │   └── {tangram,destvi,tacco,cell2location}/
│   └── canonical/
│       ├── cells.parquet
│       ├── neighborhoods.parquet
│       ├── stage_edges.parquet
│       ├── split_manifest.json
│       └── feature_spec.yaml
└── runs/
    ├── v1_complete/
    │   ├── weights/
    │   │   ├── final_model.pt
    │   │   └── best_model_*.pt
    │   ├── figures/
    │   ├── results.json
    │   └── config.json
    ├── ablations/
    │   ├── table3_main_results.csv
    │   ├── table3_main_results.tex
    │   ├── figure7_ablation_heatmap.png
    │   └── statistical_comparisons.csv
    └── publication_figures/
        ├── main/
        ├── supplementary/
        └── tikz/
```

## Estimated Timeline

| Stage | Duration | Cumulative |
|-------|----------|------------|
| 1. HLCA Mapping | 6h | 6h |
| 2. LuCA Mapping | 6h | 12h |
| 3. Spatial Benchmark | 24h | 24h (parallel) |
| 4. Data Preparation | 4h | 28h |
| 5. Training | 24h | 52h |
| 6. Ablations | 48h | 100h |
| 7. Figures | 4h | 104h |

**Total:** ~104 hours (~4.3 days) wall time

Note: Stages 1-3 overlap, so actual wall time is less.

## Troubleshooting

### LuCA Model Fails with BlockPlacement Error

```bash
# Create pandas 1.5.x compatible environment
conda create -n luca_compat python=3.11
conda activate luca_compat
pip install pandas==1.5.3 scvi-tools torch --index-url https://download.pytorch.org/whl/cu124
```

### GPU Not Detected

```bash
# Verify CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# Reinstall PyTorch if needed
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### Out of Memory

- Reduce batch_size in training scripts
- Use --hpc flag for chunked processing
- Request more memory in SBATCH header

### Job Dependency Failed

```bash
# Check failed job
sacct -j <FAILED_JOBID> --format=JobID,State,ExitCode,Reason

# Resubmit from failed stage
sbatch scripts/hpc_step<N>_*.sbatch
# Then resubmit subsequent stages with new dependency
```

## Doctrine Compliance

This pipeline follows StageBridge doctrine:

1. **Cells as primary learning units** - not lesions, not bags
2. **Receiver-centered niche modeling** - ReceiverCenteredNicheEncoder
3. **Dual-reference structure** - HLCA + LuCA geometry
4. **Representation learning first** - SSL pretraining (70% weight)
5. **Progression/transition as downstream** - Flow matching
6. **V1 scope discipline** - No v2 features (phase portraits, hypergraphs)

## Contact

For issues with this pipeline, check:
1. Logs in `/home/booka/StageBridge/logs/`
2. CLAUDE.md for debugging guidance
3. docs/PROJECT_DOCTRINE.md for conceptual alignment
