# StageBridge HPC Guide

Complete guide for running StageBridge on High Performance Computing clusters.

**Contents:**
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Pipeline Execution](#pipeline-execution)
4. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Data

Download from GEO and place in `data/raw/`:

```bash
# snRNA-seq (GSE308103)
wget -O data/raw/GSE308103_RAW.tar \
    "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE308nnn/GSE308103/suppl/GSE308103_RAW.tar"

# Visium spatial (GSE307534)
wget -O data/raw/GSE307534_RAW.tar \
    "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307534/suppl/GSE307534_RAW.tar"

# WES (GSE307529)
wget -O data/raw/GSE307529_RAW.tar \
    "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307529/suppl/GSE307529_RAW.tar"
```

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x V100 16GB | 4x A100 40GB |
| RAM | 64GB | 128GB |
| CPU | 8 cores | 16+ cores |
| Storage | 200GB | 500GB |
| Time | 24h | 48-72h |

---

## Environment Setup

### Iris HPC (Miniforge)

Iris uses `miniforge3` (not Anaconda due to licensing).

```bash
# SSH to Iris
ssh your_username@iris.mskcc.org

# Load miniforge module
module load miniforge3

# Create environment (use /data/ for large envs)
conda env create -f envs/environment.yaml \
    --prefix /data/your_labname/envs/stagebridge

# Activate
conda activate /data/your_labname/envs/stagebridge

# Register Jupyter kernel
python -m ipykernel install --user --name stagebridge
```

### Generic HPC (Conda)

```bash
# Create environment
conda env create -f envs/environment.yaml -n stagebridge

# Activate
conda activate stagebridge
```

### Transfer Code to HPC

```bash
# From local machine
rsync -avz --progress \
    --exclude='outputs/' \
    --exclude='data/raw/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    /path/to/StageBridge/ \
    USERNAME@hpc-login:~/StageBridge/
```

---

## Pipeline Execution

### Quick Start

```bash
# Submit entire pipeline with SLURM dependencies
sbatch scripts/hpc_v1_master_pipeline.sbatch
```

### Pipeline Stages

```
Stage 1: HLCA Mapping (6h, 1 GPU)
    ↓
Stage 2: LuCA Mapping (6h, 1 GPU)
    ↓                              Stage 3: Spatial Benchmark (24h, 1 GPU)
    ↓                                  ↓
    +----------------------------------+
                   ↓
Stage 4: Data Preparation (4h, 1 GPU)
                   ↓
Stage 5: Training - SSL + Transition (12h, 4 GPU)
                   ↓
Stage 6: Evaluation (2h, 1 GPU)
                   ↓
Stage 7: Publication Figures (1h, 1 GPU)
```

### Manual Execution

```bash
# Stage 1: HLCA mapping
sbatch scripts/hpc_step1_hlca.sbatch

# Stage 2: LuCA mapping (after Stage 1)
sbatch --dependency=afterok:$HLCA_JOB scripts/hpc_step2_luca.sbatch

# Stage 3: Spatial benchmark (parallel with 1-2)
sbatch scripts/hpc_step3_spatial.sbatch

# Stage 4: Data prep (after 1, 2, 3)
sbatch --dependency=afterok:$LUCA_JOB:$SPATIAL_JOB scripts/hpc_step4_data_prep.sbatch

# Stage 5: Training (after Stage 4)
sbatch --dependency=afterok:$PREP_JOB scripts/hpc_step5_training.sbatch

# Stage 6: Evaluation
sbatch --dependency=afterok:$TRAIN_JOB scripts/hpc_step6_evaluation.sbatch

# Stage 7: Figures
sbatch --dependency=afterok:$EVAL_JOB scripts/hpc_step7_figures.sbatch
```

### Monitor Jobs

```bash
# Check queue
squeue -u $USER

# View logs
tail -f logs/stagebridge_*.log

# Check GPU usage
nvidia-smi
```

---

## Troubleshooting

### CUDA Issues

```bash
# Verify CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues

```bash
# Request more memory in sbatch
#SBATCH --mem=128G

# Or use chunked processing
python -m stagebridge.pipelines.run_reference --hpc --chunk-size 10000
```

### Module Conflicts

```bash
# Clear and reload
module purge
module load miniforge3 cuda/12.4
```

### Common Errors

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size or use `--hpc` flag |
| `Module not found` | Activate conda environment |
| `Permission denied` | Check file permissions on /data/ |
| `Job killed` | Request more time/memory in sbatch |

---

## Reference

- **Pipeline README**: `stagebridge/pipelines/README.md`
- **SLURM scripts**: `scripts/hpc_*.sbatch`
- **Environment file**: `envs/environment.yaml`
