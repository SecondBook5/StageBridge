# StageBridge V1 - HPC Execution Guide

Complete guide for running StageBridge on High Performance Computing clusters.

---

## Prerequisites

### Required Data Files

Download these datasets from GEO and place in `data/raw/`:

```bash
# On your local machine, download:
# 1. snRNA-seq: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308103
wget -O data/raw/GSE308103_RAW.tar "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE308nnn/GSE308103/suppl/GSE308103_RAW.tar"

# 2. Visium spatial: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307534
wget -O data/raw/GSE307534_RAW.tar "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307534/suppl/GSE307534_RAW.tar"

# 3. WES: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307529
wget -O data/raw/GSE307529_RAW.tar "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307529/suppl/GSE307529_RAW.tar"
```

### System Requirements

- **GPU**: 1x NVIDIA GPU with 16GB+ VRAM (V100, A100, RTX 3090/4090)
- **Memory**: 128GB RAM recommended
- **CPU**: 16+ cores
- **Storage**: 500GB+ for data and outputs
- **Time**: 48-72 hours for full pipeline

---

## Quick Start

### 1. Transfer Repository to HPC

```bash
# On your local machine
rsync -avz --progress \
    --exclude='outputs/' \
    --exclude='data/raw/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    /home/booka/projects/StageBridge/ \
    USERNAME@hpc-login.university.edu:~/StageBridge/
```

### 2. Transfer Data Files

```bash
# Transfer raw data (if downloaded locally)
rsync -avz --progress \
    data/raw/ \
    USERNAME@hpc-login.university.edu:~/StageBridge/data/raw/
```

### 3. SSH to HPC

```bash
ssh USERNAME@hpc-login.university.edu
cd ~/StageBridge
```

### 4. Setup Environment

```bash
# Make setup script executable
chmod +x hpc_setup.sh

# Run setup (takes ~15 minutes)
./hpc_setup.sh

# Activate environment
conda activate stagebridge
```

### 5. Update SLURM Scripts

Edit job parameters in `run_hpc_full.slurm`:

```bash
# Update these lines:
#SBATCH --mail-user=YOUR_EMAIL@example.com  # Your email
#SBATCH --partition=gpu                      # Your GPU partition name
#SBATCH --account=YOUR_ACCOUNT               # Your account/allocation (if needed)

# Also update module names if different on your system:
module load cuda/12.1    # Check: module avail cuda
module load gcc/11.2.0   # Check: module avail gcc
```

### 6. Run Quick Test (30 minutes)

```bash
# Test that everything works
sbatch run_hpc_test.slurm

# Monitor job
squeue -u $USER
tail -f logs/stagebridge_test_*.out
```

### 7. Run Full Pipeline (48-72 hours)

```bash
# Submit full job
sbatch run_hpc_full.slurm

# Check job status
squeue -u $USER

# Monitor progress
tail -f logs/stagebridge_*.out

# Check GPU usage
ssh <compute-node>
nvidia-smi
```

---

## Pipeline Steps

The full pipeline runs these steps automatically:

### Step 0: Reference Download (~1-2 hours)
- Downloads HLCA (Human Lung Cell Atlas)
- Downloads LuCA (Lung Cancer Atlas)
- Output: `data/references/`

### Step 1: Data Preprocessing (~2-3 hours)
- Extracts raw GEO archives
- Processes snRNA-seq data
- Processes Visium spatial data
- Processes WES features
- Integrates with references
- Generates canonical artifacts
- Output: `data/processed/luad/`

### Step 2: Spatial Backend Benchmark (~2-4 hours)
- Runs Tangram
- Runs DestVI
- Runs TACCO
- Compares performance
- Selects canonical backend
- Output: `outputs/luad_v1_comprehensive/spatial_benchmark/`

### Step 3: Model Training (~15-20 hours)
- Trains full model across 5 folds
- 50 epochs per fold
- Saves attention weights
- Computes metrics (W-distance, MSE, MAE)
- Output: `outputs/luad_v1_comprehensive/training/fold_*/`

### Step 4: Ablation Suite (~20-30 hours)
- Runs 8 ablations × 5 folds = 40 models
- Compares to full model
- Generates comparison tables
- Output: `outputs/luad_v1_comprehensive/ablations/`

### Step 5: Analysis & Figures (~1-2 hours)
- Transformer attention analysis
- Biological interpretation
- Niche influence extraction
- Pathway signatures
- Output: `outputs/luad_v1_comprehensive/transformer_analysis/` and `biology/`

---

##  Monitoring & Debugging

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job <JOB_ID>

# Cancel job
scancel <JOB_ID>
```

### Monitor Output

```bash
# Watch main output
tail -f logs/stagebridge_*.out

# Check errors
tail -f logs/stagebridge_*.err

# Check GPU usage (once job is running)
ssh <compute-node-name>
watch -n 1 nvidia-smi
```

### Common Issues

**1. Out of Memory**
```bash
# Edit run_hpc_full.slurm:
#SBATCH --mem=256G  # Increase memory
# Or reduce batch size:
--batch_size 16
```

**2. GPU Out of Memory**
```bash
# In run_v1_full.py, reduce batch size:
--batch_size 16
# Or use gradient accumulation
```

**3. Job Time Limit**
```bash
# Edit run_hpc_full.slurm:
#SBATCH --time=96:00:00  # Increase to 96 hours
```

**4. Module Not Found**
```bash
# Check available modules
module avail cuda
module avail gcc

# Update module names in SLURM script
```

---

## Expected Outputs

After completion, you should have:

```
outputs/luad_v1_comprehensive/
 spatial_benchmark/
    tangram/
    destvi/
    tacco/
    backend_comparison.json
 training/
    fold_0/
       best_model.pt
       results.json
       training_log.csv
    fold_1/ ... fold_4/
    training_results_all_folds.csv
 ablations/
    full_model/
    no_niche/
    no_wes/
    ... (8 ablations)
    all_results.csv
    table3_main_results.csv
 transformer_analysis/
    attention_patterns.png
    attention_entropy.csv
    transformer_summary.md
 biology/
     niche_influence.png
     biological_summary.md
```

---

##  Download Results

After job completes, download results to local machine:

```bash
# On your local machine
rsync -avz --progress \
    USERNAME@hpc-login.university.edu:~/StageBridge/outputs/luad_v1_comprehensive/ \
    ./outputs/luad_v1_comprehensive/
```

---

##  Advanced Options

### Run Only Specific Steps

```bash
# Skip data preprocessing if already done
# Comment out Step 1 in run_hpc_full.slurm

# Run only training
sbatch -J training_only --wrap="bash -c '
source activate stagebridge
for fold in {0..4}; do
    python stagebridge/pipelines/run_v1_full.py --data_dir data/processed/luad --fold $fold --n_epochs 50 --batch_size 32 --output_dir outputs/training/fold_$fold --niche_encoder transformer --use_set_encoder --use_wes
done
'"
```

### Parallel Training Across Nodes

```bash
# Submit each fold as separate job
for fold in {0..4}; do
    sbatch --job-name=fold_$fold \
           --output=logs/fold_${fold}_%j.out \
           --wrap="source activate stagebridge && python stagebridge/pipelines/run_v1_full.py --data_dir data/processed/luad --fold $fold --n_epochs 50 --batch_size 32 --output_dir outputs/training/fold_$fold --niche_encoder transformer --use_set_encoder --use_wes"
done
```

### Interactive Session (for debugging)

```bash
# Request interactive GPU node
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=4:00:00 --pty bash

# Once on node, activate and test
conda activate stagebridge
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Performance Benchmarks

Expected runtimes on different systems:

| System | GPU | Full Pipeline | Training Only | Test Run |
|--------|-----|---------------|---------------|----------|
| V100 | 1x 32GB | 52 hours | 18 hours | 25 min |
| A100 | 1x 40GB | 38 hours | 12 hours | 18 min |
| RTX 4090 | 1x 24GB | 45 hours | 15 hours | 22 min |

---

## Support

If you encounter issues:

1. Check logs: `tail -f logs/stagebridge_*.{out,err}`
2. Verify GPU: `nvidia-smi`
3. Check environment: `conda list`
4. Test imports: `python -c "import torch, anndata, scanpy"`
5. Check disk space: `df -h`

---

##  Checklist

Before submitting full job:

- [ ] Data files transferred to `data/raw/`
- [ ] Environment setup complete (`conda activate stagebridge` works)
- [ ] SLURM script updated (email, partition, account)
- [ ] Test job completed successfully
- [ ] Logs directory exists: `mkdir -p logs`
- [ ] Sufficient disk space (500GB+)
- [ ] GPU partition accessible
- [ ] Can load required modules

---

## Quick Commands Reference

```bash
# Setup
chmod +x hpc_setup.sh
./hpc_setup.sh
conda activate stagebridge

# Submit jobs
sbatch run_hpc_test.slurm      # Test (30 min)
sbatch run_hpc_full.slurm      # Full (48-72 hours)

# Monitor
squeue -u $USER                 # Job status
tail -f logs/stagebridge_*.out  # Watch progress
scancel <JOB_ID>               # Cancel job

# Download results
rsync -avz USERNAME@hpc:~/StageBridge/outputs/ ./outputs/
```

---

**Ready to run! Start with the test job, then launch the full pipeline.** 
