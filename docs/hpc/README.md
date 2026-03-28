# StageBridge HPC Guide

Complete guide for running StageBridge on High Performance Computing clusters using Snakemake.

---

## Quick Start

```bash
# Dry run (see what would execute)
snakemake -n --profile workflow/slurm

# Full execution with SLURM
snakemake --profile workflow/slurm --jobs 20
```

---

## Prerequisites

### Required Data

Download from GEO and place in `$DATA/raw/`:

```bash
# snRNA-seq (GSE308103)
wget -O $DATA/raw/GSE308103_RAW.tar \
    "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE308nnn/GSE308103/suppl/GSE308103_RAW.tar"

# Visium spatial (GSE307534)
wget -O $DATA/raw/GSE307534_RAW.tar \
    "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307534/suppl/GSE307534_RAW.tar"

# WES (GSE307529)
wget -O $DATA/raw/GSE307529_RAW.tar \
    "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307529/suppl/GSE307529_RAW.tar"
```

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x V100 16GB | 4x A100/H100 |
| RAM | 64GB | 128GB |
| CPU | 8 cores | 16+ cores |
| Storage | 200GB | 500GB |

---

## Environment Setup

```bash
# Clone repository
git clone https://github.com/SecondBook5/StageBridge.git
cd StageBridge

# Create conda environment
micromamba env create -f environment.yml
micromamba activate stagebridge

# Install package
pip install -e ".[all]"

# Set data root
export STAGEBRIDGE_DATA_ROOT=/path/to/your/data
```

---

## Pipeline Execution with Snakemake

### Configuration

Edit `workflow/config.yaml`:

```yaml
data_root: "/scratch/your_username/stagebridge"
```

### Run Pipeline

```bash
# Full pipeline
snakemake --profile workflow/slurm --jobs 20

# Specific target
snakemake publication_figures --profile workflow/slurm

# Dry run
snakemake -n --profile workflow/slurm
```

### Pipeline DAG

```
hlca_mapping ──┬──> add_cell_type_labels ──> validate_markers
               │              │
               │              v
               └──> fuse_embeddings   spatial_backend_sample (448 jobs)
                          │                    │
luca_mapping ─────────────┘                    v
                               spatial_comparison ──> data_preparation
                                                           │
                                           ┌───────────────┼───────────────┐
                                           v               v               v
                                    training (15×)   baselines (60×)     hpo
                                           │               │
                                           v               v
                                     aggregate_cv    aggregate_baselines
                                           │               │
                                           └───────┬───────┘
                                                   v
                                            ablation (14×)
                                                   │
                                                   v
                                         publication_figures
```

### Monitor Jobs

```bash
# Check queue
squeue -u $USER

# Watch progress
watch -n 30 'squeue -u $USER'

# View Snakemake logs
tail -f $DATA/runs/logs/*.log
```

---

## Troubleshooting

### CUDA Issues

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -c "import torch; print(torch.cuda.is_available())"
```

### Common Errors

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Snakemake profiles set appropriate batch sizes |
| `Module not found` | Activate conda environment |
| `Job killed` | Check `workflow/slurm/config.yaml` for resource settings |

---

## Reference

- **Workflow config**: `workflow/config.yaml`
- **SLURM profile**: `workflow/slurm/config.yaml`
- **Pipeline README**: `workflow/README.md`
