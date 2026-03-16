#!/bin/bash
#SBATCH --job-name=stagebridge_v1
#SBATCH --output=logs/stagebridge_%j.out
#SBATCH --error=logs/stagebridge_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

################################################################################
# StageBridge V1 Full Pipeline - HPC Execution
################################################################################

echo "=========================================="
echo "StageBridge V1 Full Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules (adjust for your HPC system)
module purge
module load cuda/12.1
module load gcc/11.2.0

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stagebridge

# Verify GPU
echo "GPU Info:"
nvidia-smi
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directories
mkdir -p logs
mkdir -p outputs/luad_v1_comprehensive
mkdir -p data/raw
mkdir -p data/processed/luad
mkdir -p data/references

# Change to project directory
cd $SLURM_SUBMIT_DIR

echo "=========================================="
echo "Step 0: Download Reference Atlases"
echo "=========================================="
python -c "
from stagebridge.pipelines.complete_data_prep import download_reference_atlases
from pathlib import Path

references = download_reference_atlases(
    output_dir='data/references',
    download_hlca=True,
    download_luca=True,
)
print(f' HLCA: {references[\"hlca\"]}')
print(f' LuCA: {references[\"luca\"]}')
"

echo ""
echo "=========================================="
echo "Step 1: Data Preprocessing"
echo "=========================================="
python stagebridge/pipelines/complete_data_prep.py \
    --raw_dir data/raw \
    --output_dir data/processed/luad \
    --reference_dir data/references

echo ""
echo "=========================================="
echo "Step 2: Spatial Backend Benchmark"
echo "=========================================="
python stagebridge/pipelines/run_spatial_benchmark.py \
    --snrna data/processed/luad/snrna_merged.h5ad \
    --spatial data/processed/luad/spatial_merged.h5ad \
    --output_dir outputs/luad_v1_comprehensive/spatial_benchmark \
    --backends tangram destvi tacco

echo ""
echo "=========================================="
echo "Step 3: Model Training (All Folds)"
echo "=========================================="
for fold in {0..4}; do
    echo ""
    echo "--- Training Fold $fold ---"
    python stagebridge/pipelines/run_v1_full.py \
        --data_dir data/processed/luad \
        --fold $fold \
        --n_epochs 50 \
        --batch_size 32 \
        --output_dir outputs/luad_v1_comprehensive/training/fold_$fold \
        --niche_encoder transformer \
        --use_set_encoder \
        --use_wes \
        --save_attention
done

echo ""
echo "=========================================="
echo "Step 4: Ablation Suite (All 8 Ablations)"
echo "=========================================="
python stagebridge/pipelines/run_ablations.py \
    --data_dir data/processed/luad \
    --output_dir outputs/luad_v1_comprehensive/ablations \
    --n_folds 5 \
    --n_epochs 50 \
    --batch_size 32

echo ""
echo "=========================================="
echo "Step 5: Analysis & Figure Generation"
echo "=========================================="
python -c "
import sys
sys.path.insert(0, '.')

from pathlib import Path
import pandas as pd
import torch

# Load trained model
from stagebridge.pipelines.run_v1_full import StageBridgeV1Full
from stagebridge.data.loaders import get_dataloader
from stagebridge.analysis.transformer_analysis import generate_transformer_report
from stagebridge.analysis.biological_interpretation import (
    InfluenceTensorExtractor,
    extract_pathway_signatures,
    visualize_niche_influence,
    generate_biological_summary,
)

print('Loading model and data...')
model_path = Path('outputs/luad_v1_comprehensive/training/fold_0/best_model.pt')
model = StageBridgeV1Full(
    latent_dim=32,
    niche_encoder_type='transformer',
    use_set_encoder=True,
    use_wes=True,
)
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

test_loader = get_dataloader(
    data_dir='data/processed/luad',
    fold=0,
    split='test',
    batch_size=32,
    latent_dim=32,
)

# Transformer analysis
print('Generating transformer analysis...')
generate_transformer_report(
    model=model,
    test_loader=test_loader,
    output_dir=Path('outputs/luad_v1_comprehensive/transformer_analysis'),
)

# Biological interpretation
print('Extracting biological insights...')
cells_df = pd.read_parquet('data/processed/luad/cells.parquet')
neighborhoods_df = pd.read_parquet('data/processed/luad/neighborhoods.parquet')

extractor = InfluenceTensorExtractor(model, device='cpu')
influence_df = extractor.compute_influence_tensor(test_loader, cell_type_mapping={})

pathway_df = extract_pathway_signatures(neighborhoods_df)

visualize_niche_influence(
    influence_df,
    output_path=Path('outputs/luad_v1_comprehensive/biology/niche_influence.png'),
)

generate_biological_summary(
    influence_df,
    pathway_df,
    output_dir=Path('outputs/luad_v1_comprehensive/biology'),
)

print(' Analysis complete!')
"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Output directory: outputs/luad_v1_comprehensive"
echo ""
echo "Results:"
echo "  - Training: outputs/luad_v1_comprehensive/training/"
echo "  - Ablations: outputs/luad_v1_comprehensive/ablations/"
echo "  - Analysis: outputs/luad_v1_comprehensive/transformer_analysis/"
echo "  - Biology: outputs/luad_v1_comprehensive/biology/"
echo ""
