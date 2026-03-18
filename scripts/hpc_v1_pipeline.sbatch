#!/bin/bash
#SBATCH --job-name=v1_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/reference_mapping_%j.log

cd /home/booka/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env

DATA=/scratch/chaunzt1/stagebridge

# Step 3: Reference mapping (HLCA + LuCA)
echo "=== Step 3: Reference Mapping ==="
python -m stagebridge.pipelines.run_reference \
    --data-root $DATA \
    --hpc \
    --luca $DATA/references/luca/luca_core_atlas.h5ad

# Step 4: Spatial benchmark (Tangram/DestVI/TACCO)
echo "=== Step 4: Spatial Benchmark ==="
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
    --spatial $DATA/processed/luad_evo/spatial_merged.h5ad \
    --output_dir $DATA/processed/luad_evo/spatial_benchmark

# Step 5: Complete data prep (canonical format)
echo "=== Step 5: Complete Data Prep ==="
python -m stagebridge.pipelines.complete_data_prep \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
    --spatial $DATA/processed/luad_evo/spatial_merged.h5ad \
    --wes $DATA/processed/luad_evo/wes_features.parquet \
    --spatial_backend_dir $DATA/processed/luad_evo/spatial_benchmark \
    --output_dir $DATA/processed/luad_evo/canonical

# Step 6: Training (both semi-synthetic + real)
echo "=== Step 6: V1 Complete Training ==="
python -m stagebridge.pipelines.run_v1_complete \
    --data_dir $DATA/processed/luad_evo/canonical \
    --output_dir $DATA/runs/v1_complete

echo "=== Pipeline Complete ==="
