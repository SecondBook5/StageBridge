#!/bin/bash
#SBATCH --job-name=sb_ablate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/ablations_%j.log

# =============================================================================
# Stage 6: Ablation Studies
# =============================================================================
#
# Runs comprehensive ablation studies across 5-fold CV:
#
# Tier 1 Ablations (Core Architecture):
# 1. full_model (baseline)
# 2. no_niche (remove niche conditioning)
# 3. no_wes (remove WES regularization)
# 4. pooled_niche (mean pool instead of transformer)
# 5. hlca_only (no LuCA reference)
# 6. luca_only (no HLCA reference)
# 7. deterministic (no stochastic dynamics)
# 8. flat_hierarchy (no Set Transformer)
#
# Tier 2 Ablations (Fusion & SSL):
# 9. learned_fusion
# 10. weighted_fusion
# 11. equal_loss_weights
# 12. no_auxiliary_losses
# 13. no_calibration
# 14. temperature_calibration
#
# Baseline Ladder (per doctrine):
# 1. MeanPoolMLP (weakest floor)
# 2. MaxPoolMLP
# 3. DeepSets
# 4. SetTransformer
# 5. HierarchicalSetTransformer
# 6. GraphSAGE
# 7. GAT
# 8. StageBridge (full model)
#
# Produces:
# - table3_main_results.csv (Nature Methods Table 3)
# - table3_main_results.tex (LaTeX)
# - figure7_ablation_heatmap.png
# - statistical_comparisons.csv
# - all_results.csv (raw data)
#
# Prerequisites:
# - Stage 5 (Training) completed
#
# Runtime: ~24-48 hours (14 ablations x 5 folds)
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Stage 6: Ablation Studies"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

# Environment setup
cd /home/booka/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env

# Data paths
DATA=/scratch/chaunzt1/stagebridge
CANONICAL=$DATA/processed/luad_evo/canonical
TRAINING_OUTPUT=$DATA/runs/v1_complete
OUTPUT=$DATA/runs/ablations

# GPU setup
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo ""
echo "Configuration:"
echo "  Data: $CANONICAL"
echo "  Training Output: $TRAINING_OUTPUT"
echo "  Ablation Output: $OUTPUT"
echo ""

# Verify prerequisites
echo "Verifying prerequisites..."
if [ ! -f "$TRAINING_OUTPUT/weights/final_model.pt" ]; then
    echo "ERROR: Training not complete. Run Stage 5 first."
    exit 1
fi
echo "  [OK] Training complete"

# Show GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

mkdir -p "$OUTPUT"

# Define ablation tiers
TIER1_ABLATIONS="full_model no_niche no_wes pooled_niche hlca_only luca_only deterministic flat_hierarchy"
TIER2_ABLATIONS="learned_fusion weighted_fusion equal_loss_weights no_auxiliary_losses"

echo "Tier 1 Ablations: $TIER1_ABLATIONS"
echo "Tier 2 Ablations: $TIER2_ABLATIONS"
echo ""

# Run ablation suite
echo "Running ablation studies..."
echo "This will take 24-48 hours for full 5-fold CV..."
echo ""

python -m stagebridge.pipelines.run_ablations \
    --data_dir "$CANONICAL" \
    --output_dir "$OUTPUT" \
    --n_folds 5 \
    --batch_size 64 \
    --n_epochs 50 \
    --lr 1e-3 \
    --ablations $TIER1_ABLATIONS $TIER2_ABLATIONS

# Verify outputs
echo ""
echo "Verifying outputs..."
EXPECTED_FILES=(
    "$OUTPUT/all_results.csv"
    "$OUTPUT/table3_main_results.csv"
    "$OUTPUT/table3_main_results.tex"
    "$OUTPUT/figure7_ablation_heatmap.png"
    "$OUTPUT/statistical_comparisons.csv"
)

ALL_OK=true
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  [OK] $f ($SIZE)"
    else
        echo "  [MISSING] $f"
        ALL_OK=false
    fi
done

# Show results summary
if [ -f "$OUTPUT/table3_main_results.csv" ]; then
    echo ""
    echo "Table 3 Preview (Main Results):"
    echo "--------------------------------"
    head -15 "$OUTPUT/table3_main_results.csv" | column -t -s','
fi

# Show statistical comparisons
if [ -f "$OUTPUT/statistical_comparisons.csv" ]; then
    echo ""
    echo "Significant Differences from Full Model (p < 0.05):"
    python -c "
import pandas as pd
df = pd.read_csv('$OUTPUT/statistical_comparisons.csv')
sig = df[df['significant'] == True]
for _, row in sig.iterrows():
    print(f\"  {row['ablation']} ({row['metric']}): d={row['cohens_d']:.3f}, p={row['p_value']:.4f}\")
if len(sig) == 0:
    print('  No significant differences found')
"
fi

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Some expected outputs are missing"
fi

echo ""
echo "=============================================="
echo "Stage 6 Complete: $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - $OUTPUT/table3_main_results.csv (Nature Methods Table 3)"
echo "  - $OUTPUT/table3_main_results.tex (LaTeX)"
echo "  - $OUTPUT/figure7_ablation_heatmap.png"
echo "  - $OUTPUT/statistical_comparisons.csv"
echo "  - $OUTPUT/all_results.csv"
echo ""
echo "Next: Stage 7 (Publication Figures)"
