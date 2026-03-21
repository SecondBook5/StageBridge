#!/bin/bash
#SBATCH --job-name=dual_ref_knn
#SBATCH --output=logs/dual_reference_%j.out
#SBATCH --error=logs/dual_reference_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL

# Dual-reference mapping using k-NN projection for BOTH references
# This ensures methodological consistency between HLCA and LuCA
# Single environment - no pandas version conflicts

set -e

cd $SLURM_SUBMIT_DIR
export DATA=/scratch/chaunzt1/stagebridge
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs

echo "=============================================="
echo "Dual Reference Mapping Pipeline (k-NN mode)"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Method: k-NN projection (consistent for both references)"
echo "=============================================="

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/stagebridge_env

# ============================================
# Step 1: HLCA k-NN Mapping
# ============================================
echo ""
echo "=============================================="
echo "Step 1: HLCA Mapping (k-NN projection)"
echo "Started: $(date)"
echo "=============================================="

python -m stagebridge.pipelines.run_reference \
    --data-root $DATA \
    --hpc \
    --hlca-only \
    --no-model-based \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad

echo "HLCA completed: $(date)"

# ============================================
# Step 2: LuCA k-NN Mapping
# ============================================
echo ""
echo "=============================================="
echo "Step 2: LuCA Mapping (k-NN projection)"
echo "Started: $(date)"
echo "=============================================="

python -m stagebridge.pipelines.run_reference \
    --data-root $DATA \
    --hpc \
    --luca-only \
    --no-model-based \
    --snrna $DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad

echo "LuCA completed: $(date)"

# ============================================
# Step 3: Merge and generate figures
# ============================================
echo ""
echo "=============================================="
echo "Step 3: Merging embeddings and generating figures"
echo "Started: $(date)"
echo "=============================================="

# Same environment throughout (k-NN has no pandas version conflicts)

python << 'PYTHON_SCRIPT'
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import anndata

DATA = "/scratch/chaunzt1/stagebridge"
output_dir = Path(f'{DATA}/processed/luad_evo/reference_geometry')
plots_dir = output_dir / 'plots'
plots_dir.mkdir(exist_ok=True)

STAGE_COLORS = {
    "Normal": "#00BA38", "AAH": "#F8766D", "AIS": "#619CFF",
    "MIA": "#E58700", "LUAD": "#A3A500", "Unknown": "#999999",
}

print('Loading embeddings...')
hlca_emb = pd.read_parquet(output_dir / 'hlca_embedding.parquet')
luca_emb = pd.read_parquet(output_dir / 'luca_embedding.parquet')

print(f'  HLCA: {hlca_emb.shape}')
print(f'  LuCA: {luca_emb.shape}')

# Load confidence data
conf_df = pd.read_parquet(output_dir / 'reference_confidence.parquet')
print(f'  Confidence: {conf_df.shape}')

# Load query for stage labels
print('Loading query for stage labels...')
query = anndata.read_h5ad(f'{DATA}/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad', backed='r')
stage_col = 'stage' if 'stage' in query.obs.columns else 'Stage'
stage_df = query.obs[[stage_col]].reset_index()
stage_df.columns = ['cell_id', 'stage']
print(f'  Stages: {stage_df["stage"].value_counts().to_dict()}')

# Merge everything on cell_id
print('Merging data...')
hlca_cols = [c for c in hlca_emb.columns if c != 'cell_id']
luca_cols = [c for c in luca_emb.columns if c != 'cell_id']

merged = hlca_emb.merge(luca_emb, on='cell_id', suffixes=('_hlca', '_luca'))
merged = merged.merge(conf_df, on='cell_id')
merged = merged.merge(stage_df, on='cell_id')
print(f'  Merged: {len(merged)} cells')

# Extract arrays
hlca_vals = merged[[f'{c}_hlca' if f'{c}_hlca' in merged.columns else c for c in hlca_cols]].values
if hlca_vals.shape[1] == 0:
    hlca_vals = merged[[c for c in merged.columns if 'latent' in c and '_luca' not in c]].values[:, :30]

luca_vals = merged[[f'{c}_luca' if f'{c}_luca' in merged.columns else c for c in luca_cols]].values
if luca_vals.shape[1] == 0:
    luca_vals = merged[[c for c in merged.columns if 'latent' in c and '_luca' in c]].values

# Fallback: if column naming is different
if hlca_vals.shape[1] == 0:
    hlca_df = hlca_emb.set_index('cell_id').loc[merged['cell_id']]
    hlca_vals = hlca_df.values
if luca_vals.shape[1] == 0:
    luca_df = luca_emb.set_index('cell_id').loc[merged['cell_id']]
    luca_vals = luca_df.values

print(f'  HLCA vals: {hlca_vals.shape}')
print(f'  LuCA vals: {luca_vals.shape}')

fused = np.concatenate([hlca_vals, luca_vals], axis=1)
print(f'  Fused shape: {fused.shape}')

stage_ids = merged['stage'].values
hlca_conf = merged['hlca_confidence'].values if 'hlca_confidence' in merged.columns else np.zeros(len(merged))
luca_conf = merged['luca_confidence'].values if 'luca_confidence' in merged.columns else np.zeros(len(merged))

# Save fused embedding
fused_df = pd.DataFrame(fused, columns=[f'latent_{i}' for i in range(fused.shape[1])])
fused_df['cell_id'] = merged['cell_id'].values
fused_df.to_parquet(output_dir / 'fused_embedding.parquet', index=False)
print(f'  Saved fused_embedding.parquet')

# ============================================
# Figure 1: Reference Mapping Summary (4-panel)
# ============================================
print('Generating reference_mapping_summary.png...')
from sklearn.decomposition import PCA

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Dual-Reference Mapping Summary (k-NN Projection)", fontsize=14, fontweight='bold')

# 1. Fused embedding PCA colored by stage
ax = axes[0, 0]
pca = PCA(n_components=2)
fused_2d = pca.fit_transform(fused)
var_explained = sum(pca.explained_variance_ratio_) * 100

for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
    mask = stage_ids == stage
    if mask.sum() > 0:
        ax.scatter(fused_2d[mask, 0], fused_2d[mask, 1],
                   c=STAGE_COLORS.get(stage, '#999999'), label=stage,
                   s=5, alpha=0.3, edgecolors='none')
ax.legend(loc='upper right', fontsize=8)
ax.set_title(f"Fused Embedding (PCA, {var_explained:.0f}% var)")
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

# 2. HLCA vs LuCA confidence scatter
ax = axes[0, 1]
for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
    mask = stage_ids == stage
    if mask.sum() > 0:
        ax.scatter(hlca_conf[mask], luca_conf[mask],
                   c=STAGE_COLORS.get(stage, '#999999'), label=stage,
                   s=5, alpha=0.3, edgecolors='none')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel("HLCA Confidence (healthy)")
ax.set_ylabel("LuCA Confidence (cancer)")
ax.set_title("Reference Confidence by Stage")
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 3. Confidence distributions
ax = axes[1, 0]
ax.hist(hlca_conf, bins=50, alpha=0.6, label='HLCA', color='blue', density=True)
ax.hist(luca_conf, bins=50, alpha=0.6, label='LuCA', color='red', density=True)
ax.set_xlabel("Calibrated Confidence")
ax.set_ylabel("Density")
ax.set_title("Confidence Distributions")
ax.legend()

# 4. Stage distribution
ax = axes[1, 1]
stage_counts = {}
for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
    count = (stage_ids == stage).sum()
    if count > 0:
        stage_counts[stage] = count
stages = list(stage_counts.keys())
counts = list(stage_counts.values())
colors = [STAGE_COLORS[s] for s in stages]
bars = ax.bar(stages, counts, color=colors, alpha=0.8)
ax.set_ylabel("Cell Count")
ax.set_title("Stage Distribution")
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{count:,}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(plots_dir / 'reference_mapping_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved reference_mapping_summary.png')

# ============================================
# Figure 2: Latent Heatmap by Stage
# ============================================
print('Generating latent_heatmap_by_stage.png...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, emb) in enumerate([("HLCA", hlca_vals), ("LuCA", luca_vals)]):
    ax = axes[idx]
    stages_present = [s for s in ["Normal", "AAH", "AIS", "MIA", "LUAD"] if (stage_ids == s).sum() > 0]
    mean_latent = np.zeros((len(stages_present), emb.shape[1]))
    for i, stage in enumerate(stages_present):
        mask = stage_ids == stage
        mean_latent[i] = emb[mask].mean(axis=0)
    im = ax.imshow(mean_latent, aspect='auto', cmap='RdBu_r')
    ax.set_yticks(range(len(stages_present)))
    ax.set_yticklabels(stages_present)
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Stage")
    ax.set_title(f"{name} Mean Latent by Stage ({emb.shape[1]}D)")
    plt.colorbar(im, ax=ax, label="Value")

plt.tight_layout()
plt.savefig(plots_dir / 'latent_heatmap_by_stage.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved latent_heatmap_by_stage.png')

# ============================================
# Figure 3: k-NN Distance Distributions
# ============================================
print('Generating knn_distance_distributions.png...')
manifest_path = output_dir / 'reference_manifest.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())

    # For k-NN mode, show distance statistics instead of training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # HLCA distance histogram from confidence
    ax = axes[0]
    ax.hist(hlca_conf, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(np.median(hlca_conf), color='red', linestyle='--', label=f'Median: {np.median(hlca_conf):.3f}')
    ax.set_xlabel('Calibrated Confidence')
    ax.set_ylabel('Count')
    ax.set_title('HLCA k-NN Confidence Distribution')
    ax.legend()

    # LuCA distance histogram from confidence
    ax = axes[1]
    ax.hist(luca_conf, bins=50, alpha=0.7, color='coral', edgecolor='white')
    ax.axvline(np.median(luca_conf), color='red', linestyle='--', label=f'Median: {np.median(luca_conf):.3f}')
    ax.set_xlabel('Calibrated Confidence')
    ax.set_ylabel('Count')
    ax.set_title('LuCA k-NN Confidence Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / 'knn_distance_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved knn_distance_distributions.png')

# ============================================
# Figure 4: UMAP colored by stage
# ============================================
print('Generating fused_umap.png (this may take a few minutes)...')
try:
    import umap

    # PCA first for speed
    pca = PCA(n_components=min(50, fused.shape[1]))
    fused_pca = pca.fit_transform(fused)

    # UMAP
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42, n_jobs=4)
    umap_coords = reducer.fit_transform(fused_pca)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By stage
    ax = axes[0]
    for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
        mask = stage_ids == stage
        if mask.sum() > 0:
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=STAGE_COLORS.get(stage, '#999999'), label=stage,
                       s=3, alpha=0.3, edgecolors='none')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Fused Embedding UMAP by Stage')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    # By HLCA-LuCA confidence ratio
    ax = axes[1]
    conf_ratio = hlca_conf / (luca_conf + 0.01)  # Higher = more healthy-like
    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                         c=np.log1p(conf_ratio), cmap='RdYlBu', s=3, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='log(HLCA/LuCA confidence)')
    ax.set_title('UMAP by Reference Balance')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.savefig(plots_dir / 'fused_umap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fused_umap.png')

    # Save UMAP coordinates
    umap_df = pd.DataFrame({'cell_id': merged['cell_id'], 'umap_1': umap_coords[:, 0], 'umap_2': umap_coords[:, 1]})
    umap_df.to_parquet(output_dir / 'umap_coordinates.parquet', index=False)
    print('  Saved umap_coordinates.parquet')

except ImportError:
    print('  UMAP not available, skipping')

# Update manifest
print('Updating manifest...')
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
else:
    manifest = {}

manifest['fused'] = {
    'n_cells': len(merged),
    'hlca_dim': hlca_vals.shape[1],
    'luca_dim': luca_vals.shape[1],
    'fused_dim': fused.shape[1],
    'merged_at': pd.Timestamp.now().isoformat()
}
manifest_path.write_text(json.dumps(manifest, indent=2))
print('Done!')
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  $DATA/processed/luad_evo/reference_geometry/"
ls -la $DATA/processed/luad_evo/reference_geometry/
echo ""
echo "Plots:"
ls -la $DATA/processed/luad_evo/reference_geometry/plots/
