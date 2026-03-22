#!/usr/bin/env python3
"""Visualize dual-reference embeddings."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("Loading embeddings...")
results_dir = Path("results")

hlca_df = pd.read_parquet(results_dir / "hlca_embedding.parquet")
luca_df = pd.read_parquet(results_dir / "luca_embedding.parquet")
fused_df = pd.read_parquet(results_dir / "fused_embedding.parquet")
conf = pd.read_parquet(results_dir / "reference_confidence.parquet")

# Extract only latent columns (numeric)
hlca_cols = [c for c in hlca_df.columns if c.startswith('hlca_latent')]
luca_cols = [c for c in luca_df.columns if c.startswith('luca_latent')]
fused_cols = [c for c in fused_df.columns if c.startswith('fused_latent')]

hlca = hlca_df[hlca_cols]
luca = luca_df[luca_cols]
fused = fused_df[fused_cols]

# Get metadata for coloring
metadata = fused_df[['donor_id', 'stage_id']].copy() if 'donor_id' in fused_df.columns else None

print(f"  HLCA: {hlca.shape}")
print(f"  LuCA: {luca.shape}")
print(f"  Fused: {fused.shape}")
print(f"  Confidence: {conf.shape}")

# Subsample for visualization (787k is too many points)
n_sample = 50000
np.random.seed(42)
idx = np.random.choice(len(hlca), size=min(n_sample, len(hlca)), replace=False)

hlca_sub = hlca.iloc[idx].values
luca_sub = luca.iloc[idx].values
fused_sub = fused.iloc[idx].values

print(f"\nSubsampled to {len(idx)} cells for visualization")

# Create figure
fig = plt.figure(figsize=(16, 12))

# 1. PCA of fused embeddings colored by stage
print("Computing PCA of fused embeddings...")
ax1 = fig.add_subplot(2, 3, 1)
pca_fused = PCA(n_components=2).fit_transform(fused_sub)

# Color by stage if available
if metadata is not None and 'stage_id' in metadata.columns:
    stage_sub = metadata['stage_id'].iloc[idx].values
    unique_stages = np.unique(stage_sub)
    stage_colors = {s: i for i, s in enumerate(unique_stages)}
    colors = [stage_colors[s] for s in stage_sub]
    scatter = ax1.scatter(pca_fused[:, 0], pca_fused[:, 1],
                          c=colors, cmap='tab10',
                          s=1, alpha=0.5, rasterized=True)
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=plt.cm.tab10(stage_colors[s]/len(unique_stages)),
                              markersize=8, label=s) for s in unique_stages[:10]]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8, title='Stage')
else:
    scatter = ax1.scatter(pca_fused[:, 0], pca_fused[:, 1],
                          c=np.arange(len(pca_fused)), cmap='viridis',
                          s=1, alpha=0.5, rasterized=True)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('Fused Embedding (40D) - by Stage', fontweight='bold')

# 2. PCA of HLCA vs LuCA side by side
print("Computing PCA of individual embeddings...")
ax2 = fig.add_subplot(2, 3, 2)
pca_hlca = PCA(n_components=2).fit_transform(hlca_sub)
ax2.scatter(pca_hlca[:, 0], pca_hlca[:, 1], s=1, alpha=0.5, c='steelblue', rasterized=True)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title('HLCA Embedding (30D) - PCA', fontweight='bold')

ax3 = fig.add_subplot(2, 3, 3)
pca_luca = PCA(n_components=2).fit_transform(luca_sub)
ax3.scatter(pca_luca[:, 0], pca_luca[:, 1], s=1, alpha=0.5, c='coral', rasterized=True)
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_title('LuCA Embedding (10D) - PCA', fontweight='bold')

# 3. Variance explained by each reference
ax4 = fig.add_subplot(2, 3, 4)
# Compute variance in each embedding
hlca_var = np.var(hlca_sub, axis=0)
luca_var = np.var(luca_sub, axis=0)
ax4.bar(['HLCA\n(30 dims)', 'LuCA\n(10 dims)'],
        [hlca_var.sum(), luca_var.sum()],
        color=['steelblue', 'coral'], edgecolor='black')
ax4.set_ylabel('Total Variance')
ax4.set_title('Embedding Variance by Reference', fontweight='bold')

# 4. Distribution of L2 norms (should be ~1 if normalized)
ax5 = fig.add_subplot(2, 3, 5)
hlca_norms = np.linalg.norm(hlca_sub, axis=1)
luca_norms = np.linalg.norm(luca_sub, axis=1)

# Check if norms have variance (L2-normalized = all 1.0)
if hlca_norms.std() > 0.001:
    ax5.hist(hlca_norms, bins=50, alpha=0.7, label=f'HLCA (mean={hlca_norms.mean():.3f})', color='steelblue')
    ax5.hist(luca_norms, bins=50, alpha=0.7, label=f'LuCA (mean={luca_norms.mean():.3f})', color='coral')
    ax5.axvline(x=1.0, color='black', linestyle='--', label='Unit norm')
    ax5.set_xlabel('L2 Norm')
    ax5.set_ylabel('Count')
    ax5.legend()
else:
    # All norms are 1.0 (L2 normalized) - show per-dimension variance instead
    hlca_dim_var = np.var(hlca_sub, axis=0)
    luca_dim_var = np.var(luca_sub, axis=0)
    ax5.bar(range(len(hlca_dim_var)), hlca_dim_var, alpha=0.7, label='HLCA', color='steelblue')
    ax5.bar(range(30, 30+len(luca_dim_var)), luca_dim_var, alpha=0.7, label='LuCA', color='coral')
    ax5.set_xlabel('Latent Dimension')
    ax5.set_ylabel('Variance')
    ax5.axvline(x=29.5, color='black', linestyle='--', alpha=0.5)
    ax5.text(15, ax5.get_ylim()[1]*0.9, 'HLCA', ha='center', fontsize=10)
    ax5.text(35, ax5.get_ylim()[1]*0.9, 'LuCA', ha='center', fontsize=10)
    ax5.legend()
ax5.set_title('Per-Dimension Variance', fontweight='bold')

# 5. HLCA PC1 vs LuCA PC1 (cross-reference structure)
ax6 = fig.add_subplot(2, 3, 6)
ax6.scatter(pca_hlca[:, 0], pca_luca[:, 0], s=1, alpha=0.3, c='purple', rasterized=True)
ax6.set_xlabel('HLCA PC1')
ax6.set_ylabel('LuCA PC1')
ax6.set_title('HLCA vs LuCA Structure', fontweight='bold')
# Add correlation
corr = np.corrcoef(pca_hlca[:, 0], pca_luca[:, 0])[0, 1]
ax6.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax6.transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(results_dir / "embedding_overview.png", dpi=150, bbox_inches='tight')
plt.savefig(results_dir / "embedding_overview.pdf", bbox_inches='tight')
print(f"\nSaved: {results_dir / 'embedding_overview.png'}")

# Second figure: Confidence analysis
print("\nGenerating confidence analysis...")
fig2, axes = plt.subplots(1, 2, figsize=(10, 4))

if 'hlca_confidence' in conf.columns and 'luca_confidence' in conf.columns:
    ax = axes[0]
    ax.hist(conf['hlca_confidence'].dropna(), bins=50, alpha=0.7, label='HLCA', color='steelblue')
    ax.hist(conf['luca_confidence'].dropna(), bins=50, alpha=0.7, label='LuCA', color='coral')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution', fontweight='bold')
    ax.legend()

    ax = axes[1]
    # Scatter of confidence vs confidence
    h_conf = conf['hlca_confidence'].values
    l_conf = conf['luca_confidence'].values
    valid = ~(np.isnan(h_conf) | np.isnan(l_conf))
    if valid.sum() > 0:
        sample_idx = np.random.choice(np.where(valid)[0], size=min(10000, valid.sum()), replace=False)
        ax.scatter(h_conf[sample_idx], l_conf[sample_idx], s=1, alpha=0.3)
        ax.set_xlabel('HLCA Confidence')
        ax.set_ylabel('LuCA Confidence')
        ax.set_title('Confidence Agreement', fontweight='bold')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
else:
    # Just show what columns we have
    print(f"  Confidence columns: {conf.columns.tolist()}")
    axes[0].text(0.5, 0.5, f"Columns:\n{chr(10).join(conf.columns[:10])}",
                 ha='center', va='center', transform=axes[0].transAxes)
    axes[1].axis('off')

plt.tight_layout()
plt.savefig(results_dir / "confidence_analysis.png", dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'confidence_analysis.png'}")

print("\nDone!")
