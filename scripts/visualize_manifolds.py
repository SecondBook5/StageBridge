#!/usr/bin/env python3
"""Generate UMAP, t-SNE, and PHATE visualizations of dual-reference embeddings."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import umap
import phate
import warnings
warnings.filterwarnings('ignore')

print("Loading embeddings...")
results_dir = Path("results")

fused_df = pd.read_parquet(results_dir / "fused_embedding.parquet")

# Extract latent columns
fused_cols = [c for c in fused_df.columns if c.startswith('fused_latent')]
fused = fused_df[fused_cols].values

# Get metadata
stage = fused_df['stage_id'].values if 'stage_id' in fused_df.columns else None
donor = fused_df['donor_id'].values if 'donor_id' in fused_df.columns else None

print(f"  Fused: {fused.shape}")
print(f"  Stages: {np.unique(stage)}")

# Subsample for computation
n_sample = 30000  # Balance between detail and speed
np.random.seed(42)
idx = np.random.choice(len(fused), size=min(n_sample, len(fused)), replace=False)

fused_sub = fused[idx]
stage_sub = stage[idx] if stage is not None else None
donor_sub = donor[idx] if donor is not None else None

print(f"\nSubsampled to {len(idx)} cells")

# Define stage colors (consistent across plots)
unique_stages = np.unique(stage_sub)
stage_order = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']  # Progression order
stage_colors = {
    'Normal': '#2ecc71',  # Green
    'AAH': '#f39c12',     # Orange
    'AIS': '#e74c3c',     # Red
    'MIA': '#9b59b6',     # Purple
    'LUAD': '#34495e',    # Dark gray
}
# Fallback for any missing stages
for s in unique_stages:
    if s not in stage_colors:
        stage_colors[s] = '#95a5a6'

colors = [stage_colors.get(s, '#95a5a6') for s in stage_sub]

# Compute embeddings
print("\n" + "="*60)
print("Computing UMAP...")
print("="*60)
umap_model = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
umap_emb = umap_model.fit_transform(fused_sub)
print("  Done!")

print("\n" + "="*60)
print("Computing t-SNE...")
print("="*60)
tsne_model = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
tsne_emb = tsne_model.fit_transform(fused_sub)
print("  Done!")

print("\n" + "="*60)
print("Computing PHATE...")
print("="*60)
phate_model = phate.PHATE(n_components=2, knn=15, decay=40, t='auto', random_state=42, n_jobs=-1)
phate_emb = phate_model.fit_transform(fused_sub)
print("  Done!")

# Create comparison figure
print("\nGenerating comparison figure...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, emb, title in zip(axes, [umap_emb, tsne_emb, phate_emb], ['UMAP', 't-SNE', 'PHATE']):
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=3, alpha=0.6, rasterized=True)
    ax.set_xlabel(f'{title}1')
    ax.set_ylabel(f'{title}2')
    ax.set_title(f'{title} of Fused Embedding (40D)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=stage_colors[s], label=s) for s in stage_order if s in unique_stages]
fig.legend(handles=legend_elements, loc='center right', fontsize=11, title='Stage',
           title_fontsize=12, bbox_to_anchor=(0.99, 0.5))

plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.savefig(results_dir / "manifold_comparison.png", dpi=200, bbox_inches='tight')
plt.savefig(results_dir / "manifold_comparison.pdf", bbox_inches='tight')
print(f"Saved: {results_dir / 'manifold_comparison.png'}")

# Create individual high-res figures
print("\nGenerating individual figures...")

for emb, name in [(umap_emb, 'umap'), (tsne_emb, 'tsne'), (phate_emb, 'phate')]:
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=5, alpha=0.7, rasterized=True)
    ax.set_xlabel(f'{name.upper()}1', fontsize=12)
    ax.set_ylabel(f'{name.upper()}2', fontsize=12)
    ax.set_title(f'Fused Dual-Reference Embedding ({name.upper()})', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    legend_elements = [Patch(facecolor=stage_colors[s], label=s) for s in stage_order if s in unique_stages]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, title='Stage')

    plt.tight_layout()
    plt.savefig(results_dir / f"fused_{name}.png", dpi=200, bbox_inches='tight')
    plt.savefig(results_dir / f"fused_{name}.pdf", bbox_inches='tight')
    print(f"  Saved: fused_{name}.png")

# Bonus: Color by donor to check batch effects
print("\nGenerating donor-colored UMAP...")
if donor_sub is not None:
    unique_donors = np.unique(donor_sub)
    donor_cmap = plt.cm.get_cmap('tab20', len(unique_donors))
    donor_color_map = {d: donor_cmap(i) for i, d in enumerate(unique_donors)}
    donor_colors = [donor_color_map[d] for d in donor_sub]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(umap_emb[:, 0], umap_emb[:, 1], c=donor_colors, s=5, alpha=0.7, rasterized=True)
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title('Fused Embedding colored by Donor (batch check)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(results_dir / "fused_umap_by_donor.png", dpi=200, bbox_inches='tight')
    print(f"  Saved: fused_umap_by_donor.png")

print("\nDone!")
