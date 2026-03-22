#!/usr/bin/env python3
"""Generate all publication figures for Nature Methods.

Snakemake script - uses snakemake.input and snakemake.output.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Snakemake provides these
fused_path = snakemake.input.fused
spatial_path = snakemake.input.spatial
training_path = snakemake.input.training
ablations_path = snakemake.input.ablations

manifest_output = snakemake.output.manifest
main_dir = Path(snakemake.output.main_figs)
supp_dir = Path(snakemake.output.supp_figs)

# Create output directories
main_dir.mkdir(parents=True, exist_ok=True)
supp_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Generating Publication Figures")
print("=" * 60)

# Publication style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

STAGE_COLORS = {
    "Normal": "#00BA38",
    "AAH": "#F8766D",
    "AIS": "#619CFF",
    "MIA": "#E58700",
    "LUAD": "#A3A500",
}

FORMATS = ['png', 'pdf', 'svg']

def save_figure(fig, path_stem, formats=FORMATS):
    """Save figure in multiple formats."""
    paths = []
    for fmt in formats:
        path = f"{path_stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        paths.append(path)
        print(f"  Saved: {path}")
    return paths

figures_generated = []

# =============================================================================
# Figure 1: Reference Geometry
# =============================================================================
print("\nFigure 1: Reference Geometry...")

try:
    fused_df = pd.read_parquet(fused_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # HLCA projection
    ax = axes[0]
    hlca_cols = [c for c in fused_df.columns if c.startswith('hlca_latent_')]
    if len(hlca_cols) >= 2:
        for stage, color in STAGE_COLORS.items():
            mask = fused_df['stage'] == stage
            if mask.sum() > 0:
                ax.scatter(
                    fused_df.loc[mask, hlca_cols[0]],
                    fused_df.loc[mask, hlca_cols[1]],
                    c=color, label=stage, alpha=0.3, s=1
                )
    ax.set_xlabel('HLCA Latent 1')
    ax.set_ylabel('HLCA Latent 2')
    ax.set_title('HLCA Projection')
    ax.legend(markerscale=5)

    # LuCA projection
    ax = axes[1]
    luca_cols = [c for c in fused_df.columns if c.startswith('luca_latent_')]
    if len(luca_cols) >= 2:
        for stage, color in STAGE_COLORS.items():
            mask = fused_df['stage'] == stage
            if mask.sum() > 0:
                ax.scatter(
                    fused_df.loc[mask, luca_cols[0]],
                    fused_df.loc[mask, luca_cols[1]],
                    c=color, label=stage, alpha=0.3, s=1
                )
    ax.set_xlabel('LuCA Latent 1')
    ax.set_ylabel('LuCA Latent 2')
    ax.set_title('LuCA Projection')

    # Fused projection
    ax = axes[2]
    fused_cols = [c for c in fused_df.columns if c.startswith('fused_latent_')]
    if len(fused_cols) >= 2:
        for stage, color in STAGE_COLORS.items():
            mask = fused_df['stage'] == stage
            if mask.sum() > 0:
                ax.scatter(
                    fused_df.loc[mask, fused_cols[0]],
                    fused_df.loc[mask, fused_cols[1]],
                    c=color, label=stage, alpha=0.3, s=1
                )
    ax.set_xlabel('Fused Latent 1')
    ax.set_ylabel('Fused Latent 2')
    ax.set_title('Fused Dual-Reference')

    plt.tight_layout()
    paths = save_figure(fig, main_dir / 'figure1_reference_geometry')
    figures_generated.append({'name': 'figure1_reference_geometry', 'paths': paths})
    plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Figure 2: Training Curves (placeholder - needs training results)
# =============================================================================
print("\nFigure 2: Training Curves...")

try:
    with open(training_path) as f:
        training_results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # SSL pretraining loss
    ax = axes[0]
    if 'ssl_loss' in training_results:
        ax.plot(training_results['ssl_loss'], 'b-', label='SSL Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('SSL Pretraining')
    ax.legend()

    # Transition loss
    ax = axes[1]
    if 'transition_loss' in training_results:
        ax.plot(training_results['transition_loss'], 'r-', label='Transition Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Transition Model')
    ax.legend()

    plt.tight_layout()
    paths = save_figure(fig, main_dir / 'figure2_training_curves')
    figures_generated.append({'name': 'figure2_training_curves', 'paths': paths})
    plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Figure 3: Spatial Backend Comparison
# =============================================================================
print("\nFigure 3: Spatial Backend Comparison...")

try:
    with open(spatial_path) as f:
        spatial_results = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6))

    if 'composite_scores' in spatial_results:
        scores = spatial_results['composite_scores']
        backends = list(scores.keys())
        vals = [scores[b] for b in backends]
        canonical = spatial_results.get('canonical_backend', '')

        colors = ['#2ecc71' if b == canonical else '#3498db' for b in backends]
        ax.barh(backends, vals, color=colors)
        ax.set_xlabel('Composite Score')
        ax.set_title('Spatial Backend Comparison')

    plt.tight_layout()
    paths = save_figure(fig, main_dir / 'figure3_spatial_comparison')
    figures_generated.append({'name': 'figure3_spatial_comparison', 'paths': paths})
    plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Figure 5: Ablation Heatmap
# =============================================================================
print("\nFigure 5: Ablation Heatmap...")

try:
    with open(ablations_path) as f:
        ablation_results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Build heatmap data
    ablations = ablation_results.get('ablations', [])
    metrics = ['transition_mae', 'flow_correlation', 'stage_accuracy']

    data = []
    for abl in ablations:
        row = []
        abl_data = ablation_results.get('results', {}).get(abl, {}).get('metrics', {})
        for m in metrics:
            row.append(abl_data.get(m, 0))
        data.append(row)

    if data:
        data = np.array(data)
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticks(range(len(ablations)))
        ax.set_yticklabels(ablations)
        plt.colorbar(im, ax=ax, label='Score')
        ax.set_title('Ablation Study Results')

    plt.tight_layout()
    paths = save_figure(fig, main_dir / 'figure5_ablation_heatmap')
    figures_generated.append({'name': 'figure5_ablation_heatmap', 'paths': paths})
    plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Generate Manifest
# =============================================================================
print("\nGenerating figure manifest...")

manifest = {
    'generated_at': datetime.now().isoformat(),
    'n_figures': len(figures_generated),
    'figures': figures_generated,
    'main_dir': str(main_dir),
    'supplementary_dir': str(supp_dir),
    'formats': FORMATS,
    'dpi': 300,
}

manifest_path = Path(manifest_output)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"Saved manifest: {manifest_path}")

print()
print("=" * 60)
print(f"Figure Generation Complete: {len(figures_generated)} figures")
print("=" * 60)
