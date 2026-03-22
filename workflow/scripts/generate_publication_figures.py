#!/usr/bin/env python3
"""Generate all publication figures for Nature Methods.

Snakemake script - uses snakemake.input and snakemake.output.

Uses LungPCA-style figures for consistency with the original Peng et al. paper.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import LungPCA style module
from stagebridge.viz import (
    configure_lungpca_style,
    STAGE_COLORS,
    STAGE_ORDER,
    save_lungpca_figure,
    plot_heatmap,
    plot_boxplot_jitter,
    plot_stacked_bar,
    get_stage_colors_list,
)

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
print("Generating Publication Figures (LungPCA Style)")
print("=" * 60)

# Configure LungPCA publication style
configure_lungpca_style()

FORMATS = ['png', 'pdf']


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
# Figure 1: Reference Geometry (LungPCA Figure 1C style)
# =============================================================================
print("\nFigure 1: Reference Geometry...")

try:
    fused_df = pd.read_parquet(fused_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor('white')

    # HLCA projection
    ax = axes[0]
    hlca_cols = [c for c in fused_df.columns if c.startswith('hlca_latent_')]
    if len(hlca_cols) >= 2:
        for stage in STAGE_ORDER:
            if stage not in fused_df.get('stage', pd.Series()).values:
                continue
            mask = fused_df['stage'] == stage
            if mask.sum() > 0:
                ax.scatter(
                    fused_df.loc[mask, hlca_cols[0]],
                    fused_df.loc[mask, hlca_cols[1]],
                    c=STAGE_COLORS.get(stage, '#d9d9d9'),
                    label=stage, alpha=0.5, s=1, rasterized=True
                )
    ax.set_xlabel('HLCA Latent 1', fontsize=6)
    ax.set_ylabel('HLCA Latent 2', fontsize=6)
    ax.set_title('HLCA Projection', fontsize=8)
    ax.legend(markerscale=5, fontsize=5, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # LuCA projection
    ax = axes[1]
    luca_cols = [c for c in fused_df.columns if c.startswith('luca_latent_')]
    if len(luca_cols) >= 2:
        for stage in STAGE_ORDER:
            if stage not in fused_df.get('stage', pd.Series()).values:
                continue
            mask = fused_df['stage'] == stage
            if mask.sum() > 0:
                ax.scatter(
                    fused_df.loc[mask, luca_cols[0]],
                    fused_df.loc[mask, luca_cols[1]],
                    c=STAGE_COLORS.get(stage, '#d9d9d9'),
                    label=stage, alpha=0.5, s=1, rasterized=True
                )
    ax.set_xlabel('LuCA Latent 1', fontsize=6)
    ax.set_ylabel('LuCA Latent 2', fontsize=6)
    ax.set_title('LuCA Projection', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Fused projection
    ax = axes[2]
    fused_cols = [c for c in fused_df.columns if c.startswith('fused_latent_')]
    if len(fused_cols) >= 2:
        for stage in STAGE_ORDER:
            if stage not in fused_df.get('stage', pd.Series()).values:
                continue
            mask = fused_df['stage'] == stage
            if mask.sum() > 0:
                ax.scatter(
                    fused_df.loc[mask, fused_cols[0]],
                    fused_df.loc[mask, fused_cols[1]],
                    c=STAGE_COLORS.get(stage, '#d9d9d9'),
                    label=stage, alpha=0.5, s=1, rasterized=True
                )
    ax.set_xlabel('Fused Latent 1', fontsize=6)
    ax.set_ylabel('Fused Latent 2', fontsize=6)
    ax.set_title('Fused Dual-Reference', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    paths = save_figure(fig, main_dir / 'figure1_reference_geometry')
    figures_generated.append({'name': 'figure1_reference_geometry', 'paths': paths})
    plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Figure 2: Training Curves (LungPCA style)
# =============================================================================
print("\nFigure 2: Training Curves...")

try:
    with open(training_path) as f:
        training_results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    fig.patch.set_facecolor('white')

    # SSL pretraining loss
    ax = axes[0]
    if 'ssl_loss' in training_results:
        epochs = range(1, len(training_results['ssl_loss']) + 1)
        ax.semilogy(epochs, training_results['ssl_loss'], 'b-', linewidth=1.5, label='SSL Loss')
        if 'ssl_val_loss' in training_results:
            ax.semilogy(epochs, training_results['ssl_val_loss'], 'b--', linewidth=1, alpha=0.7, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=6)
    ax.set_ylabel('Loss (log)', fontsize=6)
    ax.set_title('Stage 1: SSL Pretraining', fontsize=8)
    ax.legend(fontsize=5, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Transition loss
    ax = axes[1]
    if 'transition_loss' in training_results:
        epochs = range(1, len(training_results['transition_loss']) + 1)
        ax.semilogy(epochs, training_results['transition_loss'], 'r-', linewidth=1.5, label='Transition Loss')
        if 'transition_val_loss' in training_results:
            ax.semilogy(epochs, training_results['transition_val_loss'], 'r--', linewidth=1, alpha=0.7, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=6)
    ax.set_ylabel('Loss (log)', fontsize=6)
    ax.set_title('Stage 2: Transition Model', fontsize=8)
    ax.legend(fontsize=5, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    paths = save_figure(fig, main_dir / 'figure2_training_curves')
    figures_generated.append({'name': 'figure2_training_curves', 'paths': paths})
    plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Figure 3: Spatial Backend Comparison (bar chart)
# =============================================================================
print("\nFigure 3: Spatial Backend Comparison...")

try:
    with open(spatial_path) as f:
        spatial_results = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor('white')

    if 'composite_scores' in spatial_results:
        scores = spatial_results['composite_scores']
        backends = list(scores.keys())
        vals = [scores[b] for b in backends]
        canonical = spatial_results.get('canonical_backend', '')

        # Color canonical backend differently
        colors = ['#33a02c' if b == canonical else '#1f78b4' for b in backends]

        bars = ax.barh(backends, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Composite Score', fontsize=6)
        ax.set_title('Spatial Deconvolution Backend Comparison', fontsize=8)

        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=6)

    plt.tight_layout()
    paths = save_figure(fig, main_dir / 'figure3_spatial_comparison')
    figures_generated.append({'name': 'figure3_spatial_comparison', 'paths': paths})
    plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Figure 4: Model Architecture (placeholder - create in paper)
# =============================================================================
print("\nFigure 4: Model Architecture (diagram placeholder)...")
# Note: Model architecture diagram should be created separately (e.g., TikZ)

# =============================================================================
# Figure 5: Ablation Heatmap (LungPCA pheatmap style)
# =============================================================================
print("\nFigure 5: Ablation Study Results...")

try:
    with open(ablations_path) as f:
        ablation_results = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    fig.patch.set_facecolor('white')

    # Build heatmap data
    ablations = ablation_results.get('ablations', [])
    metrics = ['transition_mae', 'flow_correlation', 'stage_accuracy', 'niche_sensitivity']
    metrics = [m for m in metrics if any(
        m in ablation_results.get('results', {}).get(abl, {}).get('metrics', {})
        for abl in ablations
    )]

    data = []
    valid_ablations = []
    for abl in ablations:
        row = []
        abl_data = ablation_results.get('results', {}).get(abl, {}).get('metrics', {})
        if abl_data:
            for m in metrics:
                row.append(abl_data.get(m, np.nan))
            data.append(row)
            valid_ablations.append(abl)

    if data:
        data = np.array(data)

        # Normalize per metric for fair comparison
        data_norm = (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0) + 1e-8)

        im = plot_heatmap(
            ax, data_norm,
            row_labels=valid_ablations,
            col_labels=[m.replace('_', ' ').title() for m in metrics],
            cmap='RdYlGn',
            vmin=0, vmax=1,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Normalized Score', fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        ax.set_title('Ablation Study: Component Contributions', fontsize=8)

    plt.tight_layout()
    paths = save_figure(fig, main_dir / 'figure5_ablation_heatmap')
    figures_generated.append({'name': 'figure5_ablation_heatmap', 'paths': paths})
    plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Supplementary Figure 1: Stage Distribution by Donor
# =============================================================================
print("\nSupp Figure 1: Stage Distribution...")

try:
    fused_df = pd.read_parquet(fused_path)

    if 'donor' in fused_df.columns and 'stage' in fused_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        fig.patch.set_facecolor('white')

        # Count by donor and stage
        counts = fused_df.groupby(['donor', 'stage']).size().unstack(fill_value=0)

        # Order stages
        stage_order = [s for s in STAGE_ORDER if s in counts.columns]
        counts = counts[stage_order]

        # Stacked bar
        bottom = np.zeros(len(counts))
        for stage in stage_order:
            if stage in counts.columns:
                ax.bar(
                    range(len(counts)), counts[stage], bottom=bottom,
                    label=stage, color=STAGE_COLORS.get(stage, '#d9d9d9'),
                    edgecolor='white', linewidth=0.5
                )
                bottom += counts[stage].values

        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=5)
        ax.set_ylabel('Number of Cells', fontsize=6)
        ax.set_title('Cell Count by Donor and Stage', fontsize=8)
        ax.legend(fontsize=5, frameon=False, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        paths = save_figure(fig, supp_dir / 'supp_fig1_stage_distribution')
        figures_generated.append({'name': 'supp_fig1_stage_distribution', 'paths': paths, 'type': 'supplementary'})
        plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Supplementary Figure 2: Reference Confidence Distribution
# =============================================================================
print("\nSupp Figure 2: Reference Confidence...")

try:
    fused_df = pd.read_parquet(fused_path)

    conf_cols = [c for c in fused_df.columns if 'confidence' in c.lower()]

    if conf_cols:
        fig, axes = plt.subplots(1, len(conf_cols), figsize=(4*len(conf_cols), 4), dpi=150)
        fig.patch.set_facecolor('white')

        if len(conf_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, conf_cols):
            for stage in STAGE_ORDER:
                if stage not in fused_df.get('stage', pd.Series()).values:
                    continue
                mask = fused_df['stage'] == stage
                if mask.sum() > 0:
                    vals = fused_df.loc[mask, col].dropna()
                    ax.hist(vals, bins=50, alpha=0.5, label=stage,
                           color=STAGE_COLORS.get(stage, '#d9d9d9'))

            ax.set_xlabel(col.replace('_', ' ').title(), fontsize=6)
            ax.set_ylabel('Count', fontsize=6)
            ax.legend(fontsize=5, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        paths = save_figure(fig, supp_dir / 'supp_fig2_reference_confidence')
        figures_generated.append({'name': 'supp_fig2_reference_confidence', 'paths': paths, 'type': 'supplementary'})
        plt.close(fig)

except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Generate Manifest
# =============================================================================
print("\nGenerating figure manifest...")

manifest = {
    'generated_at': datetime.now().isoformat(),
    'style': 'LungPCA (Peng et al.)',
    'n_figures': len(figures_generated),
    'n_main': len([f for f in figures_generated if f.get('type') != 'supplementary']),
    'n_supplementary': len([f for f in figures_generated if f.get('type') == 'supplementary']),
    'figures': figures_generated,
    'main_dir': str(main_dir),
    'supplementary_dir': str(supp_dir),
    'formats': FORMATS,
    'dpi': 300,
    'stage_colors': STAGE_COLORS,
}

manifest_path = Path(manifest_output)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"Saved manifest: {manifest_path}")

print()
print("=" * 60)
print(f"Figure Generation Complete: {len(figures_generated)} figures")
print(f"  Main figures: {manifest['n_main']}")
print(f"  Supplementary: {manifest['n_supplementary']}")
print("=" * 60)
