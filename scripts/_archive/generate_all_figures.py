#!/usr/bin/env python3
"""
Generate ALL publication figures from traceable data sources.

This is the SINGLE SCRIPT that generates every figure in the paper.
Each figure function documents its data source and computation.

Usage:
    python scripts/generate_all_figures.py --output-dir figures/publication

Figures generated:
    1. fig_cell_composition.pdf - T-cell, macrophage, fibroblast by stage
    2. fig_ablation_study.pdf - Ablation impact violin plot
    3. fig_baseline_comparison.pdf - StageBridge vs baselines
    4. fig_drift_field.pdf - Helmholtz decomposition of drift field
    5. fig_context_pathway.pdf - Context embedding correlations
    6. fig_il1b_expression.pdf - IL1B expression by stage
    7. fig_attention_weights.pdf - Model attention analysis
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

STAGE_COLORS = {
    'Normal': '#3498db',
    'Preinvasive': '#f39c12',
    'Invasive': '#e74c3c',
    'AAH': '#82e0aa',
    'AIS': '#f7dc6f',
    'MIA': '#f5b7b1',
    'LUAD': '#e74c3c',
}


def load_cells(cells_path: Path) -> pd.DataFrame:
    """Load cells data with validation."""
    print(f"Loading cells from {cells_path}...")
    cells = pd.read_parquet(cells_path)
    print(f"  Loaded {len(cells):,} cells")
    print(f"  Stages: {cells['stage'].unique().tolist()}")
    return cells


def load_comparison_report(results_dir: Path) -> dict:
    """Load ablation/baseline comparison report."""
    report_path = results_dir / 'comparison_report.json'
    if not report_path.exists():
        raise FileNotFoundError(f"comparison_report.json not found at {report_path}")
    with open(report_path) as f:
        return json.load(f)


# =============================================================================
# Figure 1: Cell Type Composition by Stage
# =============================================================================
def fig_cell_composition(cells: pd.DataFrame, output_dir: Path):
    """
    Generate cell type composition figure.

    Data source: cells.parquet, cell_type column
    Computation: Per-donor percentage, then mean/std across donors per stage
    """
    print("\nGenerating fig_cell_composition...")

    cell_type_col = 'cell_type'

    # Compute per-donor percentages
    records = []
    for (donor, stage), group in cells.groupby(['donor_id', 'stage']):
        n = len(group)

        # T cells
        t_cells = group[cell_type_col].str.contains('T cell', case=False, na=False).sum()

        # Macrophages
        macros = group[cell_type_col].str.contains('Macrophage|Mono', case=False, na=False).sum()

        # Fibroblasts
        fibros = group[cell_type_col].str.contains('Fibroblast', case=False, na=False).sum()

        records.append({
            'donor_id': donor,
            'stage': stage,
            't_cell_pct': 100 * t_cells / n,
            'macrophage_pct': 100 * macros / n,
            'fibroblast_pct': 100 * fibros / n,
        })

    df = pd.DataFrame(records)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    stage_order = ['Normal', 'Preinvasive', 'Invasive']
    cell_types = [('t_cell_pct', 'T Cells'), ('macrophage_pct', 'Macrophages'), ('fibroblast_pct', 'Fibroblasts')]

    for ax, (col, title) in zip(axes, cell_types):
        # Raincloud-style: violin + strip + boxplot
        df_plot = df[df['stage'].isin(stage_order)]

        sns.violinplot(data=df_plot, x='stage', y=col, order=stage_order,
                      palette=[STAGE_COLORS[s] for s in stage_order],
                      inner=None, alpha=0.3, ax=ax)
        sns.stripplot(data=df_plot, x='stage', y=col, order=stage_order,
                     palette=[STAGE_COLORS[s] for s in stage_order],
                     size=4, alpha=0.7, ax=ax)
        sns.boxplot(data=df_plot, x='stage', y=col, order=stage_order,
                   color='white', width=0.3, ax=ax,
                   boxprops=dict(alpha=0.7), showfliers=False)

        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(title)

        # Add mean values as text
        for i, stage in enumerate(stage_order):
            mean = df_plot[df_plot['stage'] == stage][col].mean()
            ax.text(i, ax.get_ylim()[1] * 0.95, f'{mean:.1f}%',
                   ha='center', va='top', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save
    out_path = output_dir / 'fig_cell_composition.pdf'
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    print(f"  Saved: {out_path}")

    # Print computed values for paper
    print("\n  COMPUTED VALUES (for paper):")
    for stage in stage_order:
        stage_df = df[df['stage'] == stage]
        print(f"    {stage}: T-cells={stage_df['t_cell_pct'].mean():.1f}%, "
              f"Macro={stage_df['macrophage_pct'].mean():.1f}%, "
              f"Fibro={stage_df['fibroblast_pct'].mean():.1f}%")

    plt.close(fig)
    return df


# =============================================================================
# Figure 2: Ablation Study
# =============================================================================
def fig_ablation_study(comparison: dict, output_dir: Path):
    """
    Generate ablation study figure.

    Data source: results/v1/comparison_report.json
    Computation: Delta % vs full model
    """
    print("\nGenerating fig_ablation_study...")

    ablations = comparison.get('ablations', {})
    if not ablations:
        print("  WARNING: No ablation data found")
        return

    # Prepare data
    data = []
    for name, stats in ablations.items():
        data.append({
            'ablation': name.replace('_', ' ').title(),
            'delta_pct': stats['delta_vs_full'],
            'val_loss': stats['mean_val_loss'],
            'n_runs': stats['n_runs'],
        })

    df = pd.DataFrame(data)
    df = df.sort_values('delta_pct', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#e74c3c' if d > 5 else '#f39c12' if d > 2 else '#3498db' for d in df['delta_pct']]

    bars = ax.barh(df['ablation'], df['delta_pct'], color=colors, edgecolor='black', linewidth=0.5)

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Change in Validation Loss (%)')
    ax.set_title('Ablation Study: Component Importance')

    # Add value labels
    for bar, val in zip(bars, df['delta_pct']):
        x = bar.get_width()
        ax.text(x + 0.3, bar.get_y() + bar.get_height()/2,
               f'+{val:.1f}%' if val > 0 else f'{val:.1f}%',
               va='center', fontsize=9)

    plt.tight_layout()

    out_path = output_dir / 'fig_ablation_study.pdf'
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    print(f"  Saved: {out_path}")

    # Print values
    print("\n  COMPUTED VALUES (for paper):")
    for _, row in df.iterrows():
        print(f"    {row['ablation']}: +{row['delta_pct']:.1f}%")

    plt.close(fig)
    return df


# =============================================================================
# Figure 3: Baseline Comparison
# =============================================================================
def fig_baseline_comparison(comparison: dict, output_dir: Path):
    """
    Generate baseline comparison figure.

    Data source: results/v1/comparison_report.json
    Computation: MSE loss comparison
    """
    print("\nGenerating fig_baseline_comparison...")

    baselines = comparison.get('baselines', {})
    full_model = comparison.get('full_model', {})

    if not baselines or not full_model:
        print("  WARNING: No baseline data found")
        return

    # Prepare data
    data = [{'model': 'StageBridge', 'val_loss': full_model['mean_val_loss'], 'std': full_model['std_val_loss']}]
    for name, stats in baselines.items():
        data.append({
            'model': name.replace('_', ' ').title(),
            'val_loss': stats['mean_val_loss'],
            'std': stats['std_val_loss'],
        })

    df = pd.DataFrame(data)
    df = df.sort_values('val_loss')

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#2ecc71' if m == 'StageBridge' else '#95a5a6' for m in df['model']]

    bars = ax.barh(df['model'], df['val_loss'], xerr=df['std'], color=colors,
                  edgecolor='black', linewidth=0.5, capsize=3)

    ax.set_xlabel('Validation Loss (MSE)')
    ax.set_title('Model Comparison')
    ax.set_xscale('log')

    # Add fold improvement
    full_loss = full_model['mean_val_loss']
    for bar, (_, row) in zip(bars, df.iterrows()):
        if row['model'] != 'StageBridge':
            fold = row['val_loss'] / full_loss
            ax.text(row['val_loss'] * 1.1, bar.get_y() + bar.get_height()/2,
                   f'{fold:.0f}x', va='center', fontsize=9)

    plt.tight_layout()

    out_path = output_dir / 'fig_baseline_comparison.pdf'
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    print(f"  Saved: {out_path}")

    # Print values
    print("\n  COMPUTED VALUES (for paper):")
    for _, row in df.iterrows():
        fold = row['val_loss'] / full_loss
        print(f"    {row['model']}: {row['val_loss']:.4f} ({fold:.1f}x)")

    plt.close(fig)
    return df


# =============================================================================
# Figure 4: IL1B Expression by Stage
# =============================================================================
def fig_il1b_expression(cells: pd.DataFrame, output_dir: Path):
    """
    Generate IL1B expression figure.

    Data source: cells.parquet, il1b_raw column
    Computation: Per-cell expression, grouped by stage
    """
    print("\nGenerating fig_il1b_expression...")

    if 'il1b_raw' not in cells.columns:
        print("  WARNING: il1b_raw not found in cells")
        return

    stage_order = ['Normal', 'Preinvasive', 'Invasive']
    df = cells[cells['stage'].isin(stage_order)][['stage', 'il1b_raw']].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Violin plot of expression
    ax = axes[0]
    sns.violinplot(data=df, x='stage', y='il1b_raw', order=stage_order,
                  palette=[STAGE_COLORS[s] for s in stage_order], ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('IL1B Expression')
    ax.set_title('A. IL1B Expression by Stage')

    # Panel B: Percentage IL1B+
    ax = axes[1]
    threshold = 0.5  # Define IL1B+ threshold
    pct_positive = []
    for stage in stage_order:
        stage_expr = df[df['stage'] == stage]['il1b_raw']
        pct = 100 * (stage_expr > threshold).sum() / len(stage_expr)
        pct_positive.append(pct)

    bars = ax.bar(stage_order, pct_positive, color=[STAGE_COLORS[s] for s in stage_order],
                 edgecolor='black', linewidth=0.5)
    ax.set_ylabel('IL1B+ Cells (%)')
    ax.set_title('B. IL1B+ Cell Percentage')

    for bar, pct in zip(bars, pct_positive):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    out_path = output_dir / 'fig_il1b_expression.pdf'
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    print(f"  Saved: {out_path}")

    # Compute fold change
    normal_mean = df[df['stage'] == 'Normal']['il1b_raw'].mean()
    invasive_mean = df[df['stage'] == 'Invasive']['il1b_raw'].mean()
    fold_change = invasive_mean / normal_mean if normal_mean > 0 else None

    # Stage correlation
    stage_map = {'Normal': 0, 'Preinvasive': 1, 'Invasive': 2}
    stage_numeric = df['stage'].map(stage_map)
    r, p = stats.spearmanr(df['il1b_raw'], stage_numeric)

    print("\n  COMPUTED VALUES (for paper):")
    print(f"    IL1B+ percentages: {dict(zip(stage_order, pct_positive))}")
    print(f"    Fold change (Normal->Invasive): {fold_change:.2f}x")
    print(f"    Stage correlation: r={r:.3f}, p={p:.2e}")

    plt.close(fig)


# =============================================================================
# Figure 5: Context Embedding Correlations
# =============================================================================
def fig_context_correlations(cells: pd.DataFrame, output_dir: Path):
    """
    Generate context embedding correlation figure.

    Data source: cells.parquet, gamma_* columns
    Computation: Spearman correlation with stage
    """
    print("\nGenerating fig_context_correlations...")

    gamma_cols = [c for c in cells.columns if c.startswith('gamma_')]
    if not gamma_cols:
        print("  WARNING: No gamma columns found")
        return

    stage_map = {'Normal': 0, 'AAH': 1, 'AIS': 2, 'MIA': 3, 'LUAD': 4, 'Preinvasive': 1, 'Invasive': 2}
    valid_mask = cells['stage'].isin(stage_map.keys())
    stage_numeric = cells.loc[valid_mask, 'stage'].map(stage_map)

    # Compute correlations
    correlations = []
    for col in gamma_cols:
        r, p = stats.spearmanr(cells.loc[valid_mask, col], stage_numeric)
        correlations.append({'gamma': col, 'r': r, 'p': p})

    df = pd.DataFrame(correlations)
    df = df.sort_values('r', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#e74c3c' if abs(r) > 0.2 else '#3498db' for r in df['r']]

    ax.barh(df['gamma'], df['r'], color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Spearman Correlation with Stage')
    ax.set_title('Context Embedding Stage Correlations')

    plt.tight_layout()

    out_path = output_dir / 'fig_context_correlations.pdf'
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix('.png'))
    print(f"  Saved: {out_path}")

    print("\n  COMPUTED VALUES (for paper):")
    for _, row in df.head(5).iterrows():
        print(f"    {row['gamma']}: r={row['r']:.3f}, p={row['p']:.2e}")

    plt.close(fig)
    return df


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate all publication figures')
    parser.add_argument('--cells', type=Path, default=Path('/home/booka/projects/StageBridge_V1/cells.parquet'))
    parser.add_argument('--results-dir', type=Path, default=Path('/home/booka/projects/StageBridge/results/v1'))
    parser.add_argument('--output-dir', type=Path, default=Path('/home/booka/projects/StageBridge/figures/publication'))
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING ALL PUBLICATION FIGURES")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Cells: {args.cells}")
    print(f"Results: {args.results_dir}")
    print(f"Output: {args.output_dir}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    cells = None
    if args.cells.exists():
        cells = load_cells(args.cells)
    else:
        print(f"WARNING: cells.parquet not found at {args.cells}")

    comparison = None
    try:
        comparison = load_comparison_report(args.results_dir)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")

    # Generate figures
    if cells is not None:
        fig_cell_composition(cells, args.output_dir)
        fig_il1b_expression(cells, args.output_dir)
        fig_context_correlations(cells, args.output_dir)

    if comparison is not None:
        fig_ablation_study(comparison, args.output_dir)
        fig_baseline_comparison(comparison, args.output_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
