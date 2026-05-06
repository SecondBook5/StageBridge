#!/usr/bin/env python3
"""Generate publication-quality training and ablation figures.

Analyzes:
- Full model training across folds/seeds
- Ablation study results (11 ablations)
- Baseline comparisons (4 baselines)
- Learning curves and convergence
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color schemes
ABLATION_COLORS = {
    'full': '#2ecc71',           # Green - full model
    'no_niche': '#e74c3c',       # Red - remove niche context
    'no_gate': '#3498db',        # Blue
    'no_distance': '#9b59b6',    # Purple
    'no_context_refiner': '#f39c12',  # Orange
    'no_ring_pooling': '#1abc9c',     # Teal
    'hlca_only': '#e67e22',      # Dark orange
    'luca_only': '#c0392b',      # Dark red
    'frozen_encoder': '#7f8c8d', # Gray
    'gw_barycentric': '#2980b9', # Medium blue
    'gw_project_hlca': '#8e44ad',# Medium purple
    'gw_project_luca': '#16a085',# Sea green
}

BASELINE_COLORS = {
    'pooling': '#95a5a6',        # Light gray
    'deepsets': '#7f8c8d',       # Gray
    'set_transformer': '#566573', # Dark gray
    'graphsage': '#2c3e50',      # Very dark
    'stagebridge': '#2ecc71',    # Green (full model)
}

ABLATION_LABELS = {
    'full': 'StageBridge (Full)',
    'no_niche': 'No Niche Context',
    'no_gate': 'No Gating',
    'no_distance': 'No Distance Encoding',
    'no_context_refiner': 'No Context Refiner',
    'no_ring_pooling': 'No Ring Pooling',
    'hlca_only': 'HLCA Only',
    'luca_only': 'LuCA Only',
    'frozen_encoder': 'Frozen Encoder',
    'gw_barycentric': 'GW Barycentric',
    'gw_project_hlca': 'GW Project HLCA',
    'gw_project_luca': 'GW Project LuCA',
}

BASELINE_LABELS = {
    'pooling': 'Pooling MLP',
    'deepsets': 'DeepSets',
    'set_transformer': 'Set Transformer',
    'graphsage': 'GraphSAGE',
}


def load_full_results(results_dir: Path) -> dict:
    """Load full model training results."""
    results = defaultdict(list)

    for json_path in results_dir.glob("full/fold_*/seed_*/training_summary.json"):
        parts = json_path.parts
        fold = int([p for p in parts if p.startswith('fold_')][0].split('_')[1])
        seed = int([p for p in parts if p.startswith('seed_')][0].split('_')[1])

        with open(json_path) as f:
            data = json.load(f)

        # Also try to load metrics log
        metrics_path = json_path.parent / "logs" / "metrics.json"
        history = None
        if metrics_path.exists():
            with open(metrics_path) as f:
                history = json.load(f)

        results['fold'].append(fold)
        results['seed'].append(seed)
        results['ssl_loss'] = data.get('ssl', {}).get('best_val_loss', np.nan)
        results['transition_loss'].append(data.get('transition', {}).get('best_val_loss', np.nan))
        results['final_epoch'].append(data.get('transition', {}).get('final_epoch', np.nan))
        results['history'].append(history)

    return dict(results)


def load_ablation_results(results_dir: Path) -> pd.DataFrame:
    """Load all ablation study results."""
    records = []

    for json_path in results_dir.glob("ablations/*/fold_*/seed_*/ablation_*.json"):
        with open(json_path) as f:
            data = json.load(f)

        record = {
            'ablation': data.get('ablation', 'unknown'),
            'fold': data.get('fold_idx', -1),
            'seed': data.get('seed', -1),
            'ssl_loss': data.get('metrics', {}).get('ssl', {}).get('best_val_loss', np.nan),
            'transition_loss': data.get('metrics', {}).get('transition', {}).get('best_val_loss', np.nan),
            'final_epoch': data.get('metrics', {}).get('transition', {}).get('final_epoch', np.nan),
            'n_parameters': data.get('n_parameters', np.nan),
            'runtime_hours': data.get('metrics', {}).get('compute', {}).get('total_runtime_hours', np.nan),
        }
        records.append(record)

    return pd.DataFrame(records)


def load_baseline_results(results_dir: Path) -> pd.DataFrame:
    """Load all baseline results."""
    records = []

    for json_path in results_dir.glob("baselines/*/fold_*/seed_*/baseline_*.json"):
        with open(json_path) as f:
            data = json.load(f)

        record = {
            'baseline': data.get('baseline', 'unknown'),
            'fold': data.get('fold_idx', -1),
            'seed': data.get('seed', -1),
            'best_val_loss': data.get('metrics', {}).get('best_val_loss', np.nan),
            'final_train_loss': data.get('metrics', {}).get('final_train_loss', np.nan),
            'n_parameters': data.get('n_parameters', np.nan),
            'history': data.get('history', {}),
        }
        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# Figure 1: Ablation Study Summary
# =============================================================================
def fig_ablation_summary(ablation_df: pd.DataFrame, full_results: dict, output_dir: Path):
    """Main ablation study figure - bar chart with error bars."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Add full model as reference
    full_loss = np.nanmean(full_results['transition_loss'])
    full_std = np.nanstd(full_results['transition_loss'])

    # Aggregate ablations
    agg = ablation_df.groupby('ablation')['transition_loss'].agg(['mean', 'std', 'count']).reset_index()
    agg = agg.sort_values('mean')

    # Add full model
    full_row = pd.DataFrame([{
        'ablation': 'full',
        'mean': full_loss,
        'std': full_std,
        'count': len(full_results['transition_loss'])
    }])
    agg = pd.concat([full_row, agg], ignore_index=True)
    agg = agg.sort_values('mean')

    # Colors
    colors = [ABLATION_COLORS.get(a, '#888') for a in agg['ablation']]
    labels = [ABLATION_LABELS.get(a, a) for a in agg['ablation']]

    y_pos = np.arange(len(agg))
    bars = ax.barh(y_pos, agg['mean'], xerr=agg['std'], color=colors,
                   edgecolor='white', linewidth=1.5, capsize=4, error_kw={'linewidth': 2})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Validation Loss (Transition)', fontweight='bold', fontsize=12)
    ax.set_title('Ablation Study: Component Contributions', fontweight='bold', fontsize=16)

    # Add reference line for full model
    ax.axvline(full_loss, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Full Model: {full_loss:.4f}')

    # Annotate
    for i, (mean, std, count) in enumerate(zip(agg['mean'], agg['std'], agg['count'])):
        delta = ((mean - full_loss) / full_loss) * 100
        sign = '+' if delta > 0 else ''
        ax.text(mean + std + 0.0001, i, f'{mean:.4f} ({sign}{delta:.1f}%)',
                va='center', fontsize=9, fontweight='bold')

    ax.legend(loc='lower right', frameon=True)
    ax.set_xlim(0, agg['mean'].max() * 1.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_summary.pdf')
    fig.savefig(output_dir / 'ablation_summary.png')
    plt.close()
    print("Saved: ablation_summary")


# =============================================================================
# Figure 2: Ablation Heatmap (Folds x Ablations)
# =============================================================================
def fig_ablation_heatmap(ablation_df: pd.DataFrame, full_results: dict, output_dir: Path):
    """Heatmap showing ablation results across folds."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create full results dataframe
    full_df = pd.DataFrame({
        'ablation': 'full',
        'fold': full_results['fold'],
        'seed': full_results['seed'],
        'transition_loss': full_results['transition_loss']
    })

    combined = pd.concat([full_df, ablation_df[['ablation', 'fold', 'seed', 'transition_loss']]])

    # Average over seeds
    pivot = combined.groupby(['ablation', 'fold'])['transition_loss'].mean().unstack()

    # Order ablations by mean
    order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.reindex(order)

    # Rename for display
    pivot.index = [ABLATION_LABELS.get(a, a) for a in pivot.index]

    # Get full model baseline for relative coloring
    full_mean = pivot.loc['StageBridge (Full)'].mean()

    # Normalize relative to full model
    pivot_rel = (pivot - full_mean) / full_mean * 100

    sns.heatmap(pivot_rel, ax=ax, cmap='RdYlGn_r', center=0,
                annot=pivot.values, fmt='.4f',
                cbar_kws={'label': '% Change from Full Model'},
                linewidths=1, linecolor='white',
                annot_kws={'fontsize': 9})

    ax.set_xlabel('Fold', fontweight='bold', fontsize=12)
    ax.set_ylabel('Ablation', fontweight='bold', fontsize=12)
    ax.set_title('Ablation Results Across Folds\n(Color = % change, Values = absolute loss)',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / 'ablation_heatmap.pdf')
    fig.savefig(output_dir / 'ablation_heatmap.png')
    plt.close()
    print("Saved: ablation_heatmap")


# =============================================================================
# Figure 3: Baseline Comparison
# =============================================================================
def fig_baseline_comparison(baseline_df: pd.DataFrame, full_results: dict, output_dir: Path):
    """Compare StageBridge to baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Loss comparison
    ax = axes[0]

    # Get StageBridge full model
    sb_loss = np.nanmean(full_results['transition_loss'])
    sb_std = np.nanstd(full_results['transition_loss'])

    # Aggregate baselines
    agg = baseline_df.groupby('baseline')['best_val_loss'].agg(['mean', 'std']).reset_index()

    # Add StageBridge
    sb_row = pd.DataFrame([{'baseline': 'stagebridge', 'mean': sb_loss, 'std': sb_std}])
    agg = pd.concat([agg, sb_row], ignore_index=True)
    agg = agg.sort_values('mean', ascending=False)

    colors = [BASELINE_COLORS.get(b, '#888') for b in agg['baseline']]
    labels = [BASELINE_LABELS.get(b, b) if b != 'stagebridge' else 'StageBridge' for b in agg['baseline']]

    y_pos = np.arange(len(agg))
    bars = ax.barh(y_pos, agg['mean'], xerr=agg['std'], color=colors,
                   edgecolor='white', linewidth=2, capsize=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_xlabel('Validation Loss', fontweight='bold', fontsize=12)
    ax.set_title('Model Comparison', fontweight='bold', fontsize=14)

    # Annotate with improvement
    for i, (mean, baseline) in enumerate(zip(agg['mean'], agg['baseline'])):
        if baseline != 'stagebridge':
            improvement = ((mean - sb_loss) / mean) * 100
            ax.text(mean + 0.005, i, f'{improvement:.1f}% vs SB', va='center', fontsize=9)

    # Panel B: Parameter efficiency
    ax = axes[1]

    # Get parameters
    params = baseline_df.groupby('baseline')['n_parameters'].first()

    # StageBridge params (from ablation data or estimate)
    sb_params = 20_450_003  # from ablation JSON

    param_data = params.to_dict()
    param_data['stagebridge'] = sb_params

    loss_data = agg.set_index('baseline')['mean'].to_dict()

    for baseline, n_params in param_data.items():
        loss = loss_data.get(baseline, np.nan)
        if np.isnan(loss):
            continue
        color = BASELINE_COLORS.get(baseline, '#888')
        label = BASELINE_LABELS.get(baseline, baseline) if baseline != 'stagebridge' else 'StageBridge'

        ax.scatter(n_params / 1e6, loss, s=200, c=color, label=label,
                   edgecolors='white', linewidth=2, zorder=5)

    ax.set_xlabel('Parameters (Millions)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Validation Loss', fontweight='bold', fontsize=12)
    ax.set_title('Parameter Efficiency', fontweight='bold', fontsize=14)
    ax.set_xscale('log')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('StageBridge vs Baselines', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / 'baseline_comparison.pdf')
    fig.savefig(output_dir / 'baseline_comparison.png')
    plt.close()
    print("Saved: baseline_comparison")


# =============================================================================
# Figure 4: Learning Curves
# =============================================================================
def fig_learning_curves(baseline_df: pd.DataFrame, output_dir: Path):
    """Learning curves for baselines."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    baselines = ['pooling', 'deepsets', 'set_transformer', 'graphsage']

    for ax, baseline in zip(axes.flat, baselines):
        subset = baseline_df[baseline_df['baseline'] == baseline]

        for _, row in subset.iterrows():
            history = row.get('history', {})
            if not history:
                continue

            train = history.get('train_loss', [])
            val = history.get('val_loss', [])

            if train and val:
                epochs = range(1, len(train) + 1)
                ax.plot(epochs, train, alpha=0.3, color=BASELINE_COLORS[baseline], linewidth=1)
                ax.plot(epochs, val, alpha=0.6, color=BASELINE_COLORS[baseline], linewidth=2)

        # Add legend
        ax.plot([], [], color=BASELINE_COLORS[baseline], alpha=0.3, linewidth=1, label='Train')
        ax.plot([], [], color=BASELINE_COLORS[baseline], alpha=0.6, linewidth=2, label='Val')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(BASELINE_LABELS[baseline], fontweight='bold', fontsize=14)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Baseline Learning Curves', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / 'baseline_learning_curves.pdf')
    fig.savefig(output_dir / 'baseline_learning_curves.png')
    plt.close()
    print("Saved: baseline_learning_curves")


# =============================================================================
# Figure 5: Ablation Categories
# =============================================================================
def fig_ablation_categories(ablation_df: pd.DataFrame, full_results: dict, output_dir: Path):
    """Group ablations by category for clearer interpretation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    full_loss = np.nanmean(full_results['transition_loss'])

    categories = {
        'Reference Ablations': ['hlca_only', 'luca_only', 'gw_barycentric', 'gw_project_hlca', 'gw_project_luca'],
        'Architecture Ablations': ['no_niche', 'no_gate', 'no_distance', 'no_context_refiner', 'no_ring_pooling'],
        'Training Ablations': ['frozen_encoder'],
    }

    for ax, (cat_name, ablations) in zip(axes, categories.items()):
        subset = ablation_df[ablation_df['ablation'].isin(ablations)]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cat_name, fontweight='bold')
            continue

        agg = subset.groupby('ablation')['transition_loss'].agg(['mean', 'std']).reset_index()
        agg = agg.sort_values('mean')

        colors = [ABLATION_COLORS.get(a, '#888') for a in agg['ablation']]
        labels = [ABLATION_LABELS.get(a, a) for a in agg['ablation']]

        y_pos = np.arange(len(agg))
        bars = ax.barh(y_pos, agg['mean'], xerr=agg['std'], color=colors,
                       edgecolor='white', linewidth=1.5, capsize=3)

        ax.axvline(full_loss, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Validation Loss', fontweight='bold')
        ax.set_title(cat_name, fontweight='bold', fontsize=14)

        # Annotate delta
        for i, mean in enumerate(agg['mean']):
            delta = ((mean - full_loss) / full_loss) * 100
            sign = '+' if delta > 0 else ''
            ax.text(mean + 0.0001, i, f'{sign}{delta:.1f}%', va='center', fontsize=9)

    fig.suptitle('Ablation Study by Category', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / 'ablation_categories.pdf')
    fig.savefig(output_dir / 'ablation_categories.png')
    plt.close()
    print("Saved: ablation_categories")


# =============================================================================
# Figure 6: Cross-Fold Consistency
# =============================================================================
def fig_cross_fold_consistency(ablation_df: pd.DataFrame, full_results: dict, output_dir: Path):
    """Show consistency of results across folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Boxplot of transition loss by fold
    ax = axes[0]

    # Combine full and ablation
    full_df = pd.DataFrame({
        'model': 'StageBridge',
        'fold': full_results['fold'],
        'transition_loss': full_results['transition_loss']
    })

    abl_df = ablation_df[['ablation', 'fold', 'transition_loss']].copy()
    abl_df['model'] = abl_df['ablation'].map(ABLATION_LABELS)

    combined = pd.concat([full_df[['model', 'fold', 'transition_loss']],
                          abl_df[['model', 'fold', 'transition_loss']]])

    # Top 5 models by mean
    top_models = combined.groupby('model')['transition_loss'].mean().nsmallest(6).index.tolist()
    combined_top = combined[combined['model'].isin(top_models)]

    sns.boxplot(data=combined_top, x='model', y='transition_loss', ax=ax,
                palette='viridis', width=0.6)
    sns.stripplot(data=combined_top, x='model', y='transition_loss', ax=ax,
                  color='black', alpha=0.5, size=4)

    ax.set_xlabel('')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Cross-Fold Variation (Top 6 Models)', fontweight='bold', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    # Panel B: Coefficient of variation
    ax = axes[1]

    cv = combined.groupby('model')['transition_loss'].agg(
        lambda x: x.std() / x.mean() * 100 if x.mean() > 0 else 0
    ).sort_values()

    colors = plt.cm.RdYlGn_r(plt.Normalize(0, cv.max())(cv.values))

    y_pos = np.arange(len(cv))
    bars = ax.barh(y_pos, cv.values, color=colors, edgecolor='white', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cv.index, fontsize=9)
    ax.set_xlabel('Coefficient of Variation (%)', fontweight='bold')
    ax.set_title('Result Stability Across Folds', fontweight='bold', fontsize=14)

    # Annotate
    for i, v in enumerate(cv.values):
        ax.text(v + 0.1, i, f'{v:.1f}%', va='center', fontsize=9)

    fig.suptitle('Cross-Fold Consistency Analysis', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / 'cross_fold_consistency.pdf')
    fig.savefig(output_dir / 'cross_fold_consistency.png')
    plt.close()
    print("Saved: cross_fold_consistency")


# =============================================================================
# Figure 7: Summary Statistics Table
# =============================================================================
def fig_summary_table(ablation_df: pd.DataFrame, baseline_df: pd.DataFrame,
                      full_results: dict, output_dir: Path):
    """Publication-ready summary table as figure."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    # Prepare data
    rows = []

    # Full model
    rows.append({
        'Model': 'StageBridge (Full)',
        'Type': 'Full',
        'Val Loss': f"{np.nanmean(full_results['transition_loss']):.4f} +/- {np.nanstd(full_results['transition_loss']):.4f}",
        'N': len(full_results['transition_loss']),
        'vs Full': '-',
    })

    full_mean = np.nanmean(full_results['transition_loss'])

    # Ablations
    for ablation in sorted(ablation_df['ablation'].unique()):
        subset = ablation_df[ablation_df['ablation'] == ablation]
        mean = subset['transition_loss'].mean()
        std = subset['transition_loss'].std()
        delta = ((mean - full_mean) / full_mean) * 100

        rows.append({
            'Model': ABLATION_LABELS.get(ablation, ablation),
            'Type': 'Ablation',
            'Val Loss': f"{mean:.4f} +/- {std:.4f}",
            'N': len(subset),
            'vs Full': f"{delta:+.1f}%",
        })

    # Baselines
    for baseline in sorted(baseline_df['baseline'].unique()):
        subset = baseline_df[baseline_df['baseline'] == baseline]
        mean = subset['best_val_loss'].mean()
        std = subset['best_val_loss'].std()
        delta = ((mean - full_mean) / full_mean) * 100

        rows.append({
            'Model': BASELINE_LABELS.get(baseline, baseline),
            'Type': 'Baseline',
            'Val Loss': f"{mean:.4f} +/- {std:.4f}",
            'N': len(subset),
            'vs Full': f"{delta:+.1f}%",
        })

    df = pd.DataFrame(rows)

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='left',
        colWidths=[0.3, 0.12, 0.25, 0.08, 0.12]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        elif df.iloc[row-1]['Type'] == 'Full':
            cell.set_facecolor('#d5f5e3')
        elif df.iloc[row-1]['Type'] == 'Baseline':
            cell.set_facecolor('#fdebd0')
        elif row % 2 == 0:
            cell.set_facecolor('#f8f9fa')

    ax.set_title('Model Performance Summary', fontweight='bold', fontsize=16, pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / 'summary_table.pdf')
    fig.savefig(output_dir / 'summary_table.png')
    plt.close()
    print("Saved: summary_table")


# =============================================================================
# Figure 8: Component Importance Ranking
# =============================================================================
def fig_component_importance(ablation_df: pd.DataFrame, full_results: dict, output_dir: Path):
    """Rank components by their importance (impact when removed)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    full_loss = np.nanmean(full_results['transition_loss'])

    # Compute importance as % increase when removed
    importance = []
    for ablation in ablation_df['ablation'].unique():
        subset = ablation_df[ablation_df['ablation'] == ablation]
        mean_loss = subset['transition_loss'].mean()
        delta = ((mean_loss - full_loss) / full_loss) * 100

        importance.append({
            'component': ABLATION_LABELS.get(ablation, ablation),
            'ablation': ablation,
            'importance': delta,
            'loss': mean_loss,
        })

    imp_df = pd.DataFrame(importance)
    imp_df = imp_df.sort_values('importance', ascending=False)

    # Only positive (harmful when removed = important)
    imp_df_pos = imp_df[imp_df['importance'] > 0]

    colors = [ABLATION_COLORS.get(a, '#888') for a in imp_df_pos['ablation']]

    y_pos = np.arange(len(imp_df_pos))
    bars = ax.barh(y_pos, imp_df_pos['importance'], color=colors,
                   edgecolor='white', linewidth=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df_pos['component'], fontsize=11)
    ax.set_xlabel('Importance (% Loss Increase When Removed)', fontweight='bold', fontsize=12)
    ax.set_title('Component Importance Ranking', fontweight='bold', fontsize=16)

    # Annotate
    for i, (imp, loss) in enumerate(zip(imp_df_pos['importance'], imp_df_pos['loss'])):
        ax.text(imp + 0.1, i, f'+{imp:.1f}% (loss={loss:.4f})', va='center', fontsize=10)

    ax.axvline(0, color='black', linewidth=1)

    plt.tight_layout()
    fig.savefig(output_dir / 'component_importance.pdf')
    fig.savefig(output_dir / 'component_importance.png')
    plt.close()
    print("Saved: component_importance")


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate training/ablation figures")
    parser.add_argument("--input", "-i", type=str, default="results/v1",
                        help="Input directory with training results")
    parser.add_argument("--output", "-o", type=str, default="figures/publication/training",
                        help="Output directory for figures")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Training & Ablation Figure Generation")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    print("\nLoading results...")

    # Load full model results
    full_results = load_full_results(input_dir)
    print(f"  Full model: {len(full_results.get('transition_loss', []))} runs")

    # Load ablation results
    ablation_df = load_ablation_results(input_dir)
    print(f"  Ablations: {len(ablation_df)} runs across {ablation_df['ablation'].nunique()} types")

    # Load baseline results
    baseline_df = load_baseline_results(input_dir)
    print(f"  Baselines: {len(baseline_df)} runs across {baseline_df['baseline'].nunique()} types")

    print("\n" + "-" * 70)
    print("Generating figures...")
    print("-" * 70)

    if len(full_results.get('transition_loss', [])) > 0 and len(ablation_df) > 0:
        fig_ablation_summary(ablation_df, full_results, output_dir)
        fig_ablation_heatmap(ablation_df, full_results, output_dir)
        fig_ablation_categories(ablation_df, full_results, output_dir)
        fig_component_importance(ablation_df, full_results, output_dir)
        fig_cross_fold_consistency(ablation_df, full_results, output_dir)

    if len(baseline_df) > 0 and len(full_results.get('transition_loss', [])) > 0:
        fig_baseline_comparison(baseline_df, full_results, output_dir)
        fig_learning_curves(baseline_df, output_dir)

    if len(ablation_df) > 0 or len(baseline_df) > 0:
        fig_summary_table(ablation_df, baseline_df, full_results, output_dir)

    print("\n" + "=" * 70)
    print(f"All figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
