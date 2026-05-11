#!/usr/bin/env python3
"""Generate poster figures from existing results.

Usage:
    python scripts/generate_poster_figures.py --output-dir figs/poster
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Rich, saturated color palette
PALETTE = {
    'PoolingMLP': '#708090',      # slate
    'DeepSets': '#4682B4',        # steel blue
    'SetTransformer': '#4169E1',  # royal blue
    'GraphSAGE': '#8B008B',       # violet/magenta
    'StageBridge': '#228B22',     # forest green
    '128': '#4682B4',
    '256': '#228B22',
    'Learned GW': '#4169E1',
    'Precompute GW': '#8B008B',
    'Concat': '#DAA520',
    'OT-CFM': '#228B22',
    'Schrodinger Bridge': '#CB4154',
}

# Publication style
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})


def load_baseline_results(baselines_dir: Path) -> pd.DataFrame:
    """Load all baseline results into a DataFrame."""
    records = []
    for baseline_dir in baselines_dir.iterdir():
        if not baseline_dir.is_dir():
            continue
        baseline_name = baseline_dir.name
        for fold_dir in baseline_dir.glob('fold_*'):
            fold_idx = int(fold_dir.name.split('_')[1])
            for seed_dir in fold_dir.glob('seed_*'):
                seed = int(seed_dir.name.split('_')[1])
                json_file = seed_dir / f'baseline_{baseline_name}.json'
                if json_file.exists():
                    with open(json_file) as f:
                        data = json.load(f)
                    records.append({
                        'model': baseline_name,
                        'fold': fold_idx,
                        'seed': seed,
                        'val_loss': data['metrics']['best_val_loss'],
                        'n_parameters': data.get('n_parameters', 0),
                    })
    return pd.DataFrame(records)


def load_hpo_history(hpo_path: Path) -> dict:
    """Load HPO optimization history."""
    with open(hpo_path) as f:
        return json.load(f)


def add_significance_bracket(ax, x1, x2, y, p_value, height=0.03):
    """Add significance bracket between two x positions."""
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bracket_height = height * y_range

    ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y],
            color='black', linewidth=1.5, clip_on=False)

    if p_value < 0.001:
        p_text = 'p < 0.001'
    elif p_value < 0.01:
        p_text = f'p = {p_value:.3f}'
    else:
        p_text = f'p = {p_value:.2f}'

    ax.text((x1 + x2) / 2, y + bracket_height * 1.3, p_text,
            ha='center', va='bottom', fontsize=14, fontweight='bold')


def fig_baseline_comparison(baseline_df: pd.DataFrame, output_dir: Path):
    """Publication-quality violin plot comparing baselines - like the reference."""
    import matplotlib.colors as mcolors

    name_map = {
        'pooling': 'PoolingMLP',
        'deepsets': 'DeepSets',
        'set_transformer': 'SetTransformer',
        'graphsage': 'GraphSAGE',
    }
    baseline_df = baseline_df.copy()
    baseline_df['Model'] = baseline_df['model'].map(name_map)

    order = ['PoolingMLP', 'DeepSets', 'SetTransformer', 'GraphSAGE']
    palette = [PALETTE[m] for m in order]

    # Create lighter versions for violin fill
    light_palette = []
    for c in palette:
        rgb = mcolors.to_rgb(c)
        light_palette.append(tuple(min(1, x + 0.3) for x in rgb))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Violin plot with NO inner elements (we'll add our own boxplot)
    sns.violinplot(
        data=baseline_df,
        x='Model',
        y='val_loss',
        order=order,
        palette=light_palette,
        inner=None,
        linewidth=2,
        saturation=1.0,
        cut=0,
        bw_method=0.5,
        ax=ax,
    )

    # Add colored boxplots on top
    for i, (model, color) in enumerate(zip(order, palette)):
        data = baseline_df[baseline_df['Model'] == model]['val_loss'].values
        bp = ax.boxplot([data], positions=[i], widths=0.15,
                       patch_artist=True, manage_ticks=False, zorder=2)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_edgecolor('black')
        bp['boxes'][0].set_linewidth(1.5)
        bp['medians'][0].set_color('black')
        bp['medians'][0].set_linewidth(2)
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1.5)
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.5)
        for flier in bp['fliers']:
            flier.set_visible(False)

    # Overlay individual points
    for i, (model, color) in enumerate(zip(order, palette)):
        data = baseline_df[baseline_df['Model'] == model]['val_loss'].values
        np.random.seed(42 + i)
        jitter = np.random.uniform(-0.12, 0.12, len(data))
        ax.scatter(i + jitter, data, facecolor=color, edgecolor='black',
                  linewidth=0.8, s=60, alpha=0.75, zorder=3)

    # Labels with sample size
    counts = baseline_df.groupby('Model').size()
    labels = [f'{m}\n(n={counts[m]})' for m in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)

    ax.set_xlabel('')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Baseline Model Comparison', fontweight='bold', pad=20)

    # Significance test
    best_data = baseline_df[baseline_df['Model'] == 'PoolingMLP']['val_loss']
    worst_data = baseline_df[baseline_df['Model'] == 'DeepSets']['val_loss']
    _, p_value = stats.mannwhitneyu(best_data, worst_data, alternative='less')

    y_max = baseline_df['val_loss'].max()
    add_significance_bracket(ax, 0, 1, y_max * 1.02, p_value)

    ax.set_ylim(0, y_max * 1.2)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig_baseline_comparison.png')
    fig.savefig(output_dir / 'fig_baseline_comparison.pdf')
    plt.close(fig)
    print(f"Saved: fig_baseline_comparison.png/pdf")


def fig_hpo_convergence(hpo_data: dict, output_dir: Path):
    """Publication-quality HPO convergence plot."""
    trials = hpo_data['all_trials']
    values = [t['value'] for t in trials]

    running_best = []
    best_so_far = float('inf')
    for v in values:
        best_so_far = min(best_so_far, v)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(10, 7))

    # All trials - color by success/failure
    good_mask = np.array(values) < 1.0
    ax.scatter(np.where(good_mask)[0], np.array(values)[good_mask],
               s=100, c='#4682B4', edgecolor='black', linewidth=1.2,
               alpha=0.8, label='Successful trial', zorder=2)
    ax.scatter(np.where(~good_mask)[0], np.array(values)[~good_mask],
               s=100, c='#CB4154', edgecolor='black', linewidth=1.2,
               alpha=0.8, label='Failed trial', zorder=2)

    # Running best
    ax.plot(range(len(running_best)), running_best,
            color='#228B22', linewidth=3.5, label='Best so far', zorder=3)
    ax.fill_between(range(len(running_best)), running_best,
                    alpha=0.15, color='#228B22')

    ax.set_xlabel('Trial', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Hyperparameter Optimization (30 trials)', fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.set_yscale('log')
    ax.set_ylim(0.002, 50)
    ax.set_xlim(-1, 30)

    best_idx = values.index(min(values))
    ax.annotate(f'Best: {min(values):.4f}',
                xy=(best_idx, min(values)),
                xytext=(best_idx + 5, min(values) * 4),
                arrowprops=dict(arrowstyle='->', color='#228B22', lw=2.5),
                fontsize=14, fontweight='bold', color='#228B22')

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig_hpo_convergence.png')
    fig.savefig(output_dir / 'fig_hpo_convergence.pdf')
    plt.close(fig)
    print(f"Saved: fig_hpo_convergence.png/pdf")


def fig_hpo_param_importance(hpo_data: dict, output_dir: Path):
    """Show which HPO parameters mattered most - proper violin + boxplot style."""
    import matplotlib.colors as mcolors

    trials = hpo_data['all_trials']

    records = []
    for t in trials:
        records.append({
            'value': t['value'],
            'Hidden Dim': str(t['params']['hidden_dim']),
            'GW Fusion': t['params']['gw_fusion_type'].replace('_', ' ').title(),
            'Dynamics': 'OT-CFM' if t['params']['dynamics_type'] == 'ot_cfm' else 'Schrodinger Bridge',
        })
    df = pd.DataFrame(records)
    df_good = df[df['value'] < 1.0].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    def violin_box_strip(ax, data_df, x_col, y_col, order, colors, bw=0.5):
        """Helper to create violin + boxplot + strip plot combo."""
        # Light colors for violin
        light_colors = []
        for c in colors:
            rgb = mcolors.to_rgb(c)
            light_colors.append(tuple(min(1, x + 0.3) for x in rgb))

        # Violin with no inner
        sns.violinplot(data=data_df, x=x_col, y=y_col, order=order,
                      palette=light_colors, inner=None, linewidth=2,
                      cut=0, bw_method=bw, ax=ax)

        # Add boxplots
        for i, (cat, color) in enumerate(zip(order, colors)):
            vals = data_df[data_df[x_col] == cat][y_col].values
            if len(vals) > 0:
                bp = ax.boxplot([vals], positions=[i], widths=0.12,
                               patch_artist=True, manage_ticks=False, zorder=2)
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_edgecolor('black')
                bp['boxes'][0].set_linewidth(1.5)
                bp['medians'][0].set_color('black')
                bp['medians'][0].set_linewidth(2)
                for w in bp['whiskers']:
                    w.set_color('black')
                    w.set_linewidth(1.5)
                for c in bp['caps']:
                    c.set_color('black')
                    c.set_linewidth(1.5)
                for f in bp['fliers']:
                    f.set_visible(False)

        # Scatter points
        for i, (cat, color) in enumerate(zip(order, colors)):
            vals = data_df[data_df[x_col] == cat][y_col].values
            np.random.seed(42 + i)
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(i + jitter, vals, facecolor=color, edgecolor='black',
                      linewidth=0.8, s=50, alpha=0.75, zorder=3)

    # Hidden dim
    violin_box_strip(axes[0], df_good, 'Hidden Dim', 'value',
                    ['128', '256'], [PALETTE['128'], PALETTE['256']])
    axes[0].set_title('Hidden Dimension', fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Validation Loss', fontweight='bold')

    # GW fusion type
    gw_order = ['Learned Gw', 'Precompute Gw', 'Concat']
    violin_box_strip(axes[1], df_good, 'GW Fusion', 'value', gw_order,
                    [PALETTE['Learned GW'], PALETTE['Precompute GW'], PALETTE['Concat']])
    axes[1].set_title('Reference Fusion', fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(['Learned\nGW', 'Precompute\nGW', 'Concat'])

    # Dynamics type - ALL data including failures
    dyn_order = ['OT-CFM', 'Schrodinger Bridge']
    violin_box_strip(axes[2], df, 'Dynamics', 'value', dyn_order,
                    [PALETTE['OT-CFM'], PALETTE['Schrodinger Bridge']], bw=0.6)
    axes[2].set_title('Dynamics Type', fontweight='bold')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    axes[2].set_yscale('log')
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(['OT-CFM', 'Schrodinger\nBridge'])

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig_hpo_params.png')
    fig.savefig(output_dir / 'fig_hpo_params.pdf')
    plt.close(fig)
    print(f"Saved: fig_hpo_params.png/pdf")


def fig_training_curves(baseline_df: pd.DataFrame, baselines_dir: Path, output_dir: Path):
    """Publication-quality training curves with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 7))

    name_map = {'pooling': 'PoolingMLP', 'deepsets': 'DeepSets',
                'set_transformer': 'SetTransformer', 'graphsage': 'GraphSAGE'}

    for baseline in ['pooling', 'deepsets', 'set_transformer', 'graphsage']:
        all_curves = []
        for fold_dir in (baselines_dir / baseline).glob('fold_*'):
            for seed_dir in fold_dir.glob('seed_*'):
                json_path = seed_dir / f'baseline_{baseline}.json'
                if json_path.exists():
                    with open(json_path) as f:
                        data = json.load(f)
                    all_curves.append(data['history']['val_loss'])

        if all_curves:
            curves = np.array(all_curves)
            mean = np.mean(curves, axis=0)
            std = np.std(curves, axis=0)
            epochs = np.arange(len(mean))

            color = PALETTE[name_map[baseline]]
            ax.plot(epochs, mean, label=name_map[baseline], color=color, linewidth=3)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Training Curves (mean ± std, n=15)', fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig_training_curves.png')
    fig.savefig(output_dir / 'fig_training_curves.pdf')
    plt.close(fig)
    print(f"Saved: fig_training_curves.png/pdf")


def main():
    parser = argparse.ArgumentParser(description='Generate poster figures')
    parser.add_argument('--output-dir', type=Path, default=Path('figs/poster'))
    parser.add_argument('--results-dir', type=Path, default=Path('data/results'))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    baselines_dir = args.results_dir / 'baselines'
    hpo_path = args.results_dir / 'optimization_history.json'

    print("Loading baseline results...")
    baseline_df = load_baseline_results(baselines_dir)
    print(f"  Loaded {len(baseline_df)} baseline runs")

    print("Loading HPO history...")
    hpo_data = load_hpo_history(hpo_path)
    print(f"  Loaded {len(hpo_data['all_trials'])} trials")

    print("\nGenerating figures...")
    fig_hpo_convergence(hpo_data, args.output_dir)
    fig_hpo_param_importance(hpo_data, args.output_dir)
    fig_baseline_comparison(baseline_df, args.output_dir)
    fig_training_curves(baseline_df, baselines_dir, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
