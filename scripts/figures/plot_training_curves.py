"""Publication-quality training curves for StageBridge.

Professional figures with deep colors, proper uncertainty, and clean aesthetics.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import numpy as np
from scipy.ndimage import gaussian_filter1d

# === STYLE CONFIGURATION ===

# Deep, rich color palette
PALETTE = {
    'stagebridge': '#1e3a5f',     # Deep navy blue
    'stagebridge_light': '#3d6a99',
    'pooling': '#c44e52',          # Muted red
    'deepsets': '#8172b3',         # Muted purple
    'set_transformer': '#64a860',  # Forest green
    'graphsage': '#e5a84b',        # Golden amber
    'train': '#2d2d2d',            # Charcoal
    'val': '#1e3a5f',              # Navy
    'ssl': '#8b4513',              # Saddle brown
    'transition': '#1e3a5f',       # Navy
    'grid': '#e8e8e8',             # Light gray
    'background': '#fafafa',       # Off-white
}

LABELS = {
    'stagebridge': 'StageBridge (Ours)',
    'pooling': 'PoolingMLP',
    'deepsets': 'DeepSets',
    'set_transformer': 'SetTransformer',
    'graphsage': 'GraphSAGE',
}

# Figure dimensions (inches) - Nature single column = 89mm = 3.5in
SINGLE_COL = 3.5
DOUBLE_COL = 7.2
GOLDEN = 1.618


def setup_style():
    """Configure matplotlib for publication quality."""
    plt.rcParams.update({
        # Font configuration
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,

        # Figure
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.facecolor': 'white',

        # Axes
        'axes.facecolor': PALETTE['background'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.6,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Grid
        'grid.color': PALETTE['grid'],
        'grid.linewidth': 0.4,
        'grid.alpha': 0.7,

        # Ticks
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Lines
        'lines.linewidth': 1.5,
        'lines.antialiased': True,

        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': False,
    })


def load_stagebridge_metrics(results_dir: Path) -> list:
    """Load all StageBridge full model metrics."""
    metrics = []
    for metrics_file in results_dir.glob('full/fold_*/seed_*/logs/metrics.json'):
        with open(metrics_file) as f:
            data = json.load(f)
            fold = int(metrics_file.parent.parent.parent.name.split('_')[1])
            seed = int(metrics_file.parent.parent.name.split('_')[1])
            metrics.append({
                'fold': fold,
                'seed': seed,
                'history': data['history'],
            })
    return metrics


def load_baseline_metrics(results_dir: Path, baseline: str) -> list:
    """Load all runs for a baseline."""
    metrics = []
    for json_file in results_dir.glob(f'baselines/{baseline}/fold_*/seed_*/*.json'):
        with open(json_file) as f:
            data = json.load(f)
            metrics.append({
                'fold': data['fold_idx'],
                'seed': data['seed'],
                'history': data['history'],
            })
    return metrics


def smooth(y, sigma=2):
    """Gaussian smoothing for cleaner curves."""
    return gaussian_filter1d(y.astype(float), sigma=sigma)


def plot_with_uncertainty(ax, x, curves, color, label=None, alpha_fill=0.15,
                          alpha_lines=0.08, smooth_sigma=1.5, linewidth=1.8):
    """Plot mean curve with uncertainty ribbon and faint individual traces."""
    curves = np.array(curves)

    # Individual traces (very faint)
    for curve in curves:
        smoothed = smooth(curve, smooth_sigma) if smooth_sigma > 0 else curve
        ax.plot(x, smoothed, color=color, alpha=alpha_lines, linewidth=0.5, zorder=1)

    # Mean and std
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    # Smooth for display
    if smooth_sigma > 0:
        mean_smooth = smooth(mean, smooth_sigma)
        std_smooth = smooth(std, smooth_sigma)
    else:
        mean_smooth, std_smooth = mean, std

    # Uncertainty ribbon with gradient effect
    for i, alpha_mult in enumerate([0.3, 0.5, 0.7, 1.0]):
        width = (4 - i) / 4
        ax.fill_between(x,
                       mean_smooth - std_smooth * width,
                       mean_smooth + std_smooth * width,
                       color=color, alpha=alpha_fill * alpha_mult,
                       linewidth=0, zorder=2)

    # Mean line
    line, = ax.plot(x, mean_smooth, color=color, linewidth=linewidth,
                    label=label, zorder=10, solid_capstyle='round')

    return line


def format_axis(ax, xlabel, ylabel, title=None, title_loc='left'):
    """Apply consistent axis formatting."""
    ax.set_xlabel(xlabel, fontweight='medium')
    ax.set_ylabel(ylabel, fontweight='medium')
    if title:
        ax.set_title(title, fontweight='bold', loc=title_loc, pad=8)

    # Subtle spine styling
    for spine in ax.spines.values():
        spine.set_color('#666666')


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add panel label (a, b, c, etc.)"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left',
            color='#1a1a1a')


# === MAIN FIGURES ===

def fig_stagebridge_training(results_dir: Path, output_dir: Path):
    """Figure 2: StageBridge two-stage training dynamics."""
    setup_style()

    metrics = load_stagebridge_metrics(results_dir)
    if not metrics:
        print("No StageBridge metrics found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    # Collect curves
    ssl_train = np.array([m['history']['train_loss'][:50] for m in metrics])
    ssl_val = np.array([m['history']['val_loss'][:50] for m in metrics])

    trans_train = [m['history']['train_loss'][50:] for m in metrics
                   if len(m['history']['train_loss']) > 50]
    trans_val = [m['history']['val_loss'][50:] for m in metrics
                 if len(m['history']['val_loss']) > 50]

    # === Panel A: SSL Phase ===
    epochs_ssl = np.arange(50)

    # Use log scale for SSL
    ax1.set_yscale('log')

    plot_with_uncertainty(ax1, epochs_ssl, ssl_train, PALETTE['train'],
                         label='Train', smooth_sigma=1)
    plot_with_uncertainty(ax1, epochs_ssl, ssl_val, PALETTE['val'],
                         label='Validation', smooth_sigma=1)

    format_axis(ax1, 'Epoch', 'Reconstruction Loss (MSE)',
                'Stage 1: Self-Supervised Pretraining')
    ax1.legend(loc='upper right', borderpad=0.5)
    ax1.set_xlim(0, 49)
    add_panel_label(ax1, 'a')

    # === Panel B: Transition Phase ===
    if trans_train:
        max_len = max(len(c) for c in trans_train)
        trans_train_padded = np.array([
            np.pad(c, (0, max_len - len(c)), mode='edge') for c in trans_train
        ])
        trans_val_padded = np.array([
            np.pad(c, (0, max_len - len(c)), mode='edge') for c in trans_val
        ])
        epochs_trans = np.arange(max_len)

        plot_with_uncertainty(ax2, epochs_trans, trans_train_padded,
                             PALETTE['train'], label='Train', smooth_sigma=2)
        plot_with_uncertainty(ax2, epochs_trans, trans_val_padded,
                             PALETTE['val'], label='Validation', smooth_sigma=2)

    format_axis(ax2, 'Epoch', 'Flow Matching Loss',
                'Stage 2: Transition Learning (OT-CFM)')
    ax2.legend(loc='upper right', borderpad=0.5)
    add_panel_label(ax2, 'b')

    # Add n= annotation
    ax1.text(0.98, 0.02, f'n={len(metrics)} runs', transform=ax1.transAxes,
             fontsize=6, ha='right', va='bottom', color='#666666')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig2_training_dynamics.pdf')
    fig.savefig(output_dir / 'fig2_training_dynamics.png')
    plt.close()
    print(f"Saved: fig2_training_dynamics.pdf")


def fig_baseline_comparison(results_dir: Path, output_dir: Path):
    """Figure 3: Baseline comparison - 2x2 grid with train/val curves."""
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.5))
    axes = axes.flatten()

    baselines = ['pooling', 'deepsets', 'set_transformer', 'graphsage']
    panel_labels = ['a', 'b', 'c', 'd']

    for idx, baseline in enumerate(baselines):
        ax = axes[idx]
        metrics = load_baseline_metrics(results_dir, baseline)

        if not metrics:
            continue

        # Collect and pad curves
        train_curves = [m['history']['train_loss'] for m in metrics]
        val_curves = [m['history']['val_loss'] for m in metrics]

        max_len = max(len(c) for c in train_curves)
        train = np.array([np.pad(c, (0, max_len-len(c)), mode='edge') for c in train_curves])
        val = np.array([np.pad(c, (0, max_len-len(c)), mode='edge') for c in val_curves])
        epochs = np.arange(max_len)

        plot_with_uncertainty(ax, epochs, train, PALETTE['train'],
                             label='Train', smooth_sigma=2)
        plot_with_uncertainty(ax, epochs, val, PALETTE[baseline],
                             label='Validation', smooth_sigma=2)

        format_axis(ax, 'Epoch', 'Loss', LABELS[baseline])
        add_panel_label(ax, panel_labels[idx])

        # Only show legend on first panel
        if idx == 0:
            ax.legend(loc='upper right', borderpad=0.5)

        # Add n= annotation
        ax.text(0.98, 0.02, f'n={len(metrics)}', transform=ax.transAxes,
                fontsize=6, ha='right', va='bottom', color='#666666')

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_baselines.pdf')
    fig.savefig(output_dir / 'fig3_baselines.png')
    plt.close()
    print(f"Saved: fig3_baselines.pdf")


def fig_method_comparison(results_dir: Path, output_dir: Path):
    """Figure 4: All methods validation loss comparison."""
    setup_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL))

    baselines = ['pooling', 'deepsets', 'set_transformer', 'graphsage']

    # Plot baselines
    for baseline in baselines:
        metrics = load_baseline_metrics(results_dir, baseline)
        if not metrics:
            continue

        val_curves = [m['history']['val_loss'] for m in metrics]
        max_len = max(len(c) for c in val_curves)
        val = np.array([np.pad(c, (0, max_len-len(c)), mode='edge') for c in val_curves])
        epochs = np.arange(max_len)

        mean = smooth(val.mean(axis=0), 2)
        std = smooth(val.std(axis=0), 2)

        ax.fill_between(epochs, mean - std, mean + std,
                       color=PALETTE[baseline], alpha=0.12)
        ax.plot(epochs, mean, color=PALETTE[baseline], linewidth=1.2,
               label=LABELS[baseline], alpha=0.9)

    # Plot StageBridge (transition phase)
    sb_metrics = load_stagebridge_metrics(results_dir)
    if sb_metrics:
        trans_val = [m['history']['val_loss'][50:] for m in sb_metrics
                     if len(m['history']['val_loss']) > 50]
        if trans_val:
            max_len = max(len(c) for c in trans_val)
            val = np.array([np.pad(c, (0, max_len-len(c)), mode='edge') for c in trans_val])
            epochs = np.arange(max_len)

            mean = val.mean(axis=0)
            std = val.std(axis=0)

            # Gradient ribbon for StageBridge
            for i, alpha in enumerate([0.08, 0.12, 0.18]):
                width = (3 - i) / 3
                ax.fill_between(epochs, mean - std * width, mean + std * width,
                               color=PALETTE['stagebridge'], alpha=alpha)

            ax.plot(epochs, mean, color=PALETTE['stagebridge'], linewidth=2.5,
                   label=LABELS['stagebridge'], zorder=10)

    format_axis(ax, 'Epoch', 'Validation Loss', 'Method Comparison')
    ax.set_ylim(bottom=0)

    # Custom legend with better spacing
    leg = ax.legend(loc='upper right', borderpad=0.6, labelspacing=0.4)
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_comparison.pdf')
    fig.savefig(output_dir / 'fig4_comparison.png')
    plt.close()
    print(f"Saved: fig4_comparison.pdf")


def fig_final_performance(results_dir: Path, output_dir: Path):
    """Figure 5: Final performance bar chart."""
    setup_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))

    methods = ['pooling', 'deepsets', 'set_transformer', 'graphsage', 'stagebridge']
    means, stds, colors = [], [], []

    for method in methods:
        if method == 'stagebridge':
            metrics = load_stagebridge_metrics(results_dir)
            finals = [min(m['history']['val_loss'][50:]) for m in metrics
                     if len(m['history']['val_loss']) > 50] if metrics else []
        else:
            metrics = load_baseline_metrics(results_dir, method)
            finals = [min(m['history']['val_loss']) for m in metrics] if metrics else []

        means.append(np.mean(finals) if finals else 0)
        stds.append(np.std(finals) if finals else 0)
        colors.append(PALETTE[method])

    x = np.arange(len(methods))

    # Create bars with subtle gradient effect
    bars = ax.bar(x, means, yerr=stds, capsize=3,
                  color=colors, edgecolor='white', linewidth=1,
                  error_kw={'linewidth': 1, 'capthick': 1, 'color': '#444444'})

    # Highlight StageBridge bar
    bars[-1].set_edgecolor(PALETTE['stagebridge'])
    bars[-1].set_linewidth(2)

    # Value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        if mean > 0:
            # Format based on magnitude
            if mean < 0.01:
                label = f'{mean:.4f}'
            else:
                label = f'{mean:.3f}'

            va = 'bottom'
            y_pos = bar.get_height() + std + max(means) * 0.02

            ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                   ha='center', va=va, fontsize=7, fontweight='medium',
                   color='#333333')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m].replace(' (Ours)', '\n(Ours)') for m in methods],
                       rotation=0, ha='center')
    ax.set_ylabel('Best Validation Loss', fontweight='medium')
    ax.set_title('Final Performance', fontweight='bold', loc='left', pad=10)

    # Reference line at StageBridge level
    ax.axhline(y=means[-1], color=PALETTE['stagebridge'], linestyle='--',
               linewidth=0.8, alpha=0.5, zorder=0)

    # Set y limit with some headroom
    ax.set_ylim(0, max(means) * 1.35)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_performance.pdf')
    fig.savefig(output_dir / 'fig5_performance.png')
    plt.close()
    print(f"Saved: fig5_performance.pdf")


def fig_combined(results_dir: Path, output_dir: Path):
    """Combined multi-panel figure for main text."""
    setup_style()

    fig = plt.figure(figsize=(DOUBLE_COL, 6))

    # Create grid: 2 rows, top row split into 2, bottom row split into 2
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1],
                          hspace=0.32, wspace=0.28,
                          left=0.08, right=0.98, top=0.95, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])  # SSL
    ax2 = fig.add_subplot(gs[0, 1])  # Transition
    ax3 = fig.add_subplot(gs[1, 0])  # Comparison
    ax4 = fig.add_subplot(gs[1, 1])  # Bar chart

    metrics = load_stagebridge_metrics(results_dir)

    # === Panel A: SSL ===
    if metrics:
        ssl_train = np.array([m['history']['train_loss'][:50] for m in metrics])
        ssl_val = np.array([m['history']['val_loss'][:50] for m in metrics])
        epochs = np.arange(50)

        ax1.set_yscale('log')
        plot_with_uncertainty(ax1, epochs, ssl_train, PALETTE['train'],
                             'Train', smooth_sigma=1)
        plot_with_uncertainty(ax1, epochs, ssl_val, PALETTE['val'],
                             'Validation', smooth_sigma=1)

    format_axis(ax1, 'Epoch', 'Reconstruction Loss', 'SSL Pretraining')
    ax1.legend(loc='upper right', fontsize=6, borderpad=0.4)
    ax1.set_xlim(0, 49)
    add_panel_label(ax1, 'a')

    # === Panel B: Transition ===
    if metrics:
        trans_train = [m['history']['train_loss'][50:] for m in metrics
                       if len(m['history']['train_loss']) > 50]
        trans_val = [m['history']['val_loss'][50:] for m in metrics
                     if len(m['history']['val_loss']) > 50]

        if trans_train:
            max_len = max(len(c) for c in trans_train)
            train_arr = np.array([np.pad(c, (0, max_len-len(c)), mode='edge')
                                  for c in trans_train])
            val_arr = np.array([np.pad(c, (0, max_len-len(c)), mode='edge')
                                for c in trans_val])
            epochs = np.arange(max_len)

            plot_with_uncertainty(ax2, epochs, train_arr, PALETTE['train'],
                                 'Train', smooth_sigma=2)
            plot_with_uncertainty(ax2, epochs, val_arr, PALETTE['val'],
                                 'Validation', smooth_sigma=2)

    format_axis(ax2, 'Epoch', 'Flow Matching Loss', 'Transition Learning')
    ax2.legend(loc='upper right', fontsize=6, borderpad=0.4)
    add_panel_label(ax2, 'b')

    # === Panel C: Method Comparison ===
    baselines = ['pooling', 'deepsets', 'set_transformer', 'graphsage']

    for baseline in baselines:
        bl_metrics = load_baseline_metrics(results_dir, baseline)
        if bl_metrics:
            val_curves = [m['history']['val_loss'] for m in bl_metrics]
            max_len = max(len(c) for c in val_curves)
            val = np.array([np.pad(c, (0, max_len-len(c)), mode='edge')
                           for c in val_curves])
            epochs = np.arange(max_len)

            mean = smooth(val.mean(axis=0), 2)
            ax3.plot(epochs, mean, color=PALETTE[baseline], linewidth=1.0,
                    label=LABELS[baseline], alpha=0.85)

    # StageBridge
    if metrics:
        trans_val = [m['history']['val_loss'][50:] for m in metrics
                     if len(m['history']['val_loss']) > 50]
        if trans_val:
            max_len = max(len(c) for c in trans_val)
            val = np.array([np.pad(c, (0, max_len-len(c)), mode='edge')
                           for c in trans_val])
            epochs = np.arange(max_len)
            mean = val.mean(axis=0)
            std = val.std(axis=0)

            ax3.fill_between(epochs, mean - std, mean + std,
                            color=PALETTE['stagebridge'], alpha=0.2)
            ax3.plot(epochs, mean, color=PALETTE['stagebridge'], linewidth=2,
                    label=LABELS['stagebridge'], zorder=10)

    format_axis(ax3, 'Epoch', 'Validation Loss', 'Method Comparison')
    ax3.legend(loc='upper right', fontsize=5.5, borderpad=0.4, ncol=1)
    ax3.set_ylim(bottom=0)
    add_panel_label(ax3, 'c')

    # === Panel D: Bar Chart ===
    methods = ['pooling', 'deepsets', 'set_transformer', 'graphsage', 'stagebridge']
    means, stds, colors = [], [], []

    for method in methods:
        if method == 'stagebridge':
            finals = [min(m['history']['val_loss'][50:]) for m in metrics
                     if len(m['history']['val_loss']) > 50] if metrics else []
        else:
            bl_metrics = load_baseline_metrics(results_dir, method)
            finals = [min(m['history']['val_loss']) for m in bl_metrics] if bl_metrics else []

        means.append(np.mean(finals) if finals else 0)
        stds.append(np.std(finals) if finals else 0)
        colors.append(PALETTE[method])

    x = np.arange(len(methods))
    bars = ax4.bar(x, means, yerr=stds, capsize=2, color=colors,
                   edgecolor='white', linewidth=0.8,
                   error_kw={'linewidth': 0.8, 'capthick': 0.8, 'color': '#444444'})
    bars[-1].set_edgecolor(PALETTE['stagebridge'])
    bars[-1].set_linewidth(1.5)

    # Value labels
    for bar, mean, std in zip(bars, means, stds):
        if mean > 0:
            label = f'{mean:.4f}' if mean < 0.01 else f'{mean:.3f}'
            ax4.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + max(means) * 0.03,
                    label, ha='center', va='bottom', fontsize=5.5)

    ax4.set_xticks(x)
    ax4.set_xticklabels([LABELS[m].replace(' (Ours)', '\n(Ours)').replace('Set', 'Set\n')
                         for m in methods], fontsize=6, rotation=0, ha='center')
    ax4.set_ylabel('Best Validation Loss', fontsize=8)
    ax4.set_title('Final Performance', fontweight='bold', loc='left')
    ax4.set_ylim(0, max(means) * 1.4)
    ax4.axhline(y=means[-1], color=PALETTE['stagebridge'], linestyle='--',
               linewidth=0.6, alpha=0.4, zorder=0)
    add_panel_label(ax4, 'd')

    fig.savefig(output_dir / 'fig2_combined.pdf')
    fig.savefig(output_dir / 'fig2_combined.png')
    plt.close()
    print(f"Saved: fig2_combined.pdf")


def print_table(results_dir: Path):
    """Print results table."""
    print("\n" + "=" * 70)
    print(f"{'Method':<22} {'Val Loss':>12} {'Std':>10} {'Improvement':>12} {'N':>5}")
    print("=" * 70)

    methods = ['pooling', 'deepsets', 'set_transformer', 'graphsage']
    results = []

    for method in methods:
        metrics = load_baseline_metrics(results_dir, method)
        if metrics:
            finals = [min(m['history']['val_loss']) for m in metrics]
            results.append((LABELS[method], np.mean(finals), np.std(finals), len(finals)))

    # StageBridge
    sb_metrics = load_stagebridge_metrics(results_dir)
    if sb_metrics:
        finals = [min(m['history']['val_loss'][50:]) for m in sb_metrics
                 if len(m['history']['val_loss']) > 50]
        if finals:
            sb_mean = np.mean(finals)
            results.append((LABELS['stagebridge'], sb_mean, np.std(finals), len(finals)))

    # Sort by loss (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # Get StageBridge loss for improvement calc
    sb_loss = next((r[1] for r in results if 'StageBridge' in r[0]), None)

    for name, mean, std, n in results:
        if sb_loss and 'StageBridge' not in name:
            improvement = f'{mean/sb_loss:.1f}x'
        else:
            improvement = '-'

        if 'StageBridge' in name:
            print(f"\033[1m{name:<22} {mean:>12.5f} {std:>10.5f} {improvement:>12} {n:>5}\033[0m")
        else:
            print(f"{name:<22} {mean:>12.5f} {std:>10.5f} {improvement:>12} {n:>5}")

    print("=" * 70)


if __name__ == '__main__':
    results_dir = Path('/home/booka/projects/StageBridge/results/v1')
    output_dir = Path('/home/booka/projects/StageBridge/figures/publication')

    print("Generating publication figures...\n")

    fig_stagebridge_training(results_dir, output_dir)
    fig_baseline_comparison(results_dir, output_dir)
    fig_method_comparison(results_dir, output_dir)
    fig_final_performance(results_dir, output_dir)
    fig_combined(results_dir, output_dir)

    print_table(results_dir)
    print("\nDone!")
