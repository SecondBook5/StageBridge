#!/usr/bin/env python3
"""Generate publication-quality COMMOT cell-cell communication figures.

COMMOT (COMMunication analysis by Optimal Transport) uses optimal transport
to infer spatial cell-cell communication from spatial transcriptomics data.

Outputs:
- sender_scores.npy: [n_cells, n_lr_pairs] - sending potential per cell
- receiver_scores.npy: [n_cells, n_lr_pairs] - receiving potential per cell
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge, PathPatch, FancyArrowPatch
from matplotlib.path import Path as MPath
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

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

# Color palettes
FOLD_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
ACTIVITY_CMAP = 'viridis'
DIVERGING_CMAP = 'RdBu_r'


def load_commot_results(results_dir: Path) -> dict:
    """Load COMMOT results from a fold directory."""
    results = {}

    json_path = results_dir / "commot_results.json"
    if json_path.exists():
        with open(json_path) as f:
            results['metadata'] = json.load(f)

    sender_path = results_dir / "sender_scores.npy"
    receiver_path = results_dir / "receiver_scores.npy"

    if sender_path.exists():
        results['sender_scores'] = np.load(sender_path)
    if receiver_path.exists():
        results['receiver_scores'] = np.load(receiver_path)

    lr_path = results_dir / "lr_pairs.json"
    if lr_path.exists():
        with open(lr_path) as f:
            results['lr_pairs'] = json.load(f)

    return results


def load_all_folds(base_dir: Path) -> dict:
    """Load results from all complete folds."""
    all_results = {}

    for fold_dir in sorted(base_dir.glob("fold_*")):
        fold_idx = int(fold_dir.name.split("_")[1])
        results = load_commot_results(fold_dir)
        if results.get('sender_scores') is not None and results.get('receiver_scores') is not None:
            all_results[fold_idx] = results
            print(f"  Fold {fold_idx}: {results['sender_scores'].shape[0]:,} cells, "
                  f"{results['sender_scores'].shape[1]} L-R pairs")

    return all_results


# =============================================================================
# Figure 1: Score Distribution Violin Plot
# =============================================================================
def fig_score_distributions_violin(results: dict, output_dir: Path):
    """Violin plot of sender/receiver score distributions across folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data for violin plot
    sender_data = []
    receiver_data = []
    fold_labels = []

    for fold_idx, fold_data in sorted(results.items()):
        # Sample for visualization (log-transform non-zero)
        sender = fold_data['sender_scores'].flatten()
        receiver = fold_data['receiver_scores'].flatten()

        # Take non-zero values and log transform
        sender_nz = sender[sender > 0]
        receiver_nz = receiver[receiver > 0]

        # Subsample for plotting
        n_sample = min(50000, len(sender_nz))
        if n_sample > 0:
            idx_s = np.random.choice(len(sender_nz), n_sample, replace=False)
            idx_r = np.random.choice(len(receiver_nz), min(n_sample, len(receiver_nz)), replace=False)
            sender_data.extend(np.log10(sender_nz[idx_s]))
            receiver_data.extend(np.log10(receiver_nz[idx_r]))
            fold_labels.extend([f'Fold {fold_idx}'] * n_sample)

    # Create DataFrames
    sender_df = pd.DataFrame({'log10(Score)': sender_data, 'Fold': fold_labels[:len(sender_data)]})
    receiver_df = pd.DataFrame({'log10(Score)': receiver_data, 'Fold': fold_labels[:len(receiver_data)]})

    # Sender violin
    sns.violinplot(data=sender_df, x='Fold', y='log10(Score)', ax=axes[0],
                   palette=FOLD_COLORS[:len(results)], inner='quartile', cut=0)
    axes[0].set_title('Sender Score Distribution', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('log$_{10}$(Sender Score)', fontweight='bold')

    # Add median annotations
    medians = sender_df.groupby('Fold')['log10(Score)'].median()
    for i, (fold, med) in enumerate(medians.items()):
        axes[0].text(i, med + 0.1, f'{10**med:.2e}', ha='center', fontsize=9, fontweight='bold')

    # Receiver violin
    sns.violinplot(data=receiver_df, x='Fold', y='log10(Score)', ax=axes[1],
                   palette=FOLD_COLORS[:len(results)], inner='quartile', cut=0)
    axes[1].set_title('Receiver Score Distribution', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('log$_{10}$(Receiver Score)', fontweight='bold')

    medians = receiver_df.groupby('Fold')['log10(Score)'].median()
    for i, (fold, med) in enumerate(medians.items()):
        axes[1].text(i, med + 0.1, f'{10**med:.2e}', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('COMMOT Score Distributions (Non-zero Values)', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / 'commot_score_violins.pdf')
    fig.savefig(output_dir / 'commot_score_violins.png')
    plt.close()
    print("Saved: commot_score_violins")


# =============================================================================
# Figure 2: L-R Pair Activity Heatmap with Clustering
# =============================================================================
def fig_lr_activity_heatmap(results: dict, output_dir: Path, top_n: int = 50):
    """Clustered heatmap of top L-R pair activity across folds."""
    fold_data = next(iter(results.values()))

    # Find minimum number of L-R pairs across all folds
    min_pairs = min(fd['sender_scores'].shape[1] for fd in results.values())
    top_n = min(top_n, min_pairs)

    # Get total activity per L-R pair (using common pairs only)
    total_activity = fold_data['sender_scores'][:, :min_pairs].sum(axis=0) + \
                     fold_data['receiver_scores'][:, :min_pairs].sum(axis=0)
    top_idx = np.argsort(total_activity)[-top_n:][::-1]

    # Build matrix: L-R pairs x folds
    n_folds = len(results)
    activity_matrix = np.zeros((top_n, n_folds))

    for j, (fold_idx, fd) in enumerate(sorted(results.items())):
        s = fd['sender_scores'][:, top_idx].sum(axis=0)
        r = fd['receiver_scores'][:, top_idx].sum(axis=0)
        activity_matrix[:, j] = s + r

    # Normalize per fold for comparison
    activity_norm = activity_matrix / activity_matrix.sum(axis=0, keepdims=True)

    # Cluster L-R pairs
    if top_n > 2:
        linkage = hierarchy.linkage(pdist(activity_norm), method='ward')
        order = hierarchy.leaves_list(linkage)
    else:
        order = np.arange(top_n)

    activity_clustered = activity_norm[order, :]

    # Get labels
    lr_labels = [f'LR_{top_idx[i]}' for i in order]
    fold_labels = [f'Fold {i}' for i in sorted(results.keys())]

    # Plot
    fig, ax = plt.subplots(figsize=(8 + n_folds, 12))

    sns.heatmap(activity_clustered, ax=ax, cmap='YlOrRd',
                xticklabels=fold_labels, yticklabels=lr_labels,
                cbar_kws={'label': 'Normalized Activity', 'shrink': 0.5},
                linewidths=0.5, linecolor='white')

    ax.set_xlabel('Cross-Validation Fold', fontweight='bold', fontsize=12)
    ax.set_ylabel('Ligand-Receptor Pair', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {top_n} L-R Pairs Activity (Hierarchically Clustered)',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / 'commot_lr_heatmap_clustered.pdf')
    fig.savefig(output_dir / 'commot_lr_heatmap_clustered.png')
    plt.close()
    print("Saved: commot_lr_heatmap_clustered")


# =============================================================================
# Figure 3: Sender-Receiver Coupling Density Plot
# =============================================================================
def fig_sender_receiver_coupling(results: dict, output_dir: Path):
    """2D density plot of sender vs receiver activity per cell."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (fold_idx, fold_data) in zip(axes, sorted(results.items())):
        sender_total = fold_data['sender_scores'].sum(axis=1)
        receiver_total = fold_data['receiver_scores'].sum(axis=1)

        # Log transform for better visualization
        sender_log = np.log10(sender_total + 1)
        receiver_log = np.log10(receiver_total + 1)

        # 2D KDE
        sns.kdeplot(x=sender_log, y=receiver_log, ax=ax, cmap='magma',
                    fill=True, levels=20, thresh=0.05)

        # Add scatter for context
        n_sample = min(2000, len(sender_log))
        idx = np.random.choice(len(sender_log), n_sample, replace=False)
        ax.scatter(sender_log[idx], receiver_log[idx], alpha=0.1, s=5, c='white', zorder=1)

        # Correlation
        corr = np.corrcoef(sender_total, receiver_total)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.set_xlabel('log$_{10}$(Sender + 1)', fontweight='bold')
        ax.set_ylabel('log$_{10}$(Receiver + 1)', fontweight='bold')
        ax.set_title(f'Fold {fold_idx}', fontweight='bold', fontsize=14)

        # Identity line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'w--', alpha=0.5, linewidth=2, label='y=x')

    fig.suptitle('Sender-Receiver Coupling per Cell', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / 'commot_sender_receiver_density.pdf')
    fig.savefig(output_dir / 'commot_sender_receiver_density.png')
    plt.close()
    print("Saved: commot_sender_receiver_density")


# =============================================================================
# Figure 4: Cell Activity Classification
# =============================================================================
def fig_cell_activity_classification(results: dict, output_dir: Path):
    """Classify cells by their sender/receiver balance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fold_data = next(iter(results.values()))
    sender_total = fold_data['sender_scores'].sum(axis=1)
    receiver_total = fold_data['receiver_scores'].sum(axis=1)

    # Compute sender/receiver ratio (log scale)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.log2((sender_total + 1) / (receiver_total + 1))

    # Panel A: Distribution of ratios
    ax = axes[0]
    sns.histplot(ratio, bins=50, ax=ax, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Balanced')
    ax.axvline(-1, color='orange', linestyle=':', linewidth=2, label='Receiver-biased')
    ax.axvline(1, color='green', linestyle=':', linewidth=2, label='Sender-biased')

    ax.set_xlabel('log$_2$(Sender/Receiver)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cell Count', fontweight='bold', fontsize=12)
    ax.set_title('Sender-Receiver Balance Distribution', fontweight='bold', fontsize=14)
    ax.legend(frameon=False)

    # Panel B: Cell classification pie chart
    ax = axes[1]

    strong_sender = (ratio > 1).sum()
    weak_sender = ((ratio > 0) & (ratio <= 1)).sum()
    balanced = (np.abs(ratio) <= 0.1).sum()
    weak_receiver = ((ratio < 0) & (ratio >= -1)).sum()
    strong_receiver = (ratio < -1).sum()

    sizes = [strong_sender, weak_sender, balanced, weak_receiver, strong_receiver]
    labels = ['Strong Sender\n(ratio > 2)', 'Weak Sender\n(1 < ratio < 2)',
              'Balanced\n(ratio ~ 1)', 'Weak Receiver\n(0.5 < ratio < 1)',
              'Strong Receiver\n(ratio < 0.5)']
    colors = ['#27ae60', '#82e0aa', '#f7dc6f', '#f5b7b1', '#e74c3c']
    explode = [0.02, 0, 0, 0, 0.02]

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, explode=explode,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    ax.set_title('Cell Communication Phenotypes', fontweight='bold', fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / 'commot_cell_classification.pdf')
    fig.savefig(output_dir / 'commot_cell_classification.png')
    plt.close()
    print("Saved: commot_cell_classification")


# =============================================================================
# Figure 5: Top L-R Pairs Lollipop Chart
# =============================================================================
def fig_top_lr_lollipop(results: dict, output_dir: Path, top_n: int = 25):
    """Lollipop chart of top L-R pairs with sender/receiver breakdown."""
    fig, ax = plt.subplots(figsize=(10, 10))

    fold_data = next(iter(results.values()))

    sender_totals = fold_data['sender_scores'].sum(axis=0)
    receiver_totals = fold_data['receiver_scores'].sum(axis=0)
    combined = sender_totals + receiver_totals

    top_idx = np.argsort(combined)[-top_n:][::-1]

    labels = [f'LR_{i}' for i in top_idx]
    sender_vals = sender_totals[top_idx]
    receiver_vals = receiver_totals[top_idx]
    total_vals = combined[top_idx]

    y_pos = np.arange(len(labels))

    # Normalize for display
    max_val = total_vals.max()
    sender_norm = sender_vals / max_val
    receiver_norm = receiver_vals / max_val

    # Draw stems
    for i, (s, r, t) in enumerate(zip(sender_norm, receiver_norm, total_vals)):
        ax.plot([0, s + r], [i, i], color='#bdc3c7', linewidth=2, zorder=1)

    # Draw dots
    ax.scatter(sender_norm, y_pos, s=150, c='#3498db', label='Sender', zorder=3, edgecolors='white', linewidth=1.5)
    ax.scatter(sender_norm + receiver_norm, y_pos, s=150, c='#e74c3c', label='Receiver', zorder=3, edgecolors='white', linewidth=1.5)

    # Connect sender to receiver
    for i, (s, r) in enumerate(zip(sender_norm, receiver_norm)):
        ax.plot([s, s + r], [i, i], color='#e74c3c', linewidth=3, alpha=0.7, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel('Normalized Communication Score', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {top_n} L-R Pairs: Sender vs Receiver Contribution',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

    ax.set_xlim(-0.05, 1.1)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    fig.savefig(output_dir / 'commot_top_lr_lollipop.pdf')
    fig.savefig(output_dir / 'commot_top_lr_lollipop.png')
    plt.close()
    print("Saved: commot_top_lr_lollipop")


# =============================================================================
# Figure 6: Cross-Fold Reproducibility
# =============================================================================
def fig_fold_reproducibility(results: dict, output_dir: Path):
    """Pairwise correlation of L-R pair rankings across folds."""
    if len(results) < 2:
        print("  Skipping fold reproducibility: need at least 2 folds")
        return

    folds = sorted(results.keys())
    n_folds = len(folds)

    # Get L-R totals for each fold
    lr_totals = {}
    min_pairs = float('inf')
    for fold_idx in folds:
        fd = results[fold_idx]
        total = fd['sender_scores'].sum(axis=0) + fd['receiver_scores'].sum(axis=0)
        lr_totals[fold_idx] = total
        min_pairs = min(min_pairs, len(total))

    # Compute correlation matrix
    corr_matrix = np.zeros((n_folds, n_folds))
    for i, f1 in enumerate(folds):
        for j, f2 in enumerate(folds):
            # Use rank correlation (Spearman) for robustness
            r, _ = stats.spearmanr(lr_totals[f1][:min_pairs], lr_totals[f2][:min_pairs])
            corr_matrix[i, j] = r

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Correlation heatmap
    ax = axes[0]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, ax=ax, mask=mask, cmap='RdYlGn', center=0,
                annot=True, fmt='.3f', square=True,
                xticklabels=[f'Fold {f}' for f in folds],
                yticklabels=[f'Fold {f}' for f in folds],
                cbar_kws={'label': 'Spearman r', 'shrink': 0.8},
                linewidths=2, linecolor='white',
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})

    ax.set_title('L-R Pair Ranking Reproducibility', fontweight='bold', fontsize=14)

    # Panel B: Scatter of fold 0 vs fold 1
    ax = axes[1]
    if len(folds) >= 2:
        f0, f1 = folds[0], folds[1]
        x = np.log10(lr_totals[f0][:min_pairs] + 1)
        y = np.log10(lr_totals[f1][:min_pairs] + 1)

        ax.scatter(x, y, alpha=0.5, s=30, c='#9b59b6', edgecolors='white', linewidth=0.5)

        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

        # Identity line
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, 'k:', alpha=0.5, label='y = x')

        r, _ = stats.spearmanr(lr_totals[f0][:min_pairs], lr_totals[f1][:min_pairs])
        ax.text(0.05, 0.95, f'Spearman r = {r:.3f}', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(f'Fold {f0} log$_{{10}}$(Activity + 1)', fontweight='bold')
        ax.set_ylabel(f'Fold {f1} log$_{{10}}$(Activity + 1)', fontweight='bold')
        ax.set_title('L-R Activity: Fold 0 vs Fold 1', fontweight='bold', fontsize=14)
        ax.legend(loc='lower right', frameon=True)

    fig.suptitle('COMMOT Cross-Fold Reproducibility Analysis', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / 'commot_fold_reproducibility.pdf')
    fig.savefig(output_dir / 'commot_fold_reproducibility.png')
    plt.close()
    print("Saved: commot_fold_reproducibility")


# =============================================================================
# Figure 7: Communication Hub Analysis
# =============================================================================
def fig_communication_hubs(results: dict, output_dir: Path, top_n: int = 100):
    """Identify communication hub cells (high sender AND receiver)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fold_data = next(iter(results.values()))
    sender_total = fold_data['sender_scores'].sum(axis=1)
    receiver_total = fold_data['receiver_scores'].sum(axis=1)

    # Identify hubs (top percentile in both)
    sender_thresh = np.percentile(sender_total, 90)
    receiver_thresh = np.percentile(receiver_total, 90)

    is_hub = (sender_total >= sender_thresh) & (receiver_total >= receiver_thresh)
    is_sender_only = (sender_total >= sender_thresh) & (receiver_total < receiver_thresh)
    is_receiver_only = (sender_total < sender_thresh) & (receiver_total >= receiver_thresh)
    is_inactive = (sender_total < np.percentile(sender_total, 25)) & \
                  (receiver_total < np.percentile(receiver_total, 25))

    # Panel A: Scatter with hub highlighting
    ax = axes[0]

    # Background cells
    ax.scatter(np.log10(sender_total + 1), np.log10(receiver_total + 1),
               alpha=0.1, s=10, c='#bdc3c7', label='Other')

    # Sender-dominant
    ax.scatter(np.log10(sender_total[is_sender_only] + 1),
               np.log10(receiver_total[is_sender_only] + 1),
               alpha=0.6, s=30, c='#3498db', label=f'Sender-dominant ({is_sender_only.sum():,})')

    # Receiver-dominant
    ax.scatter(np.log10(sender_total[is_receiver_only] + 1),
               np.log10(receiver_total[is_receiver_only] + 1),
               alpha=0.6, s=30, c='#e74c3c', label=f'Receiver-dominant ({is_receiver_only.sum():,})')

    # Hubs
    ax.scatter(np.log10(sender_total[is_hub] + 1),
               np.log10(receiver_total[is_hub] + 1),
               alpha=0.8, s=60, c='#f39c12', edgecolors='black', linewidth=1,
               label=f'Communication Hubs ({is_hub.sum():,})')

    # Threshold lines
    ax.axhline(np.log10(receiver_thresh + 1), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(np.log10(sender_thresh + 1), color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('log$_{10}$(Sender Activity + 1)', fontweight='bold', fontsize=12)
    ax.set_ylabel('log$_{10}$(Receiver Activity + 1)', fontweight='bold', fontsize=12)
    ax.set_title('Communication Hub Identification', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', frameon=True, fontsize=9)

    # Panel B: Cell type breakdown
    ax = axes[1]

    categories = ['Hubs\n(High S+R)', 'Sender-\ndominant', 'Receiver-\ndominant', 'Inactive\n(Low S+R)', 'Other']
    counts = [is_hub.sum(), is_sender_only.sum(), is_receiver_only.sum(), is_inactive.sum(),
              len(sender_total) - is_hub.sum() - is_sender_only.sum() - is_receiver_only.sum() - is_inactive.sum()]
    colors = ['#f39c12', '#3498db', '#e74c3c', '#95a5a6', '#ecf0f1']

    bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=2)

    for bar, count in zip(bars, counts):
        pct = count / len(sender_total) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('Number of Cells', fontweight='bold', fontsize=12)
    ax.set_title('Cell Communication Phenotypes', fontweight='bold', fontsize=14)
    ax.set_ylim(0, max(counts) * 1.2)

    fig.suptitle('COMMOT Communication Hub Analysis', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / 'commot_communication_hubs.pdf')
    fig.savefig(output_dir / 'commot_communication_hubs.png')
    plt.close()
    print("Saved: commot_communication_hubs")


# =============================================================================
# Figure 8: L-R Pair Specificity vs Activity
# =============================================================================
def fig_lr_specificity(results: dict, output_dir: Path):
    """Plot L-R pair specificity (Gini coefficient) vs total activity."""
    fig, ax = plt.subplots(figsize=(10, 8))

    fold_data = next(iter(results.values()))
    combined = fold_data['sender_scores'] + fold_data['receiver_scores']

    # Compute Gini coefficient for each L-R pair (cell-level specificity)
    n_pairs = combined.shape[1]
    gini = np.zeros(n_pairs)
    activity = np.zeros(n_pairs)

    for i in range(n_pairs):
        vals = combined[:, i]
        vals_sorted = np.sort(vals)
        n = len(vals)
        cumsum = np.cumsum(vals_sorted)
        gini[i] = (2 * np.sum((np.arange(1, n+1) * vals_sorted)) - (n+1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)
        activity[i] = vals.sum()

    # Color by activity quantile
    activity_log = np.log10(activity + 1)
    colors = plt.cm.viridis(plt.Normalize()(activity_log))

    scatter = ax.scatter(activity_log, gini, c=activity_log, cmap='viridis',
                         s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Highlight extremes
    high_gini = gini > np.percentile(gini, 95)
    low_gini = gini < np.percentile(gini, 5)

    ax.scatter(activity_log[high_gini], gini[high_gini], c='red', s=100,
               marker='^', label=f'High specificity (top 5%)', edgecolors='black', linewidth=1, zorder=5)
    ax.scatter(activity_log[low_gini], gini[low_gini], c='blue', s=100,
               marker='v', label=f'Low specificity (bottom 5%)', edgecolors='black', linewidth=1, zorder=5)

    ax.set_xlabel('log$_{10}$(Total Activity + 1)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Gini Coefficient (Cell Specificity)', fontweight='bold', fontsize=12)
    ax.set_title('L-R Pair Specificity vs Activity', fontweight='bold', fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('log$_{10}$(Activity + 1)', fontweight='bold')

    ax.legend(loc='upper left', frameon=True)

    # Add interpretation text
    ax.text(0.98, 0.02, 'High Gini = concentrated in few cells\nLow Gini = broadly active',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            style='italic', color='#555',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / 'commot_lr_specificity.pdf')
    fig.savefig(output_dir / 'commot_lr_specificity.png')
    plt.close()
    print("Saved: commot_lr_specificity")


# =============================================================================
# Figure 9: Summary Statistics Panel
# =============================================================================
def fig_summary_panel(results: dict, output_dir: Path):
    """Multi-panel summary of COMMOT analysis."""
    fig = plt.figure(figsize=(16, 12))

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fold_data = next(iter(results.values()))
    meta = fold_data.get('metadata', {}).get('metrics', {})

    # Panel A: Key metrics table
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')

    sender = fold_data['sender_scores']
    receiver = fold_data['receiver_scores']

    metrics = [
        ['Cells Analyzed', f"{sender.shape[0]:,}"],
        ['L-R Pairs', f"{sender.shape[1]:,}"],
        ['Database', meta.get('database', 'CellChat')],
        ['Genes Used', f"{meta.get('n_genes', 'N/A'):,}"],
        ['Sender Sparsity', f"{(sender == 0).mean() * 100:.1f}%"],
        ['Receiver Sparsity', f"{(receiver == 0).mean() * 100:.1f}%"],
        ['Mean Sender/Cell', f"{sender.sum(axis=1).mean():.2f}"],
        ['Mean Receiver/Cell', f"{receiver.sum(axis=1).mean():.2f}"],
    ]

    table = ax.table(cellText=metrics, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='left', colWidths=[0.5, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#ecf0f1')

    ax.set_title('COMMOT Analysis Summary', fontweight='bold', fontsize=14, pad=20)

    # Panel B: Activity distribution
    ax = fig.add_subplot(gs[0, 1])
    total_activity = sender.sum(axis=1) + receiver.sum(axis=1)
    sns.histplot(np.log10(total_activity + 1), bins=50, ax=ax, color='#9b59b6', edgecolor='white')
    ax.axvline(np.log10(np.median(total_activity) + 1), color='red', linestyle='--',
               linewidth=2, label=f'Median: {np.median(total_activity):.1f}')
    ax.set_xlabel('log$_{10}$(Total Activity + 1)', fontweight='bold')
    ax.set_ylabel('Cell Count', fontweight='bold')
    ax.set_title('Per-Cell Communication Activity', fontweight='bold', fontsize=14)
    ax.legend(frameon=False)

    # Panel C: L-R pair activity distribution
    ax = fig.add_subplot(gs[0, 2])
    lr_activity = sender.sum(axis=0) + receiver.sum(axis=0)
    sns.histplot(np.log10(lr_activity + 1), bins=50, ax=ax, color='#e67e22', edgecolor='white')
    ax.axvline(np.log10(np.median(lr_activity) + 1), color='red', linestyle='--',
               linewidth=2, label=f'Median: {np.median(lr_activity):.1f}')
    ax.set_xlabel('log$_{10}$(L-R Pair Activity + 1)', fontweight='bold')
    ax.set_ylabel('L-R Pair Count', fontweight='bold')
    ax.set_title('Per-L-R Pair Communication', fontweight='bold', fontsize=14)
    ax.legend(frameon=False)

    # Panel D: Sender vs Receiver per cell (hexbin)
    ax = fig.add_subplot(gs[1, 0])
    sender_total = sender.sum(axis=1)
    receiver_total = receiver.sum(axis=1)
    hb = ax.hexbin(np.log10(sender_total + 1), np.log10(receiver_total + 1),
                   gridsize=40, cmap='YlOrRd', mincnt=1)
    ax.set_xlabel('log$_{10}$(Sender + 1)', fontweight='bold')
    ax.set_ylabel('log$_{10}$(Receiver + 1)', fontweight='bold')
    ax.set_title('Sender vs Receiver Activity', fontweight='bold', fontsize=14)
    plt.colorbar(hb, ax=ax, label='Cell Count')

    # Panel E: Top 10 L-R pairs
    ax = fig.add_subplot(gs[1, 1])
    top_10_idx = np.argsort(lr_activity)[-10:][::-1]
    top_10_vals = lr_activity[top_10_idx]
    top_10_labels = [f'LR_{i}' for i in top_10_idx]

    colors = plt.cm.viridis(np.linspace(0.8, 0.2, 10))
    bars = ax.barh(range(10), top_10_vals, color=colors, edgecolor='white')
    ax.set_yticks(range(10))
    ax.set_yticklabels(top_10_labels)
    ax.invert_yaxis()
    ax.set_xlabel('Total Communication Score', fontweight='bold')
    ax.set_title('Top 10 L-R Pairs', fontweight='bold', fontsize=14)

    # Panel F: Fold comparison (if multiple)
    ax = fig.add_subplot(gs[1, 2])
    if len(results) > 1:
        fold_means = []
        fold_labels = []
        for fold_idx, fd in sorted(results.items()):
            s_mean = fd['sender_scores'].mean()
            r_mean = fd['receiver_scores'].mean()
            fold_means.append([s_mean, r_mean])
            fold_labels.append(f'Fold {fold_idx}')

        fold_means = np.array(fold_means)
        x = np.arange(len(fold_labels))
        width = 0.35

        ax.bar(x - width/2, fold_means[:, 0], width, label='Sender', color='#3498db', edgecolor='white')
        ax.bar(x + width/2, fold_means[:, 1], width, label='Receiver', color='#e74c3c', edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels)
        ax.set_ylabel('Mean Score', fontweight='bold')
        ax.set_title('Cross-Fold Consistency', fontweight='bold', fontsize=14)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, 'Single fold\n(no comparison)', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='#888')
        ax.axis('off')

    fig.suptitle('COMMOT Cell-Cell Communication Analysis', fontweight='bold', fontsize=18, y=0.98)

    fig.savefig(output_dir / 'commot_summary_panel.pdf')
    fig.savefig(output_dir / 'commot_summary_panel.png')
    plt.close()
    print("Saved: commot_summary_panel")


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate publication-quality COMMOT figures")
    parser.add_argument("--input", "-i", type=str,
                        default="results/external/commot",
                        help="Input directory with COMMOT results")
    parser.add_argument("--output", "-o", type=str,
                        default="figures/publication/commot",
                        help="Output directory for figures")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMMOT Publication Figure Generation")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    print("\nLoading results...")
    results = load_all_folds(input_dir)

    if not results:
        print("ERROR: No complete COMMOT results found!")
        return

    print(f"\nLoaded {len(results)} complete fold(s)")

    print("\n" + "-" * 70)
    print("Generating figures...")
    print("-" * 70)

    fig_score_distributions_violin(results, output_dir)
    fig_lr_activity_heatmap(results, output_dir)
    fig_sender_receiver_coupling(results, output_dir)
    fig_cell_activity_classification(results, output_dir)
    fig_top_lr_lollipop(results, output_dir)
    fig_fold_reproducibility(results, output_dir)
    fig_communication_hubs(results, output_dir)
    fig_lr_specificity(results, output_dir)
    fig_summary_panel(results, output_dir)

    print("\n" + "=" * 70)
    print(f"All figures saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
