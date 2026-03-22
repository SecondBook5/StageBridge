#!/usr/bin/env python3
"""Generate publication-quality training curve visualizations.

Reads metrics CSV files from training runs and generates:
- Train vs Val loss curves (detect overfitting)
- Learning rate schedule visualization
- GPU memory usage over time
- Multi-fold comparison plots

Usage:
    python workflow/scripts/plot_training_curves.py \
        --input_dir /path/to/training \
        --output_dir /path/to/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication-quality settings
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# Color palette
COLORS = {
    'train': '#2563EB',      # Blue
    'val': '#DC2626',        # Red
    'ssl': '#059669',        # Green
    'transition': '#7C3AED', # Purple
    'lr': '#F59E0B',         # Amber
    'memory': '#6366F1',     # Indigo
}


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load metrics CSV file."""
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)


def plot_loss_curves(df: pd.DataFrame, output_path: Path, title: str = "Training Curves"):
    """Plot train vs validation loss curves."""
    if df.empty or 'train_loss' not in df.columns:
        print("  Skipping loss curves - no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Split by phase
    ssl_df = df[df['phase'] == 'ssl'] if 'phase' in df.columns else df
    trans_df = df[df['phase'] == 'transition'] if 'phase' in df.columns else pd.DataFrame()

    # SSL phase
    ax = axes[0]
    if not ssl_df.empty:
        epochs = ssl_df['epoch'].values
        ax.plot(epochs, ssl_df['train_loss'], color=COLORS['train'], label='Train', linewidth=2)
        if 'val_loss' in ssl_df.columns:
            ax.plot(epochs, ssl_df['val_loss'], color=COLORS['val'], label='Val', linewidth=2)

        # Mark best epoch
        if 'val_loss' in ssl_df.columns:
            best_idx = ssl_df['val_loss'].idxmin()
            best_epoch = ssl_df.loc[best_idx, 'epoch']
            best_val = ssl_df.loc[best_idx, 'val_loss']
            ax.axvline(best_epoch, color='gray', linestyle='--', alpha=0.5)
            ax.scatter([best_epoch], [best_val], color=COLORS['val'], s=100, zorder=5, marker='*')
            ax.annotate(f'Best: {best_val:.4f}', (best_epoch, best_val),
                       xytext=(10, 10), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Stage 1: SSL Pretraining')
    ax.legend(loc='upper right')
    ax.set_yscale('log')

    # Transition phase
    ax = axes[1]
    if not trans_df.empty:
        epochs = trans_df['epoch'].values
        ax.plot(epochs, trans_df['train_loss'], color=COLORS['train'], label='Train', linewidth=2)
        if 'val_loss' in trans_df.columns:
            ax.plot(epochs, trans_df['val_loss'], color=COLORS['val'], label='Val', linewidth=2)

        # Mark best epoch
        if 'val_loss' in trans_df.columns:
            best_idx = trans_df['val_loss'].idxmin()
            best_epoch = trans_df.loc[best_idx, 'epoch']
            best_val = trans_df.loc[best_idx, 'val_loss']
            ax.axvline(best_epoch, color='gray', linestyle='--', alpha=0.5)
            ax.scatter([best_epoch], [best_val], color=COLORS['val'], s=100, zorder=5, marker='*')
            ax.annotate(f'Best: {best_val:.4f}', (best_epoch, best_val),
                       xytext=(10, 10), textcoords='offset points', fontsize=9)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Stage 2: Transition Model')
        ax.legend(loc='upper right')
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No transition data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Stage 2: Transition Model')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_lr_schedule(df: pd.DataFrame, output_path: Path):
    """Plot learning rate schedule over training."""
    if df.empty or 'learning_rate' not in df.columns:
        print("  Skipping LR plot - no data")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    epochs = df['epoch'].values
    lr = df['learning_rate'].values

    ax.plot(epochs, lr, color=COLORS['lr'], linewidth=2)
    ax.fill_between(epochs, 0, lr, alpha=0.2, color=COLORS['lr'])

    # Mark phase transitions
    if 'phase' in df.columns:
        ssl_epochs = df[df['phase'] == 'ssl']['epoch'].max()
        if pd.notna(ssl_epochs):
            ax.axvline(ssl_epochs, color='gray', linestyle='--', alpha=0.7)
            ax.text(ssl_epochs, ax.get_ylim()[1] * 0.9, ' Transition\n Phase',
                   fontsize=9, va='top')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Warmup + Cosine Decay)')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_gpu_memory(df: pd.DataFrame, output_path: Path):
    """Plot GPU memory usage over training."""
    mem_col = None
    for col in ['gpu_memory_allocated_gb', 'gpu_memory_gb', 'memory_gb']:
        if col in df.columns:
            mem_col = col
            break

    if df.empty or mem_col is None:
        print("  Skipping memory plot - no data")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    epochs = df['epoch'].values
    memory = df[mem_col].values

    ax.plot(epochs, memory, color=COLORS['memory'], linewidth=2)
    ax.fill_between(epochs, 0, memory, alpha=0.2, color=COLORS['memory'])

    # Add peak annotation
    peak_idx = np.argmax(memory)
    ax.scatter([epochs[peak_idx]], [memory[peak_idx]], color=COLORS['memory'],
              s=100, zorder=5, marker='v')
    ax.annotate(f'Peak: {memory[peak_idx]:.1f} GB',
               (epochs[peak_idx], memory[peak_idx]),
               xytext=(10, 10), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('GPU Memory (GB)')
    ax.set_title('GPU Memory Usage During Training')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_overfitting_gap(df: pd.DataFrame, output_path: Path):
    """Plot the gap between train and val loss to detect overfitting."""
    if df.empty or 'train_loss' not in df.columns or 'val_loss' not in df.columns:
        print("  Skipping overfitting plot - no data")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    epochs = df['epoch'].values
    gap = df['val_loss'].values - df['train_loss'].values

    # Color by whether overfitting (gap > 0 and increasing)
    ax.plot(epochs, gap, color='#6B7280', linewidth=2)
    ax.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='#DC2626', label='Overfitting')
    ax.fill_between(epochs, 0, gap, where=(gap <= 0), alpha=0.3, color='#059669', label='Underfitting')

    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss - Train Loss')
    ax.set_title('Generalization Gap (Overfitting Detection)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_multi_fold_comparison(input_dir: Path, output_path: Path, n_folds: int = 5, seeds: list = None):
    """Plot loss curves across multiple folds for variance visualization."""
    seeds = seeds or [42, 123, 456]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect data from all folds/seeds
    all_ssl_losses = []
    all_trans_losses = []

    for fold in range(n_folds):
        for seed in seeds:
            metrics_path = input_dir / f"fold{fold}_seed{seed}" / "metrics" / "training_metrics.csv"
            if not metrics_path.exists():
                continue

            df = pd.read_csv(metrics_path)

            ssl_df = df[df['phase'] == 'ssl'] if 'phase' in df.columns else df
            trans_df = df[df['phase'] == 'transition'] if 'phase' in df.columns else pd.DataFrame()

            if not ssl_df.empty and 'val_loss' in ssl_df.columns:
                all_ssl_losses.append(ssl_df['val_loss'].values)

            if not trans_df.empty and 'val_loss' in trans_df.columns:
                all_trans_losses.append(trans_df['val_loss'].values)

    # Plot SSL phase
    ax = axes[0]
    if all_ssl_losses:
        # Pad to same length
        max_len = max(len(x) for x in all_ssl_losses)
        padded = [np.pad(x, (0, max_len - len(x)), constant_values=np.nan) for x in all_ssl_losses]
        stacked = np.array(padded)

        mean_loss = np.nanmean(stacked, axis=0)
        std_loss = np.nanstd(stacked, axis=0)
        epochs = np.arange(1, len(mean_loss) + 1)

        ax.plot(epochs, mean_loss, color=COLORS['ssl'], linewidth=2, label='Mean')
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                       alpha=0.3, color=COLORS['ssl'])

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title(f'SSL Phase ({len(all_ssl_losses)} runs)')
        ax.legend()
        ax.set_yscale('log')

    # Plot Transition phase
    ax = axes[1]
    if all_trans_losses:
        max_len = max(len(x) for x in all_trans_losses)
        padded = [np.pad(x, (0, max_len - len(x)), constant_values=np.nan) for x in all_trans_losses]
        stacked = np.array(padded)

        mean_loss = np.nanmean(stacked, axis=0)
        std_loss = np.nanstd(stacked, axis=0)
        epochs = np.arange(1, len(mean_loss) + 1)

        ax.plot(epochs, mean_loss, color=COLORS['transition'], linewidth=2, label='Mean')
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                       alpha=0.3, color=COLORS['transition'])

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title(f'Transition Phase ({len(all_trans_losses)} runs)')
        ax.legend()
        ax.set_yscale('log')

    fig.suptitle('Cross-Validation Loss Curves (Mean ± Std)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training curve visualizations")
    parser.add_argument("--input_dir", type=str, required=True, help="Training output directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Figure output directory")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold to plot (default: all)")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to plot (default: all)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating training curves from {input_dir}")

    # Plot individual run if specified
    if args.fold is not None and args.seed is not None:
        run_dir = input_dir / f"fold{args.fold}_seed{args.seed}"
        metrics_path = run_dir / "metrics" / "training_metrics.csv"

        print(f"\nPlotting fold {args.fold}, seed {args.seed}")
        df = load_metrics(metrics_path)

        if not df.empty:
            plot_loss_curves(df, output_dir / f"loss_curves_fold{args.fold}_seed{args.seed}.png",
                           title=f"Training Curves (Fold {args.fold}, Seed {args.seed})")
            plot_lr_schedule(df, output_dir / f"lr_schedule_fold{args.fold}_seed{args.seed}.png")
            plot_gpu_memory(df, output_dir / f"gpu_memory_fold{args.fold}_seed{args.seed}.png")
            plot_overfitting_gap(df, output_dir / f"overfitting_fold{args.fold}_seed{args.seed}.png")
        else:
            print(f"  No metrics found at {metrics_path}")
    else:
        # Plot representative run (fold 0, seed 42)
        print("\nPlotting representative run (fold 0, seed 42)")
        rep_metrics = input_dir / "fold0_seed42" / "metrics" / "training_metrics.csv"
        df = load_metrics(rep_metrics)

        if not df.empty:
            plot_loss_curves(df, output_dir / "loss_curves_representative.png",
                           title="Training Curves (Fold 0, Seed 42)")
            plot_lr_schedule(df, output_dir / "lr_schedule.png")
            plot_gpu_memory(df, output_dir / "gpu_memory.png")
            plot_overfitting_gap(df, output_dir / "overfitting_gap.png")

        # Plot multi-fold comparison
        print("\nPlotting multi-fold comparison")
        plot_multi_fold_comparison(input_dir, output_dir / "loss_curves_all_folds.png")

    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
