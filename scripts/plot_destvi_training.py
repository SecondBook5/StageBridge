#!/usr/bin/env python3
"""
Generate DestVI training loss curves for publication.

Usage:
    python scripts/plot_destvi_training.py --spatial-benchmark-dir $DATA/runs/spatial_benchmark/hlca/destvi

Outputs:
    - destvi_loss_curves.pdf (publication-ready)
    - destvi_loss_curves.png (for quick viewing)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
})


def load_training_history(sample_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load CondSCVI and DestVI training history from a sample directory."""
    condscvi_path = sample_dir / "condscvi_training_history.csv"
    destvi_path = sample_dir / "destvi_training_history.csv"

    condscvi_df = pd.read_csv(condscvi_path) if condscvi_path.exists() else None
    destvi_df = pd.read_csv(destvi_path) if destvi_path.exists() else None

    return condscvi_df, destvi_df


def plot_single_sample(condscvi_df: pd.DataFrame, destvi_df: pd.DataFrame,
                       sample_name: str, output_path: Path):
    """Plot training curves for a single sample."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # CondSCVI loss
    ax1 = axes[0]
    if condscvi_df is not None and len(condscvi_df) > 0:
        # Find ELBO columns (scvi-tools naming varies)
        elbo_cols = [c for c in condscvi_df.columns if 'elbo' in c.lower()]
        for col in elbo_cols:
            label = 'Validation' if 'val' in col.lower() else 'Training'
            linestyle = '--' if 'val' in col.lower() else '-'
            ax1.plot(condscvi_df[col], label=label, linestyle=linestyle)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('ELBO')
        ax1.set_title(f'CondSCVI Training ({sample_name})')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No CondSCVI history', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('CondSCVI Training')

    # DestVI loss
    ax2 = axes[1]
    if destvi_df is not None and len(destvi_df) > 0:
        elbo_cols = [c for c in destvi_df.columns if 'elbo' in c.lower() or 'loss' in c.lower()]
        for col in elbo_cols:
            label = 'Validation' if 'val' in col.lower() else 'Training'
            linestyle = '--' if 'val' in col.lower() else '-'
            ax2.plot(destvi_df[col], label=label, linestyle=linestyle)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ELBO / Loss')
        ax2.set_title(f'DestVI Training ({sample_name})')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No DestVI history', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('DestVI Training')

    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.with_suffix('.pdf')}")


def plot_aggregate(all_histories: list[dict], output_path: Path):
    """Plot aggregate training curves across all samples."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Aggregate CondSCVI
    ax1 = axes[0]
    condscvi_losses = []
    for h in all_histories:
        if h['condscvi'] is not None:
            # Find training ELBO column
            elbo_cols = [c for c in h['condscvi'].columns if 'elbo' in c.lower() and 'val' not in c.lower()]
            if elbo_cols:
                condscvi_losses.append(h['condscvi'][elbo_cols[0]].values)

    if condscvi_losses:
        # Pad to same length
        max_len = max(len(l) for l in condscvi_losses)
        padded = np.array([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in condscvi_losses])
        mean_loss = np.nanmean(padded, axis=0)
        std_loss = np.nanstd(padded, axis=0)
        epochs = np.arange(len(mean_loss))

        ax1.plot(epochs, mean_loss, 'b-', label='Mean')
        ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('ELBO')
        ax1.set_title(f'CondSCVI Training (n={len(condscvi_losses)} samples)')
        ax1.legend()

    # Aggregate DestVI
    ax2 = axes[1]
    destvi_losses = []
    for h in all_histories:
        if h['destvi'] is not None:
            elbo_cols = [c for c in h['destvi'].columns if 'elbo' in c.lower() and 'val' not in c.lower()]
            if not elbo_cols:
                elbo_cols = [c for c in h['destvi'].columns if 'loss' in c.lower()]
            if elbo_cols:
                destvi_losses.append(h['destvi'][elbo_cols[0]].values)

    if destvi_losses:
        max_len = max(len(l) for l in destvi_losses)
        padded = np.array([np.pad(l, (0, max_len - len(l)), constant_values=np.nan) for l in destvi_losses])
        mean_loss = np.nanmean(padded, axis=0)
        std_loss = np.nanstd(padded, axis=0)
        epochs = np.arange(len(mean_loss))

        ax2.plot(epochs, mean_loss, 'r-', label='Mean')
        ax2.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ELBO / Loss')
        ax2.set_title(f'DestVI Training (n={len(destvi_losses)} samples)')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description='Generate DestVI training loss curves')
    parser.add_argument('--spatial-benchmark-dir', type=str, required=True,
                        help='Path to spatial benchmark results (e.g., $DATA/runs/spatial_benchmark/hlca/destvi)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures (default: spatial-benchmark-dir/figures)')
    parser.add_argument('--sample', type=str, default=None,
                        help='Plot single sample only (e.g., "sample_001")')
    args = parser.parse_args()

    benchmark_dir = Path(args.spatial_benchmark_dir)
    output_dir = Path(args.output_dir) if args.output_dir else benchmark_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find sample directories
    samples_dir = benchmark_dir / "samples"
    if not samples_dir.exists():
        # Maybe the benchmark_dir IS the samples dir
        samples_dir = benchmark_dir

    sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    print(f"Found {len(sample_dirs)} sample directories")

    if args.sample:
        # Plot single sample
        sample_dir = samples_dir / args.sample
        if not sample_dir.exists():
            print(f"Sample directory not found: {sample_dir}")
            return
        condscvi_df, destvi_df = load_training_history(sample_dir)
        plot_single_sample(condscvi_df, destvi_df, args.sample, output_dir / f"destvi_loss_{args.sample}")
    else:
        # Plot all samples and aggregate
        all_histories = []
        for sample_dir in sample_dirs:
            sample_name = sample_dir.name
            condscvi_df, destvi_df = load_training_history(sample_dir)

            if condscvi_df is not None or destvi_df is not None:
                all_histories.append({
                    'name': sample_name,
                    'condscvi': condscvi_df,
                    'destvi': destvi_df,
                })
                # Plot individual
                plot_single_sample(condscvi_df, destvi_df, sample_name,
                                   output_dir / f"destvi_loss_{sample_name}")

        print(f"\nLoaded history from {len(all_histories)} samples")

        # Aggregate plot
        if all_histories:
            plot_aggregate(all_histories, output_dir / "destvi_loss_aggregate")
            print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
