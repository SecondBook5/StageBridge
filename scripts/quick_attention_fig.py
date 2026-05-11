#!/usr/bin/env python3
"""Quick attention and drift figures from inference output."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def make_drift_figure(inf_dir: Path, data_dir: Path, out_path: Path):
    """Create drift/velocity streamline figure."""
    disp_file = inf_dir / 'displacements.npy'
    pred_file = inf_dir / 'predictions.parquet'
    umap_file = data_dir / 'embeddings' / 'umap_embedding.parquet'
    cells_file = data_dir / 'cells.parquet'

    if not disp_file.exists():
        print(f"No displacements: {disp_file}")
        return

    disp = np.load(disp_file)
    print(f"Displacements: shape={disp.shape}, range=[{disp.min():.3f}, {disp.max():.3f}]")

    # Load predictions to get cell_ids
    pred_df = pd.read_parquet(pred_file)

    # Load UMAP
    if not umap_file.exists():
        print(f"No UMAP file: {umap_file}")
        return

    umap_df = pd.read_parquet(umap_file)
    if 'UMAP1' in umap_df.columns:
        umap_all = umap_df[['UMAP1', 'UMAP2']].values
    else:
        umap_all = umap_df.iloc[:, :2].values

    # Load cells for stage info
    cells_df = pd.read_parquet(cells_file)

    # Match cell_ids to get UMAP coords for predicted cells
    # This is approximate - assumes order matches snRNA subset
    n_pred = len(pred_df)
    n_umap = len(umap_all)

    if n_pred > n_umap:
        print(f"More predictions ({n_pred}) than UMAP points ({n_umap}), truncating")
        n = n_umap
    else:
        n = n_pred

    umap = umap_all[:n]
    disp = disp[:n]

    # Project displacements to 2D (use first 2 dims scaled by magnitude)
    disp_mag = np.linalg.norm(disp, axis=1, keepdims=True)
    disp_2d = disp[:, :2]
    disp_2d = disp_2d * disp_mag / (np.linalg.norm(disp_2d, axis=1, keepdims=True) + 1e-8)

    # Get stages
    stages = cells_df['stage'].values[:n] if 'stage' in cells_df.columns else None

    stage_colors = {
        'Normal': '#228B22',
        'Preinvasive': '#4169E1',
        'Invasive': '#CB4154',
        'AAH': '#4682B4',
        'AIS': '#4169E1',
        'MIA': '#8B008B',
        'LUAD': '#CB4154',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Quiver plot colored by stage
    ax = axes[0]
    # Subsample for clarity
    step = max(1, n // 2000)
    idx = np.arange(0, n, step)

    if stages is not None:
        for stage, color in stage_colors.items():
            mask = stages[idx] == stage
            if mask.sum() > 0:
                ax.quiver(umap[idx][mask, 0], umap[idx][mask, 1],
                         disp_2d[idx][mask, 0], disp_2d[idx][mask, 1],
                         color=color, alpha=0.6, scale=50, width=0.003,
                         label=stage)
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.quiver(umap[idx, 0], umap[idx, 1], disp_2d[idx, 0], disp_2d[idx, 1],
                 alpha=0.6, scale=50, width=0.003)

    ax.set_title('Predicted Drift by Stage')
    ax.set_xticks([]); ax.set_yticks([])

    # 2. Streamlines via griddata interpolation
    ax = axes[1]
    from scipy.interpolate import griddata

    # Create grid
    xi = np.linspace(umap[:, 0].min(), umap[:, 0].max(), 30)
    yi = np.linspace(umap[:, 1].min(), umap[:, 1].max(), 30)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate velocities
    U = griddata(umap, disp_2d[:, 0], (Xi, Yi), method='linear')
    V = griddata(umap, disp_2d[:, 1], (Xi, Yi), method='linear')

    # Fill NaN
    U = np.nan_to_num(U, nan=0)
    V = np.nan_to_num(V, nan=0)

    # Background scatter
    ax.scatter(umap[::step, 0], umap[::step, 1], c='lightgray', s=1, alpha=0.3, rasterized=True)

    # Streamlines
    speed = np.sqrt(U**2 + V**2)
    lw = 2 * speed / (speed.max() + 1e-8)
    ax.streamplot(xi, yi, U, V, color=speed, cmap='coolwarm', linewidth=lw, density=1.5, arrowsize=1)

    ax.set_title('Drift Streamlines')
    ax.set_xticks([]); ax.set_yticks([])

    # 3. Drift magnitude on UMAP
    ax = axes[2]
    mag = np.linalg.norm(disp_2d, axis=1)
    scatter = ax.scatter(umap[:, 0], umap[:, 1], c=mag, cmap='magma', s=1, alpha=0.5, rasterized=True)
    plt.colorbar(scatter, ax=ax, shrink=0.7, label='Drift Magnitude')
    ax.set_title('Drift Magnitude')
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved drift figure: {out_path}")


def main(inf_dir: Path, out_path: Path, data_dir: Path = None):
    attn_file = inf_dir / 'attention_weights.npz'
    if not attn_file.exists():
        print(f"No attention file: {attn_file}")
        return

    attn = np.load(attn_file)['attention']
    print(f"Attention: shape={attn.shape}, range=[{attn.min():.4f}, {attn.max():.4f}], mean={attn.mean():.4f}")

    if attn.max() < 0.001:
        print("WARNING: Attention near zero, skipping attention figure")
        # Still make drift figure
        if data_dir is not None:
            drift_path = out_path.parent / out_path.name.replace('attention_', 'drift_')
            make_drift_figure(inf_dir, data_dir, drift_path)
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Heatmap (sample of cells x neighbors)
    ax = axes[0]
    n_show = min(100, len(attn))
    im = ax.imshow(attn[:n_show], aspect='auto', cmap='viridis')
    ax.set_xlabel('Neighbor Position')
    ax.set_ylabel('Cell')
    ax.set_title(f'Attention Heatmap (first {n_show} cells)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. Distribution
    ax = axes[1]
    ax.hist(attn.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Attention Weight')
    ax.set_ylabel('Count')
    ax.set_title(f'Attention Distribution')
    ax.axvline(attn.mean(), color='red', linestyle='--', label=f'mean={attn.mean():.4f}')
    ax.legend()

    # 3. Mean attention per neighbor position (distance decay)
    ax = axes[2]
    mean_per_pos = attn.mean(axis=0)
    ax.plot(range(len(mean_per_pos)), mean_per_pos, 'o-', markersize=3)
    ax.set_xlabel('Neighbor Position (sorted by distance)')
    ax.set_ylabel('Mean Attention')
    ax.set_title('Attention vs Distance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Also make drift figure if data_dir provided
    if data_dir is not None:
        drift_path = out_path.parent / out_path.name.replace('attention_', 'drift_')
        make_drift_figure(inf_dir, data_dir, drift_path)


if __name__ == '__main__':
    inf_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    data_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    main(inf_dir, out_path, data_dir)
