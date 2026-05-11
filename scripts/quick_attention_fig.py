#!/usr/bin/env python3
"""Quick attention figure from inference output."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main(inf_dir: Path, out_path: Path):
    attn_file = inf_dir / 'attention_weights.npz'
    if not attn_file.exists():
        print(f"No attention file: {attn_file}")
        return

    attn = np.load(attn_file)['attention']
    print(f"Attention: shape={attn.shape}, range=[{attn.min():.4f}, {attn.max():.4f}], mean={attn.mean():.4f}")

    if attn.max() < 0.001:
        print("WARNING: Attention near zero, skipping figure")
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

if __name__ == '__main__':
    inf_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    main(inf_dir, out_path)
