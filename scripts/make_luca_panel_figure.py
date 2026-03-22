#!/usr/bin/env python
"""Create panel figure from LuCA retraining outputs."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def main():
    results_dir = Path("results/luca_retrain")
    output_dir = Path("docs/paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    training = mpimg.imread(results_dir / "training_curves.png")
    scib = mpimg.imread(results_dir / "scib_comparison.png")
    umap = mpimg.imread(results_dir / "umap_comparison.png")

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(12, 10), dpi=300)

    # Layout: 2 columns on top, 1 spanning bottom
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.25, wspace=0.15)

    # (A) Training curves - top left
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(training)
    ax_a.axis('off')
    ax_a.set_title('A', fontsize=14, fontweight='bold', loc='left', x=-0.05, y=1.02)

    # (B) scIB comparison - top right
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(scib)
    ax_b.axis('off')
    ax_b.set_title('B', fontsize=14, fontweight='bold', loc='left', x=-0.05, y=1.02)

    # (C) UMAP comparison - bottom spanning both columns
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.imshow(umap)
    ax_c.axis('off')
    ax_c.set_title('C', fontsize=14, fontweight='bold', loc='left', x=-0.02, y=1.02)

    # Add figure caption/title
    fig.suptitle('LuCA scANVI Model Retraining Validation', fontsize=14, fontweight='bold', y=0.98)

    # Save
    output_path = output_dir / "luca_retraining_panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    # Also save PDF for publication
    pdf_path = output_dir / "luca_retraining_panel.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")

    plt.close()

    print("\nPanel figure created with:")
    print("  (A) Training curves: scVI (329 epochs) + scANVI (34 epochs)")
    print("  (B) scIB benchmark: Retrained vs Original model comparison")
    print("  (C) UMAP embeddings: Original (0.669) vs Retrained (0.673)")


if __name__ == "__main__":
    main()
