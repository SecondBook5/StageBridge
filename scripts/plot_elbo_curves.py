#!/usr/bin/env python3
"""Plot ELBO training curves from reference mapping log."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Extracted from training log output
# HLCA: epochs 1-58, early stopped when validation didn't improve for 15 epochs
hlca_epochs = list(range(1, 59))
hlca_train_loss = [
    733, 694, 681, 675, 673, 671, 670, 670, 669, 669,  # 1-10
    669, 669, 669, 668, 668, 668, 668, 669, 669, 669,  # 11-20
    669, 669, 669, 669, 669, 669, 670, 670, 670, 670,  # 21-30
    670, 671, 671, 671, 671, 671, 672, 672, 672, 672,  # 31-40
    672, 673, 673, 673, 673, 674, 674, 674, 674, 675,  # 41-50
    675, 675, 675, 675, 676, 676, 676, 677,            # 51-58
]

# LuCA: epochs 1-64, early stopped
luca_epochs = list(range(1, 65))
luca_train_loss = [
    2010, 1950, 1940, 1930, 1930, 1930, 1930, 1920, 1920, 1920,  # 1-10
    1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920,  # 11-20
    1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920,  # 21-30
    1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920,  # 31-40
    1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920,  # 41-50
    1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920,  # 51-60
    1930, 1930, 1930, 1930,                                      # 61-64
]

# Best validation ELBO (from early stopping message)
hlca_best_val = 763.368
luca_best_val = 1968.988

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# HLCA
ax = axes[0]
ax.plot(hlca_epochs, hlca_train_loss, 'b-', linewidth=2, label='Train ELBO')
ax.axhline(y=hlca_best_val, color='r', linestyle='--', alpha=0.7, label=f'Best Val ELBO: {hlca_best_val:.1f}')
ax.axvline(x=58, color='gray', linestyle=':', alpha=0.5, label='Early stop (epoch 58)')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('ELBO', fontsize=11)
ax.set_title('HLCA scArches Surgery', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 60)

# LuCA
ax = axes[1]
ax.plot(luca_epochs, luca_train_loss, 'b-', linewidth=2, label='Train ELBO')
ax.axhline(y=luca_best_val, color='r', linestyle='--', alpha=0.7, label=f'Best Val ELBO: {luca_best_val:.1f}')
ax.axvline(x=64, color='gray', linestyle=':', alpha=0.5, label='Early stop (epoch 64)')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('ELBO', fontsize=11)
ax.set_title('LuCA scArches Surgery', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 70)

plt.tight_layout()

# Save
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "reference_elbo_curves.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

# Also save PDF for paper
plt.savefig(output_dir / "reference_elbo_curves.pdf", bbox_inches='tight')
print(f"Saved: {output_dir / 'reference_elbo_curves.pdf'}")

plt.show()
