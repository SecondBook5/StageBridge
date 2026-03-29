#!/usr/bin/env python
"""
Add cycling cell states to HLCA and LuCA references.
"""

import scanpy as sc
from pathlib import Path
import sys

# Add StageBridge to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stagebridge.spatial_mapping.lung_markers import add_cycling_cell_states

# Paths from workflow/config.yaml
DATA_ROOT = Path("/scratch/chaunzt1/stagebridge")
HLCA_PATH = DATA_ROOT / "references/hlca/hlca_reference.h5ad"
LUCA_PATH = DATA_ROOT / "references/luca/luca_core_atlas.h5ad"


def process_reference(path: Path, name: str, output_suffix: str = "_with_cycling"):
    """Add cycling states to a reference."""
    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return None

    # Load
    print(f"Loading {path}...")
    adata = sc.read_h5ad(path)
    print(f"Loaded: {adata.shape}")

    # Add cycling states
    print("\nAdding cycling cell states...")
    add_cycling_cell_states(adata, cell_type_key='cell_type')

    # Show results
    print("\nCell state distribution:")
    print(adata.obs['cell_state'].value_counts().head(20))

    # Count cycling
    cycling_states = adata.obs['cell_state'].str.contains('cycling')
    n_cycling = cycling_states.sum()
    pct_cycling = 100 * n_cycling / len(adata)
    print(f"\nTotal cycling cells: {n_cycling} ({pct_cycling:.1f}%)")

    # Save
    output_path = path.parent / f"{path.stem}{output_suffix}.h5ad"
    print(f"\nSaving to: {output_path}")
    adata.write(output_path)

    print(f"✓ Done: {name}")
    return output_path


def main():
    """Process both references."""
    print("="*60)
    print("ADD CYCLING STATES TO REFERENCES")
    print("="*60)

    # Process HLCA
    hlca_out = process_reference(HLCA_PATH, "HLCA")

    # Process LuCA
    luca_out = process_reference(LUCA_PATH, "LuCA")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if hlca_out:
        print(f"✓ HLCA: {hlca_out}")
    if luca_out:
        print(f"✓ LuCA: {luca_out}")

    print("\nNext steps:")
    print("1. Update your pipeline config to use the new references:")
    print(f"   reference_path: {hlca_out}")
    print(f"   reference_path: {luca_out}")
    print("2. Restart the spatial benchmark with updated references")


if __name__ == "__main__":
    main()
