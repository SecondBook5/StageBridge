#!/usr/bin/env python
"""
Check if HLCA and LuCA references have cycling states.
If not, need to add them before running deconvolution.
"""

import scanpy as sc
import sys
from pathlib import Path

# Paths from workflow/config.yaml
DATA_ROOT = Path("/scratch/chaunzt1/stagebridge")
HLCA_PATH = DATA_ROOT / "references/hlca/hlca_reference.h5ad"
LUCA_PATH = DATA_ROOT / "references/luca/luca_core_atlas.h5ad"


def check_reference(path: Path, name: str) -> bool:
    """Check if reference has cycling states."""
    print(f"\n{'='*60}")
    print(f"Checking {name}: {path}")
    print(f"{'='*60}")

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return False

    adata = sc.read_h5ad(path)
    print(f"Shape: {adata.shape}")

    # Check cell type column
    if 'cell_type' not in adata.obs.columns:
        print("ERROR: No 'cell_type' column in .obs")
        return False

    cell_types = adata.obs['cell_type'].unique()
    print(f"\nTotal cell types: {len(cell_types)}")
    print(f"\nFirst 20 cell types:")
    for ct in sorted(cell_types)[:20]:
        print(f"  - {ct}")

    # Check for cycling
    has_cycling = any('cycl' in str(ct).lower() for ct in cell_types)

    if has_cycling:
        print(f"\n✓ HAS CYCLING STATES")
        cycling_types = [ct for ct in cell_types if 'cycl' in str(ct).lower()]
        print(f"Cycling cell types ({len(cycling_types)}):")
        for ct in cycling_types[:10]:
            n_cells = (adata.obs['cell_type'] == ct).sum()
            print(f"  - {ct}: {n_cells} cells")
    else:
        print(f"\n✗ NO CYCLING STATES")
        print(f"\nNeed to add cycling states with:")
        print(f"  from stagebridge.spatial_mapping.lung_markers import add_cycling_cell_states")
        print(f"  add_cycling_cell_states(adata)")

    return has_cycling


def main():
    """Check both references."""
    print("="*60)
    print("REFERENCE CYCLING CHECK")
    print("="*60)

    hlca_has_cycling = check_reference(HLCA_PATH, "HLCA")
    luca_has_cycling = check_reference(LUCA_PATH, "LuCA")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"HLCA has cycling: {'✓' if hlca_has_cycling else '✗'}")
    print(f"LuCA has cycling: {'✓' if luca_has_cycling else '✗'}")

    if not (hlca_has_cycling and luca_has_cycling):
        print(f"\n{'='*60}")
        print("DECISION: STOP CURRENT RUN")
        print(f"{'='*60}")
        print("Your references don't have cycling states.")
        print("Current benchmark will miss cycling cell populations.")
        print("\nSteps:")
        print("1. Stop the current Snakemake run (Ctrl+C)")
        print("2. Add cycling states to references:")
        print("   python scripts/add_cycling_to_references.py")
        print("3. Restart benchmark with updated references")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("DECISION: CONTINUE CURRENT RUN")
        print(f"{'='*60}")
        print("References have cycling states. Current run is good.")
        print("You can apply abundance stratification to results afterward.")
        sys.exit(0)


if __name__ == "__main__":
    main()
