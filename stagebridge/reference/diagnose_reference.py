"""Diagnose and clean reference latent integrity issues.

Quick, memory-efficient checks for NaN/corruption in reference latents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def diagnose_latent_integrity(
    ref_path: Path,
    latent_key: str = "X_scVI",
    chunk_size: int = 50000,
) -> dict[str, Any]:
    """Diagnose latent integrity without loading full file.

    Uses h5py for fast, memory-efficient access.

    Returns
    -------
    dict with:
        - n_cells: total cells
        - latent_dim: latent dimensions
        - n_cells_any_nan: cells with at least one NaN
        - n_cells_all_nan: cells with all NaN (completely invalid)
        - n_cells_valid: cells with no NaN (usable)
        - nan_fraction: fraction of all values that are NaN
        - valid_fraction: fraction of cells that are valid
        - recommendation: "usable", "needs_cleaning", or "unusable"
    """
    import h5py

    report = {
        "reference_path": str(ref_path),
        "latent_key": latent_key,
    }

    with h5py.File(ref_path, 'r') as f:
        # Check if latent exists
        if 'obsm' not in f or latent_key not in f['obsm']:
            report["status"] = "error"
            report["error"] = f"Latent key '{latent_key}' not found in obsm"
            if 'obsm' in f:
                report["available_keys"] = list(f['obsm'].keys())
            return report

        latent = f['obsm'][latent_key]
        n_cells, latent_dim = latent.shape

        report["n_cells"] = int(n_cells)
        report["latent_dim"] = int(latent_dim)

        # Process in chunks for memory efficiency
        n_cells_any_nan = 0
        n_cells_all_nan = 0
        total_nan = 0

        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = latent[start:end, :]

            nan_per_row = np.isnan(chunk).sum(axis=1)
            n_cells_any_nan += int((nan_per_row > 0).sum())
            n_cells_all_nan += int((nan_per_row == latent_dim).sum())
            total_nan += int(np.isnan(chunk).sum())

        n_cells_valid = n_cells - n_cells_any_nan

        report["n_cells_any_nan"] = n_cells_any_nan
        report["n_cells_all_nan"] = n_cells_all_nan
        report["n_cells_partial_nan"] = n_cells_any_nan - n_cells_all_nan
        report["n_cells_valid"] = n_cells_valid
        report["total_nan_values"] = total_nan
        report["total_values"] = n_cells * latent_dim
        report["nan_fraction"] = float(total_nan / (n_cells * latent_dim))
        report["valid_cell_fraction"] = float(n_cells_valid / n_cells)

        # Recommendation
        if n_cells_valid == n_cells:
            report["recommendation"] = "usable"
            report["recommendation_detail"] = "All cells have valid latents"
        elif n_cells_valid / n_cells >= 0.9:
            report["recommendation"] = "needs_cleaning"
            report["recommendation_detail"] = f"Filter {n_cells_any_nan} invalid cells ({100*(1-n_cells_valid/n_cells):.1f}%)"
        elif n_cells_valid / n_cells >= 0.5:
            report["recommendation"] = "needs_cleaning_major"
            report["recommendation_detail"] = f"Filter {n_cells_any_nan} invalid cells ({100*(1-n_cells_valid/n_cells):.1f}%) - significant data loss"
        else:
            report["recommendation"] = "unusable"
            report["recommendation_detail"] = f"Only {100*n_cells_valid/n_cells:.1f}% valid cells - reference needs regeneration"

        report["status"] = "complete"

    return report


def clean_reference_latent(
    ref_path: Path,
    output_path: Path,
    latent_key: str = "X_scVI",
    min_valid_fraction: float = 0.5,
) -> dict[str, Any]:
    """Create cleaned reference with invalid latent rows removed.

    Parameters
    ----------
    ref_path : Path
        Path to original reference h5ad
    output_path : Path
        Path for cleaned reference h5ad
    latent_key : str
        Key in obsm containing latent embeddings
    min_valid_fraction : float
        Minimum fraction of valid cells required (else abort)

    Returns
    -------
    dict with cleaning report
    """
    import anndata

    report = {
        "source_path": str(ref_path),
        "output_path": str(output_path),
        "latent_key": latent_key,
    }

    # First diagnose
    diag = diagnose_latent_integrity(ref_path, latent_key)
    report["diagnosis"] = diag

    if diag.get("status") == "error":
        report["status"] = "error"
        report["error"] = diag.get("error")
        return report

    valid_fraction = diag["valid_cell_fraction"]

    if valid_fraction < min_valid_fraction:
        report["status"] = "aborted"
        report["error"] = f"Only {valid_fraction:.1%} valid cells, below threshold {min_valid_fraction:.1%}"
        return report

    if valid_fraction == 1.0:
        report["status"] = "no_cleaning_needed"
        report["n_cells_removed"] = 0
        return report

    # Load and filter
    print(f"Loading reference from {ref_path}...")
    adata = anndata.read_h5ad(ref_path)

    n_before = adata.n_obs

    # Identify valid cells (no NaN in latent)
    latent = np.asarray(adata.obsm[latent_key])
    valid_mask = ~np.isnan(latent).any(axis=1)

    # Filter
    adata_clean = adata[valid_mask].copy()
    n_after = adata_clean.n_obs
    n_removed = n_before - n_after

    print(f"Removed {n_removed:,} cells with invalid latent ({100*n_removed/n_before:.1f}%)")
    print(f"Cleaned reference: {n_after:,} cells")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving cleaned reference to {output_path}...")
    adata_clean.write_h5ad(output_path)

    report["status"] = "complete"
    report["n_cells_before"] = n_before
    report["n_cells_after"] = n_after
    report["n_cells_removed"] = n_removed
    report["removal_fraction"] = float(n_removed / n_before)

    return report


def main():
    """CLI for reference diagnosis and cleaning."""
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose and clean reference latent integrity")
    parser.add_argument("reference", type=Path, help="Path to reference h5ad")
    parser.add_argument("--latent-key", default="X_scVI", help="Key in obsm for latent")
    parser.add_argument("--diagnose-only", action="store_true", help="Only diagnose, don't clean")
    parser.add_argument("--output", type=Path, help="Output path for cleaned reference")
    parser.add_argument("--report", type=Path, help="Path to save JSON report")

    args = parser.parse_args()

    # Diagnose
    print(f"Diagnosing {args.reference}...")
    report = diagnose_latent_integrity(args.reference, args.latent_key)

    print("\n=== Latent Integrity Report ===")
    print(f"  Total cells: {report.get('n_cells', 'N/A'):,}")
    print(f"  Latent dim: {report.get('latent_dim', 'N/A')}")
    print(f"  Valid cells: {report.get('n_cells_valid', 'N/A'):,} ({report.get('valid_cell_fraction', 0):.1%})")
    print(f"  Cells with any NaN: {report.get('n_cells_any_nan', 'N/A'):,}")
    print(f"  Cells with all NaN: {report.get('n_cells_all_nan', 'N/A'):,}")
    print(f"  Recommendation: {report.get('recommendation', 'N/A')}")
    print(f"  Detail: {report.get('recommendation_detail', 'N/A')}")

    # Clean if requested
    if not args.diagnose_only and report.get("recommendation") in ("needs_cleaning", "needs_cleaning_major"):
        if args.output is None:
            # Default output path
            args.output = args.reference.parent / f"{args.reference.stem}_cleaned.h5ad"

        print("\nCleaning reference...")
        clean_report = clean_reference_latent(args.reference, args.output, args.latent_key)
        report["cleaning"] = clean_report

        if clean_report.get("status") == "complete":
            print(f"\nCleaned reference saved to: {args.output}")

    # Save report
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.report}")

    return report


if __name__ == "__main__":
    main()
