#!/usr/bin/env python3
"""Validate cell type markers - Snakemake wrapper.

Core logic in stagebridge.validation.markers
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata

from stagebridge.validation.markers import validate_all_cell_types


def main():
    parser = argparse.ArgumentParser(description="Validate cell type markers")
    parser.add_argument("--adata", type=str, required=True, help="Input h5ad with cell types")
    parser.add_argument("--cell_type_col", type=str, default="cell_type")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--min_cells", type=int, default=10)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.adata}")
    adata = anndata.read_h5ad(args.adata)
    print(f"  Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    results = validate_all_cell_types(
        adata, cell_type_col=args.cell_type_col, min_cells=args.min_cells
    )

    enrichment_df = results.pop("enrichment_df", None)
    if enrichment_df is not None:
        enrichment_path = output_path.with_suffix(".enrichment.csv")
        enrichment_df.to_csv(enrichment_path, index=False)
        print(f"Enrichment table saved to: {enrichment_path}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Validation report saved to: {output_path}")

    n_pass = results["summary"]["n_pass"]
    n_fail = results["summary"]["n_fail"]
    print(f"\nSummary: {n_pass} pass, {n_fail} fail")

    if not results["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
