#!/usr/bin/env python3
"""Validate cell type predictions against known marker genes.

This is a critical QC step to ensure HLCA cell type predictions are biologically
meaningful before downstream analysis.

Validation approach:
1. For each predicted cell type, check expression of known markers
2. Compute marker enrichment (mean expression in predicted vs other cells)
3. Flag cell types with low marker concordance

Known markers for lung cell types (from literature):
- Epithelial: EPCAM, KRT18, KRT19
- Alveolar Type 2: SFTPC, SFTPB, LAMP3
- Alveolar Type 1: AGER, PDPN, HOPX
- Club cells: SCGB1A1, SCGB3A2
- Ciliated: FOXJ1, TPPP3
- Basal: KRT5, KRT17, TP63
- Immune (PTPRC+): PTPRC (CD45)
- T cells: CD3D, CD3E
- B cells: CD79A, MS4A1
- Macrophages: CD68, CD14, MARCO
- Fibroblasts: COL1A1, DCN
- Endothelial: PECAM1, VWF

Outputs:
- marker_validation.json: Full validation report
- marker_enrichment.csv: Enrichment scores per cell type
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import stats


# Known marker genes for lung cell types
# Format: {cell_type_pattern: [marker_genes]}
# Patterns match HLCA cell type labels (case-insensitive, partial match)
MARKER_GENES = {
    # Epithelial lineage
    "epithelial": ["EPCAM", "KRT18", "KRT19"],
    "alveolar type 2": ["SFTPC", "SFTPB", "SFTPA1", "LAMP3"],
    "at2": ["SFTPC", "SFTPB", "LAMP3"],
    "alveolar type 1": ["AGER", "PDPN", "HOPX"],
    "at1": ["AGER", "PDPN", "HOPX"],
    "club": ["SCGB1A1", "SCGB3A2"],
    "ciliated": ["FOXJ1", "TPPP3", "PIFO"],
    "basal": ["KRT5", "KRT17", "TP63"],

    # Immune lineage
    "immune": ["PTPRC"],
    "t cell": ["CD3D", "CD3E", "CD3G"],
    "cd4": ["CD3D", "CD4"],
    "cd8": ["CD3D", "CD8A", "CD8B"],
    "b cell": ["CD79A", "MS4A1", "CD19"],
    "plasma": ["JCHAIN", "MZB1", "SDC1"],
    "macrophage": ["CD68", "CD14", "MARCO"],
    "monocyte": ["CD14", "FCGR3A", "S100A8"],
    "dendritic": ["ITGAX", "CD1C", "CLEC9A"],
    "mast": ["TPSAB1", "CPA3", "KIT"],
    "nk": ["NCAM1", "NKG7", "GNLY"],

    # Stromal lineage
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "smooth muscle": ["ACTA2", "MYH11", "TAGLN"],
    "endothelial": ["PECAM1", "VWF", "CDH5", "ERG"],
    "pericyte": ["PDGFRB", "RGS5", "NOTCH3"],
}


def compute_marker_enrichment(
    adata: anndata.AnnData,
    cell_type_col: str,
    markers: list[str],
    min_cells: int = 10,
) -> pd.DataFrame:
    """Compute marker enrichment for each cell type.

    Enrichment = mean(marker in cell type) / mean(marker in other cells)
    Also computes statistical significance via Mann-Whitney U test.
    """
    # Get available markers
    available_markers = [m for m in markers if m in adata.var_names]

    if not available_markers:
        return pd.DataFrame()

    # Get expression matrix (densify if sparse)
    X = adata[:, available_markers].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    rows = []
    cell_types = adata.obs[cell_type_col].unique()

    for cell_type in cell_types:
        mask = adata.obs[cell_type_col] == cell_type
        n_cells = mask.sum()

        if n_cells < min_cells:
            continue

        X_in = X[mask]
        X_out = X[~mask]

        for i, marker in enumerate(available_markers):
            expr_in = X_in[:, i]
            expr_out = X_out[:, i]

            mean_in = float(np.mean(expr_in))
            mean_out = float(np.mean(expr_out))

            # Enrichment ratio (avoid division by zero)
            if mean_out > 0:
                enrichment = mean_in / mean_out
            else:
                enrichment = np.inf if mean_in > 0 else 1.0

            # Percent expressing (>0)
            pct_in = float(np.mean(expr_in > 0) * 100)
            pct_out = float(np.mean(expr_out > 0) * 100)

            # Statistical test
            if len(expr_in) >= 3 and len(expr_out) >= 3:
                try:
                    stat, pval = stats.mannwhitneyu(expr_in, expr_out, alternative="greater")
                except Exception:
                    pval = 1.0
            else:
                pval = 1.0

            rows.append({
                "cell_type": cell_type,
                "marker": marker,
                "n_cells": n_cells,
                "mean_in": mean_in,
                "mean_out": mean_out,
                "enrichment": enrichment,
                "pct_expressing_in": pct_in,
                "pct_expressing_out": pct_out,
                "pval": pval,
            })

    return pd.DataFrame(rows)


def validate_cell_type(cell_type: str, enrichment_df: pd.DataFrame) -> dict:
    """Validate a single cell type against expected markers."""

    cell_type_lower = cell_type.lower()

    # Find matching marker set
    matched_markers = []
    for pattern, markers in MARKER_GENES.items():
        if pattern in cell_type_lower:
            matched_markers.extend(markers)

    if not matched_markers:
        return {
            "cell_type": cell_type,
            "status": "no_markers_defined",
            "message": f"No marker genes defined for '{cell_type}'",
        }

    # Check enrichment for matched markers
    cell_df = enrichment_df[enrichment_df["cell_type"] == cell_type]

    if cell_df.empty:
        return {
            "cell_type": cell_type,
            "status": "no_enrichment_data",
            "message": "No enrichment data available",
        }

    # Check each expected marker
    marker_results = []
    for marker in set(matched_markers):
        marker_row = cell_df[cell_df["marker"] == marker]

        if marker_row.empty:
            marker_results.append({
                "marker": marker,
                "status": "not_found",
                "enrichment": None,
            })
        else:
            row = marker_row.iloc[0]
            enrichment = row["enrichment"]
            pval = row["pval"]

            # Consider enriched if ratio > 1.5 and p < 0.05
            is_enriched = enrichment > 1.5 and pval < 0.05

            marker_results.append({
                "marker": marker,
                "status": "enriched" if is_enriched else "not_enriched",
                "enrichment": enrichment,
                "pval": pval,
                "pct_expressing": row["pct_expressing_in"],
            })

    # Compute overall score
    n_expected = len([m for m in marker_results if m["status"] != "not_found"])
    n_enriched = len([m for m in marker_results if m["status"] == "enriched"])

    if n_expected == 0:
        concordance = 0.0
        status = "no_markers_available"
    else:
        concordance = n_enriched / n_expected
        if concordance >= 0.5:
            status = "pass"
        elif concordance > 0:
            status = "warn"
        else:
            status = "fail"

    return {
        "cell_type": cell_type,
        "status": status,
        "concordance": concordance,
        "n_markers_expected": n_expected,
        "n_markers_enriched": n_enriched,
        "marker_results": marker_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate cell type markers")
    parser.add_argument("--adata", type=str, required=True, help="Input h5ad with cell types")
    parser.add_argument("--cell_type_col", type=str, default="cell_type", help="Column with cell type labels")
    parser.add_argument("--output", type=str, required=True, help="Output validation report path")
    parser.add_argument("--min_cells", type=int, default=10, help="Minimum cells per type")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.adata}")
    adata = anndata.read_h5ad(args.adata)
    print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    if args.cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{args.cell_type_col}' not found. "
                        f"Available: {adata.obs.columns.tolist()}")

    cell_types = adata.obs[args.cell_type_col].unique()
    print(f"  Cell types: {len(cell_types)}")

    # Get all markers to test
    all_markers = list(set(m for markers in MARKER_GENES.values() for m in markers))
    available_markers = [m for m in all_markers if m in adata.var_names]
    print(f"\nMarkers: {len(available_markers)}/{len(all_markers)} available")

    # Compute enrichment
    print("\nComputing marker enrichment...")
    enrichment_df = compute_marker_enrichment(
        adata, args.cell_type_col, available_markers, min_cells=args.min_cells
    )

    # Validate each cell type
    print("\nValidating cell types...")
    validations = []
    for cell_type in cell_types:
        result = validate_cell_type(cell_type, enrichment_df)
        validations.append(result)

        status_emoji = {
            "pass": "[PASS]",
            "warn": "[WARN]",
            "fail": "[FAIL]",
            "no_markers_defined": "[SKIP]",
            "no_markers_available": "[SKIP]",
            "no_enrichment_data": "[SKIP]",
        }
        print(f"  {status_emoji.get(result['status'], '[?]')} {cell_type}: "
              f"{result.get('concordance', 0):.0%} marker concordance")

    # Summary
    n_pass = len([v for v in validations if v["status"] == "pass"])
    n_warn = len([v for v in validations if v["status"] == "warn"])
    n_fail = len([v for v in validations if v["status"] == "fail"])
    n_skip = len([v for v in validations if v["status"] not in ["pass", "warn", "fail"]])

    overall_valid = n_fail == 0

    # Save enrichment table
    enrichment_path = output_path.with_suffix(".enrichment.csv")
    enrichment_df.to_csv(enrichment_path, index=False)
    print(f"\nEnrichment table saved to: {enrichment_path}")

    # Build report
    report = {
        "valid": overall_valid,
        "summary": {
            "n_cell_types": len(cell_types),
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
            "n_skip": n_skip,
        },
        "validations": validations,
        "markers_available": available_markers,
        "markers_missing": [m for m in all_markers if m not in adata.var_names],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Validation report saved to: {output_path}")

    # Final status
    print("\n" + "=" * 60)
    if overall_valid:
        print("[PASS] Marker validation passed")
    else:
        print(f"[FAIL] Marker validation failed: {n_fail} cell types with low concordance")
        for v in validations:
            if v["status"] == "fail":
                print(f"  - {v['cell_type']}")
    print("=" * 60)

    if not overall_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
