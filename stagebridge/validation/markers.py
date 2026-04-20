"""Marker gene validation for cell type predictions.

Validates HLCA/LuCA cell type predictions against known marker genes.
Critical QC step to ensure biological validity before downstream analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    import anndata
except ImportError:
    anndata = None


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
    adata: Any,
    cell_type_col: str,
    markers: list[str],
    min_cells: int = 10,
) -> pd.DataFrame:
    """Compute marker enrichment for each cell type.

    Enrichment = mean(marker in cell type) / mean(marker in other cells)
    Also computes statistical significance via Mann-Whitney U test.

    Args:
        adata: AnnData object with expression data
        cell_type_col: Column name for cell type labels
        markers: List of marker genes to test
        min_cells: Minimum cells per type

    Returns:
        DataFrame with enrichment statistics
    """
    available_markers = [m for m in markers if m in adata.var_names]

    if not available_markers:
        return pd.DataFrame()

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

            if mean_out > 0:
                enrichment = mean_in / mean_out
            else:
                enrichment = np.inf if mean_in > 0 else 1.0

            pct_in = float(np.mean(expr_in > 0) * 100)
            pct_out = float(np.mean(expr_out > 0) * 100)

            if len(expr_in) >= 3 and len(expr_out) >= 3:
                try:
                    _, pval = stats.mannwhitneyu(expr_in, expr_out, alternative="greater")
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


def validate_cell_type(
    cell_type: str,
    enrichment_df: pd.DataFrame,
    enrichment_threshold: float = 1.5,
    pval_threshold: float = 0.05,
) -> dict[str, Any]:
    """Validate a single cell type against expected markers.

    Args:
        cell_type: Cell type name to validate
        enrichment_df: DataFrame from compute_marker_enrichment
        enrichment_threshold: Minimum enrichment ratio for "enriched" status
        pval_threshold: Maximum p-value for significance

    Returns:
        Dictionary with validation results
    """
    cell_type_lower = cell_type.lower()

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

    cell_df = enrichment_df[enrichment_df["cell_type"] == cell_type]

    if cell_df.empty:
        return {
            "cell_type": cell_type,
            "status": "no_enrichment_data",
            "message": "No enrichment data available",
        }

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

            is_enriched = enrichment > enrichment_threshold and pval < pval_threshold

            marker_results.append({
                "marker": marker,
                "status": "enriched" if is_enriched else "not_enriched",
                "enrichment": enrichment,
                "pval": pval,
                "pct_expressing": row["pct_expressing_in"],
            })

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


def validate_all_cell_types(
    adata: Any,
    cell_type_col: str = "cell_type",
    min_cells: int = 10,
) -> dict[str, Any]:
    """Validate all cell types in an AnnData object.

    Args:
        adata: AnnData object with expression data
        cell_type_col: Column name for cell type labels
        min_cells: Minimum cells per type

    Returns:
        Dictionary with full validation report
    """
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_col}' not found")

    all_markers = list(set(m for markers in MARKER_GENES.values() for m in markers))
    available_markers = [m for m in all_markers if m in adata.var_names]

    enrichment_df = compute_marker_enrichment(
        adata, cell_type_col, available_markers, min_cells=min_cells
    )

    cell_types = adata.obs[cell_type_col].unique()
    validations = []
    for cell_type in cell_types:
        result = validate_cell_type(cell_type, enrichment_df)
        validations.append(result)

    n_pass = len([v for v in validations if v["status"] == "pass"])
    n_warn = len([v for v in validations if v["status"] == "warn"])
    n_fail = len([v for v in validations if v["status"] == "fail"])
    n_skip = len([v for v in validations if v["status"] not in ["pass", "warn", "fail"]])

    return {
        "valid": n_fail == 0,
        "summary": {
            "n_cell_types": len(cell_types),
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
            "n_skip": n_skip,
        },
        "validations": validations,
        "enrichment_df": enrichment_df,
        "markers_available": available_markers,
        "markers_missing": [m for m in all_markers if m not in adata.var_names],
    }
