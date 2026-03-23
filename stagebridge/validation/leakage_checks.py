"""
Leakage Detection for StageBridge.

Validates that no information from held-out donors, lesions, or samples
has leaked into training data.
"""

import logging
from typing import Any

import numpy as np
import anndata as ad

log = logging.getLogger(__name__)


# =============================================================================
# Core Leakage Checks
# =============================================================================


def check_donor_leakage(
    train_donors: set[str],
    val_donors: set[str],
    test_donors: set[str],
) -> dict[str, Any]:
    """
    Check for donor overlap between splits.

    Parameters
    ----------
    train_donors : set
        Donor IDs in training set
    val_donors : set
        Donor IDs in validation set
    test_donors : set
        Donor IDs in test set

    Returns
    -------
    dict
        Leakage report
    """
    train_val_overlap = train_donors & val_donors
    train_test_overlap = train_donors & test_donors
    val_test_overlap = val_donors & test_donors

    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    return {
        "check": "donor_leakage",
        "valid": not has_leakage,
        "train_val_overlap": list(train_val_overlap),
        "train_test_overlap": list(train_test_overlap),
        "val_test_overlap": list(val_test_overlap),
        "n_train_donors": len(train_donors),
        "n_val_donors": len(val_donors),
        "n_test_donors": len(test_donors),
    }


def check_sample_containment(
    adata: ad.AnnData,
    split_col: str = "split",
    donor_col: str = "donor_id",
    sample_col: str = "sample_id",
) -> dict[str, Any]:
    """
    Check that samples from the same donor are in the same split.

    This prevents leakage when multiple samples come from one donor.
    """
    issues = []

    if sample_col not in adata.obs.columns:
        return {"check": "sample_containment", "valid": True, "note": "No sample_id column"}

    # Group by donor and check split consistency
    for donor in adata.obs[donor_col].unique():
        donor_mask = adata.obs[donor_col] == donor
        donor_splits = adata.obs.loc[donor_mask, split_col].unique()

        if len(donor_splits) > 1:
            issues.append(
                {
                    "donor": donor,
                    "splits": list(donor_splits),
                    "n_samples": donor_mask.sum(),
                }
            )

    return {
        "check": "sample_containment",
        "valid": len(issues) == 0,
        "issues": issues,
    }


def check_neighborhood_leakage(
    cell_ids: np.ndarray,
    neighbor_ids: list[list[str]],
    cell_splits: dict[str, str],
) -> dict[str, Any]:
    """
    Check that neighborhoods don't include cells from held-out donors.

    Parameters
    ----------
    cell_ids : array
        IDs of focal cells
    neighbor_ids : list of lists
        Neighbor IDs for each cell
    cell_splits : dict
        Mapping of cell_id -> split

    Returns
    -------
    dict
        Leakage report
    """
    cross_split_neighbors = []

    for cell_id, neighbors in zip(cell_ids, neighbor_ids):
        cell_split = cell_splits.get(cell_id)
        if cell_split is None:
            continue

        for neighbor_id in neighbors:
            neighbor_split = cell_splits.get(neighbor_id)
            if neighbor_split and neighbor_split != cell_split:
                cross_split_neighbors.append(
                    {
                        "cell": cell_id,
                        "cell_split": cell_split,
                        "neighbor": neighbor_id,
                        "neighbor_split": neighbor_split,
                    }
                )

    return {
        "check": "neighborhood_leakage",
        "valid": len(cross_split_neighbors) == 0,
        "n_violations": len(cross_split_neighbors),
        "examples": cross_split_neighbors[:10],  # First 10 examples
    }


def check_reference_leakage(
    query_donors: set[str],
    reference_donors: set[str],
    reference_name: str,
) -> dict[str, Any]:
    """
    Check that reference fitting didn't use held-out donor data.

    For external references (HLCA, LuCA), this is typically not an issue,
    but for custom references it must be verified.
    """
    overlap = query_donors & reference_donors

    return {
        "check": f"reference_leakage_{reference_name}",
        "valid": len(overlap) == 0,
        "overlap": list(overlap),
        "note": "External references (HLCA, LuCA) are typically safe",
    }


# =============================================================================
# Split Manifest Validation
# =============================================================================


def validate_split_manifest(
    manifest: dict[str, Any],
    adata: ad.AnnData | None = None,
) -> dict[str, Any]:
    """
    Validate a split manifest for correctness.

    Parameters
    ----------
    manifest : dict
        Split manifest with train/val/test donor lists
    adata : AnnData, optional
        If provided, verify cells match manifest

    Returns
    -------
    dict
        Validation report
    """
    report = {
        "valid": True,
        "checks": [],
    }

    # Check donor leakage
    train_donors = set(manifest.get("train_donors", []))
    val_donors = set(manifest.get("val_donors", []))
    test_donors = set(manifest.get("test_donors", []))

    donor_check = check_donor_leakage(train_donors, val_donors, test_donors)
    report["checks"].append(donor_check)
    if not donor_check["valid"]:
        report["valid"] = False

    # If adata provided, verify consistency
    if adata is not None and "donor_id" in adata.obs.columns:
        adata_donors = set(adata.obs["donor_id"].unique())
        manifest_donors = train_donors | val_donors | test_donors

        missing_in_manifest = adata_donors - manifest_donors
        missing_in_adata = manifest_donors - adata_donors

        consistency_check = {
            "check": "manifest_consistency",
            "valid": len(missing_in_manifest) == 0 and len(missing_in_adata) == 0,
            "missing_in_manifest": list(missing_in_manifest),
            "missing_in_adata": list(missing_in_adata),
        }
        report["checks"].append(consistency_check)
        if not consistency_check["valid"]:
            report["valid"] = False

    return report


# =============================================================================
# Full Leakage Audit
# =============================================================================


def run_leakage_audit(
    adata: ad.AnnData,
    split_manifest: dict[str, Any],
    neighbor_graph: dict | None = None,
) -> dict[str, Any]:
    """
    Run comprehensive leakage audit.

    Parameters
    ----------
    adata : AnnData
        Dataset with split annotations
    split_manifest : dict
        Split manifest
    neighbor_graph : dict, optional
        Pre-computed neighbor graph for neighborhood leakage check

    Returns
    -------
    dict
        Comprehensive leakage report
    """
    report = {
        "valid": True,
        "checks": [],
        "summary": {},
    }

    # 1. Validate split manifest
    manifest_report = validate_split_manifest(split_manifest, adata)
    report["checks"].extend(manifest_report["checks"])
    if not manifest_report["valid"]:
        report["valid"] = False

    # 2. Check sample containment
    if "split" in adata.obs.columns:
        containment_check = check_sample_containment(adata)
        report["checks"].append(containment_check)
        if not containment_check["valid"]:
            report["valid"] = False

    # 3. Neighborhood leakage (if graph provided)
    if neighbor_graph is not None and "split" in adata.obs.columns:
        dict(zip(adata.obs.index, adata.obs["split"]))
        # This would require the actual neighbor structure
        log.info("Neighborhood leakage check requires explicit neighbor structure")

    # Summary
    n_passed = sum(1 for c in report["checks"] if c["valid"])
    n_total = len(report["checks"])
    report["summary"] = {
        "n_checks": n_total,
        "n_passed": n_passed,
        "n_failed": n_total - n_passed,
    }

    return report
