"""
Split validation for StageBridge.

Ensures train/val/test splits are correctly constructed and reproducible.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def create_split_manifest(
    donor_ids: list[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    stratify_by: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Create a reproducible donor-level split manifest.

    Parameters
    ----------
    donor_ids : list
        All unique donor IDs
    train_frac, val_frac, test_frac : float
        Split fractions (must sum to 1.0)
    seed : int
        Random seed for reproducibility
    stratify_by : dict, optional
        Mapping of donor_id -> stratum for stratified splitting

    Returns
    -------
    dict
        Split manifest
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    donors = np.array(sorted(donor_ids))
    n = len(donors)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    manifest = {
        "train_donors": donors[train_idx].tolist(),
        "val_donors": donors[val_idx].tolist(),
        "test_donors": donors[test_idx].tolist(),
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_total_donors": n,
        "hash": _compute_manifest_hash(donors, seed),
    }

    return manifest


def _compute_manifest_hash(donors: np.ndarray, seed: int) -> str:
    """Compute deterministic hash for manifest verification."""
    content = f"{sorted(donors.tolist())}_{seed}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def verify_split_reproducibility(
    manifest: dict[str, Any],
    donor_ids: list[str],
) -> dict[str, Any]:
    """
    Verify that a manifest can be reproduced from the same inputs.

    Parameters
    ----------
    manifest : dict
        Existing manifest
    donor_ids : list
        Current donor IDs

    Returns
    -------
    dict
        Verification report
    """
    # Recreate manifest
    recreated = create_split_manifest(
        donor_ids,
        train_frac=manifest["train_frac"],
        val_frac=manifest["val_frac"],
        test_frac=manifest["test_frac"],
        seed=manifest["seed"],
    )

    # Compare
    matches = {
        "train_match": set(manifest["train_donors"]) == set(recreated["train_donors"]),
        "val_match": set(manifest["val_donors"]) == set(recreated["val_donors"]),
        "test_match": set(manifest["test_donors"]) == set(recreated["test_donors"]),
        "hash_match": manifest.get("hash") == recreated["hash"],
    }

    return {
        "reproducible": all(matches.values()),
        "matches": matches,
        "original_hash": manifest.get("hash"),
        "recreated_hash": recreated["hash"],
    }


def assign_cells_to_splits(
    cell_df: pd.DataFrame,
    manifest: dict[str, Any],
    donor_col: str = "donor_id",
) -> pd.DataFrame:
    """
    Assign cells to splits based on donor manifest.

    Parameters
    ----------
    cell_df : DataFrame
        Cell metadata with donor column
    manifest : dict
        Split manifest
    donor_col : str
        Name of donor column

    Returns
    -------
    DataFrame
        Cell metadata with 'split' column added
    """
    cell_df = cell_df.copy()

    train_set = set(manifest["train_donors"])
    val_set = set(manifest["val_donors"])
    test_set = set(manifest["test_donors"])

    def get_split(donor):
        if donor in train_set:
            return "train"
        elif donor in val_set:
            return "val"
        elif donor in test_set:
            return "test"
        else:
            return "unknown"

    cell_df["split"] = cell_df[donor_col].apply(get_split)

    # Log split sizes
    split_counts = cell_df["split"].value_counts()
    log.info(f"Split assignment: {split_counts.to_dict()}")

    if "unknown" in split_counts.index:
        log.warning(f"{split_counts['unknown']} cells have unknown split (donor not in manifest)")

    return cell_df


def validate_split_balance(
    cell_df: pd.DataFrame,
    split_col: str = "split",
    stage_col: str = "stage",
) -> dict[str, Any]:
    """
    Check that splits are reasonably balanced across stages.

    Parameters
    ----------
    cell_df : DataFrame
        Cell metadata with split and stage columns
    split_col : str
        Name of split column
    stage_col : str
        Name of stage column

    Returns
    -------
    dict
        Balance report
    """
    if stage_col not in cell_df.columns:
        return {"valid": True, "note": "No stage column for balance check"}

    # Compute stage proportions per split
    crosstab = pd.crosstab(cell_df[split_col], cell_df[stage_col], normalize="index")

    # Check for severe imbalance (>2x difference in stage proportion)
    issues = []
    for stage in crosstab.columns:
        props = crosstab[stage]
        if props.max() > 2 * props.min() and props.min() > 0:
            issues.append(
                {
                    "stage": stage,
                    "min_prop": props.min(),
                    "max_prop": props.max(),
                    "ratio": props.max() / props.min(),
                }
            )

    return {
        "valid": len(issues) == 0,
        "stage_proportions": crosstab.to_dict(),
        "imbalance_issues": issues,
    }


def save_split_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    """Save split manifest to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Saved split manifest to {output_path}")
    return output_path


def load_split_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load split manifest from JSON."""
    with open(manifest_path) as f:
        return json.load(f)
