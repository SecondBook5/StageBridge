"""Canonical data contract for StageBridge v1.

This module defines THE canonical contract for StageBridge data artifacts.
All other code MUST conform to this contract or fail hard.

Canonical artifacts:
- cells.parquet: Cell-level features with fused embeddings
- neighborhoods.parquet: 9-token niche structure per cell
- split_manifest.json: Donor-held-out CV splits

DO NOT create alternative contracts. DO NOT silently degrade.
If data doesn't match, FAIL LOUDLY.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# =============================================================================
# CANONICAL CONSTANTS - THE SINGLE SOURCE OF TRUTH
# =============================================================================

# Stage names - EXACTLY these, in this order
CANONICAL_STAGES = ("Normal", "AAH", "AIS", "MIA", "LUAD")
STAGE_TO_INDEX = {s: i for i, s in enumerate(CANONICAL_STAGES)}
INDEX_TO_STAGE = {i: s for i, s in enumerate(CANONICAL_STAGES)}

# Latent dimensions
CANONICAL_LATENT_DIM = 40  # Fused = HLCA + LuCA
CANONICAL_HLCA_DIM = 30
CANONICAL_LUCA_DIM = 10

# WES features - EXACTLY these, in this order
CANONICAL_WES_COLS = (
    "tmb",
    "kras_mut",
    "egfr_mut",
    "tp53_mut",
    "stk11_mut",
    "keap1_mut",
    "nfe2l2_mut",
    "rb1_mut",
)
CANONICAL_WES_DIM = len(CANONICAL_WES_COLS)

# Niche token structure - 9 tokens
CANONICAL_N_TOKENS = 9
TOKEN_TYPES = (
    "receiver",  # 0: Central cell
    "ring1",     # 1: Innermost ring
    "ring2",     # 2
    "ring3",     # 3
    "ring4",     # 4: Outermost ring
    "hlca",      # 5: HLCA reference embedding
    "luca",      # 6: LuCA reference embedding
    "pathway",   # 7: Pathway activity (EMT, CAF, immune)
    "stats",     # 8: Summary statistics
)

# Pathway features in token 7
PATHWAY_FEATURES = ("emt_score", "caf_fraction", "immune_fraction", "il1b_score")

# Data types - cells.parquet must distinguish these
DATA_TYPES = ("snrna", "spatial")

# Required columns in cells.parquet
REQUIRED_CELL_COLS = (
    "cell_id",
    "donor_id",
    "stage",
    "data_type",  # MUST distinguish snrna vs spatial
)

# Required columns in neighborhoods.parquet
REQUIRED_NEIGHBORHOOD_COLS = (
    "cell_id",
    "donor_id",
    "tokens",  # List of token dicts
)


@dataclass
class ContractViolation:
    """A specific contract violation."""
    severity: str  # "error" or "warning"
    category: str
    message: str
    details: dict | None = None


class CanonicalContractValidator:
    """Validates data artifacts against the canonical contract.

    Usage:
        validator = CanonicalContractValidator(data_dir)
        validator.validate_all()  # Raises if ANY errors

        # Or check specific artifacts
        validator.validate_cells()
        validator.validate_neighborhoods()
        validator.validate_splits()
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.violations: list[ContractViolation] = []

    def _add_error(self, category: str, message: str, details: dict | None = None):
        self.violations.append(ContractViolation("error", category, message, details))

    def _add_warning(self, category: str, message: str, details: dict | None = None):
        self.violations.append(ContractViolation("warning", category, message, details))

    def validate_all(self, raise_on_error: bool = True) -> list[ContractViolation]:
        """Validate all canonical artifacts.

        Args:
            raise_on_error: If True, raises ValueError on any error-level violation.

        Returns:
            List of all violations found.
        """
        self.violations = []

        # Check file existence first
        cells_path = self.data_dir / "cells.parquet"
        neighborhoods_path = self.data_dir / "neighborhoods.parquet"
        splits_path = self.data_dir / "split_manifest.json"

        if not cells_path.exists():
            self._add_error("files", f"cells.parquet not found at {cells_path}")
        if not neighborhoods_path.exists():
            self._add_error("files", f"neighborhoods.parquet not found at {neighborhoods_path}")
        if not splits_path.exists():
            self._add_error("files", f"split_manifest.json not found at {splits_path}")

        # If core files missing, can't continue
        errors = [v for v in self.violations if v.severity == "error"]
        if errors and raise_on_error:
            raise ValueError(f"Contract violations: {[e.message for e in errors]}")

        # Validate each artifact
        if cells_path.exists():
            self.validate_cells()
        if neighborhoods_path.exists():
            self.validate_neighborhoods()
        if splits_path.exists():
            self.validate_splits()

        # Cross-validate
        if cells_path.exists() and neighborhoods_path.exists():
            self.validate_cell_neighborhood_alignment()

        errors = [v for v in self.violations if v.severity == "error"]
        if errors and raise_on_error:
            error_msgs = "\n".join([f"  - [{e.category}] {e.message}" for e in errors])
            raise ValueError(f"Canonical contract violations:\n{error_msgs}")

        return self.violations

    def validate_cells(self) -> pd.DataFrame:
        """Validate cells.parquet against canonical contract."""
        cells_path = self.data_dir / "cells.parquet"
        cells = pd.read_parquet(cells_path)

        # Required columns
        for col in REQUIRED_CELL_COLS:
            if col not in cells.columns:
                self._add_error("cells", f"Missing required column: {col}")

        # Stage values must be canonical
        if "stage" in cells.columns:
            unique_stages = set(cells["stage"].unique())
            invalid_stages = unique_stages - set(CANONICAL_STAGES)
            if invalid_stages:
                self._add_error(
                    "cells",
                    f"Non-canonical stages found: {invalid_stages}. "
                    f"Must be one of {CANONICAL_STAGES}",
                    {"invalid_stages": list(invalid_stages)}
                )

        # Data type must be present and valid
        if "data_type" in cells.columns:
            unique_types = set(cells["data_type"].unique())
            invalid_types = unique_types - set(DATA_TYPES)
            if invalid_types:
                self._add_error(
                    "cells",
                    f"Invalid data_type values: {invalid_types}. "
                    f"Must be one of {DATA_TYPES}",
                )
        else:
            self._add_error(
                "cells",
                "Missing 'data_type' column. Cannot distinguish snRNA from spatial cells."
            )

        # Fused embedding columns
        fused_cols = [f"z_fused_{i}" for i in range(CANONICAL_LATENT_DIM)]
        missing_fused = [c for c in fused_cols if c not in cells.columns]
        if missing_fused:
            self._add_error(
                "cells",
                f"Missing fused embedding columns: {missing_fused[:5]}... "
                f"(expected {CANONICAL_LATENT_DIM} columns z_fused_0 to z_fused_{CANONICAL_LATENT_DIM-1})"
            )

        # WES columns (optional but must be complete if present)
        wes_present = [c for c in CANONICAL_WES_COLS if c in cells.columns]
        if 0 < len(wes_present) < CANONICAL_WES_DIM:
            self._add_error(
                "cells",
                f"Partial WES columns: found {len(wes_present)}/{CANONICAL_WES_DIM}. "
                f"WES must be all or nothing. Missing: {set(CANONICAL_WES_COLS) - set(wes_present)}"
            )

        # Check for NaN in critical columns
        for col in ["cell_id", "donor_id", "stage"]:
            if col in cells.columns and cells[col].isna().any():
                n_nan = cells[col].isna().sum()
                self._add_error("cells", f"Column '{col}' has {n_nan} NaN values")

        return cells

    def validate_neighborhoods(self) -> pd.DataFrame:
        """Validate neighborhoods.parquet against canonical contract."""
        neighborhoods_path = self.data_dir / "neighborhoods.parquet"
        neighborhoods = pd.read_parquet(neighborhoods_path)

        # Required columns
        for col in REQUIRED_NEIGHBORHOOD_COLS:
            if col not in neighborhoods.columns:
                self._add_error("neighborhoods", f"Missing required column: {col}")

        # Validate token structure
        if "tokens" in neighborhoods.columns:
            # Sample check
            n_check = min(100, len(neighborhoods))
            sample_indices = np.random.choice(len(neighborhoods), n_check, replace=False)

            for idx in sample_indices:
                tokens = neighborhoods.iloc[idx]["tokens"]

                if not isinstance(tokens, list):
                    self._add_error(
                        "neighborhoods",
                        f"Row {idx}: 'tokens' must be a list, got {type(tokens)}"
                    )
                    continue

                if len(tokens) != CANONICAL_N_TOKENS:
                    self._add_warning(
                        "neighborhoods",
                        f"Row {idx}: Expected {CANONICAL_N_TOKENS} tokens, got {len(tokens)}"
                    )

                # Check token structure
                for token in tokens:
                    if not isinstance(token, dict):
                        self._add_error(
                            "neighborhoods",
                            f"Row {idx}: Token must be dict, got {type(token)}"
                        )
                        continue

                    if "token_idx" not in token:
                        self._add_error("neighborhoods", f"Row {idx}: Token missing 'token_idx'")
                    if "token_type" not in token:
                        self._add_error("neighborhoods", f"Row {idx}: Token missing 'token_type'")

        return neighborhoods

    def validate_splits(self) -> dict:
        """Validate split_manifest.json against canonical contract."""
        splits_path = self.data_dir / "split_manifest.json"

        with open(splits_path) as f:
            splits = json.load(f)

        # Must have folds
        if "folds" not in splits:
            self._add_error("splits", "Missing 'folds' key in split_manifest.json")
            return splits

        folds = splits["folds"]
        if not isinstance(folds, list) or len(folds) == 0:
            self._add_error("splits", "'folds' must be a non-empty list")
            return splits

        # Each fold must have train/val/test donors
        for i, fold in enumerate(folds):
            for key in ["train_donors", "val_donors", "test_donors"]:
                if key not in fold:
                    self._add_error("splits", f"Fold {i} missing '{key}'")
                elif not isinstance(fold[key], list):
                    self._add_error("splits", f"Fold {i} '{key}' must be a list")

            # Check for donor overlap (leakage)
            if all(k in fold for k in ["train_donors", "val_donors", "test_donors"]):
                train = set(fold["train_donors"])
                val = set(fold["val_donors"])
                test = set(fold["test_donors"])

                train_val_overlap = train & val
                train_test_overlap = train & test
                val_test_overlap = val & test

                if train_val_overlap:
                    self._add_error(
                        "splits",
                        f"Fold {i}: train/val donor overlap (LEAKAGE): {train_val_overlap}"
                    )
                if train_test_overlap:
                    self._add_error(
                        "splits",
                        f"Fold {i}: train/test donor overlap (LEAKAGE): {train_test_overlap}"
                    )
                if val_test_overlap:
                    self._add_error(
                        "splits",
                        f"Fold {i}: val/test donor overlap (LEAKAGE): {val_test_overlap}"
                    )

        return splits

    def validate_cell_neighborhood_alignment(self):
        """Validate that cells and neighborhoods are properly aligned."""
        cells = pd.read_parquet(self.data_dir / "cells.parquet")
        neighborhoods = pd.read_parquet(self.data_dir / "neighborhoods.parquet")

        cell_ids = set(cells["cell_id"])
        neighborhood_cell_ids = set(neighborhoods["cell_id"])

        # Check for cells without neighborhoods
        cells_without_neighborhoods = cell_ids - neighborhood_cell_ids
        if cells_without_neighborhoods:
            n_missing = len(cells_without_neighborhoods)
            pct = 100 * n_missing / len(cell_ids)

            # If spatial cells don't have neighborhoods, that's an error
            if "data_type" in cells.columns:
                spatial_cells = set(cells[cells["data_type"] == "spatial"]["cell_id"])
                spatial_without_niche = spatial_cells & cells_without_neighborhoods
                if spatial_without_niche:
                    self._add_error(
                        "alignment",
                        f"{len(spatial_without_niche)} spatial cells have no neighborhoods. "
                        f"Spatial cells MUST have neighborhoods."
                    )

                # snRNA cells without neighborhoods is a warning
                snrna_cells = set(cells[cells["data_type"] == "snrna"]["cell_id"])
                snrna_without_niche = snrna_cells & cells_without_neighborhoods
                if snrna_without_niche:
                    self._add_warning(
                        "alignment",
                        f"{len(snrna_without_niche)} snRNA cells have no neighborhoods. "
                        f"These will use degenerate (receiver-copy) niches."
                    )
            else:
                self._add_error(
                    "alignment",
                    f"{n_missing} cells ({pct:.1f}%) have no neighborhoods. "
                    f"Cannot determine if this is acceptable without 'data_type' column."
                )

        # Check for orphan neighborhoods
        orphan_neighborhoods = neighborhood_cell_ids - cell_ids
        if orphan_neighborhoods:
            self._add_error(
                "alignment",
                f"{len(orphan_neighborhoods)} neighborhoods have no matching cell in cells.parquet"
            )


def validate_canonical_contract(data_dir: str | Path) -> list[ContractViolation]:
    """Convenience function to validate canonical contract.

    Raises ValueError on any error-level violation.
    """
    validator = CanonicalContractValidator(data_dir)
    return validator.validate_all(raise_on_error=True)


def enforce_canonical_stages(stages: pd.Series) -> pd.Series:
    """Convert stages to canonical format, raising on invalid values.

    DO NOT silently coerce. Fail if stages don't match.
    """
    unique = set(stages.unique())
    invalid = unique - set(CANONICAL_STAGES) - {None, np.nan}

    if invalid:
        raise ValueError(
            f"Non-canonical stages found: {invalid}. "
            f"Valid stages are: {CANONICAL_STAGES}. "
            f"DO NOT silently coerce stages - fix the data source."
        )

    return stages


def get_stage_index(stage: str) -> int:
    """Get numeric index for a canonical stage name."""
    if stage not in STAGE_TO_INDEX:
        raise ValueError(
            f"Invalid stage '{stage}'. Must be one of {CANONICAL_STAGES}"
        )
    return STAGE_TO_INDEX[stage]


def get_stage_name(index: int) -> str:
    """Get stage name from numeric index."""
    if index not in INDEX_TO_STAGE:
        raise ValueError(
            f"Invalid stage index {index}. Must be 0-{len(CANONICAL_STAGES)-1}"
        )
    return INDEX_TO_STAGE[index]
