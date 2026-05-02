"""Cancer type support module extending contracts.py.

This module provides cancer type-aware versions of functions from contracts.py,
allowing StageBridge to be used with cancer types other than LUAD.

The approach is additive - it doesn't modify contracts.py but provides
alternative functions that accept cancer_type parameters.

Usage:
    from stagebridge.config.cancer_support import (
        get_stage_system_for_cancer,
        get_valid_stages,
        get_stage_colors,
        validate_contract_for_cancer,
        set_default_cancer_type,
    )

    # Set default cancer type (affects all subsequent calls without explicit cancer_type)
    set_default_cancer_type("pdac")

    # Get stages for PDAC
    stages, s2i, i2s = get_stage_system_for_cancer("3", cancer_type="pdac")

    # Validate data for PDAC
    validate_contract_for_cancer("/path/to/data", cancer_type="pdac")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd

from stagebridge.config.cancer_types import (
    get_cancer_config,
)


# =============================================================================
# Default Cancer Type Management
# =============================================================================

_DEFAULT_CANCER_TYPE = os.environ.get("STAGEBRIDGE_CANCER_TYPE", "luad")


def set_default_cancer_type(cancer_type: str) -> None:
    """Set the default cancer type for this session.

    Args:
        cancer_type: Cancer type name (e.g., "luad", "pdac")

    This affects all subsequent calls that don't explicitly specify cancer_type.
    """
    global _DEFAULT_CANCER_TYPE
    # Validate by attempting to load config
    get_cancer_config(cancer_type)
    _DEFAULT_CANCER_TYPE = cancer_type


def get_default_cancer_type() -> str:
    """Get the current default cancer type."""
    return _DEFAULT_CANCER_TYPE


# =============================================================================
# Stage System Functions (Cancer Type Aware)
# =============================================================================

def get_stage_system_for_cancer(
    system: Literal["3", "4", "5", "full"] = "3",
    cancer_type: str | None = None,
) -> tuple[tuple[str, ...], dict[str, int], dict[int, str]]:
    """Get stage names and mappings for a cancer type.

    This is the cancer type-aware version of contracts.get_stage_system().

    Args:
        system: "3", "4", "5", or "full" stage system
        cancer_type: Cancer type (default: current default)

    Returns:
        (stage_names, stage_to_index, index_to_stage)
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    # For LUAD with standard systems, delegate to contracts for backward compatibility
    if cancer_type == "luad" and system in ("3", "4", "5"):
        from stagebridge.contracts import get_stage_system
        return get_stage_system(system)

    # For other cancer types or "full" system, use config
    config = get_cancer_config(cancer_type)
    stages = config.stages.get_stages(system)

    s2i = {s: i for i, s in enumerate(stages)}
    i2s = {i: s for i, s in enumerate(stages)}
    return stages, s2i, i2s


def convert_stage_for_cancer(
    stage: str,
    to_system: Literal["3", "4"],
    cancer_type: str | None = None,
) -> str:
    """Convert a stage name to a coarser system.

    Args:
        stage: Stage name to convert
        to_system: Target system ("3" or "4")
        cancer_type: Cancer type (default: current default)

    Returns:
        Stage name in target system
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    # For LUAD, delegate to contracts for backward compatibility
    if cancer_type == "luad":
        from stagebridge.contracts import convert_stage
        return convert_stage(stage, to_system)

    # For other cancer types, use config
    config = get_cancer_config(cancer_type)
    if to_system == "3":
        return config.stages.convert_to_3(stage)
    else:
        return config.stages.convert_to_4(stage)


def get_valid_stages(cancer_type: str | None = None) -> set[str]:
    """Get all valid stage names for a cancer type.

    Returns stages from all systems (3, 4, full) combined.
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    if cancer_type == "luad":
        from stagebridge.contracts import STAGES_3, STAGES_4, STAGES_5
        return set(STAGES_3) | set(STAGES_4) | set(STAGES_5)

    config = get_cancer_config(cancer_type)
    valid = set(config.stages.stages_3) | set(config.stages.stages_full)
    if config.stages.stages_4:
        valid |= set(config.stages.stages_4)
    return valid


def get_stage_colors(cancer_type: str | None = None) -> dict[str, str]:
    """Get stage colors for visualization.

    Args:
        cancer_type: Cancer type (default: current default)

    Returns:
        Dict mapping stage name to hex color
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    config = get_cancer_config(cancer_type)
    colors = config.stages.stage_colors.copy()
    colors.setdefault("Unknown", "#696969")
    return colors


def stage_to_idx_for_cancer(
    stage: str,
    system: Literal["3", "4", "5", "full"] = "3",
    cancer_type: str | None = None,
) -> int:
    """Get numeric index for a stage name.

    Args:
        stage: Stage name
        system: Stage system
        cancer_type: Cancer type (default: current default)

    Returns:
        Numeric index for the stage
    """
    _, s2i, _ = get_stage_system_for_cancer(system, cancer_type)
    if stage not in s2i:
        raise ValueError(f"Invalid stage '{stage}' for {system}-stage system")
    return s2i[stage]


def idx_to_stage_for_cancer(
    idx: int,
    system: Literal["3", "4", "5", "full"] = "3",
    cancer_type: str | None = None,
) -> str:
    """Get stage name from numeric index.

    Args:
        idx: Numeric index
        system: Stage system
        cancer_type: Cancer type (default: current default)

    Returns:
        Stage name
    """
    _, _, i2s = get_stage_system_for_cancer(system, cancer_type)
    if idx not in i2s:
        raise ValueError(f"Invalid index {idx} for {system}-stage system")
    return i2s[idx]


# =============================================================================
# Reference Atlas Functions
# =============================================================================

def get_reference_dims(cancer_type: str | None = None) -> dict[str, int]:
    """Get reference atlas dimensions for a cancer type.

    Args:
        cancer_type: Cancer type (default: current default)

    Returns:
        Dict mapping reference name to latent dimension
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    if cancer_type == "luad":
        from stagebridge.contracts import HLCA_DIM, LUCA_DIM
        return {"hlca": HLCA_DIM, "luca": LUCA_DIM}

    config = get_cancer_config(cancer_type)
    return {name: ref.latent_dim for name, ref in config.references.items()}


def get_fused_dim_for_cancer(
    method: Literal["concat", "weighted", "gated", "film"] = "concat",
    cancer_type: str | None = None,
) -> int:
    """Get output dimension for a fusion method.

    Args:
        method: Fusion method
        cancer_type: Cancer type (default: current default)

    Returns:
        Output dimension of fused embedding
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    ref_dims = get_reference_dims(cancer_type)

    if not ref_dims:
        # No references configured - return 0 for no-reference mode
        return 0

    dims = list(ref_dims.values())
    total_dim = sum(dims)
    max_dim = max(dims) if dims else 0

    if method == "concat":
        return total_dim
    elif method in ("weighted", "gated", "film"):
        return max_dim
    else:
        raise ValueError(f"Unknown fusion method: {method}")


# =============================================================================
# Token Structure Functions
# =============================================================================

def get_token_structure(cancer_type: str | None = None) -> tuple[int, tuple[str, ...]]:
    """Get token structure for a cancer type.

    The token structure depends on the reference atlas configuration:
    - Dual reference: 9 tokens (receiver, 4 rings, ref1, ref2, pathway, stats)
    - Single reference: 8 tokens (receiver, 4 rings, ref, pathway, stats)
    - No reference: 7 tokens (receiver, 4 rings, pathway, stats)

    Args:
        cancer_type: Cancer type (default: current default)

    Returns:
        (n_tokens, token_names)
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    # For LUAD, use standard 9-token structure
    if cancer_type == "luad":
        from stagebridge.contracts import N_TOKENS, TOKEN_NAMES
        return N_TOKENS, TOKEN_NAMES

    config = get_cancer_config(cancer_type)

    # Base tokens: receiver + 4 rings
    base_tokens = ["receiver", "ring1", "ring2", "ring3", "ring4"]

    # Add reference tokens based on mode
    if config.reference_mode == "dual" and config.primary_reference and config.secondary_reference:
        base_tokens.append(config.primary_reference)
        base_tokens.append(config.secondary_reference)
    elif config.reference_mode == "single" and config.primary_reference:
        base_tokens.append(config.primary_reference)

    # Add pathway and stats
    base_tokens.extend(["pathway", "stats"])

    return len(base_tokens), tuple(base_tokens)


def get_token_type_ids(cancer_type: str | None = None) -> dict[str, int]:
    """Get token type IDs for a cancer type.

    Args:
        cancer_type: Cancer type (default: current default)

    Returns:
        Dict mapping token name/type to ID
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    # For LUAD, use standard type IDs
    if cancer_type == "luad":
        from stagebridge.contracts import TOKEN_TYPE_IDS
        return TOKEN_TYPE_IDS.copy()

    config = get_cancer_config(cancer_type)

    type_ids = {
        "receiver": 0,
        "spatial": 1,  # Covers ring1-4
    }

    next_id = 2
    if config.reference_mode == "dual" and config.primary_reference and config.secondary_reference:
        type_ids[config.primary_reference] = next_id
        next_id += 1
        type_ids[config.secondary_reference] = next_id
        next_id += 1
    elif config.reference_mode == "single" and config.primary_reference:
        type_ids[config.primary_reference] = next_id
        next_id += 1

    type_ids["pathway"] = next_id
    type_ids["stats"] = next_id + 1

    return type_ids


# =============================================================================
# Contract Validation (Cancer Type Aware)
# =============================================================================

def validate_contract_for_cancer(
    data_dir: str | Path,
    cancer_type: str | None = None,
    raise_on_error: bool = True,
) -> list:
    """Validate canonical contract for a specific cancer type.

    Args:
        data_dir: Path to canonical data directory
        cancer_type: Cancer type (default: current default)
        raise_on_error: Whether to raise on validation errors

    Returns:
        List of contract violations
    """
    from stagebridge.contracts import (
        ContractViolation,
        CELLS_REQUIRED_COLS,
        DATA_TYPES,
        NEIGHBORHOODS_REQUIRED_COLS,
        NEIGHBORHOODS_TOKENS_COL,
        NEIGHBORHOODS_RING_COLS,
    )
    import json
    import numpy as np

    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    data_dir = Path(data_dir)
    violations = []
    valid_stages = get_valid_stages(cancer_type)

    def _error(category: str, message: str):
        violations.append(ContractViolation("error", category, message))

    def _warning(category: str, message: str):
        violations.append(ContractViolation("warning", category, message))

    # Check required files
    cells_path = data_dir / "cells.parquet"
    neighborhoods_path = data_dir / "neighborhoods.parquet"
    splits_path = data_dir / "split_manifest.json"

    if not cells_path.exists():
        _error("files", f"cells.parquet not found at {cells_path}")
    if not neighborhoods_path.exists():
        _error("files", f"neighborhoods.parquet not found at {neighborhoods_path}")
    if not splits_path.exists():
        _error("files", f"split_manifest.json not found at {splits_path}")

    errors = [v for v in violations if v.severity == "error"]
    if errors and raise_on_error:
        raise ValueError(f"Contract violations: {[e.message for e in errors]}")

    # Validate cells
    if cells_path.exists():
        cells = pd.read_parquet(cells_path)

        for col in CELLS_REQUIRED_COLS:
            if col not in cells.columns:
                _error("cells", f"Missing required column: {col}")

        if "stage" in cells.columns:
            unique = set(cells["stage"].unique())
            invalid = unique - valid_stages
            if invalid:
                _error("cells", f"Invalid stages for {cancer_type}: {invalid}")

        if "data_type" in cells.columns:
            invalid = set(cells["data_type"].unique()) - set(DATA_TYPES)
            if invalid:
                _error("cells", f"Invalid data_type: {invalid}")

        for col in ["cell_id", "donor_id", "stage"]:
            if col in cells.columns and cells[col].isna().any():
                _error("cells", f"NaN values in {col}")

    # Validate neighborhoods
    if neighborhoods_path.exists():
        neighborhoods = pd.read_parquet(neighborhoods_path)

        for col in NEIGHBORHOODS_REQUIRED_COLS:
            if col not in neighborhoods.columns:
                _error("neighborhoods", f"Missing required column: {col}")

        has_tokens_col = NEIGHBORHOODS_TOKENS_COL in neighborhoods.columns
        has_ring_cols = all(c in neighborhoods.columns for c in NEIGHBORHOODS_RING_COLS)

        if not has_tokens_col and not has_ring_cols:
            _error("neighborhoods", f"Missing token structure")

        if has_tokens_col:
            n_tokens, _ = get_token_structure(cancer_type)
            n_check = min(100, len(neighborhoods))
            indices = np.random.choice(len(neighborhoods), n_check, replace=False)

            for idx in indices:
                tokens = neighborhoods.iloc[idx][NEIGHBORHOODS_TOKENS_COL]
                if not isinstance(tokens, (list, np.ndarray)):
                    _error("neighborhoods", f"Row {idx}: tokens not list/array")
                    continue
                if len(tokens) != n_tokens:
                    _warning("neighborhoods", f"Row {idx}: {len(tokens)} tokens, expected {n_tokens}")

    # Validate splits
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)

        if "folds" not in splits:
            _error("splits", "Missing 'folds' key")
        else:
            for i, fold in enumerate(splits["folds"]):
                for key in ["train_donors", "val_donors", "test_donors"]:
                    if key not in fold:
                        _error("splits", f"Fold {i} missing '{key}'")

                if all(k in fold for k in ["train_donors", "val_donors", "test_donors"]):
                    train = set(fold["train_donors"])
                    val = set(fold["val_donors"])
                    test = set(fold["test_donors"])

                    if train & val:
                        _error("splits", f"Fold {i}: train/val overlap (LEAKAGE)")
                    if train & test:
                        _error("splits", f"Fold {i}: train/test overlap (LEAKAGE)")
                    if val & test:
                        _error("splits", f"Fold {i}: val/test overlap (LEAKAGE)")

    errors = [v for v in violations if v.severity == "error"]
    if errors and raise_on_error:
        msgs = "\n".join([f"  - [{e.category}] {e.message}" for e in errors])
        raise ValueError(f"Contract violations:\n{msgs}")

    return violations


# =============================================================================
# Known Biology Functions
# =============================================================================

def get_known_mechanisms(cancer_type: str | None = None):
    """Get known biological mechanisms for a cancer type.

    Args:
        cancer_type: Cancer type (default: current default)

    Returns:
        Tuple of BiologicalMechanism objects
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    config = get_cancer_config(cancer_type)
    return config.known_mechanisms


def get_cell_markers(cancer_type: str | None = None):
    """Get cell type markers for a cancer type.

    Args:
        cancer_type: Cancer type (default: current default)

    Returns:
        CellTypeMarkers object or None
    """
    if cancer_type is None:
        cancer_type = _DEFAULT_CANCER_TYPE

    config = get_cancer_config(cancer_type)
    return config.cell_markers
