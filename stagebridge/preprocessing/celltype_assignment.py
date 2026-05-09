"""Cell type assignment utilities for deconvolved spatial data.

Assigns cell types to 'mixed' cells using dominant DestVI gamma coefficients.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def assign_celltypes_from_gamma(
    cells: pd.DataFrame,
    gamma_cols: list[str] | None = None,
    celltype_col: str = "cell_type",
    output_col: str = "cell_type_assigned",
    mixed_label: str = "mixed",
) -> pd.DataFrame:
    """Assign cell types to 'mixed' cells using dominant DestVI gamma.

    For cells labeled 'mixed' (from deconvolution), assigns a cell type
    based on which gamma coefficient is dominant.

    Args:
        cells: DataFrame with cell_type and gamma columns
        gamma_cols: List of gamma column names (default: gamma_0 through gamma_9)
        celltype_col: Column name for existing cell types
        output_col: Column name for assigned cell types
        mixed_label: Label used for mixed cells

    Returns:
        DataFrame with output_col added
    """
    cells = cells.copy()

    if gamma_cols is None:
        gamma_cols = [f"gamma_{i}" for i in range(10)]

    available_gamma = [c for c in gamma_cols if c in cells.columns]
    if not available_gamma:
        logger.warning("No gamma columns found, returning original cell types")
        cells[output_col] = cells[celltype_col]
        return cells

    logger.info(f"Using {len(available_gamma)} gamma columns for assignment")

    gamma_to_ct = _build_gamma_mapping(cells, available_gamma, celltype_col, mixed_label)
    logger.info(f"Built gamma->celltype mapping: {gamma_to_ct}")

    gamma_matrix = cells[available_gamma].values
    dominant_idx = np.argmax(gamma_matrix, axis=1)

    cells[output_col] = cells[celltype_col].copy()
    mixed_mask = cells[celltype_col] == mixed_label
    n_mixed = mixed_mask.sum()
    logger.info(f"Assigning types to {n_mixed:,} mixed cells")

    for idx, ct in gamma_to_ct.items():
        if idx < len(available_gamma):
            assign_mask = mixed_mask & (dominant_idx == idx)
            cells.loc[assign_mask, output_col] = ct
            logger.info(f"  gamma_{idx} -> {ct}: {assign_mask.sum():,} cells")

    remaining = cells[output_col] == mixed_label
    if remaining.sum() > 0:
        logger.warning(f"{remaining.sum():,} cells still unassigned (gamma index not in mapping)")
        for idx in range(len(available_gamma)):
            if idx not in gamma_to_ct:
                assign_mask = remaining & (dominant_idx == idx)
                if assign_mask.sum() > 0:
                    cells.loc[assign_mask, output_col] = f"Type_{idx}"
                    logger.info(f"  gamma_{idx} -> Type_{idx}: {assign_mask.sum():,} cells")

    return cells


def _build_gamma_mapping(
    cells: pd.DataFrame,
    gamma_cols: list[str],
    celltype_col: str,
    mixed_label: str,
) -> dict[int, str]:
    """Build mapping from gamma index to cell type using non-mixed cells."""
    assigned = cells[cells[celltype_col] != mixed_label]

    gamma_to_ct = {}
    for ct in assigned[celltype_col].unique():
        if pd.isna(ct):
            continue
        mask = assigned[celltype_col] == ct
        gamma_vals = assigned.loc[mask, gamma_cols].values
        dominant = np.argmax(gamma_vals, axis=1)
        most_common = pd.Series(dominant).value_counts().index[0]
        gamma_to_ct[most_common] = ct

    return gamma_to_ct


def run_celltype_assignment(
    cells_path: str | Path,
    output_path: str | Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run cell type assignment on cells.parquet file.

    Args:
        cells_path: Path to cells.parquet
        output_path: Path to save results (optional)
        **kwargs: Additional arguments to assign_celltypes_from_gamma

    Returns:
        DataFrame with assigned cell types
    """
    cells_path = Path(cells_path)
    logger.info(f"Loading {cells_path}")
    cells = pd.read_parquet(cells_path)
    logger.info(f"Loaded {len(cells):,} cells")

    result = assign_celltypes_from_gamma(cells, **kwargs)

    logger.info(f"Final distribution:\n{result['cell_type_assigned'].value_counts()}")

    if output_path:
        output_path = Path(output_path)
        result.to_parquet(output_path)
        logger.info(f"Saved to {output_path}")

    return result
