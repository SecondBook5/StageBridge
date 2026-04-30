"""WES feature integration for cells.parquet.

Merges per-sample WES features (from variant-calling-pipeline) into cells.parquet
for model training. WES features are sample-level but get broadcast to all cells
in that sample.

Usage:
    from stagebridge.genomics.wes_integration import merge_wes_into_cells

    cells = merge_wes_into_cells(
        cells_path="data/cells.parquet",
        wes_path="wes_features.parquet",
        output_path="data/cells_with_wes.parquet",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from stagebridge.contracts import WES_COLS

logger = logging.getLogger(__name__)


def load_wes_features(wes_path: str | Path) -> pd.DataFrame:
    """Load WES features from parquet.

    Args:
        wes_path: Path to wes_features.parquet

    Returns:
        DataFrame with sample_id + WES_COLS
    """
    wes_path = Path(wes_path)
    if not wes_path.exists():
        raise FileNotFoundError(f"WES features not found: {wes_path}")

    df = pd.read_parquet(wes_path)

    if "sample_id" not in df.columns:
        raise ValueError("wes_features.parquet must have 'sample_id' column")

    missing = [col for col in WES_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing WES columns: {missing}")

    return df


def merge_wes_into_cells(
    cells_path: str | Path,
    wes_path: str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Merge WES features into cells.parquet.

    WES features are per-sample but get broadcast to all cells in that sample.
    This is the standard approach for lesion-level genomic features.

    Args:
        cells_path: Path to cells.parquet
        wes_path: Path to wes_features.parquet (from variant-calling-pipeline)
        output_path: If provided, save merged result here
        overwrite: Whether to overwrite existing WES columns

    Returns:
        Merged DataFrame

    Raises:
        ValueError: If WES columns already exist and overwrite=False
    """
    cells_path = Path(cells_path)
    wes_path = Path(wes_path)

    logger.info(f"Loading cells from {cells_path}")
    cells = pd.read_parquet(cells_path)
    n_cells = len(cells)

    existing_wes = [col for col in WES_COLS if col in cells.columns]
    if existing_wes and not overwrite:
        raise ValueError(
            f"WES columns already exist in cells.parquet: {existing_wes}. "
            "Use overwrite=True to replace them."
        )

    if existing_wes and overwrite:
        logger.warning(f"Overwriting existing WES columns: {existing_wes}")
        cells = cells.drop(columns=existing_wes)

    logger.info(f"Loading WES features from {wes_path}")
    wes = load_wes_features(wes_path)
    n_samples = len(wes)

    merge_cols = ["sample_id"] + list(WES_COLS)
    wes_merge = wes[merge_cols].copy()

    logger.info(f"Merging WES features: {n_samples} samples -> {n_cells} cells")
    cells = cells.merge(wes_merge, on="sample_id", how="left")

    n_with_wes = cells[WES_COLS[0]].notna().sum()
    n_without = n_cells - n_with_wes
    if n_without > 0:
        logger.warning(f"{n_without} cells have no matching WES data (will be NaN)")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cells.to_parquet(output_path, index=False)
        logger.info(f"Saved merged cells to {output_path}")

    return cells


def validate_wes_coverage(
    cells: pd.DataFrame,
    wes: pd.DataFrame,
) -> dict:
    """Check WES coverage over cells.

    Args:
        cells: cells.parquet DataFrame
        wes: wes_features.parquet DataFrame

    Returns:
        Coverage statistics
    """
    cells_samples = set(cells["sample_id"].dropna().unique())
    wes_samples = set(wes["sample_id"].dropna().unique())

    covered = cells_samples & wes_samples
    missing = cells_samples - wes_samples
    extra = wes_samples - cells_samples

    cells_covered = cells[cells["sample_id"].isin(covered)]

    return {
        "n_cells_total": len(cells),
        "n_cells_covered": len(cells_covered),
        "coverage_pct": 100 * len(cells_covered) / len(cells) if len(cells) > 0 else 0,
        "n_samples_cells": len(cells_samples),
        "n_samples_wes": len(wes_samples),
        "n_samples_covered": len(covered),
        "missing_in_wes": sorted(missing),
        "extra_in_wes": sorted(extra),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge WES features into cells.parquet")
    parser.add_argument("--cells", type=Path, required=True, help="Path to cells.parquet")
    parser.add_argument("--wes", type=Path, required=True, help="Path to wes_features.parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WES columns")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    merge_wes_into_cells(
        cells_path=args.cells,
        wes_path=args.wes,
        output_path=args.output,
        overwrite=args.overwrite,
    )
