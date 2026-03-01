"""
Manifest helpers — shared utilities for building and validating
sample manifests used by both snRNA and spatial pipelines.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)

REQUIRED_SNRNA_COLS = {"sample_id", "input_path", "gsm", "patient_id", "stage"}
REQUIRED_SPATIAL_COLS = {"sample_id", "sample_dir", "gsm", "patient_id", "stage"}


def load_manifest(csv_path: Path, required_cols: set[str] | None = None) -> pd.DataFrame:
    """Load and validate a manifest CSV.

    Parameters
    ----------
    csv_path : Path
        Path to the manifest CSV.
    required_cols : set[str] or None
        If provided, raises ValueError if any column is missing.

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Manifest CSV not found: {csv_path}\n"
            f"Generate it first with build_snrna_manifest() or build_spatial_manifest()."
        )
    df = pd.read_csv(csv_path)
    if required_cols:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Manifest {csv_path} is missing required columns: {missing}\n"
                f"Found: {list(df.columns)}"
            )
    log.info("Loaded manifest: %d rows from %s", len(df), csv_path)
    return df


def validate_snrna_manifest(csv_path: Path) -> pd.DataFrame:
    """Load and validate a snRNA manifest CSV."""
    return load_manifest(csv_path, required_cols=REQUIRED_SNRNA_COLS)


def validate_spatial_manifest(csv_path: Path) -> pd.DataFrame:
    """Load and validate a spatial manifest CSV."""
    return load_manifest(csv_path, required_cols=REQUIRED_SPATIAL_COLS)
