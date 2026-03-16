"""Typed schema objects for the active StageBridge data layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True, frozen=True)
class DatasetManifest:
    """Named dataset contract used by data-layer readers."""

    name: str
    modality: str
    stage_column: str = "stage"
    donor_column: str = "patient_id"


@dataclass(slots=True, frozen=True)
class LuadEvoPaths:
    """Resolved asset paths for the active LUAD evolution cohort."""

    data_root: Path
    snrna_h5ad: Path
    snrna_latent_h5ad: Path
    hlca_labels_parquet: Path
    spatial_h5ad: Path
    spatial_tangram_h5ad: Path
    tangram_scores_parquet: Path
    niche_token_bank_zarr: Path
    wes_features_path: Path
    hlca_h5ad: Path | None


@dataclass(slots=True, frozen=True)
class LatentCohort:
    """In-memory latent table for transition learning."""

    latent: np.ndarray
    obs: pd.DataFrame
    feature_names: tuple[str, ...]
    source_path: Path
    latent_key: str

    @property
    def n_obs(self) -> int:
        return int(self.latent.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.latent.shape[1])


@dataclass(slots=True, frozen=True)
class SpatialCohort:
    """Spot-level Tangram spatial mapping table used by the context model."""

    compositions: np.ndarray
    coords: np.ndarray
    obs: pd.DataFrame
    feature_names: tuple[str, ...]
    source_path: Path

    @property
    def n_spots(self) -> int:
        return int(self.compositions.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.compositions.shape[1])


@dataclass(slots=True, frozen=True)
class WESCohort:
    """Per-donor, per-stage WES feature table."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    source_path: Path

    @property
    def n_rows(self) -> int:
        return int(self.frame.shape[0])
