"""
Base adapter class for dataset-specific handling.

Adapters encapsulate dataset-specific logic for:
- Raw data loading and discovery
- Metadata harmonization (column mapping, ID normalization)
- QC configuration defaults
- Export settings

Subclass this for each dataset (LUAD-Evo, BrainMets, etc.).

Usage:
    class MyDatasetAdapter(DatasetAdapter):
        def load_raw(self) -> AnnData:
            ...

        def harmonize_metadata(self, adata: AnnData) -> AnnData:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration classes
# ---------------------------------------------------------------------------


@dataclass
class AdapterConfig:
    """Configuration for a dataset adapter."""

    name: str
    data_root: Path | None = None
    modality: str = "snRNA"
    donor_column: str = "donor_id"
    sample_column: str = "sample_id"
    stage_column: str = "stage"
    raw_count_layer: str = "counts"
    extra_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "data_root": str(self.data_root) if self.data_root else None,
            "modality": self.modality,
            "donor_column": self.donor_column,
            "sample_column": self.sample_column,
            "stage_column": self.stage_column,
            "raw_count_layer": self.raw_count_layer,
            "extra_config": self.extra_config,
        }


@dataclass
class ColumnMapping:
    """Mapping of source columns to canonical names."""

    donor_id: str | None = None  # Source column for donor_id
    sample_id: str | None = None  # Source column for sample_id
    stage: str | None = None  # Source column for stage
    modality: str | None = None  # Source column for modality
    batch: str | None = None  # Source column for batch
    cell_type: str | None = None  # Source column for cell type

    # Additional mappings
    extra_mappings: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary (source -> target)."""
        result = {
            "donor_id": self.donor_id,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "modality": self.modality,
            "batch": self.batch,
            "cell_type": self.cell_type,
        }
        result.update(self.extra_mappings)
        return {k: v for k, v in result.items() if v is not None}


# ---------------------------------------------------------------------------
# Base adapter class
# ---------------------------------------------------------------------------


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Subclasses must implement:
    - load_raw(): Load raw data
    - harmonize_metadata(): Harmonize column names and IDs
    - get_qc_config(): Return dataset-specific QC configuration
    - get_column_mapping(): Return column name mappings

    Optional overrides:
    - run_qc(): Custom QC logic
    - export(): Custom export logic
    - validate(): Custom validation logic
    """

    def __init__(
        self,
        config: AdapterConfig | dict[str, Any] | None = None,
        data_root: Path | str | None = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        config : AdapterConfig or dict, optional
            Adapter configuration.
        data_root : Path, optional
            Data root directory (overrides config).
        """
        if isinstance(config, dict):
            self._config = AdapterConfig(**config)
        elif config is not None:
            self._config = config
        else:
            self._config = AdapterConfig(name=self.__class__.__name__)

        if data_root is not None:
            self._config.data_root = Path(data_root)

    @property
    def config(self) -> AdapterConfig:
        """Get adapter configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._config.name

    @property
    def data_root(self) -> Path | None:
        """Get data root path."""
        return self._config.data_root

    # -------------------------------------------------------------------------
    # Abstract methods (must implement)
    # -------------------------------------------------------------------------

    @abstractmethod
    def load_raw(self) -> Any:  # AnnData
        """Load raw data.

        Returns
        -------
        AnnData
            Raw AnnData object with original column names.
        """

    @abstractmethod
    def harmonize_metadata(self, adata: Any) -> Any:  # AnnData
        """Harmonize metadata column names and values.

        Should:
        - Rename columns to canonical names (donor_id, sample_id, stage, etc.)
        - Normalize ID formats
        - Apply stage ontology
        - Add modality column

        Parameters
        ----------
        adata : AnnData
            AnnData with original metadata.

        Returns
        -------
        AnnData
            AnnData with harmonized metadata.
        """

    @abstractmethod
    def get_qc_config(self) -> Any:  # QCConfig
        """Get dataset-specific QC configuration.

        Returns
        -------
        QCConfig
            QC configuration with appropriate thresholds.
        """

    @abstractmethod
    def get_column_mapping(self) -> ColumnMapping:
        """Get column name mapping for this dataset.

        Returns
        -------
        ColumnMapping
            Mapping from source columns to canonical names.
        """

    # -------------------------------------------------------------------------
    # Optional methods (can override)
    # -------------------------------------------------------------------------

    def discover_files(self) -> dict[str, list[Path]]:
        """Discover raw data files.

        Returns
        -------
        dict
            Dictionary of file type -> list of paths.
        """
        if self.data_root is None:
            raise ValueError("data_root not set")

        from stagebridge.data.ingest import discover_raw_files

        result = discover_raw_files(self.data_root)
        return {
            "matrix": [f.path for f in result.matrix_files],
            "metadata": [f.path for f in result.metadata_files],
            "coordinates": [f.path for f in result.coordinate_files],
            "images": [f.path for f in result.image_files],
            "archives": [f.path for f in result.archives],
        }

    def run_qc(
        self,
        adata: Any,  # AnnData
        config: Any | None = None,  # QCConfig
    ) -> tuple[Any, Any]:  # (AnnData, QCResult)
        """Run QC filtering.

        Parameters
        ----------
        adata : AnnData
            Input AnnData.
        config : QCConfig, optional
            QC config (uses default if not provided).

        Returns
        -------
        tuple[AnnData, QCResult]
            Filtered AnnData and QC result.
        """
        from stagebridge.data.qc import run_qc

        if config is None:
            config = self.get_qc_config()

        return run_qc(
            adata,
            config,
            donor_column=self._config.donor_column,
            stage_column=self._config.stage_column,
        )

    def export(
        self,
        adata: Any,  # AnnData
        output_dir: Path | str,
        **kwargs: Any,
    ) -> Any:  # ExportResult
        """Export processed data.

        Parameters
        ----------
        adata : AnnData
            Processed AnnData.
        output_dir : Path
            Output directory.
        **kwargs
            Additional export options.

        Returns
        -------
        ExportResult
            Export result.
        """
        from stagebridge.data.export import export_canonical_dataset

        return export_canonical_dataset(
            adata,
            output_dir=output_dir,
            dataset_name=self.name,
            donor_column=self._config.donor_column,
            sample_column=self._config.sample_column,
            stage_column=self._config.stage_column,
            **kwargs,
        )

    def validate(self, adata: Any) -> tuple[bool, list[str]]:  # AnnData
        """Validate processed data.

        Parameters
        ----------
        adata : AnnData
            AnnData to validate.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list of issues)
        """
        issues = []

        # Check required columns
        required = {
            self._config.donor_column,
            self._config.sample_column,
            self._config.stage_column,
        }
        for col in required:
            if col not in adata.obs.columns:
                issues.append(f"Missing required column: {col}")

        # Check for empty data
        if adata.n_obs == 0:
            issues.append("AnnData has 0 observations")
        if adata.n_vars == 0:
            issues.append("AnnData has 0 variables")

        # Check for raw counts
        if self._config.raw_count_layer not in adata.layers:
            issues.append(f"Missing raw counts layer: {self._config.raw_count_layer}")

        return len(issues) == 0, issues

    def get_stage_order(self) -> list[str]:
        """Get canonical stage order for this dataset.

        Returns
        -------
        list[str]
            Stage labels in biological order.
        """
        return ["Normal", "AAH", "AIS", "MIA", "LUAD"]

    def get_marker_genes(self) -> dict[str, list[str]]:
        """Get marker gene sets for this dataset.

        Returns
        -------
        dict
            Category -> list of marker genes.
        """
        return {}

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def _apply_column_mapping(
        self,
        obs: Any,  # pd.DataFrame
        mapping: ColumnMapping,
    ) -> Any:  # pd.DataFrame
        """Apply column mapping to obs DataFrame.

        Parameters
        ----------
        obs : DataFrame
            Original obs DataFrame.
        mapping : ColumnMapping
            Column mapping.

        Returns
        -------
        DataFrame
            DataFrame with renamed columns.
        """
        import pandas as pd

        obs = obs.copy()
        rename_map = {}

        # Build rename map
        for target, source in mapping.to_dict().items():
            if source is not None and source in obs.columns and target != source:
                rename_map[source] = target

        if rename_map:
            obs = obs.rename(columns=rename_map)
            log.info("Renamed columns: %s", rename_map)

        return obs

    def _normalize_ids(
        self,
        obs: Any,  # pd.DataFrame
        column: str,
        *,
        strip_prefix: str | None = None,
        add_prefix: str | None = None,
    ) -> Any:  # pd.DataFrame
        """Normalize ID column.

        Parameters
        ----------
        obs : DataFrame
            DataFrame to modify.
        column : str
            Column name to normalize.
        strip_prefix : str, optional
            Prefix to remove.
        add_prefix : str, optional
            Prefix to add.

        Returns
        -------
        DataFrame
            DataFrame with normalized IDs.
        """
        import re

        if column not in obs.columns:
            return obs

        obs = obs.copy()
        ids = obs[column].astype(str)

        # Strip prefix
        if strip_prefix:
            pattern = f"^{re.escape(strip_prefix)}"
            ids = ids.str.replace(pattern, "", regex=True)

        # Add prefix
        if add_prefix:
            ids = add_prefix + ids

        # Strip whitespace
        ids = ids.str.strip()

        obs[column] = ids
        return obs

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, data_root={self.data_root})"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def apply_stage_mapping(
    stage_column: Any,  # pd.Series
    mapping: dict[str, str],
    *,
    default: str = "Unknown",
) -> Any:  # pd.Series
    """Apply stage label mapping.

    Parameters
    ----------
    stage_column : Series
        Stage labels.
    mapping : dict
        Source -> target stage mapping.
    default : str
        Default for unmapped stages.

    Returns
    -------
    Series
        Mapped stage labels.
    """
    import pandas as pd

    return stage_column.astype(str).map(lambda x: mapping.get(x, default))
