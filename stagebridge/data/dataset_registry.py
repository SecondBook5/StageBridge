"""
Dataset registration and tracking for StageBridge.

This module handles:
- Dataset registration with modality tracking
- Donor/sample/stage enumeration
- Dataset lookup and listing
- Registry persistence

Usage:
    from stagebridge.data.dataset_registry import DatasetRegistry

    registry = DatasetRegistry(registry_dir="data/registry")
    registry.register_dataset("luad_evo", modality="snrna", paths={...})
    info = registry.get_dataset("luad_evo")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DatasetInfo:
    """Information about a registered dataset."""

    name: str
    modality: str  # snRNA, snATAC, spatial, wes, multi
    paths: dict[str, str]  # Key paths (h5ad, parquet, etc.)
    n_donors: int = 0
    n_samples: int = 0
    n_cells: int = 0
    n_spots: int = 0
    n_genes: int = 0
    stages: list[str] = field(default_factory=list)
    donors: list[str] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"
    description: str = ""
    source_url: str | None = None
    processed: bool = False
    validated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "modality": self.modality,
            "paths": self.paths,
            "n_donors": self.n_donors,
            "n_samples": self.n_samples,
            "n_cells": self.n_cells,
            "n_spots": self.n_spots,
            "n_genes": self.n_genes,
            "stages": self.stages,
            "donors": self.donors,
            "registered_at": self.registered_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "description": self.description,
            "source_url": self.source_url,
            "processed": self.processed,
            "validated": self.validated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            modality=data["modality"],
            paths=data.get("paths", {}),
            n_donors=data.get("n_donors", 0),
            n_samples=data.get("n_samples", 0),
            n_cells=data.get("n_cells", 0),
            n_spots=data.get("n_spots", 0),
            n_genes=data.get("n_genes", 0),
            stages=data.get("stages", []),
            donors=data.get("donors", []),
            registered_at=data.get("registered_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            source_url=data.get("source_url"),
            processed=data.get("processed", False),
            validated=data.get("validated", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ModalityInfo:
    """Information about a data modality within a dataset."""

    modality: str
    datasets: list[str] = field(default_factory=list)
    total_cells: int = 0
    total_spots: int = 0
    total_donors: int = 0
    stages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class DatasetRegistry:
    """Registry for tracking datasets and modalities.

    Provides:
    - Dataset registration and lookup
    - Modality tracking
    - Donor/stage enumeration across datasets
    - Persistence to JSON

    Parameters
    ----------
    registry_dir : Path, optional
        Directory for registry persistence (default: in-memory only).
    """

    def __init__(self, registry_dir: str | Path | None = None) -> None:
        """Initialize the registry.

        Parameters
        ----------
        registry_dir : Path, optional
            Directory for persistence. If None, registry is in-memory only.
        """
        self._datasets: dict[str, DatasetInfo] = {}
        self._registry_dir = Path(registry_dir) if registry_dir else None

        if self._registry_dir is not None:
            self._registry_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def _registry_path(self) -> Path | None:
        """Return path to registry JSON file."""
        if self._registry_dir is None:
            return None
        return self._registry_dir / "registry.json"

    def _load(self) -> None:
        """Load registry from disk."""
        path = self._registry_path()
        if path is None or not path.exists():
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for dataset_data in data.get("datasets", []):
                info = DatasetInfo.from_dict(dataset_data)
                self._datasets[info.name] = info

            log.info("Loaded %d datasets from registry", len(self._datasets))
        except Exception as e:
            log.warning("Failed to load registry: %s", e)

    def _save(self) -> None:
        """Save registry to disk."""
        path = self._registry_path()
        if path is None:
            return

        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "n_datasets": len(self._datasets),
            "datasets": [info.to_dict() for info in self._datasets.values()],
        }

        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            log.debug("Saved registry: %d datasets", len(self._datasets))
        except Exception as e:
            log.warning("Failed to save registry: %s", e)

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register_dataset(
        self,
        name: str,
        modality: str,
        paths: dict[str, str] | None = None,
        *,
        n_donors: int = 0,
        n_samples: int = 0,
        n_cells: int = 0,
        n_spots: int = 0,
        n_genes: int = 0,
        stages: list[str] | None = None,
        donors: list[str] | None = None,
        description: str = "",
        source_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> DatasetInfo:
        """Register a dataset.

        Parameters
        ----------
        name : str
            Unique dataset name.
        modality : str
            Data modality (snRNA, snATAC, spatial, wes, multi).
        paths : dict, optional
            Key paths (e.g., {"h5ad": "/path/to/data.h5ad"}).
        n_donors, n_samples, n_cells, n_spots, n_genes : int
            Dataset statistics.
        stages : list[str], optional
            Stage labels in dataset.
        donors : list[str], optional
            Donor IDs in dataset.
        description : str
            Dataset description.
        source_url : str, optional
            Source URL (e.g., GEO accession).
        metadata : dict, optional
            Additional metadata.
        overwrite : bool
            Whether to overwrite existing registration.

        Returns
        -------
        DatasetInfo
            The registered dataset info.
        """
        if name in self._datasets and not overwrite:
            raise ValueError(
                f"Dataset '{name}' already registered. Use overwrite=True to replace."
            )

        info = DatasetInfo(
            name=name,
            modality=modality,
            paths=paths or {},
            n_donors=n_donors,
            n_samples=n_samples,
            n_cells=n_cells,
            n_spots=n_spots,
            n_genes=n_genes,
            stages=stages or [],
            donors=donors or [],
            description=description,
            source_url=source_url,
            metadata=metadata or {},
        )

        self._datasets[name] = info
        self._save()

        log.info(
            "Registered dataset '%s' (%s): %d donors, %d cells, %d stages",
            name,
            modality,
            n_donors,
            n_cells,
            len(stages or []),
        )

        return info

    def register_from_adata(
        self,
        name: str,
        adata: Any,  # AnnData
        modality: str,
        *,
        h5ad_path: str | Path | None = None,
        donor_column: str = "donor_id",
        sample_column: str = "sample_id",
        stage_column: str = "stage",
        description: str = "",
        source_url: str | None = None,
        overwrite: bool = False,
    ) -> DatasetInfo:
        """Register a dataset from AnnData.

        Automatically extracts statistics and metadata from adata.

        Parameters
        ----------
        name : str
            Dataset name.
        adata : AnnData
            AnnData object.
        modality : str
            Data modality.
        h5ad_path : Path, optional
            Path to h5ad file.
        donor_column, sample_column, stage_column : str
            Column names in adata.obs.
        description, source_url : str
            Metadata.
        overwrite : bool
            Whether to overwrite existing.

        Returns
        -------
        DatasetInfo
            The registered dataset info.
        """
        # Extract statistics
        n_cells = adata.n_obs
        n_genes = adata.n_vars
        n_spots = n_cells if modality == "spatial" else 0

        donors = []
        if donor_column in adata.obs.columns:
            donors = sorted(adata.obs[donor_column].astype(str).unique().tolist())

        samples = []
        if sample_column in adata.obs.columns:
            samples = sorted(adata.obs[sample_column].astype(str).unique().tolist())

        stages = []
        if stage_column in adata.obs.columns:
            stages = sorted(adata.obs[stage_column].astype(str).unique().tolist())

        paths = {}
        if h5ad_path is not None:
            paths["h5ad"] = str(h5ad_path)

        return self.register_dataset(
            name=name,
            modality=modality,
            paths=paths,
            n_donors=len(donors),
            n_samples=len(samples),
            n_cells=n_cells if modality != "spatial" else 0,
            n_spots=n_spots,
            n_genes=n_genes,
            stages=stages,
            donors=donors,
            description=description,
            source_url=source_url,
            overwrite=overwrite,
        )

    def update_dataset(
        self,
        name: str,
        *,
        processed: bool | None = None,
        validated: bool | None = None,
        paths: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DatasetInfo:
        """Update a registered dataset.

        Parameters
        ----------
        name : str
            Dataset name.
        processed : bool, optional
            Whether dataset has been processed.
        validated : bool, optional
            Whether dataset has been validated.
        paths : dict, optional
            Additional paths to add.
        metadata : dict, optional
            Additional metadata to add.
        **kwargs
            Other fields to update.

        Returns
        -------
        DatasetInfo
            Updated dataset info.
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not registered")

        info = self._datasets[name]

        if processed is not None:
            info.processed = processed
        if validated is not None:
            info.validated = validated
        if paths is not None:
            info.paths.update(paths)
        if metadata is not None:
            info.metadata.update(metadata)

        # Update other fields
        for key, value in kwargs.items():
            if hasattr(info, key):
                setattr(info, key, value)

        info.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()

        log.info("Updated dataset '%s'", name)
        return info

    def unregister_dataset(self, name: str) -> None:
        """Remove a dataset from the registry.

        Parameters
        ----------
        name : str
            Dataset name.
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not registered")

        del self._datasets[name]
        self._save()
        log.info("Unregistered dataset '%s'", name)

    # -------------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------------

    def get_dataset(self, name: str) -> DatasetInfo:
        """Get dataset info by name.

        Parameters
        ----------
        name : str
            Dataset name.

        Returns
        -------
        DatasetInfo
            Dataset information.

        Raises
        ------
        KeyError
            If dataset not found.
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(self._datasets.keys())}")
        return self._datasets[name]

    def has_dataset(self, name: str) -> bool:
        """Check if dataset is registered.

        Parameters
        ----------
        name : str
            Dataset name.

        Returns
        -------
        bool
            True if registered.
        """
        return name in self._datasets

    def list_datasets(
        self,
        *,
        modality: str | None = None,
        processed: bool | None = None,
        validated: bool | None = None,
    ) -> list[str]:
        """List registered dataset names.

        Parameters
        ----------
        modality : str, optional
            Filter by modality.
        processed : bool, optional
            Filter by processed status.
        validated : bool, optional
            Filter by validated status.

        Returns
        -------
        list[str]
            Dataset names matching filters.
        """
        names = []
        for name, info in self._datasets.items():
            if modality is not None and info.modality != modality:
                continue
            if processed is not None and info.processed != processed:
                continue
            if validated is not None and info.validated != validated:
                continue
            names.append(name)
        return sorted(names)

    def get_all_datasets(self) -> dict[str, DatasetInfo]:
        """Get all registered datasets.

        Returns
        -------
        dict
            Dictionary of dataset name -> DatasetInfo.
        """
        return dict(self._datasets)

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def get_modality_info(self, modality: str) -> ModalityInfo:
        """Get aggregated information for a modality.

        Parameters
        ----------
        modality : str
            Modality name.

        Returns
        -------
        ModalityInfo
            Aggregated modality information.
        """
        datasets = [name for name, info in self._datasets.items() if info.modality == modality]

        total_cells = sum(self._datasets[name].n_cells for name in datasets)
        total_spots = sum(self._datasets[name].n_spots for name in datasets)

        all_donors = set()
        all_stages = set()
        for name in datasets:
            all_donors.update(self._datasets[name].donors)
            all_stages.update(self._datasets[name].stages)

        return ModalityInfo(
            modality=modality,
            datasets=datasets,
            total_cells=total_cells,
            total_spots=total_spots,
            total_donors=len(all_donors),
            stages=sorted(all_stages),
        )

    def get_all_donors(self) -> list[str]:
        """Get all unique donor IDs across datasets.

        Returns
        -------
        list[str]
            Sorted list of unique donor IDs.
        """
        all_donors = set()
        for info in self._datasets.values():
            all_donors.update(info.donors)
        return sorted(all_donors)

    def get_all_stages(self) -> list[str]:
        """Get all unique stage labels across datasets.

        Returns
        -------
        list[str]
            Sorted list of unique stages.
        """
        all_stages = set()
        for info in self._datasets.values():
            all_stages.update(info.stages)
        return sorted(all_stages)

    def get_modalities(self) -> list[str]:
        """Get all modalities in registry.

        Returns
        -------
        list[str]
            List of modalities.
        """
        return sorted(set(info.modality for info in self._datasets.values()))

    def get_donor_datasets(self, donor_id: str) -> list[str]:
        """Get datasets containing a specific donor.

        Parameters
        ----------
        donor_id : str
            Donor ID.

        Returns
        -------
        list[str]
            Dataset names containing this donor.
        """
        return [name for name, info in self._datasets.items() if donor_id in info.donors]

    def get_stage_datasets(self, stage: str) -> list[str]:
        """Get datasets containing a specific stage.

        Parameters
        ----------
        stage : str
            Stage label.

        Returns
        -------
        list[str]
            Dataset names containing this stage.
        """
        return [name for name, info in self._datasets.items() if stage in info.stages]

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Get registry summary.

        Returns
        -------
        dict
            Summary statistics.
        """
        return {
            "n_datasets": len(self._datasets),
            "modalities": self.get_modalities(),
            "n_donors": len(self.get_all_donors()),
            "n_stages": len(self.get_all_stages()),
            "total_cells": sum(info.n_cells for info in self._datasets.values()),
            "total_spots": sum(info.n_spots for info in self._datasets.values()),
            "processed": len(self.list_datasets(processed=True)),
            "validated": len(self.list_datasets(validated=True)),
            "datasets": self.list_datasets(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DatasetRegistry(n_datasets={len(self._datasets)}, "
            f"modalities={self.get_modalities()})"
        )

    def __len__(self) -> int:
        """Number of registered datasets."""
        return len(self._datasets)

    def __contains__(self, name: str) -> bool:
        """Check if dataset is registered."""
        return name in self._datasets
