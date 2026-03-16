"""Tests for stagebridge.data.dataset_registry module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stagebridge.data.dataset_registry import (
    DatasetInfo,
    DatasetRegistry,
    ModalityInfo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path) -> DatasetRegistry:
    """Create a DatasetRegistry with persistence."""
    return DatasetRegistry(registry_dir=tmp_path / "registry")


@pytest.fixture
def memory_registry() -> DatasetRegistry:
    """Create an in-memory DatasetRegistry."""
    return DatasetRegistry(registry_dir=None)


@pytest.fixture
def populated_registry(tmp_path: Path) -> DatasetRegistry:
    """Create a registry with some datasets."""
    reg = DatasetRegistry(registry_dir=tmp_path / "registry")

    reg.register_dataset(
        name="luad_evo_snrna",
        modality="snRNA",
        n_donors=10,
        n_cells=50000,
        stages=["Normal", "AAH", "AIS", "MIA", "LUAD"],
        donors=[f"D{i}" for i in range(1, 11)],
    )

    reg.register_dataset(
        name="luad_evo_spatial",
        modality="spatial",
        n_donors=10,
        n_spots=20000,
        stages=["Normal", "AAH", "AIS", "MIA", "LUAD"],
        donors=[f"D{i}" for i in range(1, 11)],
    )

    reg.register_dataset(
        name="brainmets_snrna",
        modality="snRNA",
        n_donors=5,
        n_cells=25000,
        stages=["Primary", "Metastasis"],
        donors=[f"B{i}" for i in range(1, 6)],
    )

    return reg


# ---------------------------------------------------------------------------
# DatasetInfo tests
# ---------------------------------------------------------------------------


class TestDatasetInfo:
    """Tests for DatasetInfo."""

    def test_create_dataset_info(self) -> None:
        """Test creating DatasetInfo."""
        info = DatasetInfo(
            name="test_dataset",
            modality="snRNA",
            paths={"h5ad": "/path/to/data.h5ad"},
            n_donors=5,
            n_cells=10000,
        )

        assert info.name == "test_dataset"
        assert info.modality == "snRNA"
        assert info.n_donors == 5
        assert info.n_cells == 10000

    def test_dataset_info_to_dict(self) -> None:
        """Test DatasetInfo serialization."""
        info = DatasetInfo(
            name="test",
            modality="spatial",
            paths={},
            n_spots=5000,
        )

        d = info.to_dict()

        assert d["name"] == "test"
        assert d["modality"] == "spatial"
        assert d["n_spots"] == 5000

    def test_dataset_info_from_dict(self) -> None:
        """Test DatasetInfo deserialization."""
        d = {
            "name": "test",
            "modality": "snRNA",
            "paths": {},
            "n_cells": 1000,
            "stages": ["A", "B"],
        }

        info = DatasetInfo.from_dict(d)

        assert info.name == "test"
        assert info.n_cells == 1000
        assert info.stages == ["A", "B"]


# ---------------------------------------------------------------------------
# Registry basic operations tests
# ---------------------------------------------------------------------------


class TestRegistryBasicOperations:
    """Tests for basic registry operations."""

    def test_register_dataset(self, registry: DatasetRegistry) -> None:
        """Test registering a dataset."""
        info = registry.register_dataset(
            name="test_dataset",
            modality="snRNA",
            n_cells=1000,
        )

        assert info.name == "test_dataset"
        assert "test_dataset" in registry

    def test_register_duplicate_raises(self, registry: DatasetRegistry) -> None:
        """Test that registering duplicate raises error."""
        registry.register_dataset(name="test", modality="snRNA")

        with pytest.raises(ValueError, match="already registered"):
            registry.register_dataset(name="test", modality="snRNA")

    def test_register_duplicate_with_overwrite(self, registry: DatasetRegistry) -> None:
        """Test overwriting existing registration."""
        registry.register_dataset(name="test", modality="snRNA", n_cells=100)
        registry.register_dataset(name="test", modality="snRNA", n_cells=200, overwrite=True)

        info = registry.get_dataset("test")
        assert info.n_cells == 200

    def test_get_dataset(self, registry: DatasetRegistry) -> None:
        """Test getting a dataset."""
        registry.register_dataset(name="test", modality="snRNA")

        info = registry.get_dataset("test")

        assert info.name == "test"

    def test_get_nonexistent_raises(self, registry: DatasetRegistry) -> None:
        """Test getting nonexistent dataset raises error."""
        with pytest.raises(KeyError):
            registry.get_dataset("nonexistent")

    def test_has_dataset(self, registry: DatasetRegistry) -> None:
        """Test has_dataset method."""
        registry.register_dataset(name="test", modality="snRNA")

        assert registry.has_dataset("test") is True
        assert registry.has_dataset("other") is False

    def test_unregister_dataset(self, registry: DatasetRegistry) -> None:
        """Test unregistering a dataset."""
        registry.register_dataset(name="test", modality="snRNA")
        registry.unregister_dataset("test")

        assert "test" not in registry

    def test_unregister_nonexistent_raises(self, registry: DatasetRegistry) -> None:
        """Test unregistering nonexistent dataset raises error."""
        with pytest.raises(KeyError):
            registry.unregister_dataset("nonexistent")

    def test_list_datasets(self, populated_registry: DatasetRegistry) -> None:
        """Test listing datasets."""
        datasets = populated_registry.list_datasets()

        assert len(datasets) == 3
        assert "luad_evo_snrna" in datasets
        assert "luad_evo_spatial" in datasets
        assert "brainmets_snrna" in datasets

    def test_list_datasets_filter_modality(self, populated_registry: DatasetRegistry) -> None:
        """Test filtering datasets by modality."""
        snrna_datasets = populated_registry.list_datasets(modality="snRNA")
        spatial_datasets = populated_registry.list_datasets(modality="spatial")

        assert len(snrna_datasets) == 2
        assert len(spatial_datasets) == 1


# ---------------------------------------------------------------------------
# Registry update tests
# ---------------------------------------------------------------------------


class TestRegistryUpdate:
    """Tests for registry update operations."""

    def test_update_dataset(self, registry: DatasetRegistry) -> None:
        """Test updating dataset info."""
        registry.register_dataset(name="test", modality="snRNA")

        registry.update_dataset("test", processed=True)

        info = registry.get_dataset("test")
        assert info.processed is True

    def test_update_paths(self, registry: DatasetRegistry) -> None:
        """Test updating dataset paths."""
        registry.register_dataset(name="test", modality="snRNA", paths={})

        registry.update_dataset("test", paths={"h5ad": "/new/path.h5ad"})

        info = registry.get_dataset("test")
        assert info.paths["h5ad"] == "/new/path.h5ad"

    def test_update_metadata(self, registry: DatasetRegistry) -> None:
        """Test updating dataset metadata."""
        registry.register_dataset(name="test", modality="snRNA")

        registry.update_dataset("test", metadata={"key": "value"})

        info = registry.get_dataset("test")
        assert info.metadata["key"] == "value"

    def test_update_nonexistent_raises(self, registry: DatasetRegistry) -> None:
        """Test updating nonexistent dataset raises error."""
        with pytest.raises(KeyError):
            registry.update_dataset("nonexistent", processed=True)


# ---------------------------------------------------------------------------
# Registry aggregation tests
# ---------------------------------------------------------------------------


class TestRegistryAggregation:
    """Tests for registry aggregation operations."""

    def test_get_modality_info(self, populated_registry: DatasetRegistry) -> None:
        """Test getting modality info."""
        info = populated_registry.get_modality_info("snRNA")

        assert isinstance(info, ModalityInfo)
        assert info.modality == "snRNA"
        assert len(info.datasets) == 2
        assert info.total_cells == 75000  # 50000 + 25000

    def test_get_all_donors(self, populated_registry: DatasetRegistry) -> None:
        """Test getting all donors."""
        donors = populated_registry.get_all_donors()

        # Should have D1-D10 and B1-B5
        assert len(donors) == 15

    def test_get_all_stages(self, populated_registry: DatasetRegistry) -> None:
        """Test getting all stages."""
        stages = populated_registry.get_all_stages()

        # Should have LUAD stages + brainmets stages
        expected = ["AAH", "AIS", "LUAD", "MIA", "Metastasis", "Normal", "Primary"]
        assert sorted(stages) == expected

    def test_get_modalities(self, populated_registry: DatasetRegistry) -> None:
        """Test getting all modalities."""
        modalities = populated_registry.get_modalities()

        assert set(modalities) == {"snRNA", "spatial"}

    def test_get_donor_datasets(self, populated_registry: DatasetRegistry) -> None:
        """Test getting datasets for a donor."""
        datasets = populated_registry.get_donor_datasets("D1")

        assert "luad_evo_snrna" in datasets
        assert "luad_evo_spatial" in datasets
        assert "brainmets_snrna" not in datasets

    def test_get_stage_datasets(self, populated_registry: DatasetRegistry) -> None:
        """Test getting datasets for a stage."""
        datasets = populated_registry.get_stage_datasets("Normal")

        assert "luad_evo_snrna" in datasets
        assert "luad_evo_spatial" in datasets
        assert "brainmets_snrna" not in datasets


# ---------------------------------------------------------------------------
# Registry persistence tests
# ---------------------------------------------------------------------------


class TestRegistryPersistence:
    """Tests for registry persistence."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading registry."""
        registry_dir = tmp_path / "registry"

        # Create and populate registry
        reg1 = DatasetRegistry(registry_dir=registry_dir)
        reg1.register_dataset(name="test", modality="snRNA", n_cells=1000)

        # Create new registry from same directory
        reg2 = DatasetRegistry(registry_dir=registry_dir)

        assert "test" in reg2
        assert reg2.get_dataset("test").n_cells == 1000

    def test_registry_file_exists(self, tmp_path: Path) -> None:
        """Test that registry file is created."""
        registry_dir = tmp_path / "registry"

        reg = DatasetRegistry(registry_dir=registry_dir)
        reg.register_dataset(name="test", modality="snRNA")

        registry_path = registry_dir / "registry.json"
        assert registry_path.exists()

    def test_registry_file_valid_json(self, tmp_path: Path) -> None:
        """Test that registry file is valid JSON."""
        registry_dir = tmp_path / "registry"

        reg = DatasetRegistry(registry_dir=registry_dir)
        reg.register_dataset(name="test", modality="snRNA")

        registry_path = registry_dir / "registry.json"
        with registry_path.open("r") as f:
            data = json.load(f)

        assert "datasets" in data
        assert len(data["datasets"]) == 1


# ---------------------------------------------------------------------------
# Registry container operations tests
# ---------------------------------------------------------------------------


class TestRegistryContainerOps:
    """Tests for container-like operations."""

    def test_len(self, populated_registry: DatasetRegistry) -> None:
        """Test len() on registry."""
        assert len(populated_registry) == 3

    def test_contains(self, populated_registry: DatasetRegistry) -> None:
        """Test 'in' operator."""
        assert "luad_evo_snrna" in populated_registry
        assert "nonexistent" not in populated_registry

    def test_repr(self, populated_registry: DatasetRegistry) -> None:
        """Test string representation."""
        repr_str = repr(populated_registry)

        assert "DatasetRegistry" in repr_str
        assert "n_datasets=3" in repr_str


# ---------------------------------------------------------------------------
# Registry summary tests
# ---------------------------------------------------------------------------


class TestRegistrySummary:
    """Tests for registry summary."""

    def test_summary(self, populated_registry: DatasetRegistry) -> None:
        """Test summary method."""
        summary = populated_registry.summary()

        assert summary["n_datasets"] == 3
        assert "snRNA" in summary["modalities"]
        assert "spatial" in summary["modalities"]
        assert summary["total_cells"] == 75000
        assert summary["total_spots"] == 20000

    def test_empty_registry_summary(self, memory_registry: DatasetRegistry) -> None:
        """Test summary of empty registry."""
        summary = memory_registry.summary()

        assert summary["n_datasets"] == 0
        assert summary["total_cells"] == 0
