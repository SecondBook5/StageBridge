"""Tests for spatial_backends adapters that wrap spatial_mapping implementations."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from stagebridge.spatial_backends.adapters import (
    AdapterConfig,
    TangramAdapter,
    DestVIAdapter,
    TACCOAdapter,
    get_adapter,
    _convert_to_backend_result,
    ADAPTERS,
)
from stagebridge.spatial_backends.base import BackendMappingResult


# ---------------------------------------------------------------------------
# AdapterConfig tests
# ---------------------------------------------------------------------------


class TestAdapterConfig:
    """Tests for AdapterConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AdapterConfig()

        assert config.execution_mode == "force_rebuild"
        assert config.stages is None
        assert config.donors is None
        assert config.max_spots_per_stage is None
        assert config.seed == 42
        assert config.extra is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AdapterConfig(
            execution_mode="load_precomputed",
            stages=["Normal", "AAH"],
            donors=["D1", "D2"],
            max_spots_per_stage=100,
            seed=123,
            extra={"custom_param": True},
        )

        assert config.execution_mode == "load_precomputed"
        assert config.stages == ["Normal", "AAH"]
        assert config.donors == ["D1", "D2"]
        assert config.max_spots_per_stage == 100
        assert config.seed == 123
        assert config.extra == {"custom_param": True}


# ---------------------------------------------------------------------------
# Convert result tests
# ---------------------------------------------------------------------------


class TestConvertToBackendResult:
    """Tests for _convert_to_backend_result function."""

    @pytest.fixture
    def mock_mapping_result(self):
        """Create a mock SpatialMappingResult."""
        from stagebridge.spatial_mapping.base import SpatialMappingResult

        # Create mock compositions
        n_spots = 10
        n_celltypes = 3
        compositions = np.random.dirichlet(np.ones(n_celltypes), size=n_spots)

        obs = pd.DataFrame(
            {"stage": ["Normal"] * n_spots, "donor_id": ["D1"] * n_spots},
            index=[f"spot_{i}" for i in range(n_spots)],
        )

        return SpatialMappingResult(
            compositions=compositions,
            obs=obs,
            feature_names=["CellType_A", "CellType_B", "CellType_C"],
            method="tangram",
            status="completed",
            provider_version="1.0.0",
            execution_mode="force_rebuild",
            qc={"mean_entropy": 0.5, "n_spots": n_spots},
            provenance={"source": "test"},
            notes="Test result",
        )

    def test_basic_conversion(self, mock_mapping_result) -> None:
        """Test basic result conversion."""
        backend_result = _convert_to_backend_result(mock_mapping_result, runtime_seconds=10.5)

        assert isinstance(backend_result, BackendMappingResult)
        assert isinstance(backend_result.cell_type_proportions, pd.DataFrame)
        assert len(backend_result.cell_type_proportions) == 10
        assert backend_result.cell_type_proportions.shape[1] == 3

    def test_confidence_computed(self, mock_mapping_result) -> None:
        """Test that confidence is computed from entropy."""
        backend_result = _convert_to_backend_result(mock_mapping_result)

        assert isinstance(backend_result.confidence, pd.Series)
        assert len(backend_result.confidence) == 10
        # Confidence should be 1 - entropy, so between 0 and 1
        assert (backend_result.confidence >= 0).all()
        assert (backend_result.confidence <= 1).all()

    def test_upstream_metrics_extracted(self, mock_mapping_result) -> None:
        """Test that upstream metrics are extracted from QC."""
        backend_result = _convert_to_backend_result(mock_mapping_result)

        assert "mean_entropy" in backend_result.upstream_metrics
        assert "n_spots" in backend_result.upstream_metrics
        assert backend_result.upstream_metrics["n_spots"] == 10

    def test_metadata_preserved(self, mock_mapping_result) -> None:
        """Test that metadata is preserved."""
        backend_result = _convert_to_backend_result(mock_mapping_result, runtime_seconds=10.5)

        assert backend_result.metadata["backend"] == "tangram"
        assert backend_result.metadata["status"] == "completed"
        assert backend_result.metadata["runtime_seconds"] == 10.5
        assert "provenance" in backend_result.metadata

    def test_empty_result(self) -> None:
        """Test conversion of empty result."""
        from stagebridge.spatial_mapping.base import SpatialMappingResult

        empty_result = SpatialMappingResult(
            compositions=None,
            obs=None,
            feature_names=[],
            method="tangram",
            status="failed",
        )

        backend_result = _convert_to_backend_result(empty_result)

        assert backend_result.cell_type_proportions.empty
        assert len(backend_result.confidence) == 0


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestTangramAdapter:
    """Tests for TangramAdapter."""

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        adapter = TangramAdapter()

        assert adapter.adapter_config.execution_mode == "force_rebuild"
        assert adapter.adapter_config.seed == 42

    def test_initialization_with_config(self) -> None:
        """Test adapter initialization with custom config."""
        config = AdapterConfig(
            execution_mode="load_precomputed",
            stages=["Normal", "AAH"],
        )
        adapter = TangramAdapter(config=config)

        assert adapter.adapter_config.execution_mode == "load_precomputed"
        assert adapter.adapter_config.stages == ["Normal", "AAH"]

    def test_build_cfg(self) -> None:
        """Test _build_cfg generates proper config."""
        config = AdapterConfig(
            execution_mode="rebuild_cached",
            extra={"marker_genes": ["TP63", "KRT5"]},
        )
        adapter = TangramAdapter(config=config)
        adapter.config = {"base_config": True}

        cfg = adapter._build_cfg()

        assert cfg["base_config"] is True
        assert cfg["spatial_mapping"]["method"] == "tangram"
        assert cfg["spatial_mapping"]["execution_mode"] == "rebuild_cached"
        assert cfg["spatial_mapping"]["marker_genes"] == ["TP63", "KRT5"]


class TestDestVIAdapter:
    """Tests for DestVIAdapter."""

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        adapter = DestVIAdapter()

        assert adapter.adapter_config.execution_mode == "force_rebuild"

    def test_build_cfg(self) -> None:
        """Test _build_cfg generates proper config."""
        adapter = DestVIAdapter()
        adapter.config = {}

        cfg = adapter._build_cfg()

        assert cfg["spatial_mapping"]["method"] == "destvi"


class TestTACCOAdapter:
    """Tests for TACCOAdapter."""

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        adapter = TACCOAdapter()

        assert adapter.adapter_config.execution_mode == "force_rebuild"

    def test_build_cfg(self) -> None:
        """Test _build_cfg generates proper config."""
        adapter = TACCOAdapter()
        adapter.config = {}

        cfg = adapter._build_cfg()

        assert cfg["spatial_mapping"]["method"] == "tacco"


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestGetAdapter:
    """Tests for get_adapter factory function."""

    def test_get_tangram_adapter(self) -> None:
        """Test getting Tangram adapter."""
        adapter = get_adapter("tangram")

        assert isinstance(adapter, TangramAdapter)

    def test_get_destvi_adapter(self) -> None:
        """Test getting DestVI adapter."""
        adapter = get_adapter("destvi")

        assert isinstance(adapter, DestVIAdapter)

    def test_get_tacco_adapter(self) -> None:
        """Test getting TACCO adapter."""
        adapter = get_adapter("tacco")

        assert isinstance(adapter, TACCOAdapter)

    def test_case_insensitive(self) -> None:
        """Test that method names are case-insensitive."""
        assert isinstance(get_adapter("Tangram"), TangramAdapter)
        assert isinstance(get_adapter("DESTVI"), DestVIAdapter)
        assert isinstance(get_adapter("TaCCo"), TACCOAdapter)

    def test_unknown_method(self) -> None:
        """Test error on unknown method."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_adapter("unknown_method")

    def test_with_config(self) -> None:
        """Test getting adapter with config."""
        config = AdapterConfig(stages=["Normal"])
        adapter = get_adapter("tangram", config=config)

        assert adapter.adapter_config.stages == ["Normal"]

    def test_adapter_registry(self) -> None:
        """Test ADAPTERS registry is complete."""
        assert "tangram" in ADAPTERS
        assert "destvi" in ADAPTERS
        assert "tacco" in ADAPTERS
        assert len(ADAPTERS) == 3


# ---------------------------------------------------------------------------
# Integration with __init__.py exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Test that adapters are properly exported from module."""

    def test_import_from_package(self) -> None:
        """Test importing adapters from stagebridge.spatial_backends."""
        from stagebridge.spatial_backends import (
            AdapterConfig,
            TangramAdapter,
            DestVIAdapter,
            TACCOAdapter,
            get_adapter,
        )

        assert AdapterConfig is not None
        assert TangramAdapter is not None
        assert DestVIAdapter is not None
        assert TACCOAdapter is not None
        assert get_adapter is not None

    def test_get_backend_with_adapter_flag(self) -> None:
        """Test get_backend with use_adapter=True."""
        from stagebridge.spatial_backends import get_backend, TangramAdapter, TangramBackend

        # Direct backend (default)
        direct = get_backend("tangram")
        assert direct is TangramBackend

        # Adapter backend
        adapter = get_backend("tangram", use_adapter=True)
        assert adapter is TangramAdapter
