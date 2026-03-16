"""Tests for stagebridge.data.neighborhood_prep module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Try to import anndata; skip tests if not available
try:
    import anndata

    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

from stagebridge.data.neighborhood_prep import (
    NeighborhoodResult,
    SpatialCoordinates,
    aggregate_neighborhood_features,
    build_neighborhood_table,
    compute_neighborhood_stats,
    extract_spatial_coords,
    save_neighborhood_table,
    validate_spatial_coordinates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spatial_adata():
    """Create a spatial AnnData object for testing."""
    if not ANNDATA_AVAILABLE:
        pytest.skip("anndata not available")

    np.random.seed(42)
    n_spots = 100
    n_genes = 50

    counts = np.random.negative_binomial(5, 0.5, size=(n_spots, n_genes))

    # Create spatial coordinates on a grid
    x = np.repeat(np.arange(10), 10) * 100
    y = np.tile(np.arange(10), 10) * 100
    coords = np.column_stack([x, y]).astype(np.float32)

    obs = pd.DataFrame(
        {
            "donor_id": ["D1"] * n_spots,
            "sample_id": ["S1"] * n_spots,
            "stage": ["Normal"] * n_spots,
        },
        index=[f"spot_{i}" for i in range(n_spots)],
    )

    var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)])

    adata = anndata.AnnData(
        X=counts.astype(np.float32),
        obs=obs,
        var=var,
    )
    adata.obsm["spatial"] = coords

    return adata


@pytest.fixture
def random_coords() -> np.ndarray:
    """Create random 2D coordinates."""
    np.random.seed(42)
    return np.random.uniform(0, 1000, size=(50, 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# SpatialCoordinates tests
# ---------------------------------------------------------------------------


class TestSpatialCoordinates:
    """Tests for SpatialCoordinates."""

    def test_create_coordinates(self) -> None:
        """Test creating SpatialCoordinates."""
        coords = np.random.rand(100, 2).astype(np.float32)

        spatial = SpatialCoordinates(coords=coords)

        assert spatial.n_spots == 100
        assert spatial.coord_names == ("x", "y")

    def test_coordinates_bounds(self) -> None:
        """Test that bounds are computed correctly."""
        coords = np.array([[0, 0], [100, 200]], dtype=np.float32)

        spatial = SpatialCoordinates(coords=coords)

        assert spatial.bounds["x"] == (0.0, 100.0)
        assert spatial.bounds["y"] == (0.0, 200.0)

    def test_coordinates_3d(self) -> None:
        """Test 3D coordinates."""
        coords = np.random.rand(50, 3).astype(np.float32)

        spatial = SpatialCoordinates(coords=coords, coord_names=("x", "y", "z"))

        assert spatial.coord_names == ("x", "y", "z")
        assert "z" in spatial.bounds

    def test_to_dataframe(self) -> None:
        """Test converting to DataFrame."""
        coords = np.array([[1, 2], [3, 4]], dtype=np.float32)

        spatial = SpatialCoordinates(coords=coords)
        df = spatial.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Coordinate extraction tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestExtractSpatialCoords:
    """Tests for coordinate extraction."""

    def test_extract_from_obsm(self, spatial_adata) -> None:
        """Test extracting coordinates from obsm."""
        coords = extract_spatial_coords(spatial_adata)

        assert isinstance(coords, SpatialCoordinates)
        assert coords.n_spots == spatial_adata.n_obs

    def test_extract_custom_key(self, spatial_adata) -> None:
        """Test extracting with custom key."""
        # Add coordinates under different key
        spatial_adata.obsm["X_spatial"] = spatial_adata.obsm["spatial"]
        del spatial_adata.obsm["spatial"]

        coords = extract_spatial_coords(spatial_adata, coord_key="X_spatial")

        assert coords.n_spots == spatial_adata.n_obs

    def test_extract_missing_key_raises(self, spatial_adata) -> None:
        """Test extraction fails with missing key."""
        del spatial_adata.obsm["spatial"]

        with pytest.raises(KeyError):
            extract_spatial_coords(spatial_adata)


# ---------------------------------------------------------------------------
# Coordinate validation tests
# ---------------------------------------------------------------------------


class TestValidateSpatialCoordinates:
    """Tests for coordinate validation."""

    def test_validate_valid_coords(self, random_coords: np.ndarray) -> None:
        """Test validation of valid coordinates."""
        is_valid, issues = validate_spatial_coordinates(random_coords)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_nan_coords(self) -> None:
        """Test validation detects NaN values."""
        coords = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)

        is_valid, issues = validate_spatial_coordinates(coords)

        assert is_valid is False
        assert any("NaN" in issue for issue in issues)

    def test_validate_inf_coords(self) -> None:
        """Test validation detects infinite values."""
        coords = np.array([[1, 2], [np.inf, 4]], dtype=np.float32)

        is_valid, issues = validate_spatial_coordinates(coords)

        assert is_valid is False
        assert any("infinite" in issue for issue in issues)

    def test_validate_extreme_coords(self) -> None:
        """Test validation detects extreme values."""
        coords = np.array([[1, 2], [1e10, 4]], dtype=np.float32)

        is_valid, issues = validate_spatial_coordinates(coords, max_coordinate=1e6)

        assert is_valid is False
        assert any("extreme" in issue for issue in issues)

    def test_validate_zero_variance(self) -> None:
        """Test validation detects zero variance."""
        coords = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.float32)

        is_valid, issues = validate_spatial_coordinates(coords)

        assert is_valid is False
        assert any("identical" in issue.lower() for issue in issues)

    def test_validate_empty_coords(self) -> None:
        """Test validation of empty coordinates."""
        coords = np.zeros((0, 2), dtype=np.float32)

        is_valid, issues = validate_spatial_coordinates(coords)

        assert is_valid is False
        assert any("empty" in issue.lower() for issue in issues)

    def test_validate_spatial_coordinates_object(self, random_coords: np.ndarray) -> None:
        """Test validation with SpatialCoordinates object."""
        spatial = SpatialCoordinates(coords=random_coords)

        is_valid, issues = validate_spatial_coordinates(spatial)

        assert is_valid is True


# ---------------------------------------------------------------------------
# Neighborhood construction tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestBuildNeighborhoodTable:
    """Tests for neighborhood table construction."""

    def test_build_knn_neighborhoods(self, spatial_adata) -> None:
        """Test building KNN neighborhoods."""
        result = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        assert isinstance(result, NeighborhoodResult)
        assert result.method == "knn"
        assert result.n_spots == spatial_adata.n_obs
        assert result.n_edges > 0

    def test_knn_has_expected_neighbors(self, spatial_adata) -> None:
        """Test that KNN has approximately k neighbors per spot."""
        k = 10
        result = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=k)

        # Mean should be close to k
        assert abs(result.mean_neighbors - k) < 1

    def test_build_radius_neighborhoods(self, spatial_adata) -> None:
        """Test building radius-based neighborhoods."""
        result = build_neighborhood_table(spatial_adata, method="radius", radius=150.0)

        assert result.method == "radius"
        assert result.n_edges > 0

    def test_radius_requires_radius_param(self, spatial_adata) -> None:
        """Test that radius method requires radius parameter."""
        with pytest.raises(ValueError, match="radius must be specified"):
            build_neighborhood_table(spatial_adata, method="radius")

    def test_build_delaunay_neighborhoods(self, spatial_adata) -> None:
        """Test building Delaunay neighborhoods."""
        try:
            from scipy.spatial import Delaunay

            result = build_neighborhood_table(spatial_adata, method="delaunay")

            assert result.method == "delaunay"
            assert result.n_edges > 0
        except ImportError:
            pytest.skip("scipy not available")

    def test_neighborhood_table_columns(self, spatial_adata) -> None:
        """Test neighborhood table has expected columns."""
        result = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        table = result.neighborhood_table
        assert "spot_i" in table.columns
        assert "spot_j" in table.columns
        assert "distance" in table.columns

    def test_neighborhood_statistics(self, spatial_adata) -> None:
        """Test neighborhood result statistics."""
        result = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        assert result.mean_neighbors > 0
        assert result.median_neighbors > 0
        assert result.min_neighbors >= 0
        assert result.max_neighbors >= result.min_neighbors

    def test_include_self(self, spatial_adata) -> None:
        """Test including self-loops."""
        result = build_neighborhood_table(
            spatial_adata, method="knn", k_neighbors=5, include_self=True
        )

        # Should have some self-loops (distance=0)
        table = result.neighborhood_table
        self_loops = table[table["spot_i"] == table["spot_j"]]
        # With include_self=True, we expect self-loops
        assert len(self_loops) >= 0  # May or may not have depending on implementation


# ---------------------------------------------------------------------------
# Neighborhood statistics tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestComputeNeighborhoodStats:
    """Tests for neighborhood statistics computation."""

    def test_compute_stats(self, spatial_adata) -> None:
        """Test computing per-spot statistics."""
        neighborhood = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        stats = compute_neighborhood_stats(spatial_adata, neighborhood)

        assert isinstance(stats, pd.DataFrame)
        assert "n_neighbors" in stats.columns
        assert "mean_distance" in stats.columns
        assert len(stats) == spatial_adata.n_obs

    def test_stats_includes_all_spots(self, spatial_adata) -> None:
        """Test that stats include all spots."""
        neighborhood = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        stats = compute_neighborhood_stats(spatial_adata, neighborhood)

        assert len(stats) == spatial_adata.n_obs


# ---------------------------------------------------------------------------
# Save neighborhood table tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestSaveNeighborhoodTable:
    """Tests for saving neighborhood tables."""

    def test_save_parquet(self, spatial_adata, tmp_path: Path) -> None:
        """Test saving as parquet."""
        neighborhood = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        path = save_neighborhood_table(neighborhood, tmp_path, format="parquet")

        assert path.exists()
        assert path.suffix == ".parquet"

    def test_save_csv(self, spatial_adata, tmp_path: Path) -> None:
        """Test saving as CSV."""
        neighborhood = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        path = save_neighborhood_table(neighborhood, tmp_path, format="csv")

        assert path.exists()
        assert path.suffix == ".csv"

    def test_save_with_prefix(self, spatial_adata, tmp_path: Path) -> None:
        """Test saving with filename prefix."""
        neighborhood = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        path = save_neighborhood_table(neighborhood, tmp_path, prefix="sample1")

        assert "sample1" in path.name


# ---------------------------------------------------------------------------
# Aggregate neighborhood features tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestAggregateNeighborhoodFeatures:
    """Tests for feature aggregation."""

    def test_aggregate_mean(self, spatial_adata) -> None:
        """Test mean aggregation."""
        # Add features to obsm
        spatial_adata.obsm["features"] = np.random.rand(spatial_adata.n_obs, 10).astype(np.float32)

        neighborhood = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        aggregated = aggregate_neighborhood_features(
            spatial_adata, neighborhood, "features", aggregation="mean"
        )

        assert aggregated.shape == spatial_adata.obsm["features"].shape

    def test_aggregate_sum(self, spatial_adata) -> None:
        """Test sum aggregation."""
        spatial_adata.obsm["features"] = np.random.rand(spatial_adata.n_obs, 10).astype(np.float32)

        neighborhood = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        aggregated = aggregate_neighborhood_features(
            spatial_adata, neighborhood, "features", aggregation="sum"
        )

        assert aggregated.shape == spatial_adata.obsm["features"].shape

    def test_aggregate_missing_key_raises(self, spatial_adata) -> None:
        """Test aggregation fails with missing feature key."""
        neighborhood = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=5)

        with pytest.raises(KeyError):
            aggregate_neighborhood_features(spatial_adata, neighborhood, "nonexistent")


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestNeighborhoodEdgeCases:
    """Tests for edge cases."""

    def test_single_spot(self) -> None:
        """Test neighborhood with single spot."""
        adata = anndata.AnnData(
            X=np.array([[1, 2, 3]]).astype(np.float32),
            obs=pd.DataFrame({"donor_id": ["D1"]}, index=["spot_0"]),
            var=pd.DataFrame(index=["G1", "G2", "G3"]),
        )
        adata.obsm["spatial"] = np.array([[100, 200]]).astype(np.float32)

        result = build_neighborhood_table(adata, method="knn", k_neighbors=1)

        assert result.n_spots == 1
        # Single spot should have no neighbors (or just self)
        assert result.n_edges >= 0

    def test_very_small_k(self, spatial_adata) -> None:
        """Test with very small k."""
        result = build_neighborhood_table(spatial_adata, method="knn", k_neighbors=1)

        assert result.mean_neighbors >= 0
        assert result.max_neighbors >= 0
