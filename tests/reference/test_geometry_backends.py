"""Tests for geometry backend implementations."""

from __future__ import annotations

import numpy as np
import pytest

from stagebridge.geometry import EuclideanBackend, GeometryBackend, get_geometry_backend


class TestEuclideanBackend:
    """Tests for EuclideanBackend."""

    def test_backend_name(self) -> None:
        """Backend name is correct."""
        backend = EuclideanBackend()
        assert backend.name == "euclidean"

    def test_distance_1d(self) -> None:
        """Distance computation for 1D arrays."""
        backend = EuclideanBackend()
        x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        y = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dist = backend.distance(x, y)
        assert np.isclose(dist, 1.0)

    def test_distance_2d(self) -> None:
        """Distance computation for 2D arrays (batch)."""
        backend = EuclideanBackend()
        x = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        y = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        dists = backend.distance(x, y)
        assert dists.shape == (2,)
        assert np.isclose(dists[0], 1.0)
        assert np.isclose(dists[1], 5.0)

    def test_midpoint(self) -> None:
        """Midpoint computation."""
        backend = EuclideanBackend()
        x = np.array([0.0, 0.0], dtype=np.float32)
        y = np.array([2.0, 4.0], dtype=np.float32)
        mid = backend.midpoint(x, y)
        assert np.allclose(mid, [1.0, 2.0])

    def test_midpoint_batch(self) -> None:
        """Midpoint computation for batch."""
        backend = EuclideanBackend()
        x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        y = np.array([[2.0, 4.0], [3.0, 3.0]], dtype=np.float32)
        mid = backend.midpoint(x, y)
        assert mid.shape == (2, 2)
        assert np.allclose(mid[0], [1.0, 2.0])
        assert np.allclose(mid[1], [2.0, 2.0])

    def test_interpolate_t0(self) -> None:
        """Interpolation at t=0 returns start."""
        backend = EuclideanBackend()
        x = np.array([0.0, 0.0], dtype=np.float32)
        y = np.array([1.0, 1.0], dtype=np.float32)
        result = backend.interpolate(x, y, 0.0)
        assert np.allclose(result, x)

    def test_interpolate_t1(self) -> None:
        """Interpolation at t=1 returns end."""
        backend = EuclideanBackend()
        x = np.array([0.0, 0.0], dtype=np.float32)
        y = np.array([1.0, 1.0], dtype=np.float32)
        result = backend.interpolate(x, y, 1.0)
        assert np.allclose(result, y)

    def test_interpolate_t05(self) -> None:
        """Interpolation at t=0.5 returns midpoint."""
        backend = EuclideanBackend()
        x = np.array([0.0, 0.0], dtype=np.float32)
        y = np.array([2.0, 4.0], dtype=np.float32)
        result = backend.interpolate(x, y, 0.5)
        expected = backend.midpoint(x, y)
        assert np.allclose(result, expected)

    def test_project_identity(self) -> None:
        """Project is identity for Euclidean backend."""
        backend = EuclideanBackend()
        x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        result = backend.project(x)
        assert np.allclose(result, x)
        assert result.dtype == np.float32

    def test_centroid_uniform(self) -> None:
        """Centroid with uniform weights is mean."""
        backend = EuclideanBackend()
        points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        centroid = backend.centroid(points)
        expected = np.mean(points, axis=0)
        assert np.allclose(centroid, expected)

    def test_centroid_weighted(self) -> None:
        """Centroid with weights."""
        backend = EuclideanBackend()
        points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        weights = np.array([1.0, 3.0], dtype=np.float32)
        centroid = backend.centroid(points, weights)
        # Weighted: (0*1 + 1*3) / 4 = 0.75
        assert np.isclose(centroid[0], 0.75)
        assert np.isclose(centroid[1], 0.0)

    def test_pairwise_distances_self(self) -> None:
        """Pairwise distances to self."""
        backend = EuclideanBackend()
        x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        dists = backend.pairwise_distances(x)
        assert dists.shape == (3, 3)
        # Diagonal should be zero
        assert np.allclose(np.diag(dists), 0.0)
        # Symmetric
        assert np.allclose(dists, dists.T)
        # d(0,1) = 1, d(0,2) = 1, d(1,2) = sqrt(2)
        assert np.isclose(dists[0, 1], 1.0)
        assert np.isclose(dists[0, 2], 1.0)
        assert np.isclose(dists[1, 2], np.sqrt(2))

    def test_pairwise_distances_different(self) -> None:
        """Pairwise distances between two sets."""
        backend = EuclideanBackend()
        x = np.array([[0.0, 0.0]], dtype=np.float32)
        y = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        dists = backend.pairwise_distances(x, y)
        assert dists.shape == (1, 2)
        assert np.isclose(dists[0, 0], 1.0)
        assert np.isclose(dists[0, 1], 1.0)


class TestGeometryBackendFactory:
    """Tests for get_geometry_backend factory."""

    def test_get_euclidean(self) -> None:
        """Get euclidean backend by name."""
        backend = get_geometry_backend("euclidean")
        assert isinstance(backend, EuclideanBackend)

    def test_get_euclidean_case_insensitive(self) -> None:
        """Backend name is case insensitive."""
        backend = get_geometry_backend("EUCLIDEAN")
        assert isinstance(backend, EuclideanBackend)

    def test_unknown_backend_raises(self) -> None:
        """Unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown geometry backend"):
            get_geometry_backend("hyperbolic")


class TestGeometryBackendProtocol:
    """Tests that EuclideanBackend satisfies GeometryBackend protocol."""

    def test_protocol_compliance(self) -> None:
        """EuclideanBackend satisfies GeometryBackend protocol."""
        backend = EuclideanBackend()
        assert isinstance(backend, GeometryBackend)

    def test_all_methods_exist(self) -> None:
        """All protocol methods exist on backend."""
        backend = EuclideanBackend()
        assert hasattr(backend, "name")
        assert hasattr(backend, "distance")
        assert hasattr(backend, "midpoint")
        assert hasattr(backend, "interpolate")
        assert hasattr(backend, "project")
        assert hasattr(backend, "centroid")
        assert hasattr(backend, "pairwise_distances")
