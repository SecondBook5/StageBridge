"""Geometry backend implementations for reference embedding operations.

This module defines the GeometryBackend protocol and concrete implementations.
The design follows "Euclidean-first, geometry-ready": starting with a stable
Euclidean backend while structuring code so spherical/hyperbolic extensions
require no rewrites.

Supported backends:
- EuclideanBackend: Default flat geometry (L2 distances, linear interpolation)
- Future: SphericalBackend, HyperbolicBackend
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GeometryBackend(Protocol):
    """Abstract protocol for geometry operations on latent embeddings.

    All methods work with batched inputs where the first dimension is batch size.
    Points are represented as float32 arrays of shape (n_points, n_dims).

    This protocol enables future extension to non-Euclidean geometries
    (spherical, hyperbolic) without changing downstream code.
    """

    @property
    def name(self) -> str:
        """Return the backend name for logging and serialization."""
        ...

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between corresponding points.

        Parameters
        ----------
        x : np.ndarray
            Points of shape (n, d) or (d,)
        y : np.ndarray
            Points of shape (n, d) or (d,), same shape as x

        Returns
        -------
        np.ndarray
            Distances of shape (n,) or scalar
        """
        ...

    def midpoint(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute midpoints between corresponding pairs of points.

        Parameters
        ----------
        x : np.ndarray
            Points of shape (n, d) or (d,)
        y : np.ndarray
            Points of shape (n, d) or (d,), same shape as x

        Returns
        -------
        np.ndarray
            Midpoints of shape (n, d) or (d,)
        """
        ...

    def interpolate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Interpolate along geodesics between points.

        Parameters
        ----------
        x : np.ndarray
            Start points of shape (n, d) or (d,)
        y : np.ndarray
            End points of shape (n, d) or (d,)
        t : float
            Interpolation parameter in [0, 1]. t=0 returns x, t=1 returns y.

        Returns
        -------
        np.ndarray
            Interpolated points of same shape as x
        """
        ...

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project points onto the manifold (identity for Euclidean).

        For non-Euclidean backends, this ensures points lie on the manifold.

        Parameters
        ----------
        x : np.ndarray
            Points of shape (n, d) or (d,)

        Returns
        -------
        np.ndarray
            Projected points of same shape
        """
        ...

    def centroid(self, points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Compute (weighted) centroid of a set of points.

        Parameters
        ----------
        points : np.ndarray
            Points of shape (n, d)
        weights : np.ndarray, optional
            Weights of shape (n,). If None, uniform weights are used.

        Returns
        -------
        np.ndarray
            Centroid of shape (d,)
        """
        ...

    def pairwise_distances(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Compute full pairwise distance matrix.

        Parameters
        ----------
        x : np.ndarray
            Points of shape (n, d)
        y : np.ndarray, optional
            Points of shape (m, d). If None, compute self-distances.

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n, m) or (n, n)
        """
        ...


class EuclideanBackend:
    """Default Euclidean geometry backend using L2 distances.

    This is the primary backend for StageBridge V1. It provides stable,
    well-understood operations on flat latent spaces.

    All operations are vectorized for performance on large cell populations.
    """

    @property
    def name(self) -> str:
        """Return backend name."""
        return "euclidean"

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute L2 distances between corresponding points.

        Parameters
        ----------
        x : np.ndarray
            Points of shape (n, d) or (d,)
        y : np.ndarray
            Points of shape (n, d) or (d,)

        Returns
        -------
        np.ndarray
            L2 distances
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        diff = x - y
        if diff.ndim == 1:
            return np.sqrt(np.sum(diff**2))
        return np.sqrt(np.sum(diff**2, axis=-1))

    def midpoint(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute midpoints (simple average in Euclidean space).

        Parameters
        ----------
        x : np.ndarray
            Points of shape (n, d) or (d,)
        y : np.ndarray
            Points of shape (n, d) or (d,)

        Returns
        -------
        np.ndarray
            Midpoints
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        return (x + y) / 2.0

    def interpolate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation between points.

        Parameters
        ----------
        x : np.ndarray
            Start points of shape (n, d) or (d,)
        y : np.ndarray
            End points of shape (n, d) or (d,)
        t : float
            Interpolation parameter in [0, 1]

        Returns
        -------
        np.ndarray
            Interpolated points
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        t = float(t)
        return x + t * (y - x)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project points (identity in Euclidean space).

        Parameters
        ----------
        x : np.ndarray
            Points of shape (n, d) or (d,)

        Returns
        -------
        np.ndarray
            Same points, ensured to be float32
        """
        return np.asarray(x, dtype=np.float32)

    def centroid(self, points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Compute weighted centroid (weighted mean in Euclidean space).

        Parameters
        ----------
        points : np.ndarray
            Points of shape (n, d)
        weights : np.ndarray, optional
            Weights of shape (n,)

        Returns
        -------
        np.ndarray
            Centroid of shape (d,)
        """
        points = np.asarray(points, dtype=np.float32)
        if weights is None:
            return np.mean(points, axis=0)

        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)
        return np.sum(points * weights[:, np.newaxis], axis=0)

    def pairwise_distances(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Compute pairwise L2 distance matrix.

        Parameters
        ----------
        x : np.ndarray
            Points of shape (n, d)
        y : np.ndarray, optional
            Points of shape (m, d). If None, compute self-distances.

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n, m) or (n, n)
        """
        x = np.asarray(x, dtype=np.float32)
        if y is None:
            y = x
        else:
            y = np.asarray(y, dtype=np.float32)

        # Efficient computation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        x_sq = np.sum(x**2, axis=1, keepdims=True)
        y_sq = np.sum(y**2, axis=1, keepdims=True)
        xy = x @ y.T
        dist_sq = x_sq + y_sq.T - 2 * xy
        # Numerical safety: clip small negatives
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.sqrt(dist_sq)


# Future placeholders for non-Euclidean backends
# class SphericalBackend(GeometryBackend):
#     """Spherical geometry on the unit sphere (great-circle distances)."""
#     pass

# class HyperbolicBackend(GeometryBackend):
#     """Hyperbolic geometry in the Poincare ball model."""
#     pass


_BACKENDS: dict[str, type] = {
    "euclidean": EuclideanBackend,
}


def get_geometry_backend(name: str = "euclidean") -> GeometryBackend:
    """Get a geometry backend by name.

    Parameters
    ----------
    name : str
        Backend name. Currently only "euclidean" is supported.
        Future: "spherical", "hyperbolic"

    Returns
    -------
    GeometryBackend
        Instantiated backend

    Raises
    ------
    ValueError
        If backend name is not recognized
    """
    name_lower = name.lower()
    if name_lower not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(f"Unknown geometry backend '{name}'. Available: {available}")
    return _BACKENDS[name_lower]()
