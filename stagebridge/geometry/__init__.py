"""Geometry abstractions for reference embedding operations.

This module provides a geometry backend pattern that starts with stable
Euclidean operations while being structured for future spherical/hyperbolic
extensions without requiring rewrites.

Example usage:
    >>> from stagebridge.geometry import EuclideanBackend
    >>> backend = EuclideanBackend()
    >>> dist = backend.distance(x, y)
    >>> mid = backend.midpoint(x, y)
"""

from __future__ import annotations

from stagebridge.geometry.backends import (
    EuclideanBackend,
    GeometryBackend,
    get_geometry_backend,
)

__all__ = [
    "EuclideanBackend",
    "GeometryBackend",
    "get_geometry_backend",
]
