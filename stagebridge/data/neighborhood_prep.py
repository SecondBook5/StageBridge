"""
Spatial neighborhood preparation for StageBridge.

This module handles:
- Spatial coordinate extraction and validation
- Neighborhood table construction
- K-nearest neighbor and radius-based neighborhood methods
- Preparation for downstream niche modeling (without building final models)

Usage:
    from stagebridge.data.neighborhood_prep import (
        extract_spatial_coords,
        build_neighborhood_table,
        validate_spatial_coordinates,
    )

    coords = extract_spatial_coords(adata)
    neighborhoods = build_neighborhood_table(adata, method="knn", k=15)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpatialCoordinates:
    """Container for spatial coordinates."""

    coords: np.ndarray  # (n_spots, 2) or (n_spots, 3)
    coord_names: tuple[str, ...] = ("x", "y")
    units: str = "pixels"
    scale_factors: dict[str, float] = field(default_factory=dict)
    n_spots: int = 0
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute derived attributes."""
        if self.n_spots == 0:
            self.n_spots = self.coords.shape[0]

        # Compute bounds
        for i, name in enumerate(self.coord_names):
            if i < self.coords.shape[1]:
                self.bounds[name] = (
                    float(self.coords[:, i].min()),
                    float(self.coords[:, i].max()),
                )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame(self.coords, columns=list(self.coord_names))
        return df


@dataclass
class NeighborhoodResult:
    """Result of neighborhood construction."""

    method: str  # knn, radius, delaunay
    n_spots: int
    n_edges: int
    neighborhood_table: pd.DataFrame
    mean_neighbors: float
    median_neighbors: float
    min_neighbors: int
    max_neighbors: int
    params: dict[str, Any] = field(default_factory=dict)
    built_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without table)."""
        return {
            "method": self.method,
            "n_spots": self.n_spots,
            "n_edges": self.n_edges,
            "mean_neighbors": self.mean_neighbors,
            "median_neighbors": self.median_neighbors,
            "min_neighbors": self.min_neighbors,
            "max_neighbors": self.max_neighbors,
            "params": self.params,
            "built_at": self.built_at,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------


def extract_spatial_coords(
    adata: Any,  # AnnData
    *,
    coord_key: str = "spatial",
    library_id: str | None = None,
) -> SpatialCoordinates:
    """Extract spatial coordinates from AnnData.

    Looks for coordinates in obsm['spatial'] or adata.uns['spatial'].

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData object.
    coord_key : str
        Key in obsm for coordinates.
    library_id : str, optional
        Library ID for Visium data (to get scale factors).

    Returns
    -------
    SpatialCoordinates
        Extracted coordinates.
    """
    # Try obsm first
    if coord_key in adata.obsm:
        coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
        log.info("Extracted coordinates from obsm['%s']: shape %s", coord_key, coords.shape)
    elif "X_spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["X_spatial"], dtype=np.float32)
        log.info("Extracted coordinates from obsm['X_spatial']: shape %s", coords.shape)
    else:
        raise KeyError(
            f"No spatial coordinates found. Expected obsm['{coord_key}'] or obsm['X_spatial']. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )

    # Ensure 2D or 3D
    if coords.ndim == 1:
        raise ValueError(f"Coordinates must be 2D array, got shape {coords.shape}")

    if coords.shape[1] == 2:
        coord_names = ("x", "y")
    elif coords.shape[1] == 3:
        coord_names = ("x", "y", "z")
    else:
        raise ValueError(f"Coordinates must have 2 or 3 columns, got {coords.shape[1]}")

    # Get scale factors from Visium data
    scale_factors = {}
    if "spatial" in adata.uns:
        spatial_uns = adata.uns["spatial"]
        if library_id is None and len(spatial_uns) == 1:
            library_id = list(spatial_uns.keys())[0]

        if library_id is not None and library_id in spatial_uns:
            lib_data = spatial_uns[library_id]
            if "scalefactors" in lib_data:
                scale_factors = dict(lib_data["scalefactors"])
                log.info("Extracted scale factors: %s", list(scale_factors.keys()))

    return SpatialCoordinates(
        coords=coords,
        coord_names=coord_names,
        scale_factors=scale_factors,
    )


def validate_spatial_coordinates(
    coords: SpatialCoordinates | np.ndarray,
    *,
    check_finite: bool = True,
    check_range: bool = True,
    max_coordinate: float = 1e6,
) -> tuple[bool, list[str]]:
    """Validate spatial coordinates.

    Checks:
    - No NaN or infinite values
    - Reasonable coordinate range
    - Sufficient variance (not all same point)

    Parameters
    ----------
    coords : SpatialCoordinates or ndarray
        Coordinates to validate.
    check_finite : bool
        Whether to check for NaN/inf.
    check_range : bool
        Whether to check coordinate range.
    max_coordinate : float
        Maximum allowed coordinate value.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of issues)
    """
    if isinstance(coords, SpatialCoordinates):
        arr = coords.coords
    else:
        arr = coords

    issues = []

    # Check shape
    if arr.ndim != 2:
        issues.append(f"Coordinates must be 2D array, got shape {arr.shape}")
        return False, issues

    if arr.shape[0] == 0:
        issues.append("No coordinates (empty array)")
        return False, issues

    # Check for NaN/inf
    if check_finite:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0:
            issues.append(f"Found {n_nan} NaN values in coordinates")
        if n_inf > 0:
            issues.append(f"Found {n_inf} infinite values in coordinates")

    # Check range
    if check_range:
        for i in range(arr.shape[1]):
            col_min = float(np.nanmin(arr[:, i]))
            col_max = float(np.nanmax(arr[:, i]))

            if abs(col_min) > max_coordinate or abs(col_max) > max_coordinate:
                issues.append(
                    f"Coordinate column {i} has extreme values [{col_min:.1f}, {col_max:.1f}]"
                )

    # Check variance
    variance = np.nanvar(arr, axis=0)
    if np.all(variance < 1e-10):
        issues.append("All coordinates are identical (zero variance)")
    elif np.any(variance < 1e-10):
        low_var_cols = np.where(variance < 1e-10)[0]
        issues.append(f"Columns {low_var_cols.tolist()} have near-zero variance")

    is_valid = len(issues) == 0

    if is_valid:
        log.info(
            "Coordinate validation passed: %d spots, bounds x=[%.1f, %.1f], y=[%.1f, %.1f]",
            arr.shape[0],
            float(arr[:, 0].min()),
            float(arr[:, 0].max()),
            float(arr[:, 1].min()),
            float(arr[:, 1].max()),
        )
    else:
        log.warning("Coordinate validation failed with %d issues", len(issues))

    return is_valid, issues


# ---------------------------------------------------------------------------
# Neighborhood construction
# ---------------------------------------------------------------------------


def build_neighborhood_table(
    adata: Any,  # AnnData
    method: Literal["knn", "radius", "delaunay"] = "knn",
    *,
    k_neighbors: int = 15,
    radius: float | None = None,
    coord_key: str = "spatial",
    include_self: bool = False,
) -> NeighborhoodResult:
    """Build spatial neighborhood table.

    Creates a table of (spot_i, spot_j, distance) edges representing
    spatial neighbors.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData object.
    method : str
        Neighborhood method:
        - "knn": K-nearest neighbors
        - "radius": All neighbors within radius
        - "delaunay": Delaunay triangulation
    k_neighbors : int
        Number of neighbors for KNN method.
    radius : float, optional
        Radius for radius-based method.
    coord_key : str
        Key in obsm for coordinates.
    include_self : bool
        Whether to include self-loops.

    Returns
    -------
    NeighborhoodResult
        Neighborhood construction result.
    """
    # Extract coordinates
    spatial_coords = extract_spatial_coords(adata, coord_key=coord_key)
    coords = spatial_coords.coords

    n_spots = coords.shape[0]
    log.info("Building %s neighborhoods for %d spots...", method, n_spots)

    if method == "knn":
        edges = _build_knn_neighborhoods(coords, k_neighbors, include_self)
        params = {"k_neighbors": k_neighbors}
    elif method == "radius":
        if radius is None:
            raise ValueError("radius must be specified for radius-based method")
        edges = _build_radius_neighborhoods(coords, radius, include_self)
        params = {"radius": radius}
    elif method == "delaunay":
        edges = _build_delaunay_neighborhoods(coords, include_self)
        params = {}
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create neighborhood table
    table = pd.DataFrame(edges, columns=["spot_i", "spot_j", "distance"])
    table["spot_i"] = table["spot_i"].astype(int)
    table["spot_j"] = table["spot_j"].astype(int)
    table["distance"] = table["distance"].astype(np.float32)

    # Compute statistics
    neighbors_per_spot = table.groupby("spot_i").size()
    mean_neighbors = float(neighbors_per_spot.mean())
    median_neighbors = float(neighbors_per_spot.median())
    min_neighbors = int(neighbors_per_spot.min()) if len(neighbors_per_spot) > 0 else 0
    max_neighbors = int(neighbors_per_spot.max()) if len(neighbors_per_spot) > 0 else 0

    result = NeighborhoodResult(
        method=method,
        n_spots=n_spots,
        n_edges=len(table),
        neighborhood_table=table,
        mean_neighbors=mean_neighbors,
        median_neighbors=median_neighbors,
        min_neighbors=min_neighbors,
        max_neighbors=max_neighbors,
        params=params,
    )

    log.info(
        "Built %d edges, mean neighbors: %.1f, range: [%d, %d]",
        result.n_edges,
        mean_neighbors,
        min_neighbors,
        max_neighbors,
    )

    return result


def _build_knn_neighborhoods(
    coords: np.ndarray,
    k: int,
    include_self: bool,
) -> list[tuple[int, int, float]]:
    """Build KNN neighborhood edges."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise ImportError("sklearn is required for KNN neighborhood construction") from e

    n_spots = coords.shape[0]
    k_actual = min(k + 1, n_spots)  # +1 because query includes self

    knn = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree")
    knn.fit(coords)
    distances, indices = knn.kneighbors(coords)

    edges = []
    for i in range(n_spots):
        for j_idx in range(k_actual):
            j = indices[i, j_idx]
            d = distances[i, j_idx]

            if i == j and not include_self:
                continue

            edges.append((i, j, d))

    return edges


def _build_radius_neighborhoods(
    coords: np.ndarray,
    radius: float,
    include_self: bool,
) -> list[tuple[int, int, float]]:
    """Build radius-based neighborhood edges."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise ImportError("sklearn is required for radius neighborhood construction") from e

    knn = NearestNeighbors(radius=radius, algorithm="ball_tree")
    knn.fit(coords)
    distances, indices = knn.radius_neighbors(coords)

    edges = []
    for i in range(len(indices)):
        for j_idx, j in enumerate(indices[i]):
            d = distances[i][j_idx]

            if i == j and not include_self:
                continue

            edges.append((i, j, d))

    return edges


def _build_delaunay_neighborhoods(
    coords: np.ndarray,
    include_self: bool,
) -> list[tuple[int, int, float]]:
    """Build Delaunay triangulation neighborhood edges."""
    try:
        from scipy.spatial import Delaunay
    except ImportError as e:
        raise ImportError("scipy is required for Delaunay neighborhood construction") from e

    if coords.shape[1] != 2:
        raise ValueError("Delaunay triangulation requires 2D coordinates")

    tri = Delaunay(coords)
    edges_set = set()

    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                if a > b:
                    a, b = b, a
                edges_set.add((a, b))

    edges = []
    for a, b in edges_set:
        d = float(np.linalg.norm(coords[a] - coords[b]))
        edges.append((a, b, d))
        edges.append((b, a, d))  # Add reverse edge

    if include_self:
        for i in range(len(coords)):
            edges.append((i, i, 0.0))

    return edges


# ---------------------------------------------------------------------------
# Neighborhood utilities
# ---------------------------------------------------------------------------


def compute_neighborhood_stats(
    adata: Any,  # AnnData
    neighborhood_result: NeighborhoodResult,
    *,
    feature_key: str | None = None,
) -> pd.DataFrame:
    """Compute per-spot neighborhood statistics.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData object.
    neighborhood_result : NeighborhoodResult
        Neighborhood construction result.
    feature_key : str, optional
        Key in obsm for features to aggregate.

    Returns
    -------
    pd.DataFrame
        Per-spot statistics: n_neighbors, mean_distance, etc.
    """
    table = neighborhood_result.neighborhood_table

    # Group by source spot
    grouped = (
        table.groupby("spot_i")
        .agg(
            n_neighbors=("spot_j", "count"),
            mean_distance=("distance", "mean"),
            min_distance=("distance", "min"),
            max_distance=("distance", "max"),
            std_distance=("distance", "std"),
        )
        .reset_index()
    )

    # Rename and fill missing spots
    grouped = grouped.rename(columns={"spot_i": "spot_idx"})

    # Add spots with no neighbors
    all_spots = pd.DataFrame({"spot_idx": range(adata.n_obs)})
    stats = all_spots.merge(grouped, on="spot_idx", how="left")
    stats = stats.fillna(0)

    log.info(
        "Computed neighborhood stats: mean neighbors=%.1f, mean distance=%.1f",
        float(stats["n_neighbors"].mean()),
        float(stats["mean_distance"].mean()),
    )

    return stats


def save_neighborhood_table(
    result: NeighborhoodResult,
    output_dir: str | Path,
    *,
    prefix: str = "",
    format: Literal["parquet", "csv"] = "parquet",
) -> Path:
    """Save neighborhood table to file.

    Parameters
    ----------
    result : NeighborhoodResult
        Neighborhood result.
    output_dir : Path
        Output directory.
    prefix : str
        File name prefix.
    format : str
        Output format.

    Returns
    -------
    Path
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_str = f"{prefix}_" if prefix else ""
    filename = f"{prefix_str}neighborhood_{result.method}"

    if format == "parquet":
        path = output_dir / f"{filename}.parquet"
        result.neighborhood_table.to_parquet(path, index=False)
    else:
        path = output_dir / f"{filename}.csv"
        result.neighborhood_table.to_csv(path, index=False)

    log.info("Saved neighborhood table to %s", path)
    return path


def aggregate_neighborhood_features(
    adata: Any,  # AnnData
    neighborhood_result: NeighborhoodResult,
    feature_key: str,
    *,
    aggregation: Literal["mean", "sum", "max", "median"] = "mean",
) -> np.ndarray:
    """Aggregate features across neighborhoods.

    For each spot, aggregate the features of its neighbors.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData object.
    neighborhood_result : NeighborhoodResult
        Neighborhood result.
    feature_key : str
        Key in obsm for features.
    aggregation : str
        Aggregation method.

    Returns
    -------
    ndarray
        Aggregated features (n_spots, n_features).
    """
    if feature_key not in adata.obsm:
        raise KeyError(f"Feature key '{feature_key}' not found in obsm")

    features = np.asarray(adata.obsm[feature_key], dtype=np.float32)
    n_spots, n_features = features.shape

    # Initialize output
    aggregated = np.zeros((n_spots, n_features), dtype=np.float32)

    # Group neighborhoods
    table = neighborhood_result.neighborhood_table
    for spot_i, group in table.groupby("spot_i"):
        neighbor_indices = group["spot_j"].values.astype(int)
        neighbor_features = features[neighbor_indices]

        if aggregation == "mean":
            agg_features = np.mean(neighbor_features, axis=0)
        elif aggregation == "sum":
            agg_features = np.sum(neighbor_features, axis=0)
        elif aggregation == "max":
            agg_features = np.max(neighbor_features, axis=0)
        elif aggregation == "median":
            agg_features = np.median(neighbor_features, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        aggregated[int(spot_i)] = agg_features

    log.info(
        "Aggregated %s features across neighborhoods (%s)",
        feature_key,
        aggregation,
    )

    return aggregated
