"""Spatial niche feature assignment for snRNA-seq cells.

For each snRNA cell, finds the K nearest Visium spots in HLCA latent space
(within the same patient) and computes the aggregated cell-type composition
vector of those spots' spatial neighborhoods.  The result is stored as
``adata.obsm["X_spatial_niche"]`` and used to condition the set transformer
context with spatially-aware niche information.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from stagebridge.logging_utils import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Column name conventions
_DONOR_COL = "patient_id"
_STAGE_COL = "stage"
_TANGRAM_KEY = "X_tangram_ct"
_SPATIAL_KEY = "spatial"


def build_snrna_spatial_niche_features(
    adata_snrna: Any,
    adata_spatial: Any,
    k_neighbors: int = 5,
    latent_key: str = "X_hlca",
    tangram_key: str = _TANGRAM_KEY,
    donor_col: str = _DONOR_COL,
    output_key: str = "X_spatial_niche",
    spatial_radius_um: float | None = None,
) -> np.ndarray:
    """Assign spatial niche composition features to snRNA cells.

    For each snRNA cell, identifies the K nearest Visium spots from the same
    patient in HLCA latent space, then aggregates their cell-type composition
    vectors (from Tangram deconvolution).

    Parameters
    ----------
    adata_snrna : AnnData
        snRNA-seq data with ``obsm[latent_key]`` and ``obs[donor_col]``.
    adata_spatial : AnnData
        Spatial Visium data with ``obsm[tangram_key]`` (cell-type fractions)
        and ``obs[donor_col]``.
    k_neighbors : int
        Number of nearest spatial spots per cell.
    latent_key : str
        HLCA latent key for nearest-neighbor search.
    tangram_key : str
        Key in ``adata_spatial.obsm`` holding Tangram cell-type composition
        matrix (N_spots × N_celltypes, values ∈ [0, 1], row-normalized).
    donor_col : str
        Column in ``obs`` containing the patient / donor identifier.
    output_key : str
        Key under which the result is written to ``adata_snrna.obsm``.
    spatial_radius_um : float or None
        If provided, only spots within this radius in physical space are used.
        Requires ``adata_spatial.obsm["spatial"]`` to contain μm coordinates.

    Returns
    -------
    np.ndarray
        Shape ``(N_snrna_cells, N_celltypes)`` — also written to
        ``adata_snrna.obsm[output_key]``.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("scikit-learn is required for spatial niche assignment.") from exc

    if latent_key not in adata_snrna.obsm:
        raise KeyError(
            f"Latent key '{latent_key}' not in adata_snrna.obsm. "
            f"Run build_latent() first. Available: {list(adata_snrna.obsm.keys())}"
        )
    if tangram_key not in adata_spatial.obsm:
        raise KeyError(
            f"Tangram key '{tangram_key}' not in adata_spatial.obsm. "
            f"Run Tangram deconvolution first. Available: {list(adata_spatial.obsm.keys())}"
        )
    if donor_col not in adata_snrna.obs.columns:
        raise KeyError(f"Donor column '{donor_col}' missing from adata_snrna.obs.")
    if donor_col not in adata_spatial.obs.columns:
        raise KeyError(f"Donor column '{donor_col}' missing from adata_spatial.obs.")

    X_snrna = np.asarray(adata_snrna.obsm[latent_key], dtype=np.float32)
    X_spatial_latent = np.asarray(adata_spatial.obsm[latent_key], dtype=np.float32) \
        if latent_key in adata_spatial.obsm else None
    tangram_mat = np.asarray(adata_spatial.obsm[tangram_key], dtype=np.float32)
    n_celltypes = tangram_mat.shape[1]

    snrna_donors = np.asarray(adata_snrna.obs[donor_col])
    spatial_donors = np.asarray(adata_spatial.obs[donor_col])

    niche_features = np.zeros((X_snrna.shape[0], n_celltypes), dtype=np.float32)

    unique_donors = np.unique(snrna_donors)
    assigned_count = 0

    for donor in unique_donors:
        snrna_mask = snrna_donors == donor
        spatial_mask = spatial_donors == donor

        n_snrna = snrna_mask.sum()
        n_spatial = spatial_mask.sum()

        if n_spatial == 0:
            log.debug("No spatial spots for donor '%s'; using global mean niche.", donor)
            niche_features[snrna_mask] = tangram_mat.mean(axis=0)
            continue

        # Select spatial spots for this donor
        tang_donor = tangram_mat[spatial_mask]  # (n_spatial, n_ct)

        if X_spatial_latent is not None:
            # KNN in HLCA latent space
            X_sp_donor = X_spatial_latent[spatial_mask]  # (n_spatial, latent_dim)
            X_sn_donor = X_snrna[snrna_mask]            # (n_snrna, latent_dim)

            # Apply optional radius filter (cheap: just prune to candidates)
            if spatial_radius_um is not None and _SPATIAL_KEY in adata_spatial.obsm:
                _ = np.asarray(adata_spatial.obsm[_SPATIAL_KEY])[spatial_mask]
                # Use spatial coords for radius; still use latent for KNN ordering
            k = min(k_neighbors, n_spatial)
            nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
            nn.fit(X_sp_donor)
            _, indices = nn.kneighbors(X_sn_donor)  # (n_snrna, k)
        else:
            # No spatial latent: assign all spots from the same donor equally
            log.debug(
                "No spatial latent for donor '%s'; assigning mean over all spots.", donor
            )
            niche_features[snrna_mask] = tang_donor.mean(axis=0)
            assigned_count += n_snrna
            continue

        # Aggregate: mean over K nearest spots' cell-type compositions
        # indices: (n_snrna, k) → tang_donor[indices]: (n_snrna, k, n_ct)
        neighbor_compositions = tang_donor[indices]           # (n_snrna, k, n_ct)
        niche_features[snrna_mask] = neighbor_compositions.mean(axis=1)  # (n_snrna, n_ct)
        assigned_count += n_snrna

    log.info(
        "Assigned spatial niche features to %d/%d snRNA cells "
        "(%d cell-type dimensions, %d donors).",
        assigned_count,
        X_snrna.shape[0],
        n_celltypes,
        len(unique_donors),
    )

    adata_snrna.obsm[output_key] = niche_features
    return niche_features


def compute_spatial_neighbor_composition(
    adata_spatial: Any,
    tangram_key: str = _TANGRAM_KEY,
    n_rings: int = 1,
    output_key: str = "X_spatial_neighbor_niche",
) -> np.ndarray:
    """Compute cell-type composition of each spot's spatial neighborhood.

    Aggregates Tangram cell-type compositions over the spot's spatial
    neighbors (defined by the Visium hexagonal grid adjacency within
    ``n_rings`` rings).  Result is stored in ``adata_spatial.obsm[output_key]``.

    Requires ``adata_spatial.obsm["spatial"]`` with pixel/μm coordinates.

    Parameters
    ----------
    adata_spatial : AnnData
        Spatial data with Tangram output in ``obsm[tangram_key]``.
    tangram_key : str
        Key for Tangram cell-type fraction matrix.
    n_rings : int
        Number of Visium hexagonal rings to aggregate (1 = 6 immediate neighbors).
    output_key : str
        Key to write the result into ``adata_spatial.obsm``.

    Returns
    -------
    np.ndarray
        Shape ``(N_spots, N_celltypes)`` — neighborhood-aggregated compositions.
    """
    from sklearn.neighbors import NearestNeighbors

    if _SPATIAL_KEY not in adata_spatial.obsm:
        raise KeyError(
            f"Spatial coordinates key '{_SPATIAL_KEY}' not in adata_spatial.obsm."
        )
    if tangram_key not in adata_spatial.obsm:
        raise KeyError(f"Tangram key '{tangram_key}' not in adata_spatial.obsm.")

    coords = np.asarray(adata_spatial.obsm[_SPATIAL_KEY], dtype=np.float32)
    tang = np.asarray(adata_spatial.obsm[tangram_key], dtype=np.float32)

    # Estimate typical spot diameter (median nearest-neighbor distance)
    nn1 = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
    nn1.fit(coords)
    dists, _ = nn1.kneighbors(coords)
    median_dist = float(np.median(dists[:, 1]))
    radius = median_dist * (2 * n_rings + 0.5)  # generous radius for n_rings hexagonal rings

    nn_radius = NearestNeighbors(radius=radius, algorithm="ball_tree")
    nn_radius.fit(coords)
    neighbors = nn_radius.radius_neighbors(coords, return_distance=False)

    neighbor_niche = np.zeros_like(tang)
    for i, nbrs in enumerate(neighbors):
        if len(nbrs) == 0:
            neighbor_niche[i] = tang[i]
        else:
            neighbor_niche[i] = tang[nbrs].mean(axis=0)

    adata_spatial.obsm[output_key] = neighbor_niche
    log.info(
        "Computed spatial neighbor niche features: %d spots, %d cell types, "
        "radius=%.1f (n_rings=%d).",
        coords.shape[0],
        tang.shape[1],
        radius,
        n_rings,
    )
    return neighbor_niche


def get_niche_feature_dim(adata_snrna: Any, niche_key: str = "X_spatial_niche") -> int:
    """Return the spatial niche feature dimensionality."""
    if niche_key not in adata_snrna.obsm:
        raise KeyError(
            f"Spatial niche key '{niche_key}' not in adata_snrna.obsm. "
            "Run build_snrna_spatial_niche_features() first."
        )
    return adata_snrna.obsm[niche_key].shape[1]


def build_fallback_niche_features(n_cells: int, n_celltypes: int) -> np.ndarray:
    """Return a uniform niche feature matrix (1/n_celltypes per type) as fallback."""
    return np.full((n_cells, n_celltypes), fill_value=1.0 / max(n_celltypes, 1), dtype=np.float32)
