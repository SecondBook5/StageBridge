"""Query-to-reference mapping for dual-reference embedding construction.

FALLBACK MODULE: This provides k-NN based mapping when scANVI models are unavailable.
For production use with cell type prediction, use:
- hlca_mapper.py: HLCA model-based mapping with cell types
- luca_mapper.py: LuCA model-based mapping with cell types

This module provides simple k-NN and PCA projection methods that work
without trained models but do NOT provide cell type predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger
from stagebridge.geometry import EuclideanBackend, GeometryBackend
from stagebridge.reference.schema import MappingResult, ReferenceNeighborhood

log = get_logger(__name__)


def _validate_no_donor_leakage(
    query_donors: np.ndarray,
    held_out_donors: set[str] | None,
) -> None:
    """Validate that held-out donors are not in query data.

    Parameters
    ----------
    query_donors : np.ndarray
        Donor IDs from query data
    held_out_donors : set[str], optional
        Set of held-out donor IDs from split manifest

    Raises
    ------
    ValueError
        If held-out donors appear in query data
    """
    if held_out_donors is None:
        return

    query_donor_set = set(query_donors.astype(str))
    overlap = query_donor_set & held_out_donors

    if overlap:
        raise ValueError(
            f"Donor leakage detected: held-out donors {overlap} appear in query data. "
            "This violates the split manifest and would contaminate evaluation."
        )


def map_to_hlca(
    query: Any,
    hlca_reference: Any,
    *,
    method: Literal["knn_projection", "scvi_query", "pca_projection"] = "knn_projection",
    latent_key: str = "X_scanvi_emb",
    k_neighbors: int = 50,
    held_out_donors: set[str] | None = None,
    geometry: GeometryBackend | None = None,
    metadata_cols: dict[str, str] | None = None,
) -> MappingResult:
    """Map query cells to HLCA reference space.

    Parameters
    ----------
    query : AnnData
        Query data with expression matrix
    hlca_reference : AnnData or LoadedReference
        HLCA reference atlas
    method : str
        Mapping method:
        - "knn_projection": Project via weighted k-NN in gene space
        - "scvi_query": Use scVI/scANVI query mapping (requires trained model)
        - "pca_projection": Project via PCA trained on reference
    latent_key : str
        Key in reference.obsm containing latent embeddings
    k_neighbors : int
        Number of neighbors for k-NN methods
    held_out_donors : set[str], optional
        Donor IDs to exclude (for split validation)
    geometry : GeometryBackend, optional
        Geometry backend for distance computations
    metadata_cols : dict, optional
        Mapping of standard names to query obs column names

    Returns
    -------
    MappingResult
        Mapping result with embeddings and metadata
    """
    if geometry is None:
        geometry = EuclideanBackend()

    # Handle LoadedReference wrapper
    ref_adata = hlca_reference.adata if hasattr(hlca_reference, "adata") else hlca_reference

    # Extract metadata columns
    metadata_cols = metadata_cols or {}
    cell_id_col = metadata_cols.get("cell_id", None)
    donor_col = metadata_cols.get("donor_id", "donor_id")
    sample_col = metadata_cols.get("sample_id", "sample_id")
    stage_col = metadata_cols.get("stage_id", "stage")

    # Get cell IDs
    if cell_id_col and cell_id_col in query.obs.columns:
        cell_ids = query.obs[cell_id_col].astype(str).to_numpy()
    else:
        cell_ids = query.obs.index.astype(str).to_numpy()

    # Get donor IDs
    if donor_col in query.obs.columns:
        donor_ids = query.obs[donor_col].astype(str).to_numpy()
    else:
        donor_ids = np.full(query.n_obs, "unknown_donor", dtype=object)

    # Check for donor leakage
    _validate_no_donor_leakage(donor_ids, held_out_donors)

    # Get sample IDs
    if sample_col in query.obs.columns:
        sample_ids = query.obs[sample_col].astype(str).to_numpy()
    else:
        sample_ids = np.full(query.n_obs, "unknown_sample", dtype=object)

    # Get stage IDs
    if stage_col in query.obs.columns:
        stage_ids = query.obs[stage_col].astype(str).to_numpy()
    else:
        stage_ids = np.full(query.n_obs, "unknown_stage", dtype=object)

    # Get reference latent
    if latent_key not in ref_adata.obsm:
        raise KeyError(
            f"Reference missing latent key '{latent_key}'. "
            f"Available: {list(ref_adata.obsm.keys())}"
        )
    ref_latent = np.asarray(ref_adata.obsm[latent_key], dtype=np.float32)

    if method == "knn_projection":
        embeddings, neighbor_distances = _map_knn_projection(
            query=query,
            ref_adata=ref_adata,
            ref_latent=ref_latent,
            k_neighbors=k_neighbors,
            geometry=geometry,
        )
    elif method == "pca_projection":
        embeddings, neighbor_distances = _map_pca_projection(
            query=query,
            ref_adata=ref_adata,
            ref_latent=ref_latent,
            k_neighbors=k_neighbors,
            geometry=geometry,
        )
    elif method == "scvi_query":
        embeddings, neighbor_distances = _map_scvi_query(
            query=query,
            ref_adata=ref_adata,
            ref_latent=ref_latent,
            k_neighbors=k_neighbors,
        )
    else:
        raise ValueError(f"Unknown mapping method: {method}")

    return MappingResult(
        embeddings=embeddings,
        latent_dim=embeddings.shape[1],
        cell_ids=cell_ids,
        donor_ids=donor_ids,
        sample_ids=sample_ids,
        stage_ids=stage_ids,
        neighbor_distances=neighbor_distances,
        reference_name="HLCA",
        reference_latent_key=latent_key,
        n_reference_cells=ref_adata.n_obs,
        mapping_method=method,
        mapping_params={"k_neighbors": k_neighbors},
    )


def map_to_luca(
    query: Any,
    luca_reference: Any,
    *,
    method: Literal["knn_projection", "scvi_query", "pca_projection"] = "knn_projection",
    latent_key: str = "X_scVI",
    k_neighbors: int = 50,
    held_out_donors: set[str] | None = None,
    geometry: GeometryBackend | None = None,
    metadata_cols: dict[str, str] | None = None,
) -> MappingResult:
    """Map query cells to LuCa reference space.

    Parameters
    ----------
    query : AnnData
        Query data with expression matrix
    luca_reference : AnnData or LoadedReference
        LuCa reference atlas
    method : str
        Mapping method (same options as map_to_hlca)
    latent_key : str
        Key in reference.obsm containing latent embeddings
    k_neighbors : int
        Number of neighbors for k-NN methods
    held_out_donors : set[str], optional
        Donor IDs to exclude (for split validation)
    geometry : GeometryBackend, optional
        Geometry backend for distance computations
    metadata_cols : dict, optional
        Mapping of standard names to query obs column names

    Returns
    -------
    MappingResult
        Mapping result with embeddings and metadata
    """
    if geometry is None:
        geometry = EuclideanBackend()

    # Handle LoadedReference wrapper
    ref_adata = luca_reference.adata if hasattr(luca_reference, "adata") else luca_reference

    # Extract metadata columns
    metadata_cols = metadata_cols or {}
    cell_id_col = metadata_cols.get("cell_id", None)
    donor_col = metadata_cols.get("donor_id", "donor_id")
    sample_col = metadata_cols.get("sample_id", "sample_id")
    stage_col = metadata_cols.get("stage_id", "stage")

    # Get cell IDs
    if cell_id_col and cell_id_col in query.obs.columns:
        cell_ids = query.obs[cell_id_col].astype(str).to_numpy()
    else:
        cell_ids = query.obs.index.astype(str).to_numpy()

    # Get donor IDs
    if donor_col in query.obs.columns:
        donor_ids = query.obs[donor_col].astype(str).to_numpy()
    else:
        donor_ids = np.full(query.n_obs, "unknown_donor", dtype=object)

    # Check for donor leakage
    _validate_no_donor_leakage(donor_ids, held_out_donors)

    # Get sample IDs
    if sample_col in query.obs.columns:
        sample_ids = query.obs[sample_col].astype(str).to_numpy()
    else:
        sample_ids = np.full(query.n_obs, "unknown_sample", dtype=object)

    # Get stage IDs
    if stage_col in query.obs.columns:
        stage_ids = query.obs[stage_col].astype(str).to_numpy()
    else:
        stage_ids = np.full(query.n_obs, "unknown_stage", dtype=object)

    # Get reference latent
    if latent_key not in ref_adata.obsm:
        raise KeyError(
            f"Reference missing latent key '{latent_key}'. "
            f"Available: {list(ref_adata.obsm.keys())}"
        )
    ref_latent = np.asarray(ref_adata.obsm[latent_key], dtype=np.float32)

    if method == "knn_projection":
        embeddings, neighbor_distances = _map_knn_projection(
            query=query,
            ref_adata=ref_adata,
            ref_latent=ref_latent,
            k_neighbors=k_neighbors,
            geometry=geometry,
        )
    elif method == "pca_projection":
        embeddings, neighbor_distances = _map_pca_projection(
            query=query,
            ref_adata=ref_adata,
            ref_latent=ref_latent,
            k_neighbors=k_neighbors,
            geometry=geometry,
        )
    elif method == "scvi_query":
        embeddings, neighbor_distances = _map_scvi_query(
            query=query,
            ref_adata=ref_adata,
            ref_latent=ref_latent,
            k_neighbors=k_neighbors,
        )
    else:
        raise ValueError(f"Unknown mapping method: {method}")

    return MappingResult(
        embeddings=embeddings,
        latent_dim=embeddings.shape[1],
        cell_ids=cell_ids,
        donor_ids=donor_ids,
        sample_ids=sample_ids,
        stage_ids=stage_ids,
        neighbor_distances=neighbor_distances,
        reference_name="LuCa",
        reference_latent_key=latent_key,
        n_reference_cells=ref_adata.n_obs,
        mapping_method=method,
        mapping_params={"k_neighbors": k_neighbors},
    )


def _map_knn_projection(
    query: Any,
    ref_adata: Any,
    ref_latent: np.ndarray,
    k_neighbors: int,
    geometry: GeometryBackend,
) -> tuple[np.ndarray, np.ndarray]:
    """Map query cells via weighted k-NN projection in gene space.

    This is a simple but robust method that:
    1. Finds k nearest reference cells in gene expression space
    2. Computes weighted average of their latent positions

    Handles ENSG vs symbol mismatches automatically using feature_name column.
    """
    from sklearn.neighbors import NearestNeighbors
    import scipy.sparse as sp

    # Get gene expression matrices
    X_query = query.X
    if sp.issparse(X_query):
        X_query = X_query.toarray()
    X_query = np.asarray(X_query, dtype=np.float32)

    X_ref = ref_adata.X
    if sp.issparse(X_ref):
        X_ref = X_ref.toarray()
    X_ref = np.asarray(X_ref, dtype=np.float32)

    # Find common genes - handle ENSG vs symbol mismatch
    query_genes = list(query.var_names.astype(str))
    query_gene_set = set(query_genes)
    ref_var_names = list(ref_adata.var_names.astype(str))

    # Check if reference uses ENSG IDs but has feature_name column
    first_ref_gene = ref_var_names[0] if ref_var_names else ""
    if first_ref_gene.startswith("ENSG") and "feature_name" in ref_adata.var.columns:
        # Build mapping from symbol to reference index
        ref_symbols = ref_adata.var["feature_name"].astype(str).tolist()
        ref_symbol_to_idx = {sym: idx for idx, sym in enumerate(ref_symbols)}
        ref_gene_set = set(ref_symbols)
        log.info("Using feature_name for gene matching (ENSG -> symbol)")
    else:
        ref_symbol_to_idx = {g: idx for idx, g in enumerate(ref_var_names)}
        ref_gene_set = set(ref_var_names)

    common_genes = sorted(query_gene_set & ref_gene_set)
    if len(common_genes) < 100:
        log.warning(
            "Only %d common genes between query and reference. Mapping quality may be poor.",
            len(common_genes),
        )

    # Subset to common genes
    query_idx = [i for i, g in enumerate(query_genes) if g in ref_gene_set]
    ref_idx = [ref_symbol_to_idx[query_genes[i]] for i in query_idx]

    X_query_common = X_query[:, query_idx]
    X_ref_common = X_ref[:, ref_idx]

    # Normalize for distance computation
    X_query_norm = X_query_common / (np.linalg.norm(X_query_common, axis=1, keepdims=True) + 1e-8)
    X_ref_norm = X_ref_common / (np.linalg.norm(X_ref_common, axis=1, keepdims=True) + 1e-8)

    # Find k nearest neighbors
    k = min(k_neighbors, X_ref_norm.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_ref_norm)
    distances, indices = nn.kneighbors(X_query_norm)

    # Compute weighted average of reference latent positions
    # Weight by inverse distance (softmax)
    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Weighted average of latent positions
    n_query = X_query.shape[0]
    latent_dim = ref_latent.shape[1]
    embeddings = np.zeros((n_query, latent_dim), dtype=np.float32)

    for i in range(n_query):
        neighbor_latents = ref_latent[indices[i]]
        embeddings[i] = np.sum(weights[i, :, np.newaxis] * neighbor_latents, axis=0)

    # Mean neighbor distance as quality metric
    mean_distances = distances.mean(axis=1)

    return embeddings, mean_distances


def _map_pca_projection(
    query: Any,
    ref_adata: Any,
    ref_latent: np.ndarray,
    k_neighbors: int,
    geometry: GeometryBackend,
) -> tuple[np.ndarray, np.ndarray]:
    """Map query cells via PCA projection trained on reference.

    1. Fit PCA on reference gene expression
    2. Project query into same PCA space
    3. Scale to match reference latent statistics

    Handles ENSG vs symbol mismatches automatically using feature_name column.
    """
    from sklearn.decomposition import TruncatedSVD
    import scipy.sparse as sp

    # Get gene expression matrices
    X_query = query.X
    if sp.issparse(X_query):
        X_query = X_query.toarray()
    X_query = np.asarray(X_query, dtype=np.float32)

    X_ref = ref_adata.X
    if sp.issparse(X_ref):
        X_ref = X_ref.toarray()
    X_ref = np.asarray(X_ref, dtype=np.float32)

    # Find common genes - handle ENSG vs symbol mismatch
    query_genes = list(query.var_names.astype(str))
    query_gene_set = set(query_genes)
    ref_var_names = list(ref_adata.var_names.astype(str))

    # Check if reference uses ENSG IDs but has feature_name column
    first_ref_gene = ref_var_names[0] if ref_var_names else ""
    if first_ref_gene.startswith("ENSG") and "feature_name" in ref_adata.var.columns:
        ref_symbols = ref_adata.var["feature_name"].astype(str).tolist()
        ref_symbol_to_idx = {sym: idx for idx, sym in enumerate(ref_symbols)}
        ref_gene_set = set(ref_symbols)
        log.info("Using feature_name for gene matching (ENSG -> symbol)")
    else:
        ref_symbol_to_idx = {g: idx for idx, g in enumerate(ref_var_names)}
        ref_gene_set = set(ref_var_names)

    common_genes = [g for g in query_genes if g in ref_gene_set]
    if len(common_genes) < 100:
        log.warning(
            "Only %d common genes. PCA projection may be unreliable.",
            len(common_genes),
        )

    # Subset to common genes
    query_idx = [query_genes.index(g) for g in common_genes]
    ref_idx = [ref_symbol_to_idx[g] for g in common_genes]

    X_query_common = X_query[:, query_idx]
    X_ref_common = X_ref[:, ref_idx]

    # Fit PCA on reference
    latent_dim = ref_latent.shape[1]
    n_components = min(latent_dim, X_ref_common.shape[1] - 1, X_ref_common.shape[0] - 1)

    pca = TruncatedSVD(n_components=n_components, random_state=42)
    ref_pca = pca.fit_transform(X_ref_common)

    # Project query
    query_pca = pca.transform(X_query_common)

    # Scale to match reference latent statistics
    ref_mu = ref_latent.mean(axis=0)
    ref_std = ref_latent.std(axis=0) + 1e-6

    pca_mu = ref_pca.mean(axis=0)
    pca_std = ref_pca.std(axis=0) + 1e-6

    # Z-score normalize query PCA, then rescale to reference latent
    query_z = (query_pca - pca_mu) / pca_std
    embeddings = query_z * ref_std[:n_components] + ref_mu[:n_components]

    # Pad if needed
    if n_components < latent_dim:
        padded = np.zeros((embeddings.shape[0], latent_dim), dtype=np.float32)
        padded[:, :n_components] = embeddings
        padded[:, n_components:] = ref_mu[n_components:]
        embeddings = padded

    # Compute neighbor distances in latent space for quality metric
    from sklearn.neighbors import NearestNeighbors

    k = min(k_neighbors, ref_latent.shape[0])
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(ref_latent)
    distances, _ = nn.kneighbors(embeddings)
    mean_distances = distances.mean(axis=1)

    return embeddings.astype(np.float32), mean_distances.astype(np.float32)


def _map_scvi_query(
    query: Any,
    ref_adata: Any,
    ref_latent: np.ndarray,
    k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map query cells using scVI/scANVI query mapping.

    This requires a trained scVI model associated with the reference.
    Falls back to k-NN projection if model is unavailable.
    """
    log.warning("scVI query mapping not yet implemented. Falling back to k-NN projection.")
    from stagebridge.geometry import EuclideanBackend

    return _map_knn_projection(
        query=query,
        ref_adata=ref_adata,
        ref_latent=ref_latent,
        k_neighbors=k_neighbors,
        geometry=EuclideanBackend(),
    )


def compute_reference_neighborhood(
    mapping_result: MappingResult,
    reference: Any,
    *,
    k: int = 10,
    label_col: str | None = None,
) -> ReferenceNeighborhood:
    """Compute reference neighborhood summary for mapped cells.

    Parameters
    ----------
    mapping_result : MappingResult
        Result from map_to_hlca or map_to_luca
    reference : AnnData or LoadedReference
        Reference atlas
    k : int
        Number of neighbors
    label_col : str, optional
        Column in reference.obs to extract neighbor labels

    Returns
    -------
    ReferenceNeighborhood
        Neighborhood summary
    """
    from sklearn.neighbors import NearestNeighbors

    # Handle LoadedReference wrapper
    ref_adata = reference.adata if hasattr(reference, "adata") else reference

    ref_latent = np.asarray(
        ref_adata.obsm[mapping_result.reference_latent_key],
        dtype=np.float32,
    )

    k = min(k, ref_latent.shape[0])
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(ref_latent)

    distances, indices = nn.kneighbors(mapping_result.embeddings)

    neighbor_labels = None
    if label_col and label_col in ref_adata.obs.columns:
        ref_labels = ref_adata.obs[label_col].astype(str).to_numpy()
        neighbor_labels = ref_labels[indices]

    return ReferenceNeighborhood(
        k_neighbors=k,
        neighbor_indices=indices,
        neighbor_distances=distances,
        neighbor_labels=neighbor_labels,
    )
