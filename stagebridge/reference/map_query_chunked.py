"""Memory-efficient chunked query-to-reference mapping.

Designed for large references (>1M cells) and limited RAM.
Uses FAISS for approximate k-NN and processes in chunks.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


def map_query_chunked(
    query_adata: Any,
    ref_path: Path,
    *,
    latent_key: str = "X_scVI",
    k_neighbors: int = 50,
    query_chunk_size: int = 10000,
    ref_chunk_size: int = 50000,  # Reduced from 100K to 50K for memory safety
    use_faiss: bool = True,
    n_probe: int = 32,
    normalize: bool = True,
    max_gene_dims: int = 2000,
    pca_components: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map query cells to reference latent space with chunked processing.

    Memory-efficient approach:
    1. Build gene index mapping once
    2. If too many genes, use PCA-reduced space for k-NN
    3. Build FAISS index from reduced/gene space
    4. Find k-NN neighbors, get weighted average of reference latent positions

    Parameters
    ----------
    query_adata : AnnData
        Query data (can be small, already subsampled)
    ref_path : Path
        Path to reference h5ad (loaded in backed mode)
    latent_key : str
        Key in reference obsm for latent embeddings
    k_neighbors : int
        Number of neighbors for k-NN
    query_chunk_size : int
        Number of query cells per chunk
    ref_chunk_size : int
        Number of reference cells to process at once for index building
    use_faiss : bool
        Use FAISS for fast approximate k-NN (recommended)
    n_probe : int
        FAISS IVF probe parameter (higher = more accurate, slower)
    normalize : bool
        L2 normalize vectors before k-NN
    max_gene_dims : int
        If common genes exceed this, use PCA reduction (default 2000)
    pca_components : int
        Number of PCA components for dimensionality reduction (default 100)

    Returns
    -------
    embeddings : np.ndarray
        Query embeddings in reference latent space (n_query, latent_dim)
    distances : np.ndarray
        Mean k-NN distances per query cell
    info : dict
        Mapping statistics including dimensionality reduction method
    """
    import anndata
    import scipy.sparse as sp

    log.info("Starting chunked query mapping to %s", ref_path.name)

    # Load reference in backed mode
    ref_adata = anndata.read_h5ad(ref_path, backed="r")
    n_ref = ref_adata.n_obs
    n_ref_genes = ref_adata.n_vars

    log.info("Reference: %d cells, %d genes (backed mode)", n_ref, n_ref_genes)

    # Check latent exists
    if latent_key not in ref_adata.obsm:
        raise KeyError(
            f"Reference missing latent key '{latent_key}'. Available: {list(ref_adata.obsm.keys())}"
        )

    ref_latent = np.asarray(ref_adata.obsm[latent_key], dtype=np.float32)
    latent_dim = ref_latent.shape[1]
    log.info("Reference latent: %d dims", latent_dim)

    # Check for NaN in reference latent - DO NOT zero-fill, filter instead
    nan_per_cell = np.isnan(ref_latent).any(axis=1)
    n_invalid = nan_per_cell.sum()
    if n_invalid > 0:
        valid_fraction = 1.0 - n_invalid / n_ref
        if valid_fraction < 0.5:
            raise ValueError(
                f"Reference latent has {n_invalid:,} invalid cells ({100 * (1 - valid_fraction):.1f}%). "
                f"Run: python -m stagebridge.reference.diagnose_reference {ref_path} --diagnose-only"
            )
        log.warning(
            "Reference latent has %d cells with NaN (%d%%) - filtering them out. "
            "Consider running diagnose_reference.py to create a cleaned reference.",
            n_invalid,
            int(100 * n_invalid / n_ref),
        )
        # Create mask for valid cells
        valid_cell_mask = ~nan_per_cell
        ref_latent = ref_latent[valid_cell_mask]
        n_ref = ref_latent.shape[0]
        log.info("After filtering: %d valid reference cells", n_ref)

    # Build gene mapping (query symbols -> reference indices)
    query_genes = list(query_adata.var_names.astype(str))
    ref_var_names = list(ref_adata.var_names.astype(str))

    # Handle ENSG -> symbol mapping
    first_ref_gene = ref_var_names[0] if ref_var_names else ""
    if first_ref_gene.startswith("ENSG") and "feature_name" in ref_adata.var.columns:
        ref_symbols = ref_adata.var["feature_name"].astype(str).tolist()
        ref_symbol_to_idx = {sym: idx for idx, sym in enumerate(ref_symbols)}
        log.info("Using feature_name for gene matching")
    else:
        ref_symbol_to_idx = {g: idx for idx, g in enumerate(ref_var_names)}

    # Find common genes
    common_genes = [g for g in query_genes if g in ref_symbol_to_idx]
    query_gene_idx = [query_genes.index(g) for g in common_genes]
    ref_gene_idx = [ref_symbol_to_idx[g] for g in common_genes]

    n_common = len(common_genes)
    log.info("Gene overlap: %d common genes (%.1f%%)", n_common, 100 * n_common / n_ref_genes)

    if n_common < 100:
        log.warning("Low gene overlap - mapping quality may be poor")

    # Determine if we need dimensionality reduction
    use_pca = n_common > max_gene_dims
    pca_model = None
    dim_reduction_method = "none"

    if use_pca:
        log.info(
            "Gene count (%d) exceeds max (%d) - using PCA reduction to %d dims",
            n_common,
            max_gene_dims,
            pca_components,
        )
        dim_reduction_method = f"pca_{pca_components}"

        # Fit PCA on a sample of reference cells
        from sklearn.decomposition import IncrementalPCA

        pca_model = IncrementalPCA(n_components=pca_components)

        # Sample cells for PCA fitting
        n_pca_sample = min(50000, ref_adata.n_obs)
        pca_sample_idx = np.random.choice(ref_adata.n_obs, n_pca_sample, replace=False)
        pca_sample_idx.sort()

        log.info("Fitting PCA on %d sampled reference cells...", n_pca_sample)

        # Fit incrementally in chunks
        for start in range(0, len(pca_sample_idx), ref_chunk_size):
            end = min(start + ref_chunk_size, len(pca_sample_idx))
            chunk_idx = pca_sample_idx[start:end]

            chunk = ref_adata.X[chunk_idx, :]
            if sp.issparse(chunk):
                chunk = chunk.toarray()
            chunk = np.asarray(chunk[:, ref_gene_idx], dtype=np.float32)

            # Normalize before PCA
            if normalize:
                norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8
                chunk = chunk / norms

            pca_model.partial_fit(chunk)

        log.info(
            "PCA fitted. Explained variance: %.1f%%",
            100 * pca_model.explained_variance_ratio_.sum(),
        )

    # Process query in chunks to avoid memory explosion
    # For 787K cells x 15K genes, dense would be 47GB - process in batches instead
    n_query = query_adata.n_obs
    effective_dims = pca_components if use_pca else n_common

    log.info(
        "Processing query in chunks (%d cells, %d effective dims, method: %s)",
        n_query,
        effective_dims,
        dim_reduction_method,
    )

    # Prepare query expression in chunks
    X_query_chunks = []
    for q_start in range(0, n_query, query_chunk_size):
        q_end = min(q_start + query_chunk_size, n_query)

        # Get chunk of query expression for common genes only
        X_chunk = query_adata.X[q_start:q_end, :]
        if sp.issparse(X_chunk):
            X_chunk = X_chunk.toarray()
        X_chunk = np.asarray(X_chunk[:, query_gene_idx], dtype=np.float32)

        if normalize:
            X_chunk = X_chunk / (np.linalg.norm(X_chunk, axis=1, keepdims=True) + 1e-8)

        # Apply PCA to query chunk if needed
        if use_pca and pca_model is not None:
            X_chunk = pca_model.transform(X_chunk).astype(np.float32)

        X_query_chunks.append(X_chunk)

        if (q_start // query_chunk_size) % 10 == 0:
            log.info("  Prepared query chunk %d-%d / %d", q_start, q_end, n_query)

    # Concatenate processed chunks (now in reduced dimension space)
    X_query = np.vstack(X_query_chunks)
    del X_query_chunks  # Free memory

    log.info("Query prepared: %d cells, %d dims", X_query.shape[0], X_query.shape[1])

    # Track valid cell indices if we filtered NaN cells
    valid_cell_indices = None
    if "valid_cell_mask" in dir() and valid_cell_mask is not None:
        valid_cell_indices = np.where(valid_cell_mask)[0]

    # Use FAISS with streaming - never load full reference matrix
    if use_faiss:
        embeddings, distances = _map_with_faiss_streaming(
            X_query,
            ref_adata,
            ref_gene_idx,
            ref_latent,
            k_neighbors=k_neighbors,
            n_probe=n_probe,
            chunk_size=ref_chunk_size,
            normalize=normalize,
            pca_model=pca_model,
            valid_cell_indices=valid_cell_indices,
        )
    else:
        embeddings, distances = _map_with_sklearn_streaming(
            X_query,
            ref_adata,
            ref_gene_idx,
            ref_latent,
            k_neighbors=k_neighbors,
            chunk_size=ref_chunk_size,
            normalize=normalize,
            pca_model=pca_model,
            valid_cell_indices=valid_cell_indices,
        )

    # Cleanup
    ref_adata.file.close()

    # Final NaN check - report but do NOT zero-fill embeddings
    nan_embed = np.isnan(embeddings).sum()
    nan_dist = np.isnan(distances).sum()
    if nan_embed > 0 or nan_dist > 0:
        log.error(
            "Final NaN check FAILED: %d NaN in embeddings, %d in distances. "
            "This indicates a problem with the reference or mapping. "
            "Do NOT proceed with these outputs.",
            nan_embed,
            nan_dist,
        )
        # Set distances to max for NaN embeddings so confidence will be low
        nan_rows = np.isnan(embeddings).any(axis=1)
        distances[nan_rows] = np.inf

    info = {
        "n_query": n_query,
        "n_ref": n_ref,
        "n_ref_original": ref_adata.n_obs,
        "n_ref_filtered": ref_adata.n_obs - n_ref if n_ref != ref_adata.n_obs else 0,
        "n_common_genes": n_common,
        "effective_dims": effective_dims,
        "dim_reduction_method": dim_reduction_method,
        "latent_dim": latent_dim,
        "k_neighbors": k_neighbors,
        "knn_method": "faiss" if use_faiss else "sklearn",
        "nan_in_embeddings": int(nan_embed),
        "nan_in_distances": int(nan_dist),
    }

    return embeddings, distances, info


def _map_with_faiss_streaming(
    X_query: np.ndarray,
    ref_adata: Any,
    ref_gene_idx: list[int],
    ref_latent: np.ndarray,
    k_neighbors: int,
    n_probe: int,
    chunk_size: int,
    normalize: bool,
    pca_model: Any = None,
    valid_cell_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use FAISS with streaming - build index from chunks without full matrix.

    Parameters
    ----------
    pca_model : IncrementalPCA, optional
        If provided, transform reference chunks to PCA space before indexing
    valid_cell_indices : np.ndarray, optional
        If provided, only use these reference cell indices (for NaN filtering)
    """
    import scipy.sparse as sp

    try:
        import faiss

        has_faiss = True
    except ImportError:
        log.warning("FAISS not installed, falling back to streaming sklearn")
        has_faiss = False

    # Determine number of reference cells to use
    n_ref_full = ref_adata.n_obs
    if valid_cell_indices is not None:
        n_ref = len(valid_cell_indices)
        log.info("Using %d valid reference cells (filtered from %d)", n_ref, n_ref_full)
    else:
        n_ref = n_ref_full

    n_query = X_query.shape[0]
    # Dimension is PCA dims if using PCA, otherwise gene dims
    dim = X_query.shape[1]
    latent_dim = ref_latent.shape[1]

    if not has_faiss:
        return _map_with_sklearn_streaming(
            X_query,
            ref_adata,
            ref_gene_idx,
            ref_latent,
            k_neighbors,
            chunk_size,
            normalize,
            pca_model=pca_model,
            valid_cell_indices=valid_cell_indices,
        )

    log.info("Building FAISS index with streaming (n=%d, d=%d)...", n_ref, dim)

    def _process_chunk(raw_chunk):
        """Process a raw expression chunk: subset genes, normalize, optionally PCA."""
        if sp.issparse(raw_chunk):
            raw_chunk = raw_chunk.toarray()
        chunk = np.asarray(raw_chunk[:, ref_gene_idx], dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8
            chunk = chunk / norms
        if pca_model is not None:
            chunk = pca_model.transform(chunk).astype(np.float32)
        return chunk

    # Determine which cells to use
    if valid_cell_indices is not None:
        cells_to_use = valid_cell_indices
    else:
        cells_to_use = np.arange(n_ref_full)

    # For large datasets, use IVF - but need training data first
    if n_ref > 100000:
        n_clusters = min(int(np.sqrt(n_ref)), 2048)
        n_train = min(n_ref, n_clusters * 50)

        log.info("Sampling %d cells for IVF training...", n_train)
        # Sample from valid cells
        train_sample_idx = np.random.choice(len(cells_to_use), n_train, replace=False)
        train_cell_ids = cells_to_use[train_sample_idx]
        train_cell_ids.sort()

        # Load training data in chunks
        train_data = []
        for i in range(0, len(train_cell_ids), chunk_size):
            batch_ids = train_cell_ids[i : i + chunk_size]
            chunk = ref_adata.X[batch_ids, :]
            chunk = _process_chunk(chunk)
            train_data.append(chunk)

        train_data = np.vstack(train_data)
        log.info("Training IVF index with %d samples...", len(train_data))

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(train_data)
        del train_data

        index.nprobe = n_probe
        log.info("FAISS IVF: %d clusters, nprobe=%d", n_clusters, n_probe)
    else:
        index = faiss.IndexFlatIP(dim)
        log.info("FAISS flat index")

    # Stream through valid reference cells and add to index
    # Also build mapping from index position to latent position
    log.info("Adding reference vectors to index (streaming)...")
    n_added = 0
    for i in range(0, len(cells_to_use), chunk_size):
        batch_ids = cells_to_use[i : i + chunk_size]
        chunk = ref_adata.X[batch_ids, :]
        chunk = _process_chunk(chunk)
        index.add(chunk)
        n_added += len(batch_ids)

        if (i // chunk_size) % 10 == 0:
            log.info("  Added %d / %d vectors", n_added, n_ref)

    log.info("Index ready: %d vectors", index.ntotal)

    # Search in batches to avoid memory/time explosion
    # 787K queries × 584K refs is too large for one operation
    log.info("Searching k=%d neighbors (batched)...", k_neighbors)
    query_batch_size = 10000  # Search 10K query cells at a time
    all_similarities = []
    all_indices = []

    for q_start in range(0, n_query, query_batch_size):
        q_end = min(q_start + query_batch_size, n_query)
        query_batch = X_query[q_start:q_end]

        sims, idxs = index.search(query_batch, k_neighbors)
        all_similarities.append(sims)
        all_indices.append(idxs)

        if (q_start // query_batch_size) % 10 == 0:
            log.info("  Searched %d / %d query cells", q_end, n_query)

    similarities = np.vstack(all_similarities)
    indices = np.vstack(all_indices)
    log.info("Search complete: %d queries processed", n_query)

    # Convert similarities to distances
    distances = 1.0 - np.clip(similarities, -1, 1)

    # Weighted average of latent positions
    log.info("Computing weighted latent embeddings...")

    # Handle any inf/nan in distances
    distances = np.where(np.isinf(distances) | np.isnan(distances), 1e6, distances)

    weights = 1.0 / (distances + 1e-6)
    weight_sums = weights.sum(axis=1, keepdims=True)

    # Handle cells with zero weight sum - use uniform weights
    zero_weight_mask = weight_sums.flatten() < 1e-10
    if zero_weight_mask.any():
        log.warning(
            "%d cells had invalid distances - using uniform weights", zero_weight_mask.sum()
        )
        weights[zero_weight_mask] = 1.0 / k_neighbors
        weight_sums[zero_weight_mask] = 1.0

    weights = weights / weight_sums

    embeddings = np.zeros((n_query, latent_dim), dtype=np.float32)
    for i in range(n_query):
        neighbor_latents = ref_latent[indices[i]]
        embeddings[i] = np.sum(weights[i, :, np.newaxis] * neighbor_latents, axis=0)

    mean_distances = distances.mean(axis=1)

    return embeddings, mean_distances


def _map_with_sklearn_streaming(
    X_query: np.ndarray,
    ref_adata: Any,
    ref_gene_idx: list[int],
    ref_latent: np.ndarray,
    k_neighbors: int,
    chunk_size: int,
    normalize: bool,
    pca_model: Any = None,
    valid_cell_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Streaming sklearn - process reference in chunks, keep top-k per query.

    Parameters
    ----------
    pca_model : IncrementalPCA, optional
        If provided, transform reference chunks to PCA space
    valid_cell_indices : np.ndarray, optional
        If provided, only use these reference cell indices
    """
    import scipy.sparse as sp
    from sklearn.metrics.pairwise import cosine_similarity

    n_ref_full = ref_adata.n_obs
    if valid_cell_indices is not None:
        n_ref = len(valid_cell_indices)
        cells_to_use = valid_cell_indices
    else:
        n_ref = n_ref_full
        cells_to_use = np.arange(n_ref_full)

    n_query = X_query.shape[0]
    latent_dim = ref_latent.shape[1]

    log.info("Streaming sklearn k-NN (n_ref=%d, n_query=%d)...", n_ref, n_query)

    def _process_chunk(raw_chunk):
        """Process a raw expression chunk: subset genes, normalize, optionally PCA."""
        if sp.issparse(raw_chunk):
            raw_chunk = raw_chunk.toarray()
        chunk = np.asarray(raw_chunk[:, ref_gene_idx], dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8
            chunk = chunk / norms
        if pca_model is not None:
            chunk = pca_model.transform(chunk).astype(np.float32)
        return chunk

    # Track top-k neighbors per query
    # Indices here refer to position in ref_latent (which matches cells_to_use order)
    top_k_distances = np.full((n_query, k_neighbors), np.inf, dtype=np.float32)
    top_k_indices = np.zeros((n_query, k_neighbors), dtype=np.int64)

    # Stream through valid reference cells in chunks
    for i in range(0, len(cells_to_use), chunk_size):
        batch_ids = cells_to_use[i : i + chunk_size]
        chunk = ref_adata.X[batch_ids, :]
        chunk = _process_chunk(chunk)

        # Compute cosine similarity
        sims = cosine_similarity(X_query, chunk)  # (n_query, chunk_size)
        dists = 1.0 - sims

        # Update top-k
        for q in range(n_query):
            chunk_dists = dists[q]
            chunk_top_k = np.argsort(chunk_dists)[:k_neighbors]

            # Merge with existing top-k
            # Indices are into ref_latent, which is indexed by position in iteration
            all_dists = np.concatenate([top_k_distances[q], chunk_dists[chunk_top_k]])
            all_indices = np.concatenate([top_k_indices[q], chunk_top_k + i])

            merged_order = np.argsort(all_dists)[:k_neighbors]
            top_k_distances[q] = all_dists[merged_order]
            top_k_indices[q] = all_indices[merged_order]

        if (i // chunk_size) % 5 == 0:
            log.info("  Processed ref chunk %d-%d / %d", i, i + len(batch_ids), n_ref)

    # Compute weighted embeddings
    log.info("Computing weighted latent embeddings...")

    # Replace inf with large finite value to avoid 0 weights
    top_k_distances = np.where(np.isinf(top_k_distances), 1e6, top_k_distances)

    weights = 1.0 / (top_k_distances + 1e-6)
    weight_sums = weights.sum(axis=1, keepdims=True)

    # Handle cells with zero weight sum (no valid neighbors) - use uniform weights
    zero_weight_mask = weight_sums.flatten() < 1e-10
    if zero_weight_mask.any():
        log.warning(
            "%d cells had no valid neighbors - using uniform weights", zero_weight_mask.sum()
        )
        weights[zero_weight_mask] = 1.0 / k_neighbors
        weight_sums[zero_weight_mask] = 1.0

    weights = weights / weight_sums

    embeddings = np.zeros((n_query, latent_dim), dtype=np.float32)
    for i in range(n_query):
        neighbor_latents = ref_latent[top_k_indices[i]]
        embeddings[i] = np.sum(weights[i, :, np.newaxis] * neighbor_latents, axis=0)

    mean_distances = top_k_distances.mean(axis=1)

    return embeddings, mean_distances


def map_to_dual_reference_chunked(
    query_adata: Any,
    hlca_path: Path | None,
    luca_path: Path | None,
    *,
    hlca_latent_key: str = "X_scanvi_emb",
    luca_latent_key: str = "X_scVI",
    k_neighbors: int = 50,
    use_faiss: bool = True,
) -> dict[str, Any]:
    """Map query to both HLCA and LuCA references with chunked processing.

    Returns
    -------
    dict with keys:
        - hlca_embeddings: np.ndarray or None
        - luca_embeddings: np.ndarray or None
        - hlca_distances: np.ndarray or None
        - luca_distances: np.ndarray or None
        - fused_embeddings: np.ndarray
        - cell_ids: np.ndarray
        - metadata: dict
    """
    results = {
        "hlca_embeddings": None,
        "luca_embeddings": None,
        "hlca_distances": None,
        "luca_distances": None,
        "fused_embeddings": None,
        "cell_ids": query_adata.obs_names.astype(str).to_numpy(),
        "metadata": {},
    }

    # Map to HLCA
    if hlca_path is not None and hlca_path.exists():
        log.info("=== Mapping to HLCA ===")
        hlca_emb, hlca_dist, hlca_info = map_query_chunked(
            query_adata,
            hlca_path,
            latent_key=hlca_latent_key,
            k_neighbors=k_neighbors,
            use_faiss=use_faiss,
        )
        results["hlca_embeddings"] = hlca_emb
        results["hlca_distances"] = hlca_dist
        results["metadata"]["hlca"] = hlca_info
        log.info("HLCA mapping complete: %s", hlca_emb.shape)

    # Map to LuCA
    if luca_path is not None and luca_path.exists():
        log.info("=== Mapping to LuCA ===")
        luca_emb, luca_dist, luca_info = map_query_chunked(
            query_adata,
            luca_path,
            latent_key=luca_latent_key,
            k_neighbors=k_neighbors,
            use_faiss=use_faiss,
        )
        results["luca_embeddings"] = luca_emb
        results["luca_distances"] = luca_dist
        results["metadata"]["luca"] = luca_info
        log.info("LuCA mapping complete: %s", luca_emb.shape)

    # Fuse embeddings
    if results["hlca_embeddings"] is not None and results["luca_embeddings"] is not None:
        results["fused_embeddings"] = np.concatenate(
            [results["hlca_embeddings"], results["luca_embeddings"]], axis=1
        )
        log.info("Fused embeddings: %s", results["fused_embeddings"].shape)
    elif results["hlca_embeddings"] is not None:
        results["fused_embeddings"] = results["hlca_embeddings"]
    elif results["luca_embeddings"] is not None:
        results["fused_embeddings"] = results["luca_embeddings"]

    return results
