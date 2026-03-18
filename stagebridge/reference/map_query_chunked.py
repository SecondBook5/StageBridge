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
    ref_chunk_size: int = 100000,
    use_faiss: bool = True,
    n_probe: int = 32,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map query cells to reference latent space with chunked processing.

    Memory-efficient approach:
    1. Build gene index mapping once
    2. Build FAISS index from reference latent (not expression)
    3. For each query chunk:
       - Load query expression
       - Compute approximate k-NN in gene space OR use precomputed latent
       - Get weighted average of reference latent positions

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

    Returns
    -------
    embeddings : np.ndarray
        Query embeddings in reference latent space (n_query, latent_dim)
    distances : np.ndarray
        Mean k-NN distances per query cell
    info : dict
        Mapping statistics
    """
    import anndata
    import scipy.sparse as sp

    log.info("Starting chunked query mapping to %s", ref_path.name)

    # Load reference in backed mode
    ref_adata = anndata.read_h5ad(ref_path, backed='r')
    n_ref = ref_adata.n_obs
    n_ref_genes = ref_adata.n_vars

    log.info("Reference: %d cells, %d genes (backed mode)", n_ref, n_ref_genes)

    # Check latent exists
    if latent_key not in ref_adata.obsm:
        raise KeyError(f"Reference missing latent key '{latent_key}'. Available: {list(ref_adata.obsm.keys())}")

    ref_latent = np.asarray(ref_adata.obsm[latent_key], dtype=np.float32)
    latent_dim = ref_latent.shape[1]
    log.info("Reference latent: %d dims", latent_dim)

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

    # Get query expression for common genes
    X_query = query_adata.X
    if sp.issparse(X_query):
        X_query = X_query.toarray()
    X_query = np.asarray(X_query[:, query_gene_idx], dtype=np.float32)

    if normalize:
        X_query = X_query / (np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-8)

    n_query = X_query.shape[0]
    log.info("Query: %d cells, %d common genes", n_query, n_common)

    # Use FAISS with streaming - never load full reference matrix
    if use_faiss:
        embeddings, distances = _map_with_faiss_streaming(
            X_query, ref_adata, ref_gene_idx, ref_latent,
            k_neighbors=k_neighbors,
            n_probe=n_probe,
            chunk_size=ref_chunk_size,
            normalize=normalize,
        )
    else:
        embeddings, distances = _map_with_sklearn_streaming(
            X_query, ref_adata, ref_gene_idx, ref_latent,
            k_neighbors=k_neighbors,
            chunk_size=ref_chunk_size,
            normalize=normalize,
        )

    # Cleanup
    ref_adata.file.close()

    info = {
        "n_query": n_query,
        "n_ref": n_ref,
        "n_common_genes": n_common,
        "latent_dim": latent_dim,
        "k_neighbors": k_neighbors,
        "method": "faiss" if use_faiss else "sklearn",
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
) -> tuple[np.ndarray, np.ndarray]:
    """Use FAISS with streaming - build index from chunks without full matrix."""
    import scipy.sparse as sp

    try:
        import faiss
        has_faiss = True
    except ImportError:
        log.warning("FAISS not installed, falling back to streaming sklearn")
        has_faiss = False

    n_ref = ref_adata.n_obs
    n_query = X_query.shape[0]
    dim = len(ref_gene_idx)
    latent_dim = ref_latent.shape[1]

    if not has_faiss:
        return _map_with_sklearn_streaming(
            X_query, ref_adata, ref_gene_idx, ref_latent,
            k_neighbors, chunk_size, normalize
        )

    log.info("Building FAISS index with streaming (n=%d, d=%d)...", n_ref, dim)

    # For large datasets, use IVF - but need training data first
    # Sample a subset for training
    if n_ref > 100000:
        n_clusters = min(int(np.sqrt(n_ref)), 2048)
        n_train = min(n_ref, n_clusters * 50)

        log.info("Sampling %d cells for IVF training...", n_train)
        train_idx = np.random.choice(n_ref, n_train, replace=False)
        train_idx.sort()

        # Load training data
        train_data = []
        current_idx = 0
        for start in range(0, n_ref, chunk_size):
            end = min(start + chunk_size, n_ref)
            # Check which training indices fall in this chunk
            chunk_train_mask = (train_idx >= start) & (train_idx < end)
            if not chunk_train_mask.any():
                continue

            chunk_train_idx = train_idx[chunk_train_mask] - start
            chunk = ref_adata.X[start:end, :]
            if sp.issparse(chunk):
                chunk = chunk.toarray()
            chunk = np.asarray(chunk[:, ref_gene_idx], dtype=np.float32)
            if normalize:
                norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8
                chunk = chunk / norms
            train_data.append(chunk[chunk_train_idx])

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

    # Stream through reference and add to index
    log.info("Adding reference vectors to index (streaming)...")
    for start in range(0, n_ref, chunk_size):
        end = min(start + chunk_size, n_ref)
        chunk = ref_adata.X[start:end, :]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
        chunk = np.asarray(chunk[:, ref_gene_idx], dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8
            chunk = chunk / norms
        index.add(chunk)

        if (start // chunk_size) % 10 == 0:
            log.info("  Added %d / %d vectors", end, n_ref)

    log.info("Index ready: %d vectors", index.ntotal)

    # Search
    log.info("Searching k=%d neighbors...", k_neighbors)
    similarities, indices = index.search(X_query, k_neighbors)

    # Convert similarities to distances
    distances = 1.0 - np.clip(similarities, -1, 1)

    # Weighted average of latent positions
    log.info("Computing weighted latent embeddings...")
    weights = 1.0 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

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
) -> tuple[np.ndarray, np.ndarray]:
    """Streaming sklearn - process reference in chunks, keep top-k per query."""
    import scipy.sparse as sp
    from sklearn.metrics.pairwise import cosine_similarity

    n_ref = ref_adata.n_obs
    n_query = X_query.shape[0]
    latent_dim = ref_latent.shape[1]

    log.info("Streaming sklearn k-NN (n_ref=%d, n_query=%d)...", n_ref, n_query)

    # Track top-k neighbors per query
    top_k_distances = np.full((n_query, k_neighbors), np.inf, dtype=np.float32)
    top_k_indices = np.zeros((n_query, k_neighbors), dtype=np.int64)

    # Stream through reference chunks
    for start in range(0, n_ref, chunk_size):
        end = min(start + chunk_size, n_ref)
        chunk = ref_adata.X[start:end, :]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
        chunk = np.asarray(chunk[:, ref_gene_idx], dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-8
            chunk = chunk / norms

        # Compute cosine similarity
        sims = cosine_similarity(X_query, chunk)  # (n_query, chunk_size)
        dists = 1.0 - sims

        # Update top-k
        for i in range(n_query):
            chunk_dists = dists[i]
            chunk_top_k = np.argsort(chunk_dists)[:k_neighbors]

            # Merge with existing top-k
            all_dists = np.concatenate([top_k_distances[i], chunk_dists[chunk_top_k]])
            all_indices = np.concatenate([top_k_indices[i], chunk_top_k + start])

            merged_order = np.argsort(all_dists)[:k_neighbors]
            top_k_distances[i] = all_dists[merged_order]
            top_k_indices[i] = all_indices[merged_order]

        if (start // chunk_size) % 5 == 0:
            log.info("  Processed ref chunk %d-%d / %d", start, end, n_ref)

    # Compute weighted embeddings
    log.info("Computing weighted latent embeddings...")
    weights = 1.0 / (top_k_distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

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
            query_adata, hlca_path,
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
            query_adata, luca_path,
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
