"""Model-based query-to-reference mapping using scVI/scANVI with scArches surgery.

This is the principled approach: use pretrained reference models to encode
query cells directly into reference latent spaces, following the official
scvi-tools tutorial workflow.

Falls back to k-NN projection only if models are unavailable.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


def map_query_with_scanvi_model(
    query_adata: Any,
    model_path: Path,
    *,
    batch_size: int = 10000,
    surgery_epochs: int = 200,
) -> tuple[np.ndarray, dict]:
    """Map query to HLCA using scArches surgery with scANVI.

    Follows official scvi-tools HLCA tutorial workflow:
    1. Load reference model
    2. Prepare query anndata (reorder/pad genes)
    3. Load query data into model
    4. Fine-tune with surgery (scArches)
    5. Get latent representation

    Parameters
    ----------
    query_adata : AnnData
        Query data (cells × genes)
    model_path : Path
        Path to scANVI model directory or HubModel cache
    batch_size : int
        Batch size for encoding
    surgery_epochs : int
        Max epochs for scArches surgery fine-tuning (default 200)

    Returns
    -------
    embeddings : np.ndarray
        Query embeddings in reference latent space (n_cells, latent_dim)
    info : dict
        Mapping metadata
    """
    from scvi.model import SCANVI

    log.info("Loading scANVI reference model from %s", model_path)

    # Load reference model
    try:
        ref_model = SCANVI.load(str(model_path), adata=None)
        log.info("  Reference model loaded successfully")
    except Exception as e:
        raise ValueError(f"Failed to load scANVI model: {e}")

    # Prepare query anndata - this reorders genes and pads missing ones
    log.info("  Preparing query anndata (gene matching and padding)...")
    query_copy = query_adata.copy()

    try:
        SCANVI.prepare_query_anndata(query_copy, ref_model)
        log.info("  Query prepared - genes matched to reference")
    except Exception as e:
        raise ValueError(f"Failed to prepare query anndata: {e}")

    # Get gene overlap stats
    n_ref_genes = ref_model.adata_manager.get_state_registry("var_names").index.shape[0]
    n_query_genes = query_copy.n_vars
    log.info("  Reference genes: %d, Query genes after prep: %d", n_ref_genes, n_query_genes)

    # Set unlabeled category for query (required for scANVI)
    query_copy.obs["scanvi_label"] = "unlabeled"

    # Add dataset column if missing (required by scANVI surgery)
    if "dataset" not in query_copy.obs.columns:
        query_copy.obs["dataset"] = "query_dataset"

    # Load query data into model (creates query-specific model instance)
    log.info("  Loading query into scANVI model...")
    try:
        query_model = SCANVI.load_query_data(query_copy, ref_model)
        log.info("  Query model created for surgery")
    except Exception as e:
        raise ValueError(f"Failed to load query data: {e}")

    # Run scArches surgery (fine-tuning on query)
    log.info("  Running scArches surgery (max %d epochs)...", surgery_epochs)
    train_kwargs = {
        "max_epochs": surgery_epochs,
        "early_stopping": True,
        "early_stopping_monitor": "elbo_train",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "plan_kwargs": {"weight_decay": 0.0},
        "check_val_every_n_epoch": None,  # Disable validation
        "enable_progress_bar": False,  # Disable tqdm for cleaner logs
    }

    try:
        query_model.train(**train_kwargs)
        log.info("  Surgery complete")
    except Exception as e:
        log.warning("  Surgery training had issues: %s", e)
        log.warning("  Proceeding to get embeddings anyway...")

    # Get latent representation
    log.info("  Getting latent representation...")
    try:
        embeddings = query_model.get_latent_representation(
            query_copy,
            batch_size=batch_size,
        )
    except Exception as e:
        raise ValueError(f"Failed to get latent representation: {e}")

    log.info("  Encoded %d cells to %d-dimensional latent space",
             embeddings.shape[0], embeddings.shape[1])

    info = {
        "method": "scanvi_scarches",
        "model_path": str(model_path),
        "n_query": query_copy.n_obs,
        "n_genes_reference": n_ref_genes,
        "latent_dim": embeddings.shape[1],
        "surgery_epochs": surgery_epochs,
    }

    return embeddings.astype(np.float32), info


def map_query_with_scvi_model(
    query_adata: Any,
    model_path: Path,
    *,
    batch_size: int = 10000,
    surgery_epochs: int = 200,
) -> tuple[np.ndarray, dict]:
    """Map query to LuCA using scArches surgery with scVI.

    Parameters
    ----------
    query_adata : AnnData
        Query data (cells × genes)
    model_path : Path
        Path to scVI model directory
    batch_size : int
        Batch size for encoding
    surgery_epochs : int
        Max epochs for scArches surgery (default 200)

    Returns
    -------
    embeddings : np.ndarray
        Query embeddings in reference latent space (n_cells, latent_dim)
    info : dict
        Mapping metadata
    """
    from scvi.model import SCVI

    log.info("Loading scVI reference model from %s", model_path)

    # Load reference model
    try:
        ref_model = SCVI.load(str(model_path), adata=None)
        log.info("  Reference model loaded successfully")
    except Exception as e:
        raise ValueError(f"Failed to load scVI model: {e}")

    # Prepare query anndata
    log.info("  Preparing query anndata...")
    query_copy = query_adata.copy()

    try:
        SCVI.prepare_query_anndata(query_copy, ref_model)
        log.info("  Query prepared - genes matched to reference")
    except Exception as e:
        raise ValueError(f"Failed to prepare query anndata: {e}")

    n_ref_genes = ref_model.adata_manager.get_state_registry("var_names").index.shape[0]
    n_query_genes = query_copy.n_vars
    log.info("  Reference genes: %d, Query genes after prep: %d", n_ref_genes, n_query_genes)

    # Add dataset column if missing
    if "dataset" not in query_copy.obs.columns:
        query_copy.obs["dataset"] = "query_dataset"

    # Load query into model
    log.info("  Loading query into scVI model...")
    try:
        query_model = SCVI.load_query_data(query_copy, ref_model)
        log.info("  Query model created for surgery")
    except Exception as e:
        raise ValueError(f"Failed to load query data: {e}")

    # Run surgery
    log.info("  Running scArches surgery (max %d epochs)...", surgery_epochs)
    train_kwargs = {
        "max_epochs": surgery_epochs,
        "early_stopping": True,
        "early_stopping_monitor": "elbo_train",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "plan_kwargs": {"weight_decay": 0.0},
        "check_val_every_n_epoch": None,
        "enable_progress_bar": False,
    }

    try:
        query_model.train(**train_kwargs)
        log.info("  Surgery complete")
    except Exception as e:
        log.warning("  Surgery training had issues: %s", e)
        log.warning("  Proceeding to get embeddings anyway...")

    # Get latent representation
    log.info("  Getting latent representation...")
    try:
        embeddings = query_model.get_latent_representation(
            query_copy,
            batch_size=batch_size,
        )
    except Exception as e:
        raise ValueError(f"Failed to get latent representation: {e}")

    log.info("  Encoded %d cells to %d-dimensional latent space",
             embeddings.shape[0], embeddings.shape[1])

    info = {
        "method": "scvi_scarches",
        "model_path": str(model_path),
        "n_query": query_copy.n_obs,
        "n_genes_reference": n_ref_genes,
        "latent_dim": embeddings.shape[1],
        "surgery_epochs": surgery_epochs,
    }

    return embeddings.astype(np.float32), info


def map_to_dual_reference_model_based(
    query_adata: Any,
    hlca_model_path: Path | None,
    luca_model_path: Path | None,
    *,
    fallback_hlca_h5ad: Path | None = None,
    fallback_luca_h5ad: Path | None = None,
    batch_size: int = 10000,
) -> dict[str, Any]:
    """Map query to both HLCA and LuCA using pretrained models.

    Falls back to k-NN projection only if models unavailable.

    Parameters
    ----------
    query_adata : AnnData
        Query data
    hlca_model_path : Path, optional
        Path to HLCA scANVI model directory
    luca_model_path : Path, optional
        Path to LuCA scVI model directory
    fallback_hlca_h5ad : Path, optional
        Fallback HLCA h5ad for k-NN if model fails
    fallback_luca_h5ad : Path, optional
        Fallback LuCA h5ad for k-NN if model fails
    batch_size : int
        Batch size for model encoding

    Returns
    -------
    dict with keys:
        - hlca_embeddings: np.ndarray or None
        - luca_embeddings: np.ndarray or None
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
    if hlca_model_path is not None and hlca_model_path.exists():
        log.info("=== Mapping to HLCA (model-based) ===")
        try:
            hlca_emb, hlca_info = map_query_with_scanvi_model(
                query_adata,
                hlca_model_path,
                batch_size=batch_size,
            )
            results["hlca_embeddings"] = hlca_emb
            results["metadata"]["hlca"] = hlca_info
            log.info("HLCA mapping complete: %s", hlca_emb.shape)
        except Exception as e:
            log.error("HLCA model-based mapping failed: %s", e)
            if fallback_hlca_h5ad is not None and fallback_hlca_h5ad.exists():
                log.warning("Falling back to k-NN projection for HLCA")
                results["metadata"]["hlca"] = {"method": "knn_fallback", "reason": str(e)}
                # Import fallback here to avoid circular dependency
                from stagebridge.reference.map_query_chunked import map_query_chunked
                hlca_emb, hlca_dist, hlca_info = map_query_chunked(
                    query_adata, fallback_hlca_h5ad,
                    latent_key="X_scanvi_emb", k_neighbors=50, use_faiss=True,
                )
                results["hlca_embeddings"] = hlca_emb
                results["hlca_distances"] = hlca_dist
                results["metadata"]["hlca"].update(hlca_info)
            else:
                log.error("No fallback available for HLCA")

    # Map to LuCA (also uses scANVI)
    if luca_model_path is not None and luca_model_path.exists():
        log.info("=== Mapping to LuCA (model-based scANVI) ===")
        try:
            luca_emb, luca_info = map_query_with_scanvi_model(
                query_adata,
                luca_model_path,
                batch_size=batch_size,
            )
            results["luca_embeddings"] = luca_emb
            results["metadata"]["luca"] = luca_info
            log.info("LuCA mapping complete: %s", luca_emb.shape)
        except Exception as e:
            log.error("LuCA model-based mapping failed: %s", e)
            if fallback_luca_h5ad is not None and fallback_luca_h5ad.exists():
                log.warning("Falling back to k-NN projection for LuCA")
                results["metadata"]["luca"] = {"method": "knn_fallback", "reason": str(e)}
                from stagebridge.reference.map_query_chunked import map_query_chunked
                luca_emb, luca_dist, luca_info = map_query_chunked(
                    query_adata, fallback_luca_h5ad,
                    latent_key="X_scVI", k_neighbors=50, use_faiss=True,
                )
                results["luca_embeddings"] = luca_emb
                results["luca_distances"] = luca_dist
                results["metadata"]["luca"].update(luca_info)
            else:
                log.error("No fallback available for LuCA")

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
