"""
Gene-space harmonisation and dimensionality reduction.

Delegates to scanpy (sc.pp.*) wherever possible.  scanpy's implementations
are well-tested against the Seurat/Bioconductor ecosystems and handle sparse
matrices correctly.

Convention for layer/embedding storage
---------------------------------------
Raw counts    → adata.layers["counts"]     (set at load time, never overwritten)
log1p-norm    → adata.layers["log1p"]      (set by normalize_log1p)
PCA latents   → adata.obsm["X_pca"]        (set by pca_fit_transform_snrna /
                                                     pca_transform_spatial)
UMAP coords   → adata.obsm["X_umap"]       (optional, set by run_umap)
Harmony emb.  → adata.obsm["X_pca_harmony"] (set by run_harmony)

All functions that add embeddings operate in-place on the passed AnnData.
Functions that return a new object (intersect_genes, select_hvg) are documented
as such.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import anndata

from stagebridge.logging_utils import get_logger
from stagebridge.preprocessing.stage_ontology import apply_stage_ontology

log = get_logger(__name__)

if TYPE_CHECKING:
    from sklearn.decomposition import TruncatedSVD


def _require_scanpy():
    """Import scanpy lazily so scanpy issues do not break unrelated workflows."""
    try:
        import scanpy as sc  # type: ignore
    except Exception as exc:
        raise ImportError(
            "scanpy is required for this preprocessing step but could not be imported. "
            "Install/repair scanpy in the runtime environment."
        ) from exc
    return sc


# ---------------------------------------------------------------------------
# Gene intersection
# ---------------------------------------------------------------------------

def intersect_genes(
    adata_a: anndata.AnnData,
    adata_b: anndata.AnnData,
) -> tuple[anndata.AnnData, anndata.AnnData]:
    """Subset both AnnData objects to their shared gene set.

    Parameters
    ----------
    adata_a, adata_b : AnnData
        Two AnnData objects.  Both are subset to the intersection of
        ``var_names``, sorted alphabetically.

    Returns
    -------
    (adata_a_sub, adata_b_sub)
        Copies (not views) subset to the common gene set.
    """
    genes_a = set(adata_a.var_names)
    genes_b = set(adata_b.var_names)
    common  = sorted(genes_a & genes_b)

    if not common:
        raise ValueError(
            "No common genes between the two AnnData objects.\n"
            f"  A has {len(genes_a)} genes (e.g. {list(genes_a)[:5]})\n"
            f"  B has {len(genes_b)} genes (e.g. {list(genes_b)[:5]})\n"
            "Check that both datasets use the same gene symbol convention "
            "(HGNC symbols expected, e.g. TP53, BRCA1)."
        )

    log.info(
        "Gene intersection: %d common  |  only-A: %d  |  only-B: %d",
        len(common),
        len(genes_a - genes_b),
        len(genes_b - genes_a),
    )
    return adata_a[:, common].copy(), adata_b[:, common].copy()


# ---------------------------------------------------------------------------
# Normalisation — delegates to scanpy
# ---------------------------------------------------------------------------

def normalize_log1p(
    adata: anndata.AnnData,
    target_sum: float = 1e4,
    layer_in: str | None = None,
    layer_out: str = "log1p",
) -> None:
    """Depth-normalise then log1p-transform counts via scanpy.

    Uses ``sc.pp.normalize_total`` + ``sc.pp.log1p`` on a working copy so
    that the source layer (raw counts) is never modified.

    Parameters
    ----------
    adata : AnnData
        Modified in place (layer added; X is not changed).
    target_sum : float
        Library-size normalisation target (default 10 000 = CPM-like).
    layer_in : str or None
        Source layer.  ``None`` → use ``adata.X``.
    layer_out : str
        Destination layer name (default ``"log1p"``).
    """
    if layer_in is not None and layer_in not in adata.layers:
        raise KeyError(
            f"layer_in='{layer_in}' not found in adata.layers.\n"
            f"Available layers: {list(adata.layers.keys())}"
        )

    # Work on a lightweight view so we don't modify the source
    tmp = anndata.AnnData(
        X=(adata.layers[layer_in] if layer_in is not None else adata.X).copy(),
        obs=adata.obs,
        var=adata.var,
    )

    sc = _require_scanpy()
    sc.pp.normalize_total(tmp, target_sum=target_sum)
    sc.pp.log1p(tmp)

    adata.layers[layer_out] = tmp.X
    log.info(
        "normalize_log1p → adata.layers['%s']  (target_sum=%g, via scanpy)",
        layer_out,
        target_sum,
    )


# ---------------------------------------------------------------------------
# HVG selection — delegates to scanpy
# ---------------------------------------------------------------------------

def select_hvg(
    adata_snrna: anndata.AnnData,
    n_hvg: int = 2000,
    layer: str = "log1p",
    flavor: str = "seurat_v3",
) -> list[str]:
    """Select highly variable genes using scanpy.

    Parameters
    ----------
    adata_snrna : AnnData
        snRNA-seq dataset after normalize_log1p.
    n_hvg : int
        Number of HVGs to select.
    layer : str
        Layer to compute mean/variance from (default ``"log1p"``).
        For ``flavor="seurat_v3"`` this should be the raw counts layer.
    flavor : str
        Passed to ``sc.pp.highly_variable_genes``.
        ``"seurat_v3"`` (default) uses raw counts; pass ``layer="counts"``
        in that case.
        ``"seurat"`` / ``"cell_ranger"`` operate on log-normalised data.

    Returns
    -------
    list[str]
        Gene names of the selected HVGs.

    Notes
    -----
    seurat_v3 flavor expects raw (un-normalised) counts.  If you pass
    ``layer="log1p"`` with ``flavor="seurat_v3"`` a warning is emitted.
    """
    if layer not in adata_snrna.layers:
        raise KeyError(
            f"Layer '{layer}' not found.  Run normalize_log1p() first.\n"
            f"Available layers: {list(adata_snrna.layers.keys())}"
        )

    if flavor == "seurat_v3" and layer != "counts":
        log.warning(
            "flavor='seurat_v3' expects raw counts but layer='%s' was passed. "
            "Consider using layer='counts' for seurat_v3.",
            layer,
        )

    tmp = anndata.AnnData(
        X=adata_snrna.layers[layer].copy(),
        obs=adata_snrna.obs,
        var=adata_snrna.var.copy(),
    )

    sc = _require_scanpy()
    sc.pp.highly_variable_genes(
        tmp,
        n_top_genes=min(n_hvg, adata_snrna.n_vars),
        flavor=flavor,
        subset=False,
    )

    hvg_names = list(tmp.var_names[tmp.var["highly_variable"]])
    log.info(
        "select_hvg: %d HVGs selected from %d genes  (flavor=%s)",
        len(hvg_names),
        adata_snrna.n_vars,
        flavor,
    )
    return hvg_names


# ---------------------------------------------------------------------------
# PCA — fit on snRNA, project spatial
# ---------------------------------------------------------------------------

def pca_fit_transform_snrna(
    adata: anndata.AnnData,
    n_components: int = 64,
    use_layer: str = "log1p",
) -> "TruncatedSVD":
    """Fit PCA on snRNA data via scanpy and store latent in obsm["X_pca"].

    Delegates to ``sc.pp.pca`` (which uses TruncatedSVD internally — sparse-
    safe, no dense materialisation).  Also returns the raw sklearn model so
    spatial data can be projected into the same space.

    Parameters
    ----------
    adata : AnnData
        Modified in place.
    n_components : int
        Number of PCA components.
    use_layer : str
        Layer to use as input (default ``"log1p"``).

    Returns
    -------
    TruncatedSVD
        Fitted sklearn model stored in ``adata.uns["pca"]["PCs"]`` (wrapped).
    """
    if use_layer not in adata.layers:
        raise KeyError(
            f"Layer '{use_layer}' not found.  Run normalize_log1p() first.\n"
            f"Available layers: {list(adata.layers.keys())}"
        )

    n_components = min(n_components, min(adata.shape) - 1)
    log.info(
        "Fitting PCA: %d components on snRNA (shape %s, layer='%s') via scanpy ...",
        n_components,
        adata.shape,
        use_layer,
    )

    sc = _require_scanpy()
    # sc.pp.pca operates on adata.X by default; we swap in our layer temporarily.
    saved_X = adata.X
    adata.X = adata.layers[use_layer]
    try:
        sc.pp.pca(adata, n_comps=n_components, svd_solver="arpack", use_highly_variable=False)
    finally:
        adata.X = saved_X  # always restore

    # adata.obsm["X_pca"] is now set by scanpy.
    # Build a TruncatedSVD wrapper so we can project the spatial data.
    from sklearn.decomposition import TruncatedSVD
    pca_model = TruncatedSVD(n_components=n_components)
    # Populate the model components from the scanpy PCA result
    pca_model.components_ = adata.varm["PCs"].T          # (n_components, n_genes)
    pca_model.explained_variance_ratio_ = adata.uns["pca"]["variance_ratio"]

    cumvar = float(pca_model.explained_variance_ratio_.sum())
    log.info(
        "PCA done.  obsm['X_pca'] shape: %s  |  cumulative var explained: %.2f%%",
        adata.obsm["X_pca"].shape,
        100 * cumvar,
    )
    return pca_model


def pca_transform_spatial(
    adata_spatial: anndata.AnnData,
    pca_model: "TruncatedSVD",
    use_layer: str = "log1p",
) -> None:
    """Project spatial data into the snRNA PCA space.

    Parameters
    ----------
    adata_spatial : AnnData
        Modified in place; ``obsm["X_pca"]`` is set.
    pca_model : TruncatedSVD
        Fitted model returned by :func:`pca_fit_transform_snrna`.
    use_layer : str
        Layer to use (default ``"log1p"``).  Must match what was used for snRNA.
    """
    if use_layer not in adata_spatial.layers:
        raise KeyError(
            f"Layer '{use_layer}' not found in spatial AnnData.\n"
            f"Run normalize_log1p() on the spatial object first.\n"
            f"Available layers: {list(adata_spatial.layers.keys())}"
        )

    expected = pca_model.components_.shape[1]
    actual   = adata_spatial.n_vars
    if actual != expected:
        raise ValueError(
            f"Feature mismatch: spatial AnnData has {actual} genes but PCA "
            f"model was fit on {expected} genes.\n"
            f"Run intersect_genes() and subset to HVGs before calling this."
        )

    X = adata_spatial.layers[use_layer]
    adata_spatial.obsm["X_pca"] = pca_model.transform(X).astype(np.float32)
    log.info(
        "Spatial PCA projection done.  obsm['X_pca'] shape: %s",
        adata_spatial.obsm["X_pca"].shape,
    )


# ---------------------------------------------------------------------------
# Harmony batch correction
# ---------------------------------------------------------------------------

def run_harmony(
    adata: anndata.AnnData,
    batch_key: str = "patient_id",
    basis: str = "X_pca",
    out_key: str = "X_pca_harmony",
    **harmony_kwargs,
) -> None:
    """Run Harmony batch correction on PCA embeddings.

    Requires ``harmonypy``.  The corrected embedding is stored in
    ``adata.obsm[out_key]``.

    Parameters
    ----------
    adata : AnnData
        Must have ``obsm[basis]`` and ``obs[batch_key]``.
    batch_key : str
        Column in ``adata.obs`` identifying batches (default ``"patient_id"``).
    basis : str
        obsm key for input embedding (default ``"X_pca"``).
    out_key : str
        obsm key for the corrected output (default ``"X_pca_harmony"``).
    **harmony_kwargs
        Forwarded to ``harmonypy.run_harmony()``.
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError(
            "harmonypy is required for run_harmony().\n"
            "Install with:  pip install harmonypy"
        )

    if basis not in adata.obsm:
        raise KeyError(
            f"obsm key '{basis}' not found.  Run pca_fit_transform_snrna() first.\n"
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )
    if batch_key not in adata.obs.columns:
        raise KeyError(
            f"Batch key '{batch_key}' not in adata.obs.\n"
            f"Available columns: {list(adata.obs.columns)}"
        )

    log.info(
        "Running Harmony (batch_key='%s', n_batches=%d) ...",
        batch_key,
        adata.obs[batch_key].nunique(),
    )
    harmony_out = harmonypy.run_harmony(
        adata.obsm[basis],
        adata.obs,
        batch_key,
        **harmony_kwargs,
    )
    adata.obsm[out_key] = harmony_out.Z_corr.T.astype(np.float32)
    log.info(
        "Harmony done.  obsm['%s'] shape: %s",
        out_key,
        adata.obsm[out_key].shape,
    )


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(
    adata: anndata.AnnData,
    basis: str = "X_pca_harmony",
    n_neighbors: int = 15,
    n_components: int = 2,
    **umap_kwargs,
) -> None:
    """Compute UMAP on top of the corrected PCA embedding via scanpy.

    Parameters
    ----------
    adata : AnnData
        Modified in place; ``obsm["X_umap"]`` and ``obsp["connectivities"]``
        are set.
    basis : str
        obsm key to use as input (default ``"X_pca_harmony"``; fall back to
        ``"X_pca"`` if not found).
    n_neighbors : int
        Number of nearest neighbours for the kNN graph.
    n_components : int
        UMAP output dimensions (default 2).
    """
    if basis not in adata.obsm:
        fallback = "X_pca"
        log.warning(
            "obsm key '%s' not found; falling back to '%s'.", basis, fallback
        )
        basis = fallback
        if basis not in adata.obsm:
            raise KeyError(
                f"Neither '{basis}' nor 'X_pca' found in adata.obsm.\n"
                f"Run pca_fit_transform_snrna() (and optionally run_harmony()) first."
            )

    sc = _require_scanpy()
    log.info("Computing kNN graph (n_neighbors=%d, basis='%s') ...", n_neighbors, basis)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=basis)
    log.info("Running UMAP (n_components=%d) ...", n_components)
    sc.tl.umap(adata, n_components=n_components, **umap_kwargs)
    log.info("UMAP done.  obsm['X_umap'] shape: %s", adata.obsm["X_umap"].shape)


# ---------------------------------------------------------------------------
# HLCA alignment and metadata requirements
# ---------------------------------------------------------------------------

def canonicalize_gene_symbols(adata: anndata.AnnData) -> None:
    """Canonicalize gene symbols in ``adata.var_names`` in place."""
    def _canon(symbol: str) -> str:
        s = str(symbol).strip()
        s = re.sub(r"\s+", "", s)
        s = s.replace("_", "-")
        return s.upper()

    adata.var_names = pd.Index([_canon(g) for g in adata.var_names], name=adata.var_names.name)
    log.info("Canonicalized gene symbols for %d genes.", adata.n_vars)


def ensure_required_obs_fields(
    adata: anndata.AnnData,
    required_fields: tuple[str, ...] = ("stage", "patient_id", "sample_id", "modality"),
) -> None:
    """Ensure required metadata fields are present in ``adata.obs``."""
    for field in required_fields:
        if field not in adata.obs.columns:
            default = "unknown_modality" if field == "modality" else f"unknown_{field}"
            adata.obs[field] = default
    adata.obs = apply_stage_ontology(adata.obs, stage_col="stage", stage_index_col="stage_index")


def add_hlca_latent(
    adata: anndata.AnnData,
    reference: object | None = None,
    source_key: str = "X_pca",
    output_key: str = "X_hlca",
    n_components: int = 64,
) -> None:
    """Add HLCA-aligned latent embedding under ``obsm[output_key]``."""
    from stagebridge.io.hlca import map_to_hlca_latent

    map_to_hlca_latent(
        adata=adata,
        reference=reference,
        input_key=source_key,
        output_key=output_key,
        n_components=n_components,
    )
    log.info("Added HLCA-aligned latent embedding in obsm['%s']", output_key)


def align_modalities_in_latent(
    adata_snrna: anndata.AnnData,
    adata_spatial: anndata.AnnData,
    latent_key: str = "X_hlca",
) -> tuple[anndata.AnnData, anndata.AnnData]:
    """Match snRNA and spatial modalities in a shared latent space."""
    if latent_key not in adata_snrna.obsm:
        raise KeyError(f"snRNA missing obsm['{latent_key}']")
    if latent_key not in adata_spatial.obsm:
        raise KeyError(f"spatial missing obsm['{latent_key}']")

    X_rna = np.asarray(adata_snrna.obsm[latent_key], dtype=np.float32)
    X_sp = np.asarray(adata_spatial.obsm[latent_key], dtype=np.float32)
    n_dim = min(X_rna.shape[1], X_sp.shape[1])
    X_rna = X_rna[:, :n_dim]
    X_sp = X_sp[:, :n_dim]

    mu = X_rna.mean(axis=0, keepdims=True)
    sd = X_rna.std(axis=0, keepdims=True) + 1e-6
    adata_snrna.obsm[latent_key] = ((X_rna - mu) / sd).astype(np.float32)
    adata_spatial.obsm[latent_key] = ((X_sp - mu) / sd).astype(np.float32)

    ensure_required_obs_fields(adata_snrna)
    ensure_required_obs_fields(adata_spatial)
    adata_snrna.obs["modality"] = "snrna"
    adata_spatial.obs["modality"] = "spatial"

    log.info(
        "Aligned modalities in latent space '%s' with dimensionality %d.",
        latent_key,
        n_dim,
    )
    return adata_snrna, adata_spatial
