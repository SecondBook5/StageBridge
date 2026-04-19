"""CNV inference from spatial transcriptomics data.

Uses infercnvpy to infer copy number variations from spatial spots,
treating each spot as a pseudo-single-cell observation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


def run_cnv_inference(
    adata: AnnData,
    reference_key: str = "cell_type",
    reference_cat: str | list[str] = "AT2",
    gene_order_file: str | Path | None = None,
    window_size: int = 100,
    step: int = 10,
    method: Literal["infercnvpy", "copykat"] = "infercnvpy",
    use_raw: bool = True,
    layer: str | None = None,
    patient_key: str | None = None,
    cluster_resolution: float = 0.5,
) -> AnnData:
    """Run CNV inference on spatial/single-cell data.

    Parameters
    ----------
    adata
        AnnData object with gene expression. Should be filtered to
        epithelial cells/spots only for best results.
    reference_key
        Column in adata.obs containing cell type annotations.
    reference_cat
        Cell type(s) to use as reference (diploid baseline).
        Typically normal epithelial cells (AT2, AT1).
    gene_order_file
        Path to gene order file (chromosome positions).
        If None, uses default from infercnvpy.
    window_size
        Window size for smoothing CNV signal.
    step
        Step size for sliding window.
    method
        CNV inference method: "infercnvpy" or "copykat"
    use_raw
        Whether to use adata.raw for counts.
    layer
        Layer to use instead of X. Overrides use_raw.
    patient_key
        Column with patient IDs. If provided, leiden clustering is done
        per-patient (required for correct clone identification).
    cluster_resolution
        Leiden clustering resolution (default: 0.5).

    Returns
    -------
    AnnData with CNV scores in:
        - adata.obsm["X_cnv"]: CNV matrix (cells x genomic windows)
        - adata.obs["cnv_score"]: Per-cell aneuploidy score
        - adata.uns["cnv"]: CNV metadata (chromosome positions, etc.)
    """
    try:
        import infercnvpy as cnv
    except ImportError:
        raise ImportError(
            "infercnvpy is required for CNV inference. "
            "Install with: pip install infercnvpy"
        )

    logger.info(f"Running CNV inference with {method} on {adata.n_obs} cells/spots")

    # Work on a copy
    adata = adata.copy()

    # Ensure we have counts
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    elif use_raw and adata.raw is not None:
        adata = adata.raw.to_adata()

    # Normalize if not already
    if adata.X.max() > 100:  # Likely counts, not normalized
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Setup reference cells
    if isinstance(reference_cat, str):
        reference_cat = [reference_cat]

    if reference_key not in adata.obs.columns:
        raise ValueError(f"Reference key '{reference_key}' not found in adata.obs")

    # Check reference cells exist
    ref_mask = adata.obs[reference_key].isin(reference_cat)
    n_ref = ref_mask.sum()
    if n_ref < 10:
        logger.warning(
            f"Only {n_ref} reference cells found. "
            f"Consider using a different reference category."
        )

    # Add reference annotation
    adata.obs["cnv_reference"] = ref_mask

    # Get gene chromosome positions
    # First check if positions already exist in adata.var
    has_positions = all(col in adata.var.columns for col in ["chromosome", "start", "end"])

    if has_positions:
        logger.info("Using existing genomic positions from adata.var")
    elif gene_order_file is not None:
        # Use provided GTF file
        cnv.io.genomic_position_from_gtf(
            gtf_file=str(gene_order_file),
            adata=adata,
            inplace=True,
        )
    else:
        # Try biomart (fetches from Ensembl online, works with gene symbols)
        logger.info("Fetching genomic positions from Ensembl Biomart...")
        try:
            cnv.io.genomic_position_from_biomart(
                adata=adata,
                biomart_gene_id="hgnc_symbol",  # Use gene symbols
                inplace=True,
            )
        except Exception as e:
            logger.warning(f"Biomart failed ({e}), trying with ensembl_gene_id...")
            # Maybe genes are ENSG IDs
            try:
                cnv.io.genomic_position_from_biomart(
                    adata=adata,
                    biomart_gene_id="ensembl_gene_id",
                    inplace=True,
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Could not get genomic positions. Provide a GTF file via gene_order_file. "
                    f"Errors: biomart(hgnc)={e}, biomart(ensg)={e2}"
                )

    # Filter to genes with position info
    adata = adata[:, adata.var["chromosome"].notna()].copy()
    logger.info(f"Using {adata.n_vars} genes with genomic positions")

    # Run CNV inference
    cnv.tl.infercnv(
        adata,
        reference_key="cnv_reference",
        reference_cat=[True],
        window_size=window_size,
        step=step,
        dynamic_threshold=1.5,
    )

    # Cluster by CNV profile to identify clones
    # Must be done PER PATIENT - clones are patient-specific
    if patient_key is not None and patient_key in adata.obs.columns:
        logger.info(f"Clustering CNV profiles per patient (key={patient_key})")
        adata.obs["cnv_leiden"] = ""

        for patient_id in adata.obs[patient_key].unique():
            patient_mask = adata.obs[patient_key] == patient_id
            n_cells = patient_mask.sum()

            if n_cells < 20:
                logger.warning(f"Patient {patient_id}: Only {n_cells} cells, assigning single clone")
                adata.obs.loc[patient_mask, "cnv_leiden"] = "0"
                continue

            # Subset to patient
            patient_idx = np.where(patient_mask)[0]
            adata_patient = adata[patient_mask].copy()

            # PCA on CNV profiles for this patient
            cnv.tl.pca(adata_patient)
            cnv.pp.neighbors(adata_patient)
            cnv.tl.leiden(adata_patient, resolution=cluster_resolution, key_added="cnv_leiden")

            # Copy back with patient-prefixed clone IDs
            clone_labels = adata_patient.obs["cnv_leiden"].values
            adata.obs.loc[patient_mask, "cnv_leiden"] = [
                f"{patient_id}_{c}" for c in clone_labels
            ]

            n_clones = adata_patient.obs["cnv_leiden"].nunique()
            logger.info(f"Patient {patient_id}: {n_cells} cells, {n_clones} clones")

        adata.obs["cnv_leiden"] = adata.obs["cnv_leiden"].astype("category")
    else:
        # Global clustering (legacy behavior - not recommended)
        logger.warning("No patient_key provided - using global clustering (not recommended)")
        cnv.tl.pca(adata)
        cnv.pp.neighbors(adata)
        cnv.tl.leiden(adata, resolution=cluster_resolution, key_added="cnv_leiden")

    # Compute per-cell CNV score (aneuploidy)
    cnv.tl.cnv_score(adata)

    logger.info(
        f"CNV inference complete. Found {adata.obs['cnv_leiden'].nunique()} CNV clusters."
    )

    return adata


def compute_clone_cnv_profiles(
    adata: AnnData,
    clone_key: str = "cnv_leiden",
) -> pd.DataFrame:
    """Compute average CNV profile per clone.

    Parameters
    ----------
    adata
        AnnData with CNV results from run_cnv_inference.
    clone_key
        Column in adata.obs with clone assignments.

    Returns
    -------
    DataFrame with clone CNV profiles (clones x genomic windows).
    """
    if "X_cnv" not in adata.obsm:
        raise ValueError("CNV not computed. Run run_cnv_inference first.")

    cnv_matrix = adata.obsm["X_cnv"]
    clones = adata.obs[clone_key].unique()

    profiles = {}
    for clone in clones:
        mask = adata.obs[clone_key] == clone
        profiles[clone] = cnv_matrix[mask].mean(axis=0)

    return pd.DataFrame(profiles).T


def compute_clone_distance_matrix(
    clone_profiles: pd.DataFrame,
    metric: Literal["euclidean", "correlation"] = "correlation",
) -> pd.DataFrame:
    """Compute pairwise distances between clone CNV profiles.

    Parameters
    ----------
    clone_profiles
        DataFrame from compute_clone_cnv_profiles.
    metric
        Distance metric to use.

    Returns
    -------
    Pairwise distance matrix (clones x clones).
    """
    from scipy.spatial.distance import pdist, squareform

    if metric == "correlation":
        # Use 1 - correlation as distance
        distances = pdist(clone_profiles.values, metric="correlation")
    else:
        distances = pdist(clone_profiles.values, metric=metric)

    dist_matrix = squareform(distances)
    return pd.DataFrame(
        dist_matrix,
        index=clone_profiles.index,
        columns=clone_profiles.index,
    )
