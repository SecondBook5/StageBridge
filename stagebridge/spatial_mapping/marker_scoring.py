"""
Marker gene signature scoring for spatial deconvolution.

This provides a reference-free baseline that doesn't require matched scRNA-seq.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import norm
from typing import Dict, List, Literal


def score_markers_scanpy(
    spatial_adata,
    marker_dict: dict[str, list[str]],
    use_raw: bool = False,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Score marker genes using Scanpy's score_genes function.

    Args:
        spatial_adata: AnnData with spatial expression
        marker_dict: Dict mapping cell type name to list of marker genes
        use_raw: Whether to use .raw attribute
        normalize: Whether to normalize scores to [0, 1] per cell type

    Returns:
        DataFrame of cell type proportions (n_spots, n_celltypes)
    """
    adata = spatial_adata.copy()

    # Score each cell type
    scores = {}
    for cell_type, markers in marker_dict.items():
        # Filter to available genes
        available_markers = [g for g in markers if g in adata.var_names]

        if len(available_markers) == 0:
            # No markers available - assign zero
            scores[cell_type] = np.zeros(adata.n_obs)
        else:
            # Score using Scanpy
            sc.tl.score_genes(
                adata,
                gene_list=available_markers,
                score_name=f"score_{cell_type}",
                use_raw=use_raw,
            )
            scores[cell_type] = adata.obs[f"score_{cell_type}"].values

    # Convert to DataFrame
    scores_df = pd.DataFrame(scores, index=adata.obs_names)

    # Normalize to proportions
    if normalize:
        # Shift to positive (min=0)
        scores_df = scores_df - scores_df.min().min()

        # Convert to proportions (sum to 1 per spot)
        row_sums = scores_df.sum(axis=1)
        scores_df = scores_df.div(row_sums, axis=0)

        # Handle zero sums (all markers missing)
        scores_df = scores_df.fillna(1.0 / len(marker_dict))

    return scores_df


def score_markers_bootstrap(
    spatial_adata,
    marker_dict: dict[str, list[str]],
    n_bootstrap: int = 10000,
    presence_threshold_pval: float = 0.05,
    return_proportions: bool = True,
) -> pd.DataFrame:
    """
    Score markers with bootstrapping for empirical p-values.


    Args:
        spatial_adata: AnnData with spatial expression
        marker_dict: Dict mapping cell type to marker genes
        n_bootstrap: Number of bootstrap samples for background
        presence_threshold_pval: P-value threshold for calling presence
        return_proportions: If True, return proportions; else return p-values

    Returns:
        DataFrame of proportions or p-values (n_spots, n_celltypes)
    """
    adata = spatial_adata.copy()

    # Ensure normalized
    if adata.X.max() > 100:  # Likely counts
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    X = adata.X if not hasattr(adata, "layers") else adata.layers.get("normalized", adata.X)
    if hasattr(X, "toarray"):
        X = X.toarray()

    results = {}

    for cell_type, markers in marker_dict.items():
        # Filter to available markers
        available_markers = [g for g in markers if g in adata.var_names]

        if len(available_markers) == 0:
            # No markers - assign high p-value (no evidence)
            results[cell_type] = np.ones(adata.n_obs)
            continue

        marker_indices = [list(adata.var_names).index(g) for g in available_markers]
        marker_expr = X[:, marker_indices]

        # Compute enrichment score per spot
        enrichment_scores = marker_expr.mean(axis=1)

        # Generate bootstrap background distribution
        # Sample random gene sets of same size
        background_scores = []
        n_genes = len(marker_indices)

        for _ in range(n_bootstrap):
            random_indices = np.random.choice(X.shape[1], size=n_genes, replace=False)
            random_expr = X[:, random_indices].mean(axis=1)
            background_scores.append(random_expr)

        background_scores = np.array(background_scores)

        # Compute empirical p-values
        # P-value = fraction of background scores >= observed score
        p_values = []
        for i, obs_score in enumerate(enrichment_scores):
            # Background for this spot
            spot_background = background_scores[:, i]
            p_val = (spot_background >= obs_score).sum() / n_bootstrap
            p_values.append(p_val)

        results[cell_type] = np.array(p_values)

    pval_df = pd.DataFrame(results, index=adata.obs_names)

    if return_proportions:
        # Convert p-values to presence scores
        # Low p-value = high enrichment = high score
        presence_scores = 1.0 - pval_df

        # Threshold by p-value
        presence_binary = (pval_df < presence_threshold_pval).astype(float)

        # Weight by enrichment strength
        proportions = presence_scores * presence_binary

        # Normalize to sum to 1
        row_sums = proportions.sum(axis=1)
        proportions = proportions.div(row_sums, axis=0)
        proportions = proportions.fillna(1.0 / len(marker_dict))

        return proportions
    else:
        return pval_df


def get_markers_from_reference(
    reference_adata,
    cell_type_key: str = "cell_type",
    n_markers: int = 50,
    method: Literal["wilcoxon", "t-test"] = "wilcoxon",
    log_fc_threshold: float = 0.5,
) -> dict[str, list[str]]:
    """
    Extract top marker genes from reference scRNA-seq data.

    Args:
        reference_adata: Reference AnnData with cell type annotations
        cell_type_key: Column in .obs with cell type labels
        n_markers: Number of top markers to keep per cell type
        method: Statistical test method
        log_fc_threshold: Minimum log fold change

    Returns:
        Dictionary mapping cell type to marker gene list
    """
    adata = reference_adata.copy()

    # Find markers
    sc.tl.rank_genes_groups(
        adata,
        groupby=cell_type_key,
        method=method,
        use_raw=False,
    )

    # Extract top markers per cell type
    marker_dict = {}
    for cell_type in adata.obs[cell_type_key].unique():
        # Get markers for this cell type
        markers = sc.get.rank_genes_groups_df(
            adata,
            group=cell_type,
        )

        # Filter by log fold change
        markers = markers[markers["logfoldchanges"] >= log_fc_threshold]

        # Take top n
        top_markers = markers.head(n_markers)["names"].tolist()

        marker_dict[cell_type] = top_markers

    return marker_dict


def get_markers_from_database(
    cell_types: list[str],
    database: Literal["panglaodb", "cellmarker", "custom"] = "panglaodb",
    species: str = "human",
    tissue: str = "lung",
) -> dict[str, list[str]]:
    """
    Get marker genes from public databases.

    Args:
        cell_types: List of cell types to get markers for
        database: Which marker database to use
        species: "human" or "mouse"
        tissue: Tissue type for filtering

    Returns:
        Dictionary mapping cell type to marker gene list
    """
    # This is a placeholder - in practice you would:
    # 1. Query PanglaoDB API or download their table
    # 2. Filter by species and tissue
    # 3. Map cell type names to database names
    # 4. Return top N markers per cell type

    # Example structure for lung cell types:
    marker_dict = {
        "Alveolar_epithelial_type_1": ["AGER", "PDPN", "CAV1", "HOPX"],
        "Alveolar_epithelial_type_2": ["SFTPC", "SFTPB", "LAMP3", "ABCA3"],
        "Club_cells": ["SCGB1A1", "SCGB3A2", "CYP2F1"],
        "Ciliated_cells": ["FOXJ1", "TUBA1A", "DNAH5"],
        "Macrophages": ["CD68", "CD163", "C1QA", "C1QB"],
        "T_cells": ["CD3D", "CD3E", "CD3G"],
        "B_cells": ["CD79A", "CD79B", "MS4A1"],
        "Endothelial": ["PECAM1", "VWF", "CDH5"],
        "Fibroblasts": ["COL1A1", "DCN", "LUM"],
    }

    # Filter to requested cell types
    filtered = {ct: markers for ct, markers in marker_dict.items() if ct in cell_types}

    return filtered


class MarkerScoringBackend:
    """
    Wrapper to use marker gene scoring as a deconvolution backend.

    This allows direct comparison with other backends in the benchmark.
    """

    def __init__(
        self,
        marker_dict: dict[str, list[str]] | None = None,
        method: Literal["scanpy", "bootstrap"] = "scanpy",
        **kwargs
    ):
        """
        Initialize marker scoring backend.

        Args:
            marker_dict: Optional pre-defined markers
            method: Scoring method ("scanpy" or "bootstrap")
            **kwargs: Additional arguments for scoring function
        """
        self.marker_dict = marker_dict
        self.method = method
        self.kwargs = kwargs

    def run(
        self,
        spatial_adata,
        reference_adata=None,
        **run_kwargs
    ) -> pd.DataFrame:
        """
        Run marker gene scoring.

        Args:
            spatial_adata: Spatial AnnData
            reference_adata: Optional reference (to extract markers if not provided)
            **run_kwargs: Additional runtime arguments

        Returns:
            DataFrame of cell type proportions
        """
        # Get markers if not provided
        if self.marker_dict is None:
            if reference_adata is None:
                raise ValueError("Must provide either marker_dict or reference_adata")

            self.marker_dict = get_markers_from_reference(
                reference_adata,
                **self.kwargs.get("marker_extraction_kwargs", {})
            )

        # Run scoring
        if self.method == "scanpy":
            proportions = score_markers_scanpy(
                spatial_adata,
                self.marker_dict,
                **self.kwargs
            )
        elif self.method == "bootstrap":
            proportions = score_markers_bootstrap(
                spatial_adata,
                self.marker_dict,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return proportions
