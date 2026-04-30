"""Biological feature computation from expression data.

Computes:
- EMT score (mesenchymal - epithelial marker expression)
- Senescence/SASP score
- LIANA ligand-receptor scores (requires liana-py)

All functions expect an AnnData with normalized log expression.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


# =============================================================================
# Gene Signatures
# =============================================================================

EMT_MESENCHYMAL = [
    "VIM", "CDH2", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2",
    "FN1", "MMP2", "MMP9", "SPARC", "COL1A1", "COL3A1", "ACTA2",
]

EMT_EPITHELIAL = [
    "CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "CLDN1", "CLDN3", "CLDN4",
    "CLDN7", "OCLN", "TJP1",
]

SENESCENCE_CORE = [
    "CDKN1A",  # p21
    "CDKN2A",  # p16
    "TP53",
    "RB1",
    "SERPINE1",  # PAI-1
    "GLB1",  # SA-beta-gal
]

SASP_GENES = [
    "IL1A", "IL1B", "IL6", "IL8", "CXCL1", "CXCL2", "CXCL8",
    "CCL2", "CCL3", "CCL5", "CCL20",
    "MMP1", "MMP3", "MMP9", "MMP10", "MMP12",
    "IGFBP3", "IGFBP7",
    "VEGFA", "FGF2",
    "SERPINE1", "SERPINE2",
]


# =============================================================================
# Score Computation
# =============================================================================

def _score_signature(adata: "ad.AnnData", genes: list[str]) -> np.ndarray:
    """Compute mean expression of signature genes."""
    gene_mask = [g in adata.var_names for g in genes]
    valid_genes = [g for g, m in zip(genes, gene_mask) if m]

    if not valid_genes:
        return np.zeros(adata.n_obs)

    X = adata[:, valid_genes].X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    return np.mean(X, axis=1)


def compute_emt_score(adata: "ad.AnnData") -> np.ndarray:
    """Compute EMT score: mesenchymal - epithelial expression.

    Positive = mesenchymal, Negative = epithelial.

    Args:
        adata: AnnData with normalized log expression

    Returns:
        EMT scores per cell
    """
    mes_score = _score_signature(adata, EMT_MESENCHYMAL)
    epi_score = _score_signature(adata, EMT_EPITHELIAL)
    return mes_score - epi_score


def compute_senescence_score(adata: "ad.AnnData") -> np.ndarray:
    """Compute senescence score from core markers.

    Args:
        adata: AnnData with normalized log expression

    Returns:
        Senescence scores per cell
    """
    return _score_signature(adata, SENESCENCE_CORE)


def compute_sasp_score(adata: "ad.AnnData") -> np.ndarray:
    """Compute SASP (senescence-associated secretory phenotype) score.

    Args:
        adata: AnnData with normalized log expression

    Returns:
        SASP scores per cell
    """
    return _score_signature(adata, SASP_GENES)


def compute_all_signatures(adata: "ad.AnnData") -> pd.DataFrame:
    """Compute all biological signature scores.

    Args:
        adata: AnnData with normalized log expression

    Returns:
        DataFrame with cell_id and signature scores
    """
    return pd.DataFrame({
        "cell_id": adata.obs_names,
        "emt_score": compute_emt_score(adata),
        "senescence_score": compute_senescence_score(adata),
        "sasp_score": compute_sasp_score(adata),
    })


# =============================================================================
# LIANA L-R Analysis
# =============================================================================

def run_liana(
    adata: "ad.AnnData",
    cell_type_col: str = "cell_type",
    resource: str = "consensus",
    expr_prop: float = 0.1,
    n_perms: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run LIANA ligand-receptor analysis.

    Args:
        adata: AnnData with normalized log expression and cell type labels
        cell_type_col: Column in adata.obs with cell type labels
        resource: L-R database (default: consensus)
        expr_prop: Minimum expression proportion
        n_perms: Number of permutations for significance
        verbose: Print progress

    Returns:
        DataFrame with L-R interactions (from adata.uns['liana_res'])
    """
    import liana as li

    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_col}' not in adata.obs")

    # Filter cells with cell type
    mask = adata.obs[cell_type_col].notna()
    if mask.sum() < adata.n_obs:
        adata = adata[mask].copy()

    if verbose:
        print(f"Running LIANA on {adata.n_obs:,} cells...")

    li.mt.rank_aggregate(
        adata,
        groupby=cell_type_col,
        resource_name=resource,
        expr_prop=expr_prop,
        use_raw=False,
        verbose=verbose,
        n_perms=n_perms,
    )

    return adata.uns['liana_res'].copy()


def extract_il1b_interactions(lr_results: pd.DataFrame) -> pd.DataFrame:
    """Extract IL1B-IL1R1 interactions from LIANA results.

    Key for validating H1.2 hypothesis about proinflammatory niches.

    Args:
        lr_results: LIANA results DataFrame

    Returns:
        Filtered DataFrame with IL1B/IL1R1 interactions
    """
    mask = (
        lr_results['ligand_complex'].str.contains('IL1B', case=False, na=False) |
        lr_results['receptor_complex'].str.contains('IL1R1', case=False, na=False)
    )
    return lr_results[mask].copy()


def run_liana_pathway_enrichment(
    adata: "ad.AnnData",
    lr_results: pd.DataFrame | None = None,
    cell_type_col: str = "cell_type",
    organism: str = "human",
) -> pd.DataFrame:
    """Run LIANA's built-in pathway enrichment (via decoupler/gseapy).

    LIANA integrates with decoupler for functional enrichment of L-R interactions.

    Args:
        adata: AnnData with LIANA results in .uns['liana_res']
        lr_results: Optional pre-computed LIANA results
        cell_type_col: Cell type column
        organism: 'human' or 'mouse'

    Returns:
        DataFrame with pathway enrichment results
    """
    import liana as li

    if lr_results is None and 'liana_res' in adata.uns:
        lr_results = adata.uns['liana_res']

    if lr_results is None:
        raise ValueError("No LIANA results provided or found in adata.uns")

    # Use LIANA's built-in functional enrichment
    # This wraps decoupler/gseapy for pathway analysis
    try:
        enrichment = li.mt.df_to_lr(
            adata,
            groupby=cell_type_col,
            resource_name="consensus",
        )
        return enrichment
    except Exception as e:
        print(f"Pathway enrichment failed: {e}")
        return pd.DataFrame()


def compute_cell_lr_scores(
    lr_results: pd.DataFrame,
    cell_types: pd.Series,
    top_n: int = 50,
) -> pd.DataFrame:
    """Compute per-cell L-R activity scores.

    Aggregates L-R interaction strengths for each cell based on its cell type.

    Args:
        lr_results: LIANA results DataFrame
        cell_types: Series mapping cell_id to cell type
        top_n: Number of top interactions to consider

    Returns:
        DataFrame with cell_id and lr_activity_score
    """
    # Get top interactions
    top_lr = lr_results.nsmallest(top_n, 'magnitude_rank')

    # Score cells by their cell type's involvement in top interactions
    cell_scores = {}

    for cell_id, cell_type in cell_types.items():
        # Count interactions where this cell type is source or target
        as_source = (top_lr['source'] == cell_type).sum()
        as_target = (top_lr['target'] == cell_type).sum()
        cell_scores[cell_id] = as_source + as_target

    return pd.DataFrame({
        "cell_id": list(cell_scores.keys()),
        "lr_activity_score": list(cell_scores.values()),
    })


# =============================================================================
# Combined Pipeline
# =============================================================================

def compute_biological_features(
    h5ad_path: Path,
    output_path: Path | None = None,
    run_liana_analysis: bool = False,
    cell_type_col: str = "cell_type",
) -> pd.DataFrame:
    """Compute all biological features from h5ad.

    Args:
        h5ad_path: Path to h5ad file with normalized expression
        output_path: Optional path to save results
        run_liana_analysis: Whether to run LIANA (slow, ~30 min)
        cell_type_col: Column for cell types (for LIANA)

    Returns:
        DataFrame with cell_id and all computed features
    """
    import scanpy as sc

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Signature scores
    print("Computing signature scores...")
    results = compute_all_signatures(adata)
    print(f"  EMT score range: [{results['emt_score'].min():.2f}, {results['emt_score'].max():.2f}]")
    print(f"  Senescence range: [{results['senescence_score'].min():.2f}, {results['senescence_score'].max():.2f}]")
    print(f"  SASP range: [{results['sasp_score'].min():.2f}, {results['sasp_score'].max():.2f}]")

    # LIANA (optional, slow)
    if run_liana_analysis and cell_type_col in adata.obs.columns:
        print("Running LIANA L-R analysis (this takes ~30 min)...")
        lr_results = run_liana(adata, cell_type_col=cell_type_col)

        # Add per-cell L-R scores
        cell_types = adata.obs[cell_type_col]
        lr_scores = compute_cell_lr_scores(lr_results, cell_types)
        results = results.merge(lr_scores, on="cell_id", how="left")

        # Save LIANA results separately
        if output_path:
            lr_out = output_path.parent / "liana_interactions.parquet"
            lr_results.to_parquet(lr_out)
            print(f"  Saved LIANA results to {lr_out}")

    if output_path:
        results.to_parquet(output_path)
        print(f"Saved features to {output_path}")

    return results
