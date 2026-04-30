"""Regulon and gene regulatory network analysis via pySCENIC.

Computes:
- TF activity scores (AUCell)
- Regulon specificity by stage/cell type
- GRN inference from expression data

Requires: pyscenic (pip install pyscenic)
Reference databases: https://resources.aertslab.org/cistarget/
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


# =============================================================================
# Database Paths (adjust for HPC)
# =============================================================================

SCENIC_DATABASES = {
    "human": {
        "tfs": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt",
        "motifs": "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
        "annotations": "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
    },
    "mouse": {
        "tfs": "https://resources.aertslab.org/cistarget/tf_lists/allTFs_mm.txt",
        "motifs": "mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
        "annotations": "motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl",
    },
}


# =============================================================================
# GRN Inference
# =============================================================================

def run_grn_inference(
    adata: "ad.AnnData",
    tf_list: list[str] | None = None,
    n_jobs: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run GRNBoost2 for gene regulatory network inference.

    Args:
        adata: AnnData with normalized expression
        tf_list: List of transcription factors (default: load from Aerts lab)
        n_jobs: Number of parallel jobs
        verbose: Print progress

    Returns:
        DataFrame with columns: TF, target, importance
    """
    from arboreto.algo import grnboost2
    from arboreto.utils import load_tf_names

    if verbose:
        print(f"Running GRNBoost2 on {adata.n_obs:,} cells x {adata.n_vars:,} genes...")

    # Get expression matrix
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    # Load TF list if not provided
    if tf_list is None:
        tf_url = SCENIC_DATABASES["human"]["tfs"]
        tf_list = load_tf_names(tf_url)

    # Filter to TFs present in data
    tf_list = [tf for tf in tf_list if tf in adata.var_names]
    if verbose:
        print(f"  Using {len(tf_list)} TFs present in data")

    # Run GRNBoost2
    adjacencies = grnboost2(
        expression_data=expr_df,
        tf_names=tf_list,
        verbose=verbose,
        client_or_address="local",
        seed=42,
    )

    if verbose:
        print(f"  Found {len(adjacencies):,} TF-target pairs")

    return adjacencies


def prune_grn_with_motifs(
    adjacencies: pd.DataFrame,
    motif_db_path: Path,
    annotation_path: Path,
    n_jobs: int = 4,
    verbose: bool = True,
) -> list:
    """Prune GRN using motif enrichment (cistarget).

    Args:
        adjacencies: GRNBoost2 output
        motif_db_path: Path to motif rankings feather file
        annotation_path: Path to motif annotations table
        n_jobs: Number of parallel jobs
        verbose: Print progress

    Returns:
        List of regulons (pySCENIC format)
    """
    from pyscenic.prune import prune2df, df2regulons
    from ctxcore.rnkdb import FeatherRankingDatabase

    if verbose:
        print("Pruning GRN with motif enrichment...")

    # Load motif database
    dbs = [FeatherRankingDatabase(motif_db_path)]

    # Run cistarget
    df_motifs = prune2df(
        dbs,
        adjacencies,
        annotation_path,
        num_workers=n_jobs,
    )

    # Convert to regulons
    regulons = df2regulons(df_motifs)

    if verbose:
        print(f"  Found {len(regulons)} regulons")

    return regulons


# =============================================================================
# AUCell Scoring
# =============================================================================

def compute_aucell(
    adata: "ad.AnnData",
    regulons: list | dict,
    n_jobs: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute AUCell regulon activity scores.

    Args:
        adata: AnnData with normalized expression
        regulons: List of regulons or dict {regulon_name: gene_list}
        n_jobs: Number of parallel jobs
        verbose: Print progress

    Returns:
        DataFrame with cell_id x regulon activity scores
    """
    from pyscenic.aucell import aucell

    if verbose:
        print(f"Computing AUCell scores for {len(regulons)} regulons...")

    # Get expression matrix
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    # Convert dict to regulon format if needed
    if isinstance(regulons, dict):
        from ctxcore.genesig import GeneSignature
        regulons = [
            GeneSignature(name=name, gene2weight=genes if isinstance(genes, dict) else {g: 1.0 for g in genes})
            for name, genes in regulons.items()
        ]

    # Run AUCell
    auc_mtx = aucell(expr_df, regulons, num_workers=n_jobs)

    if verbose:
        print(f"  Computed activity for {auc_mtx.shape[0]:,} cells")

    return auc_mtx


def compute_regulon_specificity(
    auc_mtx: pd.DataFrame,
    cell_labels: pd.Series,
    method: str = "zscore",
) -> pd.DataFrame:
    """Compute regulon specificity scores by cell group.

    Args:
        auc_mtx: AUCell scores (cells x regulons)
        cell_labels: Series mapping cell_id to group (stage, cell_type)
        method: 'zscore' or 'fold_change'

    Returns:
        DataFrame with regulon x group specificity scores
    """
    # Align indices
    common_cells = auc_mtx.index.intersection(cell_labels.index)
    auc_mtx = auc_mtx.loc[common_cells]
    cell_labels = cell_labels.loc[common_cells]

    # Compute group means
    group_means = auc_mtx.groupby(cell_labels).mean()

    if method == "zscore":
        # Z-score normalization across groups
        global_mean = auc_mtx.mean()
        global_std = auc_mtx.std()
        specificity = (group_means - global_mean) / (global_std + 1e-8)
    elif method == "fold_change":
        # Fold change vs global mean
        global_mean = auc_mtx.mean()
        specificity = np.log2((group_means + 1e-8) / (global_mean + 1e-8))
    else:
        raise ValueError(f"Unknown method: {method}")

    return specificity.T


# =============================================================================
# Predefined Regulon Signatures
# =============================================================================

LUNG_CANCER_REGULONS = {
    "NKX2-1": [
        "SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "NAPSA", "SLC34A2",
        "LAMP3", "ABCA3", "LPCAT1", "PGC",
    ],
    "SOX2": [
        "TP63", "KRT5", "KRT14", "KRT6A", "S100A2", "SERPINB5",
    ],
    "TP53": [
        "CDKN1A", "MDM2", "BAX", "PUMA", "NOXA", "GADD45A",
        "TIGAR", "SESN1", "SESN2",
    ],
    "MYC": [
        "CCND1", "CDK4", "E2F1", "MCM2", "MCM5", "MCM7",
        "PCNA", "RFC4", "RRM2",
    ],
    "STAT3": [
        "BCL2L1", "MCL1", "VEGFA", "MMP9", "CCND1", "MYC",
        "SOCS3", "IL6",
    ],
    "NFE2L2": [  # NRF2
        "NQO1", "HMOX1", "GCLC", "GCLM", "GPX2", "TXNRD1",
        "AKR1C1", "AKR1C2", "AKR1C3",
    ],
    "HIF1A": [
        "VEGFA", "SLC2A1", "LDHA", "PGK1", "ENO1", "ALDOA",
        "CA9", "BNIP3", "EGLN1",
    ],
    "FOXM1": [
        "CCNB1", "CCNB2", "CDC25B", "PLK1", "AURKA", "AURKB",
        "CENPA", "CENPF", "TOP2A",
    ],
}


def score_predefined_regulons(
    adata: "ad.AnnData",
    regulons: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Score cells using predefined regulon gene sets.

    Simpler alternative to full pySCENIC when you have known regulons.

    Args:
        adata: AnnData with normalized expression
        regulons: Dict of {regulon_name: gene_list}, default: LUNG_CANCER_REGULONS

    Returns:
        DataFrame with cell_id and regulon scores
    """
    if regulons is None:
        regulons = LUNG_CANCER_REGULONS

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    scores = {"cell_id": adata.obs_names.tolist()}

    for reg_name, genes in regulons.items():
        # Find genes present in data
        gene_mask = [g in adata.var_names for g in genes]
        valid_genes = [g for g, m in zip(genes, gene_mask) if m]

        if not valid_genes:
            scores[reg_name] = np.zeros(adata.n_obs)
            continue

        # Mean expression of regulon genes
        gene_idx = [adata.var_names.get_loc(g) for g in valid_genes]
        scores[reg_name] = X[:, gene_idx].mean(axis=1)

    return pd.DataFrame(scores)


# =============================================================================
# Full pySCENIC Pipeline
# =============================================================================

def run_scenic_pipeline(
    h5ad_path: Path,
    output_dir: Path,
    motif_db_path: Path | None = None,
    annotation_path: Path | None = None,
    n_jobs: int = 8,
    skip_grn: bool = False,
    verbose: bool = True,
) -> dict:
    """Run full pySCENIC pipeline.

    Steps:
    1. GRN inference (GRNBoost2)
    2. Motif pruning (cistarget) - requires database files
    3. AUCell scoring

    Args:
        h5ad_path: Path to h5ad with normalized expression
        output_dir: Output directory
        motif_db_path: Path to motif rankings (skip pruning if None)
        annotation_path: Path to motif annotations
        n_jobs: Number of parallel jobs
        skip_grn: Skip GRN inference, use predefined regulons
        verbose: Print progress

    Returns:
        Dict with adjacencies, regulons, auc_mtx paths
    """
    import scanpy as sc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    results = {}

    # Step 1: GRN inference or use predefined
    if skip_grn:
        if verbose:
            print("Using predefined lung cancer regulons...")
        regulons = LUNG_CANCER_REGULONS
        results["regulon_source"] = "predefined"
    else:
        adj_path = output_dir / "adjacencies.parquet"
        if adj_path.exists():
            if verbose:
                print(f"Loading existing adjacencies from {adj_path}")
            adjacencies = pd.read_parquet(adj_path)
        else:
            adjacencies = run_grn_inference(adata, n_jobs=n_jobs, verbose=verbose)
            adjacencies.to_parquet(adj_path)
        results["adjacencies"] = adj_path

        # Step 2: Motif pruning (if databases available)
        if motif_db_path and annotation_path:
            regulons = prune_grn_with_motifs(
                adjacencies, motif_db_path, annotation_path,
                n_jobs=n_jobs, verbose=verbose
            )
            results["regulon_source"] = "scenic"
        else:
            if verbose:
                print("No motif database provided, using top TF-target pairs...")
            # Convert top adjacencies to simple regulons
            regulons = {}
            for tf in adjacencies['TF'].unique()[:50]:
                targets = adjacencies[adjacencies['TF'] == tf].nlargest(50, 'importance')['target'].tolist()
                if len(targets) >= 5:
                    regulons[tf] = targets
            results["regulon_source"] = "grn_top50"

    # Step 3: AUCell scoring
    auc_mtx = compute_aucell(adata, regulons, n_jobs=n_jobs, verbose=verbose)
    auc_path = output_dir / "aucell_scores.parquet"
    auc_mtx.to_parquet(auc_path)
    results["aucell"] = auc_path

    # Add regulon scores to summary
    scores_path = output_dir / "regulon_scores.parquet"
    auc_mtx.reset_index().rename(columns={"index": "cell_id"}).to_parquet(scores_path)
    results["scores"] = scores_path

    if verbose:
        print(f"pySCENIC results saved to {output_dir}")

    return results


def load_regulon_scores(regulon_path: Path) -> pd.DataFrame:
    """Load precomputed regulon scores."""
    return pd.read_parquet(regulon_path)
