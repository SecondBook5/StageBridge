"""Transcription factor and pathway activity scoring via decoupleR.

Provides wrapper functions for TF activity (CollecTRI), pathway activity (PROGENy),
and hallmark gene set scoring using the decoupleR v2 API.

Requires: decoupler (pip install decoupler)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def compute_tf_activity(
    adata: "ad.AnnData",
    organism: str = "human",
) -> pd.DataFrame:
    """Compute TF activity scores using CollecTRI network.

    Args:
        adata: AnnData with normalized expression
        organism: 'human' or 'mouse'

    Returns:
        DataFrame with cell_id x TF activity scores
    """
    import decoupler as dc

    collectri = dc.op.collectri(organism=organism)
    dc.mt.ulm(data=adata, net=collectri)
    tf_acts = dc.pp.get_obsm(adata=adata, key="score_ulm")

    return pd.DataFrame(
        tf_acts.X, index=tf_acts.obs.index, columns=tf_acts.var.index
    )


def compute_pathway_activity(
    adata: "ad.AnnData",
    organism: str = "human",
) -> pd.DataFrame:
    """Compute pathway activity scores using PROGENy.

    Args:
        adata: AnnData with normalized expression
        organism: 'human' or 'mouse'

    Returns:
        DataFrame with cell_id x pathway activity scores (14 pathways)
    """
    import decoupler as dc

    progeny = dc.op.progeny(organism=organism)
    dc.mt.ulm(data=adata, net=progeny)
    pathway_acts = dc.pp.get_obsm(adata=adata, key="score_ulm")

    return pd.DataFrame(
        pathway_acts.X, index=pathway_acts.obs.index, columns=pathway_acts.var.index
    )


def compute_hallmark_activity(
    adata: "ad.AnnData",
    organism: str = "human",
) -> pd.DataFrame:
    """Compute hallmark gene set activity scores (MSigDB).

    Args:
        adata: AnnData with normalized expression
        organism: 'human' or 'mouse'

    Returns:
        DataFrame with cell_id x hallmark activity scores (50 gene sets)
    """
    import decoupler as dc

    hallmark = dc.op.hallmark(organism=organism)
    dc.mt.ulm(data=adata, net=hallmark)
    hallmark_acts = dc.pp.get_obsm(adata=adata, key="score_ulm")

    return pd.DataFrame(
        hallmark_acts.X, index=hallmark_acts.obs.index, columns=hallmark_acts.var.index
    )


def rank_by_progression(
    adata: "ad.AnnData",
    activity_key: str = "score_ulm",
    order_col: str = "stage_num",
    stat: str = "dcor",
) -> pd.DataFrame:
    """Rank TFs/pathways by correlation with progression order.

    Args:
        adata: AnnData with activity scores in obsm
        activity_key: Key in obsm containing activity scores
        order_col: Column in obs containing numeric progression order
        stat: Correlation statistic ('dcor' for distance correlation, 'pearson')

    Returns:
        DataFrame with ranked features by progression correlation
    """
    import decoupler as dc

    score = dc.pp.get_obsm(adata=adata, key=activity_key)
    score.obs[order_col] = adata.obs[order_col].values
    return dc.tl.rankby_order(adata=score, order=order_col, stat=stat)


def rank_by_group(
    adata: "ad.AnnData",
    activity_key: str = "score_ulm",
    groupby: str = "stage",
    reference: str = "rest",
) -> pd.DataFrame:
    """Find marker TFs/pathways for each group.

    Args:
        adata: AnnData with activity scores in obsm
        activity_key: Key in obsm containing activity scores
        groupby: Column in obs for grouping
        reference: Reference group ('rest' or specific group)

    Returns:
        DataFrame with marker features per group
    """
    import decoupler as dc

    score = dc.pp.get_obsm(adata=adata, key=activity_key)
    score.obs[groupby] = adata.obs[groupby].values
    return dc.tl.rankby_group(adata=score, groupby=groupby, reference=reference)


def compute_pseudobulk_activity(
    adata: "ad.AnnData",
    sample_col: str = "donor_id",
    groups_col: str = "stage",
    organism: str = "human",
    min_cells: int = 10,
    min_counts: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Compute pseudobulk TF/pathway activity.

    Aggregates cells by sample and group, then scores activity.

    Args:
        adata: AnnData with raw counts
        sample_col: Column for sample identity
        groups_col: Column for grouping (e.g., stage)
        organism: 'human' or 'mouse'
        min_cells: Minimum cells per pseudobulk
        min_counts: Minimum counts per pseudobulk

    Returns:
        Dict with 'tf', 'pathway', 'hallmark' DataFrames
    """
    import decoupler as dc

    pdata = dc.pp.pseudobulk(
        adata=adata,
        sample_col=sample_col,
        groups_col=groups_col,
        mode="sum",
    )
    dc.pp.filter_samples(pdata, min_cells=min_cells, min_counts=min_counts)

    results = {}

    # TF activity
    collectri = dc.op.collectri(organism=organism)
    dc.mt.ulm(data=pdata, net=collectri)
    tf_acts = dc.pp.get_obsm(adata=pdata, key="score_ulm")
    results["tf"] = pd.DataFrame(
        tf_acts.X, index=tf_acts.obs.index, columns=tf_acts.var.index
    )

    # Pathway activity
    progeny = dc.op.progeny(organism=organism)
    dc.mt.ulm(data=pdata, net=progeny)
    pw_acts = dc.pp.get_obsm(adata=pdata, key="score_ulm")
    results["pathway"] = pd.DataFrame(
        pw_acts.X, index=pw_acts.obs.index, columns=pw_acts.var.index
    )

    # Hallmark activity
    hallmark = dc.op.hallmark(organism=organism)
    dc.mt.ulm(data=pdata, net=hallmark)
    hm_acts = dc.pp.get_obsm(adata=pdata, key="score_ulm")
    results["hallmark"] = pd.DataFrame(
        hm_acts.X, index=hm_acts.obs.index, columns=hm_acts.var.index
    )

    return results


def compute_spatial_activity(
    adata: "ad.AnnData",
    organism: str = "human",
    bw: float = 100,
    cutoff: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Compute spatially-smoothed TF/pathway activity.

    Uses KNN-based spatial weighting before scoring.

    Args:
        adata: AnnData with spatial coordinates in obsm['spatial']
        organism: 'human' or 'mouse'
        bw: Bandwidth for spatial weighting
        cutoff: Cutoff for spatial connectivity

    Returns:
        Dict with 'tf', 'pathway', 'hallmark' DataFrames
    """
    import decoupler as dc

    # Apply spatial smoothing
    dc.pp.knn(adata, key="spatial", bw=bw, cutoff=cutoff)
    adata.X = adata.obsp["spatial_connectivities"].dot(adata.X)

    results = {}

    # TF activity
    collectri = dc.op.collectri(organism=organism)
    dc.mt.ulm(data=adata, net=collectri)
    tf_acts = dc.pp.get_obsm(adata=adata, key="score_ulm")
    results["tf"] = pd.DataFrame(
        tf_acts.X, index=tf_acts.obs.index, columns=tf_acts.var.index
    )

    # Pathway activity
    progeny = dc.op.progeny(organism=organism)
    dc.mt.ulm(data=adata, net=progeny)
    pw_acts = dc.pp.get_obsm(adata=adata, key="score_ulm")
    results["pathway"] = pd.DataFrame(
        pw_acts.X, index=pw_acts.obs.index, columns=pw_acts.var.index
    )

    # Hallmark activity
    hallmark = dc.op.hallmark(organism=organism)
    dc.mt.ulm(data=adata, net=hallmark)
    hm_acts = dc.pp.get_obsm(adata=adata, key="score_ulm")
    results["hallmark"] = pd.DataFrame(
        hm_acts.X, index=hm_acts.obs.index, columns=hm_acts.var.index
    )

    return results
