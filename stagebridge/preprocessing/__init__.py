"""Preprocessing modules for StageBridge data preparation."""

from stagebridge.preprocessing.spatial_stats import (
    compute_spatial_neighbors,
    compute_nhood_enrichment,
    compute_morans_i,
    run_spatial_stats,
)
from stagebridge.preprocessing.de_analysis import (
    run_de_stage_vs_rest,
    run_de_all_stages,
)
from stagebridge.preprocessing.summary_stats import (
    compute_celltype_proportions,
    compute_stage_summary,
    run_summary_stats,
)
from stagebridge.preprocessing.embeddings import (
    compute_umap,
    compute_phate,
    compute_leiden_clustering,
    run_embeddings,
)
from stagebridge.preprocessing.qc import (
    compute_qc_metrics,
    run_qc,
)

__all__ = [
    # Spatial stats
    "compute_spatial_neighbors",
    "compute_nhood_enrichment",
    "compute_morans_i",
    "run_spatial_stats",
    # DE analysis
    "run_de_stage_vs_rest",
    "run_de_all_stages",
    # Summary stats
    "compute_celltype_proportions",
    "compute_stage_summary",
    "run_summary_stats",
    # Embeddings
    "compute_umap",
    "compute_phate",
    "compute_leiden_clustering",
    "run_embeddings",
    # QC
    "compute_qc_metrics",
    "run_qc",
]
