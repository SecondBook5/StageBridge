"""Biological analysis layer for StageBridge.

This module provides post-training analyses that connect model outputs to
biological insight from Peng et al. 2026 (Cancer Cell):

  context_sensitivity — quantifies how much population context matters per transition
  gene_attribution    — correlates gene expression with context vector dimensions
  trajectory          — Euler integration and UMAP projection of predicted cell paths
"""
from .context_sensitivity import compute_context_sensitivity
from .gene_attribution import extract_context_vectors, compute_gene_context_correlation, top_genes_per_context_dim
from .trajectory import integrate_trajectories, project_trajectories_to_umap

__all__ = [
    "compute_context_sensitivity",
    "extract_context_vectors",
    "compute_gene_context_correlation",
    "top_genes_per_context_dim",
    "integrate_trajectories",
    "project_trajectories_to_umap",
]
