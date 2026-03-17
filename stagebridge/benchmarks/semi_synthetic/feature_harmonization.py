"""
Feature harmonization across HLCA, LuCA, and progression snRNA data.

Ensures a common gene space for benchmark generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class HarmonizationReport:
    """Report on feature harmonization results."""

    source_gene_counts: dict[str, int] = field(default_factory=dict)
    shared_genes: list[str] = field(default_factory=list)
    hvg_selected: list[str] = field(default_factory=list)
    final_gene_count: int = 0
    overlap_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_gene_counts": self.source_gene_counts,
            "shared_gene_count": len(self.shared_genes),
            "shared_genes_sample": self.shared_genes[:50],
            "hvg_count": len(self.hvg_selected),
            "hvg_sample": self.hvg_selected[:50],
            "final_gene_count": self.final_gene_count,
            "overlap_matrix": self.overlap_matrix,
            "warnings": self.warnings,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class FeatureHarmonizer:
    """Harmonize features across multiple data sources."""

    def __init__(
        self,
        n_hvg: int = 2000,
        require_all_sources: bool = False,
        min_overlap_fraction: float = 0.3,
    ):
        self.n_hvg = n_hvg
        self.require_all_sources = require_all_sources
        self.min_overlap_fraction = min_overlap_fraction
        self._report: HarmonizationReport | None = None
        self._final_genes: list[str] = []

    @property
    def harmonized_genes(self) -> list[str]:
        """Get the final harmonized gene list."""
        return self._final_genes
        self._final_genes: list[str] | None = None

    def harmonize(
        self,
        gene_sets: dict[str, set[str]],
        reference_adata: Any | None = None,
    ) -> HarmonizationReport:
        """Harmonize gene sets from multiple sources.

        Args:
            gene_sets: Dictionary mapping source name to set of gene names
            reference_adata: Optional reference AnnData for HVG selection

        Returns:
            HarmonizationReport with details on harmonization
        """
        report = HarmonizationReport()

        # Record source gene counts
        for name, genes in gene_sets.items():
            report.source_gene_counts[name] = len(genes)

        # Compute pairwise overlap
        names = list(gene_sets.keys())
        for i, name1 in enumerate(names):
            report.overlap_matrix[name1] = {}
            for j, name2 in enumerate(names):
                if i == j:
                    report.overlap_matrix[name1][name2] = 1.0
                else:
                    overlap = len(gene_sets[name1] & gene_sets[name2])
                    max_size = max(len(gene_sets[name1]), len(gene_sets[name2]))
                    report.overlap_matrix[name1][name2] = overlap / max_size if max_size > 0 else 0

        # Compute shared genes
        if gene_sets:
            shared = set.intersection(*gene_sets.values())
            report.shared_genes = sorted(shared)
        else:
            report.warnings.append("No gene sets provided")
            return report

        # Check overlap threshold
        if len(report.shared_genes) < self.min_overlap_fraction * min(
            len(gs) for gs in gene_sets.values()
        ):
            report.warnings.append(
                f"Low gene overlap: {len(report.shared_genes)} shared genes "
                f"(< {self.min_overlap_fraction:.0%} of smallest source)"
            )

        # Select HVGs if reference provided
        if reference_adata is not None:
            hvg = self._select_hvg_from_reference(reference_adata, report.shared_genes)
            report.hvg_selected = hvg
            report.final_gene_count = len(hvg)
            self._final_genes = hvg
        else:
            # Use all shared genes up to n_hvg
            report.hvg_selected = report.shared_genes[: self.n_hvg]
            report.final_gene_count = len(report.hvg_selected)
            self._final_genes = report.hvg_selected

        self._report = report
        return report

    def _select_hvg_from_reference(
        self,
        adata: Any,
        candidate_genes: list[str],
    ) -> list[str]:
        """Select HVGs from a reference AnnData."""
        try:
            import scanpy as sc
        except ImportError:
            # Fallback: just take top N by variance
            log.warning("scanpy not available, using variance-based HVG selection")
            return self._variance_hvg_fallback(adata, candidate_genes)

        # Subset to candidate genes
        gene_mask = adata.var_names.isin(candidate_genes)
        if gene_mask.sum() == 0:
            return candidate_genes[: self.n_hvg]

        # Create temporary AnnData for HVG selection
        import anndata

        X_subset = adata.X[:, gene_mask]
        if hasattr(X_subset, "toarray"):
            X_subset = X_subset.toarray()

        tmp = anndata.AnnData(
            X=X_subset,
            var=pd.DataFrame(index=adata.var_names[gene_mask]),
        )

        # Normalize if needed
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)

        # Select HVGs
        n_top = min(self.n_hvg, tmp.n_vars)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat")

        hvg_names = tmp.var_names[tmp.var["highly_variable"]].tolist()
        return sorted(hvg_names)

    def _variance_hvg_fallback(
        self,
        adata: Any,
        candidate_genes: list[str],
    ) -> list[str]:
        """Fallback HVG selection using variance."""
        gene_mask = adata.var_names.isin(candidate_genes)
        X = adata.X[:, gene_mask]
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Compute variance per gene
        variances = np.var(X, axis=0)
        top_idx = np.argsort(variances)[::-1][: self.n_hvg]

        selected_genes = adata.var_names[gene_mask][top_idx].tolist()
        return sorted(selected_genes)

    def get_final_genes(self) -> list[str]:
        """Get the final harmonized gene list."""
        if self._final_genes is None:
            raise ValueError("Must call harmonize() first")
        return self._final_genes

    def subset_adata(self, adata: Any) -> Any:
        """Subset an AnnData to the harmonized gene set."""
        if self._final_genes is None:
            raise ValueError("Must call harmonize() first")

        gene_mask = adata.var_names.isin(self._final_genes)
        return adata[:, gene_mask].copy()


def compute_gene_statistics(
    adata: Any,
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Compute basic statistics for genes in an AnnData.

    Args:
        adata: AnnData object
        genes: Optional list of genes to compute stats for

    Returns:
        DataFrame with gene statistics
    """
    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        X = adata.X[:, gene_mask]
        gene_names = adata.var_names[gene_mask]
    else:
        X = adata.X
        gene_names = adata.var_names

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    stats = pd.DataFrame(
        {
            "mean": np.mean(X, axis=0),
            "variance": np.var(X, axis=0),
            "nonzero_fraction": np.mean(X > 0, axis=0),
            "max": np.max(X, axis=0),
        },
        index=gene_names,
    )

    return stats
