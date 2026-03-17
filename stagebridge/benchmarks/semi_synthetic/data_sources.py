"""
Data source loading and inspection for semi-synthetic benchmark.

Handles loading real expression data from:
- HLCA (Human Lung Cell Atlas) - healthy reference
- LuCA (Lung Cancer Atlas) - disease reference
- Progression snRNA dataset - stage-aware substrate
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
class DataSourceReport:
    """Report on available data sources for benchmark generation."""

    hlca: dict[str, Any] = field(default_factory=dict)
    luca: dict[str, Any] = field(default_factory=dict)
    progression: dict[str, Any] = field(default_factory=dict)
    shared_genes: list[str] = field(default_factory=list)
    total_cells_available: int = 0
    sources_loaded: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hlca": self.hlca,
            "luca": self.luca,
            "progression": self.progression,
            "shared_genes_count": len(self.shared_genes),
            "shared_genes_sample": self.shared_genes[:20],
            "total_cells_available": self.total_cells_available,
            "sources_loaded": self.sources_loaded,
            "warnings": self.warnings,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class LoadedDataSource:
    """Container for a loaded data source."""

    name: str
    adata: Any  # AnnData
    n_cells: int
    n_genes: int
    cell_type_column: str | None
    stage_column: str | None
    donor_column: str | None
    available_cell_types: list[str]
    available_stages: list[str]
    latent_key: str | None
    expression_layer: str | None


class DataSourceLoader:
    """Load and inspect available data sources for benchmark generation."""

    def __init__(
        self,
        hlca_path: Path | None = None,
        luca_path: Path | None = None,
        progression_path: Path | None = None,
    ):
        self.hlca_path = hlca_path
        self.luca_path = luca_path
        self.progression_path = progression_path

        self._hlca: LoadedDataSource | None = None
        self._luca: LoadedDataSource | None = None
        self._progression: LoadedDataSource | None = None
        self._report: DataSourceReport | None = None

    def discover_data_paths(self) -> dict[str, Path | None]:
        """Discover available data files in standard locations."""
        discovered: dict[str, Path | None] = {
            "hlca": None,
            "luca": None,
            "progression": None,
        }

        # Standard search paths
        search_paths = [
            Path("data/references"),
            Path("data/processed/hlca"),
            Path("data/processed/luca"),
            Path("data/processed/luad"),
            Path("data/raw"),
        ]

        # HLCA patterns
        hlca_patterns = ["*hlca*.h5ad", "*HLCA*.h5ad", "hlca_full*.h5ad"]
        for base in search_paths:
            if not base.exists():
                continue
            for pattern in hlca_patterns:
                matches = list(base.glob(pattern))
                if matches:
                    discovered["hlca"] = matches[0]
                    break
            if discovered["hlca"]:
                break

        # LuCA patterns
        luca_patterns = ["*luca*.h5ad", "*LuCA*.h5ad", "luca_extended*.h5ad"]
        for base in search_paths:
            if not base.exists():
                continue
            for pattern in luca_patterns:
                matches = list(base.glob(pattern))
                if matches:
                    discovered["luca"] = matches[0]
                    break
            if discovered["luca"]:
                break

        # Progression snRNA patterns
        progression_patterns = [
            "*snrna_merged*.h5ad",
            "*progression*.h5ad",
            "*luad_evo*.h5ad",
            "cells.h5ad",
        ]
        for base in search_paths:
            if not base.exists():
                continue
            for pattern in progression_patterns:
                matches = list(base.glob(pattern))
                if matches:
                    discovered["progression"] = matches[0]
                    break
            if discovered["progression"]:
                break

        return discovered

    def load_all(self, backed: bool = True) -> DataSourceReport:
        """Load all available data sources and generate a report."""
        try:
            import anndata
        except ImportError as e:
            raise ImportError("anndata required for data loading") from e

        report = DataSourceReport()

        # Auto-discover if paths not provided
        if not any([self.hlca_path, self.luca_path, self.progression_path]):
            discovered = self.discover_data_paths()
            self.hlca_path = self.hlca_path or discovered.get("hlca")
            self.luca_path = self.luca_path or discovered.get("luca")
            self.progression_path = self.progression_path or discovered.get("progression")

        gene_sets: list[set[str]] = []

        # Load HLCA
        if self.hlca_path and self.hlca_path.exists():
            try:
                log.info("Loading HLCA from %s", self.hlca_path)
                adata = anndata.read_h5ad(self.hlca_path, backed="r" if backed else None)
                self._hlca = self._inspect_source(adata, "hlca")
                report.hlca = self._source_to_dict(self._hlca)
                report.sources_loaded.append("hlca")
                report.total_cells_available += self._hlca.n_cells
                gene_sets.append(set(adata.var_names))
            except Exception as e:
                report.warnings.append(f"Failed to load HLCA: {e}")
        else:
            report.warnings.append(f"HLCA path not found: {self.hlca_path}")

        # Load LuCA
        if self.luca_path and self.luca_path.exists():
            try:
                log.info("Loading LuCA from %s", self.luca_path)
                adata = anndata.read_h5ad(self.luca_path, backed="r" if backed else None)
                self._luca = self._inspect_source(adata, "luca")
                report.luca = self._source_to_dict(self._luca)
                report.sources_loaded.append("luca")
                report.total_cells_available += self._luca.n_cells
                gene_sets.append(set(adata.var_names))
            except Exception as e:
                report.warnings.append(f"Failed to load LuCA: {e}")
        else:
            report.warnings.append(f"LuCA path not found: {self.luca_path}")

        # Load progression snRNA
        if self.progression_path and self.progression_path.exists():
            try:
                log.info("Loading progression snRNA from %s", self.progression_path)
                adata = anndata.read_h5ad(self.progression_path, backed="r" if backed else None)
                self._progression = self._inspect_source(adata, "progression")
                report.progression = self._source_to_dict(self._progression)
                report.sources_loaded.append("progression")
                report.total_cells_available += self._progression.n_cells
                gene_sets.append(set(adata.var_names))
            except Exception as e:
                report.warnings.append(f"Failed to load progression snRNA: {e}")
        else:
            report.warnings.append(f"Progression path not found: {self.progression_path}")

        # Compute shared genes
        if gene_sets:
            shared = gene_sets[0]
            for gs in gene_sets[1:]:
                shared = shared & gs
            report.shared_genes = sorted(shared)

        self._report = report
        return report

    def _inspect_source(self, adata: Any, name: str) -> LoadedDataSource:
        """Inspect an AnnData source for relevant metadata."""
        # Find cell type column
        cell_type_col = None
        cell_type_candidates = [
            "cell_type",
            "celltype",
            "cell_type_tumor",
            "ann_level_1",
            "ann_level_2",
            "ann_level_3",
            "ann_fine",
            "ann_coarse",
            "cluster",
        ]
        for col in cell_type_candidates:
            if col in adata.obs.columns:
                cell_type_col = col
                break

        # Find stage column
        stage_col = None
        stage_candidates = ["stage", "disease_stage", "progression_stage", "condition"]
        for col in stage_candidates:
            if col in adata.obs.columns:
                stage_col = col
                break

        # Find donor column
        donor_col = None
        donor_candidates = ["donor_id", "patient_id", "subject_id", "sample_id"]
        for col in donor_candidates:
            if col in adata.obs.columns:
                donor_col = col
                break

        # Get available cell types
        available_cell_types = []
        if cell_type_col:
            available_cell_types = sorted(adata.obs[cell_type_col].dropna().unique().tolist())

        # Get available stages
        available_stages = []
        if stage_col:
            available_stages = sorted(adata.obs[stage_col].dropna().unique().tolist())

        # Find latent embedding
        latent_key = None
        latent_candidates = [
            "X_scVI",
            "X_scanvi_emb",
            "X_pca",
            "X_pca_harmony",
            "X_latent",
        ]
        for key in latent_candidates:
            if key in adata.obsm:
                latent_key = key
                break

        # Find expression layer
        expression_layer = None
        layer_candidates = ["counts", "raw_counts", "log1p", "normalized"]
        for layer in layer_candidates:
            if layer in adata.layers:
                expression_layer = layer
                break

        return LoadedDataSource(
            name=name,
            adata=adata,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            cell_type_column=cell_type_col,
            stage_column=stage_col,
            donor_column=donor_col,
            available_cell_types=available_cell_types,
            available_stages=available_stages,
            latent_key=latent_key,
            expression_layer=expression_layer,
        )

    def _source_to_dict(self, source: LoadedDataSource) -> dict[str, Any]:
        """Convert LoadedDataSource to dictionary for report."""
        return {
            "name": source.name,
            "n_cells": source.n_cells,
            "n_genes": source.n_genes,
            "cell_type_column": source.cell_type_column,
            "stage_column": source.stage_column,
            "donor_column": source.donor_column,
            "n_cell_types": len(source.available_cell_types),
            "cell_types_sample": source.available_cell_types[:20],
            "n_stages": len(source.available_stages),
            "stages": source.available_stages,
            "latent_key": source.latent_key,
            "expression_layer": source.expression_layer,
        }

    def get_source(self, name: str) -> LoadedDataSource | None:
        """Get a loaded data source by name."""
        sources = {
            "hlca": self._hlca,
            "luca": self._luca,
            "progression": self._progression,
        }
        return sources.get(name)

    def get_report(self) -> DataSourceReport:
        """Get the data source report."""
        if self._report is None:
            self._report = self.load_all()
        return self._report

    def sample_cells_by_keywords(
        self,
        source_name: str,
        keywords: list[str],
        n_cells: int,
        stage_filter: list[str] | None = None,
        seed: int = 42,
    ) -> pd.DataFrame | None:
        """Sample cells from a source matching keywords.

        Returns a DataFrame with cell metadata and indices for expression access.
        """
        source = self.get_source(source_name)
        if source is None:
            return None

        adata = source.adata
        obs = adata.obs.copy()

        # Add index for later expression lookup
        obs["_source_idx"] = np.arange(len(obs))

        # Filter by cell type keywords
        if source.cell_type_column:
            mask = pd.Series(False, index=obs.index)
            ct_col = obs[source.cell_type_column].astype(str).str.lower()
            for kw in keywords:
                mask |= ct_col.str.contains(kw.lower(), na=False)
            obs = obs[mask]

        # Filter by stage if specified
        if stage_filter and source.stage_column:
            obs = obs[obs[source.stage_column].isin(stage_filter)]

        if len(obs) == 0:
            return None

        # Sample
        rng = np.random.default_rng(seed)
        n_sample = min(n_cells, len(obs))
        sampled_idx = rng.choice(len(obs), size=n_sample, replace=False)

        result = obs.iloc[sampled_idx].copy()
        result["_source_name"] = source_name
        result["_source_cell_type_col"] = source.cell_type_column
        result["_source_stage_col"] = source.stage_column

        return result

    def get_expression_matrix(
        self,
        source_name: str,
        indices: np.ndarray,
        genes: list[str] | None = None,
        layer: str | None = None,
    ) -> np.ndarray | None:
        """Get expression matrix for specific cells.

        Args:
            source_name: Name of the data source
            indices: Cell indices to retrieve
            genes: Optional list of genes to subset to
            layer: Optional layer name (uses X if None)

        Returns:
            Expression matrix (n_cells, n_genes) as dense numpy array
        """
        source = self.get_source(source_name)
        if source is None:
            return None

        adata = source.adata

        # Get expression data
        if layer and layer in adata.layers:
            X = adata.layers[layer][indices, :]
        else:
            X = adata.X[indices, :]

        # Convert to dense if sparse
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)

        # Subset to genes if specified
        if genes:
            gene_mask = adata.var_names.isin(genes)
            X = X[:, gene_mask]

        return X


def create_fallback_data(
    n_cells: int = 1000,
    n_genes: int = 2000,
    cell_groups: dict[str, int] | None = None,
    stages: list[str] | None = None,
    seed: int = 42,
) -> Any:
    """Create fallback synthetic data when real data is unavailable.

    This allows the benchmark pipeline to run for testing even without
    real HLCA/LuCA/progression data.
    """
    try:
        import anndata
        import scipy.sparse as sp
    except ImportError as e:
        raise ImportError("anndata and scipy required") from e

    rng = np.random.default_rng(seed)

    if cell_groups is None:
        # Create proportional cell groups that sum to n_cells
        cell_groups = {
            "AT2": int(n_cells * 0.30),
            "Fibroblast": int(n_cells * 0.20),
            "Macrophage": int(n_cells * 0.20),
            "Endothelial": int(n_cells * 0.10),
            "T_cell": int(n_cells * 0.10),
            "Other": n_cells - int(n_cells * 0.90),  # Remainder
        }

    if stages is None:
        stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]

    # Build cell metadata
    cell_types = []
    for ct, count in cell_groups.items():
        cell_types.extend([ct] * count)

    # Use actual number of cells from cell_groups
    actual_n_cells = len(cell_types)
    rng.shuffle(cell_types)

    # Assign stages with progression bias
    stage_probs = np.array([0.15, 0.2, 0.25, 0.25, 0.15])
    stage_probs = stage_probs / stage_probs.sum()
    cell_stages = rng.choice(stages, size=actual_n_cells, p=stage_probs)

    # Create expression matrix (sparse, simulated counts)
    # Different cell types have different expression patterns
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]

    # Create cell-type specific mean expression
    ct_means = {}
    for ct in set(cell_types):
        ct_means[ct] = rng.exponential(1.0, size=n_genes)

    X_data = []
    for ct in cell_types:
        # Sample from negative binomial around cell-type mean
        means = ct_means[ct]
        counts = rng.negative_binomial(n=5, p=5 / (5 + means))
        X_data.append(counts)

    X = sp.csr_matrix(np.array(X_data, dtype=np.float32))

    # Create AnnData
    obs = pd.DataFrame(
        {
            "cell_type": cell_types,
            "stage": cell_stages,
            "donor_id": [f"donor_{i % 5}" for i in range(actual_n_cells)],
        },
        index=[f"cell_{i:06d}" for i in range(actual_n_cells)],
    )

    var = pd.DataFrame(index=gene_names)

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()

    # Add simple PCA as latent
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=32, random_state=seed)
    adata.obsm["X_pca"] = svd.fit_transform(X).astype(np.float32)

    return adata
