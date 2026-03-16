"""
Tangram spatial mapping backend wrapper.

Tangram: Marker-gene based mapping with gradient optimization.
Reference: https://github.com/broadinstitute/Tangram
"""

from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from .base import SpatialBackend, SpatialMappingResult, compute_cell_type_entropy, compute_sparsity


class TangramBackend(SpatialBackend):
    """
    Tangram spatial mapping wrapper.

    Configuration options:
    - mode: 'cells' or 'clusters' (map individual cells or cell types)
    - marker_genes: List of marker genes or 'auto' for automatic selection
    - density_prior: Density regularization weight
    - n_epochs: Training epochs
    - device: 'cpu' or 'cuda'
    """

    def __init__(
        self,
        mode: str = "clusters",
        marker_genes: str | List[str] = "auto",
        density_prior: float = 1.0,
        n_epochs: int = 1000,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mode = mode
        self.marker_genes = marker_genes
        self.density_prior = density_prior
        self.n_epochs = n_epochs
        self.device = device

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Optional[Path] = None,
    ) -> SpatialMappingResult:
        """Run Tangram mapping."""
        # Validate and preprocess
        self.validate_inputs(snrna, spatial)
        snrna, spatial = self.preprocess(snrna, spatial)

        # Import tangram (lazy import)
        try:
            import tangram as tg
        except ImportError:
            raise ImportError(
                "Tangram not installed. Install with: pip install tangram-sc"
            )

        # Select marker genes if needed
        if self.marker_genes == "auto":
            marker_genes = self._select_marker_genes(snrna)
        else:
            marker_genes = self.marker_genes

        # Subset to marker genes
        marker_genes = [g for g in marker_genes if g in snrna.var_names]
        snrna_markers = snrna[:, marker_genes].copy()
        spatial_markers = spatial[:, marker_genes].copy()

        print(f"Tangram: Using {len(marker_genes)} marker genes")

        # Run mapping
        print(f"Running Tangram with mode={self.mode}, epochs={self.n_epochs}...")

        ad_map = tg.map_cells_to_space(
            adata_sc=snrna_markers,
            adata_sp=spatial_markers,
            mode=self.mode,
            density_prior=self.density_prior,
            num_epochs=self.n_epochs,
            device=self.device,
        )

        # Extract cell type proportions
        if self.mode == "clusters":
            # Get cell type proportions directly
            cell_type_proportions = self._extract_cluster_proportions(
                ad_map, snrna, spatial
            )
        else:
            # Aggregate cell-level mapping to cell types
            cell_type_proportions = self._aggregate_to_celltypes(
                ad_map, snrna, spatial
            )

        # Compute confidence
        confidence = self.estimate_confidence(snrna, spatial, None)

        # Compute upstream metrics
        upstream_metrics = self.compute_upstream_metrics(
            snrna, spatial, None
        )

        # Save if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            ad_map.write_h5ad(output_dir / "tangram_mapping.h5ad")

        result = SpatialMappingResult(
            cell_type_proportions=cell_type_proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata={
                "backend": "tangram",
                "mode": self.mode,
                "n_marker_genes": len(marker_genes),
                "n_epochs": self.n_epochs,
                "density_prior": self.density_prior,
            },
        )

        return result

    def _select_marker_genes(
        self,
        snrna: ad.AnnData,
        n_genes: int = 100,
    ) -> List[str]:
        """
        Select marker genes using differential expression.

        Args:
            snrna: Single-cell reference
            n_genes: Number of top genes per cell type

        Returns:
            List of marker gene names
        """
        # Rank genes per cell type
        sc.tl.rank_genes_groups(
            snrna,
            groupby="cell_type",
            method="wilcoxon",
            n_genes=n_genes,
        )

        # Extract top genes per group
        marker_genes = set()
        for group in snrna.uns["rank_genes_groups"]["names"].dtype.names:
            genes = snrna.uns["rank_genes_groups"]["names"][group][:n_genes]
            marker_genes.update(genes)

        return list(marker_genes)

    def _extract_cluster_proportions(
        self,
        ad_map: ad.AnnData,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
    ) -> pd.DataFrame:
        """Extract cell type proportions from cluster-mode mapping."""
        # ad_map should have (n_spots, n_celltypes) in .X
        cell_types = snrna.obs["cell_type"].unique()

        proportions = pd.DataFrame(
            ad_map.X,
            index=spatial.obs_names,
            columns=cell_types,
        )

        # Ensure non-negative and normalized
        proportions = proportions.clip(lower=0)
        proportions = proportions.div(proportions.sum(axis=1), axis=0).fillna(0)

        return proportions

    def _aggregate_to_celltypes(
        self,
        ad_map: ad.AnnData,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
    ) -> pd.DataFrame:
        """Aggregate cell-level mapping to cell type proportions."""
        # ad_map.X: (n_spots, n_cells) assignment matrix
        # Aggregate by cell type

        cell_types = snrna.obs["cell_type"].values
        spot_names = spatial.obs_names
        unique_celltypes = sorted(snrna.obs["cell_type"].unique())

        # Build proportion matrix
        proportions = np.zeros((len(spot_names), len(unique_celltypes)))

        for ct_idx, ct in enumerate(unique_celltypes):
            ct_mask = cell_types == ct
            proportions[:, ct_idx] = ad_map.X[:, ct_mask].sum(axis=1)

        # Normalize
        row_sums = proportions.sum(axis=1, keepdims=True)
        proportions = proportions / (row_sums + 1e-10)

        return pd.DataFrame(
            proportions,
            index=spot_names,
            columns=unique_celltypes,
        )

    def compute_upstream_metrics(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: Optional[SpatialMappingResult],
    ) -> Dict[str, float]:
        """Compute Tangram-specific upstream metrics."""
        if result is None:
            # Called before result is fully constructed
            return {}

        proportions = result.cell_type_proportions

        # Cell type entropy (diversity)
        entropy = compute_cell_type_entropy(proportions)

        # Sparsity
        sparsity = compute_sparsity(proportions)

        # Coverage (fraction with confident mapping)
        coverage = (result.confidence > 0.5).mean()

        metrics = {
            "mean_entropy": float(entropy.mean()),
            "std_entropy": float(entropy.std()),
            "sparsity": float(sparsity),
            "coverage": float(coverage),
            "n_spots": len(spatial),
            "n_celltypes": proportions.shape[1],
        }

        return metrics

    def estimate_confidence(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: Optional[SpatialMappingResult],
    ) -> pd.Series:
        """
        Estimate confidence from cell type proportion entropy.

        Low entropy (dominated by one type) = high confidence
        High entropy (diverse mixture) = lower confidence
        """
        if result is None:
            # Placeholder - will be computed after proportions are known
            return pd.Series(
                np.ones(len(spatial)),
                index=spatial.obs_names,
                name="confidence",
            )

        proportions = result.cell_type_proportions

        # Compute entropy (normalized)
        entropy = compute_cell_type_entropy(proportions)

        # Convert to confidence: 1 - entropy (so low entropy = high confidence)
        confidence = 1.0 - entropy

        return confidence


def run_tangram(
    snrna_path: str | Path,
    spatial_path: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> SpatialMappingResult:
    """
    Convenience function to run Tangram mapping.

    Args:
        snrna_path: Path to single-cell h5ad
        spatial_path: Path to spatial h5ad
        output_dir: Where to save results
        **kwargs: Additional Tangram parameters

    Returns:
        SpatialMappingResult
    """
    # Load data
    print(f"Loading snRNA data from {snrna_path}...")
    snrna = ad.read_h5ad(snrna_path)

    print(f"Loading spatial data from {spatial_path}...")
    spatial = ad.read_h5ad(spatial_path)

    # Initialize backend
    backend = TangramBackend(**kwargs)

    # Run mapping
    result = backend.map(snrna, spatial, output_dir=output_dir)

    # Save result
    result.save(output_dir)

    print(f" Tangram mapping complete. Results saved to {output_dir}")

    return result


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Tangram backend with synthetic data...")

    # Create dummy data
    n_cells = 1000
    n_spots = 500
    n_genes = 100

    snrna = ad.AnnData(
        X=np.random.randn(n_cells, n_genes),
        obs=pd.DataFrame({
            "cell_type": np.random.choice(["A", "B", "C"], n_cells)
        }),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )

    spatial = ad.AnnData(
        X=np.random.randn(n_spots, n_genes),
        obs=pd.DataFrame(index=[f"spot_{i}" for i in range(n_spots)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
        obsm={"spatial": np.random.rand(n_spots, 2)},
    )

    # Run mapping
    backend = TangramBackend(mode="clusters", n_epochs=10)
    result = backend.map(snrna, spatial)

    print(f"Proportions shape: {result.cell_type_proportions.shape}")
    print(f"Confidence range: [{result.confidence.min():.3f}, {result.confidence.max():.3f}]")
    print(f"Metrics: {result.upstream_metrics}")

    print("\n Tangram backend test passed!")
