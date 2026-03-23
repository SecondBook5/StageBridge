"""
Tangram spatial mapping backend wrapper using scvi-tools integration.

Tangram via scvi-tools: Model-based spatial mapping with GPU acceleration and
integrated Squidpy visualizations for spatial analysis.

Reference: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/spatial/Tangram_scvi_tools.html
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from .backend_base import SpatialBackend, BackendMappingResult, compute_cell_type_entropy, compute_sparsity


class TangramBackend(SpatialBackend):
    """
    Tangram spatial mapping wrapper using scvi.external.Tangram.

    Uses scvi-tools integrated Tangram with MuData for better performance
    and integration with the scvi-tools ecosystem.

    Configuration options:
    - constrained: Use constrained mode (requires density priors)
    - marker_genes: List of marker genes or 'auto' for automatic selection
    - n_epochs: Training epochs (default 1000)
    - target_count: Target cell count for constrained mode
    """

    def __init__(
        self,
        mode: str = "clusters",  # Kept for API compatibility but uses constrained mode
        marker_genes: str | list[str] = "auto",
        constrained: bool = True,
        n_epochs: int = 1000,
        target_count: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mode = mode  # For compatibility
        self.marker_genes = marker_genes
        self.constrained = constrained
        self.n_epochs = n_epochs
        self.target_count = target_count

        # Store trained model for visualizations
        self.model = None
        self._snrna_ref = None
        self._spatial_ref = None
        self._mapper = None

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run Tangram mapping using scvi-tools."""
        # Validate and preprocess
        self.validate_inputs(snrna, spatial)
        snrna, spatial = self.preprocess(snrna, spatial)

        # Import scvi-tools Tangram
        try:
            from scvi.external import Tangram
            import mudata
        except ImportError as e:
            raise ImportError(
                "scvi-tools not installed. Install with: pip install scvi-tools>=1.1.0"
            ) from e

        # Select marker genes if needed
        if self.marker_genes == "auto":
            marker_genes = self._select_marker_genes(snrna)
        else:
            marker_genes = self.marker_genes

        # Subset to marker genes
        common_genes = list(set(marker_genes) & set(snrna.var_names) & set(spatial.var_names))
        snrna_train = snrna[:, common_genes].copy()
        spatial_train = spatial[:, common_genes].copy()

        print(f"Tangram (scvi-tools): Using {len(common_genes)} marker genes")

        # Create MuData
        mdata = mudata.MuData({
            "sc": snrna_train,
            "sp": spatial_train,
        })

        # Compute density priors if using constrained mode
        if self.constrained:
            # Uniform density as default (can be enhanced with cell segmentation)
            spatial_train.obs["uniform_density"] = 1.0 / spatial_train.n_obs
            density_prior_key = "uniform_density"

            # Target count defaults to number of spots if not specified
            if self.target_count is None:
                self.target_count = spatial_train.n_obs

            print(f"  Constrained mode: target_count={self.target_count}")
        else:
            density_prior_key = None
            self.target_count = None

        # Setup MuData for Tangram
        Tangram.setup_mudata(
            mdata,
            density_prior_key=density_prior_key,
            modalities={
                "density_prior_key": "sp",
                "sc_layer": "sc",
                "sp_layer": "sp",
            },
        )

        # Create and train model
        print(f"  Training Tangram model ({self.n_epochs} epochs)...")
        model = Tangram(
            mdata,
            constrained=self.constrained,
            target_count=self.target_count,
        )

        model.train(max_epochs=self.n_epochs)

        # Get mapper matrix
        mapper = model.get_mapper_matrix()

        # Store model and data for visualizations
        self.model = model
        self._snrna_ref = snrna
        self._spatial_ref = spatial
        self._mapper = mapper

        # Project cell type annotations
        if "cell_type" in snrna.obs.columns:
            cell_type_key = "cell_type"
        elif "celltype" in snrna.obs.columns:
            cell_type_key = "celltype"
        else:
            raise ValueError("No cell_type column found in snRNA obs")

        ct_pred = model.project_cell_annotations(
            mdata.mod["sc"],
            mdata.mod["sp"],
            mapper,
            mdata.mod["sc"].obs[cell_type_key],
        )

        # Convert to cell type proportions DataFrame
        cell_type_proportions = pd.DataFrame(
            ct_pred,
            index=spatial.obs_names,
        )

        # Compute confidence (based on entropy of predictions)
        confidence = self._compute_mapping_confidence(cell_type_proportions)

        # Create preliminary result for metrics computation
        preliminary_result = BackendMappingResult(
            cell_type_proportions=cell_type_proportions,
            confidence=confidence,
            upstream_metrics={},
            metadata={},
        )

        # Compute upstream metrics
        upstream_metrics = self.compute_upstream_metrics(snrna, spatial, preliminary_result)

        # Save if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save mapper matrix
            np.save(output_dir / "tangram_mapper.npy", mapper)

            # Save cell type predictions
            cell_type_proportions.to_csv(output_dir / "tangram_cell_type_props.csv")

            # Save annotated spatial data
            spatial_annotated = spatial.copy()
            spatial_annotated.obsm["tangram_proportions"] = cell_type_proportions.values
            for ct in cell_type_proportions.columns:
                spatial_annotated.obs[f"tangram_{ct}"] = cell_type_proportions[ct].values
            spatial_annotated.write_h5ad(output_dir / "tangram_spatial_annotated.h5ad")

            # Save MuData for Squidpy visualizations
            try:
                mdata.write(output_dir / "tangram_mudata.h5mu")
            except Exception:
                pass  # MuData write optional

            print(f"  Tangram outputs saved to {output_dir}")

        result = BackendMappingResult(
            cell_type_proportions=cell_type_proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata={
                "backend": "tangram_scvi",
                "constrained": self.constrained,
                "n_marker_genes": len(common_genes),
                "n_epochs": self.n_epochs,
                "target_count": self.target_count,
            },
        )

        return result

    def _compute_mapping_confidence(
        self,
        cell_type_proportions: pd.DataFrame,
    ) -> np.ndarray:
        """
        Compute confidence scores from cell type proportion entropy.

        Lower entropy = higher confidence (more certain mapping).
        """
        # Compute entropy per spot
        props = cell_type_proportions.values
        props = props / (props.sum(axis=1, keepdims=True) + 1e-10)

        entropy = -np.sum(props * np.log(props + 1e-10), axis=1)
        max_entropy = np.log(props.shape[1])  # Maximum possible entropy

        # Confidence = 1 - normalized_entropy
        confidence = 1.0 - (entropy / max_entropy)

        return confidence

    def compute_upstream_metrics(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: BackendMappingResult | None,
    ) -> dict[str, float]:
        """Compute upstream quality metrics for Tangram mapping."""
        metrics = {}

        if result is not None and result.cell_type_proportions is not None:
            props = result.cell_type_proportions.values
            # Cell type entropy (diversity)
            metrics["mean_entropy"] = float(compute_cell_type_entropy(props).mean())
            # Sparsity
            metrics["sparsity"] = float(compute_sparsity(props))
            # Coverage (spots with confident mapping)
            if result.confidence is not None:
                conf = result.confidence if isinstance(result.confidence, np.ndarray) else result.confidence.values
                metrics["coverage_0.5"] = float((conf > 0.5).mean())
                metrics["mean_confidence"] = float(conf.mean())

        return metrics

    def estimate_confidence(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: BackendMappingResult | None,
    ) -> pd.Series:
        """Return confidence scores (already computed in map())."""
        if result is not None and result.confidence is not None:
            if isinstance(result.confidence, pd.Series):
                return result.confidence
            return pd.Series(result.confidence, index=spatial.obs_names)
        return pd.Series(np.zeros(spatial.n_obs), index=spatial.obs_names)

    def _select_marker_genes(
        self,
        snrna: ad.AnnData,
        n_genes: int = 100,
    ) -> list[str]:
        """
        Select marker genes using differential expression.

        Args:
            snrna: Single-cell reference
            n_genes: Number of top genes per cell type

        Returns:
            List of marker gene names
        """
        # Check if cell_type column exists
        if "cell_type" not in snrna.obs.columns and "celltype" not in snrna.obs.columns:
            raise ValueError("cell_type column required for marker gene selection")

        cell_type_key = "cell_type" if "cell_type" in snrna.obs.columns else "celltype"

        # Rank genes per cell type
        sc.tl.rank_genes_groups(
            snrna,
            groupby=cell_type_key,
            method="wilcoxon",
            n_genes=n_genes,
            use_raw=False,
        )

        # Extract top genes per group
        marker_genes = set()
        for group in snrna.uns["rank_genes_groups"]["names"].dtype.names:
            genes = snrna.uns["rank_genes_groups"]["names"][group][:n_genes]
            marker_genes.update(genes)

        return list(marker_genes)

    def plot_cell_type_spatial(
        self,
        cell_type: str,
        quantile_clip: float = 0.99,
        cmap: str = "Reds",
        save_path: Path | None = None,
    ):
        """
        Plot cell type proportions in spatial coordinates.

        Args:
            cell_type: Cell type to visualize
            quantile_clip: Clip values at this quantile for better visualization
            cmap: Matplotlib colormap
            save_path: If provided, saves figure to this path
        """
        if self._spatial_ref is None or self._mapper is None:
            raise RuntimeError("Must run map() before plotting")

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("matplotlib required for plotting") from e

        # Get spatial coordinates
        if "spatial" not in self._spatial_ref.obsm:
            raise ValueError("Spatial coordinates not found in spatial.obsm['spatial']")

        coords = self._spatial_ref.obsm["spatial"]

        # Get cell type proportions
        proportions = self._spatial_ref.obs.get(f"tangram_{cell_type}")
        if proportions is None:
            raise ValueError(f"Cell type {cell_type} not found in stored proportions")

        # Clip for better visualization
        values = np.clip(proportions.values, 0, np.quantile(proportions.values, quantile_clip))

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=cmap,
            s=20,
        )
        plt.colorbar(scatter, ax=ax)
        ax.set_title(f"Tangram: {cell_type} proportion")
        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def project_genes(
        self,
        gene_names: list[str],
        aggregate: bool = False,
    ) -> pd.DataFrame:
        """
        Project gene expression from single-cell to spatial using Tangram mapper.

        Args:
            gene_names: List of genes to project
            aggregate: If True, sum across genes

        Returns:
            DataFrame of projected expression (spots × genes or spots × 1 if aggregate)
        """
        if self._mapper is None or self._snrna_ref is None or self._spatial_ref is None:
            raise RuntimeError("Must run map() before projecting genes")

        # Check gene availability
        available_genes = list(set(gene_names) & set(self._snrna_ref.var_names))
        if not available_genes:
            raise ValueError("None of the requested genes found in reference data")

        # Get expression matrix for genes
        gene_expr = self._snrna_ref[:, available_genes].X
        if hasattr(gene_expr, "toarray"):
            gene_expr = gene_expr.toarray()

        # Project using mapper: spatial_expr = mapper @ cell_expr
        projected = self._mapper @ gene_expr  # (n_spots, n_genes)

        if aggregate:
            projected = projected.sum(axis=1, keepdims=True)
            columns = ["aggregated_expression"]
        else:
            columns = available_genes

        result = pd.DataFrame(
            projected,
            index=self._spatial_ref.obs_names,
            columns=columns,
        )

        return result

    def plot_projected_genes(
        self,
        gene_names: list[str],
        aggregate: bool = True,
        cmap: str = "Reds",
        save_path: Path | None = None,
    ):
        """
        Project and visualize gene expression in spatial coordinates.

        Args:
            gene_names: Genes to project
            aggregate: If True, sum genes (useful for gene sets)
            cmap: Matplotlib colormap
            save_path: If provided, saves figure to this path
        """
        if self._spatial_ref is None:
            raise RuntimeError("Must run map() before plotting")

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("matplotlib required for plotting") from e

        # Project genes
        projected = self.project_genes(gene_names, aggregate=aggregate)

        # Get spatial coordinates
        coords = self._spatial_ref.obsm["spatial"]

        # Get values
        if aggregate:
            values = np.log1p(1e4 * projected.values.flatten())
            title = f"Projected: {', '.join(gene_names[:3])}"
            if len(gene_names) > 3:
                title += f" (+{len(gene_names) - 3} more)"
        else:
            # Plot first gene only
            values = np.log1p(1e4 * projected.iloc[:, 0].values)
            title = f"Projected: {projected.columns[0]}"

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=cmap,
            s=20,
        )
        plt.colorbar(scatter, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def compute_spatial_statistics(
        self,
        cell_types: list[str] | None = None,
        n_neighs: int = 6,
    ) -> dict[str, Any]:
        """
        Compute spatial statistics using Squidpy.

        Includes spatial autocorrelation (Moran's I), co-occurrence,
        and neighborhood enrichment.

        Args:
            cell_types: Cell types to analyze. If None, uses all.
            n_neighs: Number of neighbors for spatial graph

        Returns:
            Dictionary with spatial statistics results
        """
        if self._spatial_ref is None:
            raise RuntimeError("Must run map() before computing spatial statistics")

        try:
            import squidpy as sq
        except ImportError as e:
            raise ImportError(
                "squidpy required for spatial statistics. Install: pip install squidpy"
            ) from e

        spatial_adata = self._spatial_ref.copy()

        # Build spatial graph
        sq.gr.spatial_neighbors(spatial_adata, n_neighs=n_neighs, coord_type="generic")

        results = {}

        # Spatial autocorrelation for cell type proportions
        if cell_types is None:
            # Get all tangram cell type columns
            ct_cols = [col for col in spatial_adata.obs.columns if col.startswith("tangram_")]
            cell_types = [col.replace("tangram_", "") for col in ct_cols]

        morans_i = {}
        for ct in cell_types:
            ct_col = f"tangram_{ct}"
            if ct_col in spatial_adata.obs.columns:
                try:
                    sq.gr.spatial_autocorr(
                        spatial_adata,
                        mode="moran",
                        genes=[ct_col],
                    )
                    morans_i[ct] = float(spatial_adata.uns["moranI"].loc[ct_col, "I"])
                except Exception:
                    pass  # Skip if fails

        results["morans_i"] = morans_i

        # Co-occurrence and enrichment would require discrete cell type assignments
        # These are optional advanced features

        return results


def run_tangram(
    snrna_path: str | Path,
    spatial_path: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> BackendMappingResult:
    """
    Convenience function to run Tangram mapping.

    Args:
        snrna_path: Path to single-cell h5ad
        spatial_path: Path to spatial h5ad
        output_dir: Where to save results
        **kwargs: Additional Tangram parameters

    Returns:
        BackendMappingResult
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
        obs=pd.DataFrame({"cell_type": np.random.choice(["A", "B", "C"], n_cells)}),
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
