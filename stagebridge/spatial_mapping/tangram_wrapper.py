"""
Tangram spatial mapping backend wrapper.

Supports two implementations:
1. Standalone tangram-sc (default, PyTorch-based)
2. scvi-tools Tangram (fallback, JAX-based)

Tangram: Deep learning-based spatial mapping of single-cell transcriptomes
to spatial transcriptomics data.

Reference: Biancalani et al., Nature Methods 2021
https://github.com/broadinstitute/Tangram
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch

from .backend_base import (
    SpatialBackend,
    BackendMappingResult,
    compute_cell_type_entropy,
    compute_sparsity,
)


class TangramBackend(SpatialBackend):
    """
    Tangram spatial mapping wrapper with fallback support.

    Tries standalone tangram-sc first (PyTorch), falls back to scvi-tools
    Tangram (JAX) if standalone fails.

    Configuration options:
    - mode: 'clusters' (cell type level) or 'cells' (single cell level)
    - marker_genes: List of marker genes or 'auto' for automatic selection
    - n_epochs: Training epochs (default 1000)
    - density_prior: 'uniform' or 'rna_count_based'
    - device: 'cuda:0' or 'cpu'
    - prefer_scvi: If True, try scvi-tools first (default False)
    - max_cells: Max cells for marker gene selection (default 100000)
    """

    def __init__(
        self,
        mode: str = "clusters",
        marker_genes: str | list[str] = "auto",
        n_epochs: int = 1000,
        density_prior: str = "uniform",
        device: str | None = None,
        prefer_scvi: bool = False,
        max_cells: int = 100000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mode = mode
        self.max_cells = max_cells
        self.marker_genes = marker_genes
        self.n_epochs = n_epochs
        self.density_prior = density_prior
        self.prefer_scvi = prefer_scvi

        # Auto-detect device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Store for later use
        self._snrna_ref = None
        self._spatial_ref = None
        self._mapper = None
        self._ad_map = None
        self._backend_used = None  # Track which implementation was used

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run Tangram mapping with fallback support.

        Tries standalone tangram-sc first, falls back to scvi-tools Tangram if needed.
        """
        print("Tangram: Starting map()...")
        print(f"  snRNA shape: {snrna.shape}, spatial shape: {spatial.shape}")

        # Validate and preprocess
        self.validate_inputs(snrna, spatial)
        snrna, spatial = self.preprocess(snrna, spatial)
        print(f"  After preprocess: snRNA {snrna.shape}, spatial {spatial.shape}")

        # Subsample for marker gene selection (memory intensive)
        if self.max_cells is not None and len(snrna) > self.max_cells:
            print(f"  Subsampling snRNA from {len(snrna)} to {self.max_cells} for marker selection...")
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(snrna))
            _, subsample_idx = train_test_split(
                indices,
                test_size=self.max_cells / len(snrna),
                stratify=snrna.obs["cell_type"],
                random_state=42,
            )
            snrna_for_markers = snrna[subsample_idx].copy()
        else:
            snrna_for_markers = snrna

        # Select marker genes if needed
        if self.marker_genes == "auto":
            print(f"  Selecting marker genes on {len(snrna_for_markers)} cells...")
            import scanpy as sc
            old_verbosity = sc.settings.verbosity
            sc.settings.verbosity = 2  # Show progress info
            marker_genes = self._select_marker_genes(snrna_for_markers)
            sc.settings.verbosity = old_verbosity
            print(f"  Selected {len(marker_genes)} marker genes")
        else:
            marker_genes = list(self.marker_genes)

        # Get common genes
        common_genes = list(set(marker_genes) & set(snrna.var_names) & set(spatial.var_names))
        if len(common_genes) < 50:
            raise ValueError(f"Only {len(common_genes)} common marker genes, need at least 50")

        print(f"Tangram: Using {len(common_genes)} marker genes, device={self.device}")

        # Determine order based on preference
        if self.prefer_scvi:
            methods = [("scvi-tools", self._map_scvi), ("standalone", self._map_standalone)]
        else:
            methods = [("standalone", self._map_standalone), ("scvi-tools", self._map_scvi)]

        last_error = None
        for method_name, method_func in methods:
            try:
                print(f"  Trying {method_name} Tangram...")
                cell_type_proportions = method_func(snrna, spatial, common_genes)
                self._backend_used = method_name
                print(f"  Success with {method_name} Tangram")
                break
            except ImportError as e:
                print(f"  {method_name} Tangram not available: {e}")
                last_error = e
                continue
            except Exception as e:
                print(f"  {method_name} Tangram failed: {e}")
                last_error = e
                continue
        else:
            # Both methods failed
            raise RuntimeError(
                f"Both Tangram implementations failed. Last error: {last_error}"
            ) from last_error

        # Ensure index matches original spatial
        cell_type_proportions.index = spatial.obs_names

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

            # Save mapper matrix if available
            if self._mapper is not None:
                np.save(output_dir / "tangram_mapper.npy", self._mapper)

            # Save cell type predictions
            cell_type_proportions.to_csv(output_dir / "tangram_cell_type_props.csv")

            # Save annotated spatial data
            spatial_annotated = spatial.copy()
            props_array = cell_type_proportions.values if hasattr(cell_type_proportions, 'values') else cell_type_proportions
            spatial_annotated.obsm["tangram_proportions"] = props_array
            if hasattr(cell_type_proportions, 'columns'):
                for ct in cell_type_proportions.columns:
                    spatial_annotated.obs[f"tangram_{ct}"] = cell_type_proportions[ct].values
            spatial_annotated.write_h5ad(output_dir / "tangram_spatial_annotated.h5ad")

            print(f"  Tangram outputs saved to {output_dir}")

        result = BackendMappingResult(
            cell_type_proportions=cell_type_proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata={
                "backend": "tangram",
                "implementation": self._backend_used,
                "mode": self.mode,
                "n_marker_genes": len(common_genes),
                "n_epochs": self.n_epochs,
                "density_prior": self.density_prior,
                "device": self.device,
            },
        )

        return result

    def _map_standalone(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        common_genes: list[str],
    ) -> pd.DataFrame:
        """Run mapping using standalone tangram-sc (PyTorch)."""
        import tangram as tg
        print(f"    tangram-sc version: {getattr(tg, '__version__', 'unknown')}")

        # Make copies to avoid modifying originals
        snrna_pp = snrna.copy()
        spatial_pp = spatial.copy()

        # Preprocess for Tangram
        tg.pp_adatas(snrna_pp, spatial_pp, genes=common_genes)

        # Get cell type key
        cell_type_key = self._get_cell_type_key(snrna_pp)

        # Run mapping
        print(f"    Training ({self.n_epochs} epochs, mode={self.mode})...")
        ad_map = tg.map_cells_to_space(
            snrna_pp,
            spatial_pp,
            mode=self.mode,
            cluster_label=cell_type_key,
            density_prior=self.density_prior,
            num_epochs=self.n_epochs,
            device=self.device,
        )

        # Store results
        self._snrna_ref = snrna
        self._spatial_ref = spatial
        self._ad_map = ad_map
        self._mapper = ad_map.X  # (n_cells, n_spots)

        # Project cell type annotations
        tg.project_cell_annotations(ad_map, spatial_pp, annotation=cell_type_key)

        # Extract cell type proportions
        if "tangram_ct_pred" not in spatial_pp.obsm:
            raise RuntimeError("Tangram did not produce cell type predictions")

        ct_pred = spatial_pp.obsm["tangram_ct_pred"]
        if isinstance(ct_pred, pd.DataFrame):
            return ct_pred
        else:
            cell_types = snrna_pp.obs[cell_type_key].cat.categories.tolist()
            return pd.DataFrame(ct_pred, index=spatial_pp.obs_names, columns=cell_types)

    def _map_scvi(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        common_genes: list[str],
    ) -> pd.DataFrame:
        """Run mapping using scvi-tools Tangram (JAX)."""
        from scvi.external import Tangram
        print("    Using scvi-tools Tangram (JAX backend)")

        # Make copies and subset to common genes
        snrna_pp = snrna[:, common_genes].copy()
        spatial_pp = spatial[:, common_genes].copy()

        # Get cell type key
        cell_type_key = self._get_cell_type_key(snrna_pp)

        # Setup anndata
        Tangram.setup_anndata(snrna_pp, labels_key=cell_type_key)
        Tangram.setup_anndata(spatial_pp)

        # Create and train model
        # scvi-tools Tangram uses 'constrained' mode (similar to 'clusters')
        model = Tangram(snrna_pp, spatial_pp)

        print(f"    Training ({self.n_epochs} epochs)...")
        model.train(max_epochs=self.n_epochs)

        # Store references
        self._snrna_ref = snrna
        self._spatial_ref = spatial
        self._mapper = None  # scvi-tools doesn't expose raw mapper easily

        # Get cell type proportions
        # scvi-tools Tangram stores predictions differently
        proportions = model.get_spatial_mapping()

        cell_types = snrna_pp.obs[cell_type_key].cat.categories.tolist()
        return pd.DataFrame(proportions, index=spatial_pp.obs_names, columns=cell_types)

    def _get_cell_type_key(self, adata: ad.AnnData) -> str:
        """Get the cell type column name from obs."""
        if "cell_type" in adata.obs.columns:
            return "cell_type"
        elif "celltype" in adata.obs.columns:
            return "celltype"
        else:
            raise ValueError("No cell_type column found in obs")

    def _compute_mapping_confidence(
        self,
        cell_type_proportions: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Compute confidence scores from cell type proportion entropy.

        Lower entropy = higher confidence (more certain mapping).
        """
        # Compute entropy per spot
        props = cell_type_proportions.values if hasattr(cell_type_proportions, 'values') else cell_type_proportions
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
            props = result.cell_type_proportions.values if hasattr(result.cell_type_proportions, 'values') else result.cell_type_proportions
            # Cell type entropy (diversity)
            metrics["mean_entropy"] = float(compute_cell_type_entropy(props).mean())
            # Sparsity
            metrics["sparsity"] = float(compute_sparsity(props))
            # Coverage (spots with confident mapping)
            if result.confidence is not None:
                conf = (
                    result.confidence
                    if isinstance(result.confidence, np.ndarray)
                    else result.confidence.values
                )
                metrics["coverage"] = float((conf > 0.5).mean())  # Required key for benchmark
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

        # Make a copy to avoid modifying original
        snrna_copy = snrna.copy()

        # Rank genes per cell type
        sc.tl.rank_genes_groups(
            snrna_copy,
            groupby=cell_type_key,
            method="wilcoxon",
            n_genes=n_genes,
            use_raw=False,
        )

        # Extract top genes per group
        marker_genes = set()
        for group in snrna_copy.uns["rank_genes_groups"]["names"].dtype.names:
            genes = snrna_copy.uns["rank_genes_groups"]["names"][group][:n_genes]
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
            DataFrame of projected expression (spots x genes or spots x 1 if aggregate)
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

        # Project using mapper: spatial_expr = mapper.T @ cell_expr
        # mapper shape: (n_cells, n_spots), so transpose for (n_spots, n_cells)
        projected = self._mapper.T @ gene_expr  # (n_spots, n_genes)

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

    def compute_spatial_statistics(
        self,
        cell_types: list[str] | None = None,
        n_neighs: int = 6,
    ) -> dict[str, Any]:
        """
        Compute spatial statistics using Squidpy.

        Includes spatial autocorrelation (Moran's I).

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

    print(f"Tangram mapping complete. Results saved to {output_dir}")

    return result
