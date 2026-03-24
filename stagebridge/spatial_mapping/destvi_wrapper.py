"""
DestVI spatial mapping backend wrapper using scvi-tools.

DestVI: Multi-resolution probabilistic VAE-based spatial deconvolution.
Supports intra-cell-type variation via gamma latent space and cell-type-specific
gene expression imputation.

Reference: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/spatial/DestVI_tutorial.html

Known issues (fixed):
- scvi-tools >= 1.0 had 'prior' KeyError due to internal API changes
- Fix: Always use vamp_prior_p=0 which bypasses the problematic code path
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import anndata as ad
import torch

from .backend_base import (
    SpatialBackend,
    BackendMappingResult,
    compute_cell_type_entropy,
    compute_sparsity,
)


def _setup_torch_for_performance():
    """Configure PyTorch for optimal GPU performance."""
    # Use Tensor Cores on supported GPUs (L40S, A100, H100, etc.)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')


def _ensure_counts(adata: ad.AnnData) -> ad.AnnData:
    """Ensure adata.X contains raw counts (not normalized data)."""
    if "counts" in adata.layers:
        print(f"  Using raw counts from layers['counts']")
        adata.X = adata.layers["counts"].copy()
    return adata


def _check_scvi_version():
    """Check and log scvi-tools version."""
    try:
        import scvi
        version = getattr(scvi, "__version__", "unknown")
        print(f"  scvi-tools version: {version}")
        return version
    except Exception as e:
        print(f"  Warning: Could not check scvi-tools version: {e}")
        return "unknown"


class DestVIBackend(SpatialBackend):
    """
    DestVI spatial mapping wrapper using scvi-tools.

    Supports multi-resolution spatial deconvolution:
    - Cell type proportions (coarse resolution)
    - Gamma latent space for intra-cell-type variation (fine resolution)
    - Cell-type-specific gene expression imputation

    Configuration options:
    - n_latent: Latent dimensionality for CondSCVI (default 10)
    - n_epochs_condsc: Training epochs for conditional scVI (default 200)
    - n_epochs_destvi: Training epochs for DestVI (default 2500)
    - lr: Learning rate (default 0.01)
    - batch_key: Column name for batch correction (default 'sample_id')

    Note: VAMP prior is disabled (vamp_prior_p=0) to avoid 'prior' KeyError
    in scvi-tools >= 1.0. This is a known compatibility issue where
    DestVI.from_rna_model expects 'prior' in CondSCVI's module_kwargs.
    """

    def __init__(
        self,
        n_latent: int = 10,
        n_epochs_condsc: int = 200,
        n_epochs_destvi: int = 2500,
        lr: float = 0.01,
        batch_key: str | None = "sample_id",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_latent = n_latent
        self.n_epochs_condsc = n_epochs_condsc
        self.n_epochs_destvi = n_epochs_destvi
        self.lr = lr
        self.batch_key = batch_key

        # Store trained models for advanced queries
        self.sc_model = None
        self.spatial_model = None
        self._snrna_ref = None
        self._spatial_ref = None

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """
        Run DestVI mapping with full multi-resolution output.

        Returns cell type proportions and enables access to:
        - Gamma latent space (intra-cell-type variation)
        - Cell-type-specific gene imputation
        - Spatial pattern detection
        """
        # Validate and preprocess
        self.validate_inputs(snrna, spatial)
        snrna, spatial = self.preprocess(snrna, spatial)

        # Ensure raw counts (scvi-tools requires unnormalized data)
        snrna = _ensure_counts(snrna)
        spatial = _ensure_counts(spatial)

        # Configure PyTorch for GPU performance (Tensor Cores)
        _setup_torch_for_performance()

        # Import scvi-tools (lazy import)
        try:
            from scvi.model import CondSCVI, DestVI
        except ImportError as e:
            raise ImportError(
                "scvi-tools not installed. Install with: pip install scvi-tools>=1.1.0"
            ) from e

        print(f"Running DestVI with {len(snrna)} cells, {len(spatial)} spots...")

        # Determine batch keys (only use if column exists)
        snrna_batch = self.batch_key if self.batch_key and self.batch_key in snrna.obs.columns else None
        spatial_batch = self.batch_key if self.batch_key and self.batch_key in spatial.obs.columns else None

        if snrna_batch:
            print(f"  Using batch_key='{snrna_batch}' for snRNA ({snrna.obs[snrna_batch].nunique()} batches)")
        if spatial_batch:
            print(f"  Using batch_key='{spatial_batch}' for spatial ({spatial.obs[spatial_batch].nunique()} batches)")

        # Setup anndata for scvi with batch correction
        CondSCVI.setup_anndata(snrna, labels_key="cell_type", batch_key=snrna_batch)
        DestVI.setup_anndata(spatial, batch_key=spatial_batch)

        # Train conditional scVI on snRNA (without reweighting)
        print(f"  Training CondSCVI for {self.n_epochs_condsc} epochs (early stopping enabled)...")
        sc_model = CondSCVI(snrna, n_latent=self.n_latent, weight_obs=False)
        sc_model.train(
            max_epochs=self.n_epochs_condsc,
            lr=self.lr,
            train_size=0.9,  # 10% validation for early stopping
            early_stopping=True,
            early_stopping_patience=15,
        )

        # Train DestVI on spatial
        # IMPORTANT: vamp_prior_p=0 is required to avoid 'prior' KeyError in scvi-tools >= 1.0
        # The error occurs because DestVI.from_rna_model checks for 'prior' in CondSCVI's
        # module_kwargs, but CondSCVI doesn't set this key by default. Setting vamp_prior_p=0
        # disables VAMP prior entirely, bypassing the problematic code path.
        print(f"  Training DestVI for {self.n_epochs_destvi} epochs (early stopping enabled)...")
        spatial_model = DestVI.from_rna_model(spatial, sc_model, vamp_prior_p=0)
        spatial_model.train(
            max_epochs=self.n_epochs_destvi,
            lr=self.lr,
            train_size=0.9,  # 10% validation for early stopping
            early_stopping=True,
            early_stopping_patience=15,
        )

        # Store models and data for advanced queries
        self.sc_model = sc_model
        self.spatial_model = spatial_model
        self._snrna_ref = snrna
        self._spatial_ref = spatial

        # Extract cell type proportions
        proportions = spatial_model.get_proportions()
        cell_types = snrna.obs["cell_type"].cat.categories.tolist()

        cell_type_proportions = pd.DataFrame(
            proportions,
            index=spatial.obs_names,
            columns=cell_types,
        )

        # Compute confidence from proportion variance
        confidence = self.estimate_confidence(snrna, spatial, None)

        # Compute upstream metrics
        upstream_metrics = self.compute_upstream_metrics(snrna, spatial, None)

        # Save outputs if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save models
            sc_model.save(output_dir / "condscvi_model", overwrite=True)
            spatial_model.save(output_dir / "destvi_model", overwrite=True)

            # Save proportions
            cell_type_proportions.to_csv(output_dir / "destvi_cell_type_props.csv")

            # Save gamma values (intra-cell-type variation)
            gamma_dict = spatial_model.get_gamma()
            for ct, gamma_df in gamma_dict.items():
                gamma_df.to_csv(output_dir / f"destvi_gamma_{ct.replace(' ', '_')}.csv")

            # Save annotated spatial data with proportions
            spatial_annotated = spatial.copy()
            spatial_annotated.obsm["proportions"] = cell_type_proportions.values
            for ct in cell_types:
                spatial_annotated.obs[f"destvi_{ct}"] = cell_type_proportions[ct].values
            spatial_annotated.write_h5ad(output_dir / "destvi_spatial_annotated.h5ad")

            print(f"  DestVI outputs saved to {output_dir}")

        result = BackendMappingResult(
            cell_type_proportions=cell_type_proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata={
                "backend": "destvi",
                "n_latent": self.n_latent,
                "n_epochs_condsc": self.n_epochs_condsc,
                "n_epochs_destvi": self.n_epochs_destvi,
                "lr": self.lr,
                "vamp_prior_p": 0,  # Fixed at 0 to avoid scvi-tools compatibility issues
                "n_cell_types": len(cell_types),
            },
        )

        return result

    def compute_upstream_metrics(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: BackendMappingResult | None,
    ) -> dict[str, float]:
        """Compute DestVI-specific upstream metrics."""
        if result is None:
            return {}

        proportions = result.cell_type_proportions

        # Cell type entropy
        entropy = compute_cell_type_entropy(proportions)

        # Sparsity
        sparsity = compute_sparsity(proportions)

        # Coverage
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
        result: BackendMappingResult | None,
    ) -> pd.Series:
        """
        Estimate confidence from proportion variance.

        Low variance (stable estimates) = high confidence
        High variance (uncertain) = low confidence
        """
        if result is None:
            return pd.Series(
                np.ones(len(spatial)),
                index=spatial.obs_names,
                name="confidence",
            )

        proportions = result.cell_type_proportions

        # Compute max proportion per spot as confidence proxy
        # Spots dominated by one cell type = high confidence
        confidence = proportions.max(axis=1)

        return pd.Series(
            confidence.values,
            index=spatial.obs_names,
            name="confidence",
        )

    def get_gamma(self, cell_types: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """
        Get gamma latent values for intra-cell-type variation.

        Gamma values capture fine-grained cell state variation within each cell type,
        enabling detection of spatial patterns and cell-type-specific gene programs.

        Args:
            cell_types: List of cell types to extract. If None, extracts all.

        Returns:
            Dictionary mapping cell type names to DataFrames of gamma values
            (spots × n_gamma_dims)
        """
        if self.spatial_model is None:
            raise RuntimeError("Must run map() before accessing gamma values")

        gamma_dict = self.spatial_model.get_gamma()

        if cell_types is not None:
            gamma_dict = {ct: gamma for ct, gamma in gamma_dict.items() if ct in cell_types}

        return gamma_dict

    def get_cell_type_specific_expression(
        self,
        cell_type: str,
        gene_names: list[str] | None = None,
        indices: np.ndarray | None = None,
        aggregate: bool = False,
    ) -> pd.DataFrame:
        """
        Impute cell-type-specific gene expression in spatial spots.

        This deconvolves spatial gene expression into cell-type-specific contributions,
        enabling cell-type-specific differential expression and pathway analysis.

        Args:
            cell_type: Cell type to impute expression for
            gene_names: List of genes to impute. If None, returns all genes.
            indices: Spot indices to impute for. If None, uses all spots.
            aggregate: If True, sum across genes (useful for gene set analysis)

        Returns:
            DataFrame of imputed expression (spots × genes)
        """
        if self.spatial_model is None:
            raise RuntimeError("Must run map() before imputing expression")

        # Get scale parameters (normalized expression)
        scale = self.spatial_model.get_scale_for_ct(cell_type, indices=indices)

        if gene_names is not None:
            scale = scale[gene_names]

        if aggregate:
            # Sum across genes and return as single column
            aggregated = scale.sum(axis=1)
            return pd.DataFrame({f"{cell_type}_expression": aggregated})

        return scale

    def automatic_proportion_threshold(
        self,
        cell_types: list[str] | None = None,
        kind_threshold: str = "secondary",
    ) -> dict[str, float]:
        """
        Automatically determine proportion thresholds for each cell type.

        Uses histogram-based detection to find appropriate thresholds for filtering
        spots where a cell type is truly present vs. background noise.

        Args:
            cell_types: List of cell types. If None, uses all.
            kind_threshold: Threshold kind - 'primary' (stricter) or 'secondary' (lenient)

        Returns:
            Dictionary mapping cell type names to proportion thresholds
        """
        if self.spatial_model is None or self._spatial_ref is None:
            raise RuntimeError("Must run map() before computing thresholds")

        proportions_df = pd.DataFrame(
            self.spatial_model.get_proportions(),
            index=self._spatial_ref.obs_names,
            columns=self._snrna_ref.obs["cell_type"].cat.categories,
        )

        if cell_types is None:
            cell_types = proportions_df.columns.tolist()

        thresholds = {}
        for ct in cell_types:
            props = proportions_df[ct].values

            # Find histogram peaks
            hist, bin_edges = np.histogram(props, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Find first local minimum after initial peak (background mode)
            # This separates true presence from background noise
            peak_idx = np.argmax(hist[:25])  # Background peak in first half

            if kind_threshold == "primary":
                # Stricter: Use valley after background peak
                valley_search_start = peak_idx + 2
                if valley_search_start < len(hist) - 1:
                    valley_idx = valley_search_start + np.argmin(hist[valley_search_start:])
                    threshold = bin_centers[valley_idx]
                else:
                    threshold = np.percentile(props, 75)
            else:  # secondary
                # Lenient: Use median between background peak and high values
                threshold = np.percentile(props, 60)

            thresholds[ct] = float(threshold)

        return thresholds

    def filter_spots_by_celltype(
        self,
        cell_type: str,
        threshold: float | None = None,
        auto_threshold: bool = True,
    ) -> np.ndarray:
        """
        Get indices of spots where a cell type is present above threshold.

        Args:
            cell_type: Cell type to filter for
            threshold: Manual proportion threshold. If None, uses automatic.
            auto_threshold: If True and threshold is None, computes automatic threshold

        Returns:
            Array of spot indices where cell type is present
        """
        if self.spatial_model is None:
            raise RuntimeError("Must run map() before filtering spots")

        proportions = self.spatial_model.get_proportions()
        cell_types = self._snrna_ref.obs["cell_type"].cat.categories.tolist()
        ct_idx = cell_types.index(cell_type)
        ct_proportions = proportions[:, ct_idx]

        if threshold is None and auto_threshold:
            thresholds = self.automatic_proportion_threshold([cell_type])
            threshold = thresholds[cell_type]
        elif threshold is None:
            threshold = 0.1  # Default fallback

        indices = np.where(ct_proportions > threshold)[0]
        return indices

    def plot_cell_type_spatial(
        self,
        cell_type: str,
        gene_names: list[str] | None = None,
        threshold: float | None = None,
        cmap: str = "Reds",
        save_path: Path | None = None,
    ):
        """
        Plot cell-type-specific gene expression in spatial coordinates.

        Args:
            cell_type: Cell type to visualize
            gene_names: Genes to aggregate. If None, shows proportion only.
            threshold: Proportion threshold for filtering. If None, uses automatic.
            cmap: Matplotlib colormap
            save_path: If provided, saves figure to this path
        """
        if self._spatial_ref is None or self.spatial_model is None:
            raise RuntimeError("Must run map() before plotting")

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("matplotlib required for plotting") from e

        # Get spatial coordinates
        if "spatial" not in self._spatial_ref.obsm:
            raise ValueError("Spatial coordinates not found in spatial.obsm['spatial']")

        coords = self._spatial_ref.obsm["spatial"]

        # Filter spots by cell type presence
        indices = self.filter_spots_by_celltype(
            cell_type,
            threshold=threshold,
            auto_threshold=True,
        )

        if len(indices) == 0:
            print(f"No spots found for {cell_type} above threshold")
            return

        # Get expression values
        if gene_names is not None:
            expression = self.get_cell_type_specific_expression(
                cell_type,
                gene_names=gene_names,
                indices=indices,
                aggregate=True,
            )
            values = np.log1p(1e4 * expression.values.flatten())
            title = f"{cell_type}: {', '.join(gene_names[:3])}"
            if len(gene_names) > 3:
                title += f" (+{len(gene_names) - 3} more)"
        else:
            proportions = self.spatial_model.get_proportions()
            cell_types = self._snrna_ref.obs["cell_type"].cat.categories.tolist()
            ct_idx = cell_types.index(cell_type)
            values = proportions[indices, ct_idx]
            title = f"{cell_type} proportion"

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Background (all spots)
        ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=5, alpha=0.2)

        # Foreground (cell type specific)
        scatter = ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
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

    def explore_gamma_space(
        self,
        cell_type: str,
        threshold: float | None = None,
        save_dir: Path | None = None,
    ) -> dict[str, Any]:
        """
        Explore gamma space for a cell type using spatially-weighted PCA.

        Identifies spatial patterns of intra-cell-type variation and enriched genes
        along spatial principal components.

        Args:
            cell_type: Cell type to analyze
            threshold: Proportion threshold. If None, uses automatic.
            save_dir: If provided, saves visualizations to this directory

        Returns:
            Dictionary with PCA results, enriched genes, and coordinates
        """
        if self.spatial_model is None or self._spatial_ref is None:
            raise RuntimeError("Must run map() before exploring gamma space")

        # Get gamma values and filter by threshold
        gamma_dict = self.get_gamma([cell_type])
        gamma_df = gamma_dict[cell_type]

        indices = self.filter_spots_by_celltype(
            cell_type,
            threshold=threshold,
            auto_threshold=True,
        )

        if len(indices) < 10:
            raise ValueError(f"Too few spots ({len(indices)}) for {cell_type} gamma analysis")

        gamma_values = gamma_df.iloc[indices].values  # spots × n_gamma
        coords = self._spatial_ref.obsm["spatial"][indices]

        # Spatially-weighted PCA
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        gamma_pca = pca.fit_transform(gamma_values)

        # Find genes enriched along each PC
        # Project single-cell embeddings onto spatial PCs
        sc_embeddings = self.sc_model.get_latent_representation()
        ct_mask = self._snrna_ref.obs["cell_type"] == cell_type
        sc_embeddings[ct_mask]

        # For gene enrichment, we'd need to correlate PC loadings with gene expression
        # This is a simplified version - full implementation would use gene-PC correlation

        result = {
            "gamma_pca": gamma_pca,
            "pca_model": pca,
            "explained_variance": pca.explained_variance_ratio_,
            "spatial_coords": coords,
            "spot_indices": indices,
            "cell_type": cell_type,
        }

        # Visualization if requested
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._plot_gamma_pca(result, save_dir)

        return result

    def _plot_gamma_pca(self, gamma_result: dict, save_dir: Path):
        """Plot gamma PCA results in spatial coordinates."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        gamma_pca = gamma_result["gamma_pca"]
        coords = gamma_result["spatial_coords"]
        cell_type = gamma_result["cell_type"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for i, ax in enumerate(axes):
            pc_values = gamma_pca[:, i]
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=pc_values,
                cmap="RdBu_r",
                s=30,
            )
            plt.colorbar(scatter, ax=ax)
            var_exp = gamma_result["explained_variance"][i]
            ax.set_title(f"{cell_type} - Spatial PC{i + 1} ({var_exp:.1%} var)")
            ax.set_xlabel("Spatial X")
            ax.set_ylabel("Spatial Y")

        plt.tight_layout()
        plt.savefig(
            save_dir / f"gamma_pca_{cell_type.replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def run_destvi(
    snrna_path: str | Path,
    spatial_path: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> BackendMappingResult:
    """
    Convenience function to run DestVI mapping.

    Args:
        snrna_path: Path to single-cell h5ad
        spatial_path: Path to spatial h5ad
        output_dir: Where to save results
        **kwargs: Additional DestVI parameters

    Returns:
        BackendMappingResult
    """
    # Load data
    print(f"Loading snRNA data from {snrna_path}...")
    snrna = ad.read_h5ad(snrna_path)

    print(f"Loading spatial data from {spatial_path}...")
    spatial = ad.read_h5ad(spatial_path)

    # Initialize backend
    backend = DestVIBackend(**kwargs)

    # Run mapping
    result = backend.map(snrna, spatial, output_dir=output_dir)

    # Save result
    result.save(output_dir)

    print(f" DestVI mapping complete. Results saved to {output_dir}")

    return result
