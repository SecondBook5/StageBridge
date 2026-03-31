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
        print("  Using raw counts from layers['counts']")
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

    Note: Uses vamp_prior_p=8 for better deconvolution quality.
    Requires prior="mog" in CondSCVI to avoid 'prior' KeyError
    in scvi-tools >= 1.0 (DestVI.from_rna_model expects 'prior' in module_kwargs).
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

        # Ensure counts layer exists
        snrna = _ensure_counts(snrna)
        spatial = _ensure_counts(spatial)

        # Setup anndata for scvi with batch correction
        # IMPORTANT: layer="counts" is required - without it scvi uses .X which may be normalized
        # Note: Only setup CondSCVI - DestVI.from_rna_model handles spatial setup internally
        CondSCVI.setup_anndata(snrna, layer="counts", labels_key="cell_type", batch_key=snrna_batch)

        # Determine accelerator once
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        # Train conditional scVI on snRNA (without reweighting)
        # IMPORTANT: prior="mog" (mixture of gaussians) is required for DestVI.from_rna_model
        # Without explicit prior, CondSCVI doesn't set the 'prior' key in init_args,
        # causing KeyError when DestVI tries to read it
        print(f"  Training CondSCVI for up to {self.n_epochs_condsc} epochs (early stopping enabled)...")
        sc_model = CondSCVI(snrna, n_latent=self.n_latent, weight_obs=False, prior="mog")
        sc_model.train(
            max_epochs=self.n_epochs_condsc,
            accelerator=accelerator,
            devices=1,
            batch_size=128,
            plan_kwargs={"lr": self.lr},
            train_size=0.9,  # 90% train, 10% validation for early stopping
            early_stopping=True,
            early_stopping_monitor="elbo_validation",
            early_stopping_patience=15,
            check_val_every_n_epoch=1,
        )
        condscvi_history = dict(sc_model.history) if hasattr(sc_model, 'history') else {}
        # Get epoch count from any available key (scvi-tools uses various key names)
        condscvi_epochs_run = max((len(v) for v in condscvi_history.values() if hasattr(v, '__len__')), default=0)
        print(f"  CondSCVI finished after {condscvi_epochs_run} epochs (keys: {list(condscvi_history.keys())[:3]})")

        # Train DestVI on spatial
        # NOTE: Official tutorial does NOT use early stopping for DestVI
        # "reducing the number of epochs leads to decreased performance"
        print(f"  Training DestVI for {self.n_epochs_destvi} epochs...")
        spatial_model = DestVI.from_rna_model(spatial, sc_model, vamp_prior_p=8)
        spatial_model.train(
            max_epochs=self.n_epochs_destvi,
            accelerator=accelerator,
            devices=1,
            batch_size=128,
            plan_kwargs={"lr": self.lr},
        )
        destvi_history = dict(spatial_model.history) if hasattr(spatial_model, 'history') else {}
        # Get epoch count from any available key (scvi-tools uses various key names)
        destvi_epochs_run = max((len(v) for v in destvi_history.values() if hasattr(v, '__len__')), default=0)
        print(f"  DestVI finished after {destvi_epochs_run} epochs (keys: {list(destvi_history.keys())[:3]})")

        # Store models, history, epoch counts, and data for advanced queries
        self.sc_model = sc_model
        self.spatial_model = spatial_model
        self._condscvi_history = condscvi_history
        self._destvi_history = destvi_history
        self._condscvi_epochs_run = condscvi_epochs_run
        self._destvi_epochs_run = destvi_epochs_run
        self._snrna_ref = snrna
        self._spatial_ref = spatial

        # CRITICAL: Save models IMMEDIATELY after training before any post-processing
        # This prevents losing 3+ hours of training if post-processing fails
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print("  Saving trained models...")
            sc_model.save(output_dir / "condscvi_model", overwrite=True)
            spatial_model.save(output_dir / "destvi_model", overwrite=True)
            print("  Models saved successfully")

        # Extract cell type proportions
        proportions = spatial_model.get_proportions()
        cell_types = snrna.obs["cell_type"].cat.categories.tolist()

        # get_proportions() returns DataFrame with proper index and columns (per tutorial)
        if isinstance(proportions, pd.DataFrame):
            cell_type_proportions = proportions
        elif isinstance(proportions, np.ndarray):
            # Fallback for older scvi-tools versions that return array
            cell_type_proportions = pd.DataFrame(
                proportions,
                index=spatial.obs_names,
                columns=cell_types,
            )
        else:
            raise TypeError(
                f"Unexpected return type from get_proportions(): {type(proportions)}. "
                f"Expected DataFrame or ndarray."
            )

        # Compute confidence from proportions (max proportion per spot)
        confidence = pd.Series(
            cell_type_proportions.max(axis=1).values,
            index=cell_type_proportions.index,
            name="confidence",
        )

        # Compute upstream metrics directly from proportions
        from .backend_base import compute_cell_type_entropy, compute_sparsity
        entropy = compute_cell_type_entropy(cell_type_proportions)
        sparsity = compute_sparsity(cell_type_proportions)
        upstream_metrics = {
            "mean_entropy": float(entropy.mean()),
            "std_entropy": float(entropy.std()),
            "sparsity": float(sparsity),
            "coverage": float((confidence > 0.5).mean()),
            "n_spots": spatial.n_obs,
            "n_celltypes": len(cell_types),
        }

        # Save additional outputs if output_dir provided
        # Note: Models already saved immediately after training above
        if output_dir:
            # Save training history / loss curves
            # Filter out scalar values (like kl_weight) that break pd.DataFrame
            def filter_history_lists(hist: dict) -> dict:
                return {k: v for k, v in hist.items() if hasattr(v, '__len__') and len(v) > 0}

            if self._condscvi_history:
                hist_lists = filter_history_lists(self._condscvi_history)
                if hist_lists:
                    condscvi_hist_df = pd.DataFrame(hist_lists)
                    condscvi_hist_df.to_csv(output_dir / "condscvi_training_history.csv", index=False)
            if self._destvi_history:
                hist_lists = filter_history_lists(self._destvi_history)
                if hist_lists:
                    destvi_hist_df = pd.DataFrame(hist_lists)
                    destvi_hist_df.to_csv(output_dir / "destvi_training_history.csv", index=False)

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
                "n_epochs_condsc_max": self.n_epochs_condsc,
                "n_epochs_destvi_max": self.n_epochs_destvi,
                "n_epochs_condsc_actual": self._condscvi_epochs_run,
                "n_epochs_destvi_actual": self._destvi_epochs_run,
                "early_stopping_triggered_condscvi": self._condscvi_epochs_run < self.n_epochs_condsc,
                "early_stopping_triggered_destvi": self._destvi_epochs_run < self.n_epochs_destvi,
                "lr": self.lr,
                "vamp_prior_p": 8,  # VAMP prior for better deconvolution
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

        # Find genes associated with each PC by correlating PC scores with imputed expression
        enriched_genes = {}
        try:
            # Get cell-type-specific imputed expression for spots in this analysis
            ct_expression = self.get_cell_type_specific_expression(cell_type, indices=indices)
            gene_names = ct_expression.columns.tolist()

            from scipy.stats import spearmanr
            for pc_idx in range(2):
                pc_scores = gamma_pca[:, pc_idx]
                correlations = []
                for gene in gene_names:
                    gene_expr = ct_expression[gene].values
                    if gene_expr.std() > 0:
                        corr, pval = spearmanr(pc_scores, gene_expr)
                        correlations.append((gene, corr, pval))
                # Sort by absolute correlation
                correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                enriched_genes[f"PC{pc_idx + 1}"] = [
                    {"gene": g, "correlation": c, "pval": p}
                    for g, c, p in correlations[:20]
                ]
        except Exception as e:
            # If imputation fails, skip gene enrichment
            enriched_genes = {"error": str(e)}

        result = {
            "gamma_pca": gamma_pca,
            "pca_model": pca,
            "explained_variance": pca.explained_variance_ratio_,
            "spatial_coords": coords,
            "spot_indices": indices,
            "cell_type": cell_type,
            "enriched_genes": enriched_genes,
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
