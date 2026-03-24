"""
Cell2location spatial mapping backend wrapper.

Cell2location is a Bayesian model for mapping single-cell reference to spatial
transcriptomics data. It estimates absolute cell abundance per location using
a negative binomial regression model.

Reference: Kleshchevnikov et al., Nature Biotechnology 2022
https://cell2location.readthedocs.io/
"""

from pathlib import Path
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
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')


def _ensure_counts(adata: ad.AnnData) -> ad.AnnData:
    """Ensure adata.X contains raw counts (not normalized data)."""
    if "counts" in adata.layers:
        print(f"  Using raw counts from layers['counts']")
        adata.X = adata.layers["counts"].copy()
    return adata


class Cell2locationBackend(SpatialBackend):
    """
    Cell2location spatial mapping wrapper.

    Two-stage Bayesian deconvolution:
    1. Train reference signature model on snRNA-seq
    2. Train spatial model to estimate cell abundances

    Configuration options:
    - n_cells_per_location: Expected cells per spot (default 30 for Visium)
    - detection_alpha: Prior on detection efficiency (default 20)
    - max_epochs_ref: Max epochs for reference model (default 250)
    - max_epochs_spatial: Max epochs for spatial model (default 30000)
    - batch_size: Training batch size (default 2500)
    """

    def __init__(
        self,
        n_cells_per_location: int = 30,
        detection_alpha: float = 20.0,
        max_epochs_ref: int = 250,
        max_epochs_spatial: int = 2500,  # Reduced from 30k (no early stopping available)
        batch_size: int = 2500,
        accelerator: str = "auto",
        batch_key: str | None = "sample_id",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_cells_per_location = n_cells_per_location
        self.detection_alpha = detection_alpha
        self.max_epochs_ref = max_epochs_ref
        self.max_epochs_spatial = max_epochs_spatial
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.batch_key = batch_key

        # Store trained models
        self.ref_model = None
        self.spatial_model = None

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run Cell2location mapping."""
        # Validate and preprocess
        self.validate_inputs(snrna, spatial)
        snrna, spatial = self.preprocess(snrna, spatial)

        # Configure PyTorch for GPU performance (Tensor Cores)
        _setup_torch_for_performance()

        # Import cell2location
        try:
            import cell2location
            from cell2location.models import RegressionModel
        except ImportError as e:
            raise ImportError(
                "cell2location not installed. Install with: pip install cell2location"
            ) from e

        # Ensure cell_type column exists
        if "cell_type" not in snrna.obs.columns:
            raise ValueError("snRNA-seq data must have 'cell_type' column in obs")

        # Get common genes
        common_genes = list(set(snrna.var_names) & set(spatial.var_names))
        if len(common_genes) < 100:
            raise ValueError(f"Only {len(common_genes)} common genes found, need at least 100")

        print(f"Cell2location: Using {len(common_genes)} common genes")

        # Subset to common genes
        snrna_sub = snrna[:, common_genes].copy()
        spatial_sub = spatial[:, common_genes].copy()

        # Ensure raw counts (cell2location needs counts, not normalized)
        if "counts" in snrna_sub.layers:
            snrna_sub.X = snrna_sub.layers["counts"]
        if "counts" in spatial_sub.layers:
            spatial_sub.X = spatial_sub.layers["counts"]

        # =====================================================================
        # Stage 1: Train reference signature model
        # =====================================================================
        print("Cell2location Stage 1: Training reference signature model...")

        # Setup reference model with batch correction if available
        snrna_batch = self.batch_key if self.batch_key and self.batch_key in snrna_sub.obs.columns else None
        if snrna_batch:
            print(f"  Using batch_key='{snrna_batch}' for reference ({snrna_sub.obs[snrna_batch].nunique()} batches)")

        cell2location.models.RegressionModel.setup_anndata(
            adata=snrna_sub,
            labels_key="cell_type",
            batch_key=snrna_batch,
        )

        self.ref_model = RegressionModel(snrna_sub)

        # Train reference model
        # Note: Cell2location doesn't implement validation_step, so no early stopping
        self.ref_model.train(
            max_epochs=self.max_epochs_ref,
            batch_size=self.batch_size,
            lr=0.002,
            accelerator=self.accelerator,
            datamodule_kwargs={"num_workers": 4},
        )

        # Export estimated cell type signatures
        snrna_sub = self.ref_model.export_posterior(
            snrna_sub,
            sample_kwargs={
                "num_samples": 1000,
                "batch_size": self.batch_size,
            },
        )

        # Get reference cell type signature
        if "means_per_cluster_mu_fg" in snrna_sub.varm:
            ref_sig = snrna_sub.varm["means_per_cluster_mu_fg"]
        else:
            raise RuntimeError("Reference model did not produce signatures")

        # Free GPU memory from reference model before Stage 2
        # Precautionary with per-sample processing (~11k spots), critical with merged data
        import torch
        del self.ref_model
        self.ref_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("  Cleared GPU memory after Stage 1")

        # =====================================================================
        # Stage 2: Train spatial deconvolution model
        # =====================================================================
        print("Cell2location Stage 2: Training spatial deconvolution model...")

        # Setup spatial model with batch correction if available
        spatial_batch = self.batch_key if self.batch_key and self.batch_key in spatial_sub.obs.columns else None
        if spatial_batch:
            print(f"  Using batch_key='{spatial_batch}' ({spatial_sub.obs[spatial_batch].nunique()} batches)")

        cell2location.models.Cell2location.setup_anndata(
            adata=spatial_sub,
            batch_key=spatial_batch,
        )

        self.spatial_model = cell2location.models.Cell2location(
            spatial_sub,
            cell_state_df=ref_sig,
            N_cells_per_location=self.n_cells_per_location,
            detection_alpha=self.detection_alpha,
        )

        # Train spatial model
        # Note: Cell2location doesn't implement validation_step, so no early stopping
        self.spatial_model.train(
            max_epochs=self.max_epochs_spatial,
            batch_size=None,  # Use full batch for spatial
            accelerator=self.accelerator,
            datamodule_kwargs={"num_workers": 4},
        )

        # Export posterior estimates
        spatial_sub = self.spatial_model.export_posterior(
            spatial_sub,
            sample_kwargs={
                "num_samples": 1000,
                "batch_size": self.batch_size if self.batch_size else spatial_sub.n_obs,
            },
        )

        # =====================================================================
        # Extract results
        # =====================================================================

        # Get cell type abundances (posterior mean)
        abundance_key = "q05_cell_abundance_w_sf"
        if abundance_key not in spatial_sub.obsm:
            # Try alternative key
            for key in spatial_sub.obsm.keys():
                if "abundance" in key.lower():
                    abundance_key = key
                    break

        if abundance_key not in spatial_sub.obsm:
            raise RuntimeError("Could not find cell abundance estimates in spatial model output")

        abundances = spatial_sub.obsm[abundance_key]

        # Convert to proportions (normalize per spot)
        if isinstance(abundances, pd.DataFrame):
            proportions = abundances.div(abundances.sum(axis=1), axis=0)
        else:
            proportions = pd.DataFrame(
                abundances / abundances.sum(axis=1, keepdims=True),
                index=spatial_sub.obs_names,
                columns=ref_sig.columns
                if hasattr(ref_sig, "columns")
                else [f"type_{i}" for i in range(abundances.shape[1])],
            )

        # Compute confidence (based on total abundance / expected)
        total_abundance = (
            abundances.sum(axis=1)
            if isinstance(abundances, np.ndarray)
            else abundances.sum(axis=1).values
        )
        confidence = pd.Series(
            np.clip(total_abundance / self.n_cells_per_location, 0, 1),
            index=spatial_sub.obs_names,
            name="confidence",
        )

        # Compute upstream metrics
        upstream_metrics = {
            "n_spots": spatial_sub.n_obs,
            "n_celltypes": proportions.shape[1],
            "n_genes_used": len(common_genes),
            "mean_entropy": float(compute_cell_type_entropy(proportions).mean()),
            "std_entropy": float(compute_cell_type_entropy(proportions).std()),
            "sparsity": float(compute_sparsity(proportions)),
            "coverage": float((confidence > 0.5).mean()),  # Required for benchmark comparison
            "mean_total_abundance": float(total_abundance.mean()),
            "ref_model_epochs": self.max_epochs_ref,
            "spatial_model_epochs": self.max_epochs_spatial,
        }

        # Build result
        result = BackendMappingResult(
            cell_type_proportions=proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata={
                "backend": "cell2location",
                "version": cell2location.__version__
                if hasattr(cell2location, "__version__")
                else "unknown",
                "n_cells_per_location": self.n_cells_per_location,
                "detection_alpha": self.detection_alpha,
                "n_common_genes": len(common_genes),
            },
        )

        # Save if output_dir provided
        if output_dir:
            result.save(output_dir)

            # Also save cell2location-specific outputs
            output_dir = Path(output_dir)
            if isinstance(abundances, pd.DataFrame):
                abundances.to_parquet(output_dir / "cell_abundances.parquet")
            else:
                pd.DataFrame(
                    abundances,
                    index=spatial_sub.obs_names,
                ).to_parquet(output_dir / "cell_abundances.parquet")

        return result

    def get_abundance_estimates(self) -> pd.DataFrame | None:
        """Get raw (unnormalized) cell abundance estimates."""
        if self.spatial_model is None:
            return None

        adata = self.spatial_model.adata
        for key in ["q05_cell_abundance_w_sf", "means_cell_abundance_w_sf"]:
            if key in adata.obsm:
                return pd.DataFrame(
                    adata.obsm[key],
                    index=adata.obs_names,
                )
        return None

    def compute_upstream_metrics(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: BackendMappingResult | None,
    ) -> dict[str, float]:
        """Compute upstream quality metrics for Cell2location mapping."""
        metrics = {}

        if result is not None and result.cell_type_proportions is not None:
            props = result.cell_type_proportions
            # Cell type entropy (diversity)
            metrics["mean_entropy"] = float(compute_cell_type_entropy(props).mean())
            # Sparsity
            metrics["sparsity"] = float(compute_sparsity(props))
            # Coverage (spots with confident mapping)
            if result.confidence is not None:
                conf = (
                    result.confidence.values
                    if isinstance(result.confidence, pd.Series)
                    else result.confidence
                )
                metrics["coverage_0.5"] = float((conf > 0.5).mean())
                metrics["mean_confidence"] = float(np.mean(conf))

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

    def plot_spatial_abundances(
        self,
        spatial: ad.AnnData,
        cell_types: list[str] | None = None,
        **kwargs,
    ):
        """Plot cell type abundances spatially using squidpy."""
        try:
            import squidpy as sq
        except ImportError as err:
            raise ImportError("squidpy required for plotting: pip install squidpy") from err

        abundances = self.get_abundance_estimates()
        if abundances is None:
            raise RuntimeError("No abundance estimates available. Run map() first.")

        # Add abundances to spatial adata
        for col in abundances.columns:
            spatial.obs[f"c2l_{col}"] = abundances[col].values

        # Select cell types to plot
        if cell_types is None:
            cell_types = list(abundances.columns)[:6]  # Top 6 by default

        plot_cols = [f"c2l_{ct}" for ct in cell_types if f"c2l_{ct}" in spatial.obs.columns]

        return sq.pl.spatial_scatter(
            spatial,
            color=plot_cols,
            **kwargs,
        )
