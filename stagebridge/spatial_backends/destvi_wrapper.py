"""
DestVI spatial mapping backend wrapper.

DestVI: Probabilistic VAE-based spatial deconvolution.
Reference: https://docs.scvi-tools.org/en/stable/user_guide/models/destvi.html
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
import anndata as ad

from .base import SpatialBackend, SpatialMappingResult, compute_cell_type_entropy, compute_sparsity


class DestVIBackend(SpatialBackend):
    """
    DestVI spatial mapping wrapper.

    Configuration options:
    - n_latent: Latent dimensionality
    - n_epochs_condsc: Training epochs for conditional scVI
    - n_epochs_destvi: Training epochs for DestVI
    - lr: Learning rate
    """

    def __init__(
        self,
        n_latent: int = 10,
        n_epochs_condsc: int = 200,
        n_epochs_destvi: int = 2500,
        lr: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_latent = n_latent
        self.n_epochs_condsc = n_epochs_condsc
        self.n_epochs_destvi = n_epochs_destvi
        self.lr = lr

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> SpatialMappingResult:
        """Run DestVI mapping."""
        # Validate and preprocess
        self.validate_inputs(snrna, spatial)
        snrna, spatial = self.preprocess(snrna, spatial)

        # Import scvi-tools (lazy import)
        try:
            import scvi
        except ImportError:
            raise ImportError(
                "scvi-tools not installed. Install with: pip install scvi-tools"
            ) from None

        print(f"Running DestVI with {len(snrna)} cells, {len(spatial)} spots...")

        # Setup anndata for scvi
        scvi.model.CondSCVI.setup_anndata(snrna, labels_key="cell_type")
        scvi.model.DestVI.setup_anndata(spatial)

        # Train conditional scVI on snRNA
        print(f"Training CondSCVI for {self.n_epochs_condsc} epochs...")
        sc_model = scvi.model.CondSCVI(snrna, n_latent=self.n_latent)
        sc_model.train(max_epochs=self.n_epochs_condsc, lr=self.lr)

        # Train DestVI on spatial
        print(f"Training DestVI for {self.n_epochs_destvi} epochs...")
        spatial_model = scvi.model.DestVI.from_rna_model(
            spatial,
            sc_model,
        )
        spatial_model.train(max_epochs=self.n_epochs_destvi, lr=self.lr)

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
        upstream_metrics = self.compute_upstream_metrics(
            snrna, spatial, None
        )

        # Save models if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            sc_model.save(output_dir / "condscvi_model", overwrite=True)
            spatial_model.save(output_dir / "destvi_model", overwrite=True)

        result = SpatialMappingResult(
            cell_type_proportions=cell_type_proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata={
                "backend": "destvi",
                "n_latent": self.n_latent,
                "n_epochs_condsc": self.n_epochs_condsc,
                "n_epochs_destvi": self.n_epochs_destvi,
                "lr": self.lr,
            },
        )

        return result

    def compute_upstream_metrics(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: SpatialMappingResult | None,
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
        result: SpatialMappingResult | None,
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


def run_destvi(
    snrna_path: str | Path,
    spatial_path: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> SpatialMappingResult:
    """
    Convenience function to run DestVI mapping.

    Args:
        snrna_path: Path to single-cell h5ad
        spatial_path: Path to spatial h5ad
        output_dir: Where to save results
        **kwargs: Additional DestVI parameters

    Returns:
        SpatialMappingResult
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
