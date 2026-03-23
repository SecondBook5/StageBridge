"""
TACCO spatial mapping backend wrapper.

TACCO: Transfer of cell-type Annotations with Compositional bias Correction using Optimal transport.
Reference: https://github.com/simonwm/tacco
"""

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

from .backend_base import (
    SpatialBackend,
    BackendMappingResult,
    compute_cell_type_entropy,
    compute_sparsity,
)


class TACCOBackend(SpatialBackend):
    """
    TACCO spatial mapping wrapper.

    Configuration options:
    - method: TACCO method ('OT', 'NMFreg', or 'NNLS')
    - epsilon: Entropic regularization for OT
    - lamb: Regularization parameter
    """

    def __init__(
        self,
        method: str = "OT",
        epsilon: float = 5e-3,
        lamb: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.method = method
        self.epsilon = epsilon
        self.lamb = lamb

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run TACCO mapping."""
        # Validate and preprocess
        self.validate_inputs(snrna, spatial)
        snrna, spatial = self.preprocess(snrna, spatial)

        # Import tacco (lazy import)
        try:
            import tacco as tc
        except ImportError:
            raise ImportError("TACCO not installed. Install with: pip install tacco") from None

        print(f"Running TACCO with method={self.method}...")

        # Run TACCO annotation
        tc.tl.annotate(
            spatial,
            snrna,
            annotation_key="cell_type",
            result_key="tacco_celltype",
            method=self.method,
            epsilon=self.epsilon if self.method == "OT" else None,
            lamb=self.lamb if self.method == "NMFreg" else None,
        )

        # Extract cell type proportions
        # TACCO stores proportions in .obsm['tacco_celltype']
        if "tacco_celltype" in spatial.obsm:
            proportions_array = spatial.obsm["tacco_celltype"]
            cell_types = snrna.obs["cell_type"].cat.categories.tolist()

            cell_type_proportions = pd.DataFrame(
                proportions_array,
                index=spatial.obs_names,
                columns=cell_types,
            )
        else:
            # Fallback: create one-hot from predicted labels
            predicted = spatial.obs["tacco_celltype"].values
            cell_types = sorted(snrna.obs["cell_type"].unique())

            proportions_array = np.zeros((len(spatial), len(cell_types)))
            for i, ct in enumerate(cell_types):
                proportions_array[:, i] = (predicted == ct).astype(float)

            cell_type_proportions = pd.DataFrame(
                proportions_array,
                index=spatial.obs_names,
                columns=cell_types,
            )

        # Compute confidence
        confidence = self.estimate_confidence(snrna, spatial, None)

        # Compute upstream metrics
        upstream_metrics = self.compute_upstream_metrics(snrna, spatial, None)

        # Save if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            spatial.write_h5ad(output_dir / "tacco_annotated_spatial.h5ad")

        result = BackendMappingResult(
            cell_type_proportions=cell_type_proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata={
                "backend": "tacco",
                "method": self.method,
                "epsilon": self.epsilon if self.method == "OT" else None,
                "lamb": self.lamb if self.method == "NMFreg" else None,
            },
        )

        return result

    def compute_upstream_metrics(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: BackendMappingResult | None,
    ) -> dict[str, float]:
        """Compute TACCO-specific upstream metrics."""
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
        Estimate confidence from proportion certainty.

        Similar to other backends: high max proportion = high confidence
        """
        if result is None:
            return pd.Series(
                np.ones(len(spatial)),
                index=spatial.obs_names,
                name="confidence",
            )

        proportions = result.cell_type_proportions

        # Max proportion as confidence
        confidence = proportions.max(axis=1)

        return pd.Series(
            confidence.values,
            index=spatial.obs_names,
            name="confidence",
        )


def run_tacco(
    snrna_path: str | Path,
    spatial_path: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> BackendMappingResult:
    """
    Convenience function to run TACCO mapping.

    Args:
        snrna_path: Path to single-cell h5ad
        spatial_path: Path to spatial h5ad
        output_dir: Where to save results
        **kwargs: Additional TACCO parameters

    Returns:
        BackendMappingResult
    """
    # Load data
    print(f"Loading snRNA data from {snrna_path}...")
    snrna = ad.read_h5ad(snrna_path)

    print(f"Loading spatial data from {spatial_path}...")
    spatial = ad.read_h5ad(spatial_path)

    # Initialize backend
    backend = TACCOBackend(**kwargs)

    # Run mapping
    result = backend.map(snrna, spatial, output_dir=output_dir)

    # Save result
    result.save(output_dir)

    print(f" TACCO mapping complete. Results saved to {output_dir}")

    return result
