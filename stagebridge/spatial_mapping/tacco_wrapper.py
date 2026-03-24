"""
TACCO spatial mapping backend wrapper.

TACCO: Transfer of cell-type Annotations with Compositional bias Correction using Optimal transport.
Reference: https://github.com/simonwm/tacco
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import anndata as ad

# Force scipy backend to avoid MKL 32-bit integer overflow on large matrices
# This must be set BEFORE importing tacco/POT/numpy with MKL
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
    - max_cells: Max cells to use from reference (subsampling). Default 150000.
                 Set to None to disable. Helps avoid MKL 32-bit integer overflow.
                 With per-sample spatial (~11k spots), 150k cells is safe.
    """

    def __init__(
        self,
        method: str = "OT",
        epsilon: float = 5e-3,
        lamb: float = 0.1,
        max_cells: int | None = 150000,  # Safe for ~11k spots per sample
        min_cells_per_type: int = 5,  # Filter rare cell types to avoid stratification errors
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.method = method
        self.epsilon = epsilon
        self.lamb = lamb
        self.max_cells = max_cells
        self.min_cells_per_type = min_cells_per_type

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

        # Filter rare cell types to avoid stratified sampling errors
        # (sklearn requires at least 2 samples per class for stratification)
        if self.min_cells_per_type > 0:
            cell_type_counts = snrna.obs["cell_type"].value_counts()
            rare_types = cell_type_counts[cell_type_counts < self.min_cells_per_type].index.tolist()
            if rare_types:
                print(f"TACCO: Filtering {len(rare_types)} rare cell types with < {self.min_cells_per_type} cells")
                mask = ~snrna.obs["cell_type"].isin(rare_types)
                snrna = snrna[mask].copy()
                # Update categories to remove filtered types
                snrna.obs["cell_type"] = snrna.obs["cell_type"].cat.remove_unused_categories()
                print(f"TACCO: {len(snrna)} cells remaining with {snrna.obs['cell_type'].nunique()} cell types")

        # Subsample reference if needed to avoid MKL 32-bit integer overflow
        # With per-sample spatial (~11k spots), 150k cells is safe (150k * 11k = 1.65B < 2^31)
        if self.max_cells is not None and len(snrna) > self.max_cells:
            print(f"TACCO: Subsampling snRNA from {len(snrna)} to {self.max_cells} cells (stratified by cell_type)")
            # Stratified subsampling to preserve cell type proportions
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(snrna))
            _, subsample_idx = train_test_split(
                indices,
                test_size=self.max_cells / len(snrna),
                stratify=snrna.obs["cell_type"],
                random_state=42,
            )
            snrna = snrna[subsample_idx].copy()

        # Subset to common genes FIRST, then filter zero-sum spots
        # (spots may become zero-sum after gene filtering)
        common_genes = list(set(snrna.var_names) & set(spatial.var_names))
        print(f"TACCO: {len(common_genes)} common genes between snRNA and spatial")

        snrna = snrna[:, common_genes].copy()
        spatial = spatial[:, common_genes].copy()

        # Now filter zero-sum spots (after gene subsetting)
        if hasattr(spatial.X, "toarray"):
            row_sums = np.array(spatial.X.sum(axis=1)).flatten()
        else:
            row_sums = np.array(spatial.X.sum(axis=1)).flatten()

        nonzero_mask = row_sums > 0
        n_zero = (~nonzero_mask).sum()
        if n_zero > 0:
            print(f"TACCO: Filtering out {n_zero} spots with zero counts (after gene filtering)")
            spatial = spatial[nonzero_mask].copy()

        if len(spatial) == 0:
            raise ValueError("No spots remaining after filtering zero-count spots")

        # Also filter zero-sum cells in snRNA
        if hasattr(snrna.X, "toarray"):
            cell_sums = np.array(snrna.X.sum(axis=1)).flatten()
        else:
            cell_sums = np.array(snrna.X.sum(axis=1)).flatten()

        nonzero_cells = cell_sums > 0
        n_zero_cells = (~nonzero_cells).sum()
        if n_zero_cells > 0:
            print(f"TACCO: Filtering out {n_zero_cells} cells with zero counts")
            snrna = snrna[nonzero_cells].copy()

        print(f"Running TACCO with method={self.method}, {len(snrna)} cells, {len(spatial)} spots...")

        # Use TACCO's own preprocessing to build reference profiles
        # This determines exactly which genes TACCO will use
        print("TACCO: Building reference profiles to determine final gene set...")
        tc.pp.construct_reference_profiles(snrna, annotation_key="cell_type")

        # Get genes that have valid profiles (non-zero for at least one cell type)
        profiles = snrna.varm["cell_type"]
        if hasattr(profiles, "toarray"):
            profiles = profiles.toarray()
        gene_has_profile = np.any(profiles > 0, axis=1)
        valid_genes = snrna.var_names[gene_has_profile].tolist()

        # Also intersect with spatial genes
        valid_genes = list(set(valid_genes) & set(spatial.var_names))
        print(f"TACCO: {len(valid_genes)} genes with valid profiles in both datasets")

        # Subset to valid genes
        if len(valid_genes) < snrna.n_vars:
            n_removed = snrna.n_vars - len(valid_genes)
            print(f"TACCO: Removing {n_removed} genes without valid profiles")
            snrna = snrna[:, valid_genes].copy()
            spatial = spatial[:, valid_genes].copy()

            # Rebuild profiles on the filtered gene set
            tc.pp.construct_reference_profiles(snrna, annotation_key="cell_type")

        # Filter zero-sum spots in spatial AFTER profile-based gene filtering
        if hasattr(spatial.X, "toarray"):
            row_sums = np.array(spatial.X.sum(axis=1)).flatten()
        else:
            row_sums = np.array(spatial.X.sum(axis=1)).flatten()

        nonzero_mask = row_sums > 0
        n_zero = (~nonzero_mask).sum()
        if n_zero > 0:
            print(f"TACCO: Filtering out {n_zero} zero-sum spots after profile gene filtering")
            spatial = spatial[nonzero_mask].copy()

        if len(spatial) == 0:
            raise ValueError("No spots remaining after filtering")

        print(f"TACCO: Final shapes - {len(snrna)} cells, {len(spatial)} spots, {snrna.n_vars} genes")

        # Run TACCO annotation (profiles already built)
        # Try requested method first, fall back to NNLS if it fails
        method_used = self.method
        try:
            tc.tl.annotate(
                spatial,
                snrna,
                annotation_key="cell_type",
                result_key="tacco_celltype",
                method=method_used,
                epsilon=self.epsilon if method_used == "OT" else None,
                lamb=self.lamb if method_used == "NMFreg" else None,
            )
        except Exception as e:
            print(f"TACCO: {method_used} failed ({e}), falling back to NNLS")
            method_used = "NNLS"
            tc.tl.annotate(
                spatial,
                snrna,
                annotation_key="cell_type",
                result_key="tacco_celltype",
                method=method_used,
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
