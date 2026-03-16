"""
Pytest fixtures for spatial backend tests.
"""

import numpy as np
import pandas as pd
import anndata as ad
import pytest


@pytest.fixture
def synthetic_snrna():
    """Create synthetic snRNA-seq data for testing."""
    n_cells = 500
    n_genes = 100
    n_celltypes = 5

    # Create expression matrix
    X = np.random.randn(n_cells, n_genes).astype(np.float32)

    # Create cell type labels
    cell_types = [f"CellType_{i}" for i in range(n_celltypes)]
    cell_type_labels = np.random.choice(cell_types, n_cells)

    # Make cell types categorical
    obs = pd.DataFrame({"cell_type": pd.Categorical(cell_type_labels, categories=cell_types)})

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def synthetic_spatial():
    """Create synthetic spatial data for testing."""
    n_spots = 200
    n_genes = 100

    # Create expression matrix
    X = np.random.randn(n_spots, n_genes).astype(np.float32)

    # Create spatial coordinates
    coords = np.random.rand(n_spots, 2) * 100

    obs = pd.DataFrame(index=[f"spot_{i}" for i in range(n_spots)])

    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords

    return adata


@pytest.fixture
def synthetic_mapping_result(synthetic_spatial, synthetic_snrna):
    """Create synthetic BackendMappingResult for testing."""
    from stagebridge.spatial_backends.base import BackendMappingResult

    n_spots = len(synthetic_spatial)
    cell_types = synthetic_snrna.obs["cell_type"].cat.categories.tolist()
    n_celltypes = len(cell_types)

    # Create random proportions (normalized)
    proportions = np.random.rand(n_spots, n_celltypes)
    proportions = proportions / proportions.sum(axis=1, keepdims=True)

    cell_type_proportions = pd.DataFrame(
        proportions,
        index=synthetic_spatial.obs_names,
        columns=cell_types,
    )

    # Create confidence scores
    confidence = pd.Series(
        np.random.rand(n_spots),
        index=synthetic_spatial.obs_names,
        name="confidence",
    )

    return BackendMappingResult(
        cell_type_proportions=cell_type_proportions,
        confidence=confidence,
        upstream_metrics={
            "mean_entropy": 0.5,
            "coverage": 0.8,
        },
        metadata={
            "backend": "test",
            "n_spots": n_spots,
        },
    )


@pytest.fixture
def synthetic_standardized_output(synthetic_mapping_result):
    """Create synthetic StandardizedOutput for testing."""
    from stagebridge.spatial_backends.standardize import standardize_backend_output

    return standardize_backend_output(
        synthetic_mapping_result,
        backend_name="test",
        backend_version="1.0.0",
    )


@pytest.fixture
def synthetic_comparison_table():
    """Create synthetic comparison table for testing."""
    return pd.DataFrame(
        {
            "backend": ["tangram", "destvi", "tacco"],
            "success": [True, True, True],
            "runtime_seconds": [10.5, 25.3, 15.2],
            "upstream_mean_entropy": [0.45, 0.52, 0.48],
            "upstream_coverage": [0.82, 0.78, 0.85],
            "upstream_sparsity": [0.15, 0.20, 0.12],
            "downstream_overall_utility": [0.72, 0.68, 0.75],
            "downstream_confidence_quality": [0.80, 0.75, 0.82],
            "spatial_local_coherence": [0.65, 0.70, 0.68],
            "spatial_smoothness": [0.58, 0.62, 0.60],
        }
    )


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "spatial_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
