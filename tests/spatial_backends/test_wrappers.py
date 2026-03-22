"""
Tests for spatial mapping backend wrappers (Tangram, DestVI, TACCO).

Tests the direct AnnData interfaces with synthetic data.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


def test_tangram_backend_imports():
    """Test that Tangram backend can be imported."""
    from stagebridge.spatial_backends import TangramBackend

    assert TangramBackend is not None


def test_destvi_backend_imports():
    """Test that DestVI backend can be imported."""
    from stagebridge.spatial_backends import DestVIBackend

    assert DestVIBackend is not None


def test_tacco_backend_imports():
    """Test that TACCO backend can be imported."""
    from stagebridge.spatial_backends import TACCOBackend

    assert TACCOBackend is not None


def test_tangram_backend_initialization():
    """Test Tangram backend initialization."""
    from stagebridge.spatial_backends import TangramBackend

    backend = TangramBackend(
        constrained=True,
        n_epochs=10,
        marker_genes="auto",
    )

    assert backend.constrained is True
    assert backend.n_epochs == 10
    assert backend.marker_genes == "auto"
    assert backend.model is None  # Not trained yet


def test_destvi_backend_initialization():
    """Test DestVI backend initialization."""
    from stagebridge.spatial_backends import DestVIBackend

    backend = DestVIBackend(
        n_latent=5,
        n_epochs_condsc=50,
        n_epochs_destvi=100,
        vamp_prior_p=25,
    )

    assert backend.n_latent == 5
    assert backend.n_epochs_condsc == 50
    assert backend.n_epochs_destvi == 100
    assert backend.vamp_prior_p == 25
    assert backend.sc_model is None  # Not trained yet
    assert backend.spatial_model is None


def test_tangram_backend_has_visualization_methods():
    """Test that Tangram backend has new visualization methods."""
    from stagebridge.spatial_backends import TangramBackend

    backend = TangramBackend()

    # Check methods exist
    assert hasattr(backend, "plot_cell_type_spatial")
    assert hasattr(backend, "project_genes")
    assert hasattr(backend, "plot_projected_genes")
    assert hasattr(backend, "compute_spatial_statistics")
    assert callable(backend.plot_cell_type_spatial)
    assert callable(backend.project_genes)


def test_destvi_backend_has_advanced_methods():
    """Test that DestVI backend has new multi-resolution methods."""
    from stagebridge.spatial_backends import DestVIBackend

    backend = DestVIBackend()

    # Check methods exist
    assert hasattr(backend, "get_gamma")
    assert hasattr(backend, "get_cell_type_specific_expression")
    assert hasattr(backend, "automatic_proportion_threshold")
    assert hasattr(backend, "filter_spots_by_celltype")
    assert hasattr(backend, "plot_cell_type_spatial")
    assert hasattr(backend, "explore_gamma_space")

    assert callable(backend.get_gamma)
    assert callable(backend.get_cell_type_specific_expression)


@pytest.mark.slow
def test_tangram_backend_map_synthetic(synthetic_snrna, synthetic_spatial, tmp_output_dir):
    """Test Tangram mapping with synthetic data."""
    scvi = pytest.importorskip("scvi", reason="scvi-tools not installed")
    pytest.importorskip("mudata", reason="mudata not installed")

    # Check scvi.external.Tangram is available (requires scvi-tools[jax])
    try:
        from scvi.external import Tangram  # noqa: F401
    except ImportError:
        pytest.skip("scvi.external.Tangram not available (install scvi-tools[jax])")

    from stagebridge.spatial_backends import TangramBackend

    backend = TangramBackend(
        constrained=True,
        n_epochs=5,  # Very short for testing
        marker_genes=synthetic_snrna.var_names[:20].tolist(),
    )

    result = backend.map(
        synthetic_snrna,
        synthetic_spatial,
        output_dir=tmp_output_dir / "tangram",
    )

    # Check result structure
    assert result.cell_type_proportions.shape[0] == len(synthetic_spatial)
    assert result.cell_type_proportions.shape[1] == len(
        synthetic_snrna.obs["cell_type"].cat.categories
    )
    assert len(result.confidence) == len(synthetic_spatial)
    assert result.metadata["backend"] == "tangram_scvi"

    # Check model was stored
    assert backend.model is not None
    assert backend._mapper is not None

    # Check outputs were saved
    assert (tmp_output_dir / "tangram" / "tangram_mapper.npy").exists()
    assert (tmp_output_dir / "tangram" / "tangram_cell_type_props.csv").exists()
    assert (tmp_output_dir / "tangram" / "tangram_spatial_annotated.h5ad").exists()


@pytest.mark.slow
@pytest.mark.xfail(
    reason="DestVI can produce NaN with synthetic data due to numerical instability",
    raises=ValueError,
    strict=False,
)
def test_destvi_backend_map_synthetic(synthetic_snrna, synthetic_spatial, tmp_output_dir):
    """Test DestVI mapping with synthetic data."""
    pytest.importorskip("scvi")

    from stagebridge.spatial_backends import DestVIBackend

    backend = DestVIBackend(
        n_latent=5,
        n_epochs_condsc=5,  # Very short for testing
        n_epochs_destvi=10,
        vamp_prior_p=10,
    )

    result = backend.map(
        synthetic_snrna,
        synthetic_spatial,
        output_dir=tmp_output_dir / "destvi",
    )

    # Check result structure
    assert result.cell_type_proportions.shape[0] == len(synthetic_spatial)
    assert result.cell_type_proportions.shape[1] == len(
        synthetic_snrna.obs["cell_type"].cat.categories
    )
    assert len(result.confidence) == len(synthetic_spatial)
    assert result.metadata["backend"] == "destvi"

    # Check models were stored
    assert backend.sc_model is not None
    assert backend.spatial_model is not None

    # Check outputs were saved
    assert (tmp_output_dir / "destvi" / "condscvi_model").exists()
    assert (tmp_output_dir / "destvi" / "destvi_model").exists()
    assert (tmp_output_dir / "destvi" / "destvi_cell_type_props.csv").exists()
    assert (tmp_output_dir / "destvi" / "destvi_spatial_annotated.h5ad").exists()

    # Check gamma files exist
    cell_types = synthetic_snrna.obs["cell_type"].cat.categories
    for ct in cell_types[:2]:
        gamma_file = tmp_output_dir / "destvi" / f"destvi_gamma_{ct.replace(' ', '_')}.csv"
        assert gamma_file.exists()


@pytest.mark.slow
@pytest.mark.xfail(
    reason="DestVI can produce NaN with synthetic data due to numerical instability",
    raises=ValueError,
    strict=False,
)
def test_destvi_get_gamma(synthetic_snrna, synthetic_spatial):
    """Test DestVI gamma extraction after mapping."""
    pytest.importorskip("scvi")

    from stagebridge.spatial_backends import DestVIBackend

    backend = DestVIBackend(
        n_latent=5,
        n_epochs_condsc=5,
        n_epochs_destvi=10,
    )

    # Map first
    result = backend.map(synthetic_snrna, synthetic_spatial)

    # Get gamma values
    cell_types = result.cell_type_proportions.columns.tolist()
    gamma_dict = backend.get_gamma(cell_types=cell_types[:2])

    assert len(gamma_dict) == 2
    for ct, gamma_df in gamma_dict.items():
        assert gamma_df.shape[0] == len(synthetic_spatial)
        assert gamma_df.shape[1] > 0  # Should have latent dimensions


@pytest.mark.slow
@pytest.mark.xfail(
    reason="DestVI can produce NaN with synthetic data due to numerical instability",
    raises=ValueError,
    strict=False,
)
def test_destvi_filter_spots(synthetic_snrna, synthetic_spatial):
    """Test DestVI spot filtering by cell type."""
    pytest.importorskip("scvi")

    from stagebridge.spatial_backends import DestVIBackend

    backend = DestVIBackend(
        n_latent=5,
        n_epochs_condsc=5,
        n_epochs_destvi=10,
    )

    # Map first
    result = backend.map(synthetic_snrna, synthetic_spatial)

    # Filter spots
    cell_types = result.cell_type_proportions.columns.tolist()
    for ct in cell_types[:2]:
        indices = backend.filter_spots_by_celltype(ct, auto_threshold=True)
        assert isinstance(indices, np.ndarray)
        assert len(indices) <= len(synthetic_spatial)
        assert indices.dtype in [np.int32, np.int64]


@pytest.mark.slow
def test_tangram_project_genes(synthetic_snrna, synthetic_spatial):
    """Test Tangram gene projection."""
    pytest.importorskip("scvi")
    pytest.importorskip("mudata")

    # Check scvi.external.Tangram is available (requires scvi-tools[jax])
    try:
        from scvi.external import Tangram  # noqa: F401
    except ImportError:
        pytest.skip("scvi.external.Tangram not available (install scvi-tools[jax])")

    from stagebridge.spatial_backends import TangramBackend

    backend = TangramBackend(
        constrained=True,
        n_epochs=5,
        marker_genes=synthetic_snrna.var_names[:20].tolist(),
    )

    # Map first
    result = backend.map(synthetic_snrna, synthetic_spatial)

    # Project genes
    genes_to_project = synthetic_snrna.var_names[:5].tolist()
    projected = backend.project_genes(genes_to_project, aggregate=False)

    assert projected.shape[0] == len(synthetic_spatial)
    assert projected.shape[1] == len(genes_to_project)
    assert all(g in projected.columns for g in genes_to_project)

    # Test aggregation
    projected_agg = backend.project_genes(genes_to_project, aggregate=True)
    assert projected_agg.shape == (len(synthetic_spatial), 1)


def test_visualization_utils_imports():
    """Test that visualization utilities can be imported."""
    from stagebridge.spatial_backends import (
        plot_proportions_spatial,
        plot_gamma_pca_spatial,
        plot_projected_genes_spatial,
        plot_proportion_distribution,
        plot_proportion_heatmap,
        plot_entropy_vs_sparsity,
        plot_spatial_autocorrelation,
        create_comprehensive_report,
    )

    # Check all imports succeeded
    assert plot_proportions_spatial is not None
    assert plot_gamma_pca_spatial is not None
    assert create_comprehensive_report is not None


def test_viz_utils_plot_proportion_distribution(synthetic_mapping_result):
    """Test proportion distribution plotting."""
    from stagebridge.spatial_backends.viz_utils import plot_proportion_distribution

    proportions = synthetic_mapping_result.cell_type_proportions

    # Should not crash (no save_path = no file created)
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        plot_proportion_distribution(proportions, kind="violin")
        plot_proportion_distribution(proportions, kind="box")
    except ImportError:
        pytest.skip("matplotlib not available")


def test_viz_utils_plot_entropy_vs_sparsity(synthetic_mapping_result):
    """Test entropy vs sparsity plotting."""
    from stagebridge.spatial_backends.viz_utils import plot_entropy_vs_sparsity

    proportions = synthetic_mapping_result.cell_type_proportions

    try:
        import matplotlib

        matplotlib.use("Agg")
        plot_entropy_vs_sparsity(proportions)
    except ImportError:
        pytest.skip("matplotlib not available")


def test_backend_factory():
    """Test backend factory function."""
    from stagebridge.spatial_backends import get_backend

    # Get direct backends
    TangramBackend = get_backend("tangram", use_adapter=False)
    DestVIBackend = get_backend("destvi", use_adapter=False)
    TACCOBackend = get_backend("tacco", use_adapter=False)

    assert TangramBackend is not None
    assert DestVIBackend is not None
    assert TACCOBackend is not None

    # Get adapters
    TangramAdapter = get_backend("tangram", use_adapter=True)
    DestVIAdapter = get_backend("destvi", use_adapter=True)
    TACCOAdapter = get_backend("tacco", use_adapter=True)

    assert TangramAdapter is not None
    assert DestVIAdapter is not None
    assert TACCOAdapter is not None


def test_backend_factory_invalid():
    """Test backend factory with invalid name."""
    from stagebridge.spatial_backends import get_backend

    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("invalid_backend")


def test_destvi_requires_mapping_before_gamma():
    """Test that DestVI raises error if gamma accessed before mapping."""
    from stagebridge.spatial_backends import DestVIBackend

    backend = DestVIBackend()

    with pytest.raises(RuntimeError, match="Must run map"):
        backend.get_gamma()


def test_tangram_requires_mapping_before_projection():
    """Test that Tangram raises error if projection before mapping."""
    from stagebridge.spatial_backends import TangramBackend

    backend = TangramBackend()

    with pytest.raises(RuntimeError, match="Must run map"):
        backend.project_genes(["gene_0"])
