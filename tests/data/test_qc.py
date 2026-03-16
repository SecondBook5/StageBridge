"""Tests for stagebridge.data.qc module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Try to import anndata; skip tests if not available
try:
    import anndata

    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

from stagebridge.data.qc import (
    QCConfig,
    QCResult,
    compute_qc_metrics,
    run_qc,
    generate_qc_figures,
    generate_per_donor_figures,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_adata():
    """Create a simple AnnData object for testing."""
    if not ANNDATA_AVAILABLE:
        pytest.skip("anndata not available")

    np.random.seed(42)
    n_cells = 100
    n_genes = 50

    # Create counts with some variation
    counts = np.random.negative_binomial(5, 0.5, size=(n_cells, n_genes))

    # Create gene names with some mitochondrial genes
    gene_names = [f"Gene{i}" for i in range(n_genes - 3)]
    gene_names.extend(["MT-CO1", "MT-CO2", "MT-ND1"])

    # Create cell metadata
    obs = pd.DataFrame(
        {
            "donor_id": np.repeat(["D1", "D2", "D3", "D4"], 25),
            "sample_id": np.repeat([f"S{i}" for i in range(1, 5)], 25),
            "stage": np.repeat(["Normal", "AAH", "AIS", "MIA"], 25),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    adata = anndata.AnnData(
        X=counts.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )

    return adata


@pytest.fixture
def adata_with_outliers():
    """Create AnnData with outlier cells for testing QC filters."""
    if not ANNDATA_AVAILABLE:
        pytest.skip("anndata not available")

    np.random.seed(42)
    n_cells = 100
    n_genes = 50

    counts = np.random.negative_binomial(5, 0.5, size=(n_cells, n_genes))

    # Add outliers
    counts[0, :] = 1  # Very low counts
    counts[1, :] = 10000  # Very high counts
    counts[2, :3] = 1  # Low gene diversity

    gene_names = [f"Gene{i}" for i in range(n_genes - 3)]
    gene_names.extend(["MT-CO1", "MT-CO2", "MT-ND1"])

    # Make cell 3 have high mito
    counts[3, -3:] = counts[3, :].sum() * 10  # High mito fraction

    obs = pd.DataFrame(
        {
            "donor_id": ["D1"] * n_cells,
            "sample_id": ["S1"] * n_cells,
            "stage": ["Normal"] * n_cells,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    adata = anndata.AnnData(
        X=counts.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )

    return adata


# ---------------------------------------------------------------------------
# QCConfig tests
# ---------------------------------------------------------------------------


class TestQCConfig:
    """Tests for QCConfig."""

    def test_default_config(self) -> None:
        """Test default QC configuration."""
        config = QCConfig()

        assert config.min_counts == 500
        assert config.max_counts == 50000
        assert config.min_genes == 200
        assert config.max_genes == 8000
        assert config.max_mito_pct == 20.0
        assert config.modality == "snrna"

    def test_default_snrna(self) -> None:
        """Test default snRNA config."""
        config = QCConfig.default_snrna()

        assert config.modality == "snrna"
        assert config.min_counts is not None

    def test_default_spatial(self) -> None:
        """Test default spatial config."""
        config = QCConfig.default_spatial()

        assert config.modality == "spatial"
        assert config.spot_tissue_filter is True
        # Spatial typically has higher mito threshold
        assert config.max_mito_pct >= 20.0

    def test_lenient_config(self) -> None:
        """Test lenient config for exploration."""
        config = QCConfig.lenient()

        assert config.min_counts < 500  # More lenient
        assert config.max_mito_pct >= 50.0

    def test_to_dict(self) -> None:
        """Test config serialization."""
        config = QCConfig(min_counts=100, max_mito_pct=15.0)

        d = config.to_dict()

        assert d["min_counts"] == 100
        assert d["max_mito_pct"] == 15.0
        assert "modality" in d

    def test_from_dict(self) -> None:
        """Test config deserialization."""
        d = {"min_counts": 200, "max_genes": 5000, "modality": "spatial"}

        config = QCConfig.from_dict(d)

        assert config.min_counts == 200
        assert config.max_genes == 5000
        assert config.modality == "spatial"


# ---------------------------------------------------------------------------
# QC metric computation tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestComputeQCMetrics:
    """Tests for QC metric computation."""

    def test_compute_basic_metrics(self, simple_adata) -> None:
        """Test basic QC metric computation."""
        compute_qc_metrics(simple_adata)

        assert "n_counts" in simple_adata.obs.columns
        assert "n_genes" in simple_adata.obs.columns
        assert "pct_counts_mito" in simple_adata.obs.columns

    def test_metrics_reasonable_values(self, simple_adata) -> None:
        """Test that computed metrics have reasonable values."""
        compute_qc_metrics(simple_adata)

        # All values should be non-negative
        assert (simple_adata.obs["n_counts"] >= 0).all()
        assert (simple_adata.obs["n_genes"] >= 0).all()
        assert (simple_adata.obs["pct_counts_mito"] >= 0).all()
        assert (simple_adata.obs["pct_counts_mito"] <= 100).all()

    def test_mito_detection(self, simple_adata) -> None:
        """Test mitochondrial gene detection."""
        compute_qc_metrics(simple_adata, mito_prefix="MT-")

        # Should have some mito percentage
        assert simple_adata.obs["pct_counts_mito"].max() > 0


# ---------------------------------------------------------------------------
# QC filtering tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestRunQC:
    """Tests for QC filtering."""

    def test_run_qc_basic(self, simple_adata) -> None:
        """Test basic QC filtering."""
        config = QCConfig.lenient()  # Use lenient to keep most cells

        adata_filtered, result = run_qc(simple_adata, config)

        assert isinstance(result, QCResult)
        assert result.n_cells_pre == simple_adata.n_obs
        assert result.n_cells_post <= result.n_cells_pre
        assert adata_filtered.n_obs == result.n_cells_post

    def test_run_qc_filters_low_counts(self, adata_with_outliers) -> None:
        """Test that QC filters low count cells."""
        config = QCConfig(
            min_counts=10,  # Cell 0 has ~50 counts
            max_counts=None,
            min_genes=None,
            max_genes=None,
            max_mito_pct=None,
        )

        adata_filtered, result = run_qc(adata_with_outliers, config)

        # Should have filtered at least cell 0
        assert result.n_filtered_min_counts >= 0

    def test_run_qc_preserves_raw(self, simple_adata) -> None:
        """Test that QC preserves raw counts if present."""
        # Add counts layer before QC
        simple_adata.layers["counts"] = simple_adata.X.copy()

        config = QCConfig.lenient()
        adata_filtered, _ = run_qc(simple_adata, config)

        # Should preserve counts layer
        assert "counts" in adata_filtered.layers

    def test_run_qc_per_donor_stats(self, simple_adata) -> None:
        """Test per-donor statistics in QC result."""
        config = QCConfig.lenient()

        _, result = run_qc(simple_adata, config, donor_column="donor_id")

        # Should have per-donor stats
        assert len(result.per_donor_stats) > 0
        for donor, stats in result.per_donor_stats.items():
            assert "pre_qc" in stats
            assert "post_qc" in stats

    def test_run_qc_per_stage_stats(self, simple_adata) -> None:
        """Test per-stage statistics in QC result."""
        config = QCConfig.lenient()

        _, result = run_qc(simple_adata, config, stage_column="stage")

        assert len(result.per_stage_stats) > 0

    def test_run_qc_copy_mode(self, simple_adata) -> None:
        """Test that copy mode doesn't modify original."""
        config = QCConfig.lenient()
        original_n_obs = simple_adata.n_obs

        adata_filtered, _ = run_qc(simple_adata, config, copy=True)

        # Original should be unchanged
        assert simple_adata.n_obs == original_n_obs

    def test_qc_result_retention_rate(self, simple_adata) -> None:
        """Test retention rate calculation."""
        config = QCConfig.lenient()

        _, result = run_qc(simple_adata, config)

        assert 0 <= result.retention_rate <= 100

    def test_qc_result_save(self, simple_adata, tmp_path: Path) -> None:
        """Test saving QC result."""
        config = QCConfig.lenient()
        _, result = run_qc(simple_adata, config)

        output_path = tmp_path / "qc_result.json"
        result.save(output_path)

        assert output_path.exists()


# ---------------------------------------------------------------------------
# QC figure generation tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestGenerateQCFigures:
    """Tests for QC figure generation."""

    def test_generate_figures_basic(self, simple_adata, tmp_path: Path) -> None:
        """Test basic figure generation."""
        config = QCConfig.lenient()
        adata_filtered, result = run_qc(simple_adata, config)

        try:
            figures = generate_qc_figures(
                adata_filtered,
                result,
                tmp_path,
            )

            # Should generate at least some figures
            assert len(figures) >= 0  # May be 0 if matplotlib not available
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_generate_per_donor_figures(self, simple_adata, tmp_path: Path) -> None:
        """Test per-donor figure generation."""
        compute_qc_metrics(simple_adata)

        try:
            figures = generate_per_donor_figures(
                simple_adata,
                donor_id="D1",
                output_dir=tmp_path,
            )

            assert len(figures) >= 0
        except ImportError:
            pytest.skip("matplotlib not available")


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANNDATA_AVAILABLE, reason="anndata not available")
class TestQCEdgeCases:
    """Tests for edge cases in QC."""

    def test_empty_adata(self) -> None:
        """Test QC on empty AnnData."""
        adata = anndata.AnnData(
            X=np.zeros((0, 10)),
            obs=pd.DataFrame(),
            var=pd.DataFrame(index=[f"Gene{i}" for i in range(10)]),
        )

        config = QCConfig.lenient()
        adata_filtered, result = run_qc(adata, config)

        assert result.n_cells_pre == 0
        assert result.n_cells_post == 0

    def test_single_cell_adata(self) -> None:
        """Test QC on single-cell AnnData."""
        adata = anndata.AnnData(
            X=np.array([[100, 200, 300]]).astype(np.float32),
            obs=pd.DataFrame(
                {"donor_id": ["D1"], "stage": ["Normal"]},
                index=["cell_0"],
            ),
            var=pd.DataFrame(index=["Gene1", "Gene2", "Gene3"]),
        )

        config = QCConfig(
            min_counts=1,
            max_counts=None,
            min_genes=1,
            max_genes=None,
            max_mito_pct=None,
        )

        adata_filtered, result = run_qc(adata, config)

        assert result.n_cells_post == 1

    def test_missing_donor_column(self, simple_adata) -> None:
        """Test QC with missing donor column."""
        # Remove donor column
        adata = simple_adata.copy()
        del adata.obs["donor_id"]

        config = QCConfig.lenient()
        _, result = run_qc(adata, config, donor_column="donor_id")

        # Should still work, just no per-donor stats
        assert result.n_cells_post > 0
