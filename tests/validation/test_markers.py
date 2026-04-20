"""Tests for stagebridge.validation.markers module."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from stagebridge.validation.markers import (
    MARKER_GENES,
    compute_marker_enrichment,
    validate_cell_type,
    validate_all_cell_types,
)


class MockAnnData:
    """Minimal AnnData mock for testing without anndata dependency."""

    def __init__(self, X, obs, var_names):
        self.X = X
        self.obs = obs
        self.var_names = var_names
        self.n_obs = X.shape[0]
        self.n_vars = X.shape[1]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            _, col_idx = key
            if isinstance(col_idx, list):
                indices = [list(self.var_names).index(g) for g in col_idx]
                return MockAnnData(
                    self.X[:, indices],
                    self.obs,
                    col_idx
                )
        return self


@pytest.fixture
def mock_adata():
    """Create mock AnnData with known marker patterns."""
    n_cells = 100
    genes = ["SFTPC", "SFTPB", "CD3D", "CD3E", "COL1A1", "EPCAM", "OTHER"]

    # Create expression matrix
    X = np.zeros((n_cells, len(genes)))

    # AT2 cells (0-30): high SFTPC, SFTPB
    X[0:30, 0] = np.random.exponential(5, 30)  # SFTPC
    X[0:30, 1] = np.random.exponential(4, 30)  # SFTPB
    X[0:30, 5] = np.random.exponential(3, 30)  # EPCAM

    # T cells (30-60): high CD3D, CD3E
    X[30:60, 2] = np.random.exponential(5, 30)  # CD3D
    X[30:60, 3] = np.random.exponential(4, 30)  # CD3E

    # Fibroblasts (60-100): high COL1A1
    X[60:100, 4] = np.random.exponential(5, 40)  # COL1A1

    # Create obs
    cell_types = ["AT2"] * 30 + ["T cell"] * 30 + ["Fibroblast"] * 40
    obs = pd.DataFrame({"cell_type": cell_types})

    return MockAnnData(X, obs, genes)


class TestMarkerGenes:
    def test_marker_genes_defined(self):
        assert "alveolar type 2" in MARKER_GENES or "at2" in MARKER_GENES
        assert "t cell" in MARKER_GENES
        assert "fibroblast" in MARKER_GENES

    def test_markers_are_lists(self):
        for cell_type, markers in MARKER_GENES.items():
            assert isinstance(markers, list)
            assert len(markers) > 0


class TestComputeMarkerEnrichment:
    def test_computes_enrichment(self, mock_adata):
        markers = ["SFTPC", "CD3D", "COL1A1"]
        df = compute_marker_enrichment(mock_adata, "cell_type", markers)

        assert len(df) > 0
        assert "cell_type" in df.columns
        assert "marker" in df.columns
        assert "enrichment" in df.columns

    def test_at2_enriched_for_sftpc(self, mock_adata):
        markers = ["SFTPC"]
        df = compute_marker_enrichment(mock_adata, "cell_type", markers)

        at2_row = df[(df["cell_type"] == "AT2") & (df["marker"] == "SFTPC")]
        assert len(at2_row) == 1
        assert at2_row.iloc[0]["enrichment"] > 1.5  # Should be enriched

    def test_missing_markers_handled(self, mock_adata):
        markers = ["NONEXISTENT_GENE"]
        df = compute_marker_enrichment(mock_adata, "cell_type", markers)
        assert len(df) == 0

    def test_min_cells_filter(self, mock_adata):
        markers = ["SFTPC"]
        df = compute_marker_enrichment(mock_adata, "cell_type", markers, min_cells=50)
        # AT2 has 30 cells, should be filtered out
        if len(df) > 0:
            assert "AT2" not in df["cell_type"].values
        else:
            assert True  # Empty df means all filtered

    def test_sparse_matrix_handled(self, mock_adata):
        mock_adata.X = sparse.csr_matrix(mock_adata.X)
        markers = ["SFTPC"]
        df = compute_marker_enrichment(mock_adata, "cell_type", markers)
        assert len(df) > 0


class TestValidateCellType:
    def test_pass_when_markers_enriched(self, mock_adata):
        markers = ["SFTPC", "SFTPB"]
        enrichment_df = compute_marker_enrichment(mock_adata, "cell_type", markers)

        # AT2 should have enriched SFTPC/SFTPB
        result = validate_cell_type("AT2", enrichment_df)
        assert result["status"] in ["pass", "warn"]  # At least some markers enriched

    def test_no_markers_defined(self, mock_adata):
        markers = ["SFTPC"]
        enrichment_df = compute_marker_enrichment(mock_adata, "cell_type", markers)

        result = validate_cell_type("UnknownCellType", enrichment_df)
        # Either no markers defined or no enrichment data (cell type not in df)
        assert result["status"] in ["no_markers_defined", "no_enrichment_data"]

    def test_concordance_computed(self, mock_adata):
        markers = ["SFTPC", "SFTPB", "LAMP3"]  # LAMP3 not in mock data
        enrichment_df = compute_marker_enrichment(mock_adata, "cell_type", markers)

        result = validate_cell_type("AT2", enrichment_df)
        assert "concordance" in result
        assert 0 <= result["concordance"] <= 1


class TestValidateAllCellTypes:
    def test_validates_all_types(self, mock_adata):
        result = validate_all_cell_types(mock_adata, "cell_type")

        assert "valid" in result
        assert "summary" in result
        assert "validations" in result
        assert len(result["validations"]) == 3  # AT2, T cell, Fibroblast

    def test_missing_column_raises(self, mock_adata):
        with pytest.raises(ValueError, match="not found"):
            validate_all_cell_types(mock_adata, "nonexistent_column")

    def test_returns_enrichment_df(self, mock_adata):
        result = validate_all_cell_types(mock_adata, "cell_type")
        assert "enrichment_df" in result
        assert isinstance(result["enrichment_df"], pd.DataFrame)
