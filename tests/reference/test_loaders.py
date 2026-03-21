"""Tests for reference loading and validation."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from stagebridge.reference.loaders import (
    LoadedReference,
    FeatureOverlapReport,
    compute_feature_overlap,
    validate_reference,
)


def _create_mock_reference(
    tmp_path: Path,
    name: str,
    n_cells: int = 100,
    n_genes: int = 500,
    latent_dim: int = 32,
    latent_key: str = "X_scanvi_emb",
    obs_cols: dict | None = None,
) -> Path:
    """Create a mock reference h5ad file."""
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])

    # Add default obs columns
    default_cols = obs_cols or {
        "ann_level_1": np.random.choice(["Epithelial", "Immune", "Stromal"], n_cells),
        "ann_level_2": np.random.choice(["AT1", "AT2", "Macrophage"], n_cells),
        "ann_level_3": np.random.choice(["AT1", "AT2", "AM"], n_cells),
    }
    for col, values in default_cols.items():
        obs[col] = values

    var = pd.DataFrame(
        {"feature_name": [f"GENE{i}" for i in range(n_genes)]},
        index=[f"ENSG0000{i:05d}" for i in range(n_genes)],
    )

    X = np.random.randn(n_cells, n_genes).astype(np.float32)
    latent = np.random.randn(n_cells, latent_dim).astype(np.float32)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm[latent_key] = latent

    path = tmp_path / f"{name}.h5ad"
    adata.write_h5ad(path)
    return path


def _create_mock_query(
    tmp_path: Path,
    n_cells: int = 50,
    n_genes: int = 400,
    gene_prefix: str = "ENSG0000",
) -> ad.AnnData:
    """Create a mock query AnnData."""
    obs = pd.DataFrame(
        {
            "donor_id": np.random.choice(["D1", "D2", "D3"], n_cells),
            "sample_id": np.random.choice(["S1", "S2"], n_cells),
            "stage": np.random.choice(["AAH", "AIS", "MIA"], n_cells),
        },
        index=[f"query_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(index=[f"{gene_prefix}{i:05d}" for i in range(n_genes)])
    X = np.random.randn(n_cells, n_genes).astype(np.float32)

    return ad.AnnData(X=X, obs=obs, var=var)


class TestValidateReference:
    """Tests for validate_reference function."""

    def test_valid_hlca_reference(self, tmp_path: Path) -> None:
        """Valid HLCA reference passes validation."""
        path = _create_mock_reference(tmp_path, "hlca_valid")
        adata = ad.read_h5ad(path)
        errors = validate_reference(adata, "HLCA", latent_key="X_scanvi_emb")
        assert errors == []

    def test_missing_obs_columns(self, tmp_path: Path) -> None:
        """Missing obs columns are detected."""
        path = _create_mock_reference(
            tmp_path,
            "hlca_missing_cols",
            obs_cols={"ann_level_1": ["A"] * 100},  # Missing level 2 and 3
        )
        adata = ad.read_h5ad(path)
        errors = validate_reference(adata, "HLCA", latent_key="X_scanvi_emb")
        assert len(errors) > 0
        assert "Missing required obs columns" in errors[0]

    def test_missing_latent_key(self, tmp_path: Path) -> None:
        """Missing latent key is detected."""
        path = _create_mock_reference(tmp_path, "hlca_no_latent", latent_key="X_other")
        adata = ad.read_h5ad(path)
        errors = validate_reference(adata, "HLCA", latent_key="X_scanvi_emb")
        assert len(errors) > 0
        assert "Missing latent embedding" in errors[0]

    def test_nan_in_latent(self, tmp_path: Path) -> None:
        """NaN values in latent are detected."""
        path = _create_mock_reference(tmp_path, "hlca_nan")
        adata = ad.read_h5ad(path)
        adata.obsm["X_scanvi_emb"][0, 0] = np.nan
        errors = validate_reference(adata, "HLCA", latent_key="X_scanvi_emb")
        assert any("NaN" in e for e in errors)


class TestComputeFeatureOverlap:
    """Tests for compute_feature_overlap function."""

    def test_full_overlap(self, tmp_path: Path) -> None:
        """Full overlap when genes match exactly."""
        ref_path = _create_mock_reference(tmp_path, "ref", n_genes=100)
        ref = ad.read_h5ad(ref_path)

        # Query with same genes
        query = ad.AnnData(
            X=np.random.randn(50, 100).astype(np.float32),
            var=ref.var.copy(),
            obs=pd.DataFrame(index=[f"q{i}" for i in range(50)]),
        )

        report = compute_feature_overlap(query, ref)
        assert report.overlap_fraction == 1.0
        assert report.shared_gene_count == 100
        assert report.status == "complete"

    def test_partial_overlap(self, tmp_path: Path) -> None:
        """Partial overlap is computed correctly."""
        ref_path = _create_mock_reference(tmp_path, "ref", n_genes=100)
        ref = ad.read_h5ad(ref_path)

        # Query with 50% overlapping genes (use feature_name symbols, not ENSG IDs)
        query_genes = list(ref.var["feature_name"][:50]) + [f"NOVEL{i}" for i in range(50)]
        query = ad.AnnData(
            X=np.random.randn(50, 100).astype(np.float32),
            var=pd.DataFrame(index=query_genes),
            obs=pd.DataFrame(index=[f"q{i}" for i in range(50)]),
        )

        report = compute_feature_overlap(query, ref)
        assert report.overlap_fraction == 0.5
        assert report.shared_gene_count == 50

    def test_low_overlap_warning(self, tmp_path: Path) -> None:
        """Low overlap produces warning status."""
        ref_path = _create_mock_reference(tmp_path, "ref", n_genes=100)
        ref = ad.read_h5ad(ref_path)

        # Query with only 10% overlapping genes (use feature_name symbols, not ENSG IDs)
        query_genes = list(ref.var["feature_name"][:10]) + [f"NOVEL{i}" for i in range(90)]
        query = ad.AnnData(
            X=np.random.randn(50, 100).astype(np.float32),
            var=pd.DataFrame(index=query_genes),
            obs=pd.DataFrame(index=[f"q{i}" for i in range(50)]),
        )

        report = compute_feature_overlap(query, ref, min_overlap_threshold=0.3)
        assert "low_overlap" in report.status

    def test_missing_genes_reported(self, tmp_path: Path) -> None:
        """Missing genes are reported in both directions."""
        ref_path = _create_mock_reference(tmp_path, "ref", n_genes=100)
        ref = ad.read_h5ad(ref_path)

        # Query with 50 shared, 50 unique query, 50 missing ref
        query_genes = list(ref.var_names[:50]) + [f"NOVEL{i}" for i in range(50)]
        query = ad.AnnData(
            X=np.random.randn(50, 100).astype(np.float32),
            var=pd.DataFrame(index=query_genes),
            obs=pd.DataFrame(index=[f"q{i}" for i in range(50)]),
        )

        report = compute_feature_overlap(query, ref)
        assert len(report.missing_in_query) > 0  # Ref genes missing from query
        assert len(report.missing_in_reference) > 0  # Query genes missing from ref


class TestFeatureOverlapReport:
    """Tests for FeatureOverlapReport dataclass."""

    def test_to_dict(self) -> None:
        """to_dict produces serializable output."""
        report = FeatureOverlapReport(
            query_gene_count=100,
            reference_gene_count=200,
            shared_gene_count=80,
            overlap_fraction=0.4,
            missing_in_query=["GENE1", "GENE2"],
            missing_in_reference=["NOVEL1"],
            status="complete",
        )

        d = report.to_dict()
        assert d["query_gene_count"] == 100
        assert d["overlap_fraction"] == 0.4
        assert d["missing_in_query_count"] == 2
        assert d["missing_in_reference_count"] == 1
        # Should be JSON serializable
        import json

        json.dumps(d)
