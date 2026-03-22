"""Tests for query-to-reference mapping."""

from __future__ import annotations


import anndata as ad
import numpy as np
import pandas as pd
import pytest

from stagebridge.reference.schema import MappingResult
from stagebridge.reference.map_query import (
    map_to_hlca,
    map_to_luca,
    _validate_no_donor_leakage,
)


def _create_mock_reference_adata(
    n_cells: int = 200,
    n_genes: int = 100,
    latent_dim: int = 16,
    latent_key: str = "X_scanvi_emb",
) -> ad.AnnData:
    """Create mock reference AnnData."""
    obs = pd.DataFrame(
        {
            "ann_level_1": np.random.choice(["Epithelial", "Immune"], n_cells),
            "ann_level_2": np.random.choice(["AT1", "AT2", "Mac"], n_cells),
        },
        index=[f"ref_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])
    X = np.random.randn(n_cells, n_genes).astype(np.float32)
    latent = np.random.randn(n_cells, latent_dim).astype(np.float32)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm[latent_key] = latent
    return adata


def _create_mock_query_adata(
    n_cells: int = 50,
    n_genes: int = 100,
    gene_names: list[str] | None = None,
) -> ad.AnnData:
    """Create mock query AnnData."""
    obs = pd.DataFrame(
        {
            "donor_id": np.random.choice(["D1", "D2", "D3"], n_cells),
            "sample_id": [f"S{i % 5}" for i in range(n_cells)],
            "stage": np.random.choice(["AAH", "AIS", "MIA"], n_cells),
        },
        index=[f"query_{i}" for i in range(n_cells)],
    )

    if gene_names is None:
        gene_names = [f"GENE{i}" for i in range(n_genes)]

    var = pd.DataFrame(index=gene_names)
    X = np.random.randn(n_cells, len(gene_names)).astype(np.float32)

    return ad.AnnData(X=X, obs=obs, var=var)


class TestMapToHLCA:
    """Tests for map_to_hlca function."""

    def test_basic_mapping(self) -> None:
        """Basic mapping produces correct output shape."""
        ref = _create_mock_reference_adata(n_cells=100, n_genes=50, latent_dim=16)
        query = _create_mock_query_adata(n_cells=30, n_genes=50)

        result = map_to_hlca(
            query,
            ref,
            method="knn_projection",
            latent_key="X_scanvi_emb",
        )

        assert isinstance(result, MappingResult)
        assert result.embeddings.shape == (30, 16)
        assert result.latent_dim == 16
        assert result.n_cells == 30
        assert result.reference_name == "HLCA"
        assert len(result.cell_ids) == 30
        assert len(result.donor_ids) == 30

    def test_metadata_preserved(self) -> None:
        """Metadata is correctly extracted from query."""
        ref = _create_mock_reference_adata()
        query = _create_mock_query_adata(n_cells=20)

        result = map_to_hlca(query, ref, latent_key="X_scanvi_emb")

        # Cell IDs should match query index
        assert list(result.cell_ids) == list(query.obs.index)

        # Donor IDs should come from obs
        assert set(result.donor_ids) <= {"D1", "D2", "D3"}

    def test_knn_projection_method(self) -> None:
        """KNN projection method runs successfully."""
        ref = _create_mock_reference_adata(n_cells=100, latent_dim=8)
        query = _create_mock_query_adata(n_cells=20)

        result = map_to_hlca(
            query,
            ref,
            method="knn_projection",
            latent_key="X_scanvi_emb",
            k_neighbors=10,
        )

        assert result.mapping_method == "knn_projection"
        assert result.embeddings.shape[1] == 8
        assert result.neighbor_distances is not None

    def test_pca_projection_method(self) -> None:
        """PCA projection method runs successfully."""
        ref = _create_mock_reference_adata(n_cells=100, latent_dim=8)
        query = _create_mock_query_adata(n_cells=20)

        result = map_to_hlca(
            query,
            ref,
            method="pca_projection",
            latent_key="X_scanvi_emb",
        )

        assert result.mapping_method == "pca_projection"
        assert result.embeddings.shape[1] == 8

    def test_to_dataframe(self) -> None:
        """to_dataframe produces correct columns."""
        ref = _create_mock_reference_adata(latent_dim=4)
        query = _create_mock_query_adata(n_cells=10)

        result = map_to_hlca(query, ref, latent_key="X_scanvi_emb")
        df = result.to_dataframe(prefix="hlca_")

        assert "cell_id" in df.columns
        assert "donor_id" in df.columns
        assert "sample_id" in df.columns
        assert "stage_id" in df.columns
        assert "hlca_latent_0" in df.columns
        assert "hlca_latent_3" in df.columns
        assert len(df) == 10


class TestMapToLuCa:
    """Tests for map_to_luca function."""

    def test_basic_mapping(self) -> None:
        """Basic LuCa mapping produces correct output."""
        ref = _create_mock_reference_adata(n_cells=100, latent_dim=12, latent_key="X_scVI")
        query = _create_mock_query_adata(n_cells=25)

        result = map_to_luca(
            query,
            ref,
            method="knn_projection",
            latent_key="X_scVI",
        )

        assert result.reference_name == "LuCa"
        assert result.embeddings.shape == (25, 12)
        assert result.reference_latent_key == "X_scVI"


class TestDonorLeakageValidation:
    """Tests for donor leakage prevention."""

    def test_no_leakage_passes(self) -> None:
        """No leakage when donors don't overlap."""
        query_donors = np.array(["D1", "D2", "D3"])
        held_out = {"D4", "D5"}

        # Should not raise
        _validate_no_donor_leakage(query_donors, held_out)

    def test_leakage_detected(self) -> None:
        """Leakage raises ValueError when donors overlap."""
        query_donors = np.array(["D1", "D2", "D3"])
        held_out = {"D2", "D4"}

        with pytest.raises(ValueError, match="Donor leakage detected"):
            _validate_no_donor_leakage(query_donors, held_out)

    def test_no_held_out_passes(self) -> None:
        """No validation when held_out is None."""
        query_donors = np.array(["D1", "D2"])

        # Should not raise
        _validate_no_donor_leakage(query_donors, None)

    def test_mapping_with_held_out_donors(self) -> None:
        """Mapping raises when held-out donors are in query."""
        ref = _create_mock_reference_adata()

        # Create query with explicit D2 donor to guarantee overlap
        obs = pd.DataFrame(
            {
                "donor_id": ["D1", "D2", "D2", "D3"] * 3,  # Explicit D2
                "sample_id": [f"S{i % 5}" for i in range(12)],
                "stage": ["AAH"] * 12,
            },
            index=[f"query_{i}" for i in range(12)],
        )
        var = pd.DataFrame(index=[f"GENE{i}" for i in range(100)])
        X = np.random.randn(12, 100).astype(np.float32)
        query = ad.AnnData(X=X, obs=obs, var=var)

        # Query has donors D1, D2, D3 - hold out D2
        with pytest.raises(ValueError, match="Donor leakage"):
            map_to_hlca(
                query,
                ref,
                latent_key="X_scanvi_emb",
                held_out_donors={"D2"},
            )


class TestMappingResult:
    """Tests for MappingResult dataclass."""

    def test_n_cells_property(self) -> None:
        """n_cells property returns correct count."""
        result = MappingResult(
            embeddings=np.random.randn(50, 16).astype(np.float32),
            latent_dim=16,
            cell_ids=np.array([f"c{i}" for i in range(50)]),
            donor_ids=np.array(["D1"] * 50),
            sample_ids=np.array(["S1"] * 50),
            stage_ids=np.array(["AAH"] * 50),
        )
        assert result.n_cells == 50
