"""Tests for dual-reference fusion operations."""

from __future__ import annotations

import numpy as np
import pytest

from stagebridge.reference.fuse import (
    FusedEmbeddingResult,
    fuse_dual_reference,
    fuse_single_reference,
)
from stagebridge.reference.schema import MappingResult


def _create_mock_mapping_result(
    n_cells: int = 50,
    latent_dim: int = 16,
    reference_name: str = "HLCA",
) -> MappingResult:
    """Create mock MappingResult."""
    return MappingResult(
        embeddings=np.random.randn(n_cells, latent_dim).astype(np.float32),
        latent_dim=latent_dim,
        cell_ids=np.array([f"cell_{i}" for i in range(n_cells)]),
        donor_ids=np.array([f"D{i % 3}" for i in range(n_cells)]),
        sample_ids=np.array([f"S{i % 5}" for i in range(n_cells)]),
        stage_ids=np.array(["AAH", "AIS", "MIA", "LUAD"][i % 4] for i in range(n_cells)),
        reference_name=reference_name,
    )


class TestFuseDualReference:
    """Tests for fuse_dual_reference function."""

    def test_concat_fusion(self) -> None:
        """Concatenation fusion produces correct dimensions."""
        hlca = _create_mock_mapping_result(n_cells=30, latent_dim=16)
        luca = _create_mock_mapping_result(n_cells=30, latent_dim=12)
        # Ensure same cell IDs
        luca = MappingResult(
            embeddings=luca.embeddings,
            latent_dim=luca.latent_dim,
            cell_ids=hlca.cell_ids,
            donor_ids=hlca.donor_ids,
            sample_ids=hlca.sample_ids,
            stage_ids=hlca.stage_ids,
        )

        fused = fuse_dual_reference(hlca, luca, method="concat", normalize=False)

        assert isinstance(fused, FusedEmbeddingResult)
        assert fused.fused_dim == 16 + 12  # Concatenated
        assert fused.n_cells == 30
        assert fused.fusion_method == "concat"
        assert np.all(fused.reference_mode_used == "both")

    def test_average_fusion_same_dims(self) -> None:
        """Average fusion works with same dimensions."""
        hlca = _create_mock_mapping_result(n_cells=20, latent_dim=16)
        luca = _create_mock_mapping_result(n_cells=20, latent_dim=16)
        luca = MappingResult(
            embeddings=luca.embeddings,
            latent_dim=luca.latent_dim,
            cell_ids=hlca.cell_ids,
            donor_ids=hlca.donor_ids,
            sample_ids=hlca.sample_ids,
            stage_ids=hlca.stage_ids,
        )

        fused = fuse_dual_reference(hlca, luca, method="average", normalize=False)

        assert fused.fused_dim == 16
        expected = (hlca.embeddings + luca.embeddings) / 2
        assert np.allclose(fused.fused_embeddings, expected)

    def test_average_fusion_different_dims_raises(self) -> None:
        """Average fusion raises with different dimensions."""
        hlca = _create_mock_mapping_result(n_cells=20, latent_dim=16)
        luca = _create_mock_mapping_result(n_cells=20, latent_dim=12)
        luca = MappingResult(
            embeddings=luca.embeddings,
            latent_dim=luca.latent_dim,
            cell_ids=hlca.cell_ids,
            donor_ids=hlca.donor_ids,
            sample_ids=hlca.sample_ids,
            stage_ids=hlca.stage_ids,
        )

        with pytest.raises(ValueError, match="same dimensions"):
            fuse_dual_reference(hlca, luca, method="average")

    def test_weighted_fusion(self) -> None:
        """Weighted fusion uses confidence scores."""
        hlca = _create_mock_mapping_result(n_cells=20, latent_dim=8)
        luca = _create_mock_mapping_result(n_cells=20, latent_dim=8)
        luca = MappingResult(
            embeddings=luca.embeddings,
            latent_dim=luca.latent_dim,
            cell_ids=hlca.cell_ids,
            donor_ids=hlca.donor_ids,
            sample_ids=hlca.sample_ids,
            stage_ids=hlca.stage_ids,
        )

        # High HLCA confidence, low LuCa confidence
        hlca_conf = np.ones(20, dtype=np.float32) * 0.9
        luca_conf = np.ones(20, dtype=np.float32) * 0.1

        fused = fuse_dual_reference(
            hlca,
            luca,
            method="weighted",
            hlca_confidence=hlca_conf,
            luca_confidence=luca_conf,
            normalize=False,
        )

        assert fused.fused_dim == 8
        # Should be mostly HLCA
        assert np.all(fused.reference_mode_used == "hlca")

    def test_cell_id_mismatch_raises(self) -> None:
        """Mismatched cell IDs raise ValueError."""
        hlca = _create_mock_mapping_result(n_cells=20)
        # Create luca with explicitly different cell IDs
        luca = MappingResult(
            embeddings=np.random.randn(20, 16).astype(np.float32),
            latent_dim=16,
            cell_ids=np.array([f"different_{i}" for i in range(20)]),  # Different!
            donor_ids=hlca.donor_ids,
            sample_ids=hlca.sample_ids,
            stage_ids=hlca.stage_ids,
        )

        with pytest.raises(ValueError, match="Cell IDs must match"):
            fuse_dual_reference(hlca, luca, method="concat")

    def test_to_dataframe_schema(self) -> None:
        """to_dataframe produces standard schema columns."""
        hlca = _create_mock_mapping_result(n_cells=10, latent_dim=4)
        luca = _create_mock_mapping_result(n_cells=10, latent_dim=4)
        luca = MappingResult(
            embeddings=luca.embeddings,
            latent_dim=luca.latent_dim,
            cell_ids=hlca.cell_ids,
            donor_ids=hlca.donor_ids,
            sample_ids=hlca.sample_ids,
            stage_ids=hlca.stage_ids,
        )

        fused = fuse_dual_reference(hlca, luca, method="concat", normalize=False)
        df = fused.to_dataframe()

        # Check metadata columns
        assert "cell_id" in df.columns
        assert "donor_id" in df.columns
        assert "sample_id" in df.columns
        assert "stage_id" in df.columns

        # Check HLCA latent columns
        for i in range(4):
            assert f"hlca_latent_{i}" in df.columns

        # Check LuCa latent columns
        for i in range(4):
            assert f"luca_latent_{i}" in df.columns

        # Check fused latent columns (8 = concat of 4+4)
        for i in range(8):
            assert f"fused_latent_{i}" in df.columns

        # Check reference mode
        assert "reference_mode_used" in df.columns

    def test_normalization(self) -> None:
        """Normalization produces zero-mean unit-variance per dimension."""
        hlca = _create_mock_mapping_result(n_cells=100, latent_dim=8)
        luca = _create_mock_mapping_result(n_cells=100, latent_dim=8)
        luca = MappingResult(
            embeddings=luca.embeddings * 10 + 5,  # Shifted and scaled
            latent_dim=luca.latent_dim,
            cell_ids=hlca.cell_ids,
            donor_ids=hlca.donor_ids,
            sample_ids=hlca.sample_ids,
            stage_ids=hlca.stage_ids,
        )

        fused = fuse_dual_reference(hlca, luca, method="concat", normalize=True)

        # After normalization, should be approximately mean=0, std=1
        assert np.abs(fused.fused_embeddings.mean()) < 0.1
        assert np.abs(fused.fused_embeddings.std() - 1.0) < 0.1


class TestFuseSingleReference:
    """Tests for fuse_single_reference function."""

    def test_hlca_only(self) -> None:
        """Single HLCA reference produces valid output."""
        hlca = _create_mock_mapping_result(n_cells=20, latent_dim=16)

        fused = fuse_single_reference(hlca, "hlca")

        assert fused.n_cells == 20
        assert fused.fused_dim == 16
        assert fused.fusion_method == "single_hlca"
        assert np.all(fused.reference_mode_used == "hlca")
        assert np.all(fused.luca_embeddings == 0)  # Dummy

    def test_luca_only(self) -> None:
        """Single LuCa reference produces valid output."""
        luca = _create_mock_mapping_result(n_cells=15, latent_dim=12)

        fused = fuse_single_reference(luca, "luca")

        assert fused.n_cells == 15
        assert fused.fused_dim == 12
        assert fused.fusion_method == "single_luca"
        assert np.all(fused.reference_mode_used == "luca")
        assert np.all(fused.hlca_embeddings == 0)  # Dummy

    def test_target_dim_padding(self) -> None:
        """Target dimension padding works correctly."""
        mapping = _create_mock_mapping_result(n_cells=10, latent_dim=8)

        # Disable normalization for this test to check padding directly
        fused = fuse_single_reference(mapping, "hlca", target_dim=16, normalize=False)

        assert fused.fused_dim == 16
        # First 8 dims should match original (no normalization)
        assert np.allclose(
            fused.fused_embeddings[:, :8],
            mapping.embeddings,
        )
        # Padded dims should be zero
        assert np.allclose(fused.fused_embeddings[:, 8:], 0.0)


class TestFusedEmbeddingResult:
    """Tests for FusedEmbeddingResult dataclass."""

    def test_n_cells_property(self) -> None:
        """n_cells property returns correct count."""
        fused = FusedEmbeddingResult(
            fused_embeddings=np.random.randn(50, 16).astype(np.float32),
            fused_dim=16,
            hlca_embeddings=np.random.randn(50, 8).astype(np.float32),
            luca_embeddings=np.random.randn(50, 8).astype(np.float32),
            hlca_dim=8,
            luca_dim=8,
            cell_ids=np.array([f"c{i}" for i in range(50)]),
            donor_ids=np.array(["D1"] * 50),
            sample_ids=np.array(["S1"] * 50),
            stage_ids=np.array(["AAH"] * 50),
            fusion_method="concat",
        )

        assert fused.n_cells == 50
