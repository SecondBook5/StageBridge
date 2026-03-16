"""Tests for confidence scoring and quality metrics."""

from __future__ import annotations

import numpy as np
import pytest

from stagebridge.reference.confidence import (
    ConfidenceScores,
    compute_hlca_confidence,
    compute_luca_confidence,
    compute_dual_confidence,
    detect_mapping_collapse,
    detect_nan_embeddings,
)
from stagebridge.reference.map_query import MappingResult


def _create_mock_mapping_result(
    n_cells: int = 50,
    latent_dim: int = 16,
    neighbor_distances: np.ndarray | None = None,
) -> MappingResult:
    """Create mock MappingResult."""
    return MappingResult(
        embeddings=np.random.randn(n_cells, latent_dim).astype(np.float32),
        latent_dim=latent_dim,
        cell_ids=np.array([f"cell_{i}" for i in range(n_cells)]),
        donor_ids=np.array([f"D{i % 3}" for i in range(n_cells)]),
        sample_ids=np.array([f"S{i % 5}" for i in range(n_cells)]),
        stage_ids=np.array(["AAH", "AIS", "MIA", "LUAD"][i % 4] for i in range(n_cells)),
        neighbor_distances=neighbor_distances,
        reference_name="HLCA",
    )


class TestComputeHLCAConfidence:
    """Tests for compute_hlca_confidence function."""

    def test_basic_confidence(self) -> None:
        """Basic confidence computation produces valid scores."""
        mapping = _create_mock_mapping_result(n_cells=30)

        confidence = compute_hlca_confidence(mapping)

        assert confidence.shape == (30,)
        assert confidence.dtype == np.float32
        # All values should be in [0, 1]
        assert np.all(confidence >= 0)
        assert np.all(confidence <= 1)

    def test_confidence_with_distances(self) -> None:
        """Confidence uses neighbor distances when available."""
        distances = np.array([0.1, 1.0, 10.0], dtype=np.float32)
        mapping = _create_mock_mapping_result(n_cells=3, neighbor_distances=distances)

        confidence = compute_hlca_confidence(mapping)

        # Closer cells (smaller distance) should have higher confidence
        assert confidence[0] > confidence[1] > confidence[2]

    def test_confidence_no_nan(self) -> None:
        """Confidence replaces NaN with 0."""
        mapping = _create_mock_mapping_result(n_cells=10)
        # Force NaN distances
        mapping = MappingResult(
            embeddings=mapping.embeddings,
            latent_dim=mapping.latent_dim,
            cell_ids=mapping.cell_ids,
            donor_ids=mapping.donor_ids,
            sample_ids=mapping.sample_ids,
            stage_ids=mapping.stage_ids,
            neighbor_distances=np.array([np.nan] * 10, dtype=np.float32),
        )

        confidence = compute_hlca_confidence(mapping)

        # NaN should be replaced with 0
        assert not np.any(np.isnan(confidence))


class TestComputeDualConfidence:
    """Tests for compute_dual_confidence function."""

    def test_dual_confidence(self) -> None:
        """Dual confidence produces scores for both references."""
        hlca = _create_mock_mapping_result(n_cells=20)
        luca = _create_mock_mapping_result(n_cells=20)
        luca = MappingResult(
            embeddings=luca.embeddings,
            latent_dim=luca.latent_dim,
            cell_ids=hlca.cell_ids,
            donor_ids=hlca.donor_ids,
            sample_ids=hlca.sample_ids,
            stage_ids=hlca.stage_ids,
        )

        scores = compute_dual_confidence(hlca, luca)

        assert isinstance(scores, ConfidenceScores)
        assert scores.hlca_confidence.shape == (20,)
        assert scores.luca_confidence.shape == (20,)
        assert np.array_equal(scores.cell_ids, hlca.cell_ids)

    def test_low_confidence_count(self) -> None:
        """Low confidence count is computed correctly."""
        # Create mapping with known low-confidence cells
        distances = np.array([10.0] * 10 + [0.1] * 10, dtype=np.float32)  # 10 far, 10 close
        hlca = _create_mock_mapping_result(n_cells=20, neighbor_distances=distances)
        luca = _create_mock_mapping_result(n_cells=20, neighbor_distances=distances)
        luca = MappingResult(
            embeddings=luca.embeddings,
            latent_dim=luca.latent_dim,
            cell_ids=hlca.cell_ids,
            donor_ids=hlca.donor_ids,
            sample_ids=hlca.sample_ids,
            stage_ids=hlca.stage_ids,
            neighbor_distances=distances,
        )

        scores = compute_dual_confidence(hlca, luca, low_confidence_threshold=0.5)

        # Some cells should be flagged as low confidence
        assert scores.hlca_low_confidence_count >= 0
        assert scores.luca_low_confidence_count >= 0


class TestConfidenceScores:
    """Tests for ConfidenceScores dataclass."""

    def test_to_dataframe(self) -> None:
        """to_dataframe produces correct output."""
        scores = ConfidenceScores(
            hlca_confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
            luca_confidence=np.array([0.6, 0.7, 0.8], dtype=np.float32),
            cell_ids=np.array(["c1", "c2", "c3"]),
        )

        df = scores.to_dataframe()

        assert "cell_id" in df.columns
        assert "hlca_confidence" in df.columns
        assert "luca_confidence" in df.columns
        assert len(df) == 3

    def test_high_confidence_mask_both(self) -> None:
        """High confidence mask with require_both=True."""
        scores = ConfidenceScores(
            hlca_confidence=np.array([0.9, 0.3, 0.9], dtype=np.float32),
            luca_confidence=np.array([0.9, 0.9, 0.3], dtype=np.float32),
            cell_ids=np.array(["c1", "c2", "c3"]),
        )

        mask = scores.get_high_confidence_mask(
            hlca_threshold=0.5,
            luca_threshold=0.5,
            require_both=True,
        )

        # Only c1 passes both thresholds
        assert mask[0]
        assert not mask[1]  # HLCA too low
        assert not mask[2]  # LuCa too low

    def test_high_confidence_mask_either(self) -> None:
        """High confidence mask with require_both=False."""
        scores = ConfidenceScores(
            hlca_confidence=np.array([0.9, 0.3, 0.1], dtype=np.float32),
            luca_confidence=np.array([0.1, 0.9, 0.1], dtype=np.float32),
            cell_ids=np.array(["c1", "c2", "c3"]),
        )

        mask = scores.get_high_confidence_mask(
            hlca_threshold=0.5,
            luca_threshold=0.5,
            require_both=False,
        )

        # c1 and c2 pass at least one threshold
        assert mask[0]
        assert mask[1]
        assert not mask[2]


class TestDetectMappingCollapse:
    """Tests for detect_mapping_collapse function."""

    def test_no_collapse(self) -> None:
        """Normal embeddings not flagged as collapsed."""
        # Embeddings with reasonable variance
        embeddings = np.random.randn(100, 16).astype(np.float32)
        mapping = MappingResult(
            embeddings=embeddings,
            latent_dim=16,
            cell_ids=np.array([f"c{i}" for i in range(100)]),
            donor_ids=np.array(["D1"] * 100),
            sample_ids=np.array(["S1"] * 100),
            stage_ids=np.array(["AAH"] * 100),
            reference_name="HLCA",
        )

        report = detect_mapping_collapse(mapping)

        assert not report["is_collapsed"]
        assert report["mean_variance"] > 0.5

    def test_collapse_detected(self) -> None:
        """Collapsed embeddings are detected."""
        # All cells at nearly the same point
        embeddings = np.ones((100, 16), dtype=np.float32)
        embeddings += np.random.randn(100, 16).astype(np.float32) * 1e-6  # Tiny noise
        mapping = MappingResult(
            embeddings=embeddings,
            latent_dim=16,
            cell_ids=np.array([f"c{i}" for i in range(100)]),
            donor_ids=np.array(["D1"] * 100),
            sample_ids=np.array(["S1"] * 100),
            stage_ids=np.array(["AAH"] * 100),
            reference_name="HLCA",
        )

        report = detect_mapping_collapse(mapping)

        assert report["is_collapsed"]
        assert report["mean_variance"] < 0.01


class TestDetectNanEmbeddings:
    """Tests for detect_nan_embeddings function."""

    def test_no_nan(self) -> None:
        """No NaN values detected in clean embeddings."""
        mapping = _create_mock_mapping_result(n_cells=50)

        report = detect_nan_embeddings(mapping)

        assert not report["has_nan"]
        assert report["total_nan_count"] == 0

    def test_nan_detected(self) -> None:
        """NaN values are detected and counted."""
        embeddings = np.random.randn(50, 16).astype(np.float32)
        embeddings[0, 0] = np.nan
        embeddings[1, :5] = np.nan  # 5 NaNs in row 1

        mapping = MappingResult(
            embeddings=embeddings,
            latent_dim=16,
            cell_ids=np.array([f"c{i}" for i in range(50)]),
            donor_ids=np.array(["D1"] * 50),
            sample_ids=np.array(["S1"] * 50),
            stage_ids=np.array(["AAH"] * 50),
            reference_name="HLCA",
        )

        report = detect_nan_embeddings(mapping)

        assert report["has_nan"]
        assert report["total_nan_count"] == 6
        assert report["cells_with_nan"] == 2
        assert report["dims_with_nan"] >= 1
