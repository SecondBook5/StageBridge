"""Tests for model-based reference mapping.

Tests the canonical implementation in hlca_mapper.py.
"""

import numpy as np


def test_hlca_mapper_imports():
    """Test that canonical hlca_mapper imports work."""
    from stagebridge.reference.hlca_mapper import (
        map_full_snrna_with_hlca,
        HLCAMappingResult,
    )

    assert callable(map_full_snrna_with_hlca)
    assert HLCAMappingResult is not None


def test_run_reference_imports():
    """Test that run_reference pipeline functions import."""
    from stagebridge.pipelines.run_reference import (
        run_hpc_reference_mapping,
        run_dual_reference_mapping,
        calibrate_confidence_percentile,
        normalize_latent_space,
    )

    assert callable(run_hpc_reference_mapping)
    assert callable(run_dual_reference_mapping)
    assert callable(calibrate_confidence_percentile)
    assert callable(normalize_latent_space)


def test_calibrate_confidence_percentile():
    """Test confidence calibration works."""
    from stagebridge.pipelines.run_reference import calibrate_confidence_percentile

    distances = np.array([0.1, 0.5, 0.9, 1.5, 2.0])
    conf, method = calibrate_confidence_percentile(distances)

    assert conf.shape == distances.shape
    assert method == "percentile_rank"
    assert conf.min() >= 0.0
    assert conf.max() <= 1.0
    # Lower distance should have higher confidence
    assert conf[0] > conf[-1]


def test_normalize_latent_space():
    """Test latent normalization works."""
    from stagebridge.pipelines.run_reference import normalize_latent_space

    embeddings = np.random.randn(100, 30).astype(np.float32)

    # L2 normalization
    normed = normalize_latent_space(embeddings, method="l2")
    norms = np.linalg.norm(normed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Z-score normalization
    normed = normalize_latent_space(embeddings, method="zscore")
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-5)


def test_batched_knn_search():
    """Test that batched k-NN search logic exists."""
    from stagebridge.reference.map_query_chunked import map_query_chunked

    assert callable(map_query_chunked)
