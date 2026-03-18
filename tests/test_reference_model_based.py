"""Smoke tests for model-based reference mapping."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np


def test_map_query_model_imports():
    """Test that model-based mapping imports work."""
    from stagebridge.reference.map_query_model import (
        map_query_with_scanvi_model,
        map_to_dual_reference_model_based,
    )
    assert callable(map_query_with_scanvi_model)
    assert callable(map_to_dual_reference_model_based)


def test_scanvi_model_basic_structure():
    """Test scANVI mapping can be called with mocks."""
    from stagebridge.reference.map_query_model import map_query_with_scanvi_model

    # Mock query data
    mock_query = Mock()
    mock_query.n_obs = 100
    mock_query.n_vars = 1000
    mock_query.var_names = [f"GENE{i}" for i in range(1000)]
    mock_query.obs = Mock()
    mock_query.copy = Mock(return_value=mock_query)

    # Mock model path
    model_path = Path("/fake/model")

    # This will fail at model loading, but proves the function structure is correct
    with pytest.raises((ValueError, ImportError, FileNotFoundError)):
        map_query_with_scanvi_model(mock_query, model_path, batch_size=10)


def test_dual_reference_fallback_logic():
    """Test that fallback to kNN is attempted when model fails."""
    from stagebridge.reference.map_query_model import map_to_dual_reference_model_based

    mock_query = Mock()
    mock_query.n_obs = 100
    mock_query.obs_names = Mock()
    mock_query.obs_names.astype = Mock(return_value=Mock())
    mock_query.obs_names.astype.return_value.to_numpy = Mock(return_value=np.array([f"cell_{i}" for i in range(100)]))

    # Non-existent paths
    hlca_model = Path("/fake/hlca_model")
    luca_model = Path("/fake/luca_model")

    # Should return results dict even if models don't exist
    results = map_to_dual_reference_model_based(
        mock_query,
        hlca_model_path=None,  # Models don't exist
        luca_model_path=None,
    )

    assert isinstance(results, dict)
    assert "hlca_embeddings" in results
    assert "luca_embeddings" in results
    assert "cell_ids" in results


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

    # Function should exist
    assert callable(map_query_chunked)
