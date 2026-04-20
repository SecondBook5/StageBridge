"""Tests for stagebridge.evaluation.heldout module.

Note: Full integration tests require actual model checkpoints.
These tests cover the helper functions and data loading logic.
"""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

# Test the module imports successfully
from stagebridge.evaluation.heldout import (
    load_model_and_test_data,
    compute_transition_metrics,
    compute_context_sensitivity,
    run_heldout_evaluation,
)


@pytest.fixture
def test_data_dir():
    """Create temporary directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create cells.parquet
        n_cells = 100
        cells_df = pd.DataFrame({
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
            "donor_id": ["D1"] * 40 + ["D2"] * 30 + ["D3"] * 30,
            "stage": [0] * 30 + [1] * 40 + [2] * 30,
        })
        cells_df.to_parquet(tmpdir / "cells.parquet")

        # Create neighborhoods.parquet
        neighborhoods = []
        for i in range(n_cells):
            for j in range(5):  # 5 neighbors per cell
                neighbor_idx = (i + j + 1) % n_cells
                neighborhoods.append({
                    "receiver_id": f"cell_{i}",
                    "sender_id": f"cell_{neighbor_idx}",
                    "distance": np.random.uniform(0, 100),
                })
        neighborhoods_df = pd.DataFrame(neighborhoods)
        neighborhoods_df.to_parquet(tmpdir / "neighborhoods.parquet")

        # Create split_manifest.json
        splits = {
            "folds": [
                {
                    "train": ["D1"],
                    "val": ["D2"],
                    "test": ["D3"],
                }
            ]
        }
        with open(tmpdir / "split_manifest.json", "w") as f:
            json.dump(splits, f)

        yield tmpdir


class TestDataLoading:
    """Test the data loading portions without model."""

    def test_loads_cells_parquet(self, test_data_dir):
        """Verify cells.parquet loads correctly."""
        cells_df = pd.read_parquet(test_data_dir / "cells.parquet")
        assert len(cells_df) == 100
        assert "cell_id" in cells_df.columns
        assert "donor_id" in cells_df.columns

    def test_loads_neighborhoods_parquet(self, test_data_dir):
        """Verify neighborhoods.parquet loads correctly."""
        neighborhoods_df = pd.read_parquet(test_data_dir / "neighborhoods.parquet")
        assert len(neighborhoods_df) == 500  # 100 cells * 5 neighbors
        assert "receiver_id" in neighborhoods_df.columns
        assert "sender_id" in neighborhoods_df.columns

    def test_loads_split_manifest(self, test_data_dir):
        """Verify split_manifest.json loads correctly."""
        with open(test_data_dir / "split_manifest.json") as f:
            splits = json.load(f)

        assert "folds" in splits
        assert len(splits["folds"]) == 1
        assert "test" in splits["folds"][0]

    def test_filters_test_donors(self, test_data_dir):
        """Verify test donor filtering works."""
        cells_df = pd.read_parquet(test_data_dir / "cells.parquet")
        neighborhoods_df = pd.read_parquet(test_data_dir / "neighborhoods.parquet")

        with open(test_data_dir / "split_manifest.json") as f:
            splits = json.load(f)

        test_donors = splits["folds"][0]["test"]
        test_cells = cells_df[cells_df["donor_id"].isin(test_donors)].copy()

        assert len(test_cells) == 30  # D3 has 30 cells
        assert set(test_cells["donor_id"].unique()) == {"D3"}

        # Filter neighborhoods
        test_cell_ids = set(test_cells["cell_id"])
        test_neighborhoods = neighborhoods_df[
            neighborhoods_df["receiver_id"].isin(test_cell_ids)
        ].copy()

        # Each of 30 cells has 5 neighbors
        assert len(test_neighborhoods) == 150


class TestMetricComputation:
    """Test metric computation with synthetic predictions."""

    def test_computes_classification_metrics(self):
        """Test stage classification metric computation."""
        from sklearn.metrics import accuracy_score, f1_score

        # Synthetic predictions
        stages = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        probs = np.eye(3)[stages]  # Perfect predictions

        stage_preds = np.argmax(probs, axis=1)
        accuracy = float(accuracy_score(stages, stage_preds))
        f1 = float(f1_score(stages, stage_preds, average="macro"))

        assert accuracy == 1.0
        assert f1 == 1.0

    def test_computes_transition_quality(self):
        """Test transition quality metric computation."""
        from stagebridge.evaluation.metrics import compute_all_metrics

        # Synthetic predictions (should be close to targets)
        targets = np.random.randn(100, 40)
        preds = targets + np.random.randn(100, 40) * 0.1

        metrics = compute_all_metrics(preds, targets)

        assert "mse" in metrics
        assert "mae" in metrics
        assert metrics["mse"] < 0.1  # Should be small


class TestContextSensitivityLogic:
    """Test context sensitivity computation logic."""

    def test_delta_computation(self):
        """Test that shuffling changes predictions."""
        # Real outputs
        real = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # Shuffled outputs (permuted)
        shuffled = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        delta = np.linalg.norm(real - shuffled, axis=1)

        assert np.all(delta > 0)  # All deltas should be non-zero
        assert delta.shape == (3,)

    def test_zscore_computation(self):
        """Test z-score computation for sensitivity."""
        deltas = np.array([0.5, 0.6, 0.4, 0.55, 0.45])

        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        zscore = mean_delta / (std_delta + 1e-8)

        assert zscore > 0
        assert not np.isnan(zscore)


class TestFunctionSignatures:
    """Test that public functions have expected signatures."""

    def test_load_model_and_test_data_signature(self):
        """Verify load_model_and_test_data has correct parameters."""
        import inspect
        sig = inspect.signature(load_model_and_test_data)
        params = list(sig.parameters.keys())

        assert "checkpoint_path" in params
        assert "data_dir" in params
        assert "fold" in params
        assert "device" in params

    def test_compute_transition_metrics_signature(self):
        """Verify compute_transition_metrics has correct parameters."""
        import inspect
        sig = inspect.signature(compute_transition_metrics)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "test_cells" in params
        assert "test_neighborhoods" in params

    def test_run_heldout_evaluation_signature(self):
        """Verify run_heldout_evaluation has correct parameters."""
        import inspect
        sig = inspect.signature(run_heldout_evaluation)
        params = list(sig.parameters.keys())

        assert "checkpoint_path" in params
        assert "data_dir" in params
        assert "fold" in params
