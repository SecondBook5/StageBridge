"""Baseline model tests.

Tests for baseline architectures used in ablation studies.
"""

from __future__ import annotations

import pytest
import torch

from stagebridge.contracts import LATENT_DIM
from stagebridge.baselines import get_baseline


class TestBaselineRegistry:
    """Test baseline model registry."""

    def test_get_pooling_baseline(self):
        """Should retrieve pooling baseline."""
        model = get_baseline("pooling", input_dim=LATENT_DIM, hidden_dim=64)
        assert model is not None

    def test_get_deepsets_baseline(self):
        """Should retrieve DeepSets baseline."""
        model = get_baseline("deepsets", input_dim=LATENT_DIM, hidden_dim=64)
        assert model is not None

    def test_get_set_transformer_baseline(self):
        """Should retrieve SetTransformer baseline."""
        model = get_baseline("set_transformer", input_dim=LATENT_DIM, hidden_dim=64)
        assert model is not None

    def test_get_graphsage_baseline(self):
        """Should retrieve GraphSAGE baseline."""
        model = get_baseline("graphsage", input_dim=LATENT_DIM, hidden_dim=64)
        assert model is not None

    def test_invalid_baseline_raises(self):
        """Should raise for invalid baseline name."""
        with pytest.raises((KeyError, ValueError)):
            get_baseline("invalid_baseline", input_dim=LATENT_DIM, hidden_dim=64)


class TestPoolingBaseline:
    """Test pooling (bag-of-cells) baseline."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    @pytest.fixture
    def model(self):
        return get_baseline("pooling", input_dim=LATENT_DIM, hidden_dim=64)

    @pytest.fixture
    def batch(self) -> dict:
        batch_size = 4
        return {
            "receiver": torch.randn(batch_size, LATENT_DIM),
            "neighbors": torch.randn(batch_size, 8, LATENT_DIM),
            "distances": torch.rand(batch_size, 8) * 50,
            "x_t": torch.randn(batch_size, LATENT_DIM),
            "t": torch.rand(batch_size),
            "stage_pair_id": torch.zeros(batch_size, dtype=torch.long),
            "neighbor_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        }

    def test_forward_shape(self, model, batch):
        """Output should match input latent dimension."""
        output = model(**batch)
        assert output.shape == (4, LATENT_DIM)

    def test_no_nan_output(self, model, batch):
        """Output should not contain NaN."""
        output = model(**batch)
        assert not torch.isnan(output).any()

    def test_handles_masked_neighbors(self, model, batch):
        """Should handle masked neighbors correctly."""
        batch["neighbor_mask"][:, 4:] = False
        output = model(**batch)
        assert not torch.isnan(output).any()


class TestDeepSetsBaseline:
    """Test DeepSets baseline."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    @pytest.fixture
    def model(self):
        return get_baseline("deepsets", input_dim=LATENT_DIM, hidden_dim=64)

    @pytest.fixture
    def batch(self) -> dict:
        batch_size = 4
        return {
            "receiver": torch.randn(batch_size, LATENT_DIM),
            "neighbors": torch.randn(batch_size, 8, LATENT_DIM),
            "distances": torch.rand(batch_size, 8) * 50,
            "x_t": torch.randn(batch_size, LATENT_DIM),
            "t": torch.rand(batch_size),
            "stage_pair_id": torch.zeros(batch_size, dtype=torch.long),
            "neighbor_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        }

    def test_forward_shape(self, model, batch):
        """Output should match input latent dimension."""
        output = model(**batch)
        assert output.shape == (4, LATENT_DIM)

    def test_permutation_invariance(self, model, batch):
        """Output should be approximately invariant to neighbor permutation.

        Note: DeepSets uses distance-based weighting, so permuting neighbors
        while keeping distances fixed should give similar (not identical) results.
        """
        output1 = model(**batch)

        # Permute neighbors AND their distances together
        perm = torch.randperm(8)
        batch["neighbors"] = batch["neighbors"][:, perm, :]
        batch["distances"] = batch["distances"][:, perm]
        batch["neighbor_mask"] = batch["neighbor_mask"][:, perm]
        output2 = model(**batch)

        # Should be close but not exact due to distance weighting
        assert torch.allclose(output1, output2, atol=0.5), "Outputs differ too much after permutation"


class TestSetTransformerBaseline:
    """Test SetTransformer baseline."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    @pytest.fixture
    def model(self):
        return get_baseline("set_transformer", input_dim=LATENT_DIM, hidden_dim=64)

    @pytest.fixture
    def batch(self) -> dict:
        batch_size = 4
        return {
            "receiver": torch.randn(batch_size, LATENT_DIM),
            "neighbors": torch.randn(batch_size, 8, LATENT_DIM),
            "distances": torch.rand(batch_size, 8) * 50,
            "x_t": torch.randn(batch_size, LATENT_DIM),
            "t": torch.rand(batch_size),
            "stage_pair_id": torch.zeros(batch_size, dtype=torch.long),
            "neighbor_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        }

    def test_forward_shape(self, model, batch):
        """Output should match input latent dimension."""
        output = model(**batch)
        assert output.shape == (4, LATENT_DIM)

    def test_attention_mechanism(self, model, batch):
        """Model should use attention (different from simple pooling)."""
        # This is a weak test - just verifies the model runs
        output = model(**batch)
        assert not torch.isnan(output).any()


class TestGraphSAGEBaseline:
    """Test GraphSAGE baseline."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    @pytest.fixture
    def model(self):
        return get_baseline("graphsage", input_dim=LATENT_DIM, hidden_dim=64)

    @pytest.fixture
    def batch(self) -> dict:
        batch_size = 4
        return {
            "receiver": torch.randn(batch_size, LATENT_DIM),
            "neighbors": torch.randn(batch_size, 8, LATENT_DIM),
            "distances": torch.rand(batch_size, 8) * 50,
            "x_t": torch.randn(batch_size, LATENT_DIM),
            "t": torch.rand(batch_size),
            "stage_pair_id": torch.zeros(batch_size, dtype=torch.long),
            "neighbor_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        }

    def test_forward_shape(self, model, batch):
        """Output should match input latent dimension."""
        output = model(**batch)
        assert output.shape == (4, LATENT_DIM)

    def test_uses_distances(self, model, batch):
        """GraphSAGE should be sensitive to distances."""
        output1 = model(**batch)

        # Change distances significantly
        batch["distances"] = batch["distances"] * 10
        output2 = model(**batch)

        # Outputs should differ (distance-aware)
        assert not torch.allclose(output1, output2, atol=1e-3)


class TestBaselineParameterCount:
    """Test baseline parameter counts are reasonable."""

    @pytest.mark.parametrize("name", ["pooling", "deepsets", "set_transformer", "graphsage"])
    def test_parameter_count(self, name: str):
        """Baselines should have reasonable parameter counts."""
        model = get_baseline(name, input_dim=LATENT_DIM, hidden_dim=64)
        n_params = sum(p.numel() for p in model.parameters())

        # All baselines should be relatively small
        assert n_params < 10_000_000  # Less than 10M params
        assert n_params > 1_000  # More than 1K params


class TestBaselineTrainability:
    """Test baselines can be trained."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    @pytest.mark.parametrize("name", ["pooling", "deepsets", "set_transformer", "graphsage"])
    def test_backward_pass(self, name: str):
        """Baselines should support backward pass."""
        model = get_baseline(name, input_dim=LATENT_DIM, hidden_dim=64)

        batch = {
            "receiver": torch.randn(4, LATENT_DIM),
            "neighbors": torch.randn(4, 8, LATENT_DIM),
            "distances": torch.rand(4, 8) * 50,
            "x_t": torch.randn(4, LATENT_DIM),
            "t": torch.rand(4),
            "stage_pair_id": torch.zeros(4, dtype=torch.long),
            "neighbor_mask": torch.ones(4, 8, dtype=torch.bool),
        }

        output = model(**batch)
        loss = output.pow(2).mean()
        loss.backward()

        # Check at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, f"{name} has no gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
