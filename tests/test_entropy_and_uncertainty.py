"""Tests for entropy loss and MC dropout uncertainty estimation."""

import numpy as np
import pandas as pd
import pytest
import torch

from stagebridge.contracts import LATENT_DIM, HLCA_DIM, LUCA_DIM


class TestEntropyLoss:
    """Test attention entropy loss computation."""

    @pytest.fixture
    def model_and_batch(self):
        """Create a model and test batch."""
        from stagebridge.models import StageBridge, StageBridgeConfig

        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_learned_ring_pooling=True,
            use_context_refiner=True,
        )
        model = StageBridge(config)

        # Create dummy batch
        B = 8
        batch = {
            "receiver": torch.randn(B, LATENT_DIM),
            "ring_cells": [torch.randn(B, 10, LATENT_DIM) for _ in range(4)],
            "ring_masks": [torch.ones(B, 10, dtype=torch.bool) for _ in range(4)],
            "hlca": torch.randn(B, HLCA_DIM),
            "luca": torch.randn(B, LUCA_DIM),
        }
        return model, batch

    def test_entropy_loss_computed_during_training(self, model_and_batch):
        """Test that entropy_loss is computed when model is in training mode."""
        model, batch = model_and_batch
        model.train()

        output = model.encode_niche(
            receiver=batch["receiver"],
            ring_cells=batch["ring_cells"],
            ring_masks=batch["ring_masks"],
            hlca=batch["hlca"],
            luca=batch["luca"],
        )

        assert output.entropy_loss is not None
        assert isinstance(output.entropy_loss, torch.Tensor)
        assert output.entropy_loss.ndim == 0  # Scalar
        assert output.entropy_loss.item() > 0  # Entropy is positive

    def test_entropy_loss_none_during_eval(self, model_and_batch):
        """Test that entropy_loss is None when model is in eval mode."""
        model, batch = model_and_batch
        model.eval()

        output = model.encode_niche(
            receiver=batch["receiver"],
            ring_cells=batch["ring_cells"],
            ring_masks=batch["ring_masks"],
            hlca=batch["hlca"],
            luca=batch["luca"],
        )

        assert output.entropy_loss is None

    def test_entropy_loss_differentiable(self, model_and_batch):
        """Test that entropy_loss gradients flow back to model parameters."""
        model, batch = model_and_batch
        model.train()

        output = model.encode_niche(
            receiver=batch["receiver"],
            ring_cells=batch["ring_cells"],
            ring_masks=batch["ring_masks"],
            hlca=batch["hlca"],
            luca=batch["luca"],
        )

        # Backward pass
        output.entropy_loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Entropy loss should have gradients flowing to model parameters"

    def test_entropy_loss_range(self, model_and_batch):
        """Test that entropy loss is in a reasonable range."""
        model, batch = model_and_batch
        model.train()

        output = model.encode_niche(
            receiver=batch["receiver"],
            ring_cells=batch["ring_cells"],
            ring_masks=batch["ring_masks"],
            hlca=batch["hlca"],
            luca=batch["luca"],
        )

        entropy = output.entropy_loss.item()
        # For uniform attention over N tokens, max entropy is log(N)
        # With 9 tokens, max entropy ~ 2.2
        # Should be positive and not too large
        assert 0 < entropy < 10, f"Entropy {entropy} outside reasonable range"


class TestMCDropoutUncertainty:
    """Test Monte Carlo dropout uncertainty estimation."""

    @pytest.fixture
    def model_and_neighborhoods(self):
        """Create a model and test neighborhoods DataFrame."""
        from stagebridge.models import StageBridge, StageBridgeConfig
        import stagebridge as sb

        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_learned_ring_pooling=True,
            use_context_refiner=True,
            use_cross_attn_drift=True,
        )
        model = StageBridge(config)
        api = sb.StageBridge(model, config, device="cpu")

        # Create test neighborhoods
        n = 20
        neighborhoods = pd.DataFrame({
            "cell_id": [f"cell_{i}" for i in range(n)],
            "donor_id": ["D1"] * n,
            "stage": ["Normal"] * (n // 2) + ["Invasive"] * (n // 2),
            "receiver_z": [np.random.randn(LATENT_DIM).tolist() for _ in range(n)],
            "ring_1_cells": [[np.random.randn(LATENT_DIM).tolist() for _ in range(3)] for _ in range(n)],
            "ring_2_cells": [[np.random.randn(LATENT_DIM).tolist() for _ in range(2)] for _ in range(n)],
            "ring_3_cells": [[np.random.randn(LATENT_DIM).tolist() for _ in range(1)] for _ in range(n)],
            "ring_4_cells": [[] for _ in range(n)],
            "hlca_z": [np.random.randn(HLCA_DIM).tolist() for _ in range(n)],
            "luca_z": [np.random.randn(LUCA_DIM).tolist() for _ in range(n)],
        })

        return api, neighborhoods

    def test_predict_with_uncertainty_returns_uncertainty(self, model_and_neighborhoods):
        """Test that predict_with_uncertainty returns uncertainty estimates."""
        api, neighborhoods = model_and_neighborhoods

        output = api.predict_with_uncertainty(
            neighborhoods=neighborhoods,
            source_stage="Normal",
            target_stage="Invasive",
            n_samples=5,  # Small for fast test
            batch_size=10,
        )

        assert output.uncertainty is not None
        assert output.uncertainty_scalar is not None
        assert output.uncertainty.shape == output.predicted_embeddings.shape
        assert output.uncertainty_scalar.shape[0] == len(neighborhoods)

    def test_uncertainty_is_positive(self, model_and_neighborhoods):
        """Test that uncertainty values are non-negative."""
        api, neighborhoods = model_and_neighborhoods

        output = api.predict_with_uncertainty(
            neighborhoods=neighborhoods,
            source_stage="Normal",
            target_stage="Invasive",
            n_samples=5,
            batch_size=10,
        )

        assert np.all(output.uncertainty >= 0)
        assert np.all(output.uncertainty_scalar >= 0)

    def test_more_samples_changes_uncertainty(self, model_and_neighborhoods):
        """Test that different n_samples gives different uncertainty estimates."""
        api, neighborhoods = model_and_neighborhoods

        # Set seed for reproducibility in first call
        torch.manual_seed(42)
        output_5 = api.predict_with_uncertainty(
            neighborhoods=neighborhoods,
            source_stage="Normal",
            target_stage="Invasive",
            n_samples=5,
            batch_size=10,
        )

        torch.manual_seed(42)
        output_10 = api.predict_with_uncertainty(
            neighborhoods=neighborhoods,
            source_stage="Normal",
            target_stage="Invasive",
            n_samples=10,
            batch_size=10,
        )

        # With more samples, uncertainty estimate should be different
        # (more accurate but not necessarily lower)
        # Just check they're not identical
        assert not np.allclose(output_5.uncertainty_scalar, output_10.uncertainty_scalar)

    def test_uncertainty_with_dropout_vs_without(self, model_and_neighborhoods):
        """Test that MC dropout produces non-zero uncertainty."""
        api, neighborhoods = model_and_neighborhoods

        output = api.predict_with_uncertainty(
            neighborhoods=neighborhoods,
            source_stage="Normal",
            target_stage="Invasive",
            n_samples=10,
            batch_size=10,
        )

        # With dropout, there should be variance between samples
        # So uncertainty should be > 0 for at least some cells
        assert np.mean(output.uncertainty_scalar) > 0

    def test_prediction_output_fields(self, model_and_neighborhoods):
        """Test that all expected fields are present in output."""
        api, neighborhoods = model_and_neighborhoods

        output = api.predict_with_uncertainty(
            neighborhoods=neighborhoods,
            source_stage="Normal",
            target_stage="Invasive",
            n_samples=5,
            batch_size=10,
        )

        # Check all fields
        assert output.predicted_embeddings is not None
        assert output.source_embeddings is not None
        assert output.context_embeddings is not None
        assert output.source_stage == "Normal"
        assert output.target_stage == "Invasive"
        assert output.uncertainty is not None
        assert output.uncertainty_scalar is not None


class TestUncertaintyPlots:
    """Test uncertainty plotting functions."""

    def test_uncertainty_plot_callable(self):
        """Test that uncertainty plot is callable."""
        import stagebridge as sb

        embeddings = np.random.randn(100, 10)
        uncertainty = np.random.rand(100)
        stages = ["Normal"] * 50 + ["Invasive"] * 50

        # Should not raise
        fig = sb.pl.uncertainty(
            embeddings,
            uncertainty,
            stages=stages,
            method="pca",
            show=False,
        )
        assert fig is not None

    def test_uncertainty_by_stage_plot_callable(self):
        """Test that uncertainty_by_stage plot is callable."""
        import stagebridge as sb

        uncertainty = np.random.rand(100)
        stages = np.array(["Normal"] * 40 + ["Preinvasive"] * 30 + ["Invasive"] * 30)

        # Should not raise
        fig = sb.pl.uncertainty_by_stage(
            uncertainty,
            stages,
            show=False,
        )
        assert fig is not None

    def test_uncertainty_plot_without_stages(self):
        """Test uncertainty plot works without stages."""
        import stagebridge as sb

        embeddings = np.random.randn(100, 10)
        uncertainty = np.random.rand(100)

        # Should not raise
        fig = sb.pl.uncertainty(
            embeddings,
            uncertainty,
            stages=None,
            method="pca",
            show=False,
        )
        assert fig is not None
