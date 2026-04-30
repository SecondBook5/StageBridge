"""Core model tests.

Tests for StageBridge model forward pass, configuration, and components.
"""

from __future__ import annotations

import pytest
import torch

from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.contracts import LATENT_DIM, HLCA_DIM, LUCA_DIM, STATS_TOKEN_DIM, STAGE_TO_IDX


class TestStageBridgeConfig:
    """Test model configuration."""

    def test_default_config(self):
        """Default config should be valid."""
        config = StageBridgeConfig()
        assert config.input_dim == LATENT_DIM
        assert config.num_stages == len(STAGE_TO_IDX)

    def test_custom_config(self):
        """Custom config values should be preserved."""
        config = StageBridgeConfig(
            hidden_dim=256,
            num_heads=8,
            num_encoder_layers=4,
        )
        assert config.hidden_dim == 256
        assert config.num_heads == 8
        assert config.num_encoder_layers == 4

    def test_config_validation(self):
        """Config should validate constraints."""
        config = StageBridgeConfig(hidden_dim=128, num_heads=4)
        assert config.hidden_dim % config.num_heads == 0


class TestStageBridgeModel:
    """Test StageBridge model."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    @pytest.fixture
    def model(self) -> StageBridge:
        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=2,
            use_learned_ring_pooling=True,
            use_context_refiner=True,
        )
        return StageBridge(config)

    @pytest.fixture
    def batch(self) -> dict:
        batch_size = 4
        max_cells = 10
        return {
            "receiver": torch.randn(batch_size, LATENT_DIM),
            "ring_cells": [torch.randn(batch_size, max_cells, LATENT_DIM) for _ in range(4)],
            "ring_masks": [torch.ones(batch_size, max_cells, dtype=torch.bool) for _ in range(4)],
            "hlca": torch.randn(batch_size, HLCA_DIM),
            "luca": torch.randn(batch_size, LUCA_DIM),
            "pathway": torch.randn(batch_size, LATENT_DIM),
            "stats": torch.randn(batch_size, STATS_TOKEN_DIM),
        }

    def test_model_creation(self, model: StageBridge):
        """Model should be created with expected components."""
        assert model.niche_tokenizer is not None
        assert model.context_refiner is not None
        assert model.drift_head is not None

    def test_parameter_count(self, model: StageBridge):
        """Model should have reasonable parameter count."""
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 100_000  # At least 100K params
        assert n_params < 100_000_000  # Less than 100M params

    def test_encode_niche(self, model: StageBridge, batch: dict):
        """encode_niche should produce valid output."""
        output = model.encode_niche(**batch)

        assert output.context is not None
        assert output.context.shape == (4, 64)  # batch_size x hidden_dim
        assert output.context_tokens is not None
        assert output.context_tokens.shape[0] == 4  # batch_size

    def test_forward_vector_field(self, model: StageBridge, batch: dict):
        """forward_vector_field should predict velocities."""
        niche_output = model.encode_niche(**batch)

        x_t = torch.randn(4, LATENT_DIM)
        t = torch.rand(4)
        stage_pair_id = torch.zeros(4, dtype=torch.long)

        velocity = model.forward_vector_field(
            x_t=x_t,
            t=t,
            context=niche_output.context,
            stage_pair_id=stage_pair_id,
            context_tokens=niche_output.context_tokens,
        )

        assert velocity.shape == (4, LATENT_DIM)
        assert not torch.isnan(velocity).any()

    def test_integrate_euler(self, model: StageBridge, batch: dict):
        """integrate_euler should produce trajectories."""
        niche_output = model.encode_niche(**batch)
        stage_pair_id = model.encode_stage_pair_tensor(0, 1, 4, batch["receiver"].device)

        x1 = model.integrate_euler(
            x0=batch["receiver"],
            context=niche_output.context,
            stage_pair_id=stage_pair_id,
            num_steps=5,
            context_tokens=niche_output.context_tokens,
        )

        assert x1.shape == batch["receiver"].shape
        assert not torch.isnan(x1).any()

    def test_encode_stage_pair_tensor(self, model: StageBridge):
        """Stage pair encoding should work for all valid pairs."""
        for src in range(3):
            for tgt in range(3):
                if src != tgt:
                    tensor = model.encode_stage_pair_tensor(src, tgt, n=2, device="cpu")
                    assert tensor.shape == (2,)
                    assert tensor.dtype == torch.long

    def test_model_eval_mode(self, model: StageBridge, batch: dict):
        """Model should work in eval mode."""
        model.eval()
        with torch.no_grad():
            output = model.encode_niche(**batch)
        assert output.context is not None

    def test_variable_ring_sizes(self, model: StageBridge):
        """Model should handle variable numbers of cells per ring."""
        batch_size = 4
        # Different number of cells in each ring
        ring_cells = [
            torch.randn(batch_size, 5, LATENT_DIM),   # Ring 1: 5 cells
            torch.randn(batch_size, 12, LATENT_DIM),  # Ring 2: 12 cells
            torch.randn(batch_size, 8, LATENT_DIM),   # Ring 3: 8 cells
            torch.randn(batch_size, 20, LATENT_DIM),  # Ring 4: 20 cells
        ]
        ring_masks = [
            torch.ones(batch_size, 5, dtype=torch.bool),
            torch.ones(batch_size, 12, dtype=torch.bool),
            torch.ones(batch_size, 8, dtype=torch.bool),
            torch.ones(batch_size, 20, dtype=torch.bool),
        ]

        output = model.encode_niche(
            receiver=torch.randn(batch_size, LATENT_DIM),
            ring_cells=ring_cells,
            ring_masks=ring_masks,
            hlca=torch.randn(batch_size, HLCA_DIM),
            luca=torch.randn(batch_size, LUCA_DIM),
            pathway=torch.randn(batch_size, LATENT_DIM),
            stats=torch.randn(batch_size, STATS_TOKEN_DIM),
        )
        assert output.context.shape == (batch_size, 64)

    def test_masked_cells(self, model: StageBridge):
        """Model should handle masked (padded) cells correctly."""
        batch_size = 4
        max_cells = 10

        ring_cells = [torch.randn(batch_size, max_cells, LATENT_DIM) for _ in range(4)]
        # Mask out some cells
        ring_masks = [torch.ones(batch_size, max_cells, dtype=torch.bool) for _ in range(4)]
        ring_masks[0][:, 5:] = False  # Only 5 valid cells in ring 1
        ring_masks[1][:, 3:] = False  # Only 3 valid cells in ring 2

        output = model.encode_niche(
            receiver=torch.randn(batch_size, LATENT_DIM),
            ring_cells=ring_cells,
            ring_masks=ring_masks,
            hlca=torch.randn(batch_size, HLCA_DIM),
            luca=torch.randn(batch_size, LUCA_DIM),
            pathway=torch.randn(batch_size, LATENT_DIM),
            stats=torch.randn(batch_size, STATS_TOKEN_DIM),
        )
        assert not torch.isnan(output.context).any()


class TestNicheTokenizer:
    """Test NicheTokenizer component."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    @pytest.fixture
    def tokenizer(self):
        from stagebridge.context.tokenizer import NicheTokenizer
        return NicheTokenizer(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_rings=4,
            num_heads=2,
            num_inducing=4,
        )

    def test_tokenizer_output_shape(self, tokenizer):
        """Tokenizer should produce correct output shape."""
        batch_size = 4
        max_cells = 10

        tokens, reconstruction, _ = tokenizer(
            receiver=torch.randn(batch_size, LATENT_DIM),
            ring_cells=[torch.randn(batch_size, max_cells, LATENT_DIM) for _ in range(4)],
            ring_masks=[torch.ones(batch_size, max_cells, dtype=torch.bool) for _ in range(4)],
            hlca=torch.randn(batch_size, HLCA_DIM),
            luca=torch.randn(batch_size, LUCA_DIM),
        )

        # 9 tokens: receiver + 4 rings + hlca + luca + pathway + stats (last two optional)
        assert tokens.shape[0] == batch_size
        assert tokens.shape[2] == 64  # hidden_dim

    def test_tokenizer_reconstruction(self, tokenizer):
        """Tokenizer should produce reconstruction for SSL."""
        batch_size = 4
        receiver = torch.randn(batch_size, LATENT_DIM)

        _, reconstruction, _ = tokenizer(
            receiver=receiver,
            ring_cells=[torch.randn(batch_size, 10, LATENT_DIM) for _ in range(4)],
            ring_masks=[torch.ones(batch_size, 10, dtype=torch.bool) for _ in range(4)],
            hlca=torch.randn(batch_size, HLCA_DIM),
            luca=torch.randn(batch_size, LUCA_DIM),
        )

        assert reconstruction.shape == receiver.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
