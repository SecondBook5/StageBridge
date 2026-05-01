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


class TestPrototypeBottlenecks:
    """Test prototype bottleneck components for interpretable niche archetypes."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

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

    def test_niche_prototype_bottleneck_disabled(self, batch: dict):
        """Model without niche prototypes should not produce prototype composition."""
        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_niche_prototypes=False,
        )
        model = StageBridge(config)
        output = model.encode_niche(**batch)

        assert output.niche_prototype_composition is None

    def test_niche_prototype_bottleneck_enabled(self, batch: dict):
        """Model with niche prototypes should produce valid prototype composition."""
        num_prototypes = 8
        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_niche_prototypes=True,
            num_niche_prototypes=num_prototypes,
        )
        model = StageBridge(config)
        output = model.encode_niche(**batch)

        # Should have prototype composition
        assert output.niche_prototype_composition is not None
        assert output.niche_prototype_composition.shape[-1] == num_prototypes

        # Composition should sum to 1 (soft assignment)
        composition_sum = output.niche_prototype_composition.sum(dim=-1)
        assert torch.allclose(composition_sum, torch.ones_like(composition_sum), atol=1e-5)

        # Context should still be valid
        assert output.context.shape == (4, 64)
        assert not torch.isnan(output.context).any()

    def test_niche_prototype_bottleneck_gradients(self, batch: dict):
        """Gradients should flow through niche prototype bottleneck."""
        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_niche_prototypes=True,
            num_niche_prototypes=8,
        )
        model = StageBridge(config)
        output = model.encode_niche(**batch)

        # Backprop through context
        loss = output.context.sum()
        loss.backward()

        # Prototype bank should have gradients
        assert model.niche_prototype_bottleneck.prototypes.grad is not None
        assert not torch.isnan(model.niche_prototype_bottleneck.prototypes.grad).any()

    def test_hierarchical_prototype_bottleneck_disabled(self):
        """Hierarchical aggregator without prototypes should work."""
        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_hierarchical=True,
            hierarchical_use_prototypes=False,
        )
        model = StageBridge(config)

        # Aggregate multiple niches
        niche_embeddings = torch.randn(2, 10, 64)  # 2 samples, 10 niches each
        result = model.aggregate_niches(niche_embeddings)

        assert "sample_embedding" in result
        assert result["sample_embedding"].shape == (2, 64)
        assert result.get("prototype_output") is None

    def test_hierarchical_prototype_bottleneck_enabled(self):
        """Hierarchical aggregator with prototypes should produce composition."""
        num_prototypes = 8
        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_hierarchical=True,
            hierarchical_use_prototypes=True,
            hierarchical_num_prototypes=num_prototypes,
        )
        model = StageBridge(config)

        # Aggregate multiple niches
        niche_embeddings = torch.randn(2, 10, 64)  # 2 samples, 10 niches each
        result = model.aggregate_niches(niche_embeddings)

        assert "sample_embedding" in result
        assert result["sample_embedding"].shape == (2, 64)

        # Should have prototype output
        assert result.get("prototype_output") is not None
        proto_out = result["prototype_output"]
        # Hierarchical bottleneck routes aggregated niche embeddings through prototypes
        # Shape is [B, num_prototypes] since it operates on per-niche level before aggregation
        assert proto_out.prototype_composition.shape[-1] == num_prototypes

    def test_dual_prototypes_enabled(self, batch: dict):
        """Both prototype bottlenecks should work together."""
        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_niche_prototypes=True,
            num_niche_prototypes=16,
            use_hierarchical=True,
            hierarchical_use_prototypes=True,
            hierarchical_num_prototypes=8,
        )
        model = StageBridge(config)

        # First encode individual niches
        output = model.encode_niche(**batch)
        assert output.niche_prototype_composition is not None
        assert output.niche_prototype_composition.shape[-1] == 16

        # Then aggregate multiple niches
        niche_embeddings = torch.randn(2, 10, 64)
        result = model.aggregate_niches(niche_embeddings)
        assert result["prototype_output"] is not None
        assert result["prototype_output"].prototype_composition.shape[-1] == 8

    def test_prototype_interpretability(self, batch: dict):
        """Prototype assignments should be interpretable (not uniform)."""
        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_niche_prototypes=True,
            num_niche_prototypes=8,
        )
        model = StageBridge(config)

        # Run multiple batches
        compositions = []
        for _ in range(5):
            batch_copy = {k: v.clone() if torch.is_tensor(v) else [t.clone() for t in v] for k, v in batch.items()}
            output = model.encode_niche(**batch_copy)
            compositions.append(output.niche_prototype_composition)

        # Stack and check variance - should not be uniform assignment
        all_comps = torch.stack(compositions)
        # Each sample should have some variance in prototype assignment
        per_sample_std = all_comps.std(dim=-1).mean()
        assert per_sample_std > 0.01, "Prototype assignments too uniform"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
