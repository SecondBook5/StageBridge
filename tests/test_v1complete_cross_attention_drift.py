"""Tests for CrossAttentionDrift and SetTransformerRefiner upgrades to V1Complete.

These tests verify that:
1. Default MLP drift behavior is preserved
2. CrossAttentionDrift variant works correctly
3. SetTransformerRefiner + CrossAttentionDrift variant works correctly
4. Invalid configs raise clear errors
5. All tensor shapes are correct
6. Backward passes work without NaNs
"""

import pytest
import torch

from stagebridge.pipelines.run_v1_complete import (
    StageBridgeV1Complete,
    SetTransformerRefiner,
)
from stagebridge.transition_model.drift_network import CrossAttentionDrift


class TestDefaultMLPBehavior:
    """Ensure default V1Complete behavior is unchanged."""

    def test_v1complete_mlp_default_still_runs(self):
        """Default instantiation uses MLP drift."""
        model = StageBridgeV1Complete()

        assert model.drift_head_type == "mlp"
        assert model.context_refiner_type == "none"
        assert model.cross_attention_drift is None
        assert model.context_refiner is None

    def test_v1complete_mlp_forward(self):
        """MLP drift forward pass works with synthetic data."""
        model = StageBridgeV1Complete(latent_dim=40)
        model.eval()

        B, K, D = 4, 9, 40
        niche = torch.randn(B, K, D)

        ctx = model.encode_niche(niche)
        assert ctx.shape == (B, 256)  # context_dim default

    def test_v1complete_mlp_transition_forward(self):
        """MLP drift transition forward works."""
        model = StageBridgeV1Complete(latent_dim=40)
        model.eval()

        B, D = 8, 40
        z_src = torch.randn(B, D)
        z_tgt = torch.randn(B, D)
        ctx = torch.randn(B, 256)
        stage_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

        with torch.no_grad():
            result = model.transition_forward(
                z_source=z_src,
                z_target=z_tgt,
                context=ctx,
                stage_indices=stage_idx,
                use_ot=True,
            )

        assert "loss_transition" in result
        assert "drift_pred" in result
        assert result["context_gate_mean"] is None  # MLP has no gate


class TestCrossAttentionDriftShapes:
    """Test CrossAttentionDrift module in isolation."""

    def test_cross_attention_drift_shapes(self):
        """CrossAttentionDrift produces correct output shape."""
        B, K = 4, 8
        latent_dim, context_dim, time_dim, stage_dim = 40, 128, 64, 32

        drift = CrossAttentionDrift(
            input_dim=latent_dim,
            context_dim=context_dim,
            time_dim=time_dim,
            stage_dim=stage_dim,
            num_heads=4,
            dropout=0.1,
        )

        x_t = torch.randn(B, latent_dim)
        time_emb = torch.randn(B, time_dim)
        context_tokens = torch.randn(B, K, context_dim)
        stage_emb = torch.randn(B, stage_dim)

        output = drift(x_t, time_emb, context_tokens, stage_emb)

        assert output.shape == (B, latent_dim)

    def test_cross_attention_drift_backward_pass(self):
        """CrossAttentionDrift backward pass has no NaNs."""
        B, K = 4, 8
        latent_dim, context_dim, time_dim, stage_dim = 40, 128, 64, 32

        drift = CrossAttentionDrift(
            input_dim=latent_dim,
            context_dim=context_dim,
            time_dim=time_dim,
            stage_dim=stage_dim,
            num_heads=4,
        )

        x_t = torch.randn(B, latent_dim, requires_grad=True)
        time_emb = torch.randn(B, time_dim)
        context_tokens = torch.randn(B, K, context_dim, requires_grad=True)
        stage_emb = torch.randn(B, stage_dim)
        target = torch.randn(B, latent_dim)

        output = drift(x_t, time_emb, context_tokens, stage_emb)
        loss = ((output - target) ** 2).mean()
        loss.backward()

        assert not torch.isnan(x_t.grad).any()
        assert not torch.isnan(context_tokens.grad).any()


class TestSetTransformerRefinerShapes:
    """Test SetTransformerRefiner module in isolation."""

    def test_set_transformer_refiner_shapes(self):
        """SetTransformerRefiner preserves input shape."""
        B, K, D = 4, 8, 128

        refiner = SetTransformerRefiner(
            dim=D,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        )

        tokens = torch.randn(B, K, D)
        output = refiner(tokens)

        assert output.shape == tokens.shape

    def test_set_transformer_refiner_with_mask(self):
        """SetTransformerRefiner handles masks."""
        B, K, D = 4, 8, 128

        refiner = SetTransformerRefiner(dim=D, num_layers=2, num_heads=4)

        tokens = torch.randn(B, K, D)
        mask = torch.ones(B, K, dtype=torch.bool)
        mask[:, -2:] = False  # Mask last 2 tokens

        output = refiner(tokens, mask=mask)
        assert output.shape == tokens.shape


class TestV1CompleteCrossAttentionVariant:
    """Test V1Complete with cross_attention drift."""

    def test_v1complete_cross_attention_variant_runs(self):
        """CrossAttention variant instantiates correctly."""
        model = StageBridgeV1Complete(
            latent_dim=40,
            drift_head="cross_attention",
            context_refiner="none",
        )

        assert model.drift_head_type == "cross_attention"
        assert model.context_refiner_type == "none"
        assert model.cross_attention_drift is not None
        assert model.context_refiner is None

    def test_v1complete_cross_attention_forward(self):
        """CrossAttention variant forward pass works."""
        model = StageBridgeV1Complete(
            latent_dim=40,
            drift_head="cross_attention",
            context_refiner="none",
        )
        model.eval()

        B, K, D = 4, 9, 40
        niche = torch.randn(B, K, D)

        ctx, tokens, attn = model.encode_niche_with_tokens(niche)

        assert ctx.shape == (B, 256)
        assert tokens is not None
        assert tokens.shape == (B, K - 1, 256)  # K-1 neighbor tokens

    def test_v1complete_cross_attention_transition_forward(self):
        """CrossAttention variant transition forward works."""
        model = StageBridgeV1Complete(
            latent_dim=40,
            drift_head="cross_attention",
            context_refiner="none",
        )
        model.eval()

        B, K, D = 8, 9, 40
        niche = torch.randn(B, K, D)
        ctx, tokens, _ = model.encode_niche_with_tokens(niche)

        z_src = torch.randn(B, D)
        z_tgt = torch.randn(B, D)
        stage_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

        with torch.no_grad():
            result = model.transition_forward(
                z_source=z_src,
                z_target=z_tgt,
                context=ctx,
                context_tokens=tokens,
                stage_indices=stage_idx,
                use_ot=True,
            )

        assert "loss_transition" in result
        assert "drift_pred" in result
        assert result["context_gate_mean"] is not None
        assert result["context_attention_entropy"] is not None


class TestV1CompleteCrossAttentionSetRefinerVariant:
    """Test V1Complete with cross_attention drift + set_transformer refiner."""

    def test_v1complete_cross_attention_set_refiner_variant_runs(self):
        """CrossAttention + SetRefiner variant instantiates correctly."""
        model = StageBridgeV1Complete(
            latent_dim=40,
            drift_head="cross_attention",
            context_refiner="set_transformer",
        )

        assert model.drift_head_type == "cross_attention"
        assert model.context_refiner_type == "set_transformer"
        assert model.cross_attention_drift is not None
        assert model.context_refiner is not None

    def test_v1complete_cross_attention_set_refiner_transition_forward(self):
        """CrossAttention + SetRefiner variant transition forward works."""
        model = StageBridgeV1Complete(
            latent_dim=40,
            drift_head="cross_attention",
            context_refiner="set_transformer",
            context_refiner_layers=2,
        )
        model.eval()

        B, K, D = 8, 9, 40
        niche = torch.randn(B, K, D)
        ctx, tokens, _ = model.encode_niche_with_tokens(niche)

        z_src = torch.randn(B, D)
        z_tgt = torch.randn(B, D)
        stage_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

        with torch.no_grad():
            result = model.transition_forward(
                z_source=z_src,
                z_target=z_tgt,
                context=ctx,
                context_tokens=tokens,
                stage_indices=stage_idx,
                use_ot=True,
            )

        assert "loss_transition" in result
        assert result["context_gate_mean"] is not None

    def test_v1complete_sample_trajectory_with_cross_attention(self):
        """Sample trajectory works with cross-attention drift."""
        model = StageBridgeV1Complete(
            latent_dim=40,
            drift_head="cross_attention",
            context_refiner="set_transformer",
        )
        model.eval()

        B, K, D = 4, 9, 40
        niche = torch.randn(B, K, D)
        ctx, tokens, _ = model.encode_niche_with_tokens(niche)

        z_src = torch.randn(B, D)
        stage_idx = torch.tensor([0, 1, 1, 2])

        with torch.no_grad():
            traj = model.sample_trajectory(
                z_source=z_src,
                context=ctx,
                context_tokens=tokens,
                stage_indices=stage_idx,
                n_steps=10,
            )

        assert traj.shape == (B, 11, D)  # n_steps + 1


class TestInvalidConfigs:
    """Test that invalid configs raise clear errors."""

    def test_invalid_drift_head_raises(self):
        """Invalid drift_head raises ValueError."""
        with pytest.raises(ValueError, match="Invalid drift_head"):
            StageBridgeV1Complete(drift_head="invalid")

    def test_invalid_context_refiner_raises(self):
        """Invalid context_refiner raises ValueError."""
        with pytest.raises(ValueError, match="Invalid context_refiner"):
            StageBridgeV1Complete(context_refiner="invalid")

    def test_set_refiner_requires_cross_attention(self):
        """set_transformer refiner requires cross_attention drift."""
        with pytest.raises(ValueError, match="requires drift_head='cross_attention'"):
            StageBridgeV1Complete(
                drift_head="mlp",
                context_refiner="set_transformer",
            )


class TestContextTokensReturned:
    """Test that context tokens are properly returned."""

    def test_context_tokens_are_returned(self):
        """V1Complete returns context_tokens from encode_niche_with_tokens."""
        model = StageBridgeV1Complete(
            latent_dim=40,
            drift_head="cross_attention",
        )
        model.eval()

        B, K, D = 4, 9, 40
        niche = torch.randn(B, K, D)

        ctx, tokens, attn = model.encode_niche_with_tokens(niche)

        assert ctx is not None
        assert tokens is not None
        assert attn is not None
        assert ctx.shape == (B, 256)
        assert tokens.shape == (B, K - 1, 256)
        assert attn.shape == (B, K - 1)

    def test_context_tokens_none_without_doctrine_encoder(self):
        """Fallback encoder returns None for context_tokens."""
        # Force fallback by disabling doctrine encoder
        model = StageBridgeV1Complete(
            latent_dim=40,
            use_doctrine_encoder=False,
        )
        model.eval()

        B, K, D = 4, 9, 40
        niche = torch.randn(B, K, D)

        ctx, tokens, attn = model.encode_niche_with_tokens(niche)

        assert ctx is not None
        assert tokens is None  # Fallback doesn't produce tokens


class TestBackwardPass:
    """Test backward passes for gradient flow."""

    def test_v1complete_cross_attention_backward(self):
        """Full model backward pass with cross-attention drift."""
        model = StageBridgeV1Complete(
            latent_dim=40,
            drift_head="cross_attention",
            context_refiner="set_transformer",
        )
        model.train()

        B, K, D = 8, 9, 40
        niche = torch.randn(B, K, D, requires_grad=True)

        ctx, tokens, _ = model.encode_niche_with_tokens(niche)

        z_src = torch.randn(B, D, requires_grad=True)
        z_tgt = torch.randn(B, D)
        stage_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

        result = model.transition_forward(
            z_source=z_src,
            z_target=z_tgt,
            context=ctx,
            context_tokens=tokens,
            stage_indices=stage_idx,
            use_ot=True,
        )

        loss = result["loss_transition"]
        loss.backward()

        # Check gradients exist and have no NaNs
        assert z_src.grad is not None
        assert not torch.isnan(z_src.grad).any()
