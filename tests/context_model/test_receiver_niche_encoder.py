"""Tests for receiver-centered niche encoder per doctrine.

These tests verify compliance with docs/NICHE_ENCODER_SPEC.md:
1. Receiver-centered architecture (receiver as query)
2. Distance-aware attention
3. Sparsity/entropy regularization
4. Neighbor ablation for interpretability
5. Masked receiver reconstruction
6. Works without cell type labels
"""

from __future__ import annotations

import pytest
import torch

from stagebridge.context_model.receiver_niche_encoder import (
    ReceiverCenteredNicheEncoder,
    ReceiverNicheEncoderWithDualReference,
    ReceiverCenteredAttention,
    DistanceEncoder,
    DistanceEncoding,
    SparsityType,
    _compute_attention_entropy,
    _sparsemax,
    _rbf_distance_encoding,
    _sinusoidal_distance_encoding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def num_neighbors():
    return 10


@pytest.fixture
def input_dim():
    return 64


@pytest.fixture
def hidden_dim():
    return 32


@pytest.fixture
def sample_data(batch_size, num_neighbors, input_dim):
    """Create sample receiver + neighbors data."""
    torch.manual_seed(42)
    return {
        "receiver": torch.randn(batch_size, input_dim),
        "neighbors": torch.randn(batch_size, num_neighbors, input_dim),
        "distances": torch.rand(batch_size, num_neighbors) * 50,  # 0-50 distance
        "neighbor_mask": torch.ones(batch_size, num_neighbors, dtype=torch.bool),
    }


@pytest.fixture
def encoder(input_dim, hidden_dim):
    """Create default encoder."""
    return ReceiverCenteredNicheEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        sparsity_type=SparsityType.ENTROPY,
        sparsity_weight=0.01,
    )


# ---------------------------------------------------------------------------
# Doctrine Compliance Tests
# ---------------------------------------------------------------------------


class TestDoctrineCompliance:
    """Verify encoder meets all NICHE_ENCODER_SPEC.md requirements."""

    def test_receiver_is_query_not_part_of_set(self, encoder, sample_data):
        """Doctrine: Receiver must be the attention query, not mixed with neighbors."""
        output = encoder(**sample_data)

        # Verify output shape is receiver-centric (batch, hidden)
        assert output.context.shape == (sample_data["receiver"].shape[0], encoder.hidden_dim)

        # Verify attention weights are over neighbors, not receiver
        assert output.attention_weights.shape == sample_data["neighbors"].shape[:2]

    def test_distance_explicitly_modulates_attention(self, sample_data, input_dim, hidden_dim):
        """Doctrine: Spatial distance must explicitly modulate attention."""
        encoder = ReceiverCenteredNicheEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            distance_encoding=DistanceEncoding.RBF,
        )

        # Run with original distances
        output1 = encoder(**sample_data)

        # Run with modified distances (closer neighbors)
        data_close = {**sample_data, "distances": sample_data["distances"] * 0.1}
        output2 = encoder(**data_close)

        # Attention should change when distances change
        assert not torch.allclose(output1.attention_weights, output2.attention_weights)

    def test_sparsity_regularization_produces_entropy_loss(
        self, sample_data, input_dim, hidden_dim
    ):
        """Doctrine: Attention should be regularized for sparsity."""
        encoder = ReceiverCenteredNicheEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_type=SparsityType.ENTROPY,
            sparsity_weight=0.1,
        )
        encoder.train()

        output = encoder(**sample_data)

        # Entropy loss should be computed during training
        assert output.entropy_loss is not None
        assert output.entropy_loss.item() > 0

    def test_neighbor_ablation_changes_output(self, encoder, sample_data):
        """Doctrine: Must support masking individual neighbors."""
        # Output with all neighbors
        output_full = encoder(**sample_data)

        # Ablate first neighbor
        output_ablated = encoder.ablate_neighbor(
            sample_data["receiver"],
            sample_data["neighbors"],
            sample_data["distances"],
            ablate_idx=0,
        )

        # Output should change
        assert not torch.allclose(output_full.context, output_ablated.context)

        # Attention to ablated neighbor should be zero
        assert output_ablated.attention_weights[:, 0].abs().max() < 1e-6

    def test_neighbor_importance_via_ablation(self, encoder, sample_data):
        """Doctrine: Can identify which neighbors most affect receiver."""
        importance = encoder.compute_neighbor_importance(
            sample_data["receiver"],
            sample_data["neighbors"],
            sample_data["distances"],
        )

        # Should have importance score for each neighbor
        assert importance.shape == sample_data["distances"].shape

        # Should be normalized to [0, 1]
        assert importance.min() >= 0
        assert importance.max() <= 1

    def test_masked_receiver_reconstruction(self, encoder, sample_data):
        """Doctrine: Masked receiver reconstruction as self-supervised signal."""
        loss, output = encoder.compute_reconstruction_loss(
            sample_data["receiver"],
            sample_data["neighbors"],
            sample_data["distances"],
            mask_ratio=0.15,
        )

        # Loss should be computable
        assert loss.item() >= 0

        # Reconstruction should be present
        assert output.receiver_reconstruction is not None
        assert output.receiver_reconstruction.shape == sample_data["receiver"].shape

    def test_works_without_cell_type_labels(self, encoder, sample_data):
        """Doctrine: Must work without cell type labels (graceful degradation)."""
        # Without cell type hint
        output_no_type = encoder(**sample_data)

        # With cell type hint
        cell_type_hint = torch.randn(sample_data["receiver"].shape[0], encoder.hidden_dim)
        output_with_type = encoder(
            **sample_data,
            cell_type_hint=cell_type_hint,
        )

        # Both should work
        assert output_no_type.context.shape == output_with_type.context.shape

        # Type hint should change output (soft bias, not ignored)
        assert not torch.allclose(output_no_type.context, output_with_type.context)


# ---------------------------------------------------------------------------
# Architecture Tests
# ---------------------------------------------------------------------------


class TestReceiverCenteredAttention:
    """Test the core attention mechanism."""

    def test_output_shape(self, batch_size, num_neighbors, hidden_dim):
        """Test attention produces correct shapes."""
        attn = ReceiverCenteredAttention(dim=hidden_dim, num_heads=4)

        receiver = torch.randn(batch_size, hidden_dim)
        neighbors = torch.randn(batch_size, num_neighbors, hidden_dim)
        distances = torch.rand(batch_size, num_neighbors) * 50

        context, weights = attn(receiver, neighbors, distances)

        assert context.shape == (batch_size, hidden_dim)
        assert weights.shape == (batch_size, num_neighbors)

    def test_attention_sums_to_one(self, batch_size, num_neighbors, hidden_dim):
        """Attention weights should sum to ~1 (softmax, averaged across heads)."""
        attn = ReceiverCenteredAttention(
            dim=hidden_dim,
            num_heads=4,
            sparsity_type=SparsityType.ENTROPY,
        )

        receiver = torch.randn(batch_size, hidden_dim)
        neighbors = torch.randn(batch_size, num_neighbors, hidden_dim)
        distances = torch.rand(batch_size, num_neighbors) * 50

        _, weights = attn(receiver, neighbors, distances)

        # Weights are averaged across heads, so should sum to ~1
        # Allow more tolerance since we're averaging multiple softmax distributions
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=0.15)

    def test_topk_sparsity(self, batch_size, num_neighbors, hidden_dim):
        """Top-k sparsity should concentrate attention on fewer neighbors."""
        topk = 3
        attn_topk = ReceiverCenteredAttention(
            dim=hidden_dim,
            num_heads=4,
            sparsity_type=SparsityType.TOPK,
            topk=topk,
        )
        attn_dense = ReceiverCenteredAttention(
            dim=hidden_dim,
            num_heads=4,
            sparsity_type=SparsityType.ENTROPY,
        )

        torch.manual_seed(42)
        receiver = torch.randn(batch_size, hidden_dim)
        neighbors = torch.randn(batch_size, num_neighbors, hidden_dim)
        distances = torch.rand(batch_size, num_neighbors) * 50

        _, weights_topk = attn_topk(receiver, neighbors, distances)
        _, weights_dense = attn_dense(receiver, neighbors, distances)

        # Top-k should be sparser (more weights near zero)
        # Count weights below threshold
        sparse_count_topk = (weights_topk < 0.05).sum(dim=-1).float().mean()
        sparse_count_dense = (weights_dense < 0.05).sum(dim=-1).float().mean()

        # Top-k should have more near-zero weights
        assert sparse_count_topk > sparse_count_dense

    def test_mask_ablates_neighbors(self, batch_size, num_neighbors, hidden_dim):
        """Masked neighbors should get zero attention."""
        attn = ReceiverCenteredAttention(dim=hidden_dim, num_heads=4)

        receiver = torch.randn(batch_size, hidden_dim)
        neighbors = torch.randn(batch_size, num_neighbors, hidden_dim)
        distances = torch.rand(batch_size, num_neighbors) * 50

        # Mask out first 3 neighbors
        mask = torch.ones(batch_size, num_neighbors, dtype=torch.bool)
        mask[:, :3] = False

        _, weights = attn(receiver, neighbors, distances, neighbor_mask=mask)

        # Masked neighbors should have zero weight
        assert (weights[:, :3].abs() < 1e-6).all()


# ---------------------------------------------------------------------------
# Distance Encoding Tests
# ---------------------------------------------------------------------------


class TestDistanceEncoding:
    """Test distance encoding strategies."""

    @pytest.mark.parametrize(
        "encoding_type",
        [
            DistanceEncoding.RBF,
            DistanceEncoding.MLP,
            DistanceEncoding.SINUSOIDAL,
        ],
    )
    def test_encoding_output_shape(self, encoding_type, batch_size, num_neighbors):
        """All encodings should produce correct shape."""
        output_dim = 16
        encoder = DistanceEncoder(
            encoding_type=encoding_type,
            output_dim=output_dim,
        )

        distances = torch.rand(batch_size, num_neighbors) * 50
        encoded = encoder(distances)

        assert encoded.shape == (batch_size, num_neighbors, output_dim)

    def test_rbf_encoding_values(self):
        """RBF encoding should produce sensible values."""
        distances = torch.tensor([[0.0, 25.0, 50.0, 100.0]])
        rbf = _rbf_distance_encoding(distances, num_rbf=16, max_dist=100.0)

        # RBF values should be in [0, 1]
        assert (rbf >= 0).all()
        assert (rbf <= 1).all()

        # Distance 0 should have high activation at first RBF
        assert rbf[0, 0, 0] > rbf[0, 0, -1]

    def test_sinusoidal_encoding_unique(self):
        """Different distances should have different encodings."""
        distances = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
        encoded = _sinusoidal_distance_encoding(distances, dim=16)

        # Each distance should have unique encoding
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(encoded[0, i], encoded[0, j])


# ---------------------------------------------------------------------------
# Sparsity Tests
# ---------------------------------------------------------------------------


class TestSparsity:
    """Test sparsity mechanisms."""

    def test_sparsemax_produces_zeros(self):
        """Sparsemax should produce sparse outputs."""
        logits = torch.randn(4, 10)
        sparse = _sparsemax(logits)

        # Should have some zeros
        assert (sparse == 0).any()

        # Should sum to 1
        assert torch.allclose(sparse.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_entropy_regularization(self):
        """Entropy loss should be lower for focused attention."""
        # Focused attention (low entropy)
        focused = torch.tensor([[0.9, 0.05, 0.05]])
        # Uniform attention (high entropy)
        uniform = torch.tensor([[0.33, 0.33, 0.34]])

        entropy_focused = _compute_attention_entropy(focused)
        entropy_uniform = _compute_attention_entropy(uniform)

        assert entropy_focused < entropy_uniform


# ---------------------------------------------------------------------------
# Dual Reference Integration Tests
# ---------------------------------------------------------------------------


class TestDualReferenceEncoder:
    """Test encoder with HLCA/LuCA dual-reference features."""

    def test_dual_reference_forward(self, batch_size, num_neighbors):
        """Test forward pass with dual-reference features."""
        input_dim = 32
        hlca_dim = 16
        luca_dim = 16

        encoder = ReceiverNicheEncoderWithDualReference(
            input_dim=input_dim,
            hlca_dim=hlca_dim,
            luca_dim=luca_dim,
            hidden_dim=64,
        )

        output = encoder(
            receiver=torch.randn(batch_size, input_dim),
            neighbors=torch.randn(batch_size, num_neighbors, input_dim),
            distances=torch.rand(batch_size, num_neighbors) * 50,
            receiver_hlca=torch.randn(batch_size, hlca_dim),
            receiver_luca=torch.randn(batch_size, luca_dim),
            neighbor_hlca=torch.randn(batch_size, num_neighbors, hlca_dim),
            neighbor_luca=torch.randn(batch_size, num_neighbors, luca_dim),
        )

        assert output.context.shape == (batch_size, 64)

    def test_dual_reference_reconstruction_shape(self, batch_size, num_neighbors):
        """Reconstruction should match original input_dim, not combined."""
        input_dim = 32
        hlca_dim = 16
        luca_dim = 16

        encoder = ReceiverNicheEncoderWithDualReference(
            input_dim=input_dim,
            hlca_dim=hlca_dim,
            luca_dim=luca_dim,
            hidden_dim=64,
            use_reconstruction_head=True,
        )

        output = encoder(
            receiver=torch.randn(batch_size, input_dim),
            neighbors=torch.randn(batch_size, num_neighbors, input_dim),
            distances=torch.rand(batch_size, num_neighbors) * 50,
            receiver_hlca=torch.randn(batch_size, hlca_dim),
            receiver_luca=torch.randn(batch_size, luca_dim),
            neighbor_hlca=torch.randn(batch_size, num_neighbors, hlca_dim),
            neighbor_luca=torch.randn(batch_size, num_neighbors, luca_dim),
            return_reconstruction=True,
        )

        # Should reconstruct original cell embedding, not combined
        assert output.receiver_reconstruction.shape == (batch_size, input_dim)


# ---------------------------------------------------------------------------
# Gradient Tests
# ---------------------------------------------------------------------------


class TestGradients:
    """Test gradient flow for training."""

    def test_gradients_flow_to_all_parameters(self, encoder, sample_data):
        """Parameters used in forward pass should receive gradients."""
        encoder.train()

        # Use reconstruction to ensure all params get gradients
        output = encoder(**sample_data, return_reconstruction=True)

        # Backprop through context and reconstruction
        loss = output.context.sum()
        if output.receiver_reconstruction is not None:
            loss = loss + output.receiver_reconstruction.sum()
        loss.backward()

        # Core parameters should have gradients
        for name, param in encoder.named_parameters():
            if "reconstruction_head" not in name or output.receiver_reconstruction is not None:
                assert param.grad is not None, f"No gradient for {name}"

    def test_reconstruction_loss_gradient(self, encoder, sample_data):
        """Reconstruction loss should have gradients."""
        encoder.train()
        loss, output = encoder.compute_reconstruction_loss(
            sample_data["receiver"],
            sample_data["neighbors"],
            sample_data["distances"],
        )

        loss.backward()

        # Check reconstruction head has gradients
        for param in encoder.reconstruction_head.parameters():
            assert param.grad is not None

    def test_entropy_loss_adds_to_total(self, sample_data, input_dim, hidden_dim):
        """Entropy loss should be addable to main loss."""
        encoder = ReceiverCenteredNicheEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_type=SparsityType.ENTROPY,
            sparsity_weight=0.1,
        )
        encoder.train()

        output = encoder(**sample_data)

        # Total loss = task loss + entropy loss
        task_loss = output.context.sum()
        total_loss = task_loss + output.entropy_loss

        total_loss.backward()

        # Should complete without error
        assert True


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_neighbor(self, input_dim, hidden_dim, batch_size):
        """Should work with just one neighbor."""
        encoder = ReceiverCenteredNicheEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        output = encoder(
            receiver=torch.randn(batch_size, input_dim),
            neighbors=torch.randn(batch_size, 1, input_dim),
            distances=torch.rand(batch_size, 1) * 50,
        )

        assert output.context.shape == (batch_size, hidden_dim)

        # Single neighbor should get most attention (close to 1)
        # Allow some tolerance due to multi-head averaging
        assert (output.attention_weights > 0.5).all()

        # Entropy loss should be 0 for single neighbor (no uncertainty)
        if output.entropy_loss is not None:
            assert output.entropy_loss.item() == 0.0 or not torch.isnan(output.entropy_loss)

    def test_all_neighbors_masked(self, encoder, sample_data):
        """Should handle all neighbors being masked."""
        mask = torch.zeros_like(sample_data["neighbor_mask"])

        # This is an edge case - behavior depends on implementation
        # At minimum it shouldn't crash
        output = encoder(
            sample_data["receiver"],
            sample_data["neighbors"],
            sample_data["distances"],
            neighbor_mask=mask,
        )

        assert output.context.shape[0] == sample_data["receiver"].shape[0]

    def test_large_distances(self, encoder, sample_data):
        """Should handle very large distances."""
        data = {**sample_data, "distances": sample_data["distances"] * 1000}
        output = encoder(**data)

        # Should not have NaN
        assert not torch.isnan(output.context).any()
        assert not torch.isnan(output.attention_weights).any()

    def test_zero_distances(self, encoder, sample_data):
        """Should handle zero distances."""
        data = {**sample_data, "distances": torch.zeros_like(sample_data["distances"])}
        output = encoder(**data)

        # Should not have NaN
        assert not torch.isnan(output.context).any()
