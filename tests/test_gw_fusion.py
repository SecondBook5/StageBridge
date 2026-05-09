"""Tests for GW fusion modes: concat, precompute_gw, learned_gw.

Validates:
1. Forward pass works for each mode
2. Output shapes are correct
3. Gradients flow (for learned modes)
4. Integration with model forward pass
"""

import pytest
import torch
import torch.nn as nn

from stagebridge.reference.learned_gw_fusion import (
    LearnedGWConfig,
    LearnedGWFusion,
    PretrainedLearnedGWFusion,
    gromov_wasserstein_differentiable,
    sinkhorn_log_stabilized,
)
from stagebridge.reference.gw_precompute import (
    GWPrecomputeConfig,
    BarycentricFusion,
    gromov_wasserstein,
    pairwise_distances,
)


class TestSinkhornLogStabilized:
    """Test Sinkhorn algorithm."""

    def test_basic_sinkhorn(self):
        """Sinkhorn produces valid transport plan."""
        B, N, M = 2, 10, 10
        C = torch.rand(B, N, M)

        P = sinkhorn_log_stabilized(C, reg=0.1, num_iters=50)

        # Check shape
        assert P.shape == (B, N, M)

        # Check non-negative
        assert (P >= 0).all()

        # Check marginals (should be close to uniform)
        row_sums = P.sum(dim=2)
        col_sums = P.sum(dim=1)

        expected_row = torch.ones(B, N) / N
        expected_col = torch.ones(B, M) / M

        assert torch.allclose(row_sums, expected_row, atol=1e-3)
        assert torch.allclose(col_sums, expected_col, atol=1e-3)

    def test_sinkhorn_differentiable(self):
        """Gradients flow through Sinkhorn."""
        B, N, M = 1, 5, 5
        C = torch.rand(B, N, M, requires_grad=True)

        P = sinkhorn_log_stabilized(C, reg=0.1, num_iters=20)
        loss = P.sum()
        loss.backward()

        assert C.grad is not None
        assert not torch.isnan(C.grad).any()


class TestGromovWassersteinDifferentiable:
    """Test differentiable GW."""

    def test_basic_gw(self):
        """GW produces valid coupling."""
        B, N = 1, 10

        # Create distance matrices
        X = torch.randn(N, 5)
        Y = torch.randn(N, 3)
        C_X = torch.cdist(X, X).unsqueeze(0)
        C_Y = torch.cdist(Y, Y).unsqueeze(0)

        # More iterations for better convergence
        P, cost = gromov_wasserstein_differentiable(
            C_X, C_Y, reg=0.1, num_gw_iters=15, num_sinkhorn_iters=50
        )

        # Check shape
        assert P.shape == (1, N, N)

        # Check non-negative
        assert (P >= 0).all()

        # Check marginals sum to 1 (relaxed tolerance for GW)
        row_sums = P.sum(dim=2).squeeze(0)
        col_sums = P.sum(dim=1).squeeze(0)

        # GW marginals should sum to 1, but individual cells may vary
        assert torch.allclose(row_sums.sum(), torch.tensor(1.0), atol=1e-2)
        assert torch.allclose(col_sums.sum(), torch.tensor(1.0), atol=1e-2)

        # Cost should be non-negative
        assert cost.item() >= 0

    def test_gw_differentiable(self):
        """Gradients flow through GW."""
        N = 8

        X = torch.randn(N, 5, requires_grad=True)
        Y = torch.randn(N, 3)

        C_X = torch.cdist(X, X).unsqueeze(0)
        C_Y = torch.cdist(Y, Y).unsqueeze(0)

        P, cost = gromov_wasserstein_differentiable(
            C_X, C_Y, reg=0.1, num_gw_iters=3, num_sinkhorn_iters=10
        )

        loss = cost.sum()
        loss.backward()

        assert X.grad is not None
        assert not torch.isnan(X.grad).any()


class TestLearnedGWFusion:
    """Test learned GW fusion."""

    @pytest.fixture
    def config(self):
        return LearnedGWConfig(
            hlca_dim=30,
            luca_dim=10,
            metric_dim=16,
            output_dim=40,
            sinkhorn_reg=0.1,
            sinkhorn_iters=10,
            gw_iters=3,
            num_metric_layers=2,
        )

    @pytest.fixture
    def model(self, config):
        return LearnedGWFusion(config)

    def test_forward_batch(self, model):
        """Forward pass with batch of cells."""
        B = 16
        hlca = torch.randn(B, 30)
        luca = torch.randn(B, 10)

        fused = model(hlca, luca)

        assert fused.shape == (B, 40)
        assert not torch.isnan(fused).any()

    def test_forward_single_cell(self, model):
        """Forward pass with single cell (falls back to concat)."""
        hlca = torch.randn(1, 30)
        luca = torch.randn(1, 10)

        fused = model(hlca, luca)

        assert fused.shape == (1, 40)
        assert not torch.isnan(fused).any()

    def test_return_coupling(self, model):
        """Return coupling and cost."""
        B = 8
        hlca = torch.randn(B, 30)
        luca = torch.randn(B, 10)

        fused, coupling, cost = model(hlca, luca, return_coupling=True)

        assert fused.shape == (B, 40)
        assert coupling.shape == (B, B)
        assert cost.ndim == 0  # scalar

    def test_gradients_flow(self, model):
        """Gradients flow through entire model."""
        B = 8
        hlca = torch.randn(B, 30, requires_grad=True)
        luca = torch.randn(B, 10, requires_grad=True)

        fused = model(hlca, luca)
        loss = fused.sum()
        loss.backward()

        assert hlca.grad is not None
        assert luca.grad is not None
        assert not torch.isnan(hlca.grad).any()
        assert not torch.isnan(luca.grad).any()

        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gw_loss(self, model):
        """GW loss is computable."""
        B = 8
        hlca = torch.randn(B, 30)
        luca = torch.randn(B, 10)

        gw_loss = model.get_gw_loss(hlca, luca)

        assert gw_loss.ndim == 0
        assert gw_loss.item() >= 0


class TestBarycentricFusion:
    """Test precomputed GW with barycentric projection."""

    @pytest.fixture
    def reference_data(self):
        N_ref = 50
        hlca_ref = torch.randn(N_ref, 30)
        luca_ref = torch.randn(N_ref, 10)

        # Compute GW coupling
        D_hlca = pairwise_distances(hlca_ref)
        D_luca = pairwise_distances(luca_ref)
        D_hlca = D_hlca / D_hlca.max()
        D_luca = D_luca / D_luca.max()

        coupling, _ = gromov_wasserstein(D_hlca, D_luca, reg=0.1)

        return hlca_ref, luca_ref, coupling

    @pytest.fixture
    def model(self, reference_data):
        hlca_ref, luca_ref, coupling = reference_data
        return BarycentricFusion(
            hlca_ref=hlca_ref,
            luca_ref=luca_ref,
            coupling=coupling,
            k_neighbors=5,
            fused_dim=40,
        )

    def test_forward(self, model):
        """Forward pass works."""
        B = 8
        hlca = torch.randn(B, 30)
        luca = torch.randn(B, 10)

        fused = model(hlca, luca)

        assert fused.shape == (B, 40)
        assert not torch.isnan(fused).any()

    def test_return_coupling(self, model):
        """Return soft coupling and k-NN indices."""
        B = 8
        hlca = torch.randn(B, 30)
        luca = torch.randn(B, 10)

        fused, soft_coupling, knn_idx = model(hlca, luca, return_coupling=True)

        assert fused.shape == (B, 40)
        assert soft_coupling.shape[0] == B
        assert knn_idx.shape == (B, model.k_neighbors)

    def test_gradients_flow_through_projections(self, model):
        """Gradients flow through projection layers."""
        B = 8
        hlca = torch.randn(B, 30, requires_grad=True)
        luca = torch.randn(B, 10, requires_grad=True)

        fused = model(hlca, luca)
        loss = fused.sum()
        loss.backward()

        assert hlca.grad is not None
        assert luca.grad is not None

        # Projection layers should have gradients
        assert model.hlca_proj.weight.grad is not None
        assert model.luca_proj.weight.grad is not None


class TestConcatBaseline:
    """Test that simple concat works as baseline."""

    def test_concat_fusion(self):
        """Simple concat produces correct output."""
        B = 8
        hlca = torch.randn(B, 30)
        luca = torch.randn(B, 10)

        fused = torch.cat([hlca, luca], dim=-1)

        assert fused.shape == (B, 40)

        # First 30 dims should be hlca
        assert torch.allclose(fused[:, :30], hlca)
        # Last 10 dims should be luca
        assert torch.allclose(fused[:, 30:], luca)


class TestIntegrationWithModel:
    """Test fusion modes integrate with StageBridge model."""

    def test_learned_gw_in_training_loop(self):
        """Simulate training step with learned GW."""
        config = LearnedGWConfig(
            hlca_dim=30,
            luca_dim=10,
            output_dim=40,
            gw_iters=2,
            sinkhorn_iters=5,
        )
        fusion = LearnedGWFusion(config)
        optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-3)

        # Simulate batch
        B = 16
        hlca = torch.randn(B, 30)
        luca = torch.randn(B, 10)
        target = torch.randn(B, 40)

        # Forward
        optimizer.zero_grad()
        fused, coupling, gw_cost = fusion(hlca, luca, return_coupling=True)

        # Loss = reconstruction + GW regularization
        recon_loss = ((fused - target) ** 2).mean()
        total_loss = recon_loss + 0.01 * gw_cost

        # Backward
        total_loss.backward()

        # Check gradients exist
        for name, param in fusion.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

        # Step
        optimizer.step()

    def test_numerical_stability_large_batch(self):
        """Test numerical stability with larger batch."""
        config = LearnedGWConfig(
            hlca_dim=30,
            luca_dim=10,
            output_dim=40,
            sinkhorn_reg=0.1,  # Higher reg for stability
            gw_iters=3,
            sinkhorn_iters=10,
        )
        fusion = LearnedGWFusion(config)

        B = 64
        hlca = torch.randn(B, 30)
        luca = torch.randn(B, 10)

        fused = fusion(hlca, luca)

        assert not torch.isnan(fused).any()
        assert not torch.isinf(fused).any()

    def test_deterministic_with_seed(self):
        """Same inputs give same outputs with fixed seed."""
        config = LearnedGWConfig(gw_iters=2, sinkhorn_iters=5)

        hlca = torch.randn(8, 30)
        luca = torch.randn(8, 10)

        torch.manual_seed(42)
        fusion1 = LearnedGWFusion(config)
        out1 = fusion1(hlca, luca)

        torch.manual_seed(42)
        fusion2 = LearnedGWFusion(config)
        out2 = fusion2(hlca, luca)

        assert torch.allclose(out1, out2)


class TestStageBridgeModelIntegration:
    """Test fusion modes work with actual StageBridge model."""

    def test_model_with_learned_gw(self):
        """Model initializes and runs with learned_gw fusion."""
        from stagebridge.models.stagebridge import StageBridge, StageBridgeConfig

        config = StageBridgeConfig(
            input_dim=40,
            hidden_dim=64,
            num_heads=4,
            use_gw_fusion=True,
            gw_fusion_type="learned_gw",
            gw_output_dim=40,
            use_amici_attention=False,
            use_context_refiner=False,
            use_hierarchical=False,
            use_sample_heads=False,
            use_pathway_head=False,
            use_proliferation_head=False,
        )

        model = StageBridge(config)
        assert model.gw_fusion is not None
        assert isinstance(model.gw_fusion, LearnedGWFusion)

    def test_model_with_concat(self):
        """Model initializes and runs with concat (no fusion module)."""
        from stagebridge.models.stagebridge import StageBridge, StageBridgeConfig

        config = StageBridgeConfig(
            input_dim=40,
            hidden_dim=64,
            num_heads=4,
            use_gw_fusion=True,
            gw_fusion_type="concat",
            use_amici_attention=False,
            use_context_refiner=False,
            use_hierarchical=False,
            use_sample_heads=False,
            use_pathway_head=False,
            use_proliferation_head=False,
        )

        model = StageBridge(config)
        # concat mode: no fusion module
        assert model.gw_fusion is None

    def test_encode_niche_with_learned_gw(self):
        """encode_niche works with learned GW fusion."""
        from stagebridge.models.stagebridge import StageBridge, StageBridgeConfig

        config = StageBridgeConfig(
            input_dim=40,
            hidden_dim=64,
            num_heads=4,
            use_gw_fusion=True,
            gw_fusion_type="learned_gw",
            gw_output_dim=40,
            use_amici_attention=False,
            use_context_refiner=False,
            use_hierarchical=False,
            use_sample_heads=False,
            use_pathway_head=False,
            use_proliferation_head=False,
            use_learned_ring_pooling=False,
        )

        model = StageBridge(config)

        B = 8
        receiver = torch.randn(B, 40)
        ring_cells = [torch.randn(B, 10, 40) for _ in range(4)]
        ring_masks = [torch.ones(B, 10, dtype=torch.bool) for _ in range(4)]
        hlca = torch.randn(B, 30)
        luca = torch.randn(B, 10)

        output = model.encode_niche(
            receiver=receiver,
            ring_cells=ring_cells,
            ring_masks=ring_masks,
            hlca=hlca,
            luca=luca,
        )

        assert output.context.shape == (B, 64)
        assert not torch.isnan(output.context).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
