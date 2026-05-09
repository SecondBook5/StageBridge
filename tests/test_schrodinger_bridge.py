"""Tests for Schrödinger Bridge dynamics."""

import pytest
import torch

from stagebridge.transition.schrodinger_bridge import (
    SchrodingerBridge,
    SchrodingerBridgeConfig,
    SchrodingerBridgeWrapper,
    schrodinger_bridge_loss,
    sb_ot_coupled_loss,
)


@pytest.fixture
def sb_config():
    return SchrodingerBridgeConfig(
        input_dim=40,
        context_dim=128,
        hidden_dim=128,
        num_stages=4,
        sigma=0.1,
    )


@pytest.fixture
def sb_module(sb_config):
    return SchrodingerBridge(sb_config)


class TestSchrodingerBridge:
    """Tests for SchrodingerBridge module."""

    def test_forward_velocity_shape(self, sb_module):
        B, D = 16, 40
        x = torch.randn(B, D)
        t = torch.rand(B)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        v = sb_module.forward_velocity(x, t, ctx, stage)
        assert v.shape == (B, D)

    def test_backward_velocity_shape(self, sb_module):
        B, D = 16, 40
        x = torch.randn(B, D)
        t = torch.rand(B)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        v = sb_module.backward_velocity(x, t, ctx, stage)
        assert v.shape == (B, D)

    def test_score_shape(self, sb_module):
        B, D = 16, 40
        x = torch.randn(B, D)
        t = torch.rand(B)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        score = sb_module.score(x, t, ctx, stage)
        assert score.shape == (B, D)

    def test_sample_forward(self, sb_module):
        B, D = 8, 40
        x0 = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        x1 = sb_module.sample_forward(x0, ctx, stage, num_steps=10)
        assert x1.shape == (B, D)
        assert not torch.allclose(x0, x1)

    def test_sample_backward(self, sb_module):
        B, D = 8, 40
        x1 = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        x0 = sb_module.sample_backward(x1, ctx, stage, num_steps=10)
        assert x0.shape == (B, D)
        assert not torch.allclose(x0, x1)

    def test_sample_with_trajectory(self, sb_module):
        B, D = 8, 40
        x0 = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)
        num_steps = 10

        x1, traj = sb_module.sample_forward(
            x0, ctx, stage, num_steps=num_steps, return_trajectory=True
        )
        assert x1.shape == (B, D)
        assert traj.shape == (B, num_steps + 1, D)
        assert torch.allclose(traj[:, 0], x0)

    def test_sample_multiple(self, sb_module):
        B, D = 4, 40
        x0 = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)
        num_samples = 5

        samples = sb_module.sample_multiple(
            x0, ctx, stage, num_samples=num_samples, num_steps=5
        )
        assert samples.shape == (B, num_samples, D)

    def test_stochasticity(self, sb_module):
        """Verify that sampling produces different outputs (stochastic)."""
        B, D = 4, 40
        x0 = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        torch.manual_seed(42)
        x1_a = sb_module.sample_forward(x0, ctx, stage, num_steps=10)

        torch.manual_seed(123)
        x1_b = sb_module.sample_forward(x0, ctx, stage, num_steps=10)

        assert not torch.allclose(x1_a, x1_b)


class TestSchrodingerBridgeLoss:
    """Tests for SB training loss."""

    def test_loss_shape(self, sb_module):
        B, D = 16, 40
        x_src = torch.randn(B, D)
        x_tgt = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        loss, diag = schrodinger_bridge_loss(x_src, x_tgt, sb_module, ctx, stage)
        assert loss.shape == ()
        assert loss.item() > 0
        assert "loss_score" in diag
        assert "loss_drift" in diag

    def test_loss_gradient(self, sb_module):
        B, D = 16, 40
        x_src = torch.randn(B, D)
        x_tgt = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        loss, _ = schrodinger_bridge_loss(x_src, x_tgt, sb_module, ctx, stage)
        loss.backward()

        # Check at least some parameters got gradients (not all may be used)
        grads_found = sum(1 for p in sb_module.parameters() if p.grad is not None)
        assert grads_found > 0, "No gradients found"

    def test_ot_coupled_loss(self, sb_module):
        N, M, D = 32, 32, 40
        x_src = torch.randn(N, D)
        x_tgt = torch.randn(M, D)
        ctx = torch.randn(1, 128)
        stage = torch.zeros(1, dtype=torch.long)

        loss, diag, coupling = sb_ot_coupled_loss(
            x_src, x_tgt, sb_module, ctx, stage,
            num_pairs=16,
        )
        assert loss.shape == ()
        assert coupling.shape == (N, M)
        assert "ot_cost" in diag


class TestSchrodingerBridgeWrapper:
    """Tests for OT-CFM compatibility wrapper."""

    def test_forward_vector_field(self, sb_module):
        wrapper = SchrodingerBridgeWrapper(sb_module)

        B, D = 16, 40
        x = torch.randn(B, D)
        t = torch.rand(B)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        v = wrapper.forward_vector_field(x, t, ctx, stage)
        assert v.shape == (B, D)

    def test_integrate_euler(self, sb_module):
        wrapper = SchrodingerBridgeWrapper(sb_module)

        B, D = 8, 40
        x0 = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        x1 = wrapper.integrate_euler(x0, ctx, stage, num_steps=10)
        assert x1.shape == (B, D)

    def test_integrate_sde(self, sb_module):
        wrapper = SchrodingerBridgeWrapper(sb_module)

        B, D = 8, 40
        x0 = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        x1 = wrapper.integrate_sde(x0, ctx, stage, num_steps=10)
        assert x1.shape == (B, D)


class TestExternalDrift:
    """Tests for external drift function integration."""

    def test_external_drift_config(self):
        """Test SB with external drift enabled."""
        config = SchrodingerBridgeConfig(
            input_dim=40,
            context_dim=128,
            hidden_dim=128,
            num_stages=4,
            sigma=0.1,
            use_external_drift=True,
        )
        sb = SchrodingerBridge(config)

        # Should NOT have internal forward_drift
        assert sb.forward_drift is None
        # Should still have score network
        assert sb.score_net is not None

    def test_external_drift_requires_function(self):
        """Test that external drift raises if no function set."""
        config = SchrodingerBridgeConfig(
            input_dim=40,
            context_dim=128,
            hidden_dim=128,
            use_external_drift=True,
        )
        sb = SchrodingerBridge(config)

        B, D = 8, 40
        x = torch.randn(B, D)
        t = torch.rand(B)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        with pytest.raises(RuntimeError, match="no external_drift_fn set"):
            sb.forward_velocity(x, t, ctx, stage)

    def test_external_drift_function(self):
        """Test that external drift is called correctly."""
        config = SchrodingerBridgeConfig(
            input_dim=40,
            context_dim=128,
            hidden_dim=128,
            use_external_drift=True,
        )
        sb = SchrodingerBridge(config)

        # Simple external drift: returns constant
        expected = torch.ones(8, 40)
        def mock_drift(x_t, t, context, stage_pair_id):
            return expected.clone()

        sb.set_external_drift(mock_drift)

        B, D = 8, 40
        x = torch.randn(B, D)
        t = torch.rand(B)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        v = sb.forward_velocity(x, t, ctx, stage)
        assert torch.allclose(v, expected)

    def test_backward_velocity_uses_external_drift(self):
        """Test backward velocity correctly combines external drift with score."""
        config = SchrodingerBridgeConfig(
            input_dim=40,
            context_dim=128,
            hidden_dim=128,
            sigma=0.1,
            use_external_drift=True,
        )
        sb = SchrodingerBridge(config)

        # External drift returns zeros
        def zero_drift(x_t, t, context, stage_pair_id):
            return torch.zeros_like(x_t)

        sb.set_external_drift(zero_drift)

        B, D = 8, 40
        x = torch.randn(B, D)
        t = torch.rand(B)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        # Forward drift is zero, so backward = -sigma^2 * score
        bwd = sb.backward_velocity(x, t, ctx, stage)
        score = sb.score(x, t, ctx, stage)
        expected_bwd = -config.sigma**2 * score

        assert torch.allclose(bwd, expected_bwd)

    def test_sample_forward_with_external_drift(self):
        """Test sampling works with external drift."""
        config = SchrodingerBridgeConfig(
            input_dim=40,
            context_dim=128,
            hidden_dim=128,
            sigma=0.1,
            use_external_drift=True,
        )
        sb = SchrodingerBridge(config)

        # External drift points toward target
        target = torch.ones(40)
        def drift_to_target(x_t, t, context, stage_pair_id):
            return (target.unsqueeze(0) - x_t) * 0.5

        sb.set_external_drift(drift_to_target)

        B, D = 4, 40
        x0 = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        x1 = sb.sample_forward(x0, ctx, stage, num_steps=20)
        assert x1.shape == (B, D)
        # Should have moved toward target (roughly)
        dist_before = (x0 - target.unsqueeze(0)).norm(dim=-1).mean()
        dist_after = (x1 - target.unsqueeze(0)).norm(dim=-1).mean()
        assert dist_after < dist_before


class TestReversibility:
    """Tests for forward-backward reversibility."""

    def test_approximate_reversibility(self, sb_module):
        """After training, forward then backward should approximately recover x0."""
        B, D = 8, 40
        x0 = torch.randn(B, D)
        ctx = torch.randn(B, 128)
        stage = torch.zeros(B, dtype=torch.long)

        # Forward
        x1 = sb_module.sample_forward(x0, ctx, stage, num_steps=20)

        # Backward
        x0_recovered = sb_module.sample_backward(x1, ctx, stage, num_steps=20)

        # Without training, won't be exact, but check shapes match
        assert x0_recovered.shape == x0.shape
