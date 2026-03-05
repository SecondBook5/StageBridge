"""Tests for cross-attention drift transformer."""
import torch

from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.training.losses import flow_matching_loss
from stagebridge.utils.types import StageBatch, StageBridgeConfig


def _xattn_cfg() -> StageBridgeConfig:
    return StageBridgeConfig(
        input_dim=8,
        hidden_dim=32,
        vector_field_hidden_dim=64,
        num_heads=4,
        num_inducing_points=8,
        num_seed_vectors=4,   # 4 context tokens for cross-attention
        max_epochs=5,
        use_cross_attn_drift=True,
    )


def test_cross_attn_drift_forward_shapes():
    """Cross-attention drift model produces correct output shapes."""
    torch.manual_seed(0)
    cfg = _xattn_cfg()
    model = StageBridgeModel(cfg)
    model.eval()

    x_set = torch.randn(32, 8)
    x_t = torch.randn(16, 8)
    t = torch.rand(16)

    with torch.no_grad():
        c_s = model.forward_set_context(x_set)
        # Cross-attn mode returns token sequence (1, k, D)
        assert c_s.ndim == 3, f"Expected 3D context tokens, got shape {c_s.shape}"
        assert c_s.shape[1] == 4, f"Expected 4 context tokens, got {c_s.shape[1]}"

        sp = model.encode_stage_pair_tensor(0, 1, n=16, device=x_t.device)
        v = model.forward_vector_field(x_t, t, c_s, sp)
        assert v.shape == (16, 8), f"Velocity shape mismatch: {v.shape}"


def test_cross_attn_drift_euler_integration():
    """Euler integration works with cross-attention drift."""
    torch.manual_seed(1)
    cfg = _xattn_cfg()
    model = StageBridgeModel(cfg)
    model.eval()

    x0 = torch.randn(16, 8)
    x_set = torch.randn(32, 8)
    with torch.no_grad():
        c_s = model.forward_set_context(x_set)
        sp = model.encode_stage_pair_tensor(0, 1, n=16, device=x0.device)
        x_pred = model.integrate_euler(x0, c_s, sp, num_steps=4)

    assert x_pred.shape == (16, 8)
    assert x_pred.isfinite().all(), "Integration produced non-finite values"


def test_cross_attn_drift_em_sigma_zero_matches_euler():
    """integrate_euler_maruyama(sigma=0) must be bitwise-identical to integrate_euler."""
    torch.manual_seed(42)
    cfg = _xattn_cfg()
    model = StageBridgeModel(cfg)
    model.eval()

    x0 = torch.randn(16, 8)
    x_set = torch.randn(32, 8)
    with torch.no_grad():
        c_s = model.forward_set_context(x_set)
        sp = model.encode_stage_pair_tensor(0, 1, n=16, device=x0.device)
        x_euler = model.integrate_euler(x0, c_s, sp, num_steps=8)
        x_em = model.integrate_euler_maruyama(x0, c_s, sp, num_steps=8, sigma=0.0)

    assert torch.allclose(x_euler, x_em, atol=0.0, rtol=0.0), (
        "integrate_euler_maruyama(sigma=0) must match integrate_euler exactly"
    )


def test_cross_attn_drift_loss_decreases():
    """Cross-attention drift model loss decreases on a simple toy problem."""
    torch.manual_seed(7)
    cfg = _xattn_cfg()
    model = StageBridgeModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    x_src = torch.randn(128, 8)
    x_tgt = x_src + 0.5
    batch = StageBatch(x_src=x_src, x_tgt=x_tgt, stage_src=0, stage_tgt=1, donor_id="D1")

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss, _, _ = flow_matching_loss(
            batch=batch,
            model=model,
            ot_epsilon=0.1,
            sinkhorn_iters=20,
            num_ot_pairs=128,
            context_consistency_weight=0.1,
            use_ot=True,
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_mlp_and_xattn_are_independent():
    """MLP-drift and cross-attn-drift models have different parameter counts."""
    cfg_mlp = StageBridgeConfig(
        input_dim=8, hidden_dim=32, vector_field_hidden_dim=64,
        num_heads=4, num_inducing_points=8, num_seed_vectors=1,
        use_cross_attn_drift=False,
    )
    cfg_xattn = StageBridgeConfig(
        input_dim=8, hidden_dim=32, vector_field_hidden_dim=64,
        num_heads=4, num_inducing_points=8, num_seed_vectors=4,
        use_cross_attn_drift=True,
    )
    n_mlp = sum(p.numel() for p in StageBridgeModel(cfg_mlp).parameters())
    n_xattn = sum(p.numel() for p in StageBridgeModel(cfg_xattn).parameters())
    assert n_mlp != n_xattn, "MLP and cross-attn models should have different param counts"
