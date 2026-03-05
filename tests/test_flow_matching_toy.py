import torch

from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.training.losses import flow_matching_loss
from stagebridge.utils.types import StageBatch, StageBridgeConfig


def _make_cfg() -> StageBridgeConfig:
    return StageBridgeConfig(
        input_dim=8,
        hidden_dim=32,
        vector_field_hidden_dim=64,
        num_heads=4,
        num_inducing_points=8,
        max_epochs=5,
    )


def test_flow_matching_loss_decreases_on_toy_problem():
    torch.manual_seed(7)

    cfg = _make_cfg()
    model = StageBridgeModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    x_src = torch.randn(128, 8)
    x_tgt = x_src + 0.5  # simple translation
    batch = StageBatch(x_src=x_src, x_tgt=x_tgt, stage_src=0, stage_tgt=1, donor_id="D1")

    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss, _, _ = flow_matching_loss(
            batch=batch,
            model=model,
            ot_epsilon=0.1,
            sinkhorn_iters=50,
            num_ot_pairs=128,
            context_consistency_weight=0.0,
            use_ot=True,
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))

    assert losses[-1] < losses[0]


def test_sb_cfm_sigma_zero_matches_euler():
    """integrate_euler_maruyama with sigma=0 must be bitwise-identical to integrate_euler."""
    torch.manual_seed(42)
    cfg = _make_cfg()
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


def test_sb_cfm_brownian_bridge_adds_noise():
    """With sigma > 0 the SB interpolant produces stochastic x_t (not equal to deterministic)."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    model = StageBridgeModel(cfg)

    x_src = torch.randn(64, 8)
    x_tgt = x_src + 1.0
    batch = StageBatch(x_src=x_src, x_tgt=x_tgt, stage_src=0, stage_tgt=1, donor_id="D1")

    # deterministic path (sigma=0)
    torch.manual_seed(5)
    _, diag0, _ = flow_matching_loss(
        batch=batch, model=model, num_ot_pairs=64, context_consistency_weight=0.0,
        use_ot=False, sigma=0.0,
    )

    # stochastic path (sigma=0.1): should still produce a valid scalar loss
    torch.manual_seed(5)
    loss_sb, diag_sb, _ = flow_matching_loss(
        batch=batch, model=model, num_ot_pairs=64, context_consistency_weight=0.0,
        use_ot=False, sigma=0.1,
    )

    assert loss_sb.isfinite(), "SB-CFM loss must be finite"
    # losses will differ because x_t is noisy — not equal to deterministic path
    assert diag0["loss_fm"] != diag_sb["loss_fm"], (
        "sigma=0.1 path should produce different loss than sigma=0 path"
    )
