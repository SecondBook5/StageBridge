from __future__ import annotations

import torch

from stagebridge.transition_model.schrodinger_bridge import edgewise_schrodinger_bridge_loss
from stagebridge.transition_model.stochastic_dynamics import EdgeWiseStochasticDynamics


def test_edgewise_schrodinger_bridge_loss_returns_finite_loss_and_coupling() -> None:
    torch.manual_seed(37)
    model = EdgeWiseStochasticDynamics(
        input_dim=4,
        context_dim=3,
        hidden_dim=24,
        time_dim=8,
        edge_dim=4,
        num_edges=4,
        dropout=0.0,
        min_diffusion_scale=1e-3,
        state_dependent_diffusion=True,
    )
    x_src = torch.randn(10, 4)
    x_tgt = x_src + 0.5 * torch.randn(10, 4)
    context = torch.randn(1, 3)
    edge_ids = torch.zeros(10, dtype=torch.long)

    loss, diagnostics, coupling = edgewise_schrodinger_bridge_loss(
        model,
        x_src=x_src,
        x_tgt=x_tgt,
        context=context,
        edge_ids=edge_ids,
        epsilon=0.1,
        sinkhorn_iters=80,
        num_ot_pairs=16,
        sigma=0.1,
        diffusion_weight=0.2,
    )

    assert torch.isfinite(loss)
    assert coupling.shape == (10, 10)
    assert diagnostics["loss_total"] >= 0.0
    assert diagnostics["loss_drift"] >= 0.0
    assert diagnostics["loss_diffusion"] >= 0.0
