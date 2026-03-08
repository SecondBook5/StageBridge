from __future__ import annotations

import torch

from stagebridge.transition_model.diffusion_network import StateDependentDiffusionNetwork


def test_state_dependent_diffusion_network_returns_positive_scales() -> None:
    torch.manual_seed(13)
    model = StateDependentDiffusionNetwork(
        input_dim=5,
        context_dim=4,
        hidden_dim=32,
        time_dim=8,
        edge_dim=4,
        num_edges=4,
        dropout=0.0,
        min_scale=1e-3,
        state_dependent=True,
    )
    x_t = torch.randn(6, 5)
    t = torch.linspace(0.1, 0.9, steps=6)
    context = torch.randn(1, 4)
    edge_ids = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)

    scales = model(x_t=x_t, t=t, context=context, edge_ids=edge_ids)

    assert scales.shape == (6, 5)
    assert torch.all(scales > 0.0)


def test_state_dependent_diffusion_changes_with_state() -> None:
    torch.manual_seed(23)
    model = StateDependentDiffusionNetwork(
        input_dim=3,
        context_dim=2,
        hidden_dim=16,
        time_dim=8,
        edge_dim=4,
        num_edges=4,
        dropout=0.0,
        min_scale=1e-3,
        state_dependent=True,
    )
    t = torch.full((2,), 0.5)
    context = torch.randn(2, 2)
    edge_ids = torch.zeros(2, dtype=torch.long)
    x_low = torch.zeros(2, 3)
    x_high = torch.ones(2, 3)

    scales_low = model(x_t=x_low, t=t, context=context, edge_ids=edge_ids)
    scales_high = model(x_t=x_high, t=t, context=context, edge_ids=edge_ids)

    assert not torch.allclose(scales_low, scales_high)
