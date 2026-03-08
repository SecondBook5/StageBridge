from __future__ import annotations

import torch

from stagebridge.transition_model.drift_network import EdgeConditionedDriftMLP


def test_edge_conditioned_drift_mlp_returns_batch_shaped_output() -> None:
    torch.manual_seed(7)
    model = EdgeConditionedDriftMLP(
        input_dim=6,
        context_dim=4,
        hidden_dim=32,
        time_dim=8,
        edge_dim=4,
        num_edges=4,
        dropout=0.0,
    )
    x_t = torch.randn(5, 6)
    t = torch.linspace(0.1, 0.9, steps=5)
    context = torch.randn(1, 4)
    edge_ids = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)

    drift = model(x_t=x_t, t=t, context=context, edge_ids=edge_ids)

    assert drift.shape == (5, 6)
    assert torch.isfinite(drift).all()


def test_edge_conditioning_changes_drift_prediction() -> None:
    torch.manual_seed(11)
    model = EdgeConditionedDriftMLP(
        input_dim=4,
        context_dim=3,
        hidden_dim=16,
        time_dim=8,
        edge_dim=4,
        num_edges=4,
        dropout=0.0,
    )
    x_t = torch.randn(3, 4)
    t = torch.full((3,), 0.5)
    context = torch.randn(3, 3)
    edge_zero = torch.zeros(3, dtype=torch.long)
    edge_one = torch.ones(3, dtype=torch.long)

    drift_zero = model(x_t=x_t, t=t, context=context, edge_ids=edge_zero)
    drift_one = model(x_t=x_t, t=t, context=context, edge_ids=edge_one)

    assert not torch.allclose(drift_zero, drift_one)
