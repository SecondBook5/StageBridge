from __future__ import annotations

import torch

from stagebridge.transition_model.stochastic_dynamics import EdgeWiseStochasticDynamics


def test_edgewise_stochastic_dynamics_sample_step_returns_shapes() -> None:
    torch.manual_seed(29)
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
    x_t = torch.randn(5, 4)
    context = torch.randn(1, 3)
    edge_ids = torch.zeros(5, dtype=torch.long)
    t = torch.full((5,), 0.5)

    x_next, drift, diffusion = model.sample_step(
        x_t,
        t=t,
        dt=0.1,
        context=context,
        edge_ids=edge_ids,
        stochastic=True,
    )

    assert x_next.shape == (5, 4)
    assert drift.shape == (5, 4)
    assert diffusion.shape == (5, 4)
    assert torch.all(diffusion > 0.0)


def test_edgewise_stochastic_rollout_differs_from_deterministic_update() -> None:
    torch.manual_seed(31)
    model = EdgeWiseStochasticDynamics(
        input_dim=3,
        context_dim=2,
        hidden_dim=16,
        time_dim=8,
        edge_dim=4,
        num_edges=4,
        dropout=0.0,
        min_diffusion_scale=1e-3,
        state_dependent_diffusion=True,
    )
    x0 = torch.zeros(6, 3)
    context = torch.randn(1, 2)
    edge_ids = torch.ones(6, dtype=torch.long)

    deterministic, _ = model.rollout(
        x0,
        context=context,
        edge_ids=edge_ids,
        num_steps=6,
        stochastic=False,
    )
    stochastic, _ = model.rollout(
        x0,
        context=context,
        edge_ids=edge_ids,
        num_steps=6,
        stochastic=True,
    )

    assert not torch.allclose(deterministic, stochastic)
