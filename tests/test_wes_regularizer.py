from __future__ import annotations

import pandas as pd
import torch

from stagebridge.transition_model.couplings import build_ot_coupling
from stagebridge.transition_model.schrodinger_bridge import edgewise_schrodinger_bridge_loss
from stagebridge.transition_model.stochastic_dynamics import EdgeWiseStochasticDynamics
from stagebridge.transition_model.wes_regularizer import lookup_wes_vectors, pairwise_wes_penalty


def test_lookup_wes_vectors_aligns_rows_and_zero_fills_missing_pairs() -> None:
    obs = pd.DataFrame(
        {
            "donor_id": ["P1", "P2", "P3"],
            "stage": ["AAH", "AIS", "MIA"],
        }
    )
    lookup = {
        ("P1", "AAH"): [1.0, 0.0],
        ("P2", "AIS"): [0.5, 0.5],
    }

    aligned = lookup_wes_vectors(obs, lookup)

    assert aligned.shape == (3, 2)
    assert torch.allclose(aligned[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(aligned[1], torch.tensor([0.5, 0.5]))
    assert torch.allclose(aligned[2], torch.zeros(2))


def test_pairwise_wes_penalty_changes_transport_coupling() -> None:
    x_src = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    x_tgt = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    src_wes = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    tgt_wes = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    base = build_ot_coupling(x_src, x_tgt, epsilon=0.05, n_iters=120)
    extra_cost = pairwise_wes_penalty(src_wes, tgt_wes, penalty_scale=1.0, normalize=True)
    regularized = build_ot_coupling(
        x_src,
        x_tgt,
        epsilon=0.05,
        n_iters=120,
        extra_cost=extra_cost,
    )

    assert extra_cost.shape == (2, 2)
    assert torch.all(extra_cost >= 0.0)
    assert regularized[0, 1] > base[0, 1]
    assert regularized[0, 0] < base[0, 0]


def test_wes_penalty_enters_bridge_loss_path() -> None:
    torch.manual_seed(41)
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
    x_src = torch.randn(6, 3)
    x_tgt = x_src + 0.2 * torch.randn(6, 3)
    context = torch.randn(1, 2)
    edge_ids = torch.zeros(6, dtype=torch.long)
    src_wes = torch.tensor([[0.0, 0.0]] * 3 + [[1.0, 1.0]] * 3, dtype=torch.float32)
    tgt_wes = torch.tensor([[1.0, 1.0]] * 3 + [[0.0, 0.0]] * 3, dtype=torch.float32)
    extra_cost = pairwise_wes_penalty(src_wes, tgt_wes, penalty_scale=1.0, normalize=True)

    base_loss, _, base_coupling = edgewise_schrodinger_bridge_loss(
        model,
        x_src=x_src,
        x_tgt=x_tgt,
        context=context,
        edge_ids=edge_ids,
        num_ot_pairs=12,
        sigma=0.1,
        diffusion_weight=0.2,
    )
    reg_loss, _, reg_coupling = edgewise_schrodinger_bridge_loss(
        model,
        x_src=x_src,
        x_tgt=x_tgt,
        context=context,
        edge_ids=edge_ids,
        num_ot_pairs=12,
        sigma=0.1,
        diffusion_weight=0.2,
        extra_cost=extra_cost,
    )

    assert torch.isfinite(base_loss)
    assert torch.isfinite(reg_loss)
    assert not torch.allclose(base_coupling, reg_coupling)
