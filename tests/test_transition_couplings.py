from __future__ import annotations

import pytest
import torch

from stagebridge.transition_model.couplings import (
    build_cost_matrix,
    build_ot_coupling,
    build_sinkhorn_coupling_from_cost,
)


def test_build_sinkhorn_coupling_preserves_uniform_marginals() -> None:
    x_src = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    x_tgt = torch.tensor(
        [
            [0.1, 0.0],
            [0.9, 0.2],
            [0.0, 1.1],
            [0.7, 0.8],
        ],
        dtype=torch.float32,
    )

    coupling = build_ot_coupling(x_src, x_tgt, epsilon=0.1, n_iters=120)

    assert coupling.shape == (3, 4)
    assert torch.all(coupling >= 0.0)
    assert torch.isclose(coupling.sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(
        coupling.sum(dim=1),
        torch.full((3,), 1.0 / 3.0),
        atol=5e-4,
    )
    assert torch.allclose(
        coupling.sum(dim=0),
        torch.full((4,), 1.0 / 4.0),
        atol=5e-4,
    )


def test_build_cost_matrix_rejects_mismatched_extra_cost_shape() -> None:
    x_src = torch.randn(2, 3)
    x_tgt = torch.randn(4, 3)

    with pytest.raises(ValueError, match="extra_cost shape"):
        build_cost_matrix(x_src, x_tgt, extra_cost=torch.zeros(2, 2))


def test_extra_cost_changes_transport_preference() -> None:
    x_src = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    x_tgt = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    base_cost = build_cost_matrix(x_src, x_tgt)
    unbiased = build_sinkhorn_coupling_from_cost(base_cost, epsilon=0.05, n_iters=120)

    extra_cost = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    biased = build_ot_coupling(x_src, x_tgt, epsilon=0.05, n_iters=120, extra_cost=extra_cost)

    assert unbiased[0, 0] > unbiased[0, 1]
    assert biased[0, 0] < unbiased[0, 0]
    assert biased[0, 1] > unbiased[0, 1]
