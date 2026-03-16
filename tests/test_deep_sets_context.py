"""Mission 3 tests for the Deep Sets context encoder."""

from __future__ import annotations

import torch

from stagebridge.context_model.set_encoder import DeepSetsContextEncoder


def test_deep_sets_context_forward_pass() -> None:
    tokens = torch.tensor(
        [
            [0.6, 0.2, 0.1, 0.1],
            [0.5, 0.2, 0.2, 0.1],
            [0.2, 0.2, 0.4, 0.2],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    encoder = DeepSetsContextEncoder(input_dim=4, hidden_dim=32, dropout=0.0)
    out = encoder(tokens)
    assert out.pooled_context.shape == (32,)
    assert out.token_embeddings.shape == (4, 32)
    assert torch.isfinite(out.pooled_context).all()
