from __future__ import annotations

import torch

from stagebridge.context_model.set_encoder import PooledContextEncoder


def test_pooled_context_encoder_returns_hidden_dim_context() -> None:
    encoder = PooledContextEncoder(input_dim=4, hidden_dim=16)
    tokens = torch.rand(12, 4)

    summary = encoder(tokens)

    assert summary.pooled_context.shape == (16,)
    assert summary.token_embeddings.shape == (12, 4)
    assert torch.isfinite(summary.pooled_context).all()
