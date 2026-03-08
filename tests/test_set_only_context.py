"""Mission 3 tests for the set-only context encoder."""
from __future__ import annotations

import torch

from stagebridge.context_model.set_encoder import TypedSetContextEncoder


def test_set_only_context_forward_pass() -> None:
    tokens = torch.tensor(
        [
            [0.6, 0.2, 0.1, 0.1],
            [0.5, 0.2, 0.2, 0.1],
            [0.2, 0.2, 0.4, 0.2],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    encoder = TypedSetContextEncoder(
        input_dim=4,
        hidden_dim=32,
        num_heads=4,
        num_inducing_points=8,
        dropout=0.0,
    )
    out = encoder(tokens)
    assert out.pooled_context.shape == (32,)
    assert out.token_embeddings.shape == (4, 32)
    assert torch.isfinite(out.pooled_context).all()


def test_set_only_context_can_emit_attention_and_token_types() -> None:
    tokens = torch.tensor(
        [
            [0.8, 0.1, 0.1, 0.0],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ],
        dtype=torch.float32,
    )
    encoder = TypedSetContextEncoder(
        input_dim=4,
        hidden_dim=32,
        num_heads=4,
        num_inducing_points=8,
        dropout=0.0,
        num_token_types=4,
    )
    out = encoder(tokens, return_attention=True)

    assert out.pooled_context.shape == (32,)
    assert out.token_embeddings.shape == (4, 32)
    assert out.token_type_ids is not None
    assert out.token_type_ids.shape == (4,)
    assert set(out.attention_maps) == {
        "isab_inducing_to_tokens",
        "isab_tokens_to_inducing",
        "sab_self_attention",
        "pma_seed_attention",
    }
    assert out.attention_maps["pma_seed_attention"].shape[-1] == 4
