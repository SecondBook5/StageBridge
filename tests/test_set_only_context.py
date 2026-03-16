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


def test_set_only_context_responds_to_spatial_coordinates_and_confidence() -> None:
    tokens = torch.tensor(
        [
            [0.8, 0.1, 0.1, 0.0],
            [0.7, 0.2, 0.1, 0.0],
            [0.1, 0.7, 0.1, 0.1],
            [0.0, 0.1, 0.1, 0.8],
        ],
        dtype=torch.float32,
    )
    coords_a = torch.tensor(
        [[0.0, 0.0], [0.1, 0.2], [1.0, 1.0], [1.2, 0.9]],
        dtype=torch.float32,
    )
    coords_b = torch.tensor(
        [[1.2, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.0]],
        dtype=torch.float32,
    )
    confidence = torch.tensor([0.9, 0.8, 0.6, 0.2], dtype=torch.float32)
    encoder = TypedSetContextEncoder(
        input_dim=4,
        hidden_dim=32,
        num_heads=4,
        num_inducing_points=8,
        dropout=0.0,
        num_token_types=4,
        use_spatial_rpe=True,
        token_dropout_rate=0.0,
        use_confidence_gate=True,
    )
    encoder.eval()

    out_a = encoder(
        tokens, token_coords=coords_a, token_confidence=confidence, return_attention=True
    )
    out_b = encoder(
        tokens, token_coords=coords_b, token_confidence=confidence, return_attention=True
    )
    out_low_conf = encoder(
        tokens,
        token_coords=coords_a,
        token_confidence=torch.zeros_like(confidence),
        return_attention=True,
    )

    assert not torch.allclose(out_a.pooled_context, out_b.pooled_context)
    assert not torch.allclose(out_a.pooled_context, out_low_conf.pooled_context)
    assert out_a.token_coords is not None
    assert out_a.token_confidence is not None
    assert out_a.diagnostics["confidence_gate_mean"] > 0.0


def test_set_only_token_dropout_is_train_only() -> None:
    tokens = torch.tensor(
        [
            [0.8, 0.1, 0.1, 0.0],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.0, 0.1, 0.1, 0.8],
        ],
        dtype=torch.float32,
    )
    coords = torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        dtype=torch.float32,
    )
    confidence = torch.ones(4, dtype=torch.float32)
    encoder = TypedSetContextEncoder(
        input_dim=4,
        hidden_dim=32,
        num_heads=4,
        num_inducing_points=8,
        dropout=0.0,
        num_token_types=4,
        token_dropout_rate=0.5,
    )

    encoder.train()
    torch.manual_seed(7)
    train_out = encoder(tokens, token_coords=coords, token_confidence=confidence)

    encoder.eval()
    torch.manual_seed(7)
    eval_out = encoder(tokens, token_coords=coords, token_confidence=confidence)

    assert not torch.allclose(train_out.pooled_context, eval_out.pooled_context)
