from __future__ import annotations

import torch

from stagebridge.context_model.set_encoder import DeepSetsTransformerHybridEncoder


def _make_tokens() -> torch.Tensor:
    return torch.tensor(
        [
            [0.8, 0.1, 0.1, 0.0],
            [0.7, 0.2, 0.1, 0.0],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.0, 0.1, 0.1, 0.8],
            [0.0, 0.1, 0.2, 0.7],
        ],
        dtype=torch.float32,
    )


def test_hybrid_encoder_emits_backbone_and_transformer_diagnostics() -> None:
    encoder = DeepSetsTransformerHybridEncoder(
        input_dim=4,
        hidden_dim=32,
        num_heads=4,
        num_inducing_points=8,
        num_seed_vectors=3,
        dropout=0.0,
        num_token_types=4,
    )
    coords = torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [0.8, 0.9], [0.9, 0.8], [1.5, 1.4], [1.6, 1.5]],
        dtype=torch.float32,
    )
    confidence = torch.tensor([0.9, 0.85, 0.8, 0.75, 0.7, 0.65], dtype=torch.float32)
    token_type_ids = torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.long)

    out = encoder(
        _make_tokens(),
        token_type_ids=token_type_ids,
        token_coords=coords,
        token_confidence=confidence,
        return_attention=True,
    )

    assert out.pooled_context.shape == (32,)
    assert out.context_tokens is not None
    assert out.context_tokens.shape == (4, 32)
    assert "pma_seed_attention" in out.attention_maps
    assert 0.0 <= out.diagnostics["hybrid_gate_mean"] <= 1.0
    assert out.diagnostics["deep_sets_context_norm"] > 0.0
    assert out.diagnostics["transformer_refinement_norm"] > 0.0


def test_hybrid_encoder_responds_to_spatial_layout() -> None:
    encoder = DeepSetsTransformerHybridEncoder(
        input_dim=4,
        hidden_dim=32,
        num_heads=4,
        num_inducing_points=8,
        num_seed_vectors=3,
        dropout=0.0,
        num_token_types=4,
    )
    tokens = _make_tokens()
    token_type_ids = torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.long)
    confidence = torch.ones(6, dtype=torch.float32)
    coords_a = torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [0.8, 0.9], [0.9, 0.8], [1.5, 1.4], [1.6, 1.5]],
        dtype=torch.float32,
    )
    coords_b = torch.flip(coords_a, dims=[0])

    out_a = encoder(tokens, token_type_ids=token_type_ids, token_coords=coords_a, token_confidence=confidence)
    out_b = encoder(tokens, token_type_ids=token_type_ids, token_coords=coords_b, token_confidence=confidence)

    assert not torch.allclose(out_a.pooled_context, out_b.pooled_context)
