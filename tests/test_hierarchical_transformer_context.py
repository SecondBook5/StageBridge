from __future__ import annotations

import torch

from stagebridge.context_model.hierarchical_transformer import TypedHierarchicalTransformerEncoder


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


def test_typed_hierarchical_transformer_emits_group_summaries_and_attention() -> None:
    encoder = TypedHierarchicalTransformerEncoder(
        input_dim=4,
        hidden_dim=32,
        num_heads=4,
        num_inducing_points=8,
        dropout=0.0,
        num_token_types=4,
        num_group_summary_tokens=3,
        num_fusion_queries=7,
        group_names=["epithelial", "stromal", "immune", "vascular"],
    )
    coords = torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [0.8, 0.9], [0.9, 0.8], [1.5, 1.4], [1.6, 1.5]],
        dtype=torch.float32,
    )
    confidence = torch.tensor([0.9, 0.85, 0.8, 0.75, 0.7, 0.65], dtype=torch.float32)
    out = encoder(
        _make_tokens(),
        token_coords=coords,
        token_confidence=confidence,
        dataset_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([1], dtype=torch.long),
        return_attention=True,
    )

    assert out.pooled_context.shape == (32,)
    assert out.context_tokens is not None
    assert out.context_tokens.shape == (7, 32)
    assert out.group_summary_tokens is not None
    assert out.group_summary_tokens.shape == (12, 32)
    assert out.relation_tokens is not None
    assert out.relation_tokens.shape == (6, 32)
    assert "fusion_query_attention" in out.attention_maps
    assert out.diagnostics["group_diagnostics"][0]["group_token_counts"]["epithelial"] == 2
    assert out.diagnostics["query_role_names"][:3] == ["source_stage", "target_stage", "transition"]


def test_dataset_and_edge_conditioning_change_hierarchical_context() -> None:
    encoder = TypedHierarchicalTransformerEncoder(
        input_dim=4,
        hidden_dim=32,
        num_heads=4,
        num_inducing_points=8,
        dropout=0.0,
        num_token_types=4,
        num_group_summary_tokens=3,
        num_fusion_queries=7,
        group_names=["epithelial", "stromal", "immune", "vascular"],
    )
    tokens = _make_tokens()
    coords = torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [0.8, 0.9], [0.9, 0.8], [1.5, 1.4], [1.6, 1.5]],
        dtype=torch.float32,
    )
    confidence = torch.ones(6, dtype=torch.float32)

    luad = encoder(tokens, token_coords=coords, token_confidence=confidence, dataset_ids=torch.tensor([0]), edge_ids=torch.tensor([1]))
    brain = encoder(tokens, token_coords=coords, token_confidence=confidence, dataset_ids=torch.tensor([1]), edge_ids=torch.tensor([1]))
    other_edge = encoder(tokens, token_coords=coords, token_confidence=confidence, dataset_ids=torch.tensor([0]), edge_ids=torch.tensor([2]))

    assert not torch.allclose(luad.pooled_context, brain.pooled_context)
    assert not torch.allclose(luad.pooled_context, other_edge.pooled_context)
