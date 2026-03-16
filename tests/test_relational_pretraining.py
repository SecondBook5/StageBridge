from __future__ import annotations

import torch

from stagebridge.context_model.hierarchical_transformer import TypedHierarchicalTransformerEncoder
from stagebridge.transition_model.relational_pretraining import (
    RelationalPretrainingConfig,
    compute_relational_auxiliary_losses,
    pretrain_relational_transformer,
    stratified_mask_token_indices,
)


def _tokens() -> torch.Tensor:
    return torch.tensor(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0],
            [0.1, 0.2, 0.7, 0.0],
            [0.0, 0.1, 0.2, 0.7],
            [0.0, 0.0, 0.2, 0.8],
        ],
        dtype=torch.float32,
    )


def _coords() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [0.8, 0.7], [0.9, 0.8], [1.6, 1.5], [1.7, 1.6]],
        dtype=torch.float32,
    )


def _confidence() -> torch.Tensor:
    return torch.tensor([0.95, 0.9, 0.85, 0.8, 0.75, 0.7], dtype=torch.float32)


def _token_type_ids() -> torch.Tensor:
    return torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.long)


def test_stratified_mask_token_indices_is_deterministic() -> None:
    token_type_ids = _token_type_ids()
    first = stratified_mask_token_indices(token_type_ids, mask_fraction=0.34, seed=13)
    second = stratified_mask_token_indices(token_type_ids, mask_fraction=0.34, seed=13)

    assert torch.equal(first, second)
    assert first.numel() >= 1


def test_relational_auxiliary_losses_include_provider_and_transfer_terms() -> None:
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
    config = RelationalPretrainingConfig(max_epochs=1, steps_per_epoch=1, seed=7)
    dataset_ids = torch.tensor([0], dtype=torch.long)
    edge_ids = torch.tensor([1], dtype=torch.long)
    provider_views = [
        {
            "method": "tacco",
            "tokens": _tokens() * 0.97,
            "coords": _coords(),
            "confidence": _confidence(),
            "token_type_ids": _token_type_ids(),
            "dataset_ids": dataset_ids.clone(),
        }
    ]
    negative_controls = [
        {
            "label": "dataset_id_mismatch",
            "tokens": torch.flip(_tokens(), dims=[0]),
            "coords": torch.flip(_coords(), dims=[0]),
            "confidence": torch.flip(_confidence(), dims=[0]),
            "token_type_ids": torch.flip(_token_type_ids(), dims=[0]),
            "dataset_ids": torch.tensor([1], dtype=torch.long),
        }
    ]

    pretraining = pretrain_relational_transformer(
        context_encoder=encoder,
        context_tokens=_tokens(),
        token_type_ids=_token_type_ids(),
        token_coords=_coords(),
        token_confidence=_confidence(),
        dataset_ids=dataset_ids,
        edge_ids=edge_ids,
        negative_controls=negative_controls,
        provider_views=provider_views,
        config=config,
    )

    assert pretraining["history"]
    assert pretraining["encoder_parameter_delta"] > 0.0
    assert pretraining["metrics"]["provider_views_used"] == 1
    assert "dataset_id_mismatch" in pretraining["metrics"]["negative_control_scores"]
    assert pretraining["metrics"]["masked_token_count"] > 0

    total, losses, metrics, summary = compute_relational_auxiliary_losses(
        context_encoder=encoder,
        heads=pretraining["heads"],
        context_tokens=_tokens(),
        token_type_ids=_token_type_ids(),
        token_coords=_coords(),
        token_confidence=_confidence(),
        dataset_ids=dataset_ids,
        edge_ids=edge_ids,
        negative_controls=negative_controls,
        provider_views=provider_views,
        config=config,
        seed=7,
        include_masked_token=True,
        include_provider_consistency=True,
        include_coordinate_corruption=True,
        include_group_relation=True,
        return_attention=True,
    )

    assert float(total.item()) >= 0.0
    assert set(losses) >= {
        "masked_token",
        "ranking",
        "provider_consistency",
        "coordinate_corruption",
        "group_relation",
    }
    assert summary.group_summary_tokens is not None
    assert metrics["provider_views_used"] == 1
