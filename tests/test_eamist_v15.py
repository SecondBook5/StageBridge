"""Tests for EA-MIST v1.5 upgrades: ordinal loss, distribution pooling, atlas contrast, monotonic reg."""

from __future__ import annotations

import torch

from stagebridge.context_model.lesion_set_transformer import EAMISTModel, NicheTransitionScoreHead
from stagebridge.context_model.losses import ordinal_stage_loss, transition_consistency_loss
from stagebridge.utils.types import LesionBagBatch


def _make_batch(batch_size: int = 2, num_instances: int = 4) -> LesionBagBatch:
    return LesionBagBatch(
        receiver_embeddings=torch.randn(batch_size, num_instances, 5),
        receiver_state_ids=torch.randint(0, 4, (batch_size, num_instances)),
        ring_compositions=torch.randn(batch_size, num_instances, 4, 6),
        hlca_features=torch.randn(batch_size, num_instances, 7),
        luca_features=torch.randn(batch_size, num_instances, 9),
        lr_pathway_summary=torch.randn(batch_size, num_instances, 8),
        neighborhood_stats=torch.randn(batch_size, num_instances, 6),
        flat_features=torch.randn(batch_size, num_instances, 59),
        center_coords=torch.randn(batch_size, num_instances, 2),
        neighborhood_mask=torch.ones(batch_size, num_instances, dtype=torch.bool),
        edge_ids=torch.ones(batch_size, dtype=torch.long),
        labels=torch.tensor([1.0, 0.0][:batch_size], dtype=torch.float32),
        label_weights=torch.ones(batch_size, dtype=torch.float32),
        stage_indices=torch.tensor([0, 4][:batch_size], dtype=torch.long),
        displacement_targets=torch.tensor([0.0, 1.0][:batch_size], dtype=torch.float32),
        edge_targets=None,
        edge_target_mask=None,
        sample_ids=[f"S{i}" for i in range(batch_size)],
        lesion_ids=[f"S{i}" for i in range(batch_size)],
        donor_ids=[f"P{i}" for i in range(batch_size)],
        patient_ids=[f"P{i}" for i in range(batch_size)],
        stages=["Normal", "LUAD"][:batch_size],
        label_sources=["synthetic"] * batch_size,
        edge_target_labels=(),
        evolution_features=torch.randn(batch_size, 3),
    )


def _make_model(**kwargs):
    defaults = dict(
        receiver_dim=5,
        sender_feature_dim=6,
        hlca_dim=7,
        luca_dim=9,
        lr_summary_dim=8,
        stats_dim=6,
        flat_feature_dim=59,
        num_receiver_states=5,
        num_rings=4,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        num_inducing_points=4,
        num_pma_seeds=1,
        dropout=0.0,
        local_encoder_type="transformer",
        use_prototypes=False,
        num_prototypes=4,
        evolution_dim=3,
        evolution_mode="gated",
        num_stage_classes=5,
        num_edge_heads=0,
        reference_feature_mode="hlca_luca",
    )
    defaults.update(kwargs)
    return EAMISTModel(**defaults)


# === Upgrade 1: Ordinal stage loss ===


def test_ordinal_stage_loss_perfect_prediction():
    """When logits perfectly predict the label, ordinal loss should be near zero."""
    logits = torch.zeros(3, 5)
    logits[0, 0] = 100.0
    logits[1, 2] = 100.0
    logits[2, 4] = 100.0
    labels = torch.tensor([0, 2, 4], dtype=torch.long)
    loss = ordinal_stage_loss(logits, labels)
    assert loss.item() < 0.01


def test_ordinal_stage_loss_far_prediction_penalized_more():
    """Predicting class 4 when true is class 0 should cost more than predicting class 1."""
    labels = torch.tensor([0], dtype=torch.long)
    logits_close = torch.zeros(1, 5)
    logits_close[0, 1] = 100.0
    logits_far = torch.zeros(1, 5)
    logits_far[0, 4] = 100.0
    loss_close = ordinal_stage_loss(logits_close, labels)
    loss_far = ordinal_stage_loss(logits_far, labels)
    assert loss_far.item() > loss_close.item()


def test_ordinal_stage_loss_handles_negative_labels():
    """Negative labels (invalid) should be skipped."""
    logits = torch.randn(4, 5)
    labels = torch.tensor([0, -1, 2, -1], dtype=torch.long)
    loss = ordinal_stage_loss(logits, labels)
    assert torch.isfinite(loss) and loss.item() >= 0


def test_ordinal_stage_loss_all_invalid():
    """If all labels are invalid, loss should be zero."""
    logits = torch.randn(3, 5)
    labels = torch.full((3,), -1, dtype=torch.long)
    loss = ordinal_stage_loss(logits, labels)
    assert loss.item() == 0.0


def test_ordinal_stage_loss_gradient_flows():
    """Gradient should flow through the ordinal loss to logits."""
    logits = torch.randn(4, 5, requires_grad=True)
    labels = torch.tensor([0, 1, 3, 4], dtype=torch.long)
    loss = ordinal_stage_loss(logits, labels)
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


# === Upgrade 2: Distribution-aware pooling ===


def test_distribution_summary_off_by_default():
    """With use_distribution_summary=False, niche_transition_scores should be None."""
    model = _make_model(use_distribution_summary=False)
    batch = _make_batch()
    output = model(batch)
    assert output.niche_transition_scores is None


def test_distribution_summary_shapes():
    """With use_distribution_summary=True, shapes should be correct."""
    model = _make_model(use_distribution_summary=True)
    batch = _make_batch(batch_size=2, num_instances=4)
    output = model(batch)
    assert output.niche_transition_scores is not None
    assert output.niche_transition_scores.shape == (2, 4)
    assert output.stage_logits.shape == (2, 5)
    assert output.displacement.shape == (2,)


def test_distribution_summary_respects_mask():
    """Masked niches should get -inf in transition scores."""
    model = _make_model(use_distribution_summary=True)
    batch = _make_batch(batch_size=2, num_instances=4)
    # Rebuild batch with custom mask
    from dataclasses import fields

    kwargs = {
        f.name: getattr(batch, f.name) for f in fields(batch) if f.name != "neighborhood_mask"
    }
    batch = LesionBagBatch(
        **kwargs,
        neighborhood_mask=torch.tensor([[True, True, True, True], [True, True, False, False]]),
    )
    output = model(batch)
    assert output.niche_transition_scores[1, 2].item() == float("-inf")
    assert output.niche_transition_scores[1, 3].item() == float("-inf")
    # Valid niches should be finite
    assert torch.isfinite(output.niche_transition_scores[0, 0])
    assert torch.isfinite(output.niche_transition_scores[1, 0])


def test_distribution_summary_gradient_flows():
    """Gradient should flow from stage loss through distribution summary features."""
    model = _make_model(use_distribution_summary=True)
    batch = _make_batch()
    output = model(batch)
    loss = output.stage_logits.sum()
    loss.backward()
    assert model.niche_transition_head is not None
    for p in model.niche_transition_head.parameters():
        if p.requires_grad:
            assert p.grad is not None


# === Upgrade 3: Atlas contrast token ===


def test_atlas_contrast_token_off_by_default():
    """Without use_atlas_contrast_token, the tokenizer should produce 9 tokens."""
    model = _make_model(use_atlas_contrast_token=False)
    batch = _make_batch()
    total = batch.receiver_embeddings.shape[0] * batch.receiver_embeddings.shape[1]
    tokens = model.local_encoder.tokenizer(
        receiver_embeddings=batch.receiver_embeddings.reshape(total, -1),
        receiver_state_ids=batch.receiver_state_ids.reshape(total),
        ring_compositions=batch.ring_compositions.reshape(total, 4, 6),
        hlca_features=batch.hlca_features.reshape(total, -1),
        luca_features=batch.luca_features.reshape(total, -1),
        lr_pathway_summary=batch.lr_pathway_summary.reshape(total, -1),
        neighborhood_stats=batch.neighborhood_stats.reshape(total, -1),
    )
    assert tokens.shape[1] == 9  # receiver + 4 rings + hlca + luca + lr + stats


def test_atlas_contrast_token_adds_10th_token():
    """With use_atlas_contrast_token, the tokenizer should produce 10 tokens."""
    model = _make_model(use_atlas_contrast_token=True)
    batch = _make_batch()
    total = batch.receiver_embeddings.shape[0] * batch.receiver_embeddings.shape[1]
    tokens = model.local_encoder.tokenizer(
        receiver_embeddings=batch.receiver_embeddings.reshape(total, -1),
        receiver_state_ids=batch.receiver_state_ids.reshape(total),
        ring_compositions=batch.ring_compositions.reshape(total, 4, 6),
        hlca_features=batch.hlca_features.reshape(total, -1),
        luca_features=batch.luca_features.reshape(total, -1),
        lr_pathway_summary=batch.lr_pathway_summary.reshape(total, -1),
        neighborhood_stats=batch.neighborhood_stats.reshape(total, -1),
    )
    assert tokens.shape[1] == 10


def test_atlas_contrast_full_forward():
    """Full model forward pass with atlas contrast token should work."""
    model = _make_model(use_atlas_contrast_token=True)
    batch = _make_batch()
    output = model(batch)
    assert output.stage_logits.shape == (2, 5)
    assert output.displacement.shape == (2,)


def test_atlas_contrast_gradient_flows():
    """Atlas contrast MLP should receive gradients."""
    model = _make_model(use_atlas_contrast_token=True)
    batch = _make_batch()
    output = model(batch)
    loss = output.stage_logits.sum()
    loss.backward()
    proj = model.local_encoder.tokenizer.atlas_contrast_proj
    assert proj is not None
    for p in proj.parameters():
        if p.requires_grad:
            assert p.grad is not None


# === Upgrade 4: Transition consistency loss ===


def test_transition_consistency_loss_basic():
    """Basic transition consistency loss computation."""
    displacement = torch.tensor([0.5, 0.8], dtype=torch.float32)
    scores = torch.tensor(
        [[0.3, 0.4, 0.6, float("-inf")], [0.7, 0.9, float("-inf"), float("-inf")]]
    )
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    loss = transition_consistency_loss(displacement, scores, mask)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_transition_consistency_loss_both_identical():
    """When displacement equals mean niche score, loss should be zero."""
    scores = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    displacement = torch.tensor([0.5, 1.0])
    loss = transition_consistency_loss(displacement, scores, mask)
    assert loss.item() < 1e-6


def test_transition_consistency_loss_no_nan():
    """Loss should never be NaN even with extreme inputs."""
    displacement = torch.tensor([0.0, 100.0, -100.0])
    scores = torch.tensor([[1e6, -1e6, 0], [0, 0, 0], [0, 0, 0]])
    mask = torch.ones(3, 3, dtype=torch.bool)
    loss = transition_consistency_loss(displacement, scores, mask)
    assert torch.isfinite(loss)


def test_transition_consistency_gradient_detaches_scores():
    """Gradient of tc loss should NOT flow through niche_transition_scores (they're detached)."""
    displacement = torch.tensor([0.5, 0.8], requires_grad=True)
    scores = torch.tensor([[0.3, 0.4], [0.7, 0.9]], requires_grad=True)
    mask = torch.ones(2, 2, dtype=torch.bool)
    loss = transition_consistency_loss(displacement, scores, mask)
    loss.backward()
    # displacement should have gradient
    assert displacement.grad is not None
    # scores should NOT because they are detached inside the loss
    assert scores.grad is None or (scores.grad.abs().sum().item() == 0.0)


# === Combined: all v1.5 upgrades together ===


def test_full_v15_model_forward():
    """Full EA-MIST v1.5 forward pass with all upgrades enabled."""
    model = _make_model(
        use_distribution_summary=True,
        use_atlas_contrast_token=True,
    )
    batch = _make_batch()
    output = model(batch)
    assert output.stage_logits.shape == (2, 5)
    assert output.displacement.shape == (2,)
    assert output.niche_transition_scores is not None
    assert output.niche_transition_scores.shape == (2, 4)


def test_full_v15_backward():
    """All v1.5 losses computed together should produce finite gradients."""
    model = _make_model(
        use_distribution_summary=True,
        use_atlas_contrast_token=True,
    )
    batch = _make_batch()
    output = model(batch)
    stage_loss = ordinal_stage_loss(output.stage_logits, batch.stage_indices)
    tc_loss = transition_consistency_loss(
        output.displacement, output.niche_transition_scores, batch.neighborhood_mask
    )
    total = stage_loss + 0.1 * tc_loss
    total.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.all(torch.isfinite(p.grad)), f"Non-finite gradient for {name}"


def test_v15_pipeline_smoke():
    """Smoke test: compute all v1.5 losses in a pipeline-like fashion."""
    from stagebridge.context_model.losses import (
        class_weighted_stage_loss,
        displacement_regression_loss,
    )

    model = _make_model(
        use_distribution_summary=True,
        use_atlas_contrast_token=True,
    )
    batch = _make_batch()
    output = model(batch)
    stage_ce = class_weighted_stage_loss(output.stage_logits, batch.stage_indices)
    ordinal = ordinal_stage_loss(output.stage_logits, batch.stage_indices)
    disp = displacement_regression_loss(output.displacement, batch.displacement_targets)
    tc = transition_consistency_loss(
        output.displacement, output.niche_transition_scores, batch.neighborhood_mask
    )
    total = 1.0 * stage_ce + 0.5 * ordinal + 0.5 * disp + 0.1 * tc
    total.backward()
    # All losses should be finite
    for name, val in [
        ("stage_ce", stage_ce),
        ("ordinal", ordinal),
        ("disp", disp),
        ("tc", tc),
        ("total", total),
    ]:
        assert torch.isfinite(val), f"{name} is not finite: {val}"
