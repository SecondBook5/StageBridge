"""Tests for the composite CCRT objective."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.contracts import CCRTShapeError, CCRTValidationError
from stagebridge.ccrt.operators import (
    ContextResidualTransportConfig,
    ContextResidualTransportOperator,
)
from stagebridge.ccrt.representations import SemanticGeometryConfig
from stagebridge.ccrt.sender_context.sparsity import (
    attention_entropy_loss,
    value_l1_sparsity_loss,
)
from stagebridge.ccrt.training import (
    CCRTTrainingBatch,
    CompositeCCRTObjective,
    CompositeCCRTObjectiveConfig,
)
from stagebridge.ccrt.transport import (
    SemanticTransportLoss,
    SemanticTransportLossConfig,
    SinkhornConfig,
)

B, K, D_R, D_S, D_Z = 2, 2, 3, 4, 2
REG, DRIFT, GROWTH, HIDDEN, HEADS = 2, 2, 1, 8, 2
N_TYPES, EMPTY_TYPE, N_EDGES = 3, 2, 1


def make_model():
    torch.manual_seed(0)
    return ContextResidualTransportOperator(
        ContextResidualTransportConfig(
            receiver_dim=D_R, sender_dim=D_S, hidden_dim=HIDDEN, num_heads=HEADS,
            num_sender_context_types=N_TYPES, empty_sender_context_type_id=EMPTY_TYPE,
            regulatory_dim=REG, drift_dim=DRIFT, growth_dim=GROWTH,
            num_transition_edges=N_EDGES,
        )
    )


def make_objective(**cfg):
    stl = SemanticTransportLoss(
        geometry=SemanticGeometryConfig(),
        native_sinkhorn=SinkhornConfig(epsilon=0.1, max_iterations=100),
        loss=SemanticTransportLossConfig(),
    )
    return CompositeCCRTObjective(
        semantic_transport_loss=stl, config=CompositeCCRTObjectiveConfig(**cfg)
    )


def make_batch(dtype=torch.float64, growth=False, semantic_dim=D_Z):
    torch.manual_seed(1)
    kwargs = dict(
        receiver_features=torch.randn(B, D_R, dtype=dtype),
        sender_features=torch.randn(B, K, D_S, dtype=dtype),
        sender_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        distance_to_receiver=torch.rand(B, K, dtype=dtype),
        sender_context_type_ids=torch.tensor([[0, 1], [0, 0]], dtype=torch.int64),
        transition_edge_index=torch.zeros(B, dtype=torch.int64),
        source_semantic_features=torch.randn(B, semantic_dim, dtype=dtype),
        target_semantic_features=torch.randn(3, semantic_dim, dtype=dtype),
    )
    if growth:
        kwargs["growth_targets"] = torch.randn(B, GROWTH, dtype=dtype)
    return CCRTTrainingBatch(**kwargs)


def _double(model):
    return model.to(dtype=torch.float64)


def test_forward_output_fields_all_scalar():
    model = _double(make_model())
    obj = make_objective().to(dtype=torch.float64)
    out = obj(model=model, batch=make_batch())
    for name in (
        "total_loss", "semantic_loss", "attention_entropy_loss",
        "sender_effect_l1_loss", "regulatory_l1_loss", "residual_drift_l2_loss",
        "residual_growth_l2_loss", "growth_supervision_loss",
    ):
        t = getattr(out, name)
        assert t.dim() == 0 and bool(torch.isfinite(t)), name


def test_total_arithmetic_exact():
    model = _double(make_model())
    obj = make_objective(
        semantic_weight=1.0, attention_entropy_weight=0.5, sender_effect_l1_weight=0.3,
        regulatory_l1_weight=0.2, residual_drift_l2_weight=0.1,
        residual_growth_l2_weight=0.4,
    ).to(dtype=torch.float64)
    out = obj(model=model, batch=make_batch())
    expected = (
        1.0 * out.semantic_loss + 0.5 * out.attention_entropy_loss
        + 0.3 * out.sender_effect_l1_loss + 0.2 * out.regulatory_l1_loss
        + 0.1 * out.residual_drift_l2_loss + 0.4 * out.residual_growth_l2_loss
    )
    assert torch.allclose(out.total_loss, expected, atol=1e-10)


def test_zero_weight_regularizers_return_finite_zero():
    model = _double(make_model())
    obj = make_objective().to(dtype=torch.float64)  # only semantic active
    out = obj(model=model, batch=make_batch())
    for name in (
        "attention_entropy_loss", "sender_effect_l1_loss", "regulatory_l1_loss",
        "residual_drift_l2_loss", "residual_growth_l2_loss", "growth_supervision_loss",
    ):
        t = getattr(out, name)
        assert float(t) == 0.0 and bool(torch.isfinite(t)), name


def test_semantic_drift_dimension_mismatch_fails():
    model = _double(make_model())  # drift_dim = 2
    obj = make_objective().to(dtype=torch.float64)
    bad = make_batch(semantic_dim=5)  # semantic dim 5 != drift dim 2
    with pytest.raises(CCRTShapeError):
        obj(model=model, batch=bad)


def test_entropy_and_l1_match_existing_functions():
    model = _double(make_model())
    obj = make_objective(attention_entropy_weight=1.0, sender_effect_l1_weight=1.0).to(
        dtype=torch.float64
    )
    out = obj(model=model, batch=make_batch())
    mo = out.model_output
    expected_entropy = attention_entropy_loss(
        mo.attention_weights, sender_mask=mo.sender_mask_with_empty
    )
    expected_l1 = value_l1_sparsity_loss(
        mo.sender_effects, sender_mask=mo.sender_mask_with_empty
    )
    assert torch.allclose(out.attention_entropy_loss, expected_entropy, atol=1e-10)
    assert torch.allclose(out.sender_effect_l1_loss, expected_l1, atol=1e-10)


def test_regulatory_and_residual_penalties_exact():
    model = _double(make_model())
    obj = make_objective(
        regulatory_l1_weight=1.0, residual_drift_l2_weight=1.0,
        residual_growth_l2_weight=1.0,
    ).to(dtype=torch.float64)
    out = obj(model=model, batch=make_batch())
    mo = out.model_output
    assert torch.allclose(out.regulatory_l1_loss, mo.regulatory_state.abs().mean(), atol=1e-10)
    assert torch.allclose(out.residual_drift_l2_loss, (mo.residual_drift ** 2).mean(), atol=1e-10)
    assert torch.allclose(out.residual_growth_l2_loss, (mo.residual_growth ** 2).mean(), atol=1e-10)


def test_mse_growth_supervision_exact():
    model = _double(make_model())
    obj = make_objective(growth_supervision_weight=1.0, growth_loss="mse").to(
        dtype=torch.float64
    )
    batch = make_batch(growth=True)
    out = obj(model=model, batch=batch)
    expected = ((out.model_output.full_growth - batch.growth_targets) ** 2).mean()
    assert torch.allclose(out.growth_supervision_loss, expected, atol=1e-10)


def test_huber_growth_supervision_finite():
    model = _double(make_model())
    obj = make_objective(growth_supervision_weight=1.0, growth_loss="huber").to(
        dtype=torch.float64
    )
    out = obj(model=model, batch=make_batch(growth=True))
    assert bool(torch.isfinite(out.growth_supervision_loss))


def test_growth_target_required_when_weighted():
    model = _double(make_model())
    obj = make_objective(growth_supervision_weight=1.0).to(dtype=torch.float64)
    with pytest.raises(CCRTValidationError):
        obj(model=model, batch=make_batch(growth=False))


def test_growth_shape_mismatch_fails():
    model = _double(make_model())
    obj = make_objective(growth_supervision_weight=1.0).to(dtype=torch.float64)
    batch = make_batch(growth=True)
    bad = CCRTTrainingBatch(
        receiver_features=batch.receiver_features,
        sender_features=batch.sender_features,
        sender_mask=batch.sender_mask,
        distance_to_receiver=batch.distance_to_receiver,
        sender_context_type_ids=batch.sender_context_type_ids,
        transition_edge_index=batch.transition_edge_index,
        source_semantic_features=batch.source_semantic_features,
        target_semantic_features=batch.target_semantic_features,
        growth_targets=torch.randn(B, GROWTH + 2, dtype=torch.float64),  # wrong G
    )
    with pytest.raises(CCRTShapeError):
        obj(model=model, batch=bad)


def test_empty_growth_mask_fails():
    model = _double(make_model())
    obj = make_objective(growth_supervision_weight=1.0).to(dtype=torch.float64)
    batch = make_batch(growth=True)
    masked = CCRTTrainingBatch(
        receiver_features=batch.receiver_features,
        sender_features=batch.sender_features,
        sender_mask=batch.sender_mask,
        distance_to_receiver=batch.distance_to_receiver,
        sender_context_type_ids=batch.sender_context_type_ids,
        transition_edge_index=batch.transition_edge_index,
        source_semantic_features=batch.source_semantic_features,
        target_semantic_features=batch.target_semantic_features,
        growth_targets=batch.growth_targets,
        growth_mask=torch.zeros(B, GROWTH, dtype=torch.bool),  # no valid entries
    )
    with pytest.raises(CCRTValidationError):
        obj(model=model, batch=masked)


def test_gradients_reach_core_groups():
    model = _double(make_model())
    obj = make_objective(growth_supervision_weight=1.0).to(dtype=torch.float64)
    out = obj(model=model, batch=make_batch(growth=True))
    out.total_loss.backward()
    assert model.attention.query_proj.weight.grad is not None
    assert any(m.weight.grad is not None for m in model.regulatory_bottleneck.mlp if hasattr(m, "weight"))
    assert model.drift_head.regulatory_map.weight.grad is not None
    # growth head gets gradient because growth supervision is active
    assert model.growth_head.regulatory_map.weight.grad is not None


def test_config_validation():
    with pytest.raises(ValueError):
        CompositeCCRTObjectiveConfig(semantic_weight=0.0)
    with pytest.raises(ValueError):
        CompositeCCRTObjectiveConfig(attention_entropy_weight=-1.0)
    with pytest.raises(ValueError):
        CompositeCCRTObjectiveConfig(growth_loss="l1")
    with pytest.raises(ValueError):
        CompositeCCRTObjectiveConfig(growth_huber_delta=0.0)
