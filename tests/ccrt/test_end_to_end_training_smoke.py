"""End-to-end CPU smoke training for CCRT (native transport only)."""

from __future__ import annotations

import torch

from stagebridge.ccrt.data import CCRTIndexRegistry
from stagebridge.ccrt.grammar import (
    GROWTH_MASS,
    SEMANTIC_DRIFT,
    BiologicalSystemSpec,
    ReceiverBehavior,
    ReceiverState,
    SenderContextType,
    TransitionEdge,
)
from stagebridge.ccrt.operators import (
    ContextResidualTransportConfig,
    ContextResidualTransportOperator,
)
from stagebridge.ccrt.representations import SemanticGeometryConfig
from stagebridge.ccrt.training import (
    CCRTTrainer,
    CCRTTrainingBatch,
    CompositeCCRTObjective,
    CompositeCCRTObjectiveConfig,
    OptimizerConfig,
    TrainerConfig,
    build_checkpoint_state,
    build_optimizer,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from stagebridge.ccrt.transport import (
    SemanticTransportLoss,
    SemanticTransportLossConfig,
    SinkhornConfig,
)

# Small dims per the milestone.
B, K, D_R, D_S, D_Z = 2, 2, 3, 4, 2
REG, DRIFT, GROWTH, HIDDEN, HEADS = 2, 2, 1, 8, 2


def make_spec():
    return BiologicalSystemSpec(
        biological_system_id="sysA",
        receiver_states=(ReceiverState("s0"), ReceiverState("s1")),
        transition_edges=(TransitionEdge("e0", "s0", "s1"),),
        sender_context_types=(SenderContextType("c0"), SenderContextType("c1")),
        receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT), ReceiverBehavior(GROWTH_MASS)),
    )


def make_registry():
    return CCRTIndexRegistry.from_system_specs([make_spec()])


def make_model(registry):
    return ContextResidualTransportOperator(
        ContextResidualTransportConfig(
            receiver_dim=D_R, sender_dim=D_S, hidden_dim=HIDDEN, num_heads=HEADS,
            num_sender_context_types=registry.num_sender_context_types,
            empty_sender_context_type_id=registry.empty_sender_context_type_index,
            regulatory_dim=REG, drift_dim=DRIFT, growth_dim=GROWTH,
            num_transition_edges=registry.num_transition_edges,
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


def make_batch(dtype=torch.float64, growth=False):
    torch.manual_seed(7)
    source = torch.randn(B, D_Z, dtype=dtype)
    # target = source shifted by a fixed nonzero displacement -> learnable signal
    target = source + torch.tensor([2.0, -1.5], dtype=dtype)
    kwargs = dict(
        receiver_features=torch.randn(B, D_R, dtype=dtype),
        sender_features=torch.randn(B, K, D_S, dtype=dtype),
        sender_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        distance_to_receiver=torch.rand(B, K, dtype=dtype),
        sender_context_type_ids=torch.tensor([[0, 1], [0, 0]], dtype=torch.int64),
        transition_edge_index=torch.zeros(B, dtype=torch.int64),
        source_semantic_features=source,
        target_semantic_features=target,
    )
    if growth:
        kwargs["growth_targets"] = torch.randn(B, GROWTH, dtype=dtype)
    return CCRTTrainingBatch(**kwargs)


def build(growth=False):
    reg = make_registry()
    model = make_model(reg)
    obj_kwargs = {"growth_supervision_weight": 1.0} if growth else {}
    objective = make_objective(**obj_kwargs)
    optimizer = build_optimizer(model.parameters(), OptimizerConfig(learning_rate=1e-2))
    trainer = CCRTTrainer(
        model=model, objective=objective, optimizer=optimizer,
        config=TrainerConfig(epochs=1, dtype="float64", seed=0),
    )
    return trainer


def test_initial_objective_finite_and_params_change():
    trainer = build()
    batch = make_batch()
    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = trainer.train_step(batch)
    assert torch.isfinite(torch.tensor(metrics.total_loss))
    after = list(trainer.model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_all_major_groups_receive_gradients():
    trainer = build()
    trainer.model.train()
    trainer.objective.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    out = trainer.objective(model=trainer.model, batch=make_batch())
    out.total_loss.backward()
    assert trainer.model.attention.query_proj.weight.grad is not None
    assert any(
        m.weight.grad is not None
        for m in trainer.model.regulatory_bottleneck.mlp
        if hasattr(m, "weight")
    )
    assert trainer.model.drift_head.regulatory_map.weight.grad is not None


def test_growth_head_receives_gradient_when_supervised():
    trainer = build(growth=True)
    trainer.model.train()
    trainer.objective.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    out = trainer.objective(model=trainer.model, batch=make_batch(growth=True))
    out.total_loss.backward()
    assert trainer.model.growth_head.regulatory_map.weight.grad is not None


def test_checkpoint_reproduces_model_output(tmp_path):
    trainer = build()
    batch = make_batch()
    trainer.train_step(batch)
    trainer.model.eval()
    with torch.no_grad():
        ref = trainer.model(
            receiver_features=batch.receiver_features,
            sender_features=batch.sender_features,
            sender_mask=batch.sender_mask,
            distance_to_receiver=batch.distance_to_receiver,
            sender_context_type_ids=batch.sender_context_type_ids,
            transition_edge_index=batch.transition_edge_index,
        ).full_drift

    state = build_checkpoint_state(
        model=trainer.model, optimizer=trainer.optimizer, epoch=0,
        global_step=trainer.global_step,
    )
    path = tmp_path / "smoke.pt"
    save_checkpoint(path, state)

    reg = make_registry()
    fresh = make_model(reg).to(dtype=torch.float64)
    restore_checkpoint(state=load_checkpoint(path), model=fresh)
    fresh.eval()
    with torch.no_grad():
        got = fresh(
            receiver_features=batch.receiver_features,
            sender_features=batch.sender_features,
            sender_mask=batch.sender_mask,
            distance_to_receiver=batch.distance_to_receiver,
            sender_context_type_ids=batch.sender_context_type_ids,
            transition_edge_index=batch.transition_edge_index,
        ).full_drift
    assert torch.allclose(ref, got, atol=1e-8)


def test_tiny_deterministic_overfit_improves():
    trainer = build()
    batch = make_batch()
    initial = trainer.evaluate_step(batch).total_loss
    best = initial
    for _ in range(25):
        m = trainer.train_step(batch)
        best = min(best, m.total_loss)
    # best post-update loss must be below the initial loss (not necessarily final)
    assert best < initial
