"""Tests for the CCRTTrainer."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.contracts import CCRTValidationError
from stagebridge.ccrt.training import (
    CCRTTrainer,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    build_optimizer,
    build_scheduler,
)

from test_composite_objective import make_batch, make_model, make_objective


def build_trainer(*, scheduler=False, epochs=1, clip=1.0, obj_kwargs=None):
    model = make_model()
    objective = make_objective(**(obj_kwargs or {}))
    optimizer = build_optimizer(model.parameters(), OptimizerConfig(learning_rate=1e-2))
    sched = None
    if scheduler:
        sched = build_scheduler(optimizer, SchedulerConfig(name="cosine", t_max=5))
    cfg = TrainerConfig(
        epochs=epochs, dtype="float64", gradient_clip_norm=clip, seed=0
    )
    return CCRTTrainer(
        model=model, objective=objective, optimizer=optimizer, config=cfg,
        scheduler=sched,
    )


def test_train_step_finite_and_increments_global_step():
    trainer = build_trainer()
    metrics = trainer.train_step(make_batch())
    assert trainer.global_step == 1
    for f in ("total_loss", "semantic_loss", "learning_rate"):
        assert torch.isfinite(torch.tensor(getattr(metrics, f)))
    assert metrics.gradient_norm is not None


def test_parameters_change_after_train_step():
    trainer = build_trainer()
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.train_step(make_batch())
    after = list(trainer.model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_evaluate_step_does_not_change_parameters():
    trainer = build_trainer()
    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = trainer.evaluate_step(make_batch())
    after = list(trainer.model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))
    assert metrics.gradient_norm is None
    assert trainer.global_step == 0


def test_gradient_clipping_finite():
    trainer = build_trainer(clip=0.5)
    metrics = trainer.train_step(make_batch())
    assert metrics.gradient_norm is not None
    assert torch.isfinite(torch.tensor(metrics.gradient_norm))


def test_fit_returns_epoch_count_and_scheduler_steps():
    trainer = build_trainer(scheduler=True, epochs=3)
    lr_before = trainer.optimizer.param_groups[0]["lr"]
    history = trainer.fit(train_batches=[make_batch(), make_batch()])
    assert len(history) == 3
    assert tuple(h.epoch for h in history) == (0, 1, 2)
    # scheduler stepped 3 times -> lr changed
    assert trainer.optimizer.param_groups[0]["lr"] != lr_before


def test_fit_metric_aggregation():
    trainer = build_trainer(epochs=1)
    b1, b2 = make_batch(), make_batch()
    history = trainer.fit(train_batches=[b1, b2])
    assert "total_loss" in history[0].train
    assert history[0].validation is None


def test_fit_with_validation():
    trainer = build_trainer(epochs=1)
    history = trainer.fit(
        train_batches=[make_batch()], validation_batches=[make_batch()]
    )
    assert history[0].validation is not None
    assert "total_loss" in history[0].validation


def test_empty_train_sequence_fails():
    trainer = build_trainer()
    with pytest.raises(CCRTValidationError):
        trainer.fit(train_batches=[])


def test_empty_validation_sequence_fails():
    trainer = build_trainer()
    with pytest.raises(CCRTValidationError):
        trainer.fit(train_batches=[make_batch()], validation_batches=[])


def test_nan_loss_detection_fails():
    trainer = build_trainer()
    batch = make_batch()
    # corrupt the source semantic features to force non-finite? Instead poison via
    # a NaN in receiver features after construction (bypass validate by using .to()
    # semantics). Simpler: monkeypatch objective to return NaN.
    bad = torch.tensor(float("nan"), dtype=torch.float64)

    class _NanObjective:
        def train(self):  # noqa: D401
            return None

        def to(self, *a, **k):
            return self

        def __call__(self, *, model, batch):  # mimic output with .total_loss
            class _Out:
                total_loss = bad
                semantic_loss = bad
                attention_entropy_loss = bad
                sender_effect_l1_loss = bad
                regulatory_l1_loss = bad
                residual_drift_l2_loss = bad
                residual_growth_l2_loss = bad
                growth_supervision_loss = bad
            return _Out()

    trainer.objective = _NanObjective()
    with pytest.raises(CCRTValidationError):
        trainer.train_step(batch)


def test_deterministic_repeated_run_matches():
    t1 = build_trainer(epochs=2)
    h1 = t1.fit(train_batches=[make_batch(), make_batch()])
    t2 = build_trainer(epochs=2)
    h2 = t2.fit(train_batches=[make_batch(), make_batch()])
    # same seed + same data ordering -> identical total loss trajectory
    for e1, e2 in zip(h1, h2):
        assert e1.train["total_loss"] == pytest.approx(e2.train["total_loss"], abs=1e-9)
    # and identical final parameters
    for p1, p2 in zip(t1.model.parameters(), t2.model.parameters()):
        assert torch.allclose(p1, p2, atol=1e-9)
