"""Fast CPU smoke tests for the synthetic benchmark machinery."""

from __future__ import annotations

import torch

from stagebridge.ccrt.synthetic import (
    SyntheticBenchmarkConfig,
    SyntheticSystemConfig,
    build_synthetic_benchmark_components,
    build_synthetic_training_sequence,
    generate_synthetic_dataset,
    predict_synthetic_batch,
    run_synthetic_benchmark_matrix,
    run_synthetic_scenario_benchmark,
)

# small + few epochs for a fast smoke test
SMOKE_SYS = SyntheticSystemConfig(
    train_batches=2, validation_batches=1, test_batches=1, batch_size=4,
    senders_per_receiver=3,
)
SMOKE_BENCH = SyntheticBenchmarkConfig(epochs=3, dtype="float64")


def test_smoke_scenario_runs():
    out = run_synthetic_scenario_benchmark(
        system=SMOKE_SYS, scenario_id="mixed_drift_growth", benchmark=SMOKE_BENCH
    )
    r = out.result
    assert r.all_metrics_finite
    assert r.initial_test_loss == r.initial_test_loss  # not NaN
    assert len(out.history) == 3
    assert r.scenario_id == "mixed_drift_growth"


def test_training_sequence_alternates_and_excludes_test():
    ds = generate_synthetic_dataset(system=SMOKE_SYS, scenario_id="drift_only")
    seq = build_synthetic_training_sequence(ds)
    # include_null default True -> alternating factual/null, 2 factual batches -> 4
    assert len(seq) == 2 * len(ds.train)
    # the exact factual batches appear (identity), test batches never do
    test_ids = {id(ex.factual_batch) for ex in ds.test}
    assert all(id(b) not in test_ids for b in seq)


def test_training_sequence_no_null_when_disabled():
    sys_nonull = SyntheticSystemConfig(
        train_batches=2, validation_batches=1, test_batches=1, batch_size=4,
        senders_per_receiver=3, include_null_context_training_pairs=False,
    )
    ds = generate_synthetic_dataset(system=sys_nonull, scenario_id="drift_only")
    seq = build_synthetic_training_sequence(ds)
    assert len(seq) == len(ds.train)


def test_parameters_change_after_training():
    ds = generate_synthetic_dataset(system=SMOKE_SYS, scenario_id="mixed_drift_growth")
    comp = build_synthetic_benchmark_components(dataset=ds, benchmark=SMOKE_BENCH)
    before = [p.detach().clone() for p in comp.model.parameters()]
    seq = build_synthetic_training_sequence(ds)
    comp.trainer.fit(train_batches=seq)
    after = list(comp.model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_teacher_and_student_parameters_independent_storage():
    ds = generate_synthetic_dataset(system=SMOKE_SYS, scenario_id="mixed_drift_growth")
    comp = build_synthetic_benchmark_components(dataset=ds, benchmark=SMOKE_BENCH)
    teacher_ptrs = {
        t.data_ptr()
        for t in vars(ds.teacher.parameters).values()
        if isinstance(t, torch.Tensor)
    }
    student_ptrs = {p.data_ptr() for p in comp.model.parameters()}
    assert teacher_ptrs.isdisjoint(student_ptrs)


def test_predict_does_not_mutate_model_mode():
    ds = generate_synthetic_dataset(system=SMOKE_SYS, scenario_id="drift_only")
    comp = build_synthetic_benchmark_components(dataset=ds, benchmark=SMOKE_BENCH)
    comp.model.train()
    predict_synthetic_batch(model=comp.model, batch=ds.test[0].factual_batch)
    assert comp.model.training  # restored to train mode


def test_matrix_preserves_order_minimal():
    scenarios = ("null_context", "drift_only", "growth_only")
    results = run_synthetic_benchmark_matrix(
        system=SMOKE_SYS, benchmark=SMOKE_BENCH, scenario_ids=scenarios
    )
    assert tuple(r.scenario_id for r in results) == scenarios
    assert all(r.all_metrics_finite for r in results)
