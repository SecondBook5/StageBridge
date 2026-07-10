"""Tests for synthetic dataset generation."""

from __future__ import annotations

import torch

from stagebridge.ccrt.synthetic import (
    SyntheticSystemConfig,
    build_synthetic_biological_system_spec,
    generate_synthetic_dataset,
)

# small system for fast tests
SYS = SyntheticSystemConfig(train_batches=2, validation_batches=1, test_batches=1, batch_size=4)


def test_system_spec_validates_and_registry_counts():
    spec = build_synthetic_biological_system_spec(SYS)
    spec.validate()
    ds = generate_synthetic_dataset(system=SYS, scenario_id="mixed_drift_growth")
    reg = ds.index_registry
    assert reg.num_real_sender_context_types == SYS.num_sender_context_types
    assert reg.num_transition_edges == SYS.num_transition_edges


def test_deterministic_generation():
    ds1 = generate_synthetic_dataset(system=SYS, scenario_id="drift_only")
    ds2 = generate_synthetic_dataset(system=SYS, scenario_id="drift_only")
    b1 = ds1.train[0].factual_batch
    b2 = ds2.train[0].factual_batch
    assert torch.equal(b1.receiver_features, b2.receiver_features)
    assert torch.equal(b1.target_semantic_features, b2.target_semantic_features)


def test_changed_seed_changes_data():
    ds1 = generate_synthetic_dataset(system=SYS, scenario_id="drift_only")
    sys2 = SyntheticSystemConfig(
        seed=1, train_batches=2, validation_batches=1, test_batches=1, batch_size=4
    )
    ds2 = generate_synthetic_dataset(system=sys2, scenario_id="drift_only")
    assert not torch.equal(
        ds1.train[0].factual_batch.receiver_features,
        ds2.train[0].factual_batch.receiver_features,
    )


def test_splits_independent():
    ds = generate_synthetic_dataset(system=SYS, scenario_id="mixed_drift_growth")
    assert len(ds.train) == 2 and len(ds.validation) == 1 and len(ds.test) == 1
    # train and test batches are not the same tensor objects / values
    train_rf = ds.train[0].factual_batch.receiver_features
    test_rf = ds.test[0].factual_batch.receiver_features
    assert train_rf is not test_rf
    assert not torch.equal(train_rf, test_rf)


def test_all_batches_validate_and_shapes():
    ds = generate_synthetic_dataset(system=SYS, scenario_id="mixed_drift_growth")
    for split in (ds.train, ds.validation, ds.test):
        for ex in split:
            ex.factual_batch.validate()
            ex.null_context_batch.validate()
            assert ex.factual_batch.semantic_dim() == SYS.semantic_dim


def test_null_masks_all_false_and_context_zero():
    ds = generate_synthetic_dataset(system=SYS, scenario_id="drift_only")
    ex = ds.train[0]
    assert not bool(ex.null_context_batch.sender_mask.any())
    assert float(ex.null_context_truth.context_state.abs().max()) == 0.0


def test_factual_truth_matches_target_before_noise():
    # with zero noise, target == teacher destination (up to permutation of rows)
    sys0 = SyntheticSystemConfig(
        train_batches=1, validation_batches=1, test_batches=1, batch_size=4,
        target_noise_std=0.0, growth_noise_std=0.0,
    )
    ds = generate_synthetic_dataset(system=sys0, scenario_id="mixed_drift_growth")
    ex = ds.train[0]
    dest = ex.factual_truth.destination_semantic_features
    tgt = ex.factual_batch.target_semantic_features
    # same set of points (order may be permuted)
    dest_sorted = torch.sort(dest.reshape(-1))[0]
    tgt_sorted = torch.sort(tgt.reshape(-1).to(dest.dtype))[0]
    assert torch.allclose(dest_sorted, tgt_sorted, atol=1e-5)


def test_hidden_truth_absent_from_batch_metadata():
    ds = generate_synthetic_dataset(system=SYS, scenario_id="mixed_drift_growth")
    meta = ds.train[0].factual_batch.metadata
    assert set(meta.keys()) == {
        "synthetic_scenario_id", "synthetic_split",
        "synthetic_batch_index", "synthetic_seed",
    }
    # no truth tensors leaked
    for v in meta.values():
        assert not isinstance(v, torch.Tensor)


def test_target_permutation_preserves_point_set():
    sys0 = SyntheticSystemConfig(
        train_batches=1, validation_batches=1, test_batches=1, batch_size=6,
        target_noise_std=0.0, growth_noise_std=0.0,
    )
    ds = generate_synthetic_dataset(system=sys0, scenario_id="drift_only")
    ex = ds.train[0]
    dest = ex.factual_truth.destination_semantic_features
    tgt = ex.factual_batch.target_semantic_features
    d_rows = {tuple(round(x, 5) for x in r.tolist()) for r in dest}
    t_rows = {tuple(round(x, 5) for x in r.to(dest.dtype).tolist()) for r in tgt}
    assert d_rows == t_rows
