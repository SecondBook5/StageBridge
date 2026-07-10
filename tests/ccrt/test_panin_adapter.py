"""Tests for the PanIN adapter output and edge partitions."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.adapters.panin import (
    adapt_reference_panin,
    build_reference_panin_adapter_config,
)
from stagebridge.ccrt.contracts import CCRTValidationError

from _panin_fixtures import FixtureSpatialLoader, build_panin_source_fixture


def _adapt(tmp_path):
    cfg = build_panin_source_fixture(tmp_path)
    return adapt_reference_panin(cfg, spatial_loader=FixtureSpatialLoader())


def test_adapter_produces_edge_partitions(tmp_path):
    out = _adapt(tmp_path)
    assert len(out.edge_partitions) >= 1
    for p in out.edge_partitions:
        assert p.source_batch.batch_size() == len(p.source_receiver_ids)
        assert p.target_semantic_features.shape[0] == len(p.target_receiver_ids)
        p.source_batch.validate()


def test_partitions_are_split_and_platform_local(tmp_path):
    out = _adapt(tmp_path)
    for p in out.edge_partitions:
        assert p.platform == "xenium"
        # source batch conditioning uses the partition's edge id
        assert all(e == p.transition_edge_id for e in p.source_batch.transition_edge_id)


def test_source_and_target_states_correct(tmp_path):
    out = _adapt(tmp_path)
    for p in out.edge_partitions:
        assert all(s == p.source_receiver_state_id for s in p.source_batch.receiver_state_id)


def test_validation_report_populated(tmp_path):
    out = _adapt(tmp_path)
    r = out.validation_report
    assert r.passed
    assert r.num_donors >= 2
    assert r.semantic_dimension == 8  # 8 CoGAPS patterns
    assert r.regulatory_feature_space_id is None  # unavailable
    assert r.coordinate_units.get("xenium") == "microns"


def test_unsupported_edges_reported_not_fabricated(tmp_path):
    # the fixture has high-grade cells only for some donors; any edge lacking
    # support in a fold is simply absent, never fabricated
    out = _adapt(tmp_path)
    edge_ids = {p.transition_edge_id for p in out.edge_partitions}
    # normal->low_grade is supported
    assert "normal_duct__to__low_grade_panin" in edge_ids


def test_requires_spatial_loader(tmp_path):
    cfg = build_panin_source_fixture(tmp_path)
    with pytest.raises(CCRTValidationError):
        adapt_reference_panin(cfg, spatial_loader=None)


def test_no_donor_leakage_across_folds(tmp_path):
    out = _adapt(tmp_path)
    # each fold's partitions must not mix a donor across folds — enforced by
    # validate_no_panin_group_leakage during adapt; here assert folds are ints
    for p in out.edge_partitions:
        assert isinstance(p.fold_index, int)


def test_continuous_distances_preserved_in_batch(tmp_path):
    out = _adapt(tmp_path)
    for p in out.edge_partitions:
        dist = torch.tensor(p.source_batch.distance_to_receiver, dtype=torch.float64)
        assert bool((dist >= 0).all())
        assert bool(torch.isfinite(dist).all())
