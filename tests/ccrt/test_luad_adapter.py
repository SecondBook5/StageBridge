"""Tests for the LUAD adapter output and edge partitions."""

from __future__ import annotations

import pytest
import torch

from stagebridge.ccrt.adapters.luad import adapt_reference_luad
from stagebridge.ccrt.contracts import CCRTValidationError

from _luad_fixtures import FixtureLUADSpatialLoader, build_luad_source_fixture


def _adapt(tmp_path):
    cfg = build_luad_source_fixture(tmp_path)
    return adapt_reference_luad(cfg, spatial_loader=FixtureLUADSpatialLoader())


def test_adapter_produces_edge_partitions(tmp_path):
    out = _adapt(tmp_path)
    assert len(out.edge_partitions) >= 1
    for p in out.edge_partitions:
        assert p.source_batch.batch_size() == len(p.source_receiver_ids)
        assert p.target_semantic_features.shape[0] == len(p.target_receiver_ids)
        p.source_batch.validate()


def test_partitions_carry_backend_and_platform(tmp_path):
    out = _adapt(tmp_path)
    for p in out.edge_partitions:
        assert p.platform == "visium"
        assert p.context_backend_id == "tangram"  # single preserved backend
        assert all(e == p.transition_edge_id for e in p.source_batch.transition_edge_id)


def test_source_states_correct(tmp_path):
    out = _adapt(tmp_path)
    for p in out.edge_partitions:
        assert all(s == p.source_receiver_state_id for s in p.source_batch.receiver_state_id)


def test_batch_never_mixes_backends(tmp_path):
    out = _adapt(tmp_path)
    # every partition is tied to exactly one backend id
    for p in out.edge_partitions:
        assert p.provenance["context_backend_id"] == p.context_backend_id


def test_validation_report_populated(tmp_path):
    out = _adapt(tmp_path)
    r = out.validation_report
    assert r.passed
    assert r.num_donors == 3
    assert r.semantic_dimension == 8
    assert r.regulatory_dimension == 34
    assert r.regulatory_feature_space_id is not None
    assert r.coordinate_units.get("visium") == "microns"
    assert r.context_counts_by_backend.get("tangram", 0) >= 1
    assert r.modality_relationship_counts.get("same_donor", 0) == 1
    assert r.context_backend_ids == ("tangram",)


def test_supported_edges_present(tmp_path):
    out = _adapt(tmp_path)
    edge_ids = {p.transition_edge_id for p in out.edge_partitions}
    assert "normal__to__aah" in edge_ids


def test_requires_spatial_loader(tmp_path):
    cfg = build_luad_source_fixture(tmp_path)
    with pytest.raises(CCRTValidationError):
        adapt_reference_luad(cfg, spatial_loader=None)


def test_continuous_distances_preserved_in_batch(tmp_path):
    out = _adapt(tmp_path)
    for p in out.edge_partitions:
        dist = torch.tensor(p.source_batch.distance_to_receiver, dtype=torch.float64)
        assert bool((dist >= 0).all())
        assert bool(torch.isfinite(dist).all())


def test_context_type_ids_are_nested_string_ids_or_none(tmp_path):
    out = _adapt(tmp_path)
    for p in out.edge_partitions:
        rows = p.source_batch.sender_context_type_ids
        for row in rows:
            for tok in row:
                assert tok is None or isinstance(tok, str)
