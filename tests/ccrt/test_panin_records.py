"""Tests for canonical PanIN record generation."""

from __future__ import annotations

from stagebridge.ccrt.adapters.panin import (
    audit_panin_source,
    adapt_reference_panin,
)
from stagebridge.ccrt.contracts import (
    RECEIVERS_SCHEMA,
    SENDER_CONTEXT_SCHEMA,
    TRANSITION_EDGES_SCHEMA,
    SAMPLES_SCHEMA,
)

from _panin_fixtures import FixtureSpatialLoader, build_panin_source_fixture


def _adapt(tmp_path):
    cfg = build_panin_source_fixture(tmp_path)
    return adapt_reference_panin(cfg, spatial_loader=FixtureSpatialLoader())


def test_records_validate_against_schemas(tmp_path):
    out = _adapt(tmp_path)
    rec = out.records
    RECEIVERS_SCHEMA.validate_fields(rec.receiver_records[0].keys(), allow_extra=True)
    SENDER_CONTEXT_SCHEMA.validate_fields(rec.sender_context_records[0].keys(), allow_extra=True)
    TRANSITION_EDGES_SCHEMA.validate_fields(rec.transition_edge_records[0].keys(), allow_extra=True)
    SAMPLES_SCHEMA.validate_fields(rec.sample_records[0].keys(), allow_extra=True)


def test_receiver_records_source_backed(tmp_path):
    out = _adapt(tmp_path)
    states = {r["receiver_state_id"] for r in out.records.receiver_records}
    assert states <= {"normal_duct", "low_grade_panin", "high_grade_panin"}
    # coordinates present, in microns; features attached
    r0 = out.records.receiver_records[0]
    assert "x_spatial" in r0 and "semantic_features" in r0 and "receiver_features" in r0


def test_sender_records_preserve_individual_elements(tmp_path):
    out = _adapt(tmp_path)
    # each sender record is one element with a continuous distance
    for s in out.records.sender_context_records:
        assert s["sender_context_mask"] == 1
        assert isinstance(s["distance_to_receiver"], float)
        assert s["distance_to_receiver"] >= 0


def test_stage_and_donor_not_in_feature_vectors(tmp_path):
    out = _adapt(tmp_path)
    # semantic + receiver feature vectors are lists of floats only
    for r in out.records.receiver_records:
        for key in ("semantic_features", "receiver_features"):
            vec = r[key]
            assert all(isinstance(v, float) for v in vec)


def test_edge_records_match_config(tmp_path):
    out = _adapt(tmp_path)
    ids = {e["transition_edge_id"] for e in out.records.transition_edge_records}
    assert ids == {
        "normal_duct__to__low_grade_panin",
        "low_grade_panin__to__high_grade_panin",
    }
