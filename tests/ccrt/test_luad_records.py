"""Tests for canonical LUAD record generation."""

from __future__ import annotations

from stagebridge.ccrt.adapters.luad import adapt_reference_luad
from stagebridge.ccrt.contracts import (
    RECEIVERS_SCHEMA,
    SAMPLES_SCHEMA,
    SENDER_CONTEXT_SCHEMA,
    TRANSITION_EDGES_SCHEMA,
)

from _luad_fixtures import FixtureLUADSpatialLoader, build_luad_source_fixture


def _adapt(tmp_path):
    cfg = build_luad_source_fixture(tmp_path)
    return adapt_reference_luad(cfg, spatial_loader=FixtureLUADSpatialLoader())


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
    assert states <= {"normal", "aah", "ais", "mia", "invasive_luad"}
    r0 = out.records.receiver_records[0]
    assert "x_spatial" in r0 and "semantic_features" in r0 and "receiver_features" in r0


def test_sender_records_preserve_individual_components(tmp_path):
    out = _adapt(tmp_path)
    for s in out.records.sender_context_records:
        assert s["sender_context_mask"] == 1
        assert isinstance(s["distance_to_receiver"], float)
        assert s["distance_to_receiver"] >= 0
        assert s["abundance"] >= 0
        assert s["sender_source"] == "tangram"  # backend preserved on every record


def test_features_are_plain_float_lists(tmp_path):
    out = _adapt(tmp_path)
    for r in out.records.receiver_records:
        for key in ("semantic_features", "receiver_features"):
            assert all(isinstance(v, float) for v in r[key])


def test_regulatory_features_attached(tmp_path):
    out = _adapt(tmp_path)
    # regulatory space is genuinely available for LUAD
    assert any("regulatory_features" in r for r in out.records.receiver_records)


def test_modality_records_and_relationships(tmp_path):
    out = _adapt(tmp_path)
    mods = {m["modality_id"] for m in out.records.modality_records}
    assert mods == {"snrna", "visium"}
    # shared donors -> same_donor, never same_observation
    rels = {r["relationship_type"] for r in out.records.modality_relationships}
    assert rels == {"same_donor"}
    assert "same_observation" not in rels


def test_backend_ids_recorded(tmp_path):
    out = _adapt(tmp_path)
    assert out.records.backend_ids == ("tangram",)


def test_edge_records_match_config(tmp_path):
    out = _adapt(tmp_path)
    ids = {e["transition_edge_id"] for e in out.records.transition_edge_records}
    assert ids == {
        "normal__to__aah",
        "aah__to__ais",
        "ais__to__mia",
        "mia__to__invasive_luad",
    }
