"""Tests for LUAD ontology mapping."""

from __future__ import annotations

import dataclasses

import pytest

from stagebridge.ccrt.adapters.luad import (
    build_luad_ontology,
    build_reference_luad_adapter_config,
)
from stagebridge.ccrt.contracts import CCRTValidationError


def make_ontology():
    return build_luad_ontology(build_reference_luad_adapter_config("/tmp/luad"))


def test_biological_system_spec_validates():
    onto = make_ontology()
    onto.biological_system_spec.validate()
    assert onto.biological_system_spec.biological_system_id == "luad_premalignant_progression"


def test_receiver_labels_map_exactly():
    onto = make_ontology()
    assert onto.map_receiver_annotation("Normal") == "normal"
    assert onto.map_receiver_annotation("AAH") == "aah"
    assert onto.map_receiver_annotation("AIS") == "ais"
    assert onto.map_receiver_annotation("MIA") == "mia"
    assert onto.map_receiver_annotation("LUAD") == "invasive_luad"


def test_sender_labels_map_exactly():
    onto = make_ontology()
    assert onto.map_sender_context_annotation("AT2") == "at2"
    assert onto.map_sender_context_annotation("Fibroblast lineage") == "fibroblast"
    assert onto.map_sender_context_annotation("Macrophages") == "macrophage"
    assert onto.map_sender_context_annotation("Mast cells") == "mast_cell"
    assert onto.map_sender_context_annotation("T cell lineage") == "t_cell"


def test_unknown_label_fails_in_strict_mode():
    onto = make_ontology()
    with pytest.raises(CCRTValidationError):
        onto.map_receiver_annotation("mystery_state")
    with pytest.raises(CCRTValidationError):
        onto.map_sender_context_annotation("mystery_sender")


def test_excluded_label_returns_none_not_bucket():
    cfg = build_reference_luad_adapter_config("/tmp/luad")
    cfg = dataclasses.replace(cfg, strict_unknown_annotations=False, excluded_annotations=("Debris",))
    onto = build_luad_ontology(cfg)
    assert onto.map_sender_context_annotation("Debris") is None


def test_transition_edges_use_valid_states():
    onto = make_ontology()
    states = onto.biological_system_spec.receiver_state_ids
    for eid, src, tgt in onto.transition_edge_records:
        assert src in states
        assert tgt in states


def test_regulatory_mediator_available():
    onto = make_ontology()
    assert "luad_evolutionary_state" in onto.biological_system_spec.regulatory_mediator_ids


def test_regulatory_mediator_absent_when_no_block():
    cfg = build_reference_luad_adapter_config("/tmp/luad")
    cfg = dataclasses.replace(cfg, regulatory_feature_block=None)
    onto = build_luad_ontology(cfg)
    assert onto.biological_system_spec.regulatory_mediator_ids == frozenset()


def test_no_duplicate_canonical_receiver_ids():
    onto = make_ontology()
    ids = list(onto.biological_system_spec.receiver_state_ids)
    assert len(ids) == len(set(ids))
