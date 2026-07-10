"""Tests for PanIN ontology mapping."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.adapters.panin import (
    build_panin_ontology,
    build_reference_panin_adapter_config,
)
from stagebridge.ccrt.contracts import CCRTValidationError


def make_ontology():
    return build_panin_ontology(build_reference_panin_adapter_config("/tmp/panin"))


def test_biological_system_spec_validates():
    onto = make_ontology()
    onto.biological_system_spec.validate()
    assert onto.biological_system_spec.biological_system_id == "panin_progression"


def test_included_receiver_labels_map_exactly():
    onto = make_ontology()
    assert onto.map_receiver_annotation("normal epithelium") == "normal_duct"
    assert onto.map_receiver_annotation("low_grade_PanIN") == "low_grade_panin"
    assert onto.map_receiver_annotation("high_grade_PanIN") == "high_grade_panin"


def test_sender_labels_map_exactly():
    onto = make_ontology()
    assert onto.map_sender_context_annotation("panCAF") == "caf"
    assert onto.map_sender_context_annotation("apCAF") == "apcaf"
    assert onto.map_sender_context_annotation("myCAF") == "mycaf"


def test_unknown_label_fails_in_strict_mode():
    onto = make_ontology()  # strict by default
    with pytest.raises(CCRTValidationError):
        onto.map_receiver_annotation("mystery_state")
    with pytest.raises(CCRTValidationError):
        onto.map_sender_context_annotation("mystery_sender")


def test_excluded_label_returns_none_not_bucket():
    import dataclasses
    cfg = build_reference_panin_adapter_config("/tmp/panin")
    cfg = dataclasses.replace(cfg, strict_unknown_annotations=False, excluded_annotations=("fat",))
    onto = build_panin_ontology(cfg)
    # excluded label maps to None, never to a catch-all id
    assert onto.map_sender_context_annotation("fat") is None


def test_transition_edges_use_valid_states():
    onto = make_ontology()
    states = onto.biological_system_spec.receiver_state_ids
    for eid, src, tgt in onto.transition_edge_records:
        assert src in states
        assert tgt in states


def test_no_duplicate_canonical_receiver_ids():
    onto = make_ontology()
    ids = list(onto.biological_system_spec.receiver_state_ids)
    assert len(ids) == len(set(ids))
