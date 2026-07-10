"""Tests for the cross-system CCRT index registry."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts import CCRTValidationError
from stagebridge.ccrt.data import (
    QUALIFIED_ID_SEPARATOR,
    CCRTIndexRegistry,
    qualify_grammar_id,
)
from stagebridge.ccrt.grammar import (
    SEMANTIC_DRIFT,
    BiologicalSystemSpec,
    ReceiverBehavior,
    ReceiverState,
    SenderContextType,
    TransitionEdge,
)


def make_spec(sid, senders, edges=(("e0", "s0", "s1"),)):
    return BiologicalSystemSpec(
        biological_system_id=sid,
        receiver_states=(ReceiverState("s0"), ReceiverState("s1")),
        transition_edges=tuple(TransitionEdge(e, s, t) for e, s, t in edges),
        sender_context_types=tuple(SenderContextType(s) for s in senders),
        receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
    )


def test_qualify_grammar_id():
    assert qualify_grammar_id("sysA", "ctx") == f"sysA{QUALIFIED_ID_SEPARATOR}ctx"


def test_qualify_rejects_separator_and_empty():
    with pytest.raises(CCRTValidationError):
        qualify_grammar_id("sys::A", "ctx")
    with pytest.raises(CCRTValidationError):
        qualify_grammar_id("sysA", "")


def test_one_system_registry():
    reg = CCRTIndexRegistry.from_system_specs([make_spec("sysA", ["c0", "c1"])])
    assert reg.num_biological_systems == 1
    assert reg.num_real_sender_context_types == 2
    assert reg.num_sender_context_types == 3  # + empty
    assert reg.empty_sender_context_type_index == 2
    assert reg.num_transition_edges == 1


def test_two_system_registry_order_preserved():
    reg = CCRTIndexRegistry.from_system_specs(
        [make_spec("sysA", ["a0", "a1"]), make_spec("sysB", ["b0"])]
    )
    assert reg.num_biological_systems == 2
    assert reg.encode_biological_system("sysA") == 0
    assert reg.encode_biological_system("sysB") == 1
    assert reg.num_real_sender_context_types == 3
    assert reg.empty_sender_context_type_index == 3


def test_duplicate_biological_systems_fail():
    with pytest.raises(CCRTValidationError):
        CCRTIndexRegistry.from_system_specs(
            [make_spec("sysA", ["c0"]), make_spec("sysA", ["c1"])]
        )


def test_same_local_sender_name_distinct_indices():
    reg = CCRTIndexRegistry.from_system_specs(
        [make_spec("sysA", ["shared"]), make_spec("sysB", ["shared"])]
    )
    a = reg.encode_sender_context_type("sysA", "shared")
    b = reg.encode_sender_context_type("sysB", "shared")
    assert a != b


def test_same_local_edge_name_distinct_indices():
    reg = CCRTIndexRegistry.from_system_specs(
        [
            make_spec("sysA", ["c0"], edges=(("shared", "s0", "s1"),)),
            make_spec("sysB", ["c0"], edges=(("shared", "s0", "s1"),)),
        ]
    )
    assert reg.encode_transition_edge("sysA", "shared") != reg.encode_transition_edge(
        "sysB", "shared"
    )


def test_empty_sender_is_final_index():
    reg = CCRTIndexRegistry.from_system_specs([make_spec("sysA", ["c0", "c1", "c2"])])
    real = [reg.encode_sender_context_type("sysA", c) for c in ("c0", "c1", "c2")]
    assert max(real) == reg.empty_sender_context_type_index - 1
    assert reg.empty_sender_context_type_index == 3


def test_encode_decode_round_trip():
    reg = CCRTIndexRegistry.from_system_specs(
        [make_spec("sysA", ["c0", "c1"]), make_spec("sysB", ["d0"])]
    )
    idx = reg.encode_sender_context_type("sysB", "d0")
    assert reg.decode_sender_context_type(idx) == ("sysB", "d0")
    eidx = reg.encode_transition_edge("sysA", "e0")
    assert reg.decode_transition_edge(eidx) == ("sysA", "e0")
    assert reg.decode_biological_system(0) == "sysA"


def test_unknown_id_fails():
    reg = CCRTIndexRegistry.from_system_specs([make_spec("sysA", ["c0"])])
    with pytest.raises(CCRTValidationError):
        reg.encode_sender_context_type("sysA", "missing")
    with pytest.raises(CCRTValidationError):
        reg.encode_biological_system("sysZ")
    with pytest.raises(CCRTValidationError):
        reg.encode_transition_edge("sysA", "no_edge")


def test_invalid_index_fails():
    reg = CCRTIndexRegistry.from_system_specs([make_spec("sysA", ["c0"])])
    with pytest.raises(CCRTValidationError):
        reg.decode_sender_context_type(999)
    with pytest.raises(CCRTValidationError):
        reg.decode_transition_edge(999)
    with pytest.raises(CCRTValidationError):
        reg.decode_biological_system(999)


def test_empty_sender_cannot_be_decoded_as_ontology():
    reg = CCRTIndexRegistry.from_system_specs([make_spec("sysA", ["c0"])])
    with pytest.raises(CCRTValidationError):
        reg.decode_sender_context_type(reg.empty_sender_context_type_index)


def test_empty_specs_fail():
    with pytest.raises(CCRTValidationError):
        CCRTIndexRegistry.from_system_specs([])
