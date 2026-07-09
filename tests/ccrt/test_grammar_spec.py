"""Tests for BiologicalSystemSpec and grammar entities.

Two minimal, biologically-distinct specs (LUAD-like and PanIN-like) exercise the
"same grammar, different vocabulary" contract: both validate, both are the same
class, and they share NO sender-context vocabulary.
"""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts import CCRTForbiddenFieldError, CCRTLeakageError
from stagebridge.ccrt.contracts.errors import CCRTGrammarError
from stagebridge.ccrt.grammar import (
    GROWTH_MASS,
    SEMANTIC_DRIFT,
    BiologicalSystemSpec,
    CounterfactualPerturbation,
    ReceiverBehavior,
    ReceiverState,
    RegulatoryMediator,
    SenderContextType,
    SignalProgram,
    TransitionEdge,
    validate_biological_system_spec,
)


# ---------------------------------------------------------------------------
# Minimal example specs (test-only; no biology in core source)
# ---------------------------------------------------------------------------


def make_luad_like_spec() -> BiologicalSystemSpec:
    return BiologicalSystemSpec(
        biological_system_id="luad_precursor_test",
        receiver_states=(
            ReceiverState("normal_alveolar", order=0),
            ReceiverState("aah_like", order=1),
            ReceiverState("ais_like", order=2),
        ),
        transition_edges=(
            TransitionEdge(
                "normal_alveolar_to_aah_like", "normal_alveolar", "aah_like"
            ),
            TransitionEdge("aah_like_to_ais_like", "aah_like", "ais_like"),
        ),
        sender_context_types=(
            SenderContextType(
                "il1b_high_macrophage", signal_program_ids=("il1b_il1r1", "nfkb")
            ),
            SenderContextType("inflammatory_myeloid", signal_program_ids=("nfkb",)),
        ),
        signal_programs=(
            SignalProgram("il1b_il1r1"),
            SignalProgram("nfkb"),
        ),
        receiver_behaviors=(
            ReceiverBehavior(SEMANTIC_DRIFT),
            ReceiverBehavior(GROWTH_MASS),
        ),
        regulatory_mediators=(
            RegulatoryMediator("nfkb_mediator", signal_program_ids=("nfkb",)),
        ),
        counterfactual_perturbations=(
            CounterfactualPerturbation(
                "remove_il1b_high_macrophage",
                perturbation_kind="remove_sender_context",
                target_sender_context_type_ids=("il1b_high_macrophage",),
            ),
        ),
        metadata={"hypothesis": "IL1B-high macrophage niches alter drift/growth."},
    )


def make_panin_like_spec() -> BiologicalSystemSpec:
    return BiologicalSystemSpec(
        biological_system_id="panin_test",
        receiver_states=(
            ReceiverState("normal_duct", order=0),
            ReceiverState("low_grade_panin", order=1),
            ReceiverState("high_grade_panin", order=2),
        ),
        transition_edges=(
            TransitionEdge(
                "normal_duct_to_low_grade_panin", "normal_duct", "low_grade_panin"
            ),
            TransitionEdge(
                "low_grade_panin_to_high_grade_panin",
                "low_grade_panin",
                "high_grade_panin",
            ),
        ),
        sender_context_types=(
            SenderContextType("caf", signal_program_ids=("caf_inflammatory",)),
            SenderContextType("ecm_rich_stroma", signal_program_ids=("tgfb_ecm",)),
        ),
        signal_programs=(
            SignalProgram("caf_inflammatory"),
            SignalProgram("tgfb_ecm"),
        ),
        receiver_behaviors=(
            ReceiverBehavior(SEMANTIC_DRIFT),
            ReceiverBehavior(GROWTH_MASS),
        ),
        regulatory_mediators=(
            RegulatoryMediator("tgfb_ecm_mediator", signal_program_ids=("tgfb_ecm",)),
        ),
        counterfactual_perturbations=(
            CounterfactualPerturbation(
                "remove_caf",
                perturbation_kind="remove_sender_context",
                target_sender_context_type_ids=("caf",),
            ),
        ),
        metadata={"hypothesis": "CAF/ECM context alters PanIN drift/growth."},
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_both_specs_validate():
    make_luad_like_spec().validate()
    make_panin_like_spec().validate()
    # free-function entry point works too
    validate_biological_system_spec(make_luad_like_spec())
    validate_biological_system_spec(make_panin_like_spec())


def test_specs_share_same_grammar_class():
    luad = make_luad_like_spec()
    panin = make_panin_like_spec()
    assert type(luad) is type(panin) is BiologicalSystemSpec


def test_specs_need_not_share_sender_context_vocabulary():
    luad = make_luad_like_spec()
    panin = make_panin_like_spec()
    shared = luad.sender_context_type_ids & panin.sender_context_type_ids
    assert shared == frozenset(), f"unexpected shared vocabulary: {shared}"
    # yet both are valid — unification is at the grammar level, not vocabulary
    luad.validate()
    panin.validate()


def test_accessors_and_lookups():
    luad = make_luad_like_spec()
    assert luad.receiver_state_ids == {"normal_alveolar", "aah_like", "ais_like"}
    assert luad.has_receiver_behavior(SEMANTIC_DRIFT)
    assert not luad.has_receiver_behavior("nonexistent_behavior")
    edge = luad.get_transition_edge("aah_like_to_ais_like")
    assert edge.source_state_id == "aah_like"
    with pytest.raises(CCRTGrammarError):
        luad.get_transition_edge("no_such_edge")


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_invalid_edge_source_state_fails():
    with pytest.raises(CCRTGrammarError):
        BiologicalSystemSpec(
            biological_system_id="bad_edge",
            receiver_states=(ReceiverState("a"), ReceiverState("b")),
            transition_edges=(
                TransitionEdge("a_to_ghost", "a", "ghost_state"),
            ),
            sender_context_types=(SenderContextType("ctx"),),
            receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
        ).validate()


def test_duplicate_receiver_state_fails():
    with pytest.raises(CCRTGrammarError):
        BiologicalSystemSpec(
            biological_system_id="dup_states",
            receiver_states=(ReceiverState("a"), ReceiverState("a")),
            transition_edges=(TransitionEdge("a_to_a", "a", "a"),),
            sender_context_types=(SenderContextType("ctx"),),
            receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
        ).validate()


def test_sender_context_referencing_missing_signal_program_fails():
    with pytest.raises(CCRTGrammarError):
        BiologicalSystemSpec(
            biological_system_id="bad_sig_ref",
            receiver_states=(ReceiverState("a"), ReceiverState("b")),
            transition_edges=(TransitionEdge("a_to_b", "a", "b"),),
            sender_context_types=(
                SenderContextType("ctx", signal_program_ids=("ghost_program",)),
            ),
            signal_programs=(SignalProgram("real_program"),),
            receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
        ).validate()


def test_counterfactual_referencing_missing_sender_context_fails():
    with pytest.raises(CCRTGrammarError):
        BiologicalSystemSpec(
            biological_system_id="bad_cf_ref",
            receiver_states=(ReceiverState("a"), ReceiverState("b")),
            transition_edges=(TransitionEdge("a_to_b", "a", "b"),),
            sender_context_types=(SenderContextType("ctx"),),
            receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
            counterfactual_perturbations=(
                CounterfactualPerturbation(
                    "remove_ghost",
                    target_sender_context_type_ids=("ghost_ctx",),
                ),
            ),
        ).validate()


def test_metadata_with_leakage_field_fails():
    with pytest.raises(CCRTLeakageError):
        BiologicalSystemSpec(
            biological_system_id="leaky_meta",
            receiver_states=(ReceiverState("a"), ReceiverState("b")),
            transition_edges=(TransitionEdge("a_to_b", "a", "b"),),
            sender_context_types=(SenderContextType("ctx"),),
            receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
            metadata={"future_expression": "leak"},
        ).validate()


def test_metadata_with_forbidden_mechanism_key_fails():
    with pytest.raises(CCRTForbiddenFieldError):
        BiologicalSystemSpec(
            biological_system_id="worldy_meta",
            receiver_states=(ReceiverState("a"), ReceiverState("b")),
            transition_edges=(TransitionEdge("a_to_b", "a", "b"),),
            sender_context_types=(SenderContextType("ctx"),),
            receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
            metadata={"world_token": "leak"},
        ).validate()


def test_too_few_receiver_states_fails():
    with pytest.raises(CCRTGrammarError):
        BiologicalSystemSpec(
            biological_system_id="one_state",
            receiver_states=(ReceiverState("only"),),
            transition_edges=(TransitionEdge("only_to_only", "only", "only"),),
            sender_context_types=(SenderContextType("ctx"),),
            receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
        ).validate()


def test_empty_entity_id_fails_at_construction():
    with pytest.raises(CCRTGrammarError):
        ReceiverState("   ")
    with pytest.raises(CCRTGrammarError):
        TransitionEdge("edge", "", "b")


def test_forbidden_mechanism_id_in_registry_fails():
    # A registry ID that IS a forbidden mechanism term must fail validation.
    with pytest.raises(CCRTForbiddenFieldError):
        BiologicalSystemSpec(
            biological_system_id="ringy",
            receiver_states=(ReceiverState("a"), ReceiverState("b")),
            transition_edges=(TransitionEdge("a_to_b", "a", "b"),),
            sender_context_types=(SenderContextType("ring_id"),),
            receiver_behaviors=(ReceiverBehavior(SEMANTIC_DRIFT),),
        ).validate()
