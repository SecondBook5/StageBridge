"""LUAD ontology construction.

Maps source-backed LUAD labels to canonical CCRT grammar ids and builds a
``BiologicalSystemSpec`` that passes the existing grammar validation. Unknown
labels fail in strict mode and are never silently bucketed into a catch-all id.

LUAD has a genuinely available regulatory space (the lesion WES / evolutionary
block), declared here as a single registered regulatory mediator
``luad_evolutionary_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...contracts.errors import CCRTGrammarError, CCRTValidationError
from ...grammar import (
    GROWTH_MASS,
    SEMANTIC_DRIFT,
    BiologicalSystemSpec,
    ReceiverBehavior,
    ReceiverState,
    RegulatoryMediator,
    SenderContextType,
    SignalProgram,
    TransitionEdge,
)
from .config import LUADAdapterConfig

__all__ = [
    "LUADOntologyEntry",
    "LUADOntology",
    "build_luad_ontology",
]

#: Single generic signal program (no discrete per-observation program matrix).
LUAD_CONTEXT_SIGNAL_PROGRAM_ID = "luad_context_signal"
#: Single registered regulatory mediator (the lesion evolutionary state space).
LUAD_REGULATORY_MEDIATOR_ID = "luad_evolutionary_state"


@dataclass(frozen=True)
class LUADOntologyEntry:
    source_label: str
    canonical_id: str
    role: str
    included: bool
    evidence: str
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class LUADOntology:
    biological_system_spec: BiologicalSystemSpec
    receiver_entries: tuple[LUADOntologyEntry, ...]
    sender_context_entries: tuple[LUADOntologyEntry, ...]
    stage_entries: tuple[LUADOntologyEntry, ...]
    transition_edge_records: tuple[tuple[str, str, str], ...]
    receiver_map: Mapping[str, str]
    sender_map: Mapping[str, str]
    stage_map: Mapping[str, str]
    strict: bool = True
    excluded: frozenset[str] = frozenset()

    def _map(self, label: str, table: Mapping[str, str], kind: str) -> str | None:
        if label in table:
            return table[label]
        if label in self.excluded:
            return None
        if self.strict:
            raise CCRTValidationError(
                f"unknown {kind} label '{label}' (strict mode; not in map and not "
                "explicitly excluded)"
            )
        return None

    def map_receiver_annotation(self, label: str) -> str | None:
        return self._map(label, self.receiver_map, "receiver")

    def map_sender_context_annotation(self, label: str) -> str | None:
        return self._map(label, self.sender_map, "sender-context")

    def map_stage(self, label: str) -> str | None:
        return self._map(label, self.stage_map, "stage")


def build_luad_ontology(config: LUADAdapterConfig) -> LUADOntology:
    """Build the LUAD ontology + validated BiologicalSystemSpec."""
    # canonical receiver states (dedup, preserve first-seen order)
    receiver_states: list[str] = []
    for canonical in config.receiver_annotation_map.values():
        if canonical not in receiver_states:
            receiver_states.append(canonical)

    sender_types: list[str] = []
    for canonical in config.sender_context_annotation_map.values():
        if canonical not in sender_types:
            sender_types.append(canonical)

    # entries (provenance)
    receiver_entries = tuple(
        LUADOntologyEntry(
            source_label=src, canonical_id=canon, role="receiver_state",
            included=True, evidence="config.receiver_annotation_map",
        )
        for src, canon in config.receiver_annotation_map.items()
    )
    sender_entries = tuple(
        LUADOntologyEntry(
            source_label=src, canonical_id=canon, role="sender_context_type",
            included=True, evidence="config.sender_context_annotation_map",
        )
        for src, canon in config.sender_context_annotation_map.items()
    )
    stage_entries = tuple(
        LUADOntologyEntry(
            source_label=src, canonical_id=canon, role="stage",
            included=True, evidence="config.stage_map",
        )
        for src, canon in config.stage_map.items()
    )
    excluded_entries = tuple(
        LUADOntologyEntry(
            source_label=lbl, canonical_id="", role="excluded", included=False,
            evidence="config.excluded_annotations", exclusion_reason="source-excluded",
        )
        for lbl in config.excluded_annotations
    )

    # grammar objects
    grammar_states = tuple(
        ReceiverState(s, order=float(i)) for i, s in enumerate(receiver_states)
    )
    grammar_edges = tuple(
        TransitionEdge(eid, src, tgt) for eid, src, tgt in config.transition_edges
    )
    grammar_senders = tuple(
        SenderContextType(t, signal_program_ids=(LUAD_CONTEXT_SIGNAL_PROGRAM_ID,))
        for t in sender_types
    )
    signal_programs = (SignalProgram(LUAD_CONTEXT_SIGNAL_PROGRAM_ID),)
    behaviors = (ReceiverBehavior(SEMANTIC_DRIFT), ReceiverBehavior(GROWTH_MASS))

    # Regulatory mediators are genuinely available (lesion evolutionary block).
    regulatory_mediators: tuple[RegulatoryMediator, ...] = ()
    if config.regulatory_feature_block is not None:
        regulatory_mediators = (
            RegulatoryMediator(
                LUAD_REGULATORY_MEDIATOR_ID,
                signal_program_ids=(LUAD_CONTEXT_SIGNAL_PROGRAM_ID,),
                description="Lesion WES/evolutionary regulatory-mediator state (r)",
            ),
        )

    spec = BiologicalSystemSpec(
        biological_system_id=config.biological_system_id,
        receiver_states=grammar_states,
        transition_edges=grammar_edges,
        sender_context_types=grammar_senders,
        signal_programs=signal_programs,
        receiver_behaviors=behaviors,
        regulatory_mediators=regulatory_mediators,
        metadata={
            "hypothesis": (
                "epithelial-niche sender context alters LUAD premalignant "
                "epithelial transition drift and growth"
            )
        },
    )
    try:
        spec.validate()
    except CCRTGrammarError as exc:
        raise CCRTValidationError(f"LUAD BiologicalSystemSpec invalid: {exc}") from exc

    return LUADOntology(
        biological_system_spec=spec,
        receiver_entries=receiver_entries + excluded_entries,
        sender_context_entries=sender_entries,
        stage_entries=stage_entries,
        transition_edge_records=config.transition_edges,
        receiver_map=dict(config.receiver_annotation_map),
        sender_map=dict(config.sender_context_annotation_map),
        stage_map=dict(config.stage_map),
        strict=config.strict_unknown_annotations,
        excluded=frozenset(config.excluded_annotations),
    )
