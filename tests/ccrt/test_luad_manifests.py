"""Tests for the LUAD modality manifest (never upgrade relationship strength)."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.adapters.luad import (
    ALLOWED_MODALITY_RELATIONSHIP_TYPES,
    LUADModalityRecord,
    LUADModalityRelationship,
    build_luad_modality_manifest,
    validate_luad_modality_manifest,
)
from stagebridge.ccrt.contracts import CCRTValidationError


def _modalities(shared_donor: bool):
    visium_donors = ("P3", "P4") if shared_donor else ("P90", "P91")
    return [
        LUADModalityRecord(
            modality_id="snrna", accession="GSE308103", platform="snrna",
            observation_unit="cell", donor_ids=("P3", "P4"),
        ),
        LUADModalityRecord(
            modality_id="visium", accession="GSE307534", platform="visium",
            observation_unit="spot", donor_ids=visium_donors,
        ),
    ]


def test_allowed_relationship_types_documented():
    assert "same_observation" in ALLOWED_MODALITY_RELATIONSHIP_TYPES
    assert "study_associated_unmatched" in ALLOWED_MODALITY_RELATIONSHIP_TYPES
    assert "unknown" in ALLOWED_MODALITY_RELATIONSHIP_TYPES


def test_shared_donor_yields_same_donor_not_stronger():
    mods, rels = build_luad_modality_manifest(_modalities(shared_donor=True))
    assert len(rels) == 1
    assert rels[0].relationship_type == "same_donor"
    assert rels[0].shared_donor_ids  # genuinely lists the shared donors
    validate_luad_modality_manifest(mods, rels)


def test_no_shared_donor_defaults_to_study_associated_unmatched():
    mods, rels = build_luad_modality_manifest(_modalities(shared_donor=False))
    assert rels[0].relationship_type == "study_associated_unmatched"
    assert rels[0].shared_donor_ids == ()
    validate_luad_modality_manifest(mods, rels)


def test_same_observation_never_from_similar_names():
    # a same_observation relationship across separate accessions is rejected
    mods = _modalities(shared_donor=True)
    bad = LUADModalityRelationship(
        source_modality_id="snrna", target_modality_id="visium",
        relationship_type="same_observation", evidence="names look similar",
    )
    with pytest.raises(CCRTValidationError):
        validate_luad_modality_manifest(mods, [bad])


def test_same_donor_requires_actual_shared_ids():
    mods = _modalities(shared_donor=True)
    bad = LUADModalityRelationship(
        source_modality_id="snrna", target_modality_id="visium",
        relationship_type="same_donor", evidence="claimed",
    )
    with pytest.raises(CCRTValidationError):
        validate_luad_modality_manifest(mods, [bad])


def test_relationship_references_known_modalities():
    mods = _modalities(shared_donor=True)
    bad = LUADModalityRelationship(
        source_modality_id="snrna", target_modality_id="ghost",
        relationship_type="unknown", evidence="x",
    )
    with pytest.raises(CCRTValidationError):
        validate_luad_modality_manifest(mods, [bad])


def test_invalid_relationship_type_rejected():
    with pytest.raises(CCRTValidationError):
        LUADModalityRelationship(
            source_modality_id="a", target_modality_id="b",
            relationship_type="cell_matched", evidence="x",
        )


def test_relationship_requires_distinct_modalities():
    with pytest.raises(CCRTValidationError):
        LUADModalityRelationship(
            source_modality_id="a", target_modality_id="a",
            relationship_type="unknown", evidence="x",
        )


def test_duplicate_modality_id_rejected():
    dup = [
        LUADModalityRecord(modality_id="m", accession="A", platform="p", observation_unit="cell"),
        LUADModalityRecord(modality_id="m", accession="B", platform="q", observation_unit="spot"),
    ]
    with pytest.raises(CCRTValidationError):
        build_luad_modality_manifest(dup)
