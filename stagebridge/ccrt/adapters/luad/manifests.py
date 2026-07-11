"""LUAD modality manifest.

Records the verified relationship between LUAD modalities (snRNA reference and
Visium spatial) at the level supported by the source, and never upgrades a
relationship's strength from similar names. snRNA (GSE308103) and Visium
(GSE307534) are separate accessions: they share patients but are NOT cell
matched, so the strongest honest relationship between a snRNA sample and a Visium
sample is ``same_donor`` (when a patient id genuinely appears in both) or
``study_associated_unmatched`` otherwise — never ``same_observation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...contracts.errors import CCRTValidationError

__all__ = [
    "ALLOWED_MODALITY_RELATIONSHIP_TYPES",
    "LUADModalityRecord",
    "LUADModalityRelationship",
    "build_luad_modality_manifest",
    "validate_luad_modality_manifest",
]

#: Allowed modality relationship types, ordered from strongest to weakest.
#: The adapter never promotes a weaker relationship to a stronger one on the
#: basis of similar sample/patient names.
ALLOWED_MODALITY_RELATIONSHIP_TYPES = (
    "same_observation",
    "same_section",
    "same_sample",
    "same_donor",
    "study_associated_unmatched",
    "unknown",
)


@dataclass(frozen=True)
class LUADModalityRecord:
    """One modality dataset (a GEO accession) and its observation unit."""

    modality_id: str
    accession: str
    platform: str
    observation_unit: str
    donor_ids: tuple[str, ...] = ()
    sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("modality_id", "accession", "platform", "observation_unit"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise CCRTValidationError(
                    f"LUADModalityRecord.{name} must be a non-empty string"
                )
        object.__setattr__(self, "donor_ids", tuple(self.donor_ids))
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))


@dataclass(frozen=True)
class LUADModalityRelationship:
    """A verified relationship between two modalities (source and target ids)."""

    source_modality_id: str
    target_modality_id: str
    relationship_type: str
    evidence: str
    shared_donor_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_modality_id", "target_modality_id", "evidence"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise CCRTValidationError(
                    f"LUADModalityRelationship.{name} must be a non-empty string"
                )
        if self.relationship_type not in ALLOWED_MODALITY_RELATIONSHIP_TYPES:
            raise CCRTValidationError(
                f"relationship_type '{self.relationship_type}' invalid; allowed: "
                f"{list(ALLOWED_MODALITY_RELATIONSHIP_TYPES)}"
            )
        if self.source_modality_id == self.target_modality_id:
            raise CCRTValidationError(
                "modality relationship must relate two distinct modalities"
            )
        object.__setattr__(self, "shared_donor_ids", tuple(self.shared_donor_ids))


def build_luad_modality_manifest(
    modalities: Sequence[LUADModalityRecord],
    *,
    default_cross_accession_relationship: str = "study_associated_unmatched",
) -> tuple[tuple[LUADModalityRecord, ...], tuple[LUADModalityRelationship, ...]]:
    """Build the modality manifest, computing honest cross-accession relationships.

    For each ordered distinct pair of modalities, the relationship is
    ``same_donor`` only if the two modalities genuinely share at least one donor
    id; otherwise it is the (weaker) ``default_cross_accession_relationship``.
    Relationships are NEVER upgraded to ``same_observation``/``same_sample`` from
    similar names — cell-matching is not asserted for separate accessions.
    """
    if default_cross_accession_relationship not in ALLOWED_MODALITY_RELATIONSHIP_TYPES:
        raise CCRTValidationError(
            f"default relationship '{default_cross_accession_relationship}' invalid"
        )
    records = tuple(modalities)
    ids = [m.modality_id for m in records]
    if len(set(ids)) != len(ids):
        raise CCRTValidationError("duplicate modality_id in manifest")

    relationships: list[LUADModalityRelationship] = []
    for i, src in enumerate(records):
        for tgt in records[i + 1 :]:
            shared = tuple(sorted(set(src.donor_ids) & set(tgt.donor_ids)))
            if shared:
                rel_type = "same_donor"
                evidence = (
                    f"shared patient/donor id(s) {list(shared)} across "
                    f"accessions {src.accession} and {tgt.accession}; NOT cell-matched"
                )
            else:
                rel_type = default_cross_accession_relationship
                evidence = (
                    f"separate accessions {src.accession} and {tgt.accession}; no "
                    "shared donor established (not cell-matched)"
                )
            relationships.append(
                LUADModalityRelationship(
                    source_modality_id=src.modality_id,
                    target_modality_id=tgt.modality_id,
                    relationship_type=rel_type,
                    evidence=evidence,
                    shared_donor_ids=shared,
                )
            )
    return records, tuple(relationships)


def validate_luad_modality_manifest(
    modalities: Sequence[LUADModalityRecord],
    relationships: Sequence[LUADModalityRelationship],
) -> None:
    """Assert the manifest is internally consistent and honest.

    * every relationship references known modalities;
    * a ``same_donor`` relationship must actually list shared donor ids;
    * a ``same_observation`` relationship is only permitted within a single
      accession (cross-accession cell matching is never asserted).
    """
    known = {m.modality_id: m for m in modalities}
    if len(known) != len(list(modalities)):
        raise CCRTValidationError("duplicate modality_id in manifest")

    for rel in relationships:
        if rel.source_modality_id not in known:
            raise CCRTValidationError(
                f"relationship references unknown source modality "
                f"'{rel.source_modality_id}'"
            )
        if rel.target_modality_id not in known:
            raise CCRTValidationError(
                f"relationship references unknown target modality "
                f"'{rel.target_modality_id}'"
            )
        src = known[rel.source_modality_id]
        tgt = known[rel.target_modality_id]

        if rel.relationship_type == "same_donor" and not rel.shared_donor_ids:
            raise CCRTValidationError(
                f"same_donor relationship {rel.source_modality_id}->"
                f"{rel.target_modality_id} lists no shared donor ids (name "
                "similarity is not evidence of a shared donor)"
            )

        if rel.relationship_type in ("same_observation", "same_section", "same_sample"):
            if src.accession != tgt.accession:
                raise CCRTValidationError(
                    f"relationship '{rel.relationship_type}' asserted across separate "
                    f"accessions {src.accession} and {tgt.accession}; separate LUAD "
                    "accessions are not cell/section/sample matched"
                )

        # sanity: shared donor ids must be a subset of both modalities' donors
        if rel.shared_donor_ids:
            not_in = set(rel.shared_donor_ids) - (set(src.donor_ids) & set(tgt.donor_ids))
            if not_in:
                raise CCRTValidationError(
                    f"relationship claims shared donor ids {sorted(not_in)} not "
                    "actually present in both modalities"
                )
