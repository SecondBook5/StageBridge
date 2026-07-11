"""Canonical CCRT record generation for the LUAD adapter.

Reuses the existing standardized table schemas and record validators (no second
contract). Produces receiver / sender-context / transition-edge / sample records
plus a modality manifest, whose values are entirely source-backed, with
continuous micron distances and individual typed context components preserved.
Feature vectors are attached as extra columns (validated with
``allow_extra=True``); metadata, coordinates, and backend/split/target fields
never enter numeric feature vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...contracts.errors import CCRTValidationError
from ...contracts.naming import (
    TABLE_RECEIVERS,
    TABLE_SAMPLES,
    TABLE_SENDER_CONTEXT,
    TABLE_TRANSITION_EDGES,
)
from ...io.records import validate_records
from ...representations import FeatureSpaceSpec
from .config import LUADAdapterConfig
from .features import LUADFeatureBlock
from .manifests import LUADModalityRecord, LUADModalityRelationship
from .neighborhoods import LUADReceiverContextRecord, LUADSpatialSpot
from .splits import LUADFoldAssignment

__all__ = ["LUADRecordBundle", "build_luad_record_bundle"]


@dataclass(frozen=True)
class LUADRecordBundle:
    receiver_records: tuple[Mapping[str, Any], ...]
    sender_context_records: tuple[Mapping[str, Any], ...]
    transition_edge_records: tuple[Mapping[str, Any], ...]
    sample_records: tuple[Mapping[str, Any], ...]
    modality_records: tuple[Mapping[str, Any], ...]
    modality_relationships: tuple[Mapping[str, Any], ...]
    backend_ids: tuple[str, ...]
    feature_space_specs: tuple[FeatureSpaceSpec, ...]
    fold_assignments: tuple[LUADFoldAssignment, ...]
    provenance: Mapping[str, Any]


def _feature_lookup(block: LUADFeatureBlock) -> dict[str, list[float]]:
    return {
        oid: block.values[i].tolist()
        for i, oid in enumerate(block.observation_ids)
    }


def build_luad_record_bundle(
    *,
    config: LUADAdapterConfig,
    audit,
    ontology,
    receiver_features: LUADFeatureBlock,
    semantic_features: LUADFeatureBlock,
    regulatory_features: LUADFeatureBlock | None,
    receivers: Sequence[LUADSpatialSpot],
    context_records: Sequence[LUADReceiverContextRecord],
    fold_assignments: Sequence[LUADFoldAssignment],
    modalities: Sequence[LUADModalityRecord],
    modality_relationships: Sequence[LUADModalityRelationship],
) -> LUADRecordBundle:
    """Build validated canonical CCRT records from source-backed inputs."""
    bsid = config.biological_system_id
    receiver_feat = _feature_lookup(receiver_features)
    semantic_feat = _feature_lookup(semantic_features)
    regulatory_feat = _feature_lookup(regulatory_features) if regulatory_features else {}

    receiver_by_id = {r.spot_id: r for r in receivers}
    receiver_obs = [
        r for r in receivers if r.canonical_receiver_state_id is not None
    ]

    # -- receiver records (Visium spots / niches) --
    receiver_records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for r in receiver_obs:
        if r.niche_id is None:
            raise CCRTValidationError(
                f"receiver spot '{r.spot_id}' has no niche_id for feature lookup"
            )
        if r.niche_id not in receiver_feat:
            raise CCRTValidationError(
                f"receiver niche '{r.niche_id}' missing from receiver feature block"
            )
        if r.niche_id not in semantic_feat:
            raise CCRTValidationError(
                f"receiver niche '{r.niche_id}' missing from semantic feature block"
            )
        sample_ids.add(r.sample_id)
        rec: dict[str, Any] = {
            "receiver_id": r.spot_id,
            "sample_id": r.sample_id,
            "biological_system_id": bsid,
            "receiver_state_id": r.canonical_receiver_state_id,
            "x_spatial": float(r.x_microns),
            "y_spatial": float(r.y_microns),
            "section_id": r.section_id,
            "donor_id": r.donor_id,
            # feature vectors as extras (validated with allow_extra)
            "receiver_features": receiver_feat[r.niche_id],
            "semantic_features": semantic_feat[r.niche_id],
            # provenance (never a numeric feature)
            "source_annotation": r.source_stage_label or "",
            "assay_platform": r.platform,
            "observation_unit": r.observation_unit,
        }
        if regulatory_features is not None and r.lesion_id is not None:
            if r.lesion_id in regulatory_feat:
                rec["regulatory_features"] = regulatory_feat[r.lesion_id]
        receiver_records.append(rec)
    validate_records(TABLE_RECEIVERS, receiver_records, allow_extra=True)

    # -- sender-context records (individual typed components preserved) --
    sender_records: list[dict[str, Any]] = []
    backend_ids: set[str] = set()
    for c in context_records:
        recv = receiver_by_id.get(c.receiver_id)
        if recv is None:
            raise CCRTValidationError(
                f"context record references unknown receiver '{c.receiver_id}'"
            )
        backend_ids.add(c.backend_id)
        sender_records.append(
            {
                "receiver_id": c.receiver_id,
                "sender_id": c.component_id,
                "sample_id": recv.sample_id,
                "biological_system_id": bsid,
                "sender_context_type_id": c.sender_context_type_id,
                "distance_to_receiver": float(c.distance_to_receiver),
                "sender_context_mask": 1,
                "abundance": float(c.abundance),
                "uncertainty": float(c.uncertainty),
                "section_id": recv.section_id,
                "sender_source": c.backend_id,
            }
        )
    if sender_records:
        validate_records(TABLE_SENDER_CONTEXT, sender_records, allow_extra=True)

    # -- transition edge records --
    edge_records = [
        {
            "transition_edge_id": eid,
            "biological_system_id": bsid,
            "source_state_id": src,
            "target_state_id": tgt,
        }
        for eid, src, tgt in config.transition_edges
    ]
    validate_records(TABLE_TRANSITION_EDGES, edge_records, allow_extra=True)

    # -- sample records --
    sample_records = []
    for sid in sorted(sample_ids):
        donor = next(
            (r.donor_id for r in receiver_obs if r.sample_id == sid and r.donor_id),
            "",
        )
        sample_records.append(
            {
                "sample_id": sid,
                "biological_system_id": bsid,
                "donor_id": donor,
            }
        )
    validate_records(TABLE_SAMPLES, sample_records, allow_extra=True)

    # -- modality manifest (provenance only, never model features) --
    modality_records = tuple(
        {
            "modality_id": m.modality_id,
            "accession": m.accession,
            "platform": m.platform,
            "observation_unit": m.observation_unit,
        }
        for m in modalities
    )
    modality_relationship_records = tuple(
        {
            "source_modality_id": rel.source_modality_id,
            "target_modality_id": rel.target_modality_id,
            "relationship_type": rel.relationship_type,
            "evidence": rel.evidence,
        }
        for rel in modality_relationships
    )

    specs = [receiver_features.spec, semantic_features.spec]
    if regulatory_features is not None:
        specs.append(regulatory_features.spec)

    return LUADRecordBundle(
        receiver_records=tuple(receiver_records),
        sender_context_records=tuple(sender_records),
        transition_edge_records=tuple(edge_records),
        sample_records=tuple(sample_records),
        modality_records=modality_records,
        modality_relationships=modality_relationship_records,
        backend_ids=tuple(sorted(backend_ids)),
        feature_space_specs=tuple(specs),
        fold_assignments=tuple(fold_assignments),
        provenance={
            "biological_system_id": bsid,
            "source_layout_version": config.source_layout_version,
            "num_receivers": len(receiver_records),
            "num_sender_context": len(sender_records),
            "semantic_feature_space_id": semantic_features.spec.feature_space_id,
            "context_feature_space_id": config.context_backend.feature_space_id,
        },
    )
