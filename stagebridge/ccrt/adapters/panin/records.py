"""Canonical CCRT record generation for the PanIN adapter.

Reuses the existing standardized table schemas and record validators (no second
contract). Produces receiver / sender-context / transition-edge / sample records
whose values are entirely source-backed, with continuous micron distances and
individual sender elements preserved. Feature vectors are attached as extra
columns (validated with ``allow_extra=True``); metadata and coordinates never
enter numeric feature vectors.
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
from .config import PanINAdapterConfig
from .features import PanINFeatureBlock
from .neighborhoods import PanINNeighborRecord, PanINSpatialObservation
from .splits import PanINFoldAssignment

__all__ = ["PanINRecordBundle", "build_panin_record_bundle"]


@dataclass(frozen=True)
class PanINRecordBundle:
    receiver_records: tuple[Mapping[str, Any], ...]
    sender_context_records: tuple[Mapping[str, Any], ...]
    transition_edge_records: tuple[Mapping[str, Any], ...]
    sample_records: tuple[Mapping[str, Any], ...]
    feature_space_specs: tuple[FeatureSpaceSpec, ...]
    fold_assignments: tuple[PanINFoldAssignment, ...]
    provenance: Mapping[str, Any]


def _feature_lookup(block: PanINFeatureBlock) -> dict[str, list[float]]:
    return {
        oid: block.values[i].tolist()
        for i, oid in enumerate(block.observation_ids)
    }


def build_panin_record_bundle(
    *,
    config: PanINAdapterConfig,
    audit,
    ontology,
    receiver_features: PanINFeatureBlock,
    sender_features: PanINFeatureBlock,
    semantic_features: PanINFeatureBlock,
    regulatory_features: PanINFeatureBlock | None,
    spatial_observations: Sequence[PanINSpatialObservation],
    neighbors: Sequence[PanINNeighborRecord],
    fold_assignments: Sequence[PanINFoldAssignment],
) -> PanINRecordBundle:
    """Build validated canonical CCRT records from source-backed inputs."""
    bsid = config.biological_system_id
    receiver_feat = _feature_lookup(receiver_features)
    sender_feat = _feature_lookup(sender_features)
    semantic_feat = _feature_lookup(semantic_features)

    obs_by_id = {o.observation_id: o for o in spatial_observations}
    receiver_obs = [
        o for o in spatial_observations if o.canonical_receiver_state_id is not None
    ]

    # -- receiver records --
    receiver_records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for o in receiver_obs:
        if o.observation_id not in receiver_feat:
            raise CCRTValidationError(
                f"receiver '{o.observation_id}' missing from receiver feature block"
            )
        if o.observation_id not in semantic_feat:
            raise CCRTValidationError(
                f"receiver '{o.observation_id}' missing from semantic feature block"
            )
        sample_ids.add(o.sample_id)
        receiver_records.append(
            {
                "receiver_id": o.observation_id,
                "sample_id": o.sample_id,
                "biological_system_id": bsid,
                "receiver_state_id": o.canonical_receiver_state_id,
                "x_spatial": float(o.x_microns),
                "y_spatial": float(o.y_microns),
                "section_id": o.section_id,
                "donor_id": o.donor_id if o.donor_id is not None else "",
                # feature vectors as extras (validated with allow_extra)
                "receiver_features": receiver_feat[o.observation_id],
                "semantic_features": semantic_feat[o.observation_id],
                # provenance (never a numeric feature)
                "source_annotation": o.source_annotation or "",
                "assay_platform": o.platform,
                "observation_unit": o.observation_unit,
            }
        )
    validate_records(TABLE_RECEIVERS, receiver_records, allow_extra=True)

    # -- sender-context records (individual sender elements preserved) --
    sender_records: list[dict[str, Any]] = []
    for n in neighbors:
        recv = obs_by_id.get(n.receiver_id)
        send = obs_by_id.get(n.sender_id)
        if recv is None or send is None:
            raise CCRTValidationError(
                f"neighbor references unknown observation ({n.receiver_id} / {n.sender_id})"
            )
        if recv.sample_id != send.sample_id or recv.section_id != send.section_id:
            raise CCRTValidationError("cross-sample/section neighbor detected")
        if recv.platform != send.platform:
            raise CCRTValidationError("cross-platform neighbor detected")
        rec = {
            "receiver_id": n.receiver_id,
            "sender_id": n.sender_id,
            "sample_id": recv.sample_id,
            "biological_system_id": bsid,
            "sender_context_type_id": n.sender_context_type_id,
            "distance_to_receiver": float(n.distance_to_receiver),
            "sender_context_mask": 1,
            "uncertainty": float(n.uncertainty),
            "section_id": recv.section_id,
        }
        if n.sender_id in sender_feat:
            rec["sender_features"] = sender_feat[n.sender_id]
        sender_records.append(rec)
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
            (o.donor_id for o in receiver_obs if o.sample_id == sid and o.donor_id),
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

    specs = [receiver_features.spec, sender_features.spec, semantic_features.spec]
    if regulatory_features is not None:
        specs.append(regulatory_features.spec)

    return PanINRecordBundle(
        receiver_records=tuple(receiver_records),
        sender_context_records=tuple(sender_records),
        transition_edge_records=tuple(edge_records),
        sample_records=tuple(sample_records),
        feature_space_specs=tuple(specs),
        fold_assignments=tuple(fold_assignments),
        provenance={
            "biological_system_id": bsid,
            "source_layout_version": config.source_layout_version,
            "num_receivers": len(receiver_records),
            "num_sender_context": len(sender_records),
            "semantic_feature_space_id": semantic_features.spec.feature_space_id,
        },
    )
