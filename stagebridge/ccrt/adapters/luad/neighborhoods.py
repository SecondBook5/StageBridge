"""Continuous spatial context for the LUAD adapter.

Builds per-receiver typed context using exact continuous Euclidean distance in
microns between Visium spots. Deconvolved context components are hosted at spots;
a component hosted at the receiver's own spot is at distance 0, and a component
hosted at a nearby spot is at the centroid distance between the two spots.

Matching is restricted to the same (donor, sample, section, platform, BACKEND):
context is never mixed across donors, samples, sections, platforms, or
deconvolution backends. No bins, rings, radial categories, or pre-attention
averaging — individual typed components are preserved. A receiver with no context
component emits zero records (the model's empty-sender element handles it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ...contracts.errors import CCRTValidationError
from .config import LUADNeighborhoodConfig
from .context_components import LUADContextComponent

__all__ = [
    "LUADSpatialSpot",
    "LUADReceiverContextRecord",
    "build_luad_context_neighborhoods",
]


@dataclass(frozen=True)
class LUADSpatialSpot:
    """An observed Visium spot (receiver and/or context host)."""

    spot_id: str
    donor_id: str
    sample_id: str
    section_id: str
    platform: str
    x_microns: float
    y_microns: float
    observation_unit: str = "spot"
    niche_id: str | None = None
    lesion_id: str | None = None
    canonical_receiver_state_id: str | None = None
    source_stage_label: str | None = None


@dataclass(frozen=True)
class LUADReceiverContextRecord:
    """One receiver x typed context component with a continuous micron distance."""

    receiver_id: str
    component_id: str
    backend_id: str
    sender_spot_id: str
    sender_context_type_id: str
    distance_to_receiver: float
    abundance: float
    uncertainty: float
    rank: int


def _receiver_partition_key(spot: LUADSpatialSpot) -> tuple[str, str, str, str]:
    return (spot.donor_id, spot.sample_id, spot.section_id, spot.platform)


def build_luad_context_neighborhoods(
    *,
    receivers: Sequence[LUADSpatialSpot],
    context_components: Sequence[LUADContextComponent],
    spots_by_id: Mapping[str, LUADSpatialSpot],
    config: LUADNeighborhoodConfig,
) -> tuple[LUADReceiverContextRecord, ...]:
    """Build continuous, donor/sample/section/platform/backend-local context.

    ``spots_by_id`` maps every spot id (receiver or context host) to its
    ``LUADSpatialSpot`` so component host coordinates can be resolved. Distances
    are exact continuous Euclidean microns; a same-spot component is distance 0.
    """
    scale = config.coordinate_scale_to_microns

    # group components by their host-spot partition + backend (same-backend only)
    components_by_partition: dict[tuple, list[LUADContextComponent]] = {}
    for comp in context_components:
        host = spots_by_id.get(comp.spot_id)
        if host is None:
            raise CCRTValidationError(
                f"context component '{comp.component_id}' references unknown host "
                f"spot '{comp.spot_id}'"
            )
        key = (*_receiver_partition_key(host), comp.backend_id)
        components_by_partition.setdefault(key, []).append(comp)

    records: list[LUADReceiverContextRecord] = []
    for receiver in receivers:
        rx = torch.tensor(
            [receiver.x_microns * scale, receiver.y_microns * scale],
            dtype=torch.float64,
        )
        # a receiver may draw from any backend, but each partition is a single
        # backend (never mix backends within a partition/receiver context set).
        backends = sorted(
            {
                key[-1]
                for key in components_by_partition
                if key[:4] == _receiver_partition_key(receiver)
            }
        )
        for backend_id in backends:
            key = (*_receiver_partition_key(receiver), backend_id)
            candidates = components_by_partition.get(key, [])
            scored: list[tuple[float, str, str, str, LUADContextComponent]] = []
            for comp in candidates:
                if comp.spot_id == receiver.spot_id:
                    dist = 0.0  # same-spot component
                else:
                    host = spots_by_id[comp.spot_id]
                    sx = torch.tensor(
                        [host.x_microns * scale, host.y_microns * scale],
                        dtype=torch.float64,
                    )
                    dist = float(torch.linalg.vector_norm(rx - sx))
                if config.max_distance is not None and dist > config.max_distance:
                    continue  # candidate truncation only (distance stays continuous)
                scored.append(
                    (dist, comp.spot_id, comp.sender_context_type_id, comp.component_id, comp)
                )

            # deterministic sort by (distance, spot_id, sender_type, component_id)
            scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            scored = scored[: config.max_neighbors]

            for rank, (dist, _sid, _stype, _cid, comp) in enumerate(scored):
                records.append(
                    LUADReceiverContextRecord(
                        receiver_id=receiver.spot_id,
                        component_id=comp.component_id,
                        backend_id=comp.backend_id,
                        sender_spot_id=comp.spot_id,
                        sender_context_type_id=comp.sender_context_type_id,
                        distance_to_receiver=dist,
                        abundance=comp.abundance,
                        uncertainty=comp.uncertainty,
                        rank=rank,
                    )
                )
    return tuple(records)
