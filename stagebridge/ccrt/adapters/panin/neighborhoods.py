"""Continuous spatial sender neighborhoods for the PanIN adapter.

Builds per-receiver typed sender neighborhoods using exact continuous Euclidean
distance in microns, restricted to the same (sample, section, platform,
observation-unit). No bins, rings, radial categories, or pre-attention
averaging — individual sender elements are preserved. A receiver with no real
sender emits zero neighbor records (the model's empty-sender element handles it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ...contracts.errors import CCRTValidationError
from .config import PanINNeighborhoodConfig

__all__ = [
    "PanINSpatialObservation",
    "PanINNeighborRecord",
    "build_continuous_sender_neighborhoods",
]


@dataclass(frozen=True)
class PanINSpatialObservation:
    observation_id: str
    donor_id: str | None
    sample_id: str
    section_id: str
    platform: str
    observation_unit: str
    x_microns: float
    y_microns: float
    z_microns: float | None = None
    source_annotation: str | None = None
    canonical_context_type_id: str | None = None
    canonical_receiver_state_id: str | None = None


@dataclass(frozen=True)
class PanINNeighborRecord:
    receiver_id: str
    sender_id: str
    sender_context_type_id: str
    distance_to_receiver: float
    uncertainty: float
    rank: int


def _partition_key(o: PanINSpatialObservation) -> tuple[str, str, str, str]:
    return (o.sample_id, o.section_id, o.platform, o.observation_unit)


def build_continuous_sender_neighborhoods(
    *,
    receivers: Sequence[PanINSpatialObservation],
    candidate_senders: Sequence[PanINSpatialObservation],
    config: PanINNeighborhoodConfig,
) -> tuple[PanINNeighborRecord, ...]:
    """Build continuous, section/platform-local sender neighborhoods."""
    # group candidate senders by partition for same-partition-only matching
    senders_by_partition: dict[tuple, list[PanINSpatialObservation]] = {}
    for s in candidate_senders:
        if s.canonical_context_type_id is None:
            continue  # only source-typed senders participate
        senders_by_partition.setdefault(_partition_key(s), []).append(s)

    records: list[PanINNeighborRecord] = []
    for receiver in receivers:
        part = _partition_key(receiver)
        candidates = senders_by_partition.get(part, [])
        # exclude self
        candidates = [s for s in candidates if s.observation_id != receiver.observation_id]
        if not candidates:
            continue  # zero neighbor records; empty-sender handles this later

        rx = torch.tensor(
            [receiver.x_microns, receiver.y_microns], dtype=torch.float64
        )
        scored: list[tuple[float, str, PanINSpatialObservation]] = []
        for s in candidates:
            sx = torch.tensor([s.x_microns, s.y_microns], dtype=torch.float64)
            dist = float(torch.linalg.vector_norm(rx - sx))
            if config.max_distance is not None and dist > config.max_distance:
                continue  # candidate truncation only (distance stays continuous)
            scored.append((dist, s.observation_id, s))

        # deterministic sort by (distance, sender_id)
        scored.sort(key=lambda t: (t[0], t[1]))
        scored = scored[: config.max_neighbors]

        for rank, (dist, sid, s) in enumerate(scored):
            records.append(
                PanINNeighborRecord(
                    receiver_id=receiver.observation_id,
                    sender_id=sid,
                    sender_context_type_id=s.canonical_context_type_id,
                    distance_to_receiver=dist,
                    uncertainty=0.0,  # observed cell-level sender; documented in contract
                    rank=rank,
                )
            )
    return tuple(records)
