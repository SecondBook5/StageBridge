"""``CCRTTableDataset`` — standardized table records to indexed receiver items.

Consumes the in-memory standardized tables (``Sequence[Mapping[str, Any]]``),
validates each against its schema and their cross-table referential integrity,
and exposes a receiver-centered, ordered, index-addressable dataset. Each item
bundles a receiver with its typed sender-context rows and any semantic /
regulatory / edge / sample rows.

No files, no pandas, no disease-specific code. Records are shallow-copied into
read-only mappings so the dataset never mutates its inputs and callers cannot
mutate its internals.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..contracts.errors import CCRTValidationError
from ..contracts.naming import (
    TABLE_RECEIVERS,
    TABLE_REGULATORY_FEATURES,
    TABLE_SAMPLES,
    TABLE_SEMANTIC_FEATURES,
    TABLE_SENDER_CONTEXT,
    TABLE_TRANSITION_EDGES,
)
from ..io.records import validate_records
from .splits import validate_split_manifest

__all__ = ["CCRTTableDataset"]


def _freeze(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only shallow copy of a record."""
    return MappingProxyType(dict(record))


def _require_key(record: Mapping[str, Any], key: str, table: str, index: int) -> Any:
    if key not in record:
        raise CCRTValidationError(
            f"table '{table}': record at index {index} is missing '{key}'"
        )
    value = record[key]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise CCRTValidationError(
            f"table '{table}': record at index {index} has empty '{key}'"
        )
    return value


class CCRTTableDataset:
    """An ordered, validated, receiver-centered view over standardized tables."""

    def __init__(
        self,
        *,
        receivers: Sequence[Mapping[str, Any]],
        sender_context: Sequence[Mapping[str, Any]],
        semantic_features: Sequence[Mapping[str, Any]] | None = None,
        regulatory_features: Sequence[Mapping[str, Any]] | None = None,
        transition_edges: Sequence[Mapping[str, Any]] | None = None,
        samples: Sequence[Mapping[str, Any]] | None = None,
        split_manifest: Mapping[str, Any] | None = None,
        allow_extra_fields: bool = False,
    ) -> None:
        # -- schema validation (records-level) --
        validate_records(TABLE_RECEIVERS, receivers, allow_extra=allow_extra_fields)
        validate_records(
            TABLE_SENDER_CONTEXT, sender_context, allow_extra=allow_extra_fields
        )
        if semantic_features is not None:
            validate_records(
                TABLE_SEMANTIC_FEATURES,
                semantic_features,
                allow_extra=allow_extra_fields,
            )
        if regulatory_features is not None:
            validate_records(
                TABLE_REGULATORY_FEATURES,
                regulatory_features,
                allow_extra=allow_extra_fields,
            )
        if transition_edges is not None:
            validate_records(
                TABLE_TRANSITION_EDGES,
                transition_edges,
                allow_extra=allow_extra_fields,
            )
        if samples is not None:
            validate_records(TABLE_SAMPLES, samples, allow_extra=allow_extra_fields)
        if split_manifest is not None:
            validate_split_manifest(split_manifest)

        # -- index receivers (unique id, order-preserving) --
        self._receiver_ids: list[str] = []
        self._receivers_by_id: dict[str, Mapping[str, Any]] = {}
        for index, record in enumerate(receivers):
            rid = _require_key(record, "receiver_id", TABLE_RECEIVERS, index)
            if rid in self._receivers_by_id:
                raise CCRTValidationError(
                    f"table '{TABLE_RECEIVERS}': duplicate receiver_id '{rid}'"
                )
            frozen = _freeze(record)
            self._receivers_by_id[rid] = frozen
            self._receiver_ids.append(rid)

        # -- index transition edges (by id) --
        self._transition_edges_by_id: dict[str, Mapping[str, Any]] = {}
        if transition_edges is not None:
            for index, record in enumerate(transition_edges):
                edge_id = _require_key(
                    record, "transition_edge_id", TABLE_TRANSITION_EDGES, index
                )
                if edge_id in self._transition_edges_by_id:
                    raise CCRTValidationError(
                        f"table '{TABLE_TRANSITION_EDGES}': duplicate "
                        f"transition_edge_id '{edge_id}'"
                    )
                self._transition_edges_by_id[edge_id] = _freeze(record)

        # -- index samples (by id) --
        self._samples_by_id: dict[str, Mapping[str, Any]] = {}
        if samples is not None:
            for index, record in enumerate(samples):
                sample_id = _require_key(record, "sample_id", TABLE_SAMPLES, index)
                if sample_id in self._samples_by_id:
                    raise CCRTValidationError(
                        f"table '{TABLE_SAMPLES}': duplicate sample_id '{sample_id}'"
                    )
                self._samples_by_id[sample_id] = _freeze(record)

        # -- group sender_context by receiver_id (order-preserving) --
        self._sender_context_by_receiver_id: dict[str, list[Mapping[str, Any]]] = {
            rid: [] for rid in self._receiver_ids
        }
        for index, record in enumerate(sender_context):
            rid = _require_key(record, "receiver_id", TABLE_SENDER_CONTEXT, index)
            if rid not in self._receivers_by_id:
                raise CCRTValidationError(
                    f"table '{TABLE_SENDER_CONTEXT}': record at index {index} "
                    f"references unknown receiver_id '{rid}'"
                )
            self._sender_context_by_receiver_id[rid].append(_freeze(record))

        # -- index semantic features by receiver_id --
        self._semantic_by_receiver_id: dict[str, Mapping[str, Any]] = {}
        if semantic_features is not None:
            self._index_feature_table(
                semantic_features,
                TABLE_SEMANTIC_FEATURES,
                self._semantic_by_receiver_id,
            )

        # -- index regulatory features by receiver_id --
        self._regulatory_by_receiver_id: dict[str, Mapping[str, Any]] = {}
        if regulatory_features is not None:
            self._index_feature_table(
                regulatory_features,
                TABLE_REGULATORY_FEATURES,
                self._regulatory_by_receiver_id,
            )

        # -- referential integrity: receiver -> transition_edges --
        if transition_edges is not None:
            for rid in self._receiver_ids:
                receiver = self._receivers_by_id[rid]
                edge_id = receiver.get("transition_edge_id")
                if edge_id is not None and edge_id not in self._transition_edges_by_id:
                    raise CCRTValidationError(
                        f"table '{TABLE_RECEIVERS}': receiver '{rid}' references "
                        f"transition_edge_id '{edge_id}' absent from "
                        f"'{TABLE_TRANSITION_EDGES}'"
                    )

        # -- referential integrity: sample_id in receivers + sender_context --
        if samples is not None:
            for rid in self._receiver_ids:
                receiver = self._receivers_by_id[rid]
                sample_id = receiver.get("sample_id")
                if sample_id is not None and sample_id not in self._samples_by_id:
                    raise CCRTValidationError(
                        f"table '{TABLE_RECEIVERS}': receiver '{rid}' references "
                        f"sample_id '{sample_id}' absent from '{TABLE_SAMPLES}'"
                    )
            for rid, rows in self._sender_context_by_receiver_id.items():
                for row in rows:
                    sample_id = row.get("sample_id")
                    if sample_id is not None and sample_id not in self._samples_by_id:
                        raise CCRTValidationError(
                            f"table '{TABLE_SENDER_CONTEXT}': a sender row for "
                            f"receiver '{rid}' references sample_id '{sample_id}' "
                            f"absent from '{TABLE_SAMPLES}'"
                        )

        # -- freeze grouped sender context into tuples --
        self._sender_context_tuples: dict[str, tuple[Mapping[str, Any], ...]] = {
            rid: tuple(rows)
            for rid, rows in self._sender_context_by_receiver_id.items()
        }

        self._split_manifest = (
            _freeze(split_manifest) if split_manifest is not None else None
        )

    # -- internal helpers --------------------------------------------------

    def _index_feature_table(
        self,
        records: Sequence[Mapping[str, Any]],
        table: str,
        into: dict[str, Mapping[str, Any]],
    ) -> None:
        """Index a per-receiver feature table by receiver_id (unique per receiver)."""
        for index, record in enumerate(records):
            rid = _require_key(record, "receiver_id", table, index)
            if rid not in self._receivers_by_id:
                raise CCRTValidationError(
                    f"table '{table}': record at index {index} references "
                    f"unknown receiver_id '{rid}'"
                )
            if rid in into:
                raise CCRTValidationError(
                    f"table '{table}': duplicate row for receiver_id '{rid}'"
                )
            into[rid] = _freeze(record)

    # -- public API --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._receiver_ids)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        if not isinstance(index, int):
            raise TypeError(f"index must be int, got {type(index).__name__}")
        try:
            rid = self._receiver_ids[index]
        except IndexError:
            raise IndexError(
                f"dataset index {index} out of range (len={len(self)})"
            ) from None
        return self._assemble_item(rid)

    def receiver_ids(self) -> tuple[str, ...]:
        """Receiver ids in input order."""
        return tuple(self._receiver_ids)

    def get_by_receiver_id(self, receiver_id: str) -> Mapping[str, Any]:
        """Return the assembled item for a given receiver id."""
        if receiver_id not in self._receivers_by_id:
            raise CCRTValidationError(
                f"unknown receiver_id '{receiver_id}'"
            )
        return self._assemble_item(receiver_id)

    @property
    def split_manifest(self) -> Mapping[str, Any] | None:
        return self._split_manifest

    def _assemble_item(self, rid: str) -> Mapping[str, Any]:
        receiver = self._receivers_by_id[rid]
        edge_id = receiver.get("transition_edge_id")
        sample_id = receiver.get("sample_id")
        return MappingProxyType(
            {
                "receiver": receiver,
                "sender_context": self._sender_context_tuples[rid],
                "semantic_features": self._semantic_by_receiver_id.get(rid),
                "regulatory_features": self._regulatory_by_receiver_id.get(rid),
                "transition_edge": (
                    self._transition_edges_by_id.get(edge_id)
                    if edge_id is not None
                    else None
                ),
                "sample": (
                    self._samples_by_id.get(sample_id)
                    if sample_id is not None
                    else None
                ),
            }
        )
