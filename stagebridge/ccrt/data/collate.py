"""Collate dataset items into a padded, validated ``CCRTBatch``.

Takes the items produced by ``CCRTTableDataset.__getitem__`` and assembles the
receiver-centered batch tensors as nested Python lists (no numpy/torch):

* receiver / semantic / regulatory feature matrices ``[B, D]``,
* the typed sender-context payload padded to ``[B, K, D_S]`` with a boolean
  ``sender_mask``, continuous ``distance_to_receiver``, and optional
  ``uncertainty``,
* grammar-conditioning ids pulled from each receiver.

Padding uses zero vectors for absent senders; ``sender_mask`` is the single
source of truth for which slots are real. The returned batch is validated before
it is returned.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts.errors import CCRTValidationError
from ..contracts.naming import (
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
)
from .batch import CCRTBatch

__all__ = [
    "collate_ccrt_records",
    "is_numeric_sequence",
    "coerce_numeric_vector",
]


def is_numeric_sequence(value: Any) -> bool:
    """True if ``value`` is a non-string sequence of real numbers (non-empty)."""
    if isinstance(value, (str, bytes, Mapping)):
        return False
    if not isinstance(value, Sequence):
        return False
    if len(value) == 0:
        return False
    for item in value:
        # bool is a subclass of int; exclude it to avoid silent True/False rows.
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
    return True


def coerce_numeric_vector(value: Any, context: str) -> list[float]:
    """Coerce a numeric sequence to ``list[float]`` or raise ``CCRTValidationError``."""
    if not is_numeric_sequence(value):
        raise CCRTValidationError(
            f"{context}: expected a non-empty numeric sequence, got {value!r}"
        )
    return [float(x) for x in value]


def _coerce_scalar(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CCRTValidationError(f"{context}: expected a numeric scalar, got {value!r}")
    return float(value)


def _check_forbidden_keys(mapping: Mapping[str, Any], context: str) -> None:
    keys = list(mapping.keys())
    assert_no_forbidden_mechanism_fields(keys)
    assert_no_model_input_leakage_fields(keys)


def _extract_optional_feature(
    item: Mapping[str, Any],
    table_slot: str,
    feature_key: str,
    context: str,
) -> list[float] | None:
    """Extract a feature vector, preferring the table row then the receiver.

    Returns None only if the key is absent in both places.
    """
    row = item.get(table_slot)
    if isinstance(row, Mapping) and feature_key in row:
        return coerce_numeric_vector(row[feature_key], f"{context} ({table_slot})")
    receiver = item.get("receiver")
    if isinstance(receiver, Mapping) and feature_key in receiver:
        return coerce_numeric_vector(receiver[feature_key], f"{context} (receiver)")
    return None


def _require_uniform_length(
    vectors: Sequence[Sequence[float]], name: str
) -> int:
    """Return the shared length of all vectors or raise if they differ."""
    length = len(vectors[0])
    for i, vec in enumerate(vectors):
        if len(vec) != length:
            raise CCRTValidationError(
                f"{name}: inconsistent feature length; row 0 has {length} but "
                f"row {i} has {len(vec)}"
            )
    return length


def collate_ccrt_records(
    items: Sequence[Mapping[str, Any]],
    *,
    receiver_feature_key: str,
    sender_feature_key: str,
    semantic_feature_key: str | None = None,
    regulatory_feature_key: str | None = None,
    require_transition_edge: bool = True,
) -> CCRTBatch:
    """Collate dataset items into a padded, validated ``CCRTBatch``."""
    if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
        raise CCRTValidationError("collate: items must be a sequence of item mappings")
    if len(items) == 0:
        raise CCRTValidationError("collate: items must be non-empty")

    batch_size = len(items)

    # ---- validate items, gather forbidden-key hygiene ----
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise CCRTValidationError(f"collate: item {idx} must be a mapping")
        _check_forbidden_keys(item, f"collate: item {idx}")
        receiver = item.get("receiver")
        if not isinstance(receiver, Mapping):
            raise CCRTValidationError(
                f"collate: item {idx} must have a 'receiver' mapping"
            )
        _check_forbidden_keys(receiver, f"collate: item {idx} receiver")
        for slot in ("semantic_features", "regulatory_features"):
            row = item.get(slot)
            if isinstance(row, Mapping):
                _check_forbidden_keys(row, f"collate: item {idx} {slot}")
        for s_idx, srow in enumerate(item.get("sender_context", ())):
            if not isinstance(srow, Mapping):
                raise CCRTValidationError(
                    f"collate: item {idx} sender row {s_idx} must be a mapping"
                )
            _check_forbidden_keys(srow, f"collate: item {idx} sender {s_idx}")

    # ---- receiver features [B, D_R] ----
    receiver_features: list[list[float]] = []
    for idx, item in enumerate(items):
        receiver = item["receiver"]
        if receiver_feature_key not in receiver:
            raise CCRTValidationError(
                f"collate: item {idx} receiver is missing feature key "
                f"'{receiver_feature_key}'"
            )
        receiver_features.append(
            coerce_numeric_vector(
                receiver[receiver_feature_key],
                f"collate: item {idx} receiver_features",
            )
        )
    _require_uniform_length(receiver_features, "receiver_features")

    # ---- grammar conditioning ids ----
    biological_system_id: list[str] = []
    transition_edge_id: list[str] = []
    receiver_state_id: list[Any] = []
    have_receiver_state = True
    for idx, item in enumerate(items):
        receiver = item["receiver"]
        if "biological_system_id" not in receiver:
            raise CCRTValidationError(
                f"collate: item {idx} receiver is missing 'biological_system_id'"
            )
        biological_system_id.append(receiver["biological_system_id"])

        edge = receiver.get("transition_edge_id")
        if edge is None or (isinstance(edge, str) and not edge.strip()):
            if require_transition_edge:
                raise CCRTValidationError(
                    f"collate: item {idx} receiver has no 'transition_edge_id' "
                    "but require_transition_edge=True"
                )
            edge = ""
        transition_edge_id.append(edge)

        if "receiver_state_id" in receiver:
            receiver_state_id.append(receiver["receiver_state_id"])
        else:
            have_receiver_state = False

    # ---- sender context: determine K, gather real rows ----
    per_item_senders: list[tuple[Mapping[str, Any], ...]] = [
        tuple(item.get("sender_context", ())) for item in items
    ]
    max_k = max((len(rows) for rows in per_item_senders), default=0)
    # If no receiver has any sender, still produce a K=1 all-padding axis.
    k = max_k if max_k > 0 else 1

    # Infer D_S from the first real sender feature vector anywhere in the batch.
    sender_dim: int | None = None
    for rows in per_item_senders:
        for srow in rows:
            if sender_feature_key not in srow:
                raise CCRTValidationError(
                    f"collate: a sender row is missing feature key "
                    f"'{sender_feature_key}'"
                )
            vec = coerce_numeric_vector(
                srow[sender_feature_key], "collate: sender_features"
            )
            if sender_dim is None:
                sender_dim = len(vec)
            elif len(vec) != sender_dim:
                raise CCRTValidationError(
                    f"collate: inconsistent sender feature length; expected "
                    f"{sender_dim}, got {len(vec)}"
                )
    if sender_dim is None:
        # No real senders anywhere: pad vectors are zero-length would be invalid,
        # so use dimension 1 for the padded-only axis.
        sender_dim = 1

    # Decide whether uncertainty is included: only if EVERY real sender row has it.
    include_uncertainty = True
    real_sender_count = 0
    for rows in per_item_senders:
        for srow in rows:
            real_sender_count += 1
            if "uncertainty" not in srow:
                include_uncertainty = False
    if real_sender_count == 0:
        include_uncertainty = False

    # ---- build padded sender tensors ----
    sender_features: list[list[list[float]]] = []
    sender_mask: list[list[int]] = []
    distance_to_receiver: list[list[float]] = []
    uncertainty: list[list[float]] | None = [] if include_uncertainty else None

    zero_vec = [0.0] * sender_dim
    for idx, rows in enumerate(per_item_senders):
        feat_row: list[list[float]] = []
        mask_row: list[int] = []
        dist_row: list[float] = []
        unc_row: list[float] = []
        for srow in rows:
            feat_row.append(
                coerce_numeric_vector(
                    srow[sender_feature_key],
                    f"collate: item {idx} sender_features",
                )
            )
            mask_row.append(1)
            if "distance_to_receiver" not in srow:
                raise CCRTValidationError(
                    f"collate: item {idx} sender row is missing "
                    "'distance_to_receiver'"
                )
            dist_row.append(
                _coerce_scalar(
                    srow["distance_to_receiver"],
                    f"collate: item {idx} distance_to_receiver",
                )
            )
            if include_uncertainty:
                unc_row.append(
                    _coerce_scalar(
                        srow["uncertainty"], f"collate: item {idx} uncertainty"
                    )
                )
        # pad to k
        while len(feat_row) < k:
            feat_row.append(list(zero_vec))
            mask_row.append(0)
            dist_row.append(0.0)
            if include_uncertainty:
                unc_row.append(0.0)

        sender_features.append(feat_row)
        sender_mask.append(mask_row)
        distance_to_receiver.append(dist_row)
        if uncertainty is not None:
            uncertainty.append(unc_row)

    # ---- optional receiver-level feature matrices ----
    semantic_features: list[list[float]] | None = None
    if semantic_feature_key is not None:
        collected: list[list[float]] = []
        for idx, item in enumerate(items):
            vec = _extract_optional_feature(
                item,
                "semantic_features",
                semantic_feature_key,
                f"collate: item {idx} semantic_features",
            )
            if vec is None:
                raise CCRTValidationError(
                    f"collate: item {idx} has no semantic feature under "
                    f"'{semantic_feature_key}'"
                )
            collected.append(vec)
        _require_uniform_length(collected, "semantic_features")
        semantic_features = collected

    regulatory_features: list[list[float]] | None = None
    if regulatory_feature_key is not None:
        collected = []
        for idx, item in enumerate(items):
            vec = _extract_optional_feature(
                item,
                "regulatory_features",
                regulatory_feature_key,
                f"collate: item {idx} regulatory_features",
            )
            if vec is None:
                raise CCRTValidationError(
                    f"collate: item {idx} has no regulatory feature under "
                    f"'{regulatory_feature_key}'"
                )
            collected.append(vec)
        _require_uniform_length(collected, "regulatory_features")
        regulatory_features = collected

    batch = CCRTBatch(
        receiver_features=receiver_features,
        sender_features=sender_features,
        sender_mask=sender_mask,
        distance_to_receiver=distance_to_receiver,
        biological_system_id=biological_system_id,
        transition_edge_id=transition_edge_id,
        receiver_state_id=receiver_state_id if have_receiver_state else None,
        uncertainty=uncertainty,
        semantic_features=semantic_features,
        regulatory_features=regulatory_features,
    )
    batch.validate()
    return batch
