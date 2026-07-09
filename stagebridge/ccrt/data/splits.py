"""Split-manifest validation.

A placeholder validator (it does not load files): it inspects an in-memory
manifest mapping and enforces the split-leakage contract — patient/donor-aware
splits, sample-aware only as a fallback, never receiver/spot/cell-level random
splits, and disjoint train/val/test groups.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts.errors import CCRTSplitError
from ..contracts.naming import (
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
)

__all__ = [
    "ALLOWED_SPLIT_STRATEGIES",
    "FORBIDDEN_SPLIT_STRATEGIES",
    "validate_split_manifest",
]

ALLOWED_SPLIT_STRATEGIES = frozenset(
    {
        "patient_aware",
        "donor_aware",
        "sample_aware",
    }
)

FORBIDDEN_SPLIT_STRATEGIES = frozenset(
    {
        "random_receiver",
        "random_spot",
        "random_cell",
        "receiver_level_random",
        "spot_level_random",
        "cell_level_random",
    }
)

#: Group keys that may supply the group IDs listed in the split partitions.
_ALLOWED_GROUP_KEYS = frozenset(
    {"patient_id", "donor_id", "sample_id", "split_group_id"}
)

#: For each strategy, the group keys that are consistent with it. ``split_group_id``
#: is always acceptable as an abstract grouping key.
_STRATEGY_GROUP_KEYS = {
    "patient_aware": frozenset({"patient_id", "split_group_id"}),
    "donor_aware": frozenset({"donor_id", "split_group_id"}),
    "sample_aware": frozenset({"sample_id", "split_group_id"}),
}


def _get_split(manifest: Mapping[str, Any], *names: str) -> Any:
    """Return the first present key among ``names`` (e.g. 'validation'/'val')."""
    for name in names:
        if name in manifest:
            return manifest[name]
    return None


def _as_group_list(value: Any, split_name: str) -> list[Any]:
    """Coerce a split partition to a list of group ids, or raise."""
    if value is None:
        raise CCRTSplitError(f"split manifest: missing '{split_name}' split")
    if isinstance(value, (str, bytes)):
        raise CCRTSplitError(
            f"split manifest: '{split_name}' must be a list of group ids, "
            "not a bare string"
        )
    try:
        return list(value)
    except TypeError as exc:
        raise CCRTSplitError(
            f"split manifest: '{split_name}' must be an iterable of group ids"
        ) from exc


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a split manifest against the split-leakage contract.

    Required keys: ``split_strategy``, ``group_key``, ``train``,
    ``validation`` (or ``val``), ``test``.
    """
    if not isinstance(manifest, Mapping):
        raise CCRTSplitError(
            f"split manifest must be a mapping, got {type(manifest).__name__}"
        )

    # -- forbidden-field hygiene on manifest keys --
    manifest_keys = list(manifest.keys())
    assert_no_forbidden_mechanism_fields(manifest_keys)
    assert_no_model_input_leakage_fields(manifest_keys)

    # -- split_strategy --
    if "split_strategy" not in manifest:
        raise CCRTSplitError("split manifest: missing required 'split_strategy'")
    strategy = manifest["split_strategy"]
    if strategy in FORBIDDEN_SPLIT_STRATEGIES:
        raise CCRTSplitError(
            f"split manifest: forbidden split_strategy '{strategy}'. "
            "Receiver/spot/cell-level random splits are never valid for "
            "biological claims; use patient_aware / donor_aware / sample_aware."
        )
    if strategy not in ALLOWED_SPLIT_STRATEGIES:
        raise CCRTSplitError(
            f"split manifest: unknown split_strategy '{strategy}' "
            f"(allowed: {sorted(ALLOWED_SPLIT_STRATEGIES)})"
        )

    # -- group_key --
    if "group_key" not in manifest:
        raise CCRTSplitError("split manifest: missing required 'group_key'")
    group_key = manifest["group_key"]
    if group_key not in _ALLOWED_GROUP_KEYS:
        raise CCRTSplitError(
            f"split manifest: invalid group_key '{group_key}' "
            f"(allowed: {sorted(_ALLOWED_GROUP_KEYS)})"
        )
    consistent_keys = _STRATEGY_GROUP_KEYS[strategy]
    if group_key not in consistent_keys:
        raise CCRTSplitError(
            f"split manifest: group_key '{group_key}' is inconsistent with "
            f"split_strategy '{strategy}' (expected one of {sorted(consistent_keys)})"
        )

    # -- partitions --
    train = _as_group_list(_get_split(manifest, "train"), "train")
    validation = _as_group_list(
        _get_split(manifest, "validation", "val"), "validation"
    )
    test = _as_group_list(_get_split(manifest, "test"), "test")

    for split_name, groups in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if len(groups) == 0:
            raise CCRTSplitError(
                f"split manifest: '{split_name}' split must be non-empty"
            )

    # -- disjointness (the leakage guard) --
    train_set, val_set, test_set = set(train), set(validation), set(test)
    overlaps = {
        "train∩validation": sorted(train_set & val_set),
        "train∩test": sorted(train_set & test_set),
        "validation∩test": sorted(val_set & test_set),
    }
    offending = {k: v for k, v in overlaps.items() if v}
    if offending:
        raise CCRTSplitError(
            f"split manifest: train/validation/test groups must be disjoint; "
            f"overlaps found: {offending}"
        )
