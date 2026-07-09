"""Tests for split-manifest validation (split-leakage guard)."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.contracts.errors import CCRTLeakageError, CCRTSplitError
from stagebridge.ccrt.data import validate_split_manifest


def base_manifest(**overrides):
    manifest = {
        "split_strategy": "patient_aware",
        "group_key": "patient_id",
        "train": ["p1", "p2"],
        "validation": ["p3"],
        "test": ["p4"],
    }
    manifest.update(overrides)
    return manifest


def test_patient_aware_split_validates():
    validate_split_manifest(base_manifest())


def test_donor_aware_split_validates():
    validate_split_manifest(
        base_manifest(
            split_strategy="donor_aware",
            group_key="donor_id",
            train=["d1"],
            validation=["d2"],
            test=["d3"],
        )
    )


def test_sample_aware_split_validates():
    validate_split_manifest(
        base_manifest(
            split_strategy="sample_aware",
            group_key="sample_id",
            train=["s1"],
            validation=["s2"],
            test=["s3"],
        )
    )


def test_val_alias_accepted():
    m = base_manifest()
    del m["validation"]
    m["val"] = ["p3"]
    validate_split_manifest(m)


@pytest.mark.parametrize(
    "strategy", ["random_receiver", "random_spot", "random_cell"]
)
def test_forbidden_random_strategies_fail(strategy):
    with pytest.raises(CCRTSplitError):
        validate_split_manifest(base_manifest(split_strategy=strategy))


def test_unknown_strategy_fails():
    with pytest.raises(CCRTSplitError):
        validate_split_manifest(base_manifest(split_strategy="magic"))


def test_group_key_inconsistent_with_strategy_fails():
    # patient_aware strategy but donor_id group key
    with pytest.raises(CCRTSplitError):
        validate_split_manifest(
            base_manifest(split_strategy="patient_aware", group_key="donor_id")
        )


def test_split_group_id_is_universally_consistent():
    validate_split_manifest(
        base_manifest(split_strategy="patient_aware", group_key="split_group_id")
    )


def test_overlapping_train_test_fails():
    with pytest.raises(CCRTSplitError):
        validate_split_manifest(
            base_manifest(train=["p1", "p4"], test=["p4"])
        )


def test_missing_validation_fails():
    m = base_manifest()
    del m["validation"]
    with pytest.raises(CCRTSplitError):
        validate_split_manifest(m)


def test_empty_train_fails():
    with pytest.raises(CCRTSplitError):
        validate_split_manifest(base_manifest(train=[]))


def test_missing_group_key_fails():
    m = base_manifest()
    del m["group_key"]
    with pytest.raises(CCRTSplitError):
        validate_split_manifest(m)


def test_invalid_group_key_fails():
    with pytest.raises(CCRTSplitError):
        validate_split_manifest(base_manifest(group_key="cell_barcode"))


def test_manifest_key_with_leakage_field_fails():
    with pytest.raises(CCRTLeakageError):
        validate_split_manifest(base_manifest(future_expression="leak"))


def test_non_mapping_manifest_fails():
    with pytest.raises(CCRTSplitError):
        validate_split_manifest(["not", "a", "mapping"])  # type: ignore[arg-type]
