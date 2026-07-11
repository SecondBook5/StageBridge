"""Tests for donor-grouped LUAD folds."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.adapters.luad import (
    LUADFoldAssignment,
    LUADGroupRecord,
    LUADSplitConfig,
    build_grouped_luad_folds,
    validate_no_luad_group_leakage,
)
from stagebridge.ccrt.contracts.errors import CCRTSplitError


def group(gid, donor, stages, samples=("s",), sections=("sec",), platforms=("visium",), backends=("tangram",)):
    return LUADGroupRecord(
        group_id=gid, donor_id=donor, sample_ids=samples, section_ids=sections,
        platforms=platforms, backend_ids=backends, canonical_stage_ids=stages,
    )


def test_donor_stays_in_one_fold():
    groups = [group(f"P{i}", f"P{i}", ("normal", "aah")) for i in range(6)]
    cfg = LUADSplitConfig(grouping_level="donor", num_folds=3, seed=0)
    assignments = build_grouped_luad_folds(groups=groups, config=cfg)
    assert len(assignments) == 6
    assert len({a.group_id for a in assignments}) == 6
    validate_no_luad_group_leakage(assignments)


def test_deterministic_assignment():
    groups = [group(f"P{i}", f"P{i}", ("normal",)) for i in range(8)]
    cfg = LUADSplitConfig(grouping_level="donor", num_folds=4, seed=0)
    a1 = build_grouped_luad_folds(groups=groups, config=cfg)
    a2 = build_grouped_luad_folds(groups=groups, config=cfg)
    assert [(a.group_id, a.fold_index) for a in a1] == [(a.group_id, a.fold_index) for a in a2]


def test_donor_grouping_requires_donor_id():
    groups = [group("g0", None, ("normal",))]
    cfg = LUADSplitConfig(grouping_level="donor", num_folds=2)
    with pytest.raises(CCRTSplitError):
        build_grouped_luad_folds(groups=groups, config=cfg)


def test_grouping_level_recorded():
    groups = [group(f"P{i}", f"P{i}", ("normal",)) for i in range(4)]
    cfg = LUADSplitConfig(grouping_level="donor", num_folds=2)
    assignments = build_grouped_luad_folds(groups=groups, config=cfg)
    assert all(a.grouping_level == "donor" for a in assignments)


def test_all_backends_stay_with_donor():
    # a donor with multiple backends stays in a single fold (backends never split)
    groups = [
        group(f"P{i}", f"P{i}", ("normal",), backends=("tangram", "rctd"))
        for i in range(4)
    ]
    cfg = LUADSplitConfig(grouping_level="donor", num_folds=2)
    assignments = build_grouped_luad_folds(groups=groups, config=cfg)
    by_group = {a.group_id: a.fold_index for a in assignments}
    assert len(by_group) == 4  # one fold per group, backends do not fork it


def test_sample_grouping_requires_opt_in():
    groups = [group(f"g{i}", None, ("normal",)) for i in range(4)]
    cfg = LUADSplitConfig(grouping_level="sample", num_folds=2, allow_sample_level_grouping=True)
    assignments = build_grouped_luad_folds(groups=groups, config=cfg)
    assert all(a.grouping_level == "sample" for a in assignments)


def test_approximate_stage_balance():
    groups = (
        [group(f"n{i}", f"n{i}", ("normal",)) for i in range(4)]
        + [group(f"a{i}", f"a{i}", ("aah",)) for i in range(4)]
    )
    cfg = LUADSplitConfig(grouping_level="donor", num_folds=2, seed=0)
    assignments = build_grouped_luad_folds(groups=groups, config=cfg)
    by_fold = {}
    for a in assignments:
        by_fold.setdefault(a.fold_index, []).append(a.group_id)
    assert len(by_fold) == 2


def test_leakage_detection():
    bad = [
        LUADFoldAssignment(group_id="P3", fold_index=0, grouping_level="donor"),
        LUADFoldAssignment(group_id="P3", fold_index=1, grouping_level="donor"),
    ]
    with pytest.raises(CCRTSplitError):
        validate_no_luad_group_leakage(bad)
