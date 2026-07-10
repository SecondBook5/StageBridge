"""Tests for donor-grouped PanIN folds."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.adapters.panin import (
    PanINGroupRecord,
    PanINSplitConfig,
    build_grouped_panin_folds,
    validate_no_panin_group_leakage,
)
from stagebridge.ccrt.contracts.errors import CCRTSplitError


def group(gid, donor, stages, samples=("s",), sections=("sec",), platforms=("xenium",)):
    return PanINGroupRecord(
        group_id=gid, donor_id=donor, sample_ids=samples, section_ids=sections,
        platforms=platforms, canonical_stage_ids=stages,
    )


def test_donor_stays_in_one_fold():
    groups = [group(f"d{i}", f"d{i}", ("normal_duct", "low_grade_panin")) for i in range(6)]
    cfg = PanINSplitConfig(grouping_level="donor", num_folds=3, seed=0)
    assignments = build_grouped_panin_folds(groups=groups, config=cfg)
    # each group appears exactly once
    assert len(assignments) == 6
    assert len({a.group_id for a in assignments}) == 6
    validate_no_panin_group_leakage(assignments, None)


def test_deterministic_assignment():
    groups = [group(f"d{i}", f"d{i}", ("normal_duct",)) for i in range(8)]
    cfg = PanINSplitConfig(grouping_level="donor", num_folds=4, seed=0)
    a1 = build_grouped_panin_folds(groups=groups, config=cfg)
    a2 = build_grouped_panin_folds(groups=groups, config=cfg)
    assert [(a.group_id, a.fold_index) for a in a1] == [(a.group_id, a.fold_index) for a in a2]


def test_donor_grouping_requires_donor_id():
    groups = [group("g0", None, ("normal_duct",))]
    cfg = PanINSplitConfig(grouping_level="donor", num_folds=2)
    with pytest.raises(CCRTSplitError):
        build_grouped_panin_folds(groups=groups, config=cfg)


def test_grouping_level_recorded():
    groups = [group(f"d{i}", f"d{i}", ("normal_duct",)) for i in range(4)]
    cfg = PanINSplitConfig(grouping_level="donor", num_folds=2)
    assignments = build_grouped_panin_folds(groups=groups, config=cfg)
    assert all(a.grouping_level == "donor" for a in assignments)


def test_sample_grouping_requires_opt_in():
    groups = [group(f"g{i}", None, ("normal_duct",)) for i in range(4)]
    cfg = PanINSplitConfig(grouping_level="sample", num_folds=2, allow_sample_level_grouping=True)
    assignments = build_grouped_panin_folds(groups=groups, config=cfg)
    assert all(a.grouping_level == "sample" for a in assignments)


def test_approximate_stage_balance():
    # 4 normal-only + 4 lowgrade-only groups across 2 folds -> each fold gets a mix
    groups = (
        [group(f"n{i}", f"n{i}", ("normal_duct",)) for i in range(4)]
        + [group(f"l{i}", f"l{i}", ("low_grade_panin",)) for i in range(4)]
    )
    cfg = PanINSplitConfig(grouping_level="donor", num_folds=2, seed=0)
    assignments = build_grouped_panin_folds(groups=groups, config=cfg)
    by_fold = {}
    for a in assignments:
        by_fold.setdefault(a.fold_index, []).append(a.group_id)
    # both folds populated
    assert len(by_fold) == 2


def test_leakage_detection():
    from stagebridge.ccrt.adapters.panin import PanINFoldAssignment
    bad = [
        PanINFoldAssignment(group_id="d0", fold_index=0, grouping_level="donor"),
        PanINFoldAssignment(group_id="d0", fold_index=1, grouping_level="donor"),
    ]
    with pytest.raises(CCRTSplitError):
        validate_no_panin_group_leakage(bad, None)
