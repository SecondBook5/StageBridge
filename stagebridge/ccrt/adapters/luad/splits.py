"""Donor-grouped split contract for the LUAD adapter.

Groups all samples/sections/platforms/backends of a donor (= patient) into a
single fold using a deterministic greedy assignment with approximate stage
balance. No ``hash()``, no observation-level random split. Sample-level grouping
requires explicit opt-in and is never presented as donor-held-out. Every backend
of a spatial observation stays with its donor's fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ...contracts.errors import CCRTSplitError
from .config import LUADSplitConfig

__all__ = [
    "LUADGroupRecord",
    "LUADFoldAssignment",
    "build_grouped_luad_folds",
    "validate_no_luad_group_leakage",
]


@dataclass(frozen=True)
class LUADGroupRecord:
    group_id: str
    donor_id: str | None
    sample_ids: tuple[str, ...]
    section_ids: tuple[str, ...]
    platforms: tuple[str, ...]
    backend_ids: tuple[str, ...]
    canonical_stage_ids: tuple[str, ...]


@dataclass(frozen=True)
class LUADFoldAssignment:
    group_id: str
    fold_index: int
    grouping_level: str


def build_grouped_luad_folds(
    *,
    groups: Sequence[LUADGroupRecord],
    config: LUADSplitConfig,
) -> tuple[LUADFoldAssignment, ...]:
    """Assign groups to folds deterministically with approximate stage balance."""
    if not groups:
        raise CCRTSplitError("no groups to assign")

    if config.grouping_level == "donor":
        for g in groups:
            if g.donor_id is None:
                raise CCRTSplitError(
                    f"group '{g.group_id}' has no donor_id but donor grouping is "
                    "required (no silent fallback to sample grouping)"
                )
    # sample grouping already gated by LUADSplitConfig.allow_sample_level_grouping

    num_folds = config.num_folds
    # deterministic ordering: by descending stage-count then group_id, so larger
    # groups are placed first (greedy balance). No hashing.
    ordered = sorted(
        groups, key=lambda g: (-len(g.canonical_stage_ids), g.group_id)
    )

    fold_load = [0] * num_folds          # number of groups per fold
    fold_stage_counts: list[dict[str, int]] = [dict() for _ in range(num_folds)]
    assignments: list[LUADFoldAssignment] = []

    for g in ordered:
        # choose the fold that best improves stage balance, ties broken by load
        best_fold = 0
        best_key = None
        for f in range(num_folds):
            projected = dict(fold_stage_counts[f])
            for s in g.canonical_stage_ids:
                projected[s] = projected.get(s, 0) + 1
            imbalance = max(projected.values()) if projected else 0
            key = (imbalance, fold_load[f], f)
            if best_key is None or key < best_key:
                best_key = key
                best_fold = f
        fold_load[best_fold] += 1
        for s in g.canonical_stage_ids:
            fold_stage_counts[best_fold][s] = fold_stage_counts[best_fold].get(s, 0) + 1
        assignments.append(
            LUADFoldAssignment(
                group_id=g.group_id,
                fold_index=best_fold,
                grouping_level=config.grouping_level,
            )
        )

    # stable ordering of the returned assignments (by group_id)
    assignments.sort(key=lambda a: a.group_id)
    return tuple(assignments)


def validate_no_luad_group_leakage(
    assignments: Sequence[LUADFoldAssignment],
    observations=None,
) -> None:
    """Assert each group id maps to exactly one fold (no leakage across folds)."""
    seen: dict[str, int] = {}
    for a in assignments:
        if a.group_id in seen and seen[a.group_id] != a.fold_index:
            raise CCRTSplitError(
                f"group '{a.group_id}' assigned to multiple folds "
                f"({seen[a.group_id]} and {a.fold_index}) — group leakage"
            )
        seen[a.group_id] = a.fold_index
