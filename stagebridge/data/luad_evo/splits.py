"""Deterministic lesion-level split utilities for EA-MIST."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from stagebridge.utils.types import LesionBag


@dataclass(slots=True, frozen=True)
class LesionFold:
    """One deterministic train/val/test split over lesion bags."""

    fold_index: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_donors: tuple[str, ...]
    val_donors: tuple[str, ...]
    test_donors: tuple[str, ...]

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "fold_index": int(self.fold_index),
            "n_train": len(self.train_indices),
            "n_val": len(self.val_indices),
            "n_test": len(self.test_indices),
            "train_donors": list(self.train_donors),
            "val_donors": list(self.val_donors),
            "test_donors": list(self.test_donors),
        }


def _group_key_for_bag(bag: LesionBag, holdout_key: str) -> str:
    """Resolve the grouping key used for held-out splitting."""
    if holdout_key == "patient_id":
        return str(bag.patient_id)
    if holdout_key == "donor_id":
        return str(bag.donor_id)
    raise ValueError(f"Unsupported holdout_key '{holdout_key}'. Expected 'donor_id' or 'patient_id'.")


def _check_class_balance(indices: Iterable[int], bags: list[LesionBag]) -> dict[float, int]:
    """Return lesion-label counts for a subset of lesion indices."""
    counts: dict[float, int] = {}
    for index in indices:
        label = float(bags[int(index)].label)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _group_support_by_label(
    bags: list[LesionBag],
    *,
    holdout_key: str,
) -> dict[float, set[str]]:
    """Return the held-out grouping support for each lesion label."""
    support: dict[float, set[str]] = {}
    for bag in bags:
        label = float(bag.label)
        support.setdefault(label, set()).add(_group_key_for_bag(bag, holdout_key))
    return support


def _require_label_support_for_holdout(
    bags: list[LesionBag],
    *,
    holdout_key: str,
    num_folds: int,
    min_lesions_per_class: int,
) -> list[float]:
    """Validate that donor-held-out CV is statistically meaningful for this edge.

    Args:
        bags: Edge-specific lesion bags.
        holdout_key: Grouping key used for holdout.
        num_folds: Requested number of outer folds.
        min_lesions_per_class: Minimum lesions per class required in each split.

    Returns:
        The sorted label set observed in the input bags.
    """
    observed_labels = sorted({float(bag.label) for bag in bags})
    if len(observed_labels) < 2:
        raise ValueError(
            "EA-MIST requires at least two lesion labels for held-out evaluation. "
            f"Observed labels: {observed_labels}."
        )

    support = _group_support_by_label(bags, holdout_key=holdout_key)
    minimum_groups_for_any_holdout = 2
    missing_any_holdout = {
        label: sorted(groups)
        for label, groups in support.items()
        if len(groups) < minimum_groups_for_any_holdout
    }
    if missing_any_holdout:
        detail = ", ".join(
            f"label={label}: groups={groups}" for label, groups in sorted(missing_any_holdout.items())
        )
        raise ValueError(
            "Donor-held-out evaluation is not possible because at least one class has fewer than "
            f"{minimum_groups_for_any_holdout} unique {holdout_key} groups. {detail}"
        )

    insufficient_for_folds = {
        label: sorted(groups)
        for label, groups in support.items()
        if len(groups) < int(num_folds)
    }
    if insufficient_for_folds:
        detail = ", ".join(
            f"label={label}: n_groups={len(groups)} groups={groups}"
            for label, groups in sorted(insufficient_for_folds.items())
        )
        raise ValueError(
            "Every outer test fold must contain both classes for AUROC/AUPRC to be meaningful. "
            f"Requested num_folds={num_folds}, but label support is insufficient: {detail}"
        )

    lesions_per_label = {label: 0 for label in observed_labels}
    for bag in bags:
        lesions_per_label[float(bag.label)] += 1
    missing_lesions = {
        label: count
        for label, count in lesions_per_label.items()
        if count < int(min_lesions_per_class)
    }
    if missing_lesions:
        detail = ", ".join(
            f"label={label}: n_lesions={count}" for label, count in sorted(missing_lesions.items())
        )
        raise ValueError(
            "Insufficient lesions for the requested min_lesions_per_class setting. "
            f"Required min_lesions_per_class={min_lesions_per_class}. {detail}"
        )
    return observed_labels


def build_lesion_folds(
    bags: list[LesionBag],
    *,
    holdout_key: str = "donor_id",
    num_folds: int = 3,
    seed: int = 42,
    min_lesions_per_class: int = 1,
) -> list[LesionFold]:
    """Build deterministic outer folds for lesion-level weak supervision.

    Args:
        bags: Lesion bags for one edge-specific task.
        holdout_key: Grouping key for held-out splitting.
        num_folds: Number of outer folds.
        seed: Deterministic seed used only to rotate donor order reproducibly.
        min_lesions_per_class: Minimum lesions per class required in train and
            test splits.
    """

    if not bags:
        raise ValueError("Cannot build lesion folds from an empty bag list.")
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}.")
    expected_labels = _require_label_support_for_holdout(
        bags,
        holdout_key=holdout_key,
        num_folds=num_folds,
        min_lesions_per_class=min_lesions_per_class,
    )

    groups = [(_group_key_for_bag(bag, holdout_key), idx) for idx, bag in enumerate(bags)]
    unique_groups = sorted({group for group, _idx in groups})
    if len(unique_groups) < num_folds:
        raise ValueError(
            f"Need at least {num_folds} unique {holdout_key} groups, found {len(unique_groups)}."
        )

    rng = np.random.default_rng(int(seed))
    rotation = int(rng.integers(0, len(unique_groups))) if len(unique_groups) > 1 else 0
    rotated_groups = unique_groups[rotation:] + unique_groups[:rotation]
    donor_slices = [rotated_groups[i::num_folds] for i in range(num_folds)]

    folds: list[LesionFold] = []
    for fold_idx in range(num_folds):
        test_groups = tuple(sorted(donor_slices[fold_idx]))
        if num_folds == 2:
            remaining_groups = tuple(sorted(donor_slices[(fold_idx + 1) % num_folds]))
            midpoint = max(1, len(remaining_groups) // 2)
            val_groups = tuple(sorted(remaining_groups[:midpoint]))
            train_groups = tuple(sorted(remaining_groups[midpoint:]))
            if not train_groups:
                train_groups = val_groups
                val_groups = ()
        else:
            val_groups = tuple(sorted(donor_slices[(fold_idx + 1) % num_folds]))
            train_groups = tuple(sorted(group for i, groups_i in enumerate(donor_slices) if i not in {fold_idx, (fold_idx + 1) % num_folds} for group in groups_i))
        train_indices = tuple(idx for group, idx in groups if group in train_groups)
        val_indices = tuple(idx for group, idx in groups if group in val_groups)
        test_indices = tuple(idx for group, idx in groups if group in test_groups)
        if not train_indices or not test_indices:
            raise ValueError(f"Fold {fold_idx} produced an empty train or test split.")

        train_counts = _check_class_balance(train_indices, bags)
        val_counts = _check_class_balance(val_indices, bags)
        test_counts = _check_class_balance(test_indices, bags)
        for subset_name, counts in (("train", train_counts), ("val", val_counts), ("test", test_counts)):
            missing_labels = [label for label in expected_labels if float(label) not in counts]
            if missing_labels:
                raise ValueError(
                    f"Fold {fold_idx} is invalid because {subset_name} is missing label(s) {missing_labels}. "
                    f"Observed class balance: {counts}."
                )
            if min(counts.values(), default=0) < int(min_lesions_per_class):
                raise ValueError(
                    f"Fold {fold_idx} has insufficient class balance in {subset_name}: {counts}. "
                    f"Required min_lesions_per_class={min_lesions_per_class}."
                )

        folds.append(
            LesionFold(
                fold_index=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                train_donors=train_groups,
                val_donors=val_groups,
                test_donors=test_groups,
            )
        )
    return folds


def build_multitask_lesion_folds(
    bags: list[LesionBag],
    *,
    holdout_key: str = "donor_id",
    num_folds: int = 3,
    seed: int = 42,
) -> list[LesionFold]:
    """Build donor-held-out lesion folds for cohort-wide stage/displacement training."""
    if not bags:
        raise ValueError("Cannot build lesion folds from an empty bag list.")
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}.")

    groups = [(_group_key_for_bag(bag, holdout_key), idx) for idx, bag in enumerate(bags)]
    unique_groups = sorted({group for group, _idx in groups})
    if len(unique_groups) < num_folds:
        raise ValueError(
            f"Need at least {num_folds} unique {holdout_key} groups, found {len(unique_groups)}."
        )

    rng = np.random.default_rng(int(seed))
    rotation = int(rng.integers(0, len(unique_groups))) if len(unique_groups) > 1 else 0
    rotated_groups = unique_groups[rotation:] + unique_groups[:rotation]
    donor_slices = [rotated_groups[i::num_folds] for i in range(num_folds)]

    folds: list[LesionFold] = []
    for fold_idx in range(num_folds):
        test_groups = tuple(sorted(donor_slices[fold_idx]))
        if num_folds == 2:
            remaining_groups = tuple(sorted(donor_slices[(fold_idx + 1) % num_folds]))
            midpoint = max(1, len(remaining_groups) // 2)
            val_groups = tuple(sorted(remaining_groups[:midpoint]))
            train_groups = tuple(sorted(remaining_groups[midpoint:]))
            if not train_groups:
                train_groups = val_groups
                val_groups = ()
        else:
            val_groups = tuple(sorted(donor_slices[(fold_idx + 1) % num_folds]))
            train_groups = tuple(
                sorted(
                    group
                    for i, groups_i in enumerate(donor_slices)
                    if i not in {fold_idx, (fold_idx + 1) % num_folds}
                    for group in groups_i
                )
            )
        train_indices = tuple(idx for group, idx in groups if group in train_groups)
        val_indices = tuple(idx for group, idx in groups if group in val_groups)
        test_indices = tuple(idx for group, idx in groups if group in test_groups)
        if not train_indices or not test_indices:
            raise ValueError(f"Fold {fold_idx} produced an empty train or test split.")
        folds.append(
            LesionFold(
                fold_index=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                train_donors=train_groups,
                val_donors=val_groups,
                test_donors=test_groups,
            )
        )
    return folds


def assert_no_split_leakage(bags: list[LesionBag], fold: LesionFold) -> None:
    """Hard-fail if train/val/test leakage is detected."""
    subsets = {
        "train": [bags[idx] for idx in fold.train_indices],
        "val": [bags[idx] for idx in fold.val_indices],
        "test": [bags[idx] for idx in fold.test_indices],
    }
    donor_sets = {name: {bag.donor_id for bag in subset} for name, subset in subsets.items()}
    patient_sets = {name: {bag.patient_id for bag in subset} for name, subset in subsets.items()}
    for left_name, left_values in donor_sets.items():
        for right_name, right_values in donor_sets.items():
            if left_name >= right_name:
                continue
            overlap = left_values.intersection(right_values)
            if overlap:
                raise ValueError(f"Detected donor leakage between {left_name} and {right_name}: {sorted(overlap)}")
    for left_name, left_values in patient_sets.items():
        for right_name, right_values in patient_sets.items():
            if left_name >= right_name:
                continue
            overlap = left_values.intersection(right_values)
            if overlap:
                raise ValueError(f"Detected patient leakage between {left_name} and {right_name}: {sorted(overlap)}")


def summarize_fold_class_balance(bags: list[LesionBag], fold: LesionFold) -> dict[str, dict[str, int]]:
    """Return per-split label balance for one fold."""
    return {
        "train": {str(k): int(v) for k, v in _check_class_balance(fold.train_indices, bags).items()},
        "val": {str(k): int(v) for k, v in _check_class_balance(fold.val_indices, bags).items()},
        "test": {str(k): int(v) for k, v in _check_class_balance(fold.test_indices, bags).items()},
    }


def summarize_fold_stage_balance(bags: list[LesionBag], fold: LesionFold) -> dict[str, dict[str, int]]:
    """Return per-split stage balance for one cohort-wide multitask fold."""
    summary: dict[str, dict[str, int]] = {}
    subsets = {
        "train": fold.train_indices,
        "val": fold.val_indices,
        "test": fold.test_indices,
    }
    for subset_name, indices in subsets.items():
        counts: dict[str, int] = {}
        for index in indices:
            stage = str(bags[int(index)].stage)
            counts[stage] = counts.get(stage, 0) + 1
        summary[subset_name] = dict(sorted(counts.items()))
    return summary
