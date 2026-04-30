"""Split manifest handling for donor-held-out cross-validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(slots=True, frozen=True)
class FoldSpec:
    """Specification for one CV fold."""
    fold: int
    train_donors: tuple[str, ...]
    val_donors: tuple[str, ...]
    test_donors: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SplitManifest:
    """Donor-held-out cross-validation split manifest."""
    folds: tuple[FoldSpec, ...]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def get_fold(self, fold_idx: int) -> FoldSpec:
        if fold_idx < 0 or fold_idx >= len(self.folds):
            raise ValueError(f"Fold {fold_idx} not found. Available: 0-{len(self.folds)-1}")
        return self.folds[fold_idx]

    def get_donors(
        self,
        fold_idx: int,
        split: Literal["train", "val", "test"],
    ) -> tuple[str, ...]:
        """Get donors for a specific fold and split."""
        fold = self.get_fold(fold_idx)
        if split == "train":
            return fold.train_donors
        elif split == "val":
            return fold.val_donors
        elif split == "test":
            return fold.test_donors
        else:
            raise ValueError(f"Unknown split '{split}'. Expected train/val/test")


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load split manifest from JSON file.

    Expected format:
    {
        "folds": [
            {
                "fold": 0,
                "train_donors": ["donor_02", "donor_03", ...],
                "val_donors": ["donor_01"],
                "test_donors": ["donor_00"]
            },
            ...
        ]
    }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    with open(path) as f:
        data = json.load(f)

    folds = []
    for fold_data in data["folds"]:
        folds.append(FoldSpec(
            fold=fold_data["fold"],
            train_donors=tuple(fold_data.get("train_donors", [])),
            val_donors=tuple(fold_data.get("val_donors", [])),
            test_donors=tuple(fold_data.get("test_donors", [])),
        ))

    return SplitManifest(folds=tuple(folds))


def get_fold_donors(
    manifest: SplitManifest,
    fold_idx: int,
    split: Literal["train", "val", "test"],
) -> tuple[str, ...]:
    """Convenience function to get donors for a fold and split."""
    return manifest.get_donors(fold_idx, split)
