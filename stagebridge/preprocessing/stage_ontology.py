"""Canonical stage ontology utilities for lung progression modeling."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


CANONICAL_STAGE_ORDER: tuple[str, ...] = ("Normal", "AAH", "AIS", "MIA", "LUAD")

_ALIASES: dict[str, str] = {
    "normal": "Normal",
    "healthy": "Normal",
    "adjacent normal": "Normal",
    "aah": "AAH",
    "atypical adenomatous hyperplasia": "AAH",
    "ais": "AIS",
    "adenocarcinoma in situ": "AIS",
    "mia": "MIA",
    "minimally invasive adenocarcinoma": "MIA",
    "luad": "LUAD",
    "invasive luad": "LUAD",
    "adenocarcinoma": "LUAD",
}


def normalize_stage_label(label: str | None) -> str:
    """Normalize arbitrary stage labels to the canonical ontology."""
    if label is None:
        return "Unknown"
    clean = re.sub(r"\s+", " ", str(label).strip().lower())
    clean = re.sub(r"[_\-]+", " ", clean).strip()
    clean = re.sub(r"\d+$", "", clean).strip()

    if clean in _ALIASES:
        return _ALIASES[clean]

    for token, canonical in _ALIASES.items():
        if token in clean:
            return canonical

    return str(label).strip().title()


def normalize_stage_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas stage column."""
    return series.astype(str).map(normalize_stage_label)


def stage_to_index(stage: str) -> int:
    """Map canonical stage name to ordinal index."""
    normalized = normalize_stage_label(stage)
    if normalized not in CANONICAL_STAGE_ORDER:
        raise ValueError(
            f"Unknown stage '{stage}' (normalized='{normalized}'). "
            f"Expected one of {CANONICAL_STAGE_ORDER}."
        )
    return CANONICAL_STAGE_ORDER.index(normalized)


def index_to_stage(index: int) -> str:
    """Inverse mapping from stage index to name."""
    if index < 0 or index >= len(CANONICAL_STAGE_ORDER):
        raise IndexError(
            f"Stage index {index} out of range [0, {len(CANONICAL_STAGE_ORDER)-1}]"
        )
    return CANONICAL_STAGE_ORDER[index]


def ordered_transitions(order: tuple[str, ...] | None = None) -> list[tuple[str, str]]:
    """Return ordered stage-to-stage transitions."""
    seq = order or CANONICAL_STAGE_ORDER
    return list(zip(seq[:-1], seq[1:]))


def build_stage_pair_index(
    order: tuple[str, ...] | None = None,
) -> dict[tuple[str, str], int]:
    """Build a deterministic integer id for each directed stage pair."""
    seq = order or CANONICAL_STAGE_ORDER
    pair_to_id: dict[tuple[str, str], int] = {}
    idx = 0
    for src in seq:
        for tgt in seq:
            pair_to_id[(src, tgt)] = idx
            idx += 1
    return pair_to_id


def infer_stage_pair_id(stage_src: str, stage_tgt: str) -> int:
    """Map a source/target stage pair to a deterministic id."""
    src = normalize_stage_label(stage_src)
    tgt = normalize_stage_label(stage_tgt)
    pair_to_id = build_stage_pair_index()
    if (src, tgt) not in pair_to_id:
        raise ValueError(f"Cannot infer pair id for ({stage_src}, {stage_tgt}).")
    return pair_to_id[(src, tgt)]


def apply_stage_ontology(
    obs: pd.DataFrame,
    stage_col: str = "stage",
    stage_index_col: str = "stage_index",
) -> pd.DataFrame:
    """Return a copy of ``obs`` with normalized stage and stage index columns."""
    if stage_col not in obs.columns:
        raise KeyError(
            f"Missing stage column '{stage_col}'. Found: {list(obs.columns)}"
        )
    out = obs.copy()
    out[stage_col] = normalize_stage_series(out[stage_col])
    out[stage_index_col] = out[stage_col].map(
        lambda s: CANONICAL_STAGE_ORDER.index(s)
        if s in CANONICAL_STAGE_ORDER
        else -1
    )
    return out


def validate_stage_progression(obs: pd.DataFrame, stage_col: str = "stage") -> None:
    """Validate that all canonical stages are present in the table."""
    stages = set(normalize_stage_series(obs[stage_col]))
    missing = [s for s in CANONICAL_STAGE_ORDER if s not in stages]
    if missing:
        raise ValueError(
            "Missing canonical stages required for full progression: "
            f"{missing}. Present stages: {sorted(stages)}"
        )


@dataclass(slots=True, frozen=True)
class StageTransition:
    """Simple typed representation of a stage transition edge."""

    stage_src: str
    stage_tgt: str

    @property
    def pair_id(self) -> int:
        return infer_stage_pair_id(self.stage_src, self.stage_tgt)
