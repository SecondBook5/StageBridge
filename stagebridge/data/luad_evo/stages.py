"""Canonical stage ontology utilities for lung progression modeling."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


CANONICAL_STAGE_ORDER: tuple[str, ...] = ("Normal", "AAH", "AIS", "MIA", "LUAD")

# ── Grouped ordinal labels for rescue study ──────────────────────────────────
# Biologically-motivated grouping for ordinal lesion positioning.
# early_like: pre-invasive (Normal + AAH)
# intermediate_like: mixed invasive potential (AIS + MIA)
# invasive_like: established invasive adenocarcinoma (LUAD)
GROUPED_STAGE_ORDER: tuple[str, ...] = ("early_like", "intermediate_like", "invasive_like")

STAGE_TO_GROUP: dict[str, str] = {
    "Normal": "early_like",
    "AAH": "early_like",
    "AIS": "intermediate_like",
    "MIA": "intermediate_like",
    "LUAD": "invasive_like",
}

GROUPED_STAGE_INDEX: dict[str, int] = {
    "early_like": 0,
    "intermediate_like": 1,
    "invasive_like": 2,
}


def stage_to_grouped_index(stage: str) -> int:
    """Map a canonical 5-class stage label to its grouped ordinal index (0–2)."""
    normalized = normalize_stage_label(stage)
    group = STAGE_TO_GROUP.get(normalized)
    if group is None:
        raise ValueError(
            f"Unknown stage '{stage}' (normalized='{normalized}'). "
            f"Cannot map to grouped ordinal label."
        )
    return GROUPED_STAGE_INDEX[group]


def stage_to_group_label(stage: str) -> str:
    """Map a canonical 5-class stage label to its grouped label string."""
    normalized = normalize_stage_label(stage)
    group = STAGE_TO_GROUP.get(normalized)
    if group is None:
        raise ValueError(f"Unknown stage '{stage}' for grouping.")
    return group


# Extended ontology for cross-dataset modeling (Normal → ... → metastasis).
# LUAD/PRIMARY are treated as the same biological stage (primary NSCLC tumor)
# bridging the early-progression (GSE308103) and brain-mets (GSE223499) datasets.
EXTENDED_STAGE_ORDER: tuple[str, ...] = (
    "Normal", "AAH", "AIS", "MIA", "LUAD",  # early progression
    "BrainMet",                               # brain metastasis
    "ChestWallMet",                           # chest wall metastasis
)

# Adjacency edges in the extended graph (not necessarily linear).
# LUAD branches to BrainMet and ChestWallMet.
EXTENDED_STAGE_EDGES: list[tuple[str, str]] = [
    ("Normal", "AAH"),
    ("AAH", "AIS"),
    ("AIS", "MIA"),
    ("MIA", "LUAD"),
    ("LUAD", "BrainMet"),
    ("LUAD", "ChestWallMet"),
]

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
    "primary": "LUAD",
    "brain mets": "BrainMet",
    "brain_mets": "BrainMet",
    "brainmet": "BrainMet",
    "brain metastasis": "BrainMet",
    "chest wall met": "ChestWallMet",
    "chest_wall_met": "ChestWallMet",
    "chestwallmet": "ChestWallMet",
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


def stage_to_index(stage: str, *, extended: bool = False) -> int:
    """Map canonical stage name to ordinal index.

    Parameters
    ----------
    stage : str
        Stage name (raw or normalized).
    extended : bool
        When True, use the extended ontology (includes BrainMet, ChestWallMet).
        Default False uses the original 5-stage LUAD ontology.
    """
    order = EXTENDED_STAGE_ORDER if extended else CANONICAL_STAGE_ORDER
    normalized = normalize_stage_label(stage)
    if normalized not in order:
        raise ValueError(
            f"Unknown stage '{stage}' (normalized='{normalized}'). "
            f"Expected one of {order}."
        )
    return order.index(normalized)


def index_to_stage(index: int, *, extended: bool = False) -> str:
    """Inverse mapping from stage index to name."""
    order = EXTENDED_STAGE_ORDER if extended else CANONICAL_STAGE_ORDER
    if index < 0 or index >= len(order):
        raise IndexError(
            f"Stage index {index} out of range [0, {len(order)-1}]"
        )
    return order[index]


def stage_to_progression_score(stage: str, *, extended: bool = False) -> float:
    """Map a stage label to a normalized ordinal progression score."""
    order = EXTENDED_STAGE_ORDER if extended else CANONICAL_STAGE_ORDER
    if len(order) <= 1:
        return 0.0
    return float(stage_to_index(stage, extended=extended)) / float(len(order) - 1)


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
