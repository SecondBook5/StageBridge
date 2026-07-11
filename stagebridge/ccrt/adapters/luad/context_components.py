"""LUAD deconvolved typed-context components.

A deconvolution backend (here, Tangram) associates typed context components with
Visium spots. ONE backend x spot x cell-type = ONE component. Components are
inferred, never observed cells, and are never replicated by abundance into
pseudo-cells. Each component preserves its backend identity, its source cell-type
label, its abundance/score, an optional uncertainty, and an ordered feature
vector. When the source provides no uncertainty, uncertainty is 0.0 with
``uncertainty_source="not_provided"`` — abundance is NEVER reused as uncertainty.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ...contracts.errors import CCRTValidationError
from .config import LUADContextBackendConfig

__all__ = [
    "LUADContextComponent",
    "load_luad_context_components",
    "validate_luad_context_components",
]

_ALLOWED_UNCERTAINTY_SOURCES = frozenset({"provided", "not_provided"})


@dataclass(frozen=True)
class LUADContextComponent:
    """One deconvolved typed-context component (backend x spot x cell-type)."""

    component_id: str
    backend_id: str
    spot_id: str
    sender_context_type_id: str
    source_sender_label: str
    abundance: float
    uncertainty: float
    uncertainty_source: str
    feature_vector: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in (
            "component_id", "backend_id", "spot_id", "sender_context_type_id",
            "source_sender_label",
        ):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise CCRTValidationError(
                    f"LUADContextComponent.{name} must be a non-empty string"
                )
        if self.uncertainty_source not in _ALLOWED_UNCERTAINTY_SOURCES:
            raise CCRTValidationError(
                f"uncertainty_source '{self.uncertainty_source}' invalid; allowed: "
                f"{sorted(_ALLOWED_UNCERTAINTY_SOURCES)}"
            )
        object.__setattr__(self, "feature_vector", tuple(float(x) for x in self.feature_vector))


def _make_component_id(backend_id: str, spot_id: str, sender_context_type_id: str) -> str:
    return f"{backend_id}::{spot_id}::{sender_context_type_id}"


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        header = list(reader.fieldnames or [])
        rows = [dict(line) for line in reader]
    return header, rows


def load_luad_context_components(
    config: LUADContextBackendConfig,
    *,
    source_root: str | Path,
    sender_context_annotation_map: Mapping[str, str],
    strict_unknown_annotations: bool = True,
    excluded_annotations: tuple[str, ...] = (),
) -> tuple[LUADContextComponent, ...]:
    """Load one backend's deconvolved context components from a long-form table.

    The table is expected in melted form: one row per (spot, cell-type). Each row
    carries the spot id, the source cell-type label, an abundance/score, an
    optional uncertainty column, and the ordered feature columns. Unknown
    cell-type labels fail in strict mode (never silently bucketed).
    """
    path = Path(source_root) / config.source_path
    if not path.is_file():
        raise CCRTValidationError(
            f"context backend source not found: {path} (backend "
            f"'{config.backend_id}')"
        )

    suffix = path.suffix.lower()
    if suffix not in (".csv", ".tsv", ".txt"):
        raise CCRTValidationError(
            f"context backend format '{suffix}' for '{config.backend_id}' requires a "
            "lazy source reader not available in this environment; provide a CSV/TSV "
            "export or install the source-format reader explicitly"
        )

    header, rows = _read_rows(path)
    required_cols = [config.spot_id_key, config.sender_context_type_key, config.abundance_key]
    if config.uncertainty_key is not None:
        required_cols.append(config.uncertainty_key)
    required_cols.extend(config.feature_ids)
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise CCRTValidationError(
            f"context backend '{config.backend_id}' source {path} missing columns: {missing}"
        )

    excluded = frozenset(excluded_annotations)
    components: list[LUADContextComponent] = []
    for line in rows:
        source_label = str(line[config.sender_context_type_key])
        if source_label in excluded:
            continue
        canonical = sender_context_annotation_map.get(source_label)
        if canonical is None:
            if strict_unknown_annotations:
                raise CCRTValidationError(
                    f"unknown sender-context label '{source_label}' in backend "
                    f"'{config.backend_id}' (strict mode; not mapped, not excluded)"
                )
            continue

        spot_id = str(line[config.spot_id_key])
        try:
            abundance = float(line[config.abundance_key])
        except (TypeError, ValueError) as exc:
            raise CCRTValidationError(
                f"non-numeric abundance {line[config.abundance_key]!r} for backend "
                f"'{config.backend_id}' spot '{spot_id}'"
            ) from exc

        if config.uncertainty_key is not None:
            try:
                uncertainty = float(line[config.uncertainty_key])
            except (TypeError, ValueError) as exc:
                raise CCRTValidationError(
                    f"non-numeric uncertainty {line[config.uncertainty_key]!r} for "
                    f"backend '{config.backend_id}' spot '{spot_id}'"
                ) from exc
            uncertainty_source = "provided"
        else:
            # No uncertainty column: 0.0 with explicit provenance. Never reuse
            # abundance as uncertainty.
            uncertainty = 0.0
            uncertainty_source = "not_provided"

        feature_vector = []
        for fid in config.feature_ids:
            try:
                feature_vector.append(float(line[fid]))
            except (TypeError, ValueError) as exc:
                raise CCRTValidationError(
                    f"non-numeric feature {line[fid]!r} for '{fid}' in backend "
                    f"'{config.backend_id}' spot '{spot_id}'"
                ) from exc

        components.append(
            LUADContextComponent(
                component_id=_make_component_id(config.backend_id, spot_id, canonical),
                backend_id=config.backend_id,
                spot_id=spot_id,
                sender_context_type_id=canonical,
                source_sender_label=source_label,
                abundance=abundance,
                uncertainty=uncertainty,
                uncertainty_source=uncertainty_source,
                feature_vector=tuple(feature_vector),
            )
        )

    validate_luad_context_components(components, expected_feature_dim=len(config.feature_ids))
    return tuple(components)


def validate_luad_context_components(
    components,
    *,
    expected_feature_dim: int | None = None,
) -> None:
    """Assert component invariants: unique backend x spot x type; finite nonneg."""
    seen: set[str] = set()
    for c in components:
        # one backend x spot x type = one component
        if c.component_id in seen:
            raise CCRTValidationError(
                f"duplicate context component '{c.component_id}' (one backend x spot "
                "x cell-type = one component; never replicated by abundance)"
            )
        seen.add(c.component_id)

        if not (c.abundance == c.abundance) or c.abundance == float("inf") or c.abundance == float("-inf"):
            raise CCRTValidationError(
                f"context component '{c.component_id}' abundance is not finite"
            )
        if c.abundance < 0:
            raise CCRTValidationError(
                f"context component '{c.component_id}' abundance is negative"
            )
        if not (c.uncertainty == c.uncertainty) or c.uncertainty == float("inf") or c.uncertainty == float("-inf"):
            raise CCRTValidationError(
                f"context component '{c.component_id}' uncertainty is not finite"
            )
        if c.uncertainty < 0:
            raise CCRTValidationError(
                f"context component '{c.component_id}' uncertainty is negative"
            )
        if c.uncertainty_source == "not_provided" and c.uncertainty != 0.0:
            raise CCRTValidationError(
                f"context component '{c.component_id}' declares no uncertainty source "
                "but carries a non-zero uncertainty"
            )
        for x in c.feature_vector:
            if not (x == x) or x == float("inf") or x == float("-inf"):
                raise CCRTValidationError(
                    f"context component '{c.component_id}' has a non-finite feature"
                )
        if expected_feature_dim is not None and len(c.feature_vector) != expected_feature_dim:
            raise CCRTValidationError(
                f"context component '{c.component_id}' feature vector width "
                f"{len(c.feature_vector)} != expected {expected_feature_dim}"
            )
