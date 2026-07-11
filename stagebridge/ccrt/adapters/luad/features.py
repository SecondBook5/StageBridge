"""LUAD feature ingestion and registration.

Loads source-derived feature matrices into ``LUADFeatureBlock`` objects and
registers them as ``FeatureSpaceSpec``s. Values remain source-derived: no PCA,
no neural embedding, no full-dataset standardization, no silent imputation or
feature dropping. Metadata columns (stage/lesion/donor/patient/sample/section/
platform/backend/split/target/edge/outcome) and coordinates are rejected from
feature vectors.

CSV/TSV readers cover unit-test fixtures and simple source tables. Heavy source
formats (parquet/anndata) are read lazily only if a real source requires them;
those imports never happen at module import time.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ...contracts.errors import CCRTValidationError
from ...representations import FeatureSpaceRegistry, FeatureSpaceSpec
from .config import LUADFeatureBlockConfig

__all__ = [
    "LUADFeatureBlock",
    "load_luad_feature_block",
    "register_luad_feature_spaces",
    "FORBIDDEN_FEATURE_ID_TOKENS",
]

#: Feature ids that must never be treated as molecular / semantic features.
FORBIDDEN_FEATURE_ID_TOKENS = frozenset(
    {
        "stage", "lesion", "lesion_id", "grade", "donor", "donor_id", "patient",
        "patient_id", "sample", "sample_id", "section", "section_id", "platform",
        "backend", "backend_id", "split", "fold", "target_state", "transition_edge",
        "edge", "outcome", "niche_id", "spot_id",
        "x_centroid", "y_centroid", "z_centroid", "x_coordinate", "y_coordinate",
        "x_spatial", "y_spatial", "x_spatial_microns", "y_spatial_microns",
    }
)


@dataclass(frozen=True)
class LUADFeatureBlock:
    spec: FeatureSpaceSpec
    observation_ids: tuple[str, ...]
    values: torch.Tensor
    source_path: str
    source_matrix_key: str | None
    provenance: Mapping[str, Any]


def _assert_feature_ids_are_features(feature_ids: tuple[str, ...]) -> None:
    for fid in feature_ids:
        if fid.strip().lower() in FORBIDDEN_FEATURE_ID_TOKENS:
            raise CCRTValidationError(
                f"feature id '{fid}' is metadata/coordinate, not a feature; it must "
                "not enter a feature vector"
            )


def _read_csv_matrix(
    path: Path, observation_id_key: str, feature_ids: tuple[str, ...]
) -> tuple[list[str], list[list[float]]]:
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        header = reader.fieldnames or []
        if observation_id_key not in header:
            raise CCRTValidationError(
                f"observation id column '{observation_id_key}' absent from {path}"
            )
        missing = [f for f in feature_ids if f not in header]
        if missing:
            raise CCRTValidationError(
                f"configured features absent from {path}: {missing}"
            )
        obs_ids: list[str] = []
        rows: list[list[float]] = []
        for line in reader:
            obs_ids.append(str(line[observation_id_key]))
            row = []
            for fid in feature_ids:
                raw = line[fid]
                try:
                    row.append(float(raw))
                except (TypeError, ValueError) as exc:
                    raise CCRTValidationError(
                        f"non-numeric value {raw!r} for feature '{fid}' in {path}"
                    ) from exc
            rows.append(row)
    return obs_ids, rows


def load_luad_feature_block(
    config: LUADFeatureBlockConfig,
    *,
    source_root: str | Path,
) -> LUADFeatureBlock:
    """Load a feature block, preserving configured feature order."""
    _assert_feature_ids_are_features(config.feature_ids)

    path = Path(source_root) / config.source_path
    if not path.is_file():
        raise CCRTValidationError(
            f"feature source not found: {path} (feature space "
            f"'{config.feature_space_id}')"
        )

    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        obs_ids, rows = _read_csv_matrix(path, config.observation_id_key, config.feature_ids)
    else:
        # Heavy source formats (parquet/h5ad) are read lazily by a dedicated
        # helper not available in this environment. Fail clearly rather than
        # guess or load the giant objects.
        raise CCRTValidationError(
            f"feature format '{suffix}' for '{config.feature_space_id}' requires a "
            "lazy source reader not available in this environment; provide a "
            "CSV/TSV export or install the source-format reader explicitly"
        )

    if len(set(obs_ids)) != len(obs_ids):
        raise CCRTValidationError(
            f"duplicate observation ids in feature block '{config.feature_space_id}'"
        )
    if not obs_ids:
        raise CCRTValidationError(
            f"feature block '{config.feature_space_id}' has no observations"
        )

    values = torch.tensor(rows, dtype=torch.float64)
    if values.dim() != 2 or values.shape[1] != len(config.feature_ids):
        raise CCRTValidationError(
            f"feature block '{config.feature_space_id}' shape "
            f"{tuple(values.shape)} inconsistent with {len(config.feature_ids)} features"
        )
    if not bool(torch.isfinite(values).all()):
        raise CCRTValidationError(
            f"feature block '{config.feature_space_id}' contains non-finite values"
        )

    spec = FeatureSpaceSpec(
        feature_space_id=config.feature_space_id,
        role=config.role,
        dimension=len(config.feature_ids),
        feature_ids=config.feature_ids,
        metric=config.metric,
        normalization=config.normalization,
        version=config.version,
        description=config.description,
    )
    return LUADFeatureBlock(
        spec=spec,
        observation_ids=tuple(obs_ids),
        values=values,
        source_path=str(config.source_path),
        source_matrix_key=config.matrix_key,
        provenance={
            "source_path": str(path),
            "matrix_key": config.matrix_key,
            "n_observations": len(obs_ids),
            "n_features": len(config.feature_ids),
        },
    )


def register_luad_feature_spaces(
    blocks: list[LUADFeatureBlock],
) -> FeatureSpaceRegistry:
    """Register the feature spaces of a set of blocks (unique ids)."""
    registry = FeatureSpaceRegistry()
    for block in blocks:
        registry.register(block.spec)
    return registry
