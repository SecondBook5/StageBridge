"""LUAD source auditing.

Inspects the local LUAD data tree (or a source-faithful fixture) and records what
actually exists: files, platforms, observation units, columns, coordinate units,
stage/annotation labels, deconvolution backends, modality relationships, and
missing requirements. Inspection only — no mutation, no data copying, bounded
metadata reads (large matrices are recorded by path/size/mtime, never loaded or
hashed).

The dataset's biological accuracy is NOT verified (see ``SOURCE_AUDIT.md``); this
audit validates only *structure*, and flags pipeline smoke-check artifacts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...contracts.errors import CCRTValidationError
from .config import LUADAdapterConfig

__all__ = [
    "LUADSourceFile",
    "LUADSourceAudit",
    "audit_luad_source",
    "validate_reference_source_audit",
]

#: Files at or below this size get a SHA-256; larger files are recorded by
#: metadata only (never loaded/hashed). The giant h5ad objects are never read.
_MAX_HASH_BYTES = 1_000_000

#: Substrings that mark pipeline smoke-check artifacts (not final outputs).
_SMOKE_MARKERS = ("smoke_check", "_smoke.")


@dataclass(frozen=True)
class LUADSourceFile:
    relative_path: str
    file_type: str
    size_bytes: int
    modified_time: float
    sha256: str | None = None
    is_smoke_check: bool = False


@dataclass(frozen=True)
class LUADSourceAudit:
    source_root: str
    dataset_commit: str | None
    layout_version: str
    files: tuple[LUADSourceFile, ...]
    platforms: tuple[str, ...]
    observation_units: Mapping[str, str]
    donor_column: str | None
    patient_column: str | None
    sample_column: str | None
    section_column: str | None
    stage_column: str | None
    annotation_columns: tuple[str, ...]
    coordinate_columns: tuple[str, ...]
    coordinate_units: Mapping[str, str]
    stage_labels: tuple[str, ...]
    annotation_labels: tuple[str, ...]
    context_backends: tuple[str, ...]
    modality_relationships: Mapping[str, str]
    smoke_check_artifacts: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    warnings: tuple[str, ...]


def _read_commit(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    try:
        content = head.read_text(encoding="utf-8").strip()
    except OSError:  # pragma: no cover - defensive
        return None
    if content.startswith("ref:"):
        ref = content.split(" ", 1)[1].strip()
        ref_path = root / ".git" / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        return None
    return content  # detached HEAD


def _classify(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or "none"


def _is_smoke(relative_path: str) -> bool:
    lowered = relative_path.lower()
    return any(marker in lowered for marker in _SMOKE_MARKERS)


def audit_luad_source(source_root: str | Path) -> LUADSourceAudit:
    """Audit a LUAD source tree (or fixture). Inspection only."""
    root = Path(source_root)
    if not root.exists():
        raise CCRTValidationError(f"source_root does not exist: {root}")
    if not root.is_dir():
        raise CCRTValidationError(f"source_root is not a directory: {root}")

    files: list[LUADSourceFile] = []
    smoke: list[str] = []
    for p in sorted(root.rglob("*")):
        if ".git" in p.parts:
            continue
        if not p.is_file():
            continue
        stat = p.stat()
        sha = None
        if stat.st_size <= _MAX_HASH_BYTES:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
        rel = str(p.relative_to(root))
        is_smoke = _is_smoke(rel)
        if is_smoke:
            smoke.append(rel)
        files.append(
            LUADSourceFile(
                relative_path=rel,
                file_type=_classify(p),
                size_bytes=stat.st_size,
                modified_time=stat.st_mtime,
                sha256=sha,
                is_smoke_check=is_smoke,
            )
        )

    manifest = _load_manifest(root)
    if manifest is not None:
        return _audit_from_manifest(root, manifest, files, tuple(sorted(smoke)))

    # No manifest: record what exists but flag that the adapter cannot proceed
    # without the modality manifest describing platforms/columns/backends.
    return LUADSourceAudit(
        source_root=str(root),
        dataset_commit=_read_commit(root),
        layout_version="unknown",
        files=tuple(files),
        platforms=(),
        observation_units={},
        donor_column=None,
        patient_column=None,
        sample_column=None,
        section_column=None,
        stage_column=None,
        annotation_columns=(),
        coordinate_columns=(),
        coordinate_units={},
        stage_labels=(),
        annotation_labels=(),
        context_backends=(),
        modality_relationships={},
        smoke_check_artifacts=tuple(sorted(smoke)),
        missing_requirements=(
            "no LUAD source manifest (luad_source_manifest.json) found; the "
            "adapter requires a manifest describing modalities (snRNA GSE308103, "
            "Visium GSE307534), coordinate units, and the tangram context backend",
        ),
        warnings=("audit ran without a source manifest; adapter cannot proceed",),
    )


def _load_manifest(root: Path) -> Mapping[str, Any] | None:
    import json

    candidate = root / "luad_source_manifest.json"
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _audit_from_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    files: list[LUADSourceFile],
    smoke: tuple[str, ...],
) -> LUADSourceAudit:
    def _tuple(key: str) -> tuple:
        return tuple(manifest.get(key, ()) or ())

    warnings = list(_tuple("warnings"))
    if smoke:
        warnings.append(
            f"{len(smoke)} pipeline smoke-check artifact(s) present alongside full "
            "outputs; dataset accuracy is not verified (structure-only adapter)"
        )

    return LUADSourceAudit(
        source_root=str(root),
        dataset_commit=manifest.get("dataset_commit") or _read_commit(root),
        layout_version=manifest.get("layout_version", "unknown"),
        files=tuple(files),
        platforms=_tuple("platforms"),
        observation_units=dict(manifest.get("observation_units", {})),
        donor_column=manifest.get("donor_column"),
        patient_column=manifest.get("patient_column"),
        sample_column=manifest.get("sample_column"),
        section_column=manifest.get("section_column"),
        stage_column=manifest.get("stage_column"),
        annotation_columns=_tuple("annotation_columns"),
        coordinate_columns=_tuple("coordinate_columns"),
        coordinate_units=dict(manifest.get("coordinate_units", {})),
        stage_labels=_tuple("stage_labels"),
        annotation_labels=_tuple("annotation_labels"),
        context_backends=_tuple("context_backends"),
        modality_relationships=dict(manifest.get("modality_relationships", {})),
        smoke_check_artifacts=smoke,
        missing_requirements=_tuple("missing_requirements"),
        warnings=tuple(warnings),
    )


def validate_reference_source_audit(
    audit: LUADSourceAudit, config: LUADAdapterConfig
) -> None:
    """Fail if the audited source cannot support the configured adapter."""
    problems: list[str] = []

    if audit.missing_requirements:
        problems.extend(audit.missing_requirements)

    # spatial (Visium) platform: observed spots with micron coordinates
    if config.spatial_platform not in audit.platforms:
        problems.append(
            f"spatial platform '{config.spatial_platform}' not present in source "
            f"platforms {audit.platforms}"
        )
    else:
        unit = audit.observation_units.get(config.spatial_platform)
        if unit != "spot":
            problems.append(
                f"spatial platform '{config.spatial_platform}' observation unit is "
                f"'{unit}', expected 'spot'"
            )
        cu = audit.coordinate_units.get(config.spatial_platform)
        if cu != "microns":
            problems.append(
                f"coordinate units for '{config.spatial_platform}' unresolved or not "
                f"microns (got {cu!r}); the Space Ranger scalefactor must be applied"
            )

    # snRNA reference platform: molecular reference (cells), no coordinates
    if config.snrna_platform not in audit.platforms:
        problems.append(
            f"snRNA platform '{config.snrna_platform}' not present in source "
            f"platforms {audit.platforms}"
        )
    else:
        unit = audit.observation_units.get(config.snrna_platform)
        if unit != "cell":
            problems.append(
                f"snRNA platform '{config.snrna_platform}' observation unit is "
                f"'{unit}', expected 'cell'"
            )

    # deconvolution context backend present (never fabricated)
    if config.context_backend.backend_id not in audit.context_backends:
        problems.append(
            f"context backend '{config.context_backend.backend_id}' not present in "
            f"audited backends {audit.context_backends}"
        )

    # required stage labels for configured edges
    needed_states = set()
    for _edge_id, src, tgt in config.transition_edges:
        needed_states.add(src)
        needed_states.add(tgt)
    mapped_stages = {config.stage_map.get(lbl) for lbl in audit.stage_labels}
    for state in needed_states:
        if state not in mapped_stages:
            problems.append(
                f"required receiver state '{state}' not represented by any audited "
                f"stage label (stage labels: {audit.stage_labels})"
            )

    # required sender annotations present
    audited_annos = set(audit.annotation_labels)
    if audited_annos:
        mappable = audited_annos & set(config.sender_context_annotation_map)
        if not mappable:
            problems.append(
                "no audited annotation label maps to a configured sender-context type"
            )

    if problems:
        raise CCRTValidationError(
            "LUAD source audit does not support the configured adapter:\n- "
            + "\n- ".join(problems)
        )
