"""PanIN source auditing.

Inspects the read-only reference repository (or a source-faithful fixture) and
records what actually exists: files, platforms, observation units, columns,
coordinate units, stage/annotation labels, modality relationships, and missing
requirements. Inspection only — no mutation, no data copying, bounded metadata
reads (large matrices are recorded by path/size/mtime, not hashed or loaded).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...contracts.errors import CCRTValidationError
from .config import PanINAdapterConfig

__all__ = [
    "PanINSourceFile",
    "PanINSourceAudit",
    "audit_panin_source",
    "validate_reference_source_audit",
]

#: Files at or below this size get a SHA-256; larger files are recorded by
#: metadata only (never loaded/hashed).
_MAX_HASH_BYTES = 1_000_000


@dataclass(frozen=True)
class PanINSourceFile:
    relative_path: str
    file_type: str
    size_bytes: int
    modified_time: float
    sha256: str | None = None


@dataclass(frozen=True)
class PanINSourceAudit:
    source_root: str
    repository_commit: str | None
    layout_version: str
    files: tuple[PanINSourceFile, ...]
    platforms: tuple[str, ...]
    observation_units: Mapping[str, str]
    donor_column: str | None
    sample_column: str | None
    section_column: str | None
    stage_column: str | None
    annotation_columns: tuple[str, ...]
    coordinate_columns: tuple[str, ...]
    coordinate_units: Mapping[str, str]
    stage_labels: tuple[str, ...]
    annotation_labels: tuple[str, ...]
    modality_relationships: Mapping[str, str]
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


def audit_panin_source(source_root: str | Path) -> PanINSourceAudit:
    """Audit a PanIN source tree (or fixture). Inspection only."""
    root = Path(source_root)
    if not root.exists():
        raise CCRTValidationError(f"source_root does not exist: {root}")
    if not root.is_dir():
        raise CCRTValidationError(f"source_root is not a directory: {root}")

    files: list[PanINSourceFile] = []
    for p in sorted(root.rglob("*")):
        if ".git" in p.parts:
            continue
        if not p.is_file():
            continue
        stat = p.stat()
        sha = None
        if stat.st_size <= _MAX_HASH_BYTES:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
        files.append(
            PanINSourceFile(
                relative_path=str(p.relative_to(root)),
                file_type=_classify(p),
                size_bytes=stat.st_size,
                modified_time=stat.st_mtime,
                sha256=sha,
            )
        )

    # Detect an optional adapter manifest (source-faithful fixtures provide one).
    manifest = _load_manifest(root)

    if manifest is not None:
        return _audit_from_manifest(root, manifest, files)

    # No manifest: record the code-only reality (real reference repo has no data).
    return PanINSourceAudit(
        source_root=str(root),
        repository_commit=_read_commit(root),
        layout_version="unknown",
        files=tuple(files),
        platforms=(),
        observation_units={},
        donor_column=None,
        sample_column=None,
        section_column=None,
        stage_column=None,
        annotation_columns=(),
        coordinate_columns=(),
        coordinate_units={},
        stage_labels=(),
        annotation_labels=(),
        modality_relationships={},
        missing_requirements=(
            "no PanIN adapter manifest (panin_source_manifest.json) found; "
            "reference repository is code-only and biological data are hosted on "
            "GEO (Visium GSE254829, Xenium GSE267680) — download into ./data/",
        ),
        warnings=("audit ran without a source manifest; adapter cannot proceed",),
    )


def _load_manifest(root: Path) -> Mapping[str, Any] | None:
    import json

    candidate = root / "panin_source_manifest.json"
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _audit_from_manifest(
    root: Path, manifest: Mapping[str, Any], files: list[PanINSourceFile]
) -> PanINSourceAudit:
    def _tuple(key: str) -> tuple:
        return tuple(manifest.get(key, ()) or ())

    return PanINSourceAudit(
        source_root=str(root),
        repository_commit=manifest.get("repository_commit") or _read_commit(root),
        layout_version=manifest.get("layout_version", "unknown"),
        files=tuple(files),
        platforms=_tuple("platforms"),
        observation_units=dict(manifest.get("observation_units", {})),
        donor_column=manifest.get("donor_column"),
        sample_column=manifest.get("sample_column"),
        section_column=manifest.get("section_column"),
        stage_column=manifest.get("stage_column"),
        annotation_columns=_tuple("annotation_columns"),
        coordinate_columns=_tuple("coordinate_columns"),
        coordinate_units=dict(manifest.get("coordinate_units", {})),
        stage_labels=_tuple("stage_labels"),
        annotation_labels=_tuple("annotation_labels"),
        modality_relationships=dict(manifest.get("modality_relationships", {})),
        missing_requirements=_tuple("missing_requirements"),
        warnings=_tuple("warnings"),
    )


def validate_reference_source_audit(
    audit: PanINSourceAudit, config: PanINAdapterConfig
) -> None:
    """Fail if the audited source cannot support the configured adapter."""
    problems: list[str] = []

    if audit.missing_requirements:
        problems.extend(audit.missing_requirements)

    if config.primary_platform not in audit.platforms:
        problems.append(
            f"primary platform '{config.primary_platform}' not present in source "
            f"platforms {audit.platforms}"
        )
    else:
        unit = audit.observation_units.get(config.primary_platform)
        if unit != "cell":
            problems.append(
                f"primary platform '{config.primary_platform}' observation unit "
                f"is '{unit}', expected cell-resolved 'cell'"
            )
        cu = audit.coordinate_units.get(config.primary_platform)
        if cu != "microns":
            problems.append(
                f"coordinate units for '{config.primary_platform}' unresolved or "
                f"not microns (got {cu!r})"
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
            "PanIN source audit does not support the configured adapter:\n- "
            + "\n- ".join(problems)
        )
