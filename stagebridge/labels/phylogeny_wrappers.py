"""PhylogicNDT and fallback phylogeny wrappers for label repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.labels.common_schema import PHYLOGENY_SUMMARY_COLUMNS, ToolCommand
from stagebridge.labels.tool_runner import run_external_command


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    """Read one dotted config key from OmegaConf or dict payloads.

    Args:
        cfg: Config tree.
        dotted: Dotted key path.
        default: Fallback when the key is missing.
    """
    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, dotted)
        return default if value is None else value
    current: Any = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def _empty_phylogeny_table(
    manifest: pd.DataFrame, *, backend: str, qc_status: str, backend_trace: str
) -> pd.DataFrame:
    """Return one empty phylogeny row per lesion.

    Args:
        manifest: Cleaned lesion manifest.
        backend: Backend name.
        qc_status: Uniform QC flag.
        backend_trace: Uniform provenance string.
    """
    frame = manifest[["lesion_id", "patient_id", "donor_id", "stage"]].copy()
    for column in PHYLOGENY_SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["tree_available"] = False
    frame["phylogeny_qc_flag"] = qc_status
    frame["backend_used"] = backend
    frame["backend_trace"] = backend_trace
    return frame.loc[:, list(PHYLOGENY_SUMMARY_COLUMNS)]


def _normalize_phylogeny_summary(
    frame: pd.DataFrame, manifest: pd.DataFrame, *, backend: str
) -> pd.DataFrame:
    """Normalize a parse-only phylogeny summary into the common lesion schema.

    Args:
        frame: Parsed backend summary.
        manifest: Cleaned lesion manifest.
        backend: Backend used.
    """
    aliases = {
        "sample": "lesion_id",
        "patient": "patient_id",
        "trunk_burden": "trunk_mutation_burden",
        "clone_sharing": "clone_sharing_score",
        "descendant_sharing": "descendant_sharing_score",
        "progression_link": "evidence_of_progression_link",
    }
    normalized = frame.rename(columns=aliases).copy()
    merged = manifest[["lesion_id", "patient_id", "donor_id", "stage"]].merge(
        normalized,
        on=["lesion_id", "patient_id"],
        how="left",
    )
    for column in [
        "trunk_mutation_burden",
        "branch_count",
        "branch_length_mean",
        "clone_sharing_score",
        "descendant_sharing_score",
        "trunk_membership_score",
        "branch_specificity_score",
        "evidence_of_progression_link",
    ]:
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce")
    merged["tree_available"] = (
        merged.get("tree_available", pd.Series([True] * merged.shape[0]))
        .fillna(False)
        .astype(bool)
    )
    merged["phylogeny_qc_flag"] = merged.get(
        "phylogeny_qc_flag", pd.Series(["parsed_existing"] * merged.shape[0])
    )
    merged["backend_used"] = merged.get("backend_used", pd.Series([backend] * merged.shape[0]))
    merged["backend_trace"] = (
        merged["backend_used"].astype(str) + ":" + merged["phylogeny_qc_flag"].astype(str)
    )
    return merged.loc[:, list(PHYLOGENY_SUMMARY_COLUMNS)]


def run_phylogeny_backend(
    cfg: DictConfig | dict[str, Any], manifest: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run or parse the configured phylogeny backend.

    Args:
        cfg: Active config tree.
        manifest: Cleaned lesion manifest.
    """
    backend = str(_cfg_select(cfg, "labels.selected_phylogeny_backend", "none")).lower()
    parse_only = bool(_cfg_select(cfg, "labels.parse_only", True))
    dry_run = bool(_cfg_select(cfg, "labels.dry_run", False))
    if backend == "none":
        return _empty_phylogeny_table(
            manifest,
            backend="none",
            qc_status="backend_not_requested",
            backend_trace="none:not_requested",
        ), {"backend": "none", "status": "skipped"}

    summary_key = f"labels.inputs.phylogeny.{backend}_summary_path"
    summary_path_raw = _cfg_select(cfg, summary_key, None)
    if summary_path_raw:
        summary_path = Path(str(summary_path_raw))
        if summary_path.exists():
            parsed = (
                pd.read_parquet(summary_path)
                if summary_path.suffix.lower() == ".parquet"
                else pd.read_csv(summary_path)
            )
            return _normalize_phylogeny_summary(parsed, manifest, backend=backend), {
                "backend": backend,
                "status": "parsed_existing",
                "summary_path": str(summary_path),
            }
        if parse_only:
            raise FileNotFoundError(
                f"Configured {backend} phylogeny summary does not exist: {summary_path}"
            )

    if parse_only:
        return _empty_phylogeny_table(
            manifest,
            backend=backend,
            qc_status="missing_backend_output",
            backend_trace=f"{backend}:parse_only_missing",
        ), {"backend": backend, "status": "missing_parse_only_input"}

    executable_key = f"labels.external_tools.{backend}_executable"
    command_key = f"labels.external_tools.{backend}_command_template"
    executable = str(_cfg_select(cfg, executable_key, backend))
    command_template = _cfg_select(cfg, command_key, None)
    if not command_template:
        raise ValueError(f"External {backend} mode requires {command_key}.")
    artifacts_root = (
        Path(str(_cfg_select(cfg, "labels.artifacts_root", "reports/labels/artifacts"))) / backend
    )
    result = run_external_command(
        ToolCommand(
            name=backend,
            executable=executable,
            args=tuple(str(part) for part in str(command_template).split()),
            workdir=artifacts_root,
            timeout_seconds=int(_cfg_select(cfg, "labels.external_tools.timeout_seconds", 3600)),
            retries=int(_cfg_select(cfg, "labels.external_tools.retries", 0)),
            log_path=artifacts_root / "command.log",
        ),
        dry_run=dry_run,
        resume=bool(_cfg_select(cfg, "labels.resume", True)),
    )
    return _empty_phylogeny_table(
        manifest,
        backend=backend,
        qc_status=result.status,
        backend_trace=result.backend_trace,
    ), {"backend": backend, "status": result.status, "message": result.message}
