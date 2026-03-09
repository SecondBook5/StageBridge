"""PyClone-VI wrapper and lesion-level clonal summary normalization."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.labels.common_schema import CLONAL_SUMMARY_COLUMNS
from stagebridge.labels.tool_runner import run_external_command
from stagebridge.labels.common_schema import ToolCommand


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    """Read a dotted config value from OmegaConf or dict payloads.

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


def _empty_clonal_table(manifest: pd.DataFrame, *, backend: str, qc_status: str, backend_trace: str) -> pd.DataFrame:
    """Return an aligned empty clonal summary frame for every lesion.

    Args:
        manifest: Cleaned lesion manifest.
        backend: Backend name.
        qc_status: Uniform QC flag.
        backend_trace: Uniform provenance string.
    """
    frame = manifest[["lesion_id", "sample_id", "patient_id", "donor_id", "stage"]].copy()
    for column in CLONAL_SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["qc_status"] = qc_status
    frame["backend_used"] = backend
    frame["backend_trace"] = backend_trace
    return frame.loc[:, list(CLONAL_SUMMARY_COLUMNS)]


def _normalize_clonal_summary(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Normalize a parse-only PyClone-VI summary into the common lesion schema.

    Args:
        frame: Parsed PyClone-VI summary table.
        manifest: Cleaned lesion manifest.
    """
    aliases = {
        "sample": "sample_id",
        "lesion": "lesion_id",
        "patient": "patient_id",
        "num_clusters": "num_clonal_clusters",
        "dominant_fraction": "dominant_clone_fraction",
        "subclone_entropy": "subclonal_entropy",
        "shared_cluster_count": "shared_cluster_count_with_later_lesions",
        "private_cluster_count": "private_cluster_count",
        "driver_clusters": "driver_cluster_count",
    }
    normalized = frame.rename(columns=aliases).copy()
    if "lesion_id" not in normalized.columns and "sample_id" in normalized.columns:
        normalized["lesion_id"] = normalized["sample_id"].astype(str)
    merged = manifest[["lesion_id", "sample_id", "patient_id", "donor_id", "stage"]].merge(
        normalized,
        on="lesion_id",
        how="left",
    )
    for column in [
        "num_clonal_clusters",
        "dominant_clone_fraction",
        "subclonal_entropy",
        "shared_cluster_count_with_later_lesions",
        "private_cluster_count",
        "driver_cluster_count",
    ]:
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce")
    merged["qc_status"] = merged.get("qc_status", pd.Series(["parsed_existing"] * merged.shape[0]))
    merged["backend_used"] = merged.get("backend_used", pd.Series(["pyclone_vi"] * merged.shape[0]))
    merged["backend_trace"] = merged["backend_used"].astype(str) + ":" + merged["qc_status"].astype(str)
    return merged.loc[:, list(CLONAL_SUMMARY_COLUMNS)]


def run_clonal_backend(cfg: DictConfig | dict[str, Any], manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run or parse the PyClone-VI clonal layer.

    Args:
        cfg: Active config tree.
        manifest: Cleaned lesion manifest.
    """
    parse_only = bool(_cfg_select(cfg, "labels.parse_only", True))
    dry_run = bool(_cfg_select(cfg, "labels.dry_run", False))
    summary_path_raw = _cfg_select(cfg, "labels.inputs.clonal.pyclone_summary_path", None)
    if summary_path_raw:
        summary_path = Path(str(summary_path_raw))
        if summary_path.exists():
            parsed = pd.read_parquet(summary_path) if summary_path.suffix.lower() == ".parquet" else pd.read_csv(summary_path)
            return _normalize_clonal_summary(parsed, manifest), {
                "backend": "pyclone_vi",
                "status": "parsed_existing",
                "summary_path": str(summary_path),
            }
        if parse_only:
            raise FileNotFoundError(f"Configured PyClone-VI summary does not exist: {summary_path}")

    if parse_only:
        return _empty_clonal_table(
            manifest,
            backend="pyclone_vi",
            qc_status="missing_backend_output",
            backend_trace="pyclone_vi:parse_only_missing",
        ), {"backend": "pyclone_vi", "status": "missing_parse_only_input"}

    executable = str(_cfg_select(cfg, "labels.external_tools.pyclone_vi_executable", "pyclone-vi"))
    command_template = _cfg_select(cfg, "labels.external_tools.pyclone_vi_command_template", None)
    if not command_template:
        raise ValueError("External PyClone-VI mode requires labels.external_tools.pyclone_vi_command_template.")
    artifacts_root = Path(str(_cfg_select(cfg, "labels.artifacts_root", "reports/labels/artifacts"))) / "pyclone_vi"
    result = run_external_command(
        ToolCommand(
            name="pyclone_vi",
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
    return _empty_clonal_table(
        manifest,
        backend="pyclone_vi",
        qc_status=result.status,
        backend_trace=result.backend_trace,
    ), {
        "backend": "pyclone_vi",
        "status": result.status,
        "message": result.message,
    }
