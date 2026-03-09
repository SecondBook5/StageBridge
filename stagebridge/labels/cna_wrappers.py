"""CNA backend wrappers for FACETS, CNVkit, and Sequenza."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.labels.common_schema import CNA_SUMMARY_COLUMNS, ToolCommand, empty_frame
from stagebridge.labels.tool_runner import run_external_command


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    """Read one dotted config key from OmegaConf or a plain dictionary.

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


def _normalize_numeric(series: pd.Series) -> pd.Series:
    """Convert a column into a numeric float series with NaNs for bad values.

    Args:
        series: Input pandas series.
    """
    return pd.to_numeric(series, errors="coerce").astype(float)


def _normalize_cna_frame(frame: pd.DataFrame, *, backend: str, manifest: pd.DataFrame) -> pd.DataFrame:
    """Map one backend-specific summary table into the normalized CNA schema.

    Args:
        frame: Parsed backend table.
        backend: Active CNA backend name.
        manifest: Cleaned lesion manifest used for lesion alignment.
    """
    if frame.empty:
        return empty_frame(CNA_SUMMARY_COLUMNS)

    aliases = {
        "sample": "sample_id",
        "lesion": "lesion_id",
        "patient": "patient_id",
        "purity_estimate": "purity",
        "ploidy_estimate": "ploidy",
        "fga": "fraction_genome_altered",
        "focal_events": "num_focal_events",
        "arm_events": "num_arm_level_events",
        "backend": "backend_used",
    }
    normalized = frame.rename(columns=aliases).copy()
    if "lesion_id" not in normalized.columns and "sample_id" in normalized.columns:
        normalized["lesion_id"] = normalized["sample_id"].astype(str)
    normalized = manifest[
        ["lesion_id", "sample_id", "patient_id", "donor_id", "stage"]
    ].merge(normalized, on=["lesion_id"], how="left", suffixes=("", "_parsed"))
    if "sample_id_parsed" in normalized.columns:
        normalized["sample_id"] = normalized["sample_id"].fillna(normalized["sample_id_parsed"])
    if "patient_id_parsed" in normalized.columns:
        normalized["patient_id"] = normalized["patient_id"].fillna(normalized["patient_id_parsed"])
    for source, target in {
        "purity": "purity",
        "ploidy": "ploidy",
        "fraction_genome_altered": "fraction_genome_altered",
        "cna_burden": "cna_burden",
        "num_focal_events": "num_focal_events",
        "num_arm_level_events": "num_arm_level_events",
        "allele_specific_imbalance": "allele_specific_imbalance",
        "major_copy_summary": "major_copy_summary",
        "minor_copy_summary": "minor_copy_summary",
    }.items():
        if source not in normalized.columns:
            normalized[source] = pd.NA
        if target in {"major_copy_summary", "minor_copy_summary"}:
            normalized[target] = normalized[source]
        else:
            normalized[target] = _normalize_numeric(normalized[source])
    normalized["qc_status"] = normalized.get("qc_status", pd.Series(["missing_backend_output"] * normalized.shape[0]))
    normalized["backend_used"] = normalized.get("backend_used", pd.Series([backend] * normalized.shape[0]))
    normalized["backend_trace"] = normalized["backend_used"].astype(str) + ":" + normalized["qc_status"].astype(str)
    return normalized.loc[:, list(CNA_SUMMARY_COLUMNS)].copy()


def _parse_summary_path(path: Path) -> pd.DataFrame:
    """Load a parse-only backend summary from CSV/TSV/Parquet.

    Args:
        path: Existing summary artifact path.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def run_cna_backend(cfg: DictConfig | dict[str, Any], manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run or parse the configured CNA backend into a normalized summary table.

    Args:
        cfg: Active config tree.
        manifest: Cleaned lesion manifest.
    """
    backend = str(_cfg_select(cfg, "labels.selected_cna_backend", "none")).lower()
    parse_only = bool(_cfg_select(cfg, "labels.parse_only", True))
    dry_run = bool(_cfg_select(cfg, "labels.dry_run", False))
    artifacts_root = Path(str(_cfg_select(cfg, "labels.artifacts_root", "reports/labels/artifacts")))
    inputs = {
        "facets": _cfg_select(cfg, "labels.inputs.cna.facets_summary_path", None),
        "cnvkit": _cfg_select(cfg, "labels.inputs.cna.cnvkit_summary_path", None),
        "sequenza": _cfg_select(cfg, "labels.inputs.cna.sequenza_summary_path", None),
    }
    if backend == "none":
        empty = manifest[["lesion_id", "sample_id", "patient_id", "donor_id", "stage"]].copy()
        for column in CNA_SUMMARY_COLUMNS:
            if column not in empty.columns:
                empty[column] = pd.NA
        empty["qc_status"] = "backend_not_requested"
        empty["backend_used"] = "none"
        empty["backend_trace"] = "none:not_requested"
        return empty.loc[:, list(CNA_SUMMARY_COLUMNS)], {"backend": backend, "status": "skipped"}

    summary_path_raw = inputs.get(backend)
    if summary_path_raw:
        summary_path = Path(str(summary_path_raw))
        if summary_path.exists():
            parsed = _parse_summary_path(summary_path)
            return _normalize_cna_frame(parsed, backend=backend, manifest=manifest), {
                "backend": backend,
                "status": "parsed_existing",
                "summary_path": str(summary_path),
            }
        if parse_only:
            raise FileNotFoundError(f"Configured {backend} parse-only summary does not exist: {summary_path}")

    if parse_only:
        empty = manifest[["lesion_id", "sample_id", "patient_id", "donor_id", "stage"]].copy()
        for column in CNA_SUMMARY_COLUMNS:
            if column not in empty.columns:
                empty[column] = pd.NA
        empty["qc_status"] = "missing_backend_output"
        empty["backend_used"] = backend
        empty["backend_trace"] = f"{backend}:parse_only_missing"
        return empty.loc[:, list(CNA_SUMMARY_COLUMNS)], {"backend": backend, "status": "missing_parse_only_input"}

    executable = str(_cfg_select(cfg, f"labels.external_tools.{backend}_executable", backend))
    command_template = _cfg_select(cfg, f"labels.external_tools.{backend}_command_template", None)
    if not command_template:
        raise ValueError(
            f"{backend} external mode requires labels.external_tools.{backend}_command_template when parse_only=false."
        )
    command = ToolCommand(
        name=backend,
        executable=executable,
        args=tuple(str(part) for part in str(command_template).split()),
        workdir=artifacts_root / backend,
        timeout_seconds=int(_cfg_select(cfg, "labels.external_tools.timeout_seconds", 3600)),
        retries=int(_cfg_select(cfg, "labels.external_tools.retries", 0)),
        log_path=artifacts_root / backend / "command.log",
    )
    result = run_external_command(command, dry_run=dry_run, resume=bool(_cfg_select(cfg, "labels.resume", True)))
    empty = manifest[["lesion_id", "sample_id", "patient_id", "donor_id", "stage"]].copy()
    for column in CNA_SUMMARY_COLUMNS:
        if column not in empty.columns:
            empty[column] = pd.NA
    empty["qc_status"] = result.status
    empty["backend_used"] = backend
    empty["backend_trace"] = result.backend_trace
    return empty.loc[:, list(CNA_SUMMARY_COLUMNS)], {
        "backend": backend,
        "status": result.status,
        "message": result.message,
        "log_path": None if result.stdout_path is None else str(result.stdout_path),
    }
