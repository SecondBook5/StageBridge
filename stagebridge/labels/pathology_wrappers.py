"""Optional pathology and region-level evidence ingestion for label repair."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.labels.common_schema import PATHOLOGY_SUMMARY_COLUMNS


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


def run_pathology_backend(cfg: DictConfig | dict[str, Any], manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse optional QuPath or QuST lesion summaries.

    Args:
        cfg: Active config tree.
        manifest: Cleaned lesion manifest.
    """
    backend = str(_cfg_select(cfg, "labels.selected_pathology_backend", "none")).lower()
    frame = manifest[["lesion_id", "sample_id", "patient_id", "donor_id", "stage"]].copy()
    for column in PATHOLOGY_SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    if backend == "none":
        frame["pathology_qc_flag"] = "backend_not_requested"
        frame["backend_used"] = "none"
        frame["backend_trace"] = "none:not_requested"
        return frame.loc[:, list(PATHOLOGY_SUMMARY_COLUMNS)], {"backend": "none", "status": "skipped"}

    summary_path_raw = _cfg_select(cfg, f"labels.inputs.pathology.{backend}_summary_path", None)
    if not summary_path_raw:
        frame["pathology_qc_flag"] = "missing_backend_output"
        frame["backend_used"] = backend
        frame["backend_trace"] = f"{backend}:parse_only_missing"
        return frame.loc[:, list(PATHOLOGY_SUMMARY_COLUMNS)], {"backend": backend, "status": "missing_parse_only_input"}

    summary_path = Path(str(summary_path_raw))
    if not summary_path.exists():
        raise FileNotFoundError(f"Configured pathology summary does not exist: {summary_path}")
    parsed = pd.read_parquet(summary_path) if summary_path.suffix.lower() == ".parquet" else pd.read_csv(summary_path)
    aliases = {
        "sample": "sample_id",
        "lesion": "lesion_id",
        "patient": "patient_id",
        "risk_score": "pathology_risk_score",
        "stromal_support": "stromal_support_score",
        "angiogenic_support": "angiogenic_support_score",
    }
    parsed = parsed.rename(columns=aliases).copy()
    if "lesion_id" not in parsed.columns and "sample_id" in parsed.columns:
        parsed["lesion_id"] = parsed["sample_id"].astype(str)
    merged = frame.merge(parsed, on="lesion_id", how="left", suffixes=("", "_parsed"))
    for column in [
        "pathology_risk_score",
        "invasive_pattern_support",
        "stromal_support_score",
        "angiogenic_support_score",
    ]:
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce")
    merged["pathology_qc_flag"] = merged.get("pathology_qc_flag", pd.Series(["parsed_existing"] * merged.shape[0]))
    merged["backend_used"] = merged.get("backend_used", pd.Series([backend] * merged.shape[0]))
    merged["backend_trace"] = merged["backend_used"].astype(str) + ":" + merged["pathology_qc_flag"].astype(str)
    return merged.loc[:, list(PATHOLOGY_SUMMARY_COLUMNS)], {"backend": backend, "status": "parsed_existing", "summary_path": str(summary_path)}
