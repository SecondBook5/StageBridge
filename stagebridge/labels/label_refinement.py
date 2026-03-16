"""Refined binary labels and continuous targets for StageBridge label repair."""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.labels.common_schema import REFINED_LABEL_COLUMNS
from stagebridge.labels.risk_scoring import score_lesions, summarize_contributions


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


def refine_lesion_labels(
    manifest: pd.DataFrame,
    *,
    cna_summary: pd.DataFrame,
    clonal_summary: pd.DataFrame,
    phylogeny_summary: pd.DataFrame,
    pathology_summary: pd.DataFrame,
    wes_features: pd.DataFrame,
    cfg: DictConfig | dict[str, Any],
) -> pd.DataFrame:
    """Derive refined labels, risk scores, and uncertainty flags.

    Args:
        manifest: Cleaned lesion manifest.
        cna_summary: Normalized lesion-level CNA summary.
        clonal_summary: Normalized lesion-level clone summary.
        phylogeny_summary: Normalized lesion-level phylogeny summary.
        pathology_summary: Optional pathology summary.
        wes_features: Existing lesion-level WES proxy features.
        cfg: Active config tree.
    """
    merged = manifest.copy()
    merged = merged.loc[merged["edge_label"].astype(str).ne("")].reset_index(drop=True)
    wes_for_merge = wes_features.rename(
        columns={"stage": "stage", "patient_id": "patient_id"}
    ).copy()
    merged = merged.merge(
        wes_for_merge, on=["patient_id", "stage"], how="left", suffixes=("", "_wes")
    )
    merged = merged.merge(
        cna_summary.drop(
            columns=["sample_id", "patient_id", "donor_id", "stage"], errors="ignore"
        ),
        on="lesion_id",
        how="left",
        suffixes=("", "_cna"),
    )
    merged = merged.merge(
        clonal_summary.drop(
            columns=["sample_id", "patient_id", "donor_id", "stage"], errors="ignore"
        ),
        on="lesion_id",
        how="left",
        suffixes=("", "_clonal"),
    )
    merged = merged.merge(
        phylogeny_summary.drop(columns=["patient_id", "donor_id", "stage"], errors="ignore"),
        on="lesion_id",
        how="left",
        suffixes=("", "_phy"),
    )
    merged = merged.merge(
        pathology_summary.drop(
            columns=["sample_id", "patient_id", "donor_id", "stage"], errors="ignore"
        ),
        on="lesion_id",
        how="left",
        suffixes=("", "_path"),
    )

    patient_stage_sets = merged.groupby("patient_id", sort=False)["stage"].agg(
        lambda values: tuple(sorted({str(v) for v in values}))
    )
    has_later_stage = []
    stage_order = {"Normal": 0, "AAH": 1, "AIS": 2, "MIA": 3, "LUAD": 4}
    for row in merged.itertuples(index=False):
        patient_stages = patient_stage_sets.get(str(row.patient_id), ())
        current_rank = stage_order.get(str(row.stage), -1)
        has_later_stage.append(
            any(stage_order.get(stage, -1) > current_rank for stage in patient_stages)
        )
    merged["has_later_stage"] = has_later_stage

    scores, contributions = score_lesions(merged, cfg)
    positive_threshold = float(_cfg_select(cfg, "labels.thresholds.positive_score", 0.75))
    negative_threshold = float(_cfg_select(cfg, "labels.thresholds.negative_score", 0.25))
    margin = float(_cfg_select(cfg, "labels.thresholds.uncertainty_margin", 0.10))
    require_non_proxy_for_heuristic = bool(
        _cfg_select(cfg, "labels.thresholds.require_non_proxy_for_heuristic_positive", True)
    )

    refined_rows: list[dict[str, object]] = []
    for idx, row in merged.iterrows():
        score = float(scores.loc[idx])
        original_source = str(row.get("original_label_source", ""))
        original_label = row.get("original_label", pd.NA)
        is_curated = original_source.startswith("peng_")
        is_heuristic = original_source == "heuristic_edge_expansion"
        non_proxy_evidence = int(
            pd.notna(row.get("cna_burden"))
            or pd.notna(row.get("num_clonal_clusters"))
            or bool(row.get("tree_available", False))
            or pd.notna(row.get("pathology_risk_score"))
        )

        refined = "uncertain"
        exclude = False
        if pd.isna(original_label):
            exclude = True
            refined = "exclude"
        elif is_heuristic:
            if score >= positive_threshold and (
                not require_non_proxy_for_heuristic or non_proxy_evidence > 0
            ):
                refined = "positive"
            elif score <= negative_threshold and non_proxy_evidence > 0:
                refined = "negative"
            else:
                refined = "uncertain"
        elif is_curated and float(original_label) == 0.0:
            refined = "negative"
        elif (
            is_curated
            and float(original_label) == 1.0
            and score >= max(0.5, positive_threshold - margin)
        ) or score >= positive_threshold:
            refined = "positive"
        elif score <= negative_threshold:
            refined = "negative"
        else:
            refined = "uncertain"

        confidence = "low"
        if refined in {"positive", "negative"} and is_curated:
            confidence = "high"
        elif refined in {"positive", "negative"}:
            confidence = "intermediate"
        if refined == "uncertain":
            confidence = "low"
        evidence = summarize_contributions(contributions.loc[idx], positive=True)
        contraindications = summarize_contributions(contributions.loc[idx], positive=False)
        backend_trace = ";".join(
            value
            for value in [
                str(row.get("backend_trace", "")),
                str(row.get("backend_trace_clonal", "")),
                str(row.get("backend_trace_phy", "")),
                str(row.get("backend_trace_path", "")),
            ]
            if value and value != "nan"
        )
        refined_rows.append(
            {
                "lesion_id": row["lesion_id"],
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "donor_id": row["donor_id"],
                "stage": row["stage"],
                "edge_label": row["edge_label"],
                "original_label": row["original_label"],
                "refined_binary_label": refined,
                "uncertainty_flag": refined == "uncertain",
                "exclusion_flag": exclude or refined == "exclude",
                "progression_risk_score": score,
                "confidence_tier": confidence,
                "top_evidence_reasons": evidence,
                "top_contraindications": contraindications,
                "backend_trace": backend_trace or "wes_proxy_only",
            }
        )
    return pd.DataFrame(refined_rows, columns=list(REFINED_LABEL_COLUMNS))
