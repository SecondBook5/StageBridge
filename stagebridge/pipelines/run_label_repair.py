"""Target-repair and target-selection pipeline for weak lesion labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.labels import (
    build_cleaned_cohort_manifest,
    evaluate_label_support,
    generate_label_repair_reports,
    refine_lesion_labels,
    run_clonal_backend,
    run_cna_backend,
    run_pathology_backend,
    run_phylogeny_backend,
)
from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


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


def _ensure_dir(path: str | Path) -> Path:
    """Create a directory when missing and return the resolved path.

    Args:
        path: Directory path to create.
    """
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_label_manifest(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Build and save the cleaned cohort manifest for target repair.

    Args:
        cfg: Active config tree.
    """
    reports_root = _ensure_dir(_cfg_select(cfg, "labels.output_root", "reports/labels"))
    tables_root = _ensure_dir(reports_root / "tables")
    outputs = build_cleaned_cohort_manifest(cfg)
    outputs["cleaned_manifest"].to_csv(tables_root / "cleaned_cohort_manifest.csv", index=False)
    outputs["sample_to_lesion"].to_csv(tables_root / "sample_to_lesion_mapping.csv", index=False)
    outputs["donor_summary"].to_csv(tables_root / "donor_patient_summary.csv", index=False)
    outputs["availability_matrix"].to_csv(
        tables_root / "data_availability_matrix.csv", index=False
    )
    return {
        "ok": True,
        "pipeline": "label_manifest",
        "status": "complete",
        "reports_root": str(reports_root),
        "tables_root": str(tables_root),
    }


def run_label_cna(
    cfg: DictConfig | dict[str, Any], *, manifest: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the configured CNA backend and save its normalized summary.

    Args:
        cfg: Active config tree.
        manifest: Optional cleaned cohort manifest.
    """
    reports_root = _ensure_dir(_cfg_select(cfg, "labels.output_root", "reports/labels"))
    tables_root = _ensure_dir(reports_root / "tables")
    active_manifest = (
        manifest
        if manifest is not None
        else build_cleaned_cohort_manifest(cfg)["cleaned_manifest"]
    )
    summary, meta = run_cna_backend(cfg, active_manifest)
    summary.to_csv(tables_root / "lesion_cna_summary.csv", index=False)
    return summary, meta


def run_label_clonal(
    cfg: DictConfig | dict[str, Any], *, manifest: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the configured clonal backend and save its normalized summary.

    Args:
        cfg: Active config tree.
        manifest: Optional cleaned cohort manifest.
    """
    reports_root = _ensure_dir(_cfg_select(cfg, "labels.output_root", "reports/labels"))
    tables_root = _ensure_dir(reports_root / "tables")
    active_manifest = (
        manifest
        if manifest is not None
        else build_cleaned_cohort_manifest(cfg)["cleaned_manifest"]
    )
    summary, meta = run_clonal_backend(cfg, active_manifest)
    summary.to_csv(tables_root / "lesion_clone_summary.csv", index=False)
    return summary, meta


def run_label_phylogeny(
    cfg: DictConfig | dict[str, Any], *, manifest: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the configured phylogeny backend and save its normalized summary.

    Args:
        cfg: Active config tree.
        manifest: Optional cleaned cohort manifest.
    """
    reports_root = _ensure_dir(_cfg_select(cfg, "labels.output_root", "reports/labels"))
    tables_root = _ensure_dir(reports_root / "tables")
    active_manifest = (
        manifest
        if manifest is not None
        else build_cleaned_cohort_manifest(cfg)["cleaned_manifest"]
    )
    summary, meta = run_phylogeny_backend(cfg, active_manifest)
    summary.to_csv(tables_root / "lesion_phylogeny_summary.csv", index=False)
    return summary, meta


def run_label_refinement(
    cfg: DictConfig | dict[str, Any], *, cached: dict[str, pd.DataFrame] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refine lesion labels and save risk-score outputs.

    Args:
        cfg: Active config tree.
        cached: Optional intermediate tables keyed by stage name.
    """
    reports_root = _ensure_dir(_cfg_select(cfg, "labels.output_root", "reports/labels"))
    tables_root = _ensure_dir(reports_root / "tables")
    manifest_outputs = build_cleaned_cohort_manifest(cfg) if cached is None else cached
    cna_summary, _ = run_label_cna(cfg, manifest=manifest_outputs["cleaned_manifest"])
    clonal_summary, _ = run_label_clonal(cfg, manifest=manifest_outputs["cleaned_manifest"])
    phylogeny_summary, _ = run_label_phylogeny(cfg, manifest=manifest_outputs["cleaned_manifest"])
    pathology_summary, _ = run_pathology_backend(cfg, manifest_outputs["cleaned_manifest"])
    refined = refine_lesion_labels(
        manifest_outputs["cleaned_manifest"],
        cna_summary=cna_summary,
        clonal_summary=clonal_summary,
        phylogeny_summary=phylogeny_summary,
        pathology_summary=pathology_summary,
        wes_features=manifest_outputs["wes_features"],
        cfg=cfg,
    )
    refined.to_csv(tables_root / "lesion_refined_labels.csv", index=False)
    refined.loc[
        :,
        [
            "lesion_id",
            "patient_id",
            "donor_id",
            "stage",
            "edge_label",
            "progression_risk_score",
            "confidence_tier",
        ],
    ].to_csv(
        tables_root / "lesion_progression_risk_scores.csv",
        index=False,
    )
    return refined, {
        "ok": True,
        "pipeline": "label_refinement",
        "status": "complete",
        "tables_root": str(tables_root),
    }


def run_label_support(
    cfg: DictConfig | dict[str, Any], *, cached: dict[str, pd.DataFrame] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate binary and continuous target support after refinement.

    Args:
        cfg: Active config tree.
        cached: Optional intermediate tables keyed by stage name.
    """
    reports_root = _ensure_dir(_cfg_select(cfg, "labels.output_root", "reports/labels"))
    tables_root = _ensure_dir(reports_root / "tables")
    artifacts_root = _ensure_dir(reports_root / "artifacts")
    manifest_outputs = build_cleaned_cohort_manifest(cfg) if cached is None else cached
    cna_summary, _ = run_label_cna(cfg, manifest=manifest_outputs["cleaned_manifest"])
    clonal_summary, _ = run_label_clonal(cfg, manifest=manifest_outputs["cleaned_manifest"])
    phylogeny_summary, _ = run_label_phylogeny(cfg, manifest=manifest_outputs["cleaned_manifest"])
    pathology_summary, _ = run_pathology_backend(cfg, manifest_outputs["cleaned_manifest"])
    refined = refine_lesion_labels(
        manifest_outputs["cleaned_manifest"],
        cna_summary=cna_summary,
        clonal_summary=clonal_summary,
        phylogeny_summary=phylogeny_summary,
        pathology_summary=pathology_summary,
        wes_features=manifest_outputs["wes_features"],
        cfg=cfg,
    )
    edge_support, donor_support, split_report = evaluate_label_support(
        manifest_outputs["cleaned_manifest"],
        refined,
        cfg,
    )
    edge_support.to_csv(tables_root / "edge_label_support_summary.csv", index=False)
    donor_support.to_csv(tables_root / "donor_support_summary.csv", index=False)
    (artifacts_root / "split_viability_report.json").write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )
    return edge_support, donor_support, split_report


def run_label_repair(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Run the full label-repair and target-selection workflow.

    Args:
        cfg: Active config tree.
    """
    reports_root = _ensure_dir(_cfg_select(cfg, "labels.output_root", "reports/labels"))
    manifest_outputs = build_cleaned_cohort_manifest(cfg)
    cna_summary, cna_meta = run_label_cna(cfg, manifest=manifest_outputs["cleaned_manifest"])
    clonal_summary, clonal_meta = run_label_clonal(
        cfg, manifest=manifest_outputs["cleaned_manifest"]
    )
    phylogeny_summary, phylo_meta = run_label_phylogeny(
        cfg, manifest=manifest_outputs["cleaned_manifest"]
    )
    pathology_summary, pathology_meta = run_pathology_backend(
        cfg, manifest_outputs["cleaned_manifest"]
    )
    pathology_summary.to_csv(
        _ensure_dir(reports_root / "tables") / "lesion_pathology_summary.csv", index=False
    )
    refined = refine_lesion_labels(
        manifest_outputs["cleaned_manifest"],
        cna_summary=cna_summary,
        clonal_summary=clonal_summary,
        phylogeny_summary=phylogeny_summary,
        pathology_summary=pathology_summary,
        wes_features=manifest_outputs["wes_features"],
        cfg=cfg,
    )
    edge_support, donor_support, split_report = evaluate_label_support(
        manifest_outputs["cleaned_manifest"],
        refined,
        cfg,
    )
    report_manifest = generate_label_repair_reports(
        cleaned_manifest=manifest_outputs["cleaned_manifest"],
        cna_summary=cna_summary,
        clonal_summary=clonal_summary,
        phylogeny_summary=phylogeny_summary,
        pathology_summary=pathology_summary,
        refined_labels=refined,
        edge_support=edge_support,
        donor_support=donor_support,
        split_report=split_report,
        output_root=reports_root,
        cfg=cfg,
    )
    manifest_payload = {
        "cna": cna_meta,
        "clonal": clonal_meta,
        "phylogeny": phylo_meta,
        "pathology": pathology_meta,
        "report_outputs": report_manifest,
    }
    (_ensure_dir(reports_root / "artifacts") / "label_repair_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )
    log.info("Label-repair workflow complete. Reports written to %s", reports_root)
    return {
        "ok": True,
        "pipeline": "label_repair",
        "status": "complete",
        "reports_root": str(reports_root),
        "recommended_targets": edge_support.loc[
            :, ["edge_label", "recommended_target", "reason"]
        ].to_dict(orient="records"),
    }


__all__ = [
    "run_label_cna",
    "run_label_clonal",
    "run_label_manifest",
    "run_label_phylogeny",
    "run_label_refinement",
    "run_label_repair",
    "run_label_support",
]
