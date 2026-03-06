#!/usr/bin/env python
"""Data contract audit for StageBridge datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stagebridge import config as sb_config
from stagebridge.io.hlca import hlca_reference_h5ad
from stagebridge.preprocessing.stage_ontology import CANONICAL_STAGE_ORDER, normalize_stage_series
from stagebridge.utils.types import DatasetAuditReport


def _load_anndata(path: Path) -> Any:
    import anndata

    return anndata.read_h5ad(path, backed="r")


def _stage_check(obs, stage_col: str) -> tuple[bool, list[str]]:
    if stage_col not in obs.columns:
        return False, [f"missing stage column '{stage_col}'"]
    normalized = set(normalize_stage_series(obs[stage_col]).tolist())
    bad = sorted([s for s in normalized if s not in CANONICAL_STAGE_ORDER])
    if bad:
        return False, [f"non-canonical stages present: {bad}"]
    missing = [s for s in CANONICAL_STAGE_ORDER if s not in normalized]
    issues: list[str] = []
    if missing:
        issues.append(f"missing canonical stages: {missing}")
    return len(issues) == 0, issues


def _column_check(obs, required: list[str], alternatives: dict[str, list[str]] | None = None) -> tuple[bool, list[str]]:
    alternatives = alternatives or {}
    issues: list[str] = []
    for col in required:
        if col in obs.columns:
            continue
        alts = alternatives.get(col, [])
        present_alt = [a for a in alts if a in obs.columns]
        if not present_alt:
            issues.append(f"missing required column '{col}'")
    return len(issues) == 0, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit StageBridge data contracts")
    parser.add_argument("--snrna", type=Path, default=sb_config.snrna_merged_h5ad())
    parser.add_argument("--spatial", type=Path, default=sb_config.spatial_merged_h5ad())
    parser.add_argument("--hlca", type=Path, default=hlca_reference_h5ad())
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/data_audit.json"))
    parser.add_argument(
        "--allow-noncanonical-stages",
        action="store_true",
        help="Do not fail audit on non-canonical stages (still reported as issues).",
    )
    args = parser.parse_args()

    issues: list[str] = []

    snrna_exists = args.snrna.exists()
    spatial_exists = args.spatial.exists()
    hlca_exists = args.hlca.exists()

    snrna_shape = spatial_shape = hlca_shape = None
    required_ok = True
    stage_ok = True

    if not snrna_exists:
        issues.append(f"snRNA file missing: {args.snrna}")
    if not spatial_exists:
        issues.append(f"spatial file missing: {args.spatial}")
    if not hlca_exists:
        issues.append(f"HLCA file missing: {args.hlca}")

    if snrna_exists:
        ad = _load_anndata(args.snrna)
        snrna_shape = tuple(int(v) for v in ad.shape)
        ok_cols, col_issues = _column_check(ad.obs, required=["stage", "patient_id", "sample_id"])
        required_ok = required_ok and ok_cols
        issues.extend([f"snRNA: {x}" for x in col_issues])
        ok_stage, stage_issues = _stage_check(ad.obs, "stage")
        if not args.allow_noncanonical_stages:
            stage_ok = stage_ok and ok_stage
        issues.extend([f"snRNA: {x}" for x in stage_issues])
        ad.file.close()

    if spatial_exists:
        ad = _load_anndata(args.spatial)
        spatial_shape = tuple(int(v) for v in ad.shape)
        ok_cols, col_issues = _column_check(ad.obs, required=["stage", "patient_id", "sample_id"])
        required_ok = required_ok and ok_cols
        issues.extend([f"spatial: {x}" for x in col_issues])
        ok_stage, stage_issues = _stage_check(ad.obs, "stage")
        if not args.allow_noncanonical_stages:
            stage_ok = stage_ok and ok_stage
        issues.extend([f"spatial: {x}" for x in stage_issues])
        ad.file.close()

    if hlca_exists:
        ad = _load_anndata(args.hlca)
        hlca_shape = tuple(int(v) for v in ad.shape)
        ok_cols, col_issues = _column_check(
            ad.obs,
            required=["stage", "patient_id", "sample_id"],
            alternatives={
                "patient_id": ["donor_id", "donor"],
                "sample_id": ["sample", "sample_name"],
            },
        )
        required_ok = required_ok and ok_cols
        issues.extend([f"HLCA: {x}" for x in col_issues])
        if "stage" in ad.obs.columns:
            ok_stage, stage_issues = _stage_check(ad.obs, "stage")
            if not args.allow_noncanonical_stages:
                stage_ok = stage_ok and ok_stage
            issues.extend([f"HLCA: {x}" for x in stage_issues])
        ad.file.close()

    report = DatasetAuditReport(
        snrna_path=str(args.snrna),
        spatial_path=str(args.spatial),
        hlca_path=str(args.hlca),
        snrna_exists=snrna_exists,
        spatial_exists=spatial_exists,
        hlca_exists=hlca_exists,
        snrna_shape=snrna_shape,
        spatial_shape=spatial_shape,
        hlca_shape=hlca_shape,
        required_obs_columns_ok=required_ok,
        canonical_stage_values_ok=stage_ok,
        issues=issues,
    )

    payload = report.to_dict()
    payload["ok"] = (
        payload["snrna_exists"]
        and payload["spatial_exists"]
        and payload["hlca_exists"]
        and payload["required_obs_columns_ok"]
        and payload["canonical_stage_values_ok"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(json.dumps({"ok": payload["ok"], "output": str(args.output), "issues": len(issues)}, indent=2))


if __name__ == "__main__":
    main()
