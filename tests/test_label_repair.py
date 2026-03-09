from __future__ import annotations

from pathlib import Path

import pandas as pd

from stagebridge.labels.label_refinement import refine_lesion_labels
from stagebridge.labels.tool_runner import ToolCommand, run_external_command
from stagebridge.labels.viability_checks import evaluate_label_support


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lesion_id": "L1",
                "sample_id": "L1",
                "patient_id": "P1",
                "donor_id": "P1",
                "stage": "AAH",
                "edge_label": "AAH->AIS",
                "original_label": 1.0,
                "original_label_weight": 0.5,
                "original_label_source": "heuristic_edge_expansion",
                "original_label_notes": "heuristic",
                "has_spatial": True,
                "has_wes": True,
                "num_spots": 10,
                "num_patient_lesions": 2,
                "num_patient_stages": 2,
                "can_support_phylogeny": True,
                "availability_trace": "spatial;wes;non_curated_label;later_stage;phylogeny_ready",
            },
            {
                "lesion_id": "L2",
                "sample_id": "L2",
                "patient_id": "P2",
                "donor_id": "P2",
                "stage": "AAH",
                "edge_label": "AAH->AIS",
                "original_label": 0.0,
                "original_label_weight": 1.0,
                "original_label_source": "peng_fig_s3_pattern2",
                "original_label_notes": "curated negative",
                "has_spatial": True,
                "has_wes": True,
                "num_spots": 10,
                "num_patient_lesions": 2,
                "num_patient_stages": 2,
                "can_support_phylogeny": True,
                "availability_trace": "spatial;wes;curated_label;later_stage;phylogeny_ready",
            },
        ]
    )


def test_refinement_marks_heuristic_positive_uncertain_without_nonproxy() -> None:
    manifest = _manifest()
    wes = pd.DataFrame(
        [
            {"patient_id": "P1", "stage": "AAH", "tmb": 10.0, "kras_mut": 0.0, "egfr_mut": 0.0, "tp53_mut": 0.0, "stk11_mut": 0.0, "keap1_mut": 0.0, "smad4_mut": 0.0, "braf_mut": 0.0},
            {"patient_id": "P2", "stage": "AAH", "tmb": 2.0, "kras_mut": 0.0, "egfr_mut": 0.0, "tp53_mut": 0.0, "stk11_mut": 0.0, "keap1_mut": 0.0, "smad4_mut": 0.0, "braf_mut": 0.0},
        ]
    )
    empty = pd.DataFrame({"lesion_id": ["L1", "L2"]})
    refined = refine_lesion_labels(
        manifest,
        cna_summary=empty,
        clonal_summary=empty,
        phylogeny_summary=empty,
        pathology_summary=empty,
        wes_features=wes,
        cfg={"labels": {"thresholds": {"require_non_proxy_for_heuristic_positive": True}}},
    )
    row = refined.loc[refined["lesion_id"] == "L1"].iloc[0]
    assert row["refined_binary_label"] == "uncertain"
    assert bool(row["uncertainty_flag"]) is True


def test_viability_checker_rejects_single_negative_donor_binary_task() -> None:
    manifest = _manifest()
    refined = pd.DataFrame(
        [
            {
                "lesion_id": "L1",
                "sample_id": "L1",
                "patient_id": "P1",
                "donor_id": "P1",
                "stage": "AAH",
                "edge_label": "AAH->AIS",
                "original_label": 1.0,
                "refined_binary_label": "positive",
                "uncertainty_flag": False,
                "exclusion_flag": False,
                "progression_risk_score": 0.9,
                "confidence_tier": "high",
                "top_evidence_reasons": "",
                "top_contraindications": "",
                "backend_trace": "test",
            },
            {
                "lesion_id": "L2",
                "sample_id": "L2",
                "patient_id": "P2",
                "donor_id": "P2",
                "stage": "AAH",
                "edge_label": "AAH->AIS",
                "original_label": 0.0,
                "refined_binary_label": "negative",
                "uncertainty_flag": False,
                "exclusion_flag": False,
                "progression_risk_score": 0.1,
                "confidence_tier": "high",
                "top_evidence_reasons": "",
                "top_contraindications": "",
                "backend_trace": "test",
            },
            {
                "lesion_id": "L3",
                "sample_id": "L3",
                "patient_id": "P3",
                "donor_id": "P3",
                "stage": "AAH",
                "edge_label": "AAH->AIS",
                "original_label": 1.0,
                "refined_binary_label": "positive",
                "uncertainty_flag": False,
                "exclusion_flag": False,
                "progression_risk_score": 0.8,
                "confidence_tier": "high",
                "top_evidence_reasons": "",
                "top_contraindications": "",
                "backend_trace": "test",
            },
        ]
    )
    edge_support, donor_support, report = evaluate_label_support(
        manifest,
        refined,
        {"labels": {"viability": {"num_folds": 3}}},
    )
    row = edge_support.iloc[0]
    assert bool(row["binary_viable"]) is False
    assert row["recommended_target"] in {"continuous_risk", "descriptive_only"}
    assert report["edges"]["AAH->AIS"]["binary_viable"] is False
    assert donor_support["donor_id"].nunique() == 3


def test_tool_runner_reports_missing_executable() -> None:
    result = run_external_command(
        ToolCommand(
            name="fake_tool",
            executable="definitely-not-a-real-executable-stagebridge",
            args=(),
            workdir=Path("."),
        ),
        dry_run=False,
    )
    assert result.status == "missing_executable"
    assert "not found" in result.message
