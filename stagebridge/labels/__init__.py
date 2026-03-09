"""Label-repair and target-selection workflow for StageBridge.

This package focuses on repairing or falsifying weak lesion-level targets.
It is intentionally separate from predictive model code so label support can
be audited before running new learning benchmarks.
"""
from .cohort_manifest import build_cleaned_cohort_manifest
from .cna_wrappers import run_cna_backend
from .clonal_wrappers import run_clonal_backend
from .common_schema import (
    CLONAL_SUMMARY_COLUMNS,
    CNA_SUMMARY_COLUMNS,
    DONOR_SUPPORT_COLUMNS,
    PATHOLOGY_SUMMARY_COLUMNS,
    PHYLOGENY_SUMMARY_COLUMNS,
    REFINED_LABEL_COLUMNS,
)
from .label_refinement import refine_lesion_labels
from .pathology_wrappers import run_pathology_backend
from .phylogeny_wrappers import run_phylogeny_backend
from .reporting import generate_label_repair_reports
from .viability_checks import evaluate_label_support

__all__ = [
    "CLONAL_SUMMARY_COLUMNS",
    "CNA_SUMMARY_COLUMNS",
    "DONOR_SUPPORT_COLUMNS",
    "PATHOLOGY_SUMMARY_COLUMNS",
    "PHYLOGENY_SUMMARY_COLUMNS",
    "REFINED_LABEL_COLUMNS",
    "build_cleaned_cohort_manifest",
    "evaluate_label_support",
    "generate_label_repair_reports",
    "refine_lesion_labels",
    "run_clonal_backend",
    "run_cna_backend",
    "run_pathology_backend",
    "run_phylogeny_backend",
]
