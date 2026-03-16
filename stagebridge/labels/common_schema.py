"""Normalized schemas for the StageBridge label-repair workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


COHORT_MANIFEST_COLUMNS: tuple[str, ...] = (
    "lesion_id",
    "sample_id",
    "patient_id",
    "donor_id",
    "stage",
    "edge_label",
    "original_label",
    "original_label_weight",
    "original_label_source",
    "original_label_notes",
    "has_spatial",
    "has_wes",
    "num_spots",
    "num_patient_lesions",
    "num_patient_stages",
    "can_support_phylogeny",
    "availability_trace",
)

SAMPLE_TO_LESION_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "lesion_id",
    "patient_id",
    "donor_id",
    "stage",
    "edge_label",
)

DATA_AVAILABILITY_COLUMNS: tuple[str, ...] = (
    "lesion_id",
    "has_spatial",
    "has_wes",
    "has_curated_label",
    "has_heuristic_label",
    "can_support_phylogeny",
)

CNA_SUMMARY_COLUMNS: tuple[str, ...] = (
    "lesion_id",
    "sample_id",
    "patient_id",
    "donor_id",
    "stage",
    "purity",
    "ploidy",
    "fraction_genome_altered",
    "cna_burden",
    "num_focal_events",
    "num_arm_level_events",
    "allele_specific_imbalance",
    "major_copy_summary",
    "minor_copy_summary",
    "qc_status",
    "backend_used",
    "backend_trace",
)

CLONAL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "lesion_id",
    "sample_id",
    "patient_id",
    "donor_id",
    "stage",
    "num_clonal_clusters",
    "dominant_clone_fraction",
    "subclonal_entropy",
    "shared_cluster_count_with_later_lesions",
    "private_cluster_count",
    "driver_cluster_count",
    "qc_status",
    "backend_used",
    "backend_trace",
)

PHYLOGENY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "lesion_id",
    "patient_id",
    "donor_id",
    "stage",
    "tree_available",
    "trunk_mutation_burden",
    "branch_count",
    "branch_length_mean",
    "clone_sharing_score",
    "descendant_sharing_score",
    "trunk_membership_score",
    "branch_specificity_score",
    "evidence_of_progression_link",
    "phylogeny_qc_flag",
    "backend_used",
    "backend_trace",
)

PATHOLOGY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "lesion_id",
    "sample_id",
    "patient_id",
    "donor_id",
    "stage",
    "pathology_risk_score",
    "invasive_pattern_support",
    "stromal_support_score",
    "angiogenic_support_score",
    "pathology_qc_flag",
    "backend_used",
    "backend_trace",
)

REFINED_LABEL_COLUMNS: tuple[str, ...] = (
    "lesion_id",
    "sample_id",
    "patient_id",
    "donor_id",
    "stage",
    "edge_label",
    "original_label",
    "refined_binary_label",
    "uncertainty_flag",
    "exclusion_flag",
    "progression_risk_score",
    "confidence_tier",
    "top_evidence_reasons",
    "top_contraindications",
    "backend_trace",
)

EDGE_SUPPORT_COLUMNS: tuple[str, ...] = (
    "edge_label",
    "target_kind",
    "n_lesions",
    "n_donors",
    "positive_lesions",
    "negative_lesions",
    "uncertain_lesions",
    "excluded_lesions",
    "positive_donors",
    "negative_donors",
    "continuous_unique_scores",
    "binary_viable",
    "continuous_viable",
    "recommended_target",
    "reason",
)

DONOR_SUPPORT_COLUMNS: tuple[str, ...] = (
    "edge_label",
    "donor_id",
    "n_lesions",
    "positive_lesions",
    "negative_lesions",
    "uncertain_lesions",
    "excluded_lesions",
    "binary_support_status",
)


@dataclass(slots=True, frozen=True)
class ToolCommand:
    """Structured external command request for parse/run wrappers.

    This keeps backend execution auditable and serializable.

    Args:
        name: Human-readable backend name.
        executable: Executable to resolve on PATH or as an absolute path.
        args: Positional arguments that should follow the executable.
        workdir: Working directory for the command.
        timeout_seconds: Maximum runtime before the helper aborts.
        retries: Number of retry attempts after a failure.
        env: Optional environment overrides.
        log_path: Optional path to the command log file.
    """

    name: str
    executable: str
    args: tuple[str, ...]
    workdir: Path
    timeout_seconds: int = 3600
    retries: int = 0
    env: dict[str, str] | None = None
    log_path: Path | None = None


@dataclass(slots=True, frozen=True)
class ToolExecutionResult:
    """Normalized execution record for external tool wrappers.

    Args:
        command: Shell-safe argument vector that was attempted.
        return_code: Process return code. `None` means dry-run only.
        stdout_path: Captured stdout/stderr log path, if written.
        status: One of `complete`, `dry_run`, `missing_executable`, or `failed`.
        message: Human-readable diagnostic summary.
        backend_trace: Provenance string persisted into output tables.
    """

    command: tuple[str, ...]
    return_code: int | None
    stdout_path: Path | None
    status: str
    message: str
    backend_trace: str


def empty_frame(columns: tuple[str, ...]) -> Any:
    """Return an empty pandas DataFrame with the requested columns.

    Args:
        columns: Ordered schema columns for the empty frame.
    """
    import pandas as pd

    return pd.DataFrame(columns=list(columns))
