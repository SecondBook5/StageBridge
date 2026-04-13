"""Clonal evolution pattern classification.

Classifies patients into evolution patterns (1a, 1b, 2) based on
clone sharing between precursor and invasive lesions.

Pattern definitions (from Peng et al. 2025):
- Pattern 1a: Precursor clones present in LUAD + LUAD has additional subclones
- Pattern 1b: Shared clones + stage-specific clones in both
- Pattern 2: No shared clones between precursor and LUAD
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)


class ClonalPattern(str, Enum):
    """Clonal evolution pattern types."""

    PATTERN_1A = "1a"  # Direct lineage
    PATTERN_1B = "1b"  # Branched evolution
    PATTERN_2 = "2"  # Independent origins
    STABLE = "stable"  # Chromosomally stable (no CNAs)
    UNCATEGORIZED = "uncategorized"  # Cannot classify


@dataclass
class ClonalAnalysisResult:
    """Results from clonal pattern analysis for one patient."""

    patient_id: str
    pattern: ClonalPattern
    n_clones_precursor: int
    n_clones_invasive: int
    n_clones_shared: int
    shared_clone_ids: list[str]
    precursor_only_clone_ids: list[str]
    invasive_only_clone_ids: list[str]
    aneuploidy_score: float
    confidence: float  # 0-1, based on clone separation quality


def get_clone_tissue_mapping(
    adata: AnnData,
    clone_key: str = "cnv_leiden",
    stage_key: str = "stage",
    precursor_stages: list[str] | None = None,
    invasive_stages: list[str] | None = None,
) -> dict[str, set[str]]:
    """Map each clone to the tissue types where it appears.

    Parameters
    ----------
    adata
        AnnData with clone assignments and stage annotations.
    clone_key
        Column with clone IDs.
    stage_key
        Column with tissue stage.
    precursor_stages
        Stages considered precursor. Default: ["AAH", "AIS", "MIA"]
    invasive_stages
        Stages considered invasive. Default: ["LUAD"]

    Returns
    -------
    Dict mapping clone_id -> set of stage categories ("precursor", "invasive", "normal")
    """
    if precursor_stages is None:
        precursor_stages = ["AAH", "AIS", "MIA"]
    if invasive_stages is None:
        invasive_stages = ["LUAD"]

    clone_stages = {}
    for clone_id in adata.obs[clone_key].unique():
        mask = adata.obs[clone_key] == clone_id
        stages = adata.obs.loc[mask, stage_key].unique()

        categories = set()
        for stage in stages:
            if stage in precursor_stages:
                categories.add("precursor")
            elif stage in invasive_stages:
                categories.add("invasive")
            else:
                categories.add("normal")

        clone_stages[str(clone_id)] = categories

    return clone_stages


def classify_evolution_pattern(
    adata: AnnData,
    patient_id: str,
    clone_key: str = "cnv_leiden",
    stage_key: str = "stage",
    cnv_score_key: str = "cnv_score",
    precursor_stages: list[str] | None = None,
    invasive_stages: list[str] | None = None,
    aneuploidy_threshold: float = 0.1,
    min_clone_cells: int = 5,
) -> ClonalAnalysisResult:
    """Classify a patient's clonal evolution pattern.

    Parameters
    ----------
    adata
        AnnData for a single patient with CNV results.
    patient_id
        Patient identifier.
    clone_key
        Column with clone IDs.
    stage_key
        Column with tissue stage.
    cnv_score_key
        Column with per-cell CNV/aneuploidy score.
    precursor_stages
        Stages considered precursor.
    invasive_stages
        Stages considered invasive.
    aneuploidy_threshold
        Minimum aneuploidy score to consider CNAs present.
    min_clone_cells
        Minimum cells for a clone to be considered.

    Returns
    -------
    ClonalAnalysisResult with pattern classification.
    """
    if precursor_stages is None:
        precursor_stages = ["AAH", "AIS", "MIA"]
    if invasive_stages is None:
        invasive_stages = ["LUAD"]

    # Check for chromosomal stability
    mean_aneuploidy = adata.obs[cnv_score_key].mean()
    if mean_aneuploidy < aneuploidy_threshold:
        logger.info(f"Patient {patient_id}: Chromosomally stable (aneuploidy={mean_aneuploidy:.3f})")
        return ClonalAnalysisResult(
            patient_id=patient_id,
            pattern=ClonalPattern.STABLE,
            n_clones_precursor=0,
            n_clones_invasive=0,
            n_clones_shared=0,
            shared_clone_ids=[],
            precursor_only_clone_ids=[],
            invasive_only_clone_ids=[],
            aneuploidy_score=mean_aneuploidy,
            confidence=1.0,
        )

    # Filter to significant clones
    clone_counts = adata.obs[clone_key].value_counts()
    significant_clones = clone_counts[clone_counts >= min_clone_cells].index.tolist()

    if len(significant_clones) == 0:
        logger.warning(f"Patient {patient_id}: No significant clones found")
        return ClonalAnalysisResult(
            patient_id=patient_id,
            pattern=ClonalPattern.UNCATEGORIZED,
            n_clones_precursor=0,
            n_clones_invasive=0,
            n_clones_shared=0,
            shared_clone_ids=[],
            precursor_only_clone_ids=[],
            invasive_only_clone_ids=[],
            aneuploidy_score=mean_aneuploidy,
            confidence=0.0,
        )

    # Get clone-tissue mapping
    clone_stages = get_clone_tissue_mapping(
        adata,
        clone_key=clone_key,
        stage_key=stage_key,
        precursor_stages=precursor_stages,
        invasive_stages=invasive_stages,
    )

    # Classify clones
    shared_clones = []
    precursor_only = []
    invasive_only = []

    for clone_id in significant_clones:
        clone_id_str = str(clone_id)
        if clone_id_str not in clone_stages:
            continue

        stages = clone_stages[clone_id_str]

        if "precursor" in stages and "invasive" in stages:
            shared_clones.append(clone_id_str)
        elif "precursor" in stages:
            precursor_only.append(clone_id_str)
        elif "invasive" in stages:
            invasive_only.append(clone_id_str)

    n_shared = len(shared_clones)
    n_precursor_only = len(precursor_only)
    n_invasive_only = len(invasive_only)
    n_precursor = n_shared + n_precursor_only
    n_invasive = n_shared + n_invasive_only

    # Classify pattern
    if n_shared == 0:
        # No shared clones -> Pattern 2
        pattern = ClonalPattern.PATTERN_2
        confidence = 0.9 if (n_precursor > 0 and n_invasive > 0) else 0.5
    elif n_precursor_only == 0 and n_invasive_only > 0:
        # All precursor clones in LUAD, LUAD has extra -> Pattern 1a
        pattern = ClonalPattern.PATTERN_1A
        confidence = 0.9
    elif n_precursor_only > 0 or n_invasive_only > 0:
        # Shared + stage-specific clones -> Pattern 1b
        pattern = ClonalPattern.PATTERN_1B
        confidence = 0.85
    else:
        # Only shared clones, no stage-specific -> Could be 1a or 1b
        # Default to 1b (more common)
        pattern = ClonalPattern.PATTERN_1B
        confidence = 0.6

    logger.info(
        f"Patient {patient_id}: Pattern {pattern.value} "
        f"(shared={n_shared}, precursor_only={n_precursor_only}, invasive_only={n_invasive_only})"
    )

    return ClonalAnalysisResult(
        patient_id=patient_id,
        pattern=pattern,
        n_clones_precursor=n_precursor,
        n_clones_invasive=n_invasive,
        n_clones_shared=n_shared,
        shared_clone_ids=shared_clones,
        precursor_only_clone_ids=precursor_only,
        invasive_only_clone_ids=invasive_only,
        aneuploidy_score=mean_aneuploidy,
        confidence=confidence,
    )


def classify_all_patients(
    adata: AnnData,
    patient_key: str = "donor_id",
    clone_key: str = "cnv_leiden",
    stage_key: str = "stage",
    **kwargs,
) -> dict[str, ClonalAnalysisResult]:
    """Classify clonal patterns for all patients.

    Parameters
    ----------
    adata
        AnnData with CNV results for all patients.
    patient_key
        Column with patient IDs.
    clone_key
        Column with clone IDs.
    stage_key
        Column with tissue stage.
    **kwargs
        Additional arguments passed to classify_evolution_pattern.

    Returns
    -------
    Dict mapping patient_id -> ClonalAnalysisResult
    """
    results = {}

    for patient_id in adata.obs[patient_key].unique():
        patient_adata = adata[adata.obs[patient_key] == patient_id].copy()

        # Skip if no paired data
        stages = patient_adata.obs[stage_key].unique()
        has_precursor = any(s in ["AAH", "AIS", "MIA"] for s in stages)
        has_invasive = "LUAD" in stages

        if not (has_precursor and has_invasive):
            logger.info(f"Patient {patient_id}: No paired precursor+LUAD data, skipping")
            results[patient_id] = ClonalAnalysisResult(
                patient_id=patient_id,
                pattern=ClonalPattern.UNCATEGORIZED,
                n_clones_precursor=0,
                n_clones_invasive=0,
                n_clones_shared=0,
                shared_clone_ids=[],
                precursor_only_clone_ids=[],
                invasive_only_clone_ids=[],
                aneuploidy_score=0.0,
                confidence=0.0,
            )
            continue

        result = classify_evolution_pattern(
            patient_adata,
            patient_id=patient_id,
            clone_key=clone_key,
            stage_key=stage_key,
            **kwargs,
        )
        results[patient_id] = result

    return results
