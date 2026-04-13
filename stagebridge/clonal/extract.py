"""Main clonal pattern extraction pipeline.

Extracts clonal evolution patterns (1a, 1b, 2) from spatial transcriptomics
data using CNV inference, following Peng et al. 2025 methodology.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import scanpy as sc
from anndata import AnnData

from .cnv_inference import run_cnv_inference, compute_clone_cnv_profiles
from .pattern_classification import (
    ClonalPattern,
    ClonalAnalysisResult,
    classify_all_patients,
)

logger = logging.getLogger(__name__)

# Re-export
__all__ = ["extract_clonal_patterns", "ClonalPattern"]


def extract_clonal_patterns(
    adata: AnnData,
    output_dir: str | Path | None = None,
    patient_key: str = "donor_id",
    stage_key: str = "stage",
    cell_type_key: str = "cell_type",
    epithelial_types: list[str] | None = None,
    reference_types: list[str] | None = None,
    window_size: int = 100,
    min_clone_cells: int = 5,
    save_intermediate: bool = True,
) -> dict[str, str]:
    """Extract clonal evolution patterns for all patients.

    This is the main entry point for clonal pattern extraction.

    Parameters
    ----------
    adata
        AnnData with spatial transcriptomics data. Should have:
        - Raw counts in adata.X or adata.raw
        - Patient IDs in adata.obs[patient_key]
        - Stage in adata.obs[stage_key]
        - Cell types in adata.obs[cell_type_key]
    output_dir
        Directory to save results. If None, doesn't save.
    patient_key
        Column with patient IDs.
    stage_key
        Column with tissue stage (Normal, AAH, AIS, MIA, LUAD).
    cell_type_key
        Column with cell type annotations.
    epithelial_types
        Cell types to include (epithelial only for CNV).
        Default: ["AT1", "AT2", "KAC", "Tumor", "Basal", "Club", "Ciliated"]
    reference_types
        Cell types to use as diploid reference.
        Default: ["AT2"] (normal alveolar type 2)
    window_size
        Window size for CNV smoothing.
    min_clone_cells
        Minimum cells per clone.
    save_intermediate
        Whether to save intermediate results (CNV adata, etc.)

    Returns
    -------
    Dict mapping patient_id -> pattern ("1a", "1b", "2", "stable", "uncategorized")
    """
    if epithelial_types is None:
        epithelial_types = [
            "AT1", "AT2", "KAC", "Tumor", "Basal", "Club",
            "Ciliated", "Secretory", "Neuroendocrine",
            # Also include HLCA labels
            "AT2 proliferating", "Club (nasal)", "Goblet",
        ]
    if reference_types is None:
        reference_types = ["AT2", "AT2 proliferating"]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting clonal pattern extraction for {adata.n_obs} cells/spots")

    # Filter to epithelial cells/spots
    epithelial_mask = adata.obs[cell_type_key].isin(epithelial_types)
    n_epithelial = epithelial_mask.sum()
    logger.info(f"Filtering to {n_epithelial} epithelial cells/spots")

    if n_epithelial < 100:
        raise ValueError(
            f"Only {n_epithelial} epithelial cells found. "
            f"Check cell_type_key='{cell_type_key}' and epithelial_types={epithelial_types}"
        )

    adata_epi = adata[epithelial_mask].copy()

    # Run CNV inference
    logger.info("Running CNV inference...")
    adata_cnv = run_cnv_inference(
        adata_epi,
        reference_key=cell_type_key,
        reference_cat=reference_types,
        window_size=window_size,
    )

    if save_intermediate and output_dir is not None:
        cnv_path = output_dir / "epithelial_cnv.h5ad"
        logger.info(f"Saving CNV results to {cnv_path}")
        adata_cnv.write_h5ad(cnv_path)

    # Classify patterns for each patient
    logger.info("Classifying clonal patterns...")
    results = classify_all_patients(
        adata_cnv,
        patient_key=patient_key,
        stage_key=stage_key,
        min_clone_cells=min_clone_cells,
    )

    # Convert to simple pattern dict
    patterns = {
        patient_id: result.pattern.value
        for patient_id, result in results.items()
    }

    # Save results
    if output_dir is not None:
        # Save pattern mapping
        patterns_path = output_dir / "clonal_patterns.json"
        with open(patterns_path, "w") as f:
            json.dump(patterns, f, indent=2)
        logger.info(f"Saved patterns to {patterns_path}")

        # Save detailed results
        details = []
        for patient_id, result in results.items():
            details.append({
                "patient_id": result.patient_id,
                "pattern": result.pattern.value,
                "n_clones_precursor": result.n_clones_precursor,
                "n_clones_invasive": result.n_clones_invasive,
                "n_clones_shared": result.n_clones_shared,
                "shared_clone_ids": ",".join(result.shared_clone_ids),
                "precursor_only_clone_ids": ",".join(result.precursor_only_clone_ids),
                "invasive_only_clone_ids": ",".join(result.invasive_only_clone_ids),
                "aneuploidy_score": result.aneuploidy_score,
                "confidence": result.confidence,
            })
        details_df = pd.DataFrame(details)
        details_path = output_dir / "clonal_analysis_details.csv"
        details_df.to_csv(details_path, index=False)
        logger.info(f"Saved detailed results to {details_path}")

    # Log summary
    pattern_counts = pd.Series(patterns).value_counts()
    logger.info(f"Pattern distribution:\n{pattern_counts}")

    return patterns


def add_clonal_patterns_to_adata(
    adata: AnnData,
    patterns: dict[str, str],
    patient_key: str = "donor_id",
) -> AnnData:
    """Add clonal pattern annotations to AnnData.

    Parameters
    ----------
    adata
        AnnData to annotate.
    patterns
        Dict mapping patient_id -> pattern from extract_clonal_patterns.
    patient_key
        Column with patient IDs.

    Returns
    -------
    AnnData with new columns:
        - adata.obs["clonal_pattern"]: Pattern for each cell
        - adata.obs["clonal_pattern_numeric"]: Numeric encoding (1a=0, 1b=1, 2=2)
    """
    adata = adata.copy()

    # Map patterns to cells
    adata.obs["clonal_pattern"] = adata.obs[patient_key].map(patterns)
    adata.obs["clonal_pattern"] = adata.obs["clonal_pattern"].fillna("unknown")

    # Numeric encoding for modeling
    pattern_to_int = {
        "1a": 0,
        "1b": 1,
        "2": 2,
        "stable": 3,
        "uncategorized": -1,
        "unknown": -1,
    }
    adata.obs["clonal_pattern_numeric"] = (
        adata.obs["clonal_pattern"].map(pattern_to_int).fillna(-1).astype(int)
    )

    return adata


def load_clonal_patterns(path: str | Path) -> dict[str, str]:
    """Load clonal patterns from JSON file.

    Parameters
    ----------
    path
        Path to clonal_patterns.json from extract_clonal_patterns.

    Returns
    -------
    Dict mapping patient_id -> pattern.
    """
    with open(path) as f:
        return json.load(f)
