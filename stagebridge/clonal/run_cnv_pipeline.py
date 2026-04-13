#!/usr/bin/env python3
"""Optimized CNV inference pipeline for large spatial data.

Processes one patient at a time to manage memory for 35GB+ spatial data.

Usage:
    python -m stagebridge.clonal.run_cnv_pipeline \
        --spatial-h5ad /path/to/spatial_merged.h5ad \
        --celltype-props /path/to/merged_celltype_proportions.parquet \
        --output-dir results/clonal/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def assign_dominant_celltype(
    props_df: pd.DataFrame,
    epithelial_types: list[str] = ["AT2", "Basal", "Secretory", "Ciliated"],
    min_epithelial_fraction: float = 0.3,
) -> pd.DataFrame:
    """Assign dominant cell type and filter to epithelial-enriched spots."""

    # Get all cell type columns (exclude sample_id and any index)
    celltype_cols = [c for c in props_df.columns if c not in ["sample_id", "spot_id", "barcode"]]

    # Dominant cell type
    props_df["dominant_celltype"] = props_df[celltype_cols].idxmax(axis=1)

    # Epithelial fraction
    epi_cols = [c for c in epithelial_types if c in props_df.columns]
    props_df["epithelial_fraction"] = props_df[epi_cols].sum(axis=1)

    # Is epithelial (dominant type is epithelial OR high epithelial fraction)
    props_df["is_epithelial"] = (
        props_df["dominant_celltype"].isin(epithelial_types) |
        (props_df["epithelial_fraction"] >= min_epithelial_fraction)
    )

    return props_df


def run_cnv_for_patient(
    adata: AnnData,
    patient_id: str,
    celltype_assignments: pd.Series,
    reference_celltype: str = "AT2",
    window_size: int = 100,
    step: int = 10,
) -> AnnData | None:
    """Run CNV inference for a single patient's epithelial spots."""

    import infercnvpy as cnv

    # Add cell type
    adata.obs["celltype"] = celltype_assignments

    # Filter to epithelial
    epithelial_mask = adata.obs["celltype"].isin(["AT2", "Basal", "Secretory", "Ciliated"])
    n_epi = epithelial_mask.sum()

    if n_epi < 50:
        logger.warning(f"Patient {patient_id}: Only {n_epi} epithelial spots, skipping")
        return None

    adata_epi = adata[epithelial_mask].copy()
    logger.info(f"Patient {patient_id}: {n_epi} epithelial spots")

    # Check for reference cells
    ref_mask = adata_epi.obs["celltype"] == reference_celltype
    n_ref = ref_mask.sum()

    if n_ref < 10:
        # Fall back to all AT2-like cells
        logger.warning(f"Patient {patient_id}: Only {n_ref} {reference_celltype} spots, using all epithelial as reference")
        # Use spots from Normal stage as reference if available
        if "stage" in adata_epi.obs.columns:
            normal_mask = adata_epi.obs["stage"] == "Normal"
            if normal_mask.sum() >= 10:
                adata_epi.obs["cnv_reference"] = normal_mask
            else:
                # Use lowest CNV score cells later
                adata_epi.obs["cnv_reference"] = False
        else:
            adata_epi.obs["cnv_reference"] = False
    else:
        # Use Normal AT2 cells preferentially
        if "stage" in adata_epi.obs.columns:
            normal_ref = (adata_epi.obs["celltype"] == reference_celltype) & (adata_epi.obs["stage"] == "Normal")
            if normal_ref.sum() >= 10:
                adata_epi.obs["cnv_reference"] = normal_ref
            else:
                adata_epi.obs["cnv_reference"] = ref_mask
        else:
            adata_epi.obs["cnv_reference"] = ref_mask

    logger.info(f"Patient {patient_id}: {adata_epi.obs['cnv_reference'].sum()} reference cells")

    # Normalize if needed
    if adata_epi.X.max() > 100:
        sc.pp.normalize_total(adata_epi, target_sum=1e4)
        sc.pp.log1p(adata_epi)

    # Get genomic positions
    try:
        cnv.io.genomic_position_from_gtf(adata_epi, gtf_file="default", inplace=True)
    except Exception as e:
        logger.error(f"Patient {patient_id}: Failed to get genomic positions: {e}")
        return None

    # Filter to genes with positions
    has_pos = adata_epi.var["chromosome"].notna() if "chromosome" in adata_epi.var.columns else pd.Series(False, index=adata_epi.var_names)
    adata_epi = adata_epi[:, has_pos].copy()

    if adata_epi.n_vars < 1000:
        logger.warning(f"Patient {patient_id}: Only {adata_epi.n_vars} genes with positions, may affect CNV quality")

    # Run CNV inference
    try:
        if adata_epi.obs["cnv_reference"].sum() >= 10:
            cnv.tl.infercnv(
                adata_epi,
                reference_key="cnv_reference",
                reference_cat=[True],
                window_size=window_size,
                step=step,
            )
        else:
            # No reference, use built-in reference or self-reference
            logger.warning(f"Patient {patient_id}: No reference cells, using median as reference")
            cnv.tl.infercnv(
                adata_epi,
                reference_key=None,
                window_size=window_size,
                step=step,
            )
    except Exception as e:
        logger.error(f"Patient {patient_id}: CNV inference failed: {e}")
        return None

    # Compute CNV score
    try:
        cnv.tl.cnv_score(adata_epi)
    except Exception as e:
        logger.warning(f"Patient {patient_id}: CNV score failed: {e}")
        adata_epi.obs["cnv_score"] = 0.0

    # Cluster by CNV to identify clones
    try:
        cnv.tl.pca(adata_epi)
        cnv.pp.neighbors(adata_epi)
        cnv.tl.leiden(adata_epi, key_added="cnv_clone")
    except Exception as e:
        logger.warning(f"Patient {patient_id}: CNV clustering failed: {e}")
        adata_epi.obs["cnv_clone"] = "0"

    return adata_epi


def classify_pattern(
    adata: AnnData,
    patient_id: str,
    clone_key: str = "cnv_clone",
    stage_key: str = "stage",
    min_clone_spots: int = 5,
    aneuploidy_threshold: float = 0.05,
) -> dict:
    """Classify clonal evolution pattern for a patient."""

    # Check aneuploidy
    mean_cnv = adata.obs.get("cnv_score", pd.Series([0])).mean()

    if mean_cnv < aneuploidy_threshold:
        return {
            "patient_id": patient_id,
            "pattern": "stable",
            "n_clones": 0,
            "n_shared": 0,
            "n_precursor_only": 0,
            "n_invasive_only": 0,
            "aneuploidy": float(mean_cnv),
            "confidence": 1.0,
        }

    # Get clone-stage mapping
    precursor_stages = ["AAH", "AIS", "MIA"]
    invasive_stages = ["LUAD"]

    clone_counts = adata.obs[clone_key].value_counts()
    significant_clones = clone_counts[clone_counts >= min_clone_spots].index.tolist()

    shared = []
    precursor_only = []
    invasive_only = []

    for clone in significant_clones:
        clone_mask = adata.obs[clone_key] == clone
        stages = set(adata.obs.loc[clone_mask, stage_key].unique())

        in_precursor = bool(stages & set(precursor_stages))
        in_invasive = bool(stages & set(invasive_stages))

        if in_precursor and in_invasive:
            shared.append(clone)
        elif in_precursor:
            precursor_only.append(clone)
        elif in_invasive:
            invasive_only.append(clone)

    n_shared = len(shared)
    n_precursor_only = len(precursor_only)
    n_invasive_only = len(invasive_only)

    # Classify
    if n_shared == 0:
        pattern = "2"  # No shared clones
        confidence = 0.9 if (n_precursor_only > 0 and n_invasive_only > 0) else 0.5
    elif n_precursor_only == 0 and n_invasive_only > 0:
        pattern = "1a"  # All precursor clones in LUAD + extras
        confidence = 0.9
    else:
        pattern = "1b"  # Shared + stage-specific
        confidence = 0.85

    return {
        "patient_id": patient_id,
        "pattern": pattern,
        "n_clones": len(significant_clones),
        "n_shared": n_shared,
        "n_precursor_only": n_precursor_only,
        "n_invasive_only": n_invasive_only,
        "aneuploidy": float(mean_cnv),
        "confidence": confidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Run CNV inference for clonal pattern extraction")
    parser.add_argument("--spatial-h5ad", type=Path, required=True)
    parser.add_argument("--celltype-props", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/clonal"))
    parser.add_argument("--patients", nargs="*", help="Specific patients to process (default: all)")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--min-clone-spots", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load cell type proportions
    logger.info(f"Loading cell type proportions from {args.celltype_props}")
    props_df = pd.read_parquet(args.celltype_props)
    props_df = assign_dominant_celltype(props_df)
    logger.info(f"Epithelial spots: {props_df['is_epithelial'].sum()} / {len(props_df)}")

    # Load spatial data - just the obs to get patient list
    logger.info(f"Loading spatial data metadata from {args.spatial_h5ad}")
    adata_full = sc.read_h5ad(args.spatial_h5ad, backed='r')

    patients = args.patients or sorted(adata_full.obs["donor_id"].unique())
    logger.info(f"Processing {len(patients)} patients: {patients}")

    # Process each patient
    results = []
    patterns = {}

    for i, patient_id in enumerate(patients):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing patient {patient_id} ({i+1}/{len(patients)})")
        logger.info(f"{'='*60}")

        # Get patient's spot indices
        patient_mask = adata_full.obs["donor_id"] == patient_id
        n_spots = patient_mask.sum()

        if n_spots == 0:
            logger.warning(f"Patient {patient_id}: No spots found, skipping")
            continue

        # Check if patient has paired data
        patient_stages = adata_full.obs.loc[patient_mask, "stage"].unique()
        has_precursor = any(s in ["AAH", "AIS", "MIA"] for s in patient_stages)
        has_invasive = "LUAD" in patient_stages

        if not (has_precursor and has_invasive):
            logger.info(f"Patient {patient_id}: No paired precursor+LUAD (stages: {list(patient_stages)})")
            patterns[patient_id] = "uncategorized"
            results.append({
                "patient_id": patient_id,
                "pattern": "uncategorized",
                "n_clones": 0,
                "n_shared": 0,
                "n_precursor_only": 0,
                "n_invasive_only": 0,
                "aneuploidy": 0.0,
                "confidence": 0.0,
                "note": f"No paired data. Stages: {list(patient_stages)}"
            })
            continue

        # Load patient data into memory
        logger.info(f"Patient {patient_id}: Loading {n_spots} spots into memory")
        patient_indices = np.where(patient_mask)[0]
        adata_patient = adata_full[patient_indices].to_memory()

        # Get cell type assignments for this patient's spots
        # Match by sample_id
        patient_samples = adata_patient.obs["sample_id"].unique()
        props_patient = props_df[props_df["sample_id"].isin(patient_samples)].copy()

        # Create mapping from spot to dominant celltype
        # Assuming row order matches between spatial and props
        if len(props_patient) == len(adata_patient):
            celltype_assignments = props_patient["dominant_celltype"].values
        else:
            logger.warning(f"Patient {patient_id}: Props/spatial mismatch ({len(props_patient)} vs {len(adata_patient)})")
            # Try to match by index
            celltype_assignments = pd.Series("unknown", index=adata_patient.obs.index)

        # Run CNV
        try:
            adata_cnv = run_cnv_for_patient(
                adata_patient,
                patient_id,
                pd.Series(celltype_assignments, index=adata_patient.obs.index),
                window_size=args.window_size,
            )
        except Exception as e:
            logger.error(f"Patient {patient_id}: CNV failed with error: {e}")
            adata_cnv = None

        if adata_cnv is None:
            patterns[patient_id] = "uncategorized"
            results.append({
                "patient_id": patient_id,
                "pattern": "uncategorized",
                "n_clones": 0,
                "n_shared": 0,
                "n_precursor_only": 0,
                "n_invasive_only": 0,
                "aneuploidy": 0.0,
                "confidence": 0.0,
                "note": "CNV inference failed"
            })
            continue

        # Classify pattern
        result = classify_pattern(
            adata_cnv,
            patient_id,
            min_clone_spots=args.min_clone_spots,
        )

        patterns[patient_id] = result["pattern"]
        results.append(result)

        logger.info(f"Patient {patient_id}: Pattern {result['pattern']} "
                   f"(shared={result['n_shared']}, prec_only={result['n_precursor_only']}, "
                   f"inv_only={result['n_invasive_only']}, aneuploidy={result['aneuploidy']:.3f})")

        # Save intermediate result
        patient_dir = args.output_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        adata_cnv.write_h5ad(patient_dir / "cnv_results.h5ad")

        # Free memory
        del adata_patient, adata_cnv
        gc.collect()

    # Save final results
    with open(args.output_dir / "clonal_patterns.json", "w") as f:
        json.dump(patterns, f, indent=2)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "clonal_analysis_details.csv", index=False)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("CLONAL PATTERN EXTRACTION COMPLETE")
    logger.info("="*60)

    pattern_counts = pd.Series(patterns).value_counts()
    for pattern, count in pattern_counts.items():
        logger.info(f"  Pattern {pattern}: {count} patients")

    logger.info(f"\nResults saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
