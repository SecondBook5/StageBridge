#!/usr/bin/env python3
"""
AP-1 Stress Signature Validation for StageBridge.

Validates the hypothesis that AP-1/stress response activation marks progression-prone
cells and niches in early LUAD. This script handles both single-cell and spatial
transcriptomics data.

Key biological insight (from Marjanovic et al.):
- AP-1 (JUN/FOS) transcription factors mark a plastic, high-plasticity state
- This state is associated with:
  - Fetal progenitor-like intermediate
  - Enhanced progression potential
  - Response to inflammatory signaling (IL1B/TNF)
  - JNK/ERK/p38 stress pathway convergence

AP-1 Core Signature:
  JUN, JUNB, JUND, FOS, FOSB, ATF3, ATF4

Extended Stress Signature:
  + HSPA5, DNAJB9 (ER stress/UPR markers)
  + EGR1 (immediate early response)

Usage:
  python scripts/run_ap1_validation.py \
      --snrna /path/to/snrna_with_celltypes.h5ad \
      --cells /path/to/cells.parquet \
      --neighborhoods /path/to/neighborhoods.parquet \
      --output-dir /path/to/ap1_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# AP-1 stress signature genes
AP1_CORE_GENES = ["JUN", "JUNB", "JUND", "FOS", "FOSB", "ATF3", "ATF4"]
AP1_EXTENDED_GENES = AP1_CORE_GENES + ["HSPA5", "DNAJB9", "EGR1"]

# Upstream regulators that converge on AP-1
UPSTREAM_KINASES = ["MAPK8", "MAPK9", "MAPK10",  # JNK1/2/3
                    "MAPK1", "MAPK3",              # ERK1/2
                    "MAPK14", "MAPK11"]            # p38 alpha/beta

# IL1B pathway connection (links to H1.2 hypothesis)
IL1B_AP1_AXIS = ["IL1B", "IL1R1", "MYD88", "IRAK1", "TRAF6"]

# Stages for progression analysis
STAGES = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
STAGE_ORDER = {s: i for i, s in enumerate(STAGES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="AP-1 Stress Signature Validation"
    )
    parser.add_argument(
        "--snrna",
        type=Path,
        help="Path to snRNA h5ad with expression data"
    )
    parser.add_argument(
        "--cells",
        type=Path,
        required=True,
        help="Path to cells.parquet with cell metadata and embeddings"
    )
    parser.add_argument(
        "--neighborhoods",
        type=Path,
        required=True,
        help="Path to neighborhoods.parquet with spatial context"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for AP-1 validation results"
    )
    parser.add_argument(
        "--use-extended",
        action="store_true",
        help="Use extended signature (includes UPR/ER stress genes)"
    )
    return parser.parse_args()


def compute_signature_score(
    adata: sc.AnnData,
    gene_list: list[str],
    score_name: str = "signature_score"
) -> np.ndarray:
    """Compute signature score using scanpy's score_genes."""
    # Filter to genes present in data
    available_genes = [g for g in gene_list if g in adata.var_names]
    missing_genes = [g for g in gene_list if g not in adata.var_names]

    if missing_genes:
        logger.warning(f"Missing genes for {score_name}: {missing_genes}")

    if len(available_genes) < 2:
        logger.error(f"Too few genes available for {score_name}: {available_genes}")
        return np.full(adata.n_obs, np.nan)

    logger.info(f"Computing {score_name} with {len(available_genes)}/{len(gene_list)} genes")

    # Use scanpy's score_genes (z-score based)
    sc.tl.score_genes(adata, available_genes, score_name=score_name)
    return adata.obs[score_name].values


def compute_gene_expression_matrix(
    adata: sc.AnnData,
    gene_list: list[str]
) -> pd.DataFrame:
    """Extract expression matrix for specific genes."""
    available_genes = [g for g in gene_list if g in adata.var_names]

    if not available_genes:
        return pd.DataFrame()

    # Get expression values
    expr = adata[:, available_genes].X
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()

    return pd.DataFrame(
        expr,
        index=adata.obs_names,
        columns=available_genes
    )


def analyze_single_cell(
    adata: sc.AnnData,
    cells_df: pd.DataFrame,
    use_extended: bool = False
) -> dict:
    """
    Analyze AP-1 signature in single-cell RNA-seq data.

    Returns dict with:
    - Per-cell AP-1 scores
    - Stage-stratified statistics
    - Cell type enrichment
    - Correlation with progression
    """
    logger.info("=" * 60)
    logger.info("SINGLE-CELL AP-1 ANALYSIS")
    logger.info("=" * 60)

    gene_list = AP1_EXTENDED_GENES if use_extended else AP1_CORE_GENES

    # Compute main AP-1 score
    ap1_scores = compute_signature_score(adata, gene_list, "ap1_score")

    # Compute sub-signatures
    jun_scores = compute_signature_score(adata, ["JUN", "JUNB", "JUND"], "jun_score")
    fos_scores = compute_signature_score(adata, ["FOS", "FOSB"], "fos_score")
    atf_scores = compute_signature_score(adata, ["ATF3", "ATF4"], "atf_score")

    # Get individual gene expression
    ap1_expr = compute_gene_expression_matrix(adata, gene_list)

    # Build results dataframe
    results_df = pd.DataFrame({
        "cell_id": adata.obs_names,
        "ap1_score": ap1_scores,
        "jun_score": jun_scores,
        "fos_score": fos_scores,
        "atf_score": atf_scores,
    })

    # Add individual genes
    for gene in ap1_expr.columns:
        results_df[f"expr_{gene}"] = ap1_expr[gene].values

    # Merge with cell metadata
    if "cell_id" in cells_df.columns:
        results_df = results_df.merge(
            cells_df[["cell_id", "stage", "cell_type", "donor_id"]].drop_duplicates(),
            on="cell_id",
            how="left"
        )

    # Stage-stratified analysis
    stage_stats = {}
    if "stage" in results_df.columns:
        for stage in STAGES:
            mask = results_df["stage"] == stage
            if mask.sum() > 0:
                scores = results_df.loc[mask, "ap1_score"].dropna()
                stage_stats[stage] = {
                    "n_cells": int(mask.sum()),
                    "mean_ap1": float(scores.mean()) if len(scores) > 0 else None,
                    "std_ap1": float(scores.std()) if len(scores) > 0 else None,
                    "median_ap1": float(scores.median()) if len(scores) > 0 else None,
                    "pct_high": float((scores > scores.quantile(0.75)).mean() * 100) if len(scores) > 0 else None,
                }

        # Progression correlation (Spearman with stage order)
        valid_mask = results_df["stage"].isin(STAGES) & results_df["ap1_score"].notna()
        if valid_mask.sum() > 10:
            stage_numeric = results_df.loc[valid_mask, "stage"].map(STAGE_ORDER)
            rho, pval = stats.spearmanr(
                stage_numeric,
                results_df.loc[valid_mask, "ap1_score"]
            )
            stage_stats["progression_correlation"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_cells": int(valid_mask.sum()),
            }

    # Cell type enrichment
    cell_type_stats = {}
    if "cell_type" in results_df.columns:
        for ct in results_df["cell_type"].dropna().unique():
            mask = results_df["cell_type"] == ct
            scores = results_df.loc[mask, "ap1_score"].dropna()
            if len(scores) > 10:
                cell_type_stats[ct] = {
                    "n_cells": int(mask.sum()),
                    "mean_ap1": float(scores.mean()),
                    "std_ap1": float(scores.std()),
                }

        # Rank cell types by mean AP-1
        if cell_type_stats:
            ranked = sorted(
                cell_type_stats.items(),
                key=lambda x: x[1]["mean_ap1"],
                reverse=True
            )
            cell_type_stats["ranking"] = [ct for ct, _ in ranked]

    # Upstream kinase analysis
    upstream_expr = compute_gene_expression_matrix(adata, UPSTREAM_KINASES)
    upstream_stats = {}
    for kinase in upstream_expr.columns:
        expr = upstream_expr[kinase]
        if results_df["ap1_score"].notna().sum() > 10:
            rho, pval = stats.spearmanr(expr, results_df["ap1_score"], nan_policy="omit")
            upstream_stats[kinase] = {
                "correlation_with_ap1": float(rho) if not np.isnan(rho) else None,
                "p_value": float(pval) if not np.isnan(pval) else None,
            }

    # IL1B-AP1 connection (links to H1.2)
    il1b_expr = compute_gene_expression_matrix(adata, IL1B_AP1_AXIS)
    il1b_stats = {}
    for gene in il1b_expr.columns:
        expr = il1b_expr[gene]
        if results_df["ap1_score"].notna().sum() > 10:
            rho, pval = stats.spearmanr(expr, results_df["ap1_score"], nan_policy="omit")
            il1b_stats[gene] = {
                "correlation_with_ap1": float(rho) if not np.isnan(rho) else None,
                "p_value": float(pval) if not np.isnan(pval) else None,
            }

    return {
        "cell_scores": results_df,
        "stage_stats": stage_stats,
        "cell_type_stats": cell_type_stats,
        "upstream_kinase_stats": upstream_stats,
        "il1b_axis_stats": il1b_stats,
        "genes_used": gene_list,
    }


def analyze_spatial_neighborhoods(
    neighborhoods_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    sc_ap1_scores: Optional[pd.DataFrame] = None
) -> dict:
    """
    Analyze AP-1 signature in spatial neighborhood context.

    This examines whether:
    1. High-AP1 cells cluster spatially
    2. Specific neighborhood compositions associate with AP-1 activation
    3. AP-1 signal propagates through spatial niches
    """
    logger.info("=" * 60)
    logger.info("SPATIAL NEIGHBORHOOD AP-1 ANALYSIS")
    logger.info("=" * 60)

    logger.info(f"Neighborhoods: {len(neighborhoods_df):,} rows")
    logger.info(f"Columns: {list(neighborhoods_df.columns)}")

    results = {
        "n_neighborhoods": len(neighborhoods_df),
        "neighborhood_stats": {},
        "stage_spatial_stats": {},
    }

    # If we have single-cell AP-1 scores, map them to neighborhoods
    if sc_ap1_scores is not None and "cell_id" in neighborhoods_df.columns:
        # Merge AP-1 scores with neighborhood data
        merged = neighborhoods_df.merge(
            sc_ap1_scores[["cell_id", "ap1_score", "stage", "cell_type"]],
            on="cell_id",
            how="left"
        )

        # Check how many cells have AP-1 scores
        n_with_scores = merged["ap1_score"].notna().sum()
        logger.info(f"Neighborhoods with AP-1 scores: {n_with_scores:,}/{len(merged):,}")

        if n_with_scores > 0:
            # Neighborhood-level AP-1 statistics
            # Group by donor and compute local AP-1 patterns
            if "donor_id" in merged.columns:
                donor_stats = {}
                for donor in merged["donor_id"].unique():
                    donor_mask = merged["donor_id"] == donor
                    donor_scores = merged.loc[donor_mask, "ap1_score"].dropna()
                    if len(donor_scores) > 10:
                        donor_stats[donor] = {
                            "n_cells": int(len(donor_scores)),
                            "mean_ap1": float(donor_scores.mean()),
                            "std_ap1": float(donor_scores.std()),
                            "spatial_variance": float(donor_scores.var()),
                        }
                results["donor_spatial_stats"] = donor_stats

            # Stage-stratified spatial analysis
            if "stage" in merged.columns:
                for stage in STAGES:
                    mask = merged["stage"] == stage
                    scores = merged.loc[mask, "ap1_score"].dropna()
                    if len(scores) > 10:
                        # Compute spatial clustering metric (coefficient of variation)
                        results["stage_spatial_stats"][stage] = {
                            "n_neighborhoods": int(mask.sum()),
                            "mean_ap1": float(scores.mean()),
                            "std_ap1": float(scores.std()),
                            "cv": float(scores.std() / scores.mean()) if scores.mean() != 0 else None,
                        }

            # Identify AP-1 hotspots (neighborhoods with high AP-1 cells)
            ap1_threshold = merged["ap1_score"].quantile(0.9)
            hotspot_mask = merged["ap1_score"] > ap1_threshold
            results["hotspot_analysis"] = {
                "threshold": float(ap1_threshold),
                "n_hotspot_cells": int(hotspot_mask.sum()),
                "pct_hotspot": float(hotspot_mask.mean() * 100),
            }

            # Stage enrichment in hotspots
            if "stage" in merged.columns:
                hotspot_stages = merged.loc[hotspot_mask, "stage"].value_counts()
                all_stages = merged["stage"].value_counts()
                enrichment = {}
                for stage in STAGES:
                    if stage in hotspot_stages.index and stage in all_stages.index:
                        obs_pct = hotspot_stages.get(stage, 0) / hotspot_mask.sum()
                        exp_pct = all_stages.get(stage, 0) / len(merged)
                        if exp_pct > 0:
                            enrichment[stage] = {
                                "observed_pct": float(obs_pct * 100),
                                "expected_pct": float(exp_pct * 100),
                                "fold_enrichment": float(obs_pct / exp_pct),
                            }
                results["hotspot_analysis"]["stage_enrichment"] = enrichment

            # Cell type composition of hotspots
            if "cell_type" in merged.columns:
                hotspot_cts = merged.loc[hotspot_mask, "cell_type"].value_counts()
                all_cts = merged["cell_type"].value_counts()
                ct_enrichment = {}
                for ct in hotspot_cts.index:
                    obs_pct = hotspot_cts.get(ct, 0) / hotspot_mask.sum()
                    exp_pct = all_cts.get(ct, 0) / len(merged)
                    if exp_pct > 0:
                        ct_enrichment[ct] = {
                            "observed_pct": float(obs_pct * 100),
                            "expected_pct": float(exp_pct * 100),
                            "fold_enrichment": float(obs_pct / exp_pct),
                        }
                results["hotspot_analysis"]["cell_type_enrichment"] = ct_enrichment

    # Analyze neighborhood tokens if present
    if "tokens" in neighborhoods_df.columns:
        logger.info("Analyzing neighborhood token structure...")
        # Token column contains spatial context - structure varies by implementation
        # Log what we have for debugging
        sample_tokens = neighborhoods_df["tokens"].iloc[0] if len(neighborhoods_df) > 0 else None
        results["token_info"] = {
            "column_present": True,
            "sample_type": str(type(sample_tokens)),
        }

    return results


def generate_validation_report(
    sc_results: dict,
    spatial_results: dict,
    output_dir: Path
) -> dict:
    """Generate comprehensive validation report."""

    report = {
        "hypothesis": "AP-1 stress signature marks progression-prone cells and niches",
        "biological_basis": {
            "key_finding": "AP-1 (JUN/FOS) activation marks plastic, progression-prone state",
            "mechanism": "JNK/ERK/p38 stress pathways converge on AP-1",
            "connection_to_h1_2": "IL1B signaling activates AP-1 via MYD88/IRAK/TRAF6",
        },
        "single_cell": {},
        "spatial": {},
        "validation_status": {},
    }

    # Single-cell summary
    if sc_results:
        stage_stats = sc_results.get("stage_stats", {})
        prog_corr = stage_stats.get("progression_correlation", {})

        report["single_cell"] = {
            "n_cells_analyzed": len(sc_results.get("cell_scores", [])),
            "genes_used": sc_results.get("genes_used", []),
            "stage_means": {
                s: stage_stats.get(s, {}).get("mean_ap1")
                for s in STAGES if s in stage_stats
            },
            "progression_correlation": prog_corr.get("spearman_rho"),
            "progression_pvalue": prog_corr.get("p_value"),
        }

        # Top cell types by AP-1
        ct_stats = sc_results.get("cell_type_stats", {})
        if "ranking" in ct_stats:
            report["single_cell"]["top_cell_types"] = ct_stats["ranking"][:5]

        # IL1B-AP1 connection strength
        il1b_stats = sc_results.get("il1b_axis_stats", {})
        if "IL1B" in il1b_stats:
            report["single_cell"]["il1b_ap1_correlation"] = il1b_stats["IL1B"].get("correlation_with_ap1")

    # Spatial summary
    if spatial_results:
        report["spatial"] = {
            "n_neighborhoods": spatial_results.get("n_neighborhoods"),
            "stage_spatial_stats": spatial_results.get("stage_spatial_stats", {}),
        }

        hotspot = spatial_results.get("hotspot_analysis", {})
        if hotspot:
            report["spatial"]["hotspot_analysis"] = {
                "n_hotspots": hotspot.get("n_hotspot_cells"),
                "pct_hotspots": hotspot.get("pct_hotspot"),
                "stage_enrichment": hotspot.get("stage_enrichment", {}),
            }

    # Validation status
    # Check if AP-1 correlates with progression (expected: positive correlation)
    rho = report["single_cell"].get("progression_correlation")
    if rho is not None:
        if rho > 0.1:
            report["validation_status"]["ap1_progression"] = "SUPPORTED"
            report["validation_status"]["ap1_progression_detail"] = (
                f"AP-1 score positively correlates with progression (rho={rho:.3f})"
            )
        elif rho < -0.1:
            report["validation_status"]["ap1_progression"] = "REVERSED"
            report["validation_status"]["ap1_progression_detail"] = (
                f"AP-1 score negatively correlates with progression (rho={rho:.3f})"
            )
        else:
            report["validation_status"]["ap1_progression"] = "INCONCLUSIVE"
            report["validation_status"]["ap1_progression_detail"] = (
                f"No strong correlation (rho={rho:.3f})"
            )

    # Check IL1B-AP1 connection (expected: positive)
    il1b_corr = report["single_cell"].get("il1b_ap1_correlation")
    if il1b_corr is not None:
        if il1b_corr > 0.1:
            report["validation_status"]["il1b_ap1_axis"] = "SUPPORTED"
        else:
            report["validation_status"]["il1b_ap1_axis"] = "WEAK_OR_ABSENT"

    # Check for early-stage hotspot enrichment (expected: AAH/AIS enriched)
    hotspot_enrich = report.get("spatial", {}).get("hotspot_analysis", {}).get("stage_enrichment", {})
    early_stages = ["AAH", "AIS"]
    late_stages = ["MIA", "LUAD"]

    early_fe = np.mean([
        hotspot_enrich.get(s, {}).get("fold_enrichment", 1.0)
        for s in early_stages if s in hotspot_enrich
    ]) if any(s in hotspot_enrich for s in early_stages) else None

    late_fe = np.mean([
        hotspot_enrich.get(s, {}).get("fold_enrichment", 1.0)
        for s in late_stages if s in hotspot_enrich
    ]) if any(s in hotspot_enrich for s in late_stages) else None

    if early_fe is not None and late_fe is not None:
        if early_fe > late_fe:
            report["validation_status"]["early_stage_hotspots"] = "SUPPORTED"
            report["validation_status"]["early_stage_detail"] = (
                f"Early stages enriched in AP-1 hotspots (early FE={early_fe:.2f}, late FE={late_fe:.2f})"
            )
        else:
            report["validation_status"]["early_stage_hotspots"] = "NOT_SUPPORTED"

    return report


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("AP-1 STRESS SIGNATURE VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Cells: {args.cells}")
    logger.info(f"Neighborhoods: {args.neighborhoods}")
    logger.info(f"snRNA: {args.snrna}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Extended signature: {args.use_extended}")

    # Load cell metadata
    logger.info("\n[1/4] Loading cell metadata...")
    cells_df = pd.read_parquet(args.cells)
    logger.info(f"  Loaded {len(cells_df):,} cells")

    # Load neighborhoods
    logger.info("\n[2/4] Loading spatial neighborhoods...")
    neighborhoods_df = pd.read_parquet(args.neighborhoods)
    logger.info(f"  Loaded {len(neighborhoods_df):,} neighborhoods")

    # Single-cell analysis (requires h5ad with expression)
    sc_results = None
    if args.snrna and args.snrna.exists():
        logger.info("\n[3/4] Single-cell AP-1 analysis...")
        adata = sc.read_h5ad(args.snrna)
        logger.info(f"  Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        sc_results = analyze_single_cell(
            adata, cells_df, use_extended=args.use_extended
        )

        # Save cell-level scores
        sc_results["cell_scores"].to_parquet(
            args.output_dir / "ap1_cell_scores.parquet",
            index=False
        )
        logger.info(f"  Saved: ap1_cell_scores.parquet")
    else:
        logger.warning("\n[3/4] Skipping single-cell analysis (no snRNA h5ad provided)")

    # Spatial analysis
    logger.info("\n[4/4] Spatial neighborhood analysis...")
    sc_scores = sc_results["cell_scores"] if sc_results else None
    spatial_results = analyze_spatial_neighborhoods(
        neighborhoods_df, cells_df, sc_scores
    )

    # Generate report
    logger.info("\nGenerating validation report...")
    report = generate_validation_report(sc_results, spatial_results, args.output_dir)

    # Save results
    with open(args.output_dir / "ap1_validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved: ap1_validation_report.json")

    # Save detailed stats
    if sc_results:
        with open(args.output_dir / "ap1_singlecell_stats.json", "w") as f:
            # Remove DataFrame from dict before saving
            stats_to_save = {k: v for k, v in sc_results.items() if k != "cell_scores"}
            json.dump(stats_to_save, f, indent=2, default=str)
        logger.info(f"Saved: ap1_singlecell_stats.json")

    with open(args.output_dir / "ap1_spatial_stats.json", "w") as f:
        json.dump(spatial_results, f, indent=2, default=str)
    logger.info(f"Saved: ap1_spatial_stats.json")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("AP-1 VALIDATION SUMMARY")
    logger.info("=" * 70)

    if sc_results:
        stage_stats = sc_results.get("stage_stats", {})
        logger.info("\nAP-1 Score by Stage:")
        for stage in STAGES:
            if stage in stage_stats:
                s = stage_stats[stage]
                logger.info(f"  {stage}: mean={s['mean_ap1']:.4f} +/- {s['std_ap1']:.4f} (n={s['n_cells']:,})")

        prog = stage_stats.get("progression_correlation", {})
        if prog:
            logger.info(f"\nProgression correlation: rho={prog['spearman_rho']:.4f}, p={prog['p_value']:.2e}")

        il1b = sc_results.get("il1b_axis_stats", {}).get("IL1B", {})
        if il1b:
            logger.info(f"IL1B-AP1 correlation: rho={il1b['correlation_with_ap1']:.4f}")

    logger.info("\nValidation Status:")
    for key, value in report.get("validation_status", {}).items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
