#!/usr/bin/env python3
"""
AP-1 Stress Signature Validation for StageBridge.

Validates whether AP-1/stress response activation correlates with model-predicted
progression risk and niche influence. This tests whether the model's predictions
align with known stress biology.

Key biological insight (from Marjanovic et al.):
- AP-1 (JUN/FOS) transcription factors mark a plastic, high-plasticity state
- This state is associated with:
  - Fetal progenitor-like intermediate
  - Enhanced progression potential
  - Response to inflammatory signaling (IL1B/TNF)
  - JNK/ERK/p38 stress pathway convergence

The validation tests:
1. Do cells with high AP-1 have higher MODEL-PREDICTED transition probability?
2. Do cells in high niche-influence regions show elevated AP-1?
3. Is AP-1 enriched in cells the model flags as progression-prone?

Usage:
  python scripts/run_ap1_validation.py \
      --snrna /path/to/snrna_with_celltypes.h5ad \
      --cells /path/to/cells.parquet \
      --neighborhoods /path/to/neighborhoods.parquet \
      --inference-outputs /path/to/inference_outputs.parquet \
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

# Stages for analysis
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
        help="Path to cells.parquet with cell metadata"
    )
    parser.add_argument(
        "--neighborhoods",
        type=Path,
        required=True,
        help="Path to neighborhoods.parquet with spatial context"
    )
    parser.add_argument(
        "--inference-outputs",
        type=Path,
        required=True,
        help="Path to model inference outputs (parquet with transition_prob, niche_influence)"
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
    available_genes = [g for g in gene_list if g in adata.var_names]
    missing_genes = [g for g in gene_list if g not in adata.var_names]

    if missing_genes:
        logger.warning(f"Missing genes for {score_name}: {missing_genes}")

    if len(available_genes) < 2:
        logger.error(f"Too few genes available for {score_name}: {available_genes}")
        return np.full(adata.n_obs, np.nan)

    logger.info(f"Computing {score_name} with {len(available_genes)}/{len(gene_list)} genes")

    sc.tl.score_genes(adata, available_genes, score_name=score_name)
    return adata.obs[score_name].values


def analyze_model_correlation(
    ap1_scores: pd.DataFrame,
    inference_df: pd.DataFrame,
    cells_df: pd.DataFrame,
) -> dict:
    """
    Correlate AP-1 signature with MODEL outputs, not raw stage.

    This is the key validation: do cells the model predicts as high-risk
    have elevated AP-1 stress response?
    """
    logger.info("=" * 60)
    logger.info("AP-1 vs MODEL PREDICTIONS")
    logger.info("=" * 60)

    # Merge AP-1 scores with model outputs
    merged = ap1_scores.merge(
        inference_df[["cell_id", "transition_prob", "niche_influence"]],
        on="cell_id",
        how="inner"
    )
    logger.info(f"Merged {len(merged):,} cells with model outputs")

    if len(merged) < 100:
        logger.error("Too few cells for analysis")
        return {"error": "insufficient_cells"}

    results = {}

    # Core correlation: AP-1 vs transition probability
    valid = merged["ap1_score"].notna() & merged["transition_prob"].notna()
    if valid.sum() > 10:
        rho, pval = stats.spearmanr(
            merged.loc[valid, "ap1_score"],
            merged.loc[valid, "transition_prob"]
        )
        results["ap1_transition_prob"] = {
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "n_cells": int(valid.sum()),
            "interpretation": (
                "POSITIVE: High AP-1 cells have higher model-predicted progression"
                if rho > 0.05 else
                "NEGATIVE: High AP-1 cells have lower model-predicted progression"
                if rho < -0.05 else
                "WEAK: No strong relationship"
            )
        }
        logger.info(f"AP-1 vs transition_prob: rho={rho:.4f}, p={pval:.2e}")

    # AP-1 vs niche influence
    valid = merged["ap1_score"].notna() & merged["niche_influence"].notna()
    if valid.sum() > 10:
        rho, pval = stats.spearmanr(
            merged.loc[valid, "ap1_score"],
            merged.loc[valid, "niche_influence"]
        )
        results["ap1_niche_influence"] = {
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "n_cells": int(valid.sum()),
            "interpretation": (
                "High AP-1 cells are in high-influence niches"
                if rho > 0.05 else
                "High AP-1 cells are in low-influence niches"
                if rho < -0.05 else
                "No strong niche pattern"
            )
        }
        logger.info(f"AP-1 vs niche_influence: rho={rho:.4f}, p={pval:.2e}")

    # Stratified analysis: AP-1 in high-risk vs low-risk cells
    trans_median = merged["transition_prob"].median()
    high_risk = merged["transition_prob"] > merged["transition_prob"].quantile(0.75)
    low_risk = merged["transition_prob"] < merged["transition_prob"].quantile(0.25)

    ap1_high_risk = merged.loc[high_risk, "ap1_score"].dropna()
    ap1_low_risk = merged.loc[low_risk, "ap1_score"].dropna()

    if len(ap1_high_risk) > 10 and len(ap1_low_risk) > 10:
        stat, pval = stats.mannwhitneyu(ap1_high_risk, ap1_low_risk, alternative="two-sided")
        effect_size = (ap1_high_risk.mean() - ap1_low_risk.mean()) / merged["ap1_score"].std()

        results["ap1_risk_stratification"] = {
            "high_risk_mean": float(ap1_high_risk.mean()),
            "low_risk_mean": float(ap1_low_risk.mean()),
            "cohens_d": float(effect_size),
            "mannwhitney_pval": float(pval),
            "n_high_risk": int(len(ap1_high_risk)),
            "n_low_risk": int(len(ap1_low_risk)),
        }
        logger.info(f"AP-1 high vs low risk: Cohen's d={effect_size:.4f}, p={pval:.2e}")

    # Add cell type info if available
    if "cell_type" in cells_df.columns:
        merged = merged.merge(
            cells_df[["cell_id", "cell_type", "stage"]].drop_duplicates(),
            on="cell_id",
            how="left"
        )

        # Per-cell-type correlation
        cell_type_corrs = {}
        for ct in merged["cell_type"].dropna().unique():
            ct_mask = merged["cell_type"] == ct
            ct_data = merged.loc[ct_mask]
            valid = ct_data["ap1_score"].notna() & ct_data["transition_prob"].notna()
            if valid.sum() > 30:
                rho, pval = stats.spearmanr(
                    ct_data.loc[valid, "ap1_score"],
                    ct_data.loc[valid, "transition_prob"]
                )
                cell_type_corrs[ct] = {
                    "rho": float(rho),
                    "pval": float(pval),
                    "n": int(valid.sum())
                }

        # Sort by correlation strength
        if cell_type_corrs:
            sorted_cts = sorted(cell_type_corrs.items(), key=lambda x: abs(x[1]["rho"]), reverse=True)
            results["cell_type_correlations"] = dict(sorted_cts[:10])
            logger.info(f"Top cell type by |correlation|: {sorted_cts[0][0]} (rho={sorted_cts[0][1]['rho']:.3f})")

    return results


def analyze_spatial_model_correlation(
    neighborhoods_df: pd.DataFrame,
    ap1_scores: pd.DataFrame,
    inference_df: pd.DataFrame,
) -> dict:
    """Analyze AP-1 in spatial context with model predictions."""
    logger.info("=" * 60)
    logger.info("SPATIAL AP-1 vs MODEL PREDICTIONS")
    logger.info("=" * 60)

    # Merge all data
    merged = neighborhoods_df.merge(
        ap1_scores[["cell_id", "ap1_score"]],
        on="cell_id",
        how="left"
    ).merge(
        inference_df[["cell_id", "transition_prob", "niche_influence"]],
        on="cell_id",
        how="left"
    )

    n_valid = (merged["ap1_score"].notna() & merged["transition_prob"].notna()).sum()
    logger.info(f"Spatial cells with scores and predictions: {n_valid:,}")

    results = {"n_neighborhoods": len(merged)}

    # Find AP-1 hotspots
    ap1_threshold = merged["ap1_score"].quantile(0.9)
    hotspot_mask = merged["ap1_score"] > ap1_threshold

    if hotspot_mask.sum() > 10:
        # Do AP-1 hotspots have higher transition probabilities?
        hotspot_trans = merged.loc[hotspot_mask, "transition_prob"].dropna()
        nonhotspot_trans = merged.loc[~hotspot_mask, "transition_prob"].dropna()

        if len(hotspot_trans) > 10 and len(nonhotspot_trans) > 10:
            stat, pval = stats.mannwhitneyu(hotspot_trans, nonhotspot_trans, alternative="two-sided")
            results["hotspot_transition"] = {
                "hotspot_mean_trans": float(hotspot_trans.mean()),
                "nonhotspot_mean_trans": float(nonhotspot_trans.mean()),
                "mannwhitney_pval": float(pval),
                "n_hotspot": int(len(hotspot_trans)),
            }
            logger.info(
                f"AP-1 hotspot transition prob: {hotspot_trans.mean():.4f} vs "
                f"non-hotspot: {nonhotspot_trans.mean():.4f} (p={pval:.2e})"
            )

    return results


def generate_validation_report(
    model_results: dict,
    spatial_results: dict,
    stage_baseline: dict,
) -> dict:
    """Generate comprehensive validation report."""

    report = {
        "hypothesis": "AP-1 stress signature correlates with model-predicted progression risk",
        "biological_basis": {
            "key_finding": "AP-1 (JUN/FOS) marks plastic, stress-responsive state",
            "mechanism": "JNK/ERK/p38 stress pathways converge on AP-1",
            "connection_to_h1_2": "IL1B signaling activates AP-1 via MYD88/IRAK/TRAF6",
        },
        "model_validation": model_results,
        "spatial_validation": spatial_results,
        "stage_baseline": stage_baseline,
        "validation_status": {},
    }

    # Determine validation status
    ap1_trans = model_results.get("ap1_transition_prob", {})
    rho = ap1_trans.get("spearman_rho")

    if rho is not None:
        if rho > 0.1:
            report["validation_status"]["ap1_model_prediction"] = "SUPPORTED"
            report["validation_status"]["detail"] = (
                f"Cells with high AP-1 have higher model-predicted transition probability (rho={rho:.3f})"
            )
        elif rho < -0.1:
            report["validation_status"]["ap1_model_prediction"] = "INVERSE"
            report["validation_status"]["detail"] = (
                f"Cells with high AP-1 have LOWER model-predicted transition (rho={rho:.3f}). "
                "This suggests AP-1 marks protective stress response, not progression-prone state."
            )
        else:
            report["validation_status"]["ap1_model_prediction"] = "WEAK"
            report["validation_status"]["detail"] = f"No strong correlation (rho={rho:.3f})"

    # Risk stratification
    strat = model_results.get("ap1_risk_stratification", {})
    if strat.get("cohens_d") is not None:
        d = strat["cohens_d"]
        if abs(d) > 0.2:
            report["validation_status"]["risk_stratification"] = (
                "HIGH_RISK_ENRICHED" if d > 0 else "LOW_RISK_ENRICHED"
            )
        else:
            report["validation_status"]["risk_stratification"] = "NO_DIFFERENCE"

    return report


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("AP-1 STRESS SIGNATURE VALIDATION (MODEL-BASED)")
    logger.info("=" * 70)
    logger.info(f"Cells: {args.cells}")
    logger.info(f"Inference outputs: {args.inference_outputs}")
    logger.info(f"Output: {args.output_dir}")

    # Load data
    logger.info("\n[1/5] Loading cell metadata...")
    cells_df = pd.read_parquet(args.cells)
    logger.info(f"  Loaded {len(cells_df):,} cells")

    logger.info("\n[2/5] Loading model inference outputs...")
    if not args.inference_outputs.exists():
        logger.error(f"Inference outputs not found: {args.inference_outputs}")
        logger.error("Run model inference first to generate transition_prob and niche_influence")
        return 1

    inference_df = pd.read_parquet(args.inference_outputs)
    logger.info(f"  Loaded {len(inference_df):,} inference outputs")
    logger.info(f"  Columns: {list(inference_df.columns)}")

    # Load neighborhoods
    logger.info("\n[3/5] Loading spatial neighborhoods...")
    neighborhoods_df = pd.read_parquet(args.neighborhoods)
    logger.info(f"  Loaded {len(neighborhoods_df):,} neighborhoods")

    # Compute AP-1 scores from expression
    gene_list = AP1_EXTENDED_GENES if args.use_extended else AP1_CORE_GENES

    if args.snrna and args.snrna.exists():
        logger.info("\n[4/5] Computing AP-1 scores from expression...")
        adata = sc.read_h5ad(args.snrna)
        logger.info(f"  Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        ap1_scores = compute_signature_score(adata, gene_list, "ap1_score")

        ap1_df = pd.DataFrame({
            "cell_id": adata.obs_names,
            "ap1_score": ap1_scores,
        })

        # Save cell scores
        ap1_df.to_parquet(args.output_dir / "ap1_cell_scores.parquet", index=False)
        logger.info(f"  Saved: ap1_cell_scores.parquet")

        # Baseline: correlation with stage (for comparison)
        if "cell_id" in cells_df.columns:
            ap1_with_stage = ap1_df.merge(
                cells_df[["cell_id", "stage"]].drop_duplicates(),
                on="cell_id",
                how="left"
            )
            valid = ap1_with_stage["stage"].isin(STAGES) & ap1_with_stage["ap1_score"].notna()
            if valid.sum() > 10:
                stage_numeric = ap1_with_stage.loc[valid, "stage"].map(STAGE_ORDER)
                rho, pval = stats.spearmanr(stage_numeric, ap1_with_stage.loc[valid, "ap1_score"])
                stage_baseline = {
                    "ap1_vs_stage_rho": float(rho),
                    "ap1_vs_stage_pval": float(pval),
                    "note": "Baseline correlation with raw stage (not model prediction)"
                }
            else:
                stage_baseline = {}
        else:
            stage_baseline = {}
    else:
        logger.error("snRNA h5ad required to compute AP-1 scores")
        return 1

    # Core validation: correlate with model outputs
    logger.info("\n[5/5] Correlating AP-1 with model predictions...")
    model_results = analyze_model_correlation(ap1_df, inference_df, cells_df)

    # Spatial analysis
    spatial_results = analyze_spatial_model_correlation(neighborhoods_df, ap1_df, inference_df)

    # Generate report
    logger.info("\nGenerating validation report...")
    report = generate_validation_report(model_results, spatial_results, stage_baseline)

    # Save results
    with open(args.output_dir / "ap1_validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved: ap1_validation_report.json")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("AP-1 VALIDATION SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nBaseline (AP-1 vs raw stage): rho={stage_baseline.get('ap1_vs_stage_rho', 'N/A')}")

    ap1_trans = model_results.get("ap1_transition_prob", {})
    logger.info(f"AP-1 vs MODEL transition_prob: rho={ap1_trans.get('spearman_rho', 'N/A')}")

    ap1_niche = model_results.get("ap1_niche_influence", {})
    logger.info(f"AP-1 vs niche_influence: rho={ap1_niche.get('spearman_rho', 'N/A')}")

    strat = model_results.get("ap1_risk_stratification", {})
    if strat:
        logger.info(f"AP-1 in high-risk cells: {strat.get('high_risk_mean', 'N/A'):.4f}")
        logger.info(f"AP-1 in low-risk cells: {strat.get('low_risk_mean', 'N/A'):.4f}")
        logger.info(f"Cohen's d: {strat.get('cohens_d', 'N/A'):.4f}")

    logger.info("\nValidation Status:")
    for key, value in report.get("validation_status", {}).items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
