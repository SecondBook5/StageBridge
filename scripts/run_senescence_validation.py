#!/usr/bin/env python3
"""
Senescence/SASP Signature Validation for StageBridge.

Validates the dual role of cellular senescence in precancer progression:
- Early/transient senescence = tumor-suppressive barrier
- Chronic senescence with SASP = tumor-promoting microenvironment

Key biological insight (from Hoi et al. Cancer Cell 2026):
- SASP (senescence-associated secretory phenotype) creates feedforward loops
- Senescent fibroblasts secrete IL6, GDF15, matrix factors
- SASP activates ERK/p38/AKT in epithelial cells
- cGAS-STING pathway drives chronic inflammatory cytokine production

This connects to other StageBridge hypotheses:
- H1.2: SASP includes IL1B secretion (IL1B-IL1R1 axis)
- AP-1: p38 converges on AP-1 transcription factors

Senescence Core Markers:
  CDKN1A (p21), CDKN2A (p16), TP53, RB1

SASP Signature:
  IL6, IL1B, IL1A, CXCL8, CCL2, GDF15, SERPINE1 (PAI-1)
  MMP1, MMP3, MMP9 (matrix remodeling)

cGAS-STING Pathway:
  MB21D1 (cGAS), TMEM173 (STING), IRF3, IFNB1

Usage:
  python scripts/run_senescence_validation.py \
      --snrna /path/to/snrna_with_celltypes.h5ad \
      --cells /path/to/cells.parquet \
      --neighborhoods /path/to/neighborhoods.parquet \
      --output-dir /path/to/senescence_results
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


# Senescence core markers (cell cycle arrest)
SENESCENCE_CORE = ["CDKN1A", "CDKN2A", "TP53", "RB1"]

# SASP - Senescence-Associated Secretory Phenotype
SASP_CYTOKINES = ["IL6", "IL1B", "IL1A", "CXCL8", "CCL2", "CXCL1", "CXCL2"]
SASP_GROWTH_FACTORS = ["GDF15", "VEGFA", "FGF2", "HGF", "AREG"]
SASP_MATRIX = ["MMP1", "MMP3", "MMP9", "SERPINE1", "TIMP1", "FN1"]
SASP_ALL = SASP_CYTOKINES + SASP_GROWTH_FACTORS + SASP_MATRIX

# cGAS-STING inflammatory pathway
CGAS_STING = ["MB21D1", "TMEM173", "IRF3", "IFNB1", "TBK1", "NFKB1"]

# Full senescence signature
SENESCENCE_FULL = SENESCENCE_CORE + SASP_ALL + CGAS_STING

# Markers that decrease with senescence (for validation)
SENESCENCE_DECREASED = ["LMNB1", "MKI67", "PCNA", "TOP2A"]

# Stages for progression analysis
STAGES = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
STAGE_ORDER = {s: i for i, s in enumerate(STAGES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Senescence/SASP Signature Validation"
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
        help="Output directory for senescence validation results"
    )
    parser.add_argument(
        "--ap1-scores",
        type=Path,
        default=None,
        help="Path to AP-1 cell scores (for cross-validation)"
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


def compute_gene_expression_matrix(
    adata: sc.AnnData,
    gene_list: list[str]
) -> pd.DataFrame:
    """Extract expression matrix for specific genes."""
    available_genes = [g for g in gene_list if g in adata.var_names]

    if not available_genes:
        return pd.DataFrame()

    expr = adata[:, available_genes].X
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()

    return pd.DataFrame(
        expr,
        index=adata.obs_names,
        columns=available_genes
    )


def classify_senescence_state(
    senescence_score: np.ndarray,
    sasp_score: np.ndarray,
    proliferation_score: np.ndarray
) -> np.ndarray:
    """
    Classify cells into senescence states:
    - proliferating: high proliferation markers
    - quiescent: low proliferation, low senescence
    - early_senescent: high senescence, low SASP
    - chronic_senescent: high senescence, high SASP (pro-tumorigenic)
    """
    states = np.full(len(senescence_score), "unknown", dtype=object)

    # Thresholds based on quartiles
    sen_high = np.nanpercentile(senescence_score, 75)
    sasp_high = np.nanpercentile(sasp_score, 75)
    prolif_high = np.nanpercentile(proliferation_score, 50)

    # Classify
    proliferating = proliferation_score > prolif_high
    sen_pos = senescence_score > sen_high
    sasp_pos = sasp_score > sasp_high

    states[proliferating] = "proliferating"
    states[~proliferating & ~sen_pos] = "quiescent"
    states[sen_pos & ~sasp_pos] = "early_senescent"
    states[sen_pos & sasp_pos] = "chronic_senescent"

    return states


def analyze_single_cell(
    adata: sc.AnnData,
    cells_df: pd.DataFrame,
    ap1_scores_df: Optional[pd.DataFrame] = None
) -> dict:
    """
    Analyze senescence signatures in single-cell RNA-seq data.

    Returns dict with:
    - Per-cell senescence and SASP scores
    - Stage-stratified statistics
    - Senescence state classification
    - Cell type enrichment
    - Connection to AP-1 (if provided)
    """
    logger.info("=" * 60)
    logger.info("SINGLE-CELL SENESCENCE ANALYSIS")
    logger.info("=" * 60)

    # Compute signature scores
    senescence_scores = compute_signature_score(adata, SENESCENCE_CORE, "senescence_score")
    sasp_scores = compute_signature_score(adata, SASP_ALL, "sasp_score")
    sasp_cytokine_scores = compute_signature_score(adata, SASP_CYTOKINES, "sasp_cytokine_score")
    sasp_matrix_scores = compute_signature_score(adata, SASP_MATRIX, "sasp_matrix_score")
    cgas_sting_scores = compute_signature_score(adata, CGAS_STING, "cgas_sting_score")

    # Proliferation markers (should be low in senescent cells)
    prolif_genes = ["MKI67", "PCNA", "TOP2A", "MCM2"]
    prolif_scores = compute_signature_score(adata, prolif_genes, "proliferation_score")

    # Build results dataframe
    results_df = pd.DataFrame({
        "cell_id": adata.obs_names,
        "senescence_score": senescence_scores,
        "sasp_score": sasp_scores,
        "sasp_cytokine_score": sasp_cytokine_scores,
        "sasp_matrix_score": sasp_matrix_scores,
        "cgas_sting_score": cgas_sting_scores,
        "proliferation_score": prolif_scores,
    })

    # Classify senescence states
    results_df["senescence_state"] = classify_senescence_state(
        senescence_scores, sasp_scores, prolif_scores
    )

    # Add key individual genes
    key_genes = ["CDKN1A", "CDKN2A", "IL6", "IL1B", "GDF15", "MMP1"]
    gene_expr = compute_gene_expression_matrix(adata, key_genes)
    for gene in gene_expr.columns:
        results_df[f"expr_{gene}"] = gene_expr[gene].values

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
                sen_scores = results_df.loc[mask, "senescence_score"].dropna()
                sasp_scores_stage = results_df.loc[mask, "sasp_score"].dropna()

                # Count senescence states
                state_counts = results_df.loc[mask, "senescence_state"].value_counts()

                stage_stats[stage] = {
                    "n_cells": int(mask.sum()),
                    "mean_senescence": float(sen_scores.mean()) if len(sen_scores) > 0 else None,
                    "mean_sasp": float(sasp_scores_stage.mean()) if len(sasp_scores_stage) > 0 else None,
                    "pct_chronic_senescent": float(
                        state_counts.get("chronic_senescent", 0) / mask.sum() * 100
                    ),
                    "pct_early_senescent": float(
                        state_counts.get("early_senescent", 0) / mask.sum() * 100
                    ),
                    "sasp_to_senescence_ratio": float(
                        sasp_scores_stage.mean() / sen_scores.mean()
                    ) if len(sen_scores) > 0 and sen_scores.mean() != 0 else None,
                }

        # Progression correlation
        valid_mask = results_df["stage"].isin(STAGES) & results_df["senescence_score"].notna()
        if valid_mask.sum() > 10:
            stage_numeric = results_df.loc[valid_mask, "stage"].map(STAGE_ORDER)

            # Senescence vs progression
            rho_sen, pval_sen = stats.spearmanr(
                stage_numeric,
                results_df.loc[valid_mask, "senescence_score"]
            )
            # SASP vs progression
            rho_sasp, pval_sasp = stats.spearmanr(
                stage_numeric,
                results_df.loc[valid_mask, "sasp_score"]
            )

            stage_stats["progression_correlation"] = {
                "senescence_rho": float(rho_sen),
                "senescence_pval": float(pval_sen),
                "sasp_rho": float(rho_sasp),
                "sasp_pval": float(pval_sasp),
                "n_cells": int(valid_mask.sum()),
            }

    # Cell type enrichment for chronic senescence
    cell_type_stats = {}
    if "cell_type" in results_df.columns:
        for ct in results_df["cell_type"].dropna().unique():
            mask = results_df["cell_type"] == ct
            if mask.sum() > 10:
                state_counts = results_df.loc[mask, "senescence_state"].value_counts()
                cell_type_stats[ct] = {
                    "n_cells": int(mask.sum()),
                    "pct_chronic_senescent": float(
                        state_counts.get("chronic_senescent", 0) / mask.sum() * 100
                    ),
                    "mean_sasp": float(results_df.loc[mask, "sasp_score"].mean()),
                }

        # Rank by chronic senescence burden
        if cell_type_stats:
            ranked = sorted(
                cell_type_stats.items(),
                key=lambda x: x[1]["pct_chronic_senescent"],
                reverse=True
            )
            cell_type_stats["ranking_chronic"] = [ct for ct, _ in ranked]

    # IL1B connection (links to H1.2)
    il1b_stats = {}
    il1b_expr = compute_gene_expression_matrix(adata, ["IL1B", "IL6"])
    if "IL1B" in il1b_expr.columns and results_df["sasp_score"].notna().sum() > 10:
        rho, pval = stats.spearmanr(
            il1b_expr["IL1B"],
            results_df["sasp_score"],
            nan_policy="omit"
        )
        il1b_stats["il1b_sasp_correlation"] = float(rho) if not np.isnan(rho) else None
        il1b_stats["il1b_sasp_pval"] = float(pval) if not np.isnan(pval) else None

    # AP-1 connection (if provided)
    ap1_stats = {}
    if ap1_scores_df is not None and "cell_id" in ap1_scores_df.columns:
        merged = results_df.merge(
            ap1_scores_df[["cell_id", "ap1_score"]],
            on="cell_id",
            how="left"
        )
        valid = merged["sasp_score"].notna() & merged["ap1_score"].notna()
        if valid.sum() > 10:
            rho, pval = stats.spearmanr(
                merged.loc[valid, "sasp_score"],
                merged.loc[valid, "ap1_score"]
            )
            ap1_stats["sasp_ap1_correlation"] = float(rho)
            ap1_stats["sasp_ap1_pval"] = float(pval)
            logger.info(f"SASP-AP1 correlation: rho={rho:.4f}, p={pval:.2e}")

    return {
        "cell_scores": results_df,
        "stage_stats": stage_stats,
        "cell_type_stats": cell_type_stats,
        "il1b_connection": il1b_stats,
        "ap1_connection": ap1_stats,
        "genes_used": {
            "senescence_core": SENESCENCE_CORE,
            "sasp_all": SASP_ALL,
            "cgas_sting": CGAS_STING,
        },
    }


def analyze_spatial_neighborhoods(
    neighborhoods_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    sc_senescence_scores: Optional[pd.DataFrame] = None
) -> dict:
    """
    Analyze senescence in spatial neighborhood context.

    Tests whether:
    1. Chronic senescent cells cluster spatially (SASP feedforward)
    2. Senescent niches associate with specific stages
    3. SASP creates local inflammatory microenvironment
    """
    logger.info("=" * 60)
    logger.info("SPATIAL NEIGHBORHOOD SENESCENCE ANALYSIS")
    logger.info("=" * 60)

    logger.info(f"Neighborhoods: {len(neighborhoods_df):,} rows")

    results = {
        "n_neighborhoods": len(neighborhoods_df),
        "spatial_stats": {},
    }

    if sc_senescence_scores is not None and "cell_id" in neighborhoods_df.columns:
        # Merge senescence scores with neighborhoods
        score_cols = ["cell_id", "senescence_score", "sasp_score", "senescence_state"]
        if "stage" in sc_senescence_scores.columns:
            score_cols.append("stage")
        if "cell_type" in sc_senescence_scores.columns:
            score_cols.append("cell_type")

        merged = neighborhoods_df.merge(
            sc_senescence_scores[score_cols],
            on="cell_id",
            how="left"
        )

        n_with_scores = merged["senescence_score"].notna().sum()
        logger.info(f"Neighborhoods with scores: {n_with_scores:,}/{len(merged):,}")

        if n_with_scores > 0:
            # Stage-stratified spatial analysis
            if "stage" in merged.columns:
                for stage in STAGES:
                    mask = merged["stage"] == stage
                    if mask.sum() > 10:
                        sen_scores = merged.loc[mask, "senescence_score"].dropna()
                        sasp_scores = merged.loc[mask, "sasp_score"].dropna()
                        state_counts = merged.loc[mask, "senescence_state"].value_counts()

                        results["spatial_stats"][stage] = {
                            "n_neighborhoods": int(mask.sum()),
                            "mean_senescence": float(sen_scores.mean()),
                            "mean_sasp": float(sasp_scores.mean()),
                            "pct_chronic": float(
                                state_counts.get("chronic_senescent", 0) / mask.sum() * 100
                            ),
                        }

            # Identify SASP hotspots (high SASP neighborhoods)
            sasp_threshold = merged["sasp_score"].quantile(0.9)
            hotspot_mask = merged["sasp_score"] > sasp_threshold

            results["sasp_hotspots"] = {
                "threshold": float(sasp_threshold),
                "n_hotspots": int(hotspot_mask.sum()),
                "pct_hotspots": float(hotspot_mask.mean() * 100),
            }

            # Stage enrichment in SASP hotspots
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
                results["sasp_hotspots"]["stage_enrichment"] = enrichment

            # Cell type composition of SASP hotspots
            if "cell_type" in merged.columns:
                hotspot_cts = merged.loc[hotspot_mask, "cell_type"].value_counts()
                ct_enrichment = {}
                all_cts = merged["cell_type"].value_counts()
                for ct in hotspot_cts.index[:10]:  # Top 10
                    obs_pct = hotspot_cts.get(ct, 0) / hotspot_mask.sum()
                    exp_pct = all_cts.get(ct, 0) / len(merged)
                    if exp_pct > 0:
                        ct_enrichment[ct] = {
                            "fold_enrichment": float(obs_pct / exp_pct),
                        }
                results["sasp_hotspots"]["cell_type_enrichment"] = ct_enrichment

    return results


def generate_validation_report(
    sc_results: dict,
    spatial_results: dict,
    output_dir: Path
) -> dict:
    """Generate comprehensive validation report."""

    report = {
        "hypothesis": "Chronic senescence with SASP creates pro-tumorigenic microenvironment",
        "biological_basis": {
            "key_finding": "Senescence is dual: early=protective, chronic+SASP=tumorigenic",
            "mechanism": "SASP creates feedforward loops via IL6, GDF15, matrix remodeling",
            "connection_to_h1_2": "SASP includes IL1B secretion",
            "connection_to_ap1": "p38 pathway converges on AP-1",
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
            "stage_chronic_senescence": {
                s: stage_stats.get(s, {}).get("pct_chronic_senescent")
                for s in STAGES if s in stage_stats
            },
            "stage_sasp_means": {
                s: stage_stats.get(s, {}).get("mean_sasp")
                for s in STAGES if s in stage_stats
            },
            "senescence_progression_rho": prog_corr.get("senescence_rho"),
            "sasp_progression_rho": prog_corr.get("sasp_rho"),
        }

        # Top cell types by chronic senescence
        ct_stats = sc_results.get("cell_type_stats", {})
        if "ranking_chronic" in ct_stats:
            report["single_cell"]["top_chronic_cell_types"] = ct_stats["ranking_chronic"][:5]

        # IL1B connection
        il1b = sc_results.get("il1b_connection", {})
        if il1b.get("il1b_sasp_correlation"):
            report["single_cell"]["il1b_sasp_correlation"] = il1b["il1b_sasp_correlation"]

        # AP1 connection
        ap1 = sc_results.get("ap1_connection", {})
        if ap1.get("sasp_ap1_correlation"):
            report["single_cell"]["sasp_ap1_correlation"] = ap1["sasp_ap1_correlation"]

    # Spatial summary
    if spatial_results:
        report["spatial"] = {
            "n_neighborhoods": spatial_results.get("n_neighborhoods"),
            "stage_spatial_stats": spatial_results.get("spatial_stats", {}),
        }

        hotspots = spatial_results.get("sasp_hotspots", {})
        if hotspots:
            report["spatial"]["sasp_hotspots"] = {
                "n_hotspots": hotspots.get("n_hotspots"),
                "pct_hotspots": hotspots.get("pct_hotspots"),
                "stage_enrichment": hotspots.get("stage_enrichment", {}),
            }

    # Validation status
    # Check if SASP correlates with progression (expected: positive in later stages)
    sasp_rho = report["single_cell"].get("sasp_progression_rho")
    if sasp_rho is not None:
        if sasp_rho > 0.05:
            report["validation_status"]["sasp_progression"] = "SUPPORTED"
            report["validation_status"]["sasp_detail"] = (
                f"SASP increases with progression (rho={sasp_rho:.3f})"
            )
        else:
            report["validation_status"]["sasp_progression"] = "WEAK_OR_ABSENT"

    # Check chronic senescence pattern (expected: higher in intermediate stages)
    chronic_pcts = report["single_cell"].get("stage_chronic_senescence", {})
    if chronic_pcts:
        early = np.mean([chronic_pcts.get(s, 0) or 0 for s in ["AAH", "AIS"]])
        late = chronic_pcts.get("LUAD", 0) or 0
        if early > late:
            report["validation_status"]["chronic_pattern"] = "EARLY_ENRICHED"
            report["validation_status"]["chronic_detail"] = (
                f"Chronic senescence higher in early stages (early={early:.1f}%, LUAD={late:.1f}%)"
            )
        else:
            report["validation_status"]["chronic_pattern"] = "LATE_ENRICHED"

    # Check IL1B-SASP connection
    il1b_corr = report["single_cell"].get("il1b_sasp_correlation")
    if il1b_corr is not None:
        if il1b_corr > 0.2:
            report["validation_status"]["il1b_sasp_axis"] = "STRONG"
        elif il1b_corr > 0.1:
            report["validation_status"]["il1b_sasp_axis"] = "MODERATE"
        else:
            report["validation_status"]["il1b_sasp_axis"] = "WEAK"

    # Check AP1-SASP connection
    ap1_corr = report["single_cell"].get("sasp_ap1_correlation")
    if ap1_corr is not None:
        if ap1_corr > 0.2:
            report["validation_status"]["sasp_ap1_axis"] = "STRONG"
        elif ap1_corr > 0.1:
            report["validation_status"]["sasp_ap1_axis"] = "MODERATE"
        else:
            report["validation_status"]["sasp_ap1_axis"] = "WEAK"

    return report


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SENESCENCE/SASP SIGNATURE VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Cells: {args.cells}")
    logger.info(f"Neighborhoods: {args.neighborhoods}")
    logger.info(f"snRNA: {args.snrna}")
    logger.info(f"Output: {args.output_dir}")

    # Load cell metadata
    logger.info("\n[1/5] Loading cell metadata...")
    cells_df = pd.read_parquet(args.cells)
    logger.info(f"  Loaded {len(cells_df):,} cells")

    # Load neighborhoods
    logger.info("\n[2/5] Loading spatial neighborhoods...")
    neighborhoods_df = pd.read_parquet(args.neighborhoods)
    logger.info(f"  Loaded {len(neighborhoods_df):,} neighborhoods")

    # Load AP-1 scores if available (for cross-validation)
    ap1_scores_df = None
    if args.ap1_scores and args.ap1_scores.exists():
        logger.info("\n[3/5] Loading AP-1 scores for cross-validation...")
        ap1_scores_df = pd.read_parquet(args.ap1_scores)
        logger.info(f"  Loaded {len(ap1_scores_df):,} AP-1 scores")
    else:
        logger.info("\n[3/5] No AP-1 scores provided, skipping cross-validation")

    # Single-cell analysis
    sc_results = None
    if args.snrna and args.snrna.exists():
        logger.info("\n[4/5] Single-cell senescence analysis...")
        adata = sc.read_h5ad(args.snrna)
        logger.info(f"  Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        sc_results = analyze_single_cell(adata, cells_df, ap1_scores_df)

        # Save cell-level scores
        sc_results["cell_scores"].to_parquet(
            args.output_dir / "senescence_cell_scores.parquet",
            index=False
        )
        logger.info(f"  Saved: senescence_cell_scores.parquet")
    else:
        logger.warning("\n[4/5] Skipping single-cell analysis (no snRNA h5ad)")

    # Spatial analysis
    logger.info("\n[5/5] Spatial neighborhood analysis...")
    sc_scores = sc_results["cell_scores"] if sc_results else None
    spatial_results = analyze_spatial_neighborhoods(
        neighborhoods_df, cells_df, sc_scores
    )

    # Generate report
    logger.info("\nGenerating validation report...")
    report = generate_validation_report(sc_results, spatial_results, args.output_dir)

    # Save results
    with open(args.output_dir / "senescence_validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved: senescence_validation_report.json")

    if sc_results:
        with open(args.output_dir / "senescence_singlecell_stats.json", "w") as f:
            stats_to_save = {k: v for k, v in sc_results.items() if k != "cell_scores"}
            json.dump(stats_to_save, f, indent=2, default=str)
        logger.info(f"Saved: senescence_singlecell_stats.json")

    with open(args.output_dir / "senescence_spatial_stats.json", "w") as f:
        json.dump(spatial_results, f, indent=2, default=str)
    logger.info(f"Saved: senescence_spatial_stats.json")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SENESCENCE VALIDATION SUMMARY")
    logger.info("=" * 70)

    if sc_results:
        stage_stats = sc_results.get("stage_stats", {})
        logger.info("\nChronic Senescence by Stage:")
        for stage in STAGES:
            if stage in stage_stats:
                s = stage_stats[stage]
                logger.info(f"  {stage}: {s['pct_chronic_senescent']:.1f}% chronic, "
                           f"SASP mean={s['mean_sasp']:.4f} (n={s['n_cells']:,})")

        prog = stage_stats.get("progression_correlation", {})
        if prog:
            logger.info(f"\nSASP-progression correlation: rho={prog['sasp_rho']:.4f}")

        il1b = sc_results.get("il1b_connection", {})
        if il1b.get("il1b_sasp_correlation"):
            logger.info(f"IL1B-SASP correlation: rho={il1b['il1b_sasp_correlation']:.4f}")

        ap1 = sc_results.get("ap1_connection", {})
        if ap1.get("sasp_ap1_correlation"):
            logger.info(f"SASP-AP1 correlation: rho={ap1['sasp_ap1_correlation']:.4f}")

    logger.info("\nValidation Status:")
    for key, value in report.get("validation_status", {}).items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
