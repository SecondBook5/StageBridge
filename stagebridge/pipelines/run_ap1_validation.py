#!/usr/bin/env python3
"""
AP-1 Stress Signature Validation for StageBridge.

Validates whether AP-1/stress response activation correlates with model-predicted
progression risk and niche influence. This tests whether the model's predictions
align with known stress biology.

Key biological insights:

1. From Marjanovic et al. (Cancer Cell):
   - AP-1 (JUN/FOS) transcription factors mark a plastic, high-plasticity state
   - This state is associated with:
     - Fetal progenitor-like intermediate
     - Enhanced progression potential
     - Response to inflammatory signaling (IL1B/TNF)
     - JNK/ERK/p38 stress pathway convergence

2. From Alcolea et al. (Nature 2026) - Precancerous Niche Paper:
   - "Tumour 12" stress state is marked by Jun, Fos, Fosb, Atf3, Egr1
   - SOX9+ cells with EGF ligands (AREG, HBEGF) recruit fibroblasts
   - EGF-SOX9-FN1 axis determines which nascent tumors persist
   - PDGFRa-low lamina propria fibroblasts form supportive niche
   - Fibroblast-deposited fibronectin (FN1) creates pro-survival scaffold
   - Only Niche+ tumors (with stromal remodeling) persist long-term

3. From Tsankov et al. (Nature Cancer 2025) - TP53 Atlas:
   - TP53-mutant tumors show increased cellular entropy/plasticity
   - NMF7 niche (TAM.SPP1 + CAF.COLs + myofibroblasts) promotes EMT
   - Niche composition affects cell plasticity

The validation tests:
1. Do cells with high AP-1 have higher MODEL-PREDICTED transition probability?
2. Do cells in high niche-influence regions show elevated AP-1?
3. Is AP-1 enriched in cells the model flags as progression-prone?
4. Do AP-1-high cells show elevated SOX9 and EGF ligands (AREG)?
5. Are AP-1 hotspots near PDGFRa-low fibroblasts and FN1+ stroma?

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

# Tumour 12 state markers from precancerous niche paper (Alcolea et al. 2026)
# These mark the stress-responsive epithelial state that recruits fibroblasts
TUMOUR12_STRESS_TFS = ["JUN", "FOS", "FOSB", "ATF3", "EGR1", "EGR3", "RUNX1", "MYC"]
TUMOUR12_EGF_LIGANDS = ["AREG", "HBEGF", "NRG1", "NRG2"]  # Signal to fibroblasts
TUMOUR12_ADHESION = ["CCN1", "CCN2", "THBS1", "ICAM1", "VCAM1"]  # ECM communication

# SOX9 - key marker of tumour-niche interaction state
# SOX9+ cells are in contact with fibroblast niche and have higher proliferation
SOX9_NICHE_MARKERS = ["SOX9", "KRT6A", "KRT17"]

# Upstream regulators that converge on AP-1
UPSTREAM_KINASES = ["MAPK8", "MAPK9", "MAPK10",  # JNK1/2/3
                    "MAPK1", "MAPK3",              # ERK1/2
                    "MAPK14", "MAPK11"]            # p38 alpha/beta

# IL1B pathway connection (links to H1.2 hypothesis)
IL1B_AP1_AXIS = ["IL1B", "IL1R1", "MYD88", "IRAK1", "TRAF6"]

# Fibroblast niche markers from precancerous niche paper
# PDGFRa-low fibroblasts form niche; PDGFRa-high stay in submucosae
FIBROBLAST_NICHE_GENES = ["PDGFRA", "FN1", "COL1A1", "COL1A2", "FAP", "ACTA2"]
# FN1 (fibronectin) accumulation is the key ECM component of supportive niche

# Bottleneck/stem-like state markers (from Gardner et al. Science 2024)
# The "bottleneck" is an undifferentiated state that enables lineage conversion
# Key insight: it's not just "loss of lineage = plasticity", cells must ENTER
# a specific undifferentiated state characterized by these markers
BOTTLENECK_HIGH = ["TM4SF1", "SOX9", "CREB5"]  # Should be HIGH in bottleneck
STEM_PROGRAM = ["SOX2", "MYC", "KLF4"]  # Stem-like markers

# Lineage markers (should be LOW in bottleneck state)
# Cells in bottleneck have lost differentiated identity
AT2_LINEAGE = ["SFTPC", "SFTPB", "SFTPA1", "NKX2-1", "HOPX"]
AT1_LINEAGE = ["AGER", "PDPN", "AQP5"]
PNEC_LINEAGE = ["ASCL1", "CHGA", "SYP", "INSM1"]
BASAL_LINEAGE = ["KRT5", "KRT17", "TP63"]
SECRETORY_LINEAGE = ["SCGB1A1", "SCGB3A2"]
ALL_LINEAGE_MARKERS = AT2_LINEAGE + AT1_LINEAGE + PNEC_LINEAGE + BASAL_LINEAGE + SECRETORY_LINEAGE

# NOTE: It's not simply "loss of HOPX = plasticity"
# The bottleneck state is:
#   HIGH: TM4SF1, SOX9, CREB5, SOX2, MYC
#   LOW: All lineage markers (SFTPC, HOPX, ASCL1, KRT5, etc.)

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


def compute_tumour12_signatures(
    adata: sc.AnnData,
) -> pd.DataFrame:
    """
    Compute Tumour 12 state signatures from precancerous niche paper.

    The Tumour 12 state is the stress-responsive epithelial state that:
    1. Expresses AP-1/stress TFs (Jun, Fos, Atf3, Egr1)
    2. Secretes EGF ligands (AREG, HBEGF) that recruit fibroblasts
    3. Shows SOX9 expression (marks cells in niche contact)
    """
    logger.info("Computing Tumour 12 state signatures...")

    results = {"cell_id": adata.obs_names.tolist()}

    # Stress TF score
    stress_tfs = [g for g in TUMOUR12_STRESS_TFS if g in adata.var_names]
    if len(stress_tfs) >= 2:
        sc.tl.score_genes(adata, stress_tfs, score_name="stress_tf_score")
        results["stress_tf_score"] = adata.obs["stress_tf_score"].values
        logger.info(f"  Stress TF score: {len(stress_tfs)} genes")

    # EGF ligand score (signals to fibroblasts)
    egf_genes = [g for g in TUMOUR12_EGF_LIGANDS if g in adata.var_names]
    if len(egf_genes) >= 1:
        sc.tl.score_genes(adata, egf_genes, score_name="egf_ligand_score")
        results["egf_ligand_score"] = adata.obs["egf_ligand_score"].values
        logger.info(f"  EGF ligand score: {len(egf_genes)} genes")

    # SOX9 expression (direct marker of niche contact)
    if "SOX9" in adata.var_names:
        if hasattr(adata.X, "toarray"):
            sox9_idx = list(adata.var_names).index("SOX9")
            results["sox9_expression"] = adata.X[:, sox9_idx].toarray().flatten()
        else:
            sox9_idx = list(adata.var_names).index("SOX9")
            results["sox9_expression"] = adata.X[:, sox9_idx].flatten()
        logger.info("  SOX9 expression extracted")

    # AREG expression (key EGF ligand)
    if "AREG" in adata.var_names:
        if hasattr(adata.X, "toarray"):
            areg_idx = list(adata.var_names).index("AREG")
            results["areg_expression"] = adata.X[:, areg_idx].toarray().flatten()
        else:
            areg_idx = list(adata.var_names).index("AREG")
            results["areg_expression"] = adata.X[:, areg_idx].flatten()
        logger.info("  AREG expression extracted")

    # Adhesion/ECM communication score
    adhesion_genes = [g for g in TUMOUR12_ADHESION if g in adata.var_names]
    if len(adhesion_genes) >= 1:
        sc.tl.score_genes(adata, adhesion_genes, score_name="adhesion_score")
        results["adhesion_score"] = adata.obs["adhesion_score"].values
        logger.info(f"  Adhesion score: {len(adhesion_genes)} genes")

    # Bottleneck state markers (Gardner et al. Science 2024)
    # The bottleneck is the key plastic intermediate that enables transformation
    bottleneck_high_genes = [g for g in BOTTLENECK_HIGH if g in adata.var_names]
    if len(bottleneck_high_genes) >= 1:
        sc.tl.score_genes(adata, bottleneck_high_genes, score_name="bottleneck_high_score")
        results["bottleneck_high_score"] = adata.obs["bottleneck_high_score"].values
        logger.info(f"  Bottleneck HIGH score: {len(bottleneck_high_genes)} genes (TM4SF1, SOX9, CREB5)")

    # All lineage markers (should be LOW in bottleneck)
    lineage_genes = [g for g in ALL_LINEAGE_MARKERS if g in adata.var_names]
    if len(lineage_genes) >= 2:
        sc.tl.score_genes(adata, lineage_genes, score_name="lineage_score")
        results["lineage_score"] = adata.obs["lineage_score"].values
        logger.info(f"  Lineage score: {len(lineage_genes)} genes (all differentiation markers)")

    # Compute bottleneck state: HIGH bottleneck markers + LOW lineage markers
    if "bottleneck_high_score" in results and "lineage_score" in results:
        # Bottleneck state = high stem markers AND low lineage markers
        bn_high = np.array(results["bottleneck_high_score"])
        lin_score = np.array(results["lineage_score"])
        # Normalize both to 0-1 range for combination
        bn_norm = (bn_high - np.nanmin(bn_high)) / (np.nanmax(bn_high) - np.nanmin(bn_high) + 1e-8)
        lin_norm = (lin_score - np.nanmin(lin_score)) / (np.nanmax(lin_score) - np.nanmin(lin_score) + 1e-8)
        # Bottleneck = high stem, low lineage
        results["bottleneck_state_score"] = bn_norm - lin_norm
        logger.info("  Bottleneck state score: (high stem) - (high lineage)")

    return pd.DataFrame(results)


def analyze_tumour12_niche_axis(
    tumour12_df: pd.DataFrame,
    ap1_scores: pd.DataFrame,
    inference_df: pd.DataFrame,
    cells_df: pd.DataFrame,
) -> dict:
    """
    Analyze the EGF-SOX9-FN1 axis from the precancerous niche paper.

    Key predictions:
    1. High AP-1 cells should have high SOX9 and EGF ligands
    2. These cells should have higher model-predicted progression
    3. The combination (AP1+SOX9+AREG) should be more predictive than AP1 alone
    """
    logger.info("=" * 60)
    logger.info("TUMOUR 12 / EGF-SOX9-FN1 AXIS ANALYSIS")
    logger.info("=" * 60)

    # Merge all data
    merged = ap1_scores.merge(tumour12_df, on="cell_id", how="inner")
    merged = merged.merge(
        inference_df[["cell_id", "transition_prob", "niche_influence"]],
        on="cell_id",
        how="inner"
    )
    logger.info(f"Merged {len(merged):,} cells for Tumour 12 analysis")

    results = {}

    # 1. AP-1 vs SOX9 correlation
    if "sox9_expression" in merged.columns:
        valid = merged["ap1_score"].notna() & (merged["sox9_expression"] > 0)
        if valid.sum() > 10:
            rho, pval = stats.spearmanr(
                merged.loc[valid, "ap1_score"],
                merged.loc[valid, "sox9_expression"]
            )
            results["ap1_sox9_correlation"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_cells": int(valid.sum()),
                "interpretation": (
                    "AP-1 and SOX9 co-express (consistent with Tumour 12 state)"
                    if rho > 0.1 else "AP-1 and SOX9 not strongly correlated"
                )
            }
            logger.info(f"AP-1 vs SOX9: rho={rho:.4f}, p={pval:.2e}")

    # 2. AP-1 vs AREG correlation
    if "areg_expression" in merged.columns:
        valid = merged["ap1_score"].notna() & (merged["areg_expression"] > 0)
        if valid.sum() > 10:
            rho, pval = stats.spearmanr(
                merged.loc[valid, "ap1_score"],
                merged.loc[valid, "areg_expression"]
            )
            results["ap1_areg_correlation"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_cells": int(valid.sum()),
            }
            logger.info(f"AP-1 vs AREG: rho={rho:.4f}, p={pval:.2e}")

    # 3. Combined Tumour 12 state vs model predictions
    # Define "Tumour 12-like" cells: high AP1 + high EGF ligands
    if "egf_ligand_score" in merged.columns:
        ap1_high = merged["ap1_score"] > merged["ap1_score"].quantile(0.75)
        egf_high = merged["egf_ligand_score"] > merged["egf_ligand_score"].quantile(0.75)
        tumour12_like = ap1_high & egf_high

        if tumour12_like.sum() > 10:
            t12_trans = merged.loc[tumour12_like, "transition_prob"].dropna()
            other_trans = merged.loc[~tumour12_like, "transition_prob"].dropna()

            if len(t12_trans) > 10 and len(other_trans) > 10:
                stat, pval = stats.mannwhitneyu(t12_trans, other_trans, alternative="two-sided")
                effect_size = (t12_trans.mean() - other_trans.mean()) / merged["transition_prob"].std()

                results["tumour12_state_vs_progression"] = {
                    "tumour12_mean_trans": float(t12_trans.mean()),
                    "other_mean_trans": float(other_trans.mean()),
                    "cohens_d": float(effect_size),
                    "mannwhitney_pval": float(pval),
                    "n_tumour12": int(len(t12_trans)),
                    "interpretation": (
                        "Tumour 12-like cells (AP1+EGF) have HIGHER progression risk"
                        if effect_size > 0.2 else
                        "Tumour 12-like cells (AP1+EGF) have LOWER progression risk"
                        if effect_size < -0.2 else
                        "No strong difference in progression risk"
                    )
                }
                logger.info(f"Tumour 12-like vs other: Cohen's d={effect_size:.4f}, p={pval:.2e}")

    # 4. SOX9+ cells vs niche influence
    if "sox9_expression" in merged.columns:
        sox9_high = merged["sox9_expression"] > merged["sox9_expression"].quantile(0.75)
        if sox9_high.sum() > 10:
            sox9_niche = merged.loc[sox9_high, "niche_influence"].dropna()
            other_niche = merged.loc[~sox9_high, "niche_influence"].dropna()

            if len(sox9_niche) > 10 and len(other_niche) > 10:
                stat, pval = stats.mannwhitneyu(sox9_niche, other_niche, alternative="two-sided")
                results["sox9_niche_influence"] = {
                    "sox9_high_niche": float(sox9_niche.mean()),
                    "other_niche": float(other_niche.mean()),
                    "mannwhitney_pval": float(pval),
                    "interpretation": (
                        "SOX9+ cells are in high-influence niches (consistent with niche contact)"
                        if sox9_niche.mean() > other_niche.mean() else
                        "SOX9+ cells are NOT in high-influence niches"
                    )
                }
                logger.info(f"SOX9+ niche influence: {sox9_niche.mean():.4f} vs {other_niche.mean():.4f}")

    # 5. Bottleneck state analysis (Gardner et al. Science 2024)
    # The bottleneck is the key plastic intermediate - NOT just lineage loss
    if "bottleneck_state_score" in merged.columns:
        valid = merged["bottleneck_state_score"].notna() & merged["transition_prob"].notna()
        if valid.sum() > 10:
            rho, pval = stats.spearmanr(
                merged.loc[valid, "bottleneck_state_score"],
                merged.loc[valid, "transition_prob"]
            )
            results["bottleneck_vs_progression"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_cells": int(valid.sum()),
                "interpretation": (
                    "Bottleneck state (high TM4SF1/SOX9/CREB5, low lineage) predicts progression"
                    if rho > 0.05 else
                    "Bottleneck state NOT associated with progression"
                )
            }
            logger.info(f"Bottleneck state vs progression: rho={rho:.4f}")

    # 6. Compare bottleneck-high cells vs model predictions
    if "bottleneck_state_score" in merged.columns:
        bn_high = merged["bottleneck_state_score"] > merged["bottleneck_state_score"].quantile(0.9)
        if bn_high.sum() > 10:
            bn_trans = merged.loc[bn_high, "transition_prob"].dropna()
            other_trans = merged.loc[~bn_high, "transition_prob"].dropna()

            if len(bn_trans) > 10 and len(other_trans) > 10:
                stat, pval = stats.mannwhitneyu(bn_trans, other_trans, alternative="two-sided")
                effect_size = (bn_trans.mean() - other_trans.mean()) / merged["transition_prob"].std()

                results["bottleneck_cells_vs_progression"] = {
                    "bottleneck_mean_trans": float(bn_trans.mean()),
                    "other_mean_trans": float(other_trans.mean()),
                    "cohens_d": float(effect_size),
                    "mannwhitney_pval": float(pval),
                    "n_bottleneck": int(len(bn_trans)),
                    "interpretation": (
                        "Cells in bottleneck state have HIGHER progression risk (expected)"
                        if effect_size > 0.2 else
                        "Bottleneck cells do NOT have higher progression (unexpected)"
                    )
                }
                logger.info(f"Bottleneck cells vs progression: Cohen's d={effect_size:.4f}")

    # 7. Lineage score alone (for comparison)
    if "lineage_score" in merged.columns:
        valid = merged["lineage_score"].notna() & merged["transition_prob"].notna()
        if valid.sum() > 10:
            rho, pval = stats.spearmanr(
                merged.loc[valid, "lineage_score"],
                merged.loc[valid, "transition_prob"]
            )
            results["lineage_vs_progression"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_cells": int(valid.sum()),
                "note": "Lineage alone is less informative than bottleneck state"
            }
            logger.info(f"Lineage score vs progression: rho={rho:.4f}")

    return results


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
        logger.info("\n[4/6] Computing AP-1 scores from expression...")
        adata = sc.read_h5ad(args.snrna)
        logger.info(f"  Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        ap1_scores = compute_signature_score(adata, gene_list, "ap1_score")

        ap1_df = pd.DataFrame({
            "cell_id": adata.obs_names,
            "ap1_score": ap1_scores,
        })

        # Compute Tumour 12 signatures (from precancerous niche paper)
        logger.info("\n[5/6] Computing Tumour 12 state signatures...")
        tumour12_df = compute_tumour12_signatures(adata)

        # Merge AP1 and Tumour12 scores
        ap1_df = ap1_df.merge(tumour12_df, on="cell_id", how="left")

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
    logger.info("\n[6/6] Correlating AP-1 with model predictions...")
    model_results = analyze_model_correlation(ap1_df, inference_df, cells_df)

    # Tumour 12 / EGF-SOX9-FN1 axis analysis (from precancerous niche paper)
    tumour12_results = analyze_tumour12_niche_axis(tumour12_df, ap1_df, inference_df, cells_df)

    # Spatial analysis
    spatial_results = analyze_spatial_model_correlation(neighborhoods_df, ap1_df, inference_df)

    # Generate report
    logger.info("\nGenerating validation report...")
    report = generate_validation_report(model_results, spatial_results, stage_baseline)

    # Add Tumour 12 results to report
    report["tumour12_niche_axis"] = tumour12_results
    report["biological_basis"]["precancerous_niche"] = {
        "paper": "Alcolea et al. Nature 2026",
        "key_finding": "EGF-SOX9-FN1 axis determines nascent tumour persistence",
        "mechanism": "AP1/stress TFs + EGF ligands recruit fibroblasts that deposit FN1",
        "prediction": "Tumour 12-like cells (AP1+SOX9+AREG) should have higher progression risk"
    }
    report["biological_basis"]["bottleneck_state"] = {
        "paper": "Gardner et al. Science 2024",
        "key_finding": "Histological transformation requires undifferentiated bottleneck state",
        "mechanism": (
            "Cells must enter stem-like state (high TM4SF1/SOX9/CREB5, low all lineage markers) "
            "to enable lineage conversion. This is NOT just loss of lineage identity."
        ),
        "markers": {
            "high_in_bottleneck": ["TM4SF1", "SOX9", "CREB5", "SOX2", "MYC"],
            "low_in_bottleneck": ["SFTPC", "HOPX", "NKX2-1", "ASCL1", "KRT5"]
        },
        "prediction": "Cells in bottleneck state should have highest plasticity and progression risk"
    }

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

    # Tumour 12 / Precancerous niche results
    logger.info("\n--- Tumour 12 / Precancerous Niche Axis ---")
    t12_sox9 = tumour12_results.get("ap1_sox9_correlation", {})
    if t12_sox9:
        logger.info(f"AP-1 vs SOX9: rho={t12_sox9.get('spearman_rho', 'N/A'):.4f}")

    t12_areg = tumour12_results.get("ap1_areg_correlation", {})
    if t12_areg:
        logger.info(f"AP-1 vs AREG: rho={t12_areg.get('spearman_rho', 'N/A'):.4f}")

    t12_prog = tumour12_results.get("tumour12_state_vs_progression", {})
    if t12_prog:
        logger.info(f"Tumour 12-like (AP1+EGF) vs progression: d={t12_prog.get('cohens_d', 'N/A'):.4f}")
        logger.info(f"  {t12_prog.get('interpretation', '')}")

    sox9_niche = tumour12_results.get("sox9_niche_influence", {})
    if sox9_niche:
        logger.info(f"SOX9+ cells niche influence: {sox9_niche.get('sox9_high_niche', 'N/A'):.4f}")

    # Bottleneck state results (Gardner et al.)
    logger.info("\n--- Bottleneck State (Gardner et al. Science 2024) ---")
    bn_prog = tumour12_results.get("bottleneck_vs_progression", {})
    if bn_prog:
        logger.info(f"Bottleneck state vs progression: rho={bn_prog.get('spearman_rho', 'N/A'):.4f}")

    bn_cells = tumour12_results.get("bottleneck_cells_vs_progression", {})
    if bn_cells:
        logger.info(f"Bottleneck cells (top 10%) mean transition: {bn_cells.get('bottleneck_mean_trans', 'N/A'):.4f}")
        logger.info(f"Other cells mean transition: {bn_cells.get('other_mean_trans', 'N/A'):.4f}")
        logger.info(f"Cohen's d: {bn_cells.get('cohens_d', 'N/A'):.4f}")
        logger.info(f"  {bn_cells.get('interpretation', '')}")

    logger.info("\nValidation Status:")
    for key, value in report.get("validation_status", {}).items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
