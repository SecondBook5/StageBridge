#!/usr/bin/env python3
"""
Senescence/SASP Signature Validation for StageBridge.

Validates whether cellular senescence and SASP correlate with model-predicted
progression risk. This tests whether the model's predictions align with
senescence biology.

Key biological insights:

1. From Hoi et al. (Cancer Cell 2026):
   - SASP (senescence-associated secretory phenotype) creates feedforward loops
   - Senescent fibroblasts secrete IL6, GDF15, matrix factors
   - SASP activates ERK/p38/AKT in epithelial cells
   - cGAS-STING pathway drives chronic inflammatory cytokine production

2. From Alcolea et al. (Nature 2026) - Precancerous Niche Paper:
   - Fibroblasts in the precancerous niche are activated (aYAP+)
   - They deposit fibronectin (FN1) creating supportive scaffold
   - Niche+ fibroblasts lack full CAF phenotype (no FAP, no ACTA2)
   - But they show pre-CAF transitional state
   - SASP-secreting cells may contribute to niche formation

3. From Tsankov et al. (Nature Cancer 2025) - TP53 Atlas:
   - CAF.ADH1B and CAF.COLs populations enriched in TP53mut tumors
   - NMF7 niche contains myofibroblasts with SASP-like features
   - Stromal remodeling creates hypoxic, EMT-promoting environment

The validation tests:
1. Do cells with high SASP have higher MODEL-PREDICTED transition probability?
2. Is chronic senescence enriched in cells the model flags as high-risk?
3. Do high niche-influence regions show elevated SASP?
4. Are SASP-secreting fibroblasts near progression-prone epithelial cells?
5. Does FN1+ stroma (precancerous niche marker) correlate with SASP?

Usage:
  python scripts/run_senescence_validation.py \
      --snrna /path/to/snrna_with_celltypes.h5ad \
      --cells /path/to/cells.parquet \
      --neighborhoods /path/to/neighborhoods.parquet \
      --inference-outputs /path/to/inference_outputs.parquet \
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

# Precancerous niche fibroblast markers (from Alcolea et al. 2026)
# Niche+ fibroblasts: PDGFRa-low, FN1+, aYAP+, but FAP-/ACTA2- (pre-CAF state)
PRECANCER_NICHE_MATRIX = ["FN1", "COL1A1", "COL1A2", "COL3A1", "COL12A1"]
PRECANCER_NICHE_ACTIVATION = ["YAP1", "WWTR1", "CTGF", "CYR61"]  # Hippo pathway
CAF_MARKERS = ["FAP", "ACTA2", "PDPN", "S100A4"]  # Full CAF phenotype

# TP53 atlas CAF subtypes (from Tsankov et al. 2025)
# CAF.ADH1B and CAF.COLs enriched in TP53mut, correlate with poor outcome
CAF_ADH1B_MARKERS = ["ADH1B", "CFD", "APOE"]  # Cytokine/immune modulation
CAF_COLS_MARKERS = ["COL1A2", "COL3A1", "TWIST1"]  # ECM/EMT promoting

# NMF7 niche components (hypoxic, EMT-promoting)
NMF7_NICHE = ["SPP1", "COL1A1", "VIM", "HIF1A"]

# Stages for analysis
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
        "--ap1-scores",
        type=Path,
        default=None,
        help="Path to AP-1 cell scores (for cross-validation)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for senescence validation results"
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


def compute_niche_fibroblast_signatures(
    adata: sc.AnnData,
) -> pd.DataFrame:
    """
    Compute fibroblast niche signatures from precancerous niche paper.

    The precancerous niche is characterized by:
    1. FN1+ (fibronectin) matrix deposition
    2. aYAP+ (active Hippo pathway) fibroblasts
    3. Lack of full CAF markers (FAP-, ACTA2-) - pre-CAF state
    4. PDGFRa-low (lamina propria origin)
    """
    logger.info("Computing fibroblast niche signatures...")

    results = {"cell_id": adata.obs_names.tolist()}

    # FN1/ECM deposition score (key niche marker)
    ecm_genes = [g for g in PRECANCER_NICHE_MATRIX if g in adata.var_names]
    if len(ecm_genes) >= 1:
        sc.tl.score_genes(adata, ecm_genes, score_name="niche_ecm_score")
        results["niche_ecm_score"] = adata.obs["niche_ecm_score"].values
        logger.info(f"  Niche ECM score: {len(ecm_genes)} genes")

    # FN1 expression directly
    if "FN1" in adata.var_names:
        if hasattr(adata.X, "toarray"):
            fn1_idx = list(adata.var_names).index("FN1")
            results["fn1_expression"] = adata.X[:, fn1_idx].toarray().flatten()
        else:
            fn1_idx = list(adata.var_names).index("FN1")
            results["fn1_expression"] = adata.X[:, fn1_idx].flatten()
        logger.info("  FN1 expression extracted")

    # Hippo/YAP activation score
    yap_genes = [g for g in PRECANCER_NICHE_ACTIVATION if g in adata.var_names]
    if len(yap_genes) >= 1:
        sc.tl.score_genes(adata, yap_genes, score_name="yap_activation_score")
        results["yap_activation_score"] = adata.obs["yap_activation_score"].values
        logger.info(f"  YAP activation score: {len(yap_genes)} genes")

    # CAF marker score (full CAF phenotype - should be LOW in precancer niche)
    caf_genes = [g for g in CAF_MARKERS if g in adata.var_names]
    if len(caf_genes) >= 1:
        sc.tl.score_genes(adata, caf_genes, score_name="caf_score")
        results["caf_score"] = adata.obs["caf_score"].values
        logger.info(f"  CAF score: {len(caf_genes)} genes")

    # NMF7 niche score (from TP53 atlas - hypoxic/EMT promoting)
    nmf7_genes = [g for g in NMF7_NICHE if g in adata.var_names]
    if len(nmf7_genes) >= 1:
        sc.tl.score_genes(adata, nmf7_genes, score_name="nmf7_score")
        results["nmf7_score"] = adata.obs["nmf7_score"].values
        logger.info(f"  NMF7 niche score: {len(nmf7_genes)} genes")

    return pd.DataFrame(results)


def analyze_precancer_niche_sasp(
    niche_df: pd.DataFrame,
    senescence_scores: pd.DataFrame,
    inference_df: pd.DataFrame,
    cells_df: pd.DataFrame,
) -> dict:
    """
    Analyze relationship between SASP and precancerous niche formation.

    Key questions:
    1. Do SASP-high cells show elevated FN1/ECM signatures?
    2. Is FN1+ stroma associated with higher model-predicted progression?
    3. Do pre-CAF (FN1+/YAP+/FAP-) cells correlate with progression?
    """
    logger.info("=" * 60)
    logger.info("PRECANCEROUS NICHE / SASP ANALYSIS")
    logger.info("=" * 60)

    # Merge data
    merged = senescence_scores.merge(niche_df, on="cell_id", how="inner")
    merged = merged.merge(
        inference_df[["cell_id", "transition_prob", "niche_influence"]],
        on="cell_id",
        how="inner"
    )
    logger.info(f"Merged {len(merged):,} cells for niche analysis")

    results = {}

    # 1. SASP vs FN1/ECM correlation
    if "niche_ecm_score" in merged.columns:
        valid = merged["sasp_score"].notna() & merged["niche_ecm_score"].notna()
        if valid.sum() > 10:
            rho, pval = stats.spearmanr(
                merged.loc[valid, "sasp_score"],
                merged.loc[valid, "niche_ecm_score"]
            )
            results["sasp_ecm_correlation"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_cells": int(valid.sum()),
                "interpretation": (
                    "SASP and ECM/FN1 deposition are correlated (niche formation)"
                    if rho > 0.1 else "SASP and ECM not strongly correlated"
                )
            }
            logger.info(f"SASP vs ECM: rho={rho:.4f}, p={pval:.2e}")

    # 2. FN1+ cells vs model-predicted progression
    if "fn1_expression" in merged.columns:
        fn1_high = merged["fn1_expression"] > merged["fn1_expression"].quantile(0.75)
        if fn1_high.sum() > 10:
            fn1_trans = merged.loc[fn1_high, "transition_prob"].dropna()
            other_trans = merged.loc[~fn1_high, "transition_prob"].dropna()

            if len(fn1_trans) > 10 and len(other_trans) > 10:
                stat, pval = stats.mannwhitneyu(fn1_trans, other_trans, alternative="two-sided")
                effect_size = (fn1_trans.mean() - other_trans.mean()) / merged["transition_prob"].std()

                results["fn1_progression"] = {
                    "fn1_high_trans": float(fn1_trans.mean()),
                    "fn1_low_trans": float(other_trans.mean()),
                    "cohens_d": float(effect_size),
                    "mannwhitney_pval": float(pval),
                    "interpretation": (
                        "FN1+ cells/neighborhoods have HIGHER progression risk"
                        if effect_size > 0.1 else
                        "FN1+ cells/neighborhoods have LOWER progression risk"
                        if effect_size < -0.1 else
                        "No strong FN1-progression relationship"
                    )
                }
                logger.info(f"FN1+ vs progression: Cohen's d={effect_size:.4f}")

    # 3. Pre-CAF state analysis (FN1+/YAP+ but FAP-)
    if all(col in merged.columns for col in ["niche_ecm_score", "yap_activation_score", "caf_score"]):
        ecm_high = merged["niche_ecm_score"] > merged["niche_ecm_score"].quantile(0.75)
        yap_high = merged["yap_activation_score"] > merged["yap_activation_score"].quantile(0.75)
        caf_low = merged["caf_score"] < merged["caf_score"].quantile(0.5)

        pre_caf = ecm_high & yap_high & caf_low
        full_caf = ecm_high & yap_high & ~caf_low

        if pre_caf.sum() > 10:
            pre_caf_trans = merged.loc[pre_caf, "transition_prob"].dropna()
            full_caf_trans = merged.loc[full_caf, "transition_prob"].dropna() if full_caf.sum() > 10 else pd.Series()
            other_trans = merged.loc[~(pre_caf | full_caf), "transition_prob"].dropna()

            results["pre_caf_state"] = {
                "n_pre_caf": int(pre_caf.sum()),
                "n_full_caf": int(full_caf.sum()),
                "pre_caf_mean_trans": float(pre_caf_trans.mean()) if len(pre_caf_trans) > 0 else None,
                "full_caf_mean_trans": float(full_caf_trans.mean()) if len(full_caf_trans) > 0 else None,
                "other_mean_trans": float(other_trans.mean()) if len(other_trans) > 0 else None,
            }
            logger.info(f"Pre-CAF cells: {pre_caf.sum()}, Full CAF: {full_caf.sum()}")

    # 4. NMF7 niche (TP53 atlas) vs progression
    if "nmf7_score" in merged.columns:
        valid = merged["nmf7_score"].notna() & merged["transition_prob"].notna()
        if valid.sum() > 10:
            rho, pval = stats.spearmanr(
                merged.loc[valid, "nmf7_score"],
                merged.loc[valid, "transition_prob"]
            )
            results["nmf7_progression"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_cells": int(valid.sum()),
                "interpretation": (
                    "NMF7 niche (hypoxic/EMT) associated with progression"
                    if rho > 0.05 else "NMF7 niche NOT associated with progression"
                )
            }
            logger.info(f"NMF7 vs progression: rho={rho:.4f}")

    return results


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

    sen_high = np.nanpercentile(senescence_score, 75)
    sasp_high = np.nanpercentile(sasp_score, 75)
    prolif_high = np.nanpercentile(proliferation_score, 50)

    proliferating = proliferation_score > prolif_high
    sen_pos = senescence_score > sen_high
    sasp_pos = sasp_score > sasp_high

    states[proliferating] = "proliferating"
    states[~proliferating & ~sen_pos] = "quiescent"
    states[sen_pos & ~sasp_pos] = "early_senescent"
    states[sen_pos & sasp_pos] = "chronic_senescent"

    return states


def analyze_model_correlation(
    senescence_scores: pd.DataFrame,
    inference_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    ap1_scores_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Correlate senescence signatures with MODEL outputs.

    The key question: do cells the model predicts as high-risk have
    elevated SASP / chronic senescence?
    """
    logger.info("=" * 60)
    logger.info("SENESCENCE vs MODEL PREDICTIONS")
    logger.info("=" * 60)

    # Merge senescence scores with model outputs
    merged = senescence_scores.merge(
        inference_df[["cell_id", "transition_prob", "niche_influence"]],
        on="cell_id",
        how="inner"
    )
    logger.info(f"Merged {len(merged):,} cells with model outputs")

    if len(merged) < 100:
        logger.error("Too few cells for analysis")
        return {"error": "insufficient_cells"}

    results = {}

    # Core: SASP vs transition probability
    valid = merged["sasp_score"].notna() & merged["transition_prob"].notna()
    if valid.sum() > 10:
        rho, pval = stats.spearmanr(
            merged.loc[valid, "sasp_score"],
            merged.loc[valid, "transition_prob"]
        )
        results["sasp_transition_prob"] = {
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "n_cells": int(valid.sum()),
            "interpretation": (
                "POSITIVE: High SASP cells have higher model-predicted progression"
                if rho > 0.05 else
                "NEGATIVE: High SASP cells have lower model-predicted progression"
                if rho < -0.05 else
                "WEAK: No strong relationship"
            )
        }
        logger.info(f"SASP vs transition_prob: rho={rho:.4f}, p={pval:.2e}")

    # Senescence core vs transition probability
    valid = merged["senescence_score"].notna() & merged["transition_prob"].notna()
    if valid.sum() > 10:
        rho, pval = stats.spearmanr(
            merged.loc[valid, "senescence_score"],
            merged.loc[valid, "transition_prob"]
        )
        results["senescence_transition_prob"] = {
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "n_cells": int(valid.sum()),
        }
        logger.info(f"Senescence vs transition_prob: rho={rho:.4f}, p={pval:.2e}")

    # SASP vs niche influence
    valid = merged["sasp_score"].notna() & merged["niche_influence"].notna()
    if valid.sum() > 10:
        rho, pval = stats.spearmanr(
            merged.loc[valid, "sasp_score"],
            merged.loc[valid, "niche_influence"]
        )
        results["sasp_niche_influence"] = {
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "n_cells": int(valid.sum()),
        }
        logger.info(f"SASP vs niche_influence: rho={rho:.4f}, p={pval:.2e}")

    # Stratified analysis: senescence state in high-risk vs low-risk cells
    high_risk = merged["transition_prob"] > merged["transition_prob"].quantile(0.75)
    low_risk = merged["transition_prob"] < merged["transition_prob"].quantile(0.25)

    if high_risk.sum() > 10 and low_risk.sum() > 10:
        # SASP comparison
        sasp_high_risk = merged.loc[high_risk, "sasp_score"].dropna()
        sasp_low_risk = merged.loc[low_risk, "sasp_score"].dropna()

        if len(sasp_high_risk) > 10 and len(sasp_low_risk) > 10:
            stat, pval = stats.mannwhitneyu(sasp_high_risk, sasp_low_risk, alternative="two-sided")
            effect_size = (sasp_high_risk.mean() - sasp_low_risk.mean()) / merged["sasp_score"].std()

            results["sasp_risk_stratification"] = {
                "high_risk_mean": float(sasp_high_risk.mean()),
                "low_risk_mean": float(sasp_low_risk.mean()),
                "cohens_d": float(effect_size),
                "mannwhitney_pval": float(pval),
            }
            logger.info(f"SASP high vs low risk: Cohen's d={effect_size:.4f}, p={pval:.2e}")

        # Chronic senescence proportion
        chronic_high = (merged.loc[high_risk, "senescence_state"] == "chronic_senescent").mean()
        chronic_low = (merged.loc[low_risk, "senescence_state"] == "chronic_senescent").mean()

        results["chronic_senescence_by_risk"] = {
            "pct_chronic_high_risk": float(chronic_high * 100),
            "pct_chronic_low_risk": float(chronic_low * 100),
            "fold_enrichment": float(chronic_high / chronic_low) if chronic_low > 0 else None,
        }
        logger.info(f"Chronic senescence: {chronic_high*100:.1f}% in high-risk vs {chronic_low*100:.1f}% in low-risk")

    # AP-1 cross-validation if available
    if ap1_scores_df is not None:
        merged_ap1 = merged.merge(
            ap1_scores_df[["cell_id", "ap1_score"]],
            on="cell_id",
            how="left"
        )
        valid = merged_ap1["sasp_score"].notna() & merged_ap1["ap1_score"].notna()
        if valid.sum() > 10:
            rho, pval = stats.spearmanr(
                merged_ap1.loc[valid, "sasp_score"],
                merged_ap1.loc[valid, "ap1_score"]
            )
            results["sasp_ap1_correlation"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_cells": int(valid.sum()),
            }
            logger.info(f"SASP-AP1 correlation: rho={rho:.4f}")

    # Cell type analysis
    if "cell_type" in cells_df.columns:
        merged = merged.merge(
            cells_df[["cell_id", "cell_type"]].drop_duplicates(),
            on="cell_id",
            how="left"
        )

        cell_type_corrs = {}
        for ct in merged["cell_type"].dropna().unique():
            ct_data = merged.loc[merged["cell_type"] == ct]
            valid = ct_data["sasp_score"].notna() & ct_data["transition_prob"].notna()
            if valid.sum() > 30:
                rho, pval = stats.spearmanr(
                    ct_data.loc[valid, "sasp_score"],
                    ct_data.loc[valid, "transition_prob"]
                )
                cell_type_corrs[ct] = {"rho": float(rho), "pval": float(pval), "n": int(valid.sum())}

        if cell_type_corrs:
            sorted_cts = sorted(cell_type_corrs.items(), key=lambda x: abs(x[1]["rho"]), reverse=True)
            results["cell_type_correlations"] = dict(sorted_cts[:10])

    return results


def analyze_spatial_model_correlation(
    neighborhoods_df: pd.DataFrame,
    senescence_scores: pd.DataFrame,
    inference_df: pd.DataFrame,
) -> dict:
    """Analyze senescence in spatial context with model predictions."""
    logger.info("=" * 60)
    logger.info("SPATIAL SENESCENCE vs MODEL PREDICTIONS")
    logger.info("=" * 60)

    merged = neighborhoods_df.merge(
        senescence_scores[["cell_id", "sasp_score", "senescence_state"]],
        on="cell_id",
        how="left"
    ).merge(
        inference_df[["cell_id", "transition_prob", "niche_influence"]],
        on="cell_id",
        how="left"
    )

    results = {"n_neighborhoods": len(merged)}

    # SASP hotspots vs model predictions
    sasp_threshold = merged["sasp_score"].quantile(0.9)
    hotspot_mask = merged["sasp_score"] > sasp_threshold

    if hotspot_mask.sum() > 10:
        hotspot_trans = merged.loc[hotspot_mask, "transition_prob"].dropna()
        nonhotspot_trans = merged.loc[~hotspot_mask, "transition_prob"].dropna()

        if len(hotspot_trans) > 10 and len(nonhotspot_trans) > 10:
            stat, pval = stats.mannwhitneyu(hotspot_trans, nonhotspot_trans, alternative="two-sided")
            results["sasp_hotspot_transition"] = {
                "hotspot_mean_trans": float(hotspot_trans.mean()),
                "nonhotspot_mean_trans": float(nonhotspot_trans.mean()),
                "mannwhitney_pval": float(pval),
                "n_hotspot": int(len(hotspot_trans)),
            }
            logger.info(
                f"SASP hotspot transition: {hotspot_trans.mean():.4f} vs "
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
        "hypothesis": "SASP/chronic senescence correlates with model-predicted progression risk",
        "biological_basis": {
            "key_finding": "Senescence is dual: early=protective, chronic+SASP=tumorigenic",
            "mechanism": "SASP creates feedforward loops via IL6, GDF15, matrix remodeling",
            "connection_to_h1_2": "SASP includes IL1B secretion",
            "connection_to_ap1": "p38 pathway converges on AP-1",
        },
        "model_validation": model_results,
        "spatial_validation": spatial_results,
        "stage_baseline": stage_baseline,
        "validation_status": {},
    }

    # Determine validation status
    sasp_trans = model_results.get("sasp_transition_prob", {})
    rho = sasp_trans.get("spearman_rho")

    if rho is not None:
        if rho > 0.1:
            report["validation_status"]["sasp_model_prediction"] = "SUPPORTED"
            report["validation_status"]["detail"] = (
                f"Cells with high SASP have higher model-predicted transition probability (rho={rho:.3f})"
            )
        elif rho < -0.1:
            report["validation_status"]["sasp_model_prediction"] = "INVERSE"
            report["validation_status"]["detail"] = (
                f"Cells with high SASP have LOWER model-predicted transition (rho={rho:.3f})"
            )
        else:
            report["validation_status"]["sasp_model_prediction"] = "WEAK"

    # Chronic senescence enrichment
    chronic = model_results.get("chronic_senescence_by_risk", {})
    if chronic.get("fold_enrichment"):
        fe = chronic["fold_enrichment"]
        if fe > 1.2:
            report["validation_status"]["chronic_enrichment"] = "HIGH_RISK_ENRICHED"
        elif fe < 0.8:
            report["validation_status"]["chronic_enrichment"] = "LOW_RISK_ENRICHED"
        else:
            report["validation_status"]["chronic_enrichment"] = "NO_DIFFERENCE"

    return report


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SENESCENCE/SASP SIGNATURE VALIDATION (MODEL-BASED)")
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

    # Load neighborhoods
    logger.info("\n[3/5] Loading spatial neighborhoods...")
    neighborhoods_df = pd.read_parquet(args.neighborhoods)
    logger.info(f"  Loaded {len(neighborhoods_df):,} neighborhoods")

    # Load AP-1 scores if available
    ap1_scores_df = None
    if args.ap1_scores and args.ap1_scores.exists():
        logger.info("  Loading AP-1 scores for cross-validation...")
        ap1_scores_df = pd.read_parquet(args.ap1_scores)

    # Compute senescence scores from expression
    if args.snrna and args.snrna.exists():
        logger.info("\n[4/6] Computing senescence scores from expression...")
        adata = sc.read_h5ad(args.snrna)
        logger.info(f"  Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        senescence_scores = compute_signature_score(adata, SENESCENCE_CORE, "senescence_score")
        sasp_scores = compute_signature_score(adata, SASP_ALL, "sasp_score")

        # Proliferation for state classification
        prolif_genes = ["MKI67", "PCNA", "TOP2A", "MCM2"]
        prolif_scores = compute_signature_score(adata, prolif_genes, "proliferation_score")

        # Classify states
        senescence_states = classify_senescence_state(senescence_scores, sasp_scores, prolif_scores)

        scores_df = pd.DataFrame({
            "cell_id": adata.obs_names,
            "senescence_score": senescence_scores,
            "sasp_score": sasp_scores,
            "senescence_state": senescence_states,
        })

        # Compute fibroblast niche signatures (from precancerous niche paper)
        logger.info("\n[5/6] Computing fibroblast niche signatures...")
        niche_df = compute_niche_fibroblast_signatures(adata)

        # Merge niche scores into main dataframe
        scores_df = scores_df.merge(niche_df, on="cell_id", how="left")

        # Save
        scores_df.to_parquet(args.output_dir / "senescence_cell_scores.parquet", index=False)
        logger.info(f"  Saved: senescence_cell_scores.parquet")

        # Baseline: correlation with stage
        if "cell_id" in cells_df.columns:
            scores_with_stage = scores_df.merge(
                cells_df[["cell_id", "stage"]].drop_duplicates(),
                on="cell_id",
                how="left"
            )
            valid = scores_with_stage["stage"].isin(STAGES) & scores_with_stage["sasp_score"].notna()
            if valid.sum() > 10:
                stage_numeric = scores_with_stage.loc[valid, "stage"].map(STAGE_ORDER)
                rho, pval = stats.spearmanr(stage_numeric, scores_with_stage.loc[valid, "sasp_score"])
                stage_baseline = {
                    "sasp_vs_stage_rho": float(rho),
                    "sasp_vs_stage_pval": float(pval),
                    "note": "Baseline correlation with raw stage (not model prediction)"
                }
            else:
                stage_baseline = {}
        else:
            stage_baseline = {}
    else:
        logger.error("snRNA h5ad required to compute senescence scores")
        return 1

    # Core validation: correlate with model outputs
    logger.info("\n[6/6] Correlating senescence with model predictions...")
    model_results = analyze_model_correlation(scores_df, inference_df, cells_df, ap1_scores_df)

    # Precancerous niche / SASP analysis (from precancerous niche paper)
    niche_results = analyze_precancer_niche_sasp(niche_df, scores_df, inference_df, cells_df)

    # Spatial analysis
    spatial_results = analyze_spatial_model_correlation(neighborhoods_df, scores_df, inference_df)

    # Generate report
    logger.info("\nGenerating validation report...")
    report = generate_validation_report(model_results, spatial_results, stage_baseline)

    # Add precancerous niche results to report
    report["precancerous_niche_analysis"] = niche_results
    report["biological_basis"]["precancerous_niche"] = {
        "paper": "Alcolea et al. Nature 2026",
        "key_finding": "FN1+ stromal scaffold promotes nascent tumour persistence",
        "mechanism": "Pre-CAF fibroblasts (FN1+/YAP+/FAP-) form supportive niche",
        "prediction": "SASP and FN1/ECM should be correlated, and FN1+ niche should associate with progression"
    }
    report["biological_basis"]["tp53_atlas"] = {
        "paper": "Tsankov et al. Nature Cancer 2025",
        "key_finding": "NMF7 niche (TAM.SPP1 + CAF.COLs + myofibroblasts) promotes EMT",
        "prediction": "NMF7 niche signature should correlate with model-predicted progression"
    }

    # Save results
    with open(args.output_dir / "senescence_validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved: senescence_validation_report.json")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SENESCENCE VALIDATION SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nBaseline (SASP vs raw stage): rho={stage_baseline.get('sasp_vs_stage_rho', 'N/A')}")

    sasp_trans = model_results.get("sasp_transition_prob", {})
    logger.info(f"SASP vs MODEL transition_prob: rho={sasp_trans.get('spearman_rho', 'N/A')}")

    sasp_niche = model_results.get("sasp_niche_influence", {})
    logger.info(f"SASP vs niche_influence: rho={sasp_niche.get('spearman_rho', 'N/A')}")

    chronic = model_results.get("chronic_senescence_by_risk", {})
    if chronic:
        logger.info(f"Chronic senescence in high-risk: {chronic.get('pct_chronic_high_risk', 'N/A'):.1f}%")
        logger.info(f"Chronic senescence in low-risk: {chronic.get('pct_chronic_low_risk', 'N/A'):.1f}%")

    # Precancerous niche results
    logger.info("\n--- Precancerous Niche / FN1 Analysis ---")
    sasp_ecm = niche_results.get("sasp_ecm_correlation", {})
    if sasp_ecm:
        logger.info(f"SASP vs ECM/FN1: rho={sasp_ecm.get('spearman_rho', 'N/A'):.4f}")

    fn1_prog = niche_results.get("fn1_progression", {})
    if fn1_prog:
        logger.info(f"FN1+ vs progression: d={fn1_prog.get('cohens_d', 'N/A'):.4f}")
        logger.info(f"  {fn1_prog.get('interpretation', '')}")

    pre_caf = niche_results.get("pre_caf_state", {})
    if pre_caf:
        logger.info(f"Pre-CAF cells (FN1+/YAP+/FAP-): {pre_caf.get('n_pre_caf', 'N/A')}")
        logger.info(f"  Mean transition prob: {pre_caf.get('pre_caf_mean_trans', 'N/A')}")

    nmf7 = niche_results.get("nmf7_progression", {})
    if nmf7:
        logger.info(f"NMF7 niche vs progression: rho={nmf7.get('spearman_rho', 'N/A'):.4f}")

    logger.info("\nValidation Status:")
    for key, value in report.get("validation_status", {}).items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
