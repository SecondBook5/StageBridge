"""
Attention-weighted ligand-receptor interaction scoring for StageBridge.

This module provides the core biological interpretation layer that links
model attention to interpretable L-R communication patterns. Key novelty:

1. Attention weights from the receiver-centered transformer directly
   quantify which sender signals influence receiver state predictions
2. L-R interaction scores are computed by weighting prior L-R strengths
   by attention, enabling identification of stage-specific communication
3. IL1B-IL1R1 axis (Peng et al. mechanism) receives special treatment

Reference: Peng et al. 2020 Cancer Cell - IL1B+ macrophage niches
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


# L-R priors with biological support scores (from communication_builder.py)
LR_PRIORS = {
    # Inflammatory axis - primary mechanism from Peng et al.
    ("IL1B", "IL1R1"): {"family": "inflammatory", "support": 1.00, "mechanism": "IL1B+ macrophage niche signaling"},
    ("IL6", "IL6ST"): {"family": "inflammatory", "support": 0.95, "mechanism": "Inflammatory cytokine signaling"},
    ("TNF", "TNFRSF1A"): {"family": "inflammatory", "support": 0.92, "mechanism": "TNF-mediated inflammation"},
    ("OSM", "OSMR"): {"family": "inflammatory", "support": 0.80, "mechanism": "Oncostatin M signaling"},

    # Chemokine axis - migration and invasion
    ("CXCL9", "CXCR3"): {"family": "chemokine", "support": 0.85, "mechanism": "T cell recruitment"},
    ("CXCL10", "CXCR3"): {"family": "chemokine", "support": 0.85, "mechanism": "T cell recruitment"},
    ("CXCL12", "CXCR4"): {"family": "chemokine", "support": 1.00, "mechanism": "Stem cell homing/metastasis"},
    ("CXCL1", "CXCR2"): {"family": "chemokine", "support": 0.72, "mechanism": "Neutrophil recruitment"},
    ("CXCL2", "CXCR2"): {"family": "chemokine", "support": 0.72, "mechanism": "Neutrophil recruitment"},

    # TGF-beta axis - EMT and fibrosis
    ("TGFB1", "TGFBR2"): {"family": "tgfb", "support": 1.00, "mechanism": "EMT induction, fibrosis"},
    ("TGFB3", "TGFBR2"): {"family": "tgfb", "support": 0.80, "mechanism": "EMT regulation"},

    # Growth factor axis - proliferation
    ("AREG", "EGFR"): {"family": "growth_factor", "support": 0.95, "mechanism": "EGFR-driven proliferation"},
    ("EREG", "EGFR"): {"family": "growth_factor", "support": 0.85, "mechanism": "EGFR ligand"},
    ("HBEGF", "EGFR"): {"family": "growth_factor", "support": 0.80, "mechanism": "EGFR ligand"},
    ("EGF", "EGFR"): {"family": "growth_factor", "support": 0.65, "mechanism": "EGFR ligand"},
    ("HGF", "MET"): {"family": "growth_factor", "support": 0.90, "mechanism": "MET-driven invasion"},

    # Notch axis - cell fate decisions
    ("JAG1", "NOTCH1"): {"family": "notch", "support": 0.85, "mechanism": "Notch-mediated fate"},
    ("DLL4", "NOTCH1"): {"family": "notch", "support": 0.80, "mechanism": "Notch signaling"},

    # ECM axis - invasion and remodeling
    ("SPP1", "ITGAV"): {"family": "ecm", "support": 0.78, "mechanism": "Osteopontin-integrin"},
    ("COL1A1", "ITGB1"): {"family": "ecm", "support": 0.75, "mechanism": "Collagen-integrin"},
    ("FN1", "ITGB1"): {"family": "ecm", "support": 0.82, "mechanism": "Fibronectin-integrin"},

    # Other
    ("VEGFA", "KDR"): {"family": "vascular", "support": 0.78, "mechanism": "Angiogenesis"},
    ("MIF", "CD74"): {"family": "immune_modulatory", "support": 0.75, "mechanism": "Macrophage inhibition"},
    ("WNT5A", "FZD7"): {"family": "developmental", "support": 0.68, "mechanism": "Non-canonical WNT"},
}

# Sender cell type categories for interpretation
SENDER_TYPE_CATEGORIES = {
    "macrophage": ["Macrophage", "Monocyte", "DC", "cDC1", "cDC2", "pDC"],
    "fibroblast": ["Fibroblast", "myCAF", "iCAF", "CAF"],
    "t_cell": ["T_cell", "CD4", "CD8", "Treg", "NK"],
    "epithelial": ["AT2", "AT1", "Club", "Basal", "Ciliated", "Secretory"],
    "endothelial": ["Endothelial", "LEC", "BEC"],
    "other": ["B_cell", "Plasma", "Mast", "Neutrophil"],
}


@dataclass
class LRInteractionScore:
    """Score for a single L-R interaction weighted by attention."""

    ligand: str
    receptor: str
    family: str
    mechanism: str
    prior_support: float
    attention_weight: float  # Mean attention to senders expressing ligand
    ligand_expression: float  # Mean ligand expression in attended senders
    receptor_expression: float  # Receptor expression in receiver
    interaction_score: float  # Final weighted score
    sender_type_breakdown: dict[str, float] = field(default_factory=dict)
    stage: str | None = None
    confidence: str = "medium"


@dataclass
class NicheEcosystemSummary:
    """Stage-specific niche ecosystem characterization."""

    stage: str
    n_cells: int
    dominant_lr_interactions: list[LRInteractionScore]
    dominant_sender_types: dict[str, float]
    pathway_activity: dict[str, float]
    risk_level: str
    biological_interpretation: str
    key_findings: list[str]


def compute_attention_weighted_lr_scores(
    attention_weights: np.ndarray,
    sender_types: np.ndarray,
    ligand_expression: pd.DataFrame,
    receptor_expression: pd.Series,
    type_names: list[str] | None = None,
) -> list[LRInteractionScore]:
    """
    Compute attention-weighted L-R interaction scores.

    This is the core novelty: we use attention from the receiver-centered
    transformer to weight L-R interactions, directly linking model
    predictions to biological communication patterns.

    Parameters
    ----------
    attention_weights : ndarray
        Attention from receiver to senders, shape (n_senders,) or (n_heads, n_senders)
    sender_types : ndarray
        Cell type indices for each sender
    ligand_expression : DataFrame
        Ligand expression per sender (n_senders, n_ligands)
    receptor_expression : Series
        Receptor expression in the receiver cell
    type_names : list, optional
        Names for sender type indices

    Returns
    -------
    list[LRInteractionScore]
        Scored L-R interactions ranked by attention-weighted score
    """
    # Average across heads if multi-head attention
    if attention_weights.ndim == 2:
        attn = attention_weights.mean(axis=0)
    else:
        attn = attention_weights

    # Normalize attention to sum to 1
    attn = attn / (attn.sum() + 1e-10)

    scores = []

    for (ligand, receptor), info in LR_PRIORS.items():
        # Get ligand expression from senders
        if ligand not in ligand_expression.columns:
            continue
        ligand_expr = ligand_expression[ligand].values

        # Get receptor expression in receiver
        receptor_expr = receptor_expression.get(receptor, 0.0)
        if receptor_expr <= 0:
            continue

        # Attention-weighted ligand expression
        weighted_ligand = (attn * ligand_expr).sum()

        # Compute interaction score: attention × ligand × receptor × prior
        interaction_score = (
            weighted_ligand *
            receptor_expr *
            info["support"]
        )

        # Breakdown by sender type
        sender_breakdown = {}
        if type_names is not None:
            for type_idx, type_name in enumerate(type_names):
                type_mask = sender_types == type_idx
                if type_mask.sum() > 0:
                    type_contribution = (attn[type_mask] * ligand_expr[type_mask]).sum()
                    if type_contribution > 0.01:
                        sender_breakdown[type_name] = float(type_contribution)

        # Assess confidence based on expression levels and attention
        mean_attn_to_expressors = attn[ligand_expr > 0].mean() if (ligand_expr > 0).any() else 0
        if weighted_ligand > 0.5 and receptor_expr > 0.5 and mean_attn_to_expressors > 0.1:
            confidence = "high"
        elif weighted_ligand > 0.1 and receptor_expr > 0.1:
            confidence = "medium"
        else:
            confidence = "low"

        scores.append(LRInteractionScore(
            ligand=ligand,
            receptor=receptor,
            family=info["family"],
            mechanism=info["mechanism"],
            prior_support=info["support"],
            attention_weight=float(mean_attn_to_expressors),
            ligand_expression=float(weighted_ligand),
            receptor_expression=float(receptor_expr),
            interaction_score=float(interaction_score),
            sender_type_breakdown=sender_breakdown,
            confidence=confidence,
        ))

    # Sort by interaction score
    scores.sort(key=lambda x: x.interaction_score, reverse=True)

    return scores


def aggregate_lr_scores_by_stage(
    all_scores: list[tuple[str, list[LRInteractionScore]]],
) -> dict[str, pd.DataFrame]:
    """
    Aggregate L-R interaction scores across cells within each stage.

    Parameters
    ----------
    all_scores : list of (stage, scores) tuples
        Per-cell L-R scores with stage labels

    Returns
    -------
    dict
        Stage -> DataFrame of aggregated L-R statistics
    """
    stage_data: dict[str, list[dict]] = {}

    for stage, cell_scores in all_scores:
        if stage not in stage_data:
            stage_data[stage] = []

        for score in cell_scores:
            stage_data[stage].append({
                "ligand": score.ligand,
                "receptor": score.receptor,
                "family": score.family,
                "interaction_score": score.interaction_score,
                "attention_weight": score.attention_weight,
                "confidence": score.confidence,
            })

    results = {}
    for stage, records in stage_data.items():
        if not records:
            continue

        df = pd.DataFrame(records)

        # Aggregate per L-R pair
        agg = df.groupby(["ligand", "receptor", "family"]).agg({
            "interaction_score": ["mean", "std", "count"],
            "attention_weight": "mean",
        }).reset_index()

        agg.columns = [
            "ligand", "receptor", "family",
            "mean_score", "std_score", "n_cells", "mean_attention"
        ]

        # Sort by mean score
        agg = agg.sort_values("mean_score", ascending=False)

        results[stage] = agg

    return results


def identify_stage_specific_interactions(
    stage_scores: dict[str, pd.DataFrame],
    min_fold_change: float = 1.5,
    min_cells: int = 10,
) -> pd.DataFrame:
    """
    Identify L-R interactions that are significantly enriched in specific stages.

    Parameters
    ----------
    stage_scores : dict
        Stage -> aggregated L-R scores from aggregate_lr_scores_by_stage
    min_fold_change : float
        Minimum fold enrichment vs other stages
    min_cells : int
        Minimum cells required

    Returns
    -------
    DataFrame
        Stage-specific L-R interactions with enrichment statistics
    """
    from scipy import stats

    # Combine all stages
    all_dfs = []
    for stage, df in stage_scores.items():
        df = df.copy()
        df["stage"] = stage
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    results = []
    stages = list(stage_scores.keys())

    for (ligand, receptor), group in combined.groupby(["ligand", "receptor"]):
        for stage in stages:
            stage_data = group[group["stage"] == stage]
            other_data = group[group["stage"] != stage]

            if stage_data.empty or stage_data["n_cells"].values[0] < min_cells:
                continue

            stage_score = stage_data["mean_score"].values[0]

            if other_data.empty:
                other_score = 0.01
            else:
                # Weighted mean by cell count
                other_score = np.average(
                    other_data["mean_score"].values,
                    weights=other_data["n_cells"].values
                )

            fold_change = stage_score / (other_score + 1e-10)

            if fold_change >= min_fold_change:
                # Get family info
                family = stage_data["family"].values[0]
                mechanism = LR_PRIORS.get((ligand, receptor), {}).get("mechanism", "Unknown")

                results.append({
                    "stage": stage,
                    "ligand": ligand,
                    "receptor": receptor,
                    "family": family,
                    "mechanism": mechanism,
                    "stage_score": stage_score,
                    "other_score": other_score,
                    "fold_change": fold_change,
                    "n_cells": stage_data["n_cells"].values[0],
                    "mean_attention": stage_data["mean_attention"].values[0],
                })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(
        ["stage", "fold_change"], ascending=[True, False]
    )


def compute_il1b_axis_score(
    lr_scores: list[LRInteractionScore],
) -> dict[str, Any]:
    """
    Compute focused score for the IL1B-IL1R1 axis (Peng et al. mechanism).

    The IL1B+ macrophage niche is the key biological mechanism we're testing.
    This function provides a focused assessment of this axis.

    Parameters
    ----------
    lr_scores : list
        L-R interaction scores for a cell

    Returns
    -------
    dict
        IL1B axis assessment with biological interpretation
    """
    # Find IL1B-IL1R1 interaction
    il1b_score = None
    for score in lr_scores:
        if score.ligand == "IL1B" and score.receptor == "IL1R1":
            il1b_score = score
            break

    if il1b_score is None:
        return {
            "detected": False,
            "score": 0.0,
            "interpretation": "IL1B-IL1R1 axis not detected in this cell",
            "confidence": "low",
        }

    # Check for macrophage contribution
    macrophage_contribution = 0.0
    for sender_type, contrib in il1b_score.sender_type_breakdown.items():
        for mac_type in SENDER_TYPE_CATEGORIES["macrophage"]:
            if mac_type.lower() in sender_type.lower():
                macrophage_contribution += contrib

    # Interpretation based on Peng et al.
    interpretation_parts = []

    if il1b_score.interaction_score > 0.5:
        interpretation_parts.append(
            "Strong IL1B-IL1R1 signaling detected, consistent with "
            "Peng et al. IL1B+ macrophage niche"
        )
    elif il1b_score.interaction_score > 0.1:
        interpretation_parts.append(
            "Moderate IL1B-IL1R1 signaling present"
        )
    else:
        interpretation_parts.append(
            "Weak IL1B-IL1R1 signaling"
        )

    if macrophage_contribution > 0.3:
        interpretation_parts.append(
            f"Macrophages contribute {macrophage_contribution:.1%} of IL1B signal - "
            "strong macrophage niche signature"
        )

    # Related inflammatory signals
    related_inflammatory = []
    for score in lr_scores:
        if score.family == "inflammatory" and score.ligand != "IL1B":
            if score.interaction_score > 0.1:
                related_inflammatory.append(f"{score.ligand}-{score.receptor}")

    if related_inflammatory:
        interpretation_parts.append(
            f"Co-occurring inflammatory signals: {', '.join(related_inflammatory[:3])}"
        )

    return {
        "detected": True,
        "score": il1b_score.interaction_score,
        "attention_to_il1b_senders": il1b_score.attention_weight,
        "macrophage_contribution": macrophage_contribution,
        "receiver_il1r1_expression": il1b_score.receptor_expression,
        "confidence": il1b_score.confidence,
        "interpretation": "; ".join(interpretation_parts),
        "related_inflammatory_signals": related_inflammatory,
    }


def generate_niche_ecosystem_summary(
    stage_scores: dict[str, pd.DataFrame],
    pathway_activity: dict[str, pd.DataFrame] | None = None,
    risk_scores: dict[str, float] | None = None,
) -> dict[str, NicheEcosystemSummary]:
    """
    Generate comprehensive niche ecosystem summaries per stage.

    Parameters
    ----------
    stage_scores : dict
        Stage -> L-R score DataFrame
    pathway_activity : dict, optional
        Stage -> pathway activity DataFrame
    risk_scores : dict, optional
        Stage -> mean risk score

    Returns
    -------
    dict
        Stage -> NicheEcosystemSummary
    """
    stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    summaries = {}

    for stage in stage_order:
        if stage not in stage_scores:
            continue

        scores_df = stage_scores[stage]
        n_cells = int(scores_df["n_cells"].sum() / len(scores_df)) if len(scores_df) > 0 else 0

        # Top L-R interactions
        top_lr = []
        for _, row in scores_df.head(5).iterrows():
            info = LR_PRIORS.get((row["ligand"], row["receptor"]), {})
            top_lr.append(LRInteractionScore(
                ligand=row["ligand"],
                receptor=row["receptor"],
                family=row["family"],
                mechanism=info.get("mechanism", "Unknown"),
                prior_support=info.get("support", 0.5),
                attention_weight=row["mean_attention"],
                ligand_expression=0.0,  # Not available in aggregated
                receptor_expression=0.0,
                interaction_score=row["mean_score"],
                stage=stage,
            ))

        # Dominant sender types from family breakdown
        family_counts = scores_df.groupby("family")["mean_score"].sum().to_dict()

        # Get pathway activity if available
        pathway_dict = {}
        if pathway_activity is not None and stage in pathway_activity:
            pathway_dict = pathway_activity[stage].to_dict()

        # Risk level
        if risk_scores is not None and stage in risk_scores:
            risk = risk_scores[stage]
            risk_level = "high" if risk > 0.67 else ("medium" if risk > 0.33 else "low")
        else:
            # Estimate from L-R patterns
            inflammatory_score = family_counts.get("inflammatory", 0)
            growth_score = family_counts.get("growth_factor", 0)
            if inflammatory_score > 0.5 or growth_score > 0.5:
                risk_level = "high"
            elif inflammatory_score > 0.2 or growth_score > 0.2:
                risk_level = "medium"
            else:
                risk_level = "low"

        # Generate biological interpretation
        interpretation = _generate_stage_interpretation(stage, top_lr, family_counts)

        # Key findings
        key_findings = _extract_key_findings(stage, top_lr, family_counts)

        summaries[stage] = NicheEcosystemSummary(
            stage=stage,
            n_cells=n_cells,
            dominant_lr_interactions=top_lr,
            dominant_sender_types=family_counts,
            pathway_activity=pathway_dict,
            risk_level=risk_level,
            biological_interpretation=interpretation,
            key_findings=key_findings,
        )

    return summaries


def _generate_stage_interpretation(
    stage: str,
    top_lr: list[LRInteractionScore],
    family_counts: dict[str, float],
) -> str:
    """Generate human-readable interpretation for a stage."""
    parts = []

    # Stage-specific context
    stage_context = {
        "Normal": "healthy alveolar tissue",
        "AAH": "atypical adenomatous hyperplasia (earliest precursor)",
        "AIS": "adenocarcinoma in situ (non-invasive)",
        "MIA": "minimally invasive adenocarcinoma",
        "LUAD": "invasive lung adenocarcinoma",
    }

    parts.append(f"In {stage_context.get(stage, stage)}:")

    # Dominant communication patterns
    if top_lr:
        dominant = top_lr[0]
        parts.append(
            f"Dominant communication: {dominant.ligand}-{dominant.receptor} "
            f"({dominant.mechanism})"
        )

    # Family patterns
    sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_families:
        top_families = [f"{f[0]} ({f[1]:.2f})" for f in sorted_families[:3]]
        parts.append(f"Active signaling families: {', '.join(top_families)}")

    # IL1B-specific check
    il1b_present = any(lr.ligand == "IL1B" for lr in top_lr)
    if il1b_present:
        parts.append(
            "IL1B-IL1R1 axis active - consistent with Peng et al. "
            "proinflammatory niche mechanism"
        )

    return " ".join(parts)


def _extract_key_findings(
    stage: str,
    top_lr: list[LRInteractionScore],
    family_counts: dict[str, float],
) -> list[str]:
    """Extract bullet-point key findings for a stage."""
    findings = []

    # Check for IL1B
    for lr in top_lr:
        if lr.ligand == "IL1B":
            findings.append(
                f"IL1B-IL1R1 signaling detected (score={lr.interaction_score:.2f}) - "
                "Peng et al. mechanism"
            )
            break

    # Dominant family
    if family_counts:
        top_family = max(family_counts.items(), key=lambda x: x[1])
        if top_family[1] > 0.1:
            findings.append(f"Dominant signaling family: {top_family[0]}")

    # Stage-specific patterns from literature
    stage_patterns = {
        "AAH": [
            "Earliest detectable precursor stage",
            "Expected: emerging inflammatory niche",
        ],
        "AIS": [
            "Non-invasive but committed to progression",
            "Expected: peak IL1B+ macrophage infiltration",
        ],
        "MIA": [
            "Minimally invasive - critical intervention window",
            "Expected: transition from inflammatory to growth-dominant",
        ],
        "LUAD": [
            "Fully invasive carcinoma",
            "Expected: growth factor and ECM remodeling dominant",
        ],
    }

    findings.extend(stage_patterns.get(stage, []))

    return findings


def create_lr_interaction_report(
    stage_summaries: dict[str, NicheEcosystemSummary],
    stage_specific_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Create comprehensive L-R interaction report for publication.

    Parameters
    ----------
    stage_summaries : dict
        Stage -> NicheEcosystemSummary
    stage_specific_df : DataFrame, optional
        Stage-specific interactions

    Returns
    -------
    dict
        Structured report suitable for paper figures/tables
    """
    report = {
        "title": "StageBridge Attention-Weighted L-R Interaction Analysis",
        "stages": {},
        "key_mechanism": {
            "name": "IL1B-IL1R1 Axis",
            "reference": "Peng et al. 2020 Cancer Cell",
            "hypothesis": (
                "IL1B+ macrophage niches drive early LUAD progression. "
                "These niches should be most prominent in AAH/AIS stages."
            ),
        },
        "stage_progression": [],
        "intervention_targets": [],
    }

    # Per-stage summary
    stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]

    for stage in stage_order:
        if stage not in stage_summaries:
            continue

        summary = stage_summaries[stage]

        stage_report = {
            "n_cells": summary.n_cells,
            "risk_level": summary.risk_level,
            "interpretation": summary.biological_interpretation,
            "key_findings": summary.key_findings,
            "top_interactions": [
                {
                    "pair": f"{lr.ligand}-{lr.receptor}",
                    "family": lr.family,
                    "score": lr.interaction_score,
                    "mechanism": lr.mechanism,
                }
                for lr in summary.dominant_lr_interactions[:5]
            ],
            "family_activity": summary.dominant_sender_types,
        }

        report["stages"][stage] = stage_report

    # Stage progression narrative
    if "AAH" in stage_summaries and "AIS" in stage_summaries:
        aah = stage_summaries["AAH"]
        ais = stage_summaries["AIS"]

        # Check IL1B trend
        aah_il1b = next(
            (lr.interaction_score for lr in aah.dominant_lr_interactions
             if lr.ligand == "IL1B"), 0.0
        )
        ais_il1b = next(
            (lr.interaction_score for lr in ais.dominant_lr_interactions
             if lr.ligand == "IL1B"), 0.0
        )

        if ais_il1b > aah_il1b:
            report["stage_progression"].append({
                "transition": "AAH -> AIS",
                "observation": f"IL1B signaling increases ({aah_il1b:.2f} -> {ais_il1b:.2f})",
                "interpretation": "Consistent with Peng et al. proinflammatory niche expansion",
            })

    # Identify intervention targets
    if stage_specific_df is not None and not stage_specific_df.empty:
        # Interactions enriched in early stages (AAH/AIS) are intervention targets
        early_enriched = stage_specific_df[
            stage_specific_df["stage"].isin(["AAH", "AIS"])
        ].head(10)

        for _, row in early_enriched.iterrows():
            report["intervention_targets"].append({
                "target": f"{row['ligand']}-{row['receptor']}",
                "stage": row["stage"],
                "fold_enrichment": row["fold_change"],
                "mechanism": row["mechanism"],
                "rationale": (
                    f"Enriched {row['fold_change']:.1f}x in {row['stage']} vs other stages. "
                    "Blocking may disrupt progression-promoting niche."
                ),
            })

    return report


def export_lr_scores_for_visualization(
    stage_scores: dict[str, pd.DataFrame],
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Export L-R scores in format suitable for visualization.

    Parameters
    ----------
    stage_scores : dict
        Stage -> L-R score DataFrame
    output_path : str, optional
        Path to save CSV

    Returns
    -------
    DataFrame
        Combined scores for all stages
    """
    dfs = []
    for stage, df in stage_scores.items():
        df = df.copy()
        df["stage"] = stage
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Add L-R pair column for plotting
    combined["lr_pair"] = combined["ligand"] + "-" + combined["receptor"]

    if output_path:
        combined.to_csv(output_path, index=False)
        log.info(f"Exported L-R scores to {output_path}")

    return combined


__all__ = [
    "LR_PRIORS",
    "SENDER_TYPE_CATEGORIES",
    "LRInteractionScore",
    "NicheEcosystemSummary",
    "compute_attention_weighted_lr_scores",
    "aggregate_lr_scores_by_stage",
    "identify_stage_specific_interactions",
    "compute_il1b_axis_score",
    "generate_niche_ecosystem_summary",
    "create_lr_interaction_report",
    "export_lr_scores_for_visualization",
]
