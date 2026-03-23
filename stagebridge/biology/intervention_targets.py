"""
Intervention target prioritization and niche-level risk scoring for StageBridge.

This module provides the translational biology layer that connects model
outputs to actionable clinical insights:

1. Intervention target prioritization: Which L-R axes to block for
   maximum disruption of progression-promoting niches
2. Niche-level risk scoring: Aggregate cell-level predictions into
   spatial neighborhood risk assessments
3. Ecosystem-aware risk: Risk that accounts for local niche context,
   not just intrinsic cell state

Key novelty: The receiver-centered attention directly quantifies which
niche signals drive progression risk, enabling principled target selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
import logging

from .attention_lr_scoring import LR_PRIORS, SENDER_TYPE_CATEGORIES

log = logging.getLogger(__name__)


@dataclass
class InterventionTarget:
    """A prioritized intervention target with supporting evidence."""

    ligand: str
    receptor: str
    target_gene: str  # Which to block (ligand or receptor)
    priority_score: float
    rationale: str
    evidence: list[str]
    stage_enrichment: dict[str, float]  # Stage -> fold enrichment
    druggability: str  # "approved", "clinical", "preclinical", "undrugged"
    safety_considerations: list[str]
    expected_effect: str


@dataclass
class NicheRiskScore:
    """Risk score for a spatial niche/neighborhood."""

    niche_id: str
    center_cell_id: str
    n_cells: int
    intrinsic_risk: float  # Receiver-only risk
    niche_risk: float  # Niche-context-augmented risk
    niche_contribution: float  # How much niche adds to risk
    dominant_risk_pathway: str
    dominant_sender_type: str
    il1b_axis_active: bool
    risk_category: str  # "low", "intermediate", "high", "very_high"
    spatial_coords: tuple[float, float] | None = None
    confidence: str = "medium"


@dataclass
class InterventionPlan:
    """Comprehensive intervention plan for a patient/sample."""

    sample_id: str
    stage: str
    overall_risk: str
    primary_target: InterventionTarget | None
    secondary_targets: list[InterventionTarget]
    high_risk_niches: list[NicheRiskScore]
    clinical_recommendation: str
    monitoring_strategy: str
    caveats: list[str]


# Known druggability status for L-R targets
DRUGGABILITY_DATABASE = {
    # Approved drugs
    "IL1B": {"status": "approved", "drugs": ["Canakinumab", "Anakinra (IL1R blocker)"]},
    "IL1R1": {"status": "approved", "drugs": ["Anakinra"]},
    "IL6": {"status": "approved", "drugs": ["Tocilizumab", "Siltuximab"]},
    "IL6ST": {"status": "approved", "drugs": ["Tocilizumab (blocks IL6R)"]},
    "TNF": {"status": "approved", "drugs": ["Infliximab", "Adalimumab", "Etanercept"]},
    "TNFRSF1A": {"status": "approved", "drugs": ["Etanercept"]},
    "EGFR": {"status": "approved", "drugs": ["Erlotinib", "Gefitinib", "Osimertinib"]},
    "MET": {"status": "approved", "drugs": ["Capmatinib", "Tepotinib"]},
    "VEGFA": {"status": "approved", "drugs": ["Bevacizumab"]},
    "KDR": {"status": "approved", "drugs": ["Ramucirumab"]},
    "TGFB1": {"status": "clinical", "drugs": ["Fresolimumab (clinical trials)"]},
    "TGFBR2": {"status": "clinical", "drugs": ["Galunisertib (clinical trials)"]},
    # Clinical trials
    "CXCR4": {"status": "clinical", "drugs": ["Plerixafor", "BL-8040"]},
    "CXCL12": {"status": "clinical", "drugs": ["NOX-A12"]},
    "NOTCH1": {"status": "clinical", "drugs": ["Gamma-secretase inhibitors"]},
    # Preclinical
    "AREG": {"status": "preclinical", "drugs": []},
    "HGF": {"status": "preclinical", "drugs": []},
    "SPP1": {"status": "preclinical", "drugs": []},
    "FN1": {"status": "undrugged", "drugs": []},
    "COL1A1": {"status": "undrugged", "drugs": []},
}


def prioritize_intervention_targets(
    stage_scores: dict[str, pd.DataFrame],
    stage_specific_df: pd.DataFrame | None = None,
    target_stages: list[str] | None = None,
    min_score: float = 0.1,
    top_n: int = 10,
) -> list[InterventionTarget]:
    """
    Prioritize L-R axes for therapeutic intervention.

    Priority is based on:
    1. Enrichment in early/targetable stages (AAH, AIS, MIA)
    2. Attention weight (model-derived importance)
    3. Druggability of the target
    4. Biological rationale from literature

    Parameters
    ----------
    stage_scores : dict
        Stage -> aggregated L-R scores
    stage_specific_df : DataFrame, optional
        Stage-specific enrichments
    target_stages : list, optional
        Stages to prioritize (default: AAH, AIS, MIA)
    min_score : float
        Minimum interaction score to consider
    top_n : int
        Number of targets to return

    Returns
    -------
    list[InterventionTarget]
        Prioritized intervention targets
    """
    if target_stages is None:
        target_stages = ["AAH", "AIS", "MIA"]

    candidates = []

    # Get enriched interactions in target stages
    if stage_specific_df is not None and not stage_specific_df.empty:
        enriched = stage_specific_df[
            (stage_specific_df["stage"].isin(target_stages))
            & (stage_specific_df["stage_score"] >= min_score)
        ]

        for _, row in enriched.iterrows():
            ligand, receptor = row["ligand"], row["receptor"]

            # Determine which to target
            ligand_drug = DRUGGABILITY_DATABASE.get(ligand, {})
            receptor_drug = DRUGGABILITY_DATABASE.get(receptor, {})

            # Prefer more druggable target
            if _druggability_rank(ligand_drug.get("status")) >= _druggability_rank(
                receptor_drug.get("status")
            ):
                target_gene = ligand
                drug_info = ligand_drug
            else:
                target_gene = receptor
                drug_info = receptor_drug

            # Compute priority score
            priority = _compute_priority_score(
                row["stage_score"],
                row["fold_change"],
                row["mean_attention"],
                drug_info.get("status", "undrugged"),
                ligand,
                receptor,
            )

            # Build evidence list
            evidence = [
                f"Enriched {row['fold_change']:.1f}x in {row['stage']}",
                f"Mean attention weight: {row['mean_attention']:.3f}",
                f"Interaction score: {row['stage_score']:.3f}",
            ]

            if (ligand, receptor) in [("IL1B", "IL1R1"), ("IL6", "IL6ST"), ("TNF", "TNFRSF1A")]:
                evidence.append("Core inflammatory axis (Peng et al. mechanism)")

            # Stage enrichment
            stage_enrichment = {row["stage"]: row["fold_change"]}

            # Rationale
            mechanism = row.get(
                "mechanism", LR_PRIORS.get((ligand, receptor), {}).get("mechanism", "")
            )
            rationale = _build_rationale(ligand, receptor, row["stage"], mechanism)

            # Safety considerations
            safety = _get_safety_considerations(target_gene, drug_info)

            # Expected effect
            expected = _predict_intervention_effect(ligand, receptor, row["stage"])

            candidates.append(
                InterventionTarget(
                    ligand=ligand,
                    receptor=receptor,
                    target_gene=target_gene,
                    priority_score=priority,
                    rationale=rationale,
                    evidence=evidence,
                    stage_enrichment=stage_enrichment,
                    druggability=drug_info.get("status", "undrugged"),
                    safety_considerations=safety,
                    expected_effect=expected,
                )
            )

    # Sort by priority
    candidates.sort(key=lambda x: x.priority_score, reverse=True)

    # Deduplicate by L-R pair
    seen = set()
    unique_targets = []
    for target in candidates:
        pair = (target.ligand, target.receptor)
        if pair not in seen:
            seen.add(pair)
            unique_targets.append(target)
            if len(unique_targets) >= top_n:
                break

    return unique_targets


def _druggability_rank(status: str | None) -> int:
    """Rank druggability status (higher = better)."""
    ranks = {"approved": 4, "clinical": 3, "preclinical": 2, "undrugged": 1}
    return ranks.get(status or "undrugged", 0)


def _compute_priority_score(
    interaction_score: float,
    fold_change: float,
    attention: float,
    druggability: str,
    ligand: str,
    receptor: str,
) -> float:
    """Compute composite priority score."""
    # Base score from model
    base = interaction_score * fold_change * (1 + attention)

    # Druggability multiplier
    drug_mult = {
        "approved": 2.0,
        "clinical": 1.5,
        "preclinical": 1.0,
        "undrugged": 0.5,
    }.get(druggability, 0.5)

    # Biological importance multiplier (literature-based)
    bio_mult = 1.0
    if (ligand, receptor) == ("IL1B", "IL1R1"):
        bio_mult = 1.5  # Core Peng et al. mechanism
    elif ligand in ["IL6", "TNF"]:
        bio_mult = 1.3  # Key inflammatory cytokines

    return base * drug_mult * bio_mult


def _build_rationale(ligand: str, receptor: str, stage: str, mechanism: str) -> str:
    """Build human-readable rationale for intervention."""
    parts = []

    # Stage-specific rationale
    stage_rationale = {
        "AAH": "earliest intervention opportunity before commitment",
        "AIS": "non-invasive stage, high intervention benefit",
        "MIA": "critical window before full invasion",
    }
    parts.append(
        f"Target {ligand}-{receptor} in {stage}: {stage_rationale.get(stage, 'therapeutic opportunity')}"
    )

    # Mechanism
    if mechanism:
        parts.append(f"Mechanism: {mechanism}")

    # IL1B-specific
    if ligand == "IL1B":
        parts.append(
            "IL1B+ macrophage niche is key driver of early progression (Peng et al. 2020). "
            "Canakinumab (anti-IL1B) showed reduced cancer incidence in CANTOS trial."
        )

    return " ".join(parts)


def _get_safety_considerations(target_gene: str, drug_info: dict) -> list[str]:
    """Get safety considerations for targeting a gene."""
    considerations = []

    # General immunosuppression warning
    if target_gene in ["IL1B", "IL1R1", "IL6", "IL6ST", "TNF", "TNFRSF1A"]:
        considerations.append("Risk of immunosuppression and infection")

    # Specific warnings
    if target_gene == "IL1B":
        considerations.append("Monitor for opportunistic infections (CANTOS data)")

    if target_gene == "TGFB1":
        considerations.append("TGF-beta has context-dependent tumor suppressor roles")

    if target_gene == "EGFR":
        considerations.append("Skin toxicity, diarrhea common with EGFR inhibitors")

    # Add existing drug information
    if drug_info.get("drugs"):
        considerations.append(f"Existing drugs: {', '.join(drug_info['drugs'])}")

    if not considerations:
        considerations.append("Limited clinical experience with this target")

    return considerations


def _predict_intervention_effect(ligand: str, receptor: str, stage: str) -> str:
    """Predict expected effect of intervention."""
    effects = []

    # IL1B axis
    if ligand == "IL1B":
        effects.append("Disruption of IL1B+ macrophage-epithelial crosstalk")
        effects.append("Reduced inflammatory niche signaling")
        effects.append("Potential reduction in progression to invasive disease")

    # Growth factors
    elif ligand in ["AREG", "EREG", "HBEGF", "EGF"]:
        effects.append("Reduced EGFR-driven proliferation")

    # TGF-beta
    elif ligand in ["TGFB1", "TGFB3"]:
        effects.append("Reduced EMT and stromal remodeling")

    # Chemokines
    elif ligand in ["CXCL12", "CXCL9", "CXCL10"]:
        effects.append("Altered immune cell trafficking")

    if not effects:
        effects.append(f"Disruption of {ligand}-{receptor} signaling in {stage}")

    return "; ".join(effects)


def compute_niche_level_risk(
    cell_risks: pd.DataFrame,
    spatial_coords: np.ndarray,
    attention_weights: np.ndarray | None = None,
    lr_scores_per_cell: list[dict] | None = None,
    niche_radius: float = 100.0,
    min_cells: int = 3,
) -> list[NicheRiskScore]:
    """
    Compute niche-level risk scores by aggregating cell risks spatially.

    Key insight: A cell's risk depends not just on its intrinsic state but
    on its local niche context. High-risk cells in low-risk niches may
    have different trajectories than those in high-risk niches.

    Parameters
    ----------
    cell_risks : DataFrame
        Cell-level risk scores with columns ['cell_id', 'risk_score']
    spatial_coords : ndarray
        Spatial coordinates (n_cells, 2)
    attention_weights : ndarray, optional
        Attention from focal cells to neighbors
    lr_scores_per_cell : list, optional
        Per-cell L-R interaction scores
    niche_radius : float
        Radius for niche definition (microns)
    min_cells : int
        Minimum cells for valid niche

    Returns
    -------
    list[NicheRiskScore]
        Niche-level risk assessments
    """
    from scipy.spatial import KDTree

    if len(cell_risks) == 0 or len(spatial_coords) == 0:
        return []

    # Build spatial index
    tree = KDTree(spatial_coords)

    niche_scores = []
    cell_ids = cell_risks["cell_id"].values
    risks = cell_risks["risk_score"].values

    # For each cell, define its niche
    for i, (cell_id, intrinsic_risk) in enumerate(zip(cell_ids, risks)):
        # Find neighbors
        neighbor_idx = tree.query_ball_point(spatial_coords[i], niche_radius)

        if len(neighbor_idx) < min_cells:
            continue

        neighbor_risks = risks[neighbor_idx]
        n_cells = len(neighbor_idx)

        # Compute niche-aggregated risk
        if attention_weights is not None and i < len(attention_weights):
            # Attention-weighted aggregation
            attn = attention_weights[i]
            if len(attn) >= len(neighbor_idx):
                neighbor_attn = attn[neighbor_idx]
                neighbor_attn = neighbor_attn / (neighbor_attn.sum() + 1e-10)
                niche_risk = (neighbor_attn * neighbor_risks).sum()
            else:
                niche_risk = neighbor_risks.mean()
        else:
            # Simple mean
            niche_risk = neighbor_risks.mean()

        # Niche contribution
        niche_contribution = niche_risk - intrinsic_risk

        # Determine dominant pathway and sender type
        dominant_pathway = "unknown"
        dominant_sender = "unknown"
        il1b_active = False

        if lr_scores_per_cell is not None and i < len(lr_scores_per_cell):
            cell_lr = lr_scores_per_cell[i]
            if cell_lr:
                # Find dominant family
                family_scores = {}
                for lr in cell_lr:
                    family = lr.get("family", "unknown")
                    score = lr.get("interaction_score", 0)
                    family_scores[family] = family_scores.get(family, 0) + score

                if family_scores:
                    dominant_pathway = max(family_scores.items(), key=lambda x: x[1])[0]

                # Check IL1B
                il1b_active = any(
                    lr.get("ligand") == "IL1B" and lr.get("interaction_score", 0) > 0.1
                    for lr in cell_lr
                )

        # Risk category
        combined_risk = (intrinsic_risk + niche_risk) / 2
        if combined_risk > 0.75:
            category = "very_high"
        elif combined_risk > 0.5:
            category = "high"
        elif combined_risk > 0.25:
            category = "intermediate"
        else:
            category = "low"

        # Confidence based on niche size
        if n_cells >= 10:
            confidence = "high"
        elif n_cells >= 5:
            confidence = "medium"
        else:
            confidence = "low"

        niche_scores.append(
            NicheRiskScore(
                niche_id=f"niche_{i}",
                center_cell_id=str(cell_id),
                n_cells=n_cells,
                intrinsic_risk=float(intrinsic_risk),
                niche_risk=float(niche_risk),
                niche_contribution=float(niche_contribution),
                dominant_risk_pathway=dominant_pathway,
                dominant_sender_type=dominant_sender,
                il1b_axis_active=il1b_active,
                risk_category=category,
                spatial_coords=(float(spatial_coords[i, 0]), float(spatial_coords[i, 1])),
                confidence=confidence,
            )
        )

    return niche_scores


def aggregate_niche_risks_by_region(
    niche_scores: list[NicheRiskScore],
    region_labels: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Aggregate niche risks into region-level summaries.

    Parameters
    ----------
    niche_scores : list
        Niche-level risk scores
    region_labels : ndarray, optional
        Region assignment for each niche

    Returns
    -------
    DataFrame
        Region-level risk summary
    """
    if not niche_scores:
        return pd.DataFrame()

    records = []
    for niche in niche_scores:
        records.append(
            {
                "niche_id": niche.niche_id,
                "intrinsic_risk": niche.intrinsic_risk,
                "niche_risk": niche.niche_risk,
                "niche_contribution": niche.niche_contribution,
                "risk_category": niche.risk_category,
                "il1b_active": niche.il1b_axis_active,
                "dominant_pathway": niche.dominant_risk_pathway,
                "n_cells": niche.n_cells,
            }
        )

    df = pd.DataFrame(records)

    # Aggregate statistics
    summary = {
        "n_niches": len(df),
        "mean_intrinsic_risk": df["intrinsic_risk"].mean(),
        "mean_niche_risk": df["niche_risk"].mean(),
        "mean_niche_contribution": df["niche_contribution"].mean(),
        "pct_high_risk": (df["risk_category"].isin(["high", "very_high"])).mean() * 100,
        "pct_il1b_active": df["il1b_active"].mean() * 100,
        "dominant_pathway": df["dominant_pathway"].mode().iloc[0] if len(df) > 0 else "unknown",
    }

    return df, summary


def generate_intervention_plan(
    sample_id: str,
    stage: str,
    targets: list[InterventionTarget],
    niche_scores: list[NicheRiskScore],
    overall_risk: float,
) -> InterventionPlan:
    """
    Generate comprehensive intervention plan for a sample.

    Parameters
    ----------
    sample_id : str
        Sample identifier
    stage : str
        Disease stage
    targets : list
        Prioritized intervention targets
    niche_scores : list
        Niche-level risk assessments
    overall_risk : float
        Overall risk score (0-1)

    Returns
    -------
    InterventionPlan
        Actionable intervention recommendations
    """
    # Overall risk category
    if overall_risk > 0.67:
        risk_category = "high"
    elif overall_risk > 0.33:
        risk_category = "intermediate"
    else:
        risk_category = "low"

    # Primary target
    primary = targets[0] if targets else None

    # Secondary targets (next 3)
    secondary = targets[1:4] if len(targets) > 1 else []

    # High risk niches
    high_risk_niches = [n for n in niche_scores if n.risk_category in ["high", "very_high"]][:10]

    # Clinical recommendation
    recommendation = _generate_clinical_recommendation(
        stage, risk_category, primary, high_risk_niches
    )

    # Monitoring strategy
    monitoring = _generate_monitoring_strategy(stage, risk_category, primary)

    # Caveats
    caveats = [
        "MODEL-GENERATED HYPOTHESIS - Requires clinical validation",
        "Based on computational analysis of gene expression patterns",
        "Individual patient factors not considered",
        "Drug interactions and contraindications not evaluated",
    ]

    return InterventionPlan(
        sample_id=sample_id,
        stage=stage,
        overall_risk=risk_category,
        primary_target=primary,
        secondary_targets=secondary,
        high_risk_niches=high_risk_niches,
        clinical_recommendation=recommendation,
        monitoring_strategy=monitoring,
        caveats=caveats,
    )


def _generate_clinical_recommendation(
    stage: str,
    risk: str,
    primary_target: InterventionTarget | None,
    high_risk_niches: list[NicheRiskScore],
) -> str:
    """Generate clinical recommendation text."""
    parts = []

    # Stage-specific baseline
    stage_rec = {
        "AAH": "Standard surveillance with enhanced imaging",
        "AIS": "Consider surgical resection if localized",
        "MIA": "Surgical resection recommended",
        "LUAD": "Standard-of-care treatment per guidelines",
    }
    parts.append(stage_rec.get(stage, "Clinical evaluation recommended"))

    # Risk-stratified modification
    if risk == "high":
        parts.append("High-risk niche patterns detected - consider accelerated timeline")

    # Target-specific
    if primary_target:
        if primary_target.ligand == "IL1B" and primary_target.druggability == "approved":
            parts.append(
                f"IL1B axis active - potential role for anti-inflammatory "
                f"intervention ({', '.join(DRUGGABILITY_DATABASE.get('IL1B', {}).get('drugs', []))})"
            )
        elif primary_target.druggability in ["approved", "clinical"]:
            parts.append(
                f"Consider {primary_target.target_gene} blockade "
                f"(druggability: {primary_target.druggability})"
            )

    # Niche-based
    il1b_niche_count = sum(1 for n in high_risk_niches if n.il1b_axis_active)
    if il1b_niche_count > len(high_risk_niches) * 0.5:
        parts.append(
            f"{il1b_niche_count}/{len(high_risk_niches)} high-risk niches show "
            "IL1B+ macrophage signature - inflammatory microenvironment predominates"
        )

    return ". ".join(parts)


def _generate_monitoring_strategy(
    stage: str,
    risk: str,
    primary_target: InterventionTarget | None,
) -> str:
    """Generate monitoring strategy text."""
    parts = []

    # Baseline monitoring
    if risk == "high":
        parts.append("CT imaging every 3 months")
    elif risk == "intermediate":
        parts.append("CT imaging every 6 months")
    else:
        parts.append("CT imaging every 12 months per guidelines")

    # Biomarkers
    parts.append("Monitor serum inflammatory markers (CRP, IL-6)")

    # Target-specific
    if primary_target and primary_target.ligand == "IL1B":
        parts.append("Consider IL-1beta levels if anti-inflammatory intervention")

    return "; ".join(parts)


def export_intervention_report(
    plan: InterventionPlan,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Export intervention plan as structured report.

    Parameters
    ----------
    plan : InterventionPlan
        Generated intervention plan
    output_path : str, optional
        Path to save JSON report

    Returns
    -------
    dict
        Structured report
    """
    report = {
        "sample_id": plan.sample_id,
        "stage": plan.stage,
        "overall_risk": plan.overall_risk,
        "clinical_recommendation": plan.clinical_recommendation,
        "monitoring_strategy": plan.monitoring_strategy,
        "caveats": plan.caveats,
        "primary_target": None,
        "secondary_targets": [],
        "high_risk_niches": [],
    }

    if plan.primary_target:
        report["primary_target"] = {
            "pair": f"{plan.primary_target.ligand}-{plan.primary_target.receptor}",
            "target_gene": plan.primary_target.target_gene,
            "priority_score": plan.primary_target.priority_score,
            "druggability": plan.primary_target.druggability,
            "rationale": plan.primary_target.rationale,
            "expected_effect": plan.primary_target.expected_effect,
            "safety": plan.primary_target.safety_considerations,
        }

    for target in plan.secondary_targets:
        report["secondary_targets"].append(
            {
                "pair": f"{target.ligand}-{target.receptor}",
                "target_gene": target.target_gene,
                "priority_score": target.priority_score,
                "druggability": target.druggability,
            }
        )

    for niche in plan.high_risk_niches:
        report["high_risk_niches"].append(
            {
                "niche_id": niche.niche_id,
                "risk_category": niche.risk_category,
                "il1b_active": niche.il1b_axis_active,
                "dominant_pathway": niche.dominant_risk_pathway,
                "coords": niche.spatial_coords,
            }
        )

    if output_path:
        import json

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Exported intervention report to {output_path}")

    return report


__all__ = [
    "DRUGGABILITY_DATABASE",
    "InterventionTarget",
    "NicheRiskScore",
    "InterventionPlan",
    "prioritize_intervention_targets",
    "compute_niche_level_risk",
    "aggregate_niche_risks_by_region",
    "generate_intervention_plan",
    "export_intervention_report",
]
