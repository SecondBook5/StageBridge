"""Validation against known biological mechanisms.

This module tests whether StageBridge recovers established biology:
- IL1B-IL1R1 axis (Peng et al.) in preinvasive stage
- KAC/reactive pneumocyte expansion
- CAF subtype dynamics (myCAF vs iCAF)
- EMT progression markers

A model that fails to recover known biology cannot be trusted for novel discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

import numpy as np


class MechanismType(StrEnum):
    """Categories of known biological mechanisms."""
    LIGAND_RECEPTOR = "ligand_receptor"
    CELL_STATE = "cell_state"
    NICHE_COMPOSITION = "niche_composition"
    GENE_PROGRAM = "gene_program"


class ValidationStatus(StrEnum):
    """Status of mechanism validation."""
    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    NOT_DETECTED = "not_detected"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class KnownMechanism:
    """A known biological mechanism to validate against."""
    name: str
    mechanism_type: MechanismType
    expected_stage: str
    markers: tuple[str, ...]
    literature_source: str
    description: str
    priority: int = 1

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class ValidationResult:
    """Result of validating a known mechanism."""
    mechanism: KnownMechanism
    status: ValidationStatus
    score: float
    observed_stage: str | None = None
    evidence: dict = field(default_factory=dict)
    explanation: str = ""


KNOWN_MECHANISMS: tuple[KnownMechanism, ...] = (
    KnownMechanism(
        name="IL1B_IL1R1_preinvasive",
        mechanism_type=MechanismType.LIGAND_RECEPTOR,
        expected_stage="Preinvasive",
        markers=("IL1B", "IL1R1"),
        literature_source="Peng et al. 2023, Kadara et al.",
        description="IL1B+ macrophage-epithelial axis enriched in AAH/AIS vs LUAD",
        priority=1,
    ),
    KnownMechanism(
        name="KAC_progenitor_expansion",
        mechanism_type=MechanismType.CELL_STATE,
        expected_stage="Preinvasive",
        markers=("KRT5", "KRT17", "SOX9", "TP63"),
        literature_source="Peng et al. 2023",
        description="KAC/reactive pneumocyte progenitors as LUAD predecessors",
        priority=1,
    ),
    KnownMechanism(
        name="iCAF_proinflammatory",
        mechanism_type=MechanismType.NICHE_COMPOSITION,
        expected_stage="Preinvasive",
        markers=("IL6", "CXCL12", "PDGFRA"),
        literature_source="CAF literature, Peng et al.",
        description="Inflammatory CAF enrichment in early progression",
        priority=2,
    ),
    KnownMechanism(
        name="myCAF_invasive",
        mechanism_type=MechanismType.NICHE_COMPOSITION,
        expected_stage="Invasive",
        markers=("ACTA2", "COL1A1", "COL3A1", "FAP"),
        literature_source="CAF literature",
        description="Myofibroblastic CAF enrichment in invasive disease",
        priority=2,
    ),
    KnownMechanism(
        name="EMT_progression",
        mechanism_type=MechanismType.GENE_PROGRAM,
        expected_stage="Invasive",
        markers=("VIM", "CDH2", "SNAI1", "TWIST1", "ZEB1"),
        literature_source="EMT literature",
        description="EMT marker upregulation with progression",
        priority=2,
    ),
    KnownMechanism(
        name="epithelial_loss",
        mechanism_type=MechanismType.GENE_PROGRAM,
        expected_stage="Invasive",
        markers=("CDH1", "EPCAM", "KRT8", "KRT18"),
        literature_source="EMT literature",
        description="Epithelial marker downregulation (inverse correlation)",
        priority=3,
    ),
    KnownMechanism(
        name="EGFR_signaling",
        mechanism_type=MechanismType.LIGAND_RECEPTOR,
        expected_stage="Preinvasive",
        markers=("EGF", "AREG", "EGFR"),
        literature_source="Alcolea et al. 2026",
        description="EGF-SOX9 axis in precancerous niche remodeling",
        priority=2,
    ),
    KnownMechanism(
        name="immune_evasion",
        mechanism_type=MechanismType.LIGAND_RECEPTOR,
        expected_stage="Invasive",
        markers=("CD274", "PDCD1", "CTLA4"),
        literature_source="Checkpoint literature",
        description="Immune checkpoint upregulation in invasive disease",
        priority=3,
    ),
)


def validate_known_mechanisms(
    attention_by_stage: dict[str, np.ndarray],
    expression_by_stage: dict[str, np.ndarray],
    gene_names: Sequence[str],
    mechanisms: Sequence[KnownMechanism] | None = None,
    attention_threshold: float = 0.1,
    expression_threshold: float = 0.5,
) -> list[ValidationResult]:
    """Validate whether model recovers known biological mechanisms.

    Args:
        attention_by_stage: Stage -> attention weights array
        expression_by_stage: Stage -> expression matrix
        gene_names: Gene names
        mechanisms: Mechanisms to validate (default: all)
        attention_threshold: Threshold for "high attention"
        expression_threshold: Threshold for "expressed"

    Returns:
        List of ValidationResult for each mechanism
    """
    if mechanisms is None:
        mechanisms = KNOWN_MECHANISMS

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    results = []

    for mech in mechanisms:
        marker_indices = [gene_to_idx.get(m) for m in mech.markers]
        marker_indices = [i for i in marker_indices if i is not None]

        if not marker_indices:
            results.append(ValidationResult(
                mechanism=mech,
                status=ValidationStatus.NOT_DETECTED,
                score=0.0,
                explanation=f"Markers not found in gene list: {mech.markers}",
            ))
            continue

        stage_scores = {}
        for stage, expr in expression_by_stage.items():
            marker_expr = expr[:, marker_indices].mean(axis=1)
            expressed_mask = marker_expr > expression_threshold

            if stage in attention_by_stage:
                attn = attention_by_stage[stage]
                if expressed_mask.any():
                    attn_score = attn[expressed_mask].mean()
                else:
                    attn_score = 0.0
            else:
                attn_score = 0.0

            expr_score = marker_expr.mean()
            stage_scores[stage] = {
                "attention": float(attn_score),
                "expression": float(expr_score),
                "combined": float(expr_score * (1 + attn_score)),
                "n_expressing": int(expressed_mask.sum()),
            }

        expected_score = stage_scores.get(mech.expected_stage, {}).get("combined", 0)
        other_scores = [
            s["combined"] for st, s in stage_scores.items()
            if st != mech.expected_stage
        ]
        max_other = max(other_scores) if other_scores else 0

        if expected_score > 0 and expected_score > max_other:
            if expected_score > max_other * 1.5:
                status = ValidationStatus.CONFIRMED
                score = min(1.0, expected_score / (max_other + 0.01))
            else:
                status = ValidationStatus.PARTIAL
                score = expected_score / (max_other + expected_score + 0.01)
        elif expected_score > 0:
            status = ValidationStatus.PARTIAL
            score = expected_score / (max_other + expected_score + 0.01)
        else:
            status = ValidationStatus.NOT_DETECTED
            score = 0.0

        best_stage = max(stage_scores.keys(), key=lambda s: stage_scores[s]["combined"])

        results.append(ValidationResult(
            mechanism=mech,
            status=status,
            score=score,
            observed_stage=best_stage,
            evidence=stage_scores,
            explanation=_generate_explanation(mech, status, best_stage, stage_scores),
        ))

    return sorted(results, key=lambda r: (r.mechanism.priority, -r.score))


def _generate_explanation(
    mech: KnownMechanism,
    status: ValidationStatus,
    observed_stage: str,
    stage_scores: dict,
) -> str:
    """Generate human-readable explanation of validation result."""
    expected = mech.expected_stage
    expected_score = stage_scores.get(expected, {}).get("combined", 0)
    observed_score = stage_scores.get(observed_stage, {}).get("combined", 0)

    if status == ValidationStatus.CONFIRMED:
        return (
            f"CONFIRMED: {mech.name} enriched in {expected} as expected "
            f"(score={expected_score:.2f}, {expected_score/observed_score:.1f}x vs others)"
        )
    elif status == ValidationStatus.PARTIAL:
        if observed_stage == expected:
            return (
                f"PARTIAL: {mech.name} detected in {expected} but enrichment weak "
                f"(score={expected_score:.2f})"
            )
        else:
            return (
                f"PARTIAL: {mech.name} strongest in {observed_stage}, "
                f"expected {expected} (observed={observed_score:.2f}, expected={expected_score:.2f})"
            )
    elif status == ValidationStatus.CONTRADICTED:
        return (
            f"CONTRADICTED: {mech.name} shows opposite pattern - "
            f"strongest in {observed_stage}, expected {expected}"
        )
    else:
        return f"NOT DETECTED: {mech.name} markers not expressed or not attended to"


def compute_mechanism_recovery_score(
    validation_results: list[ValidationResult],
    weight_by_priority: bool = True,
) -> dict[str, float]:
    """Compute overall mechanism recovery score.

    A model should recover most priority-1 mechanisms to be trusted.

    Args:
        validation_results: Results from validate_known_mechanisms
        weight_by_priority: Weight by mechanism priority

    Returns:
        Dict with overall score and breakdown
    """
    if not validation_results:
        return {"overall": 0.0, "by_priority": {}, "by_type": {}}

    total_weight = 0.0
    weighted_score = 0.0
    by_priority: dict[int, list[float]] = {}
    by_type: dict[str, list[float]] = {}

    for r in validation_results:
        priority = r.mechanism.priority
        mtype = r.mechanism.mechanism_type.value

        weight = 1.0 / priority if weight_by_priority else 1.0
        total_weight += weight
        weighted_score += weight * r.score

        by_priority.setdefault(priority, []).append(r.score)
        by_type.setdefault(mtype, []).append(r.score)

    return {
        "overall": weighted_score / total_weight if total_weight > 0 else 0.0,
        "by_priority": {p: np.mean(s) for p, s in by_priority.items()},
        "by_type": {t: np.mean(s) for t, s in by_type.items()},
        "n_confirmed": sum(1 for r in validation_results if r.status == ValidationStatus.CONFIRMED),
        "n_partial": sum(1 for r in validation_results if r.status == ValidationStatus.PARTIAL),
        "n_not_detected": sum(1 for r in validation_results if r.status == ValidationStatus.NOT_DETECTED),
        "n_total": len(validation_results),
    }
