"""Novel biological discovery from unexplained attention patterns.

This module identifies potential novel biology that the model discovered
but was NOT part of the training signal or known mechanisms.

CRITICAL: All outputs are clearly labeled as HYPOTHESES, not findings.
These require experimental validation before publication claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import numpy as np
import pandas as pd

from stagebridge.biology.lr_scoring import LR_PAIRS_DATABASE
from stagebridge.biology.known_biology import KNOWN_MECHANISMS


class HypothesisConfidence(StrEnum):
    """Confidence level for novel hypotheses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


class HypothesisType(StrEnum):
    """Types of novel discoveries."""
    NOVEL_LR_AXIS = "novel_lr_axis"
    NOVEL_CELL_STATE = "novel_cell_state"
    NOVEL_NICHE_PATTERN = "novel_niche_pattern"
    UNEXPECTED_STAGE_ASSOCIATION = "unexpected_stage_association"
    ATTENTION_HOTSPOT = "attention_hotspot"


@dataclass
class NovelHypothesis:
    """A model-generated hypothesis requiring validation.

    IMPORTANT: This is a HYPOTHESIS, not a finding. The model identified
    this pattern as informative for stage prediction, but biological
    validity requires experimental confirmation.
    """
    hypothesis_type: HypothesisType
    description: str
    confidence: HypothesisConfidence
    evidence_score: float
    stage_association: str | None = None
    genes_involved: tuple[str, ...] = ()
    attention_score: float = 0.0
    n_cells: int = 0
    suggested_validation: str = ""
    literature_support: float = 0.0
    is_novel: bool = True

    def to_dict(self) -> dict:
        return {
            "type": self.hypothesis_type.value,
            "description": self.description,
            "confidence": self.confidence.value,
            "evidence_score": self.evidence_score,
            "stage": self.stage_association,
            "genes": list(self.genes_involved),
            "attention_score": self.attention_score,
            "n_cells": self.n_cells,
            "validation_needed": self.suggested_validation,
            "literature_support": self.literature_support,
            "is_novel": self.is_novel,
            "DISCLAIMER": "HYPOTHESIS - requires experimental validation",
        }


@dataclass
class DiscoveryResult:
    """Results of novel discovery analysis."""
    hypotheses: list[NovelHypothesis]
    known_recovered: int
    novel_discovered: int
    spurious_filtered: int
    summary: str = ""


def _get_known_genes() -> set[str]:
    """Get all genes involved in known mechanisms."""
    known = set()
    for mech in KNOWN_MECHANISMS:
        known.update(mech.markers)
    for pair in LR_PAIRS_DATABASE:
        known.add(pair.ligand)
        known.add(pair.receptor)
    return known


def generate_novel_hypotheses(
    attention_weights: np.ndarray,
    expression_matrix: np.ndarray,
    gene_names: Sequence[str],
    cell_types: Sequence[str],
    stages: Sequence[str],
    attention_percentile: float = 90.0,
    min_expression: float = 0.5,
    min_cells: int = 10,
) -> DiscoveryResult:
    """Generate hypotheses from unexplained high-attention patterns.

    The model may attend to genes/cell states not in our known mechanism
    database. These are potential novel discoveries.

    Args:
        attention_weights: [N_cells, K_neighbors] attention
        expression_matrix: [N_cells, N_genes] expression
        gene_names: Gene names
        cell_types: Cell type per cell
        stages: Stage label per cell
        attention_percentile: Threshold for "high attention"
        min_expression: Minimum expression
        min_cells: Minimum cells for pattern

    Returns:
        DiscoveryResult with novel hypotheses
    """
    known_genes = _get_known_genes()
    hypotheses = []
    spurious_count = 0

    attn_threshold = np.percentile(attention_weights.mean(axis=1), attention_percentile)
    high_attn_mask = attention_weights.mean(axis=1) > attn_threshold

    if not high_attn_mask.any():
        return DiscoveryResult(
            hypotheses=[],
            known_recovered=0,
            novel_discovered=0,
            spurious_filtered=0,
            summary="No high-attention cells found",
        )

    high_attn_expr = expression_matrix[high_attn_mask]
    high_attn_stages = np.array(stages)[high_attn_mask]
    high_attn_celltypes = np.array(cell_types)[high_attn_mask]
    high_attn_scores = attention_weights[high_attn_mask].mean(axis=1)

    gene_attention_scores = {}
    for i, gene in enumerate(gene_names):
        expressed_mask = high_attn_expr[:, i] > min_expression
        if expressed_mask.sum() >= min_cells:
            gene_attention_scores[gene] = {
                "mean_attention": float(high_attn_scores[expressed_mask].mean()),
                "n_cells": int(expressed_mask.sum()),
                "mean_expression": float(high_attn_expr[expressed_mask, i].mean()),
            }

    sorted_genes = sorted(
        gene_attention_scores.items(),
        key=lambda x: x[1]["mean_attention"],
        reverse=True,
    )

    for gene, scores in sorted_genes[:50]:
        if gene in known_genes:
            continue

        expressed_mask = high_attn_expr[:, gene_names.index(gene)] > min_expression
        stage_counts = pd.Series(high_attn_stages[expressed_mask]).value_counts()

        if len(stage_counts) == 0:
            continue

        dominant_stage = stage_counts.index[0]
        stage_specificity = stage_counts.iloc[0] / stage_counts.sum()

        if stage_specificity < 0.5:
            spurious_count += 1
            continue

        if scores["mean_attention"] < attn_threshold * 0.8:
            spurious_count += 1
            continue

        confidence = _compute_confidence(
            attention_score=scores["mean_attention"],
            n_cells=scores["n_cells"],
            stage_specificity=stage_specificity,
        )

        hypotheses.append(NovelHypothesis(
            hypothesis_type=HypothesisType.ATTENTION_HOTSPOT,
            description=(
                f"Gene {gene} shows high attention in {dominant_stage} stage "
                f"({scores['n_cells']} cells, {stage_specificity:.0%} stage-specific)"
            ),
            confidence=confidence,
            evidence_score=scores["mean_attention"] * stage_specificity,
            stage_association=dominant_stage,
            genes_involved=(gene,),
            attention_score=scores["mean_attention"],
            n_cells=scores["n_cells"],
            suggested_validation=f"Validate {gene} expression/function in {dominant_stage} samples",
            is_novel=True,
        ))

    celltype_attention = {}
    for ct in set(cell_types):
        ct_mask = np.array(cell_types) == ct
        ct_high_attn = high_attn_mask & ct_mask
        if ct_high_attn.sum() >= min_cells:
            celltype_attention[ct] = {
                "mean_attention": float(attention_weights[ct_high_attn].mean()),
                "n_cells": int(ct_high_attn.sum()),
                "fraction_high_attn": float(ct_high_attn.sum() / ct_mask.sum()),
            }

    sorted_celltypes = sorted(
        celltype_attention.items(),
        key=lambda x: x[1]["fraction_high_attn"],
        reverse=True,
    )

    for ct, scores in sorted_celltypes[:10]:
        if scores["fraction_high_attn"] < 0.3:
            continue

        ct_stages = np.array(stages)[np.array(cell_types) == ct]
        stage_counts = pd.Series(ct_stages).value_counts()
        dominant_stage = stage_counts.index[0]

        confidence = _compute_confidence(
            attention_score=scores["mean_attention"],
            n_cells=scores["n_cells"],
            stage_specificity=scores["fraction_high_attn"],
        )

        hypotheses.append(NovelHypothesis(
            hypothesis_type=HypothesisType.NOVEL_CELL_STATE,
            description=(
                f"Cell type {ct} shows disproportionate attention "
                f"({scores['fraction_high_attn']:.0%} of cells in top attention), "
                f"enriched in {dominant_stage}"
            ),
            confidence=confidence,
            evidence_score=scores["mean_attention"] * scores["fraction_high_attn"],
            stage_association=dominant_stage,
            attention_score=scores["mean_attention"],
            n_cells=scores["n_cells"],
            suggested_validation=f"Investigate {ct} role in {dominant_stage} progression",
            is_novel=True,
        ))

    known_recovered = len([g for g, _ in sorted_genes[:20] if g in known_genes])

    return DiscoveryResult(
        hypotheses=hypotheses,
        known_recovered=known_recovered,
        novel_discovered=len(hypotheses),
        spurious_filtered=spurious_count,
        summary=(
            f"Generated {len(hypotheses)} novel hypotheses from high-attention patterns. "
            f"Recovered {known_recovered} known genes in top-20. "
            f"Filtered {spurious_count} spurious associations."
        ),
    )


def _compute_confidence(
    attention_score: float,
    n_cells: int,
    stage_specificity: float,
) -> HypothesisConfidence:
    """Compute confidence level based on evidence strength."""
    score = attention_score * np.log10(n_cells + 1) * stage_specificity

    if score > 1.0 and n_cells > 100:
        return HypothesisConfidence.HIGH
    elif score > 0.5 and n_cells > 50:
        return HypothesisConfidence.MEDIUM
    elif score > 0.2 and n_cells > 20:
        return HypothesisConfidence.LOW
    else:
        return HypothesisConfidence.SPECULATIVE


def rank_hypotheses_by_confidence(
    hypotheses: list[NovelHypothesis],
) -> list[NovelHypothesis]:
    """Rank hypotheses by confidence and evidence score."""
    confidence_order = {
        HypothesisConfidence.HIGH: 0,
        HypothesisConfidence.MEDIUM: 1,
        HypothesisConfidence.LOW: 2,
        HypothesisConfidence.SPECULATIVE: 3,
    }

    return sorted(
        hypotheses,
        key=lambda h: (confidence_order[h.confidence], -h.evidence_score),
    )


def filter_spurious_associations(
    hypotheses: list[NovelHypothesis],
    min_confidence: HypothesisConfidence = HypothesisConfidence.LOW,
    min_cells: int = 20,
    min_evidence: float = 0.1,
) -> list[NovelHypothesis]:
    """Filter out likely spurious hypotheses.

    Args:
        hypotheses: Raw hypotheses
        min_confidence: Minimum confidence level
        min_cells: Minimum supporting cells
        min_evidence: Minimum evidence score

    Returns:
        Filtered hypotheses meeting criteria
    """
    confidence_order = {
        HypothesisConfidence.HIGH: 0,
        HypothesisConfidence.MEDIUM: 1,
        HypothesisConfidence.LOW: 2,
        HypothesisConfidence.SPECULATIVE: 3,
    }
    min_order = confidence_order[min_confidence]

    return [
        h for h in hypotheses
        if (
            confidence_order[h.confidence] <= min_order
            and h.n_cells >= min_cells
            and h.evidence_score >= min_evidence
        )
    ]
