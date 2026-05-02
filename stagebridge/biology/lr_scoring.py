"""Attention-weighted ligand-receptor scoring for StageBridge.

Core novelty: Weight L-R interactions by model attention to identify
which cell-cell communication axes the model finds most informative
for stage progression prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LRPair:
    """A ligand-receptor pair with metadata."""
    ligand: str
    receptor: str
    family: str
    source_celltype: str | None = None
    target_celltype: str | None = None
    literature_support: float = 0.0
    mechanism: str = ""

    @property
    def name(self) -> str:
        return f"{self.ligand}_{self.receptor}"


@dataclass
class LRScoreResult:
    """Result of attention-weighted L-R scoring."""
    pair: LRPair
    raw_score: float
    attention_weight: float
    weighted_score: float
    stage: str | None = None
    donor_id: str | None = None
    n_cells: int = 0
    confidence: float = 0.0


LR_PAIRS_DATABASE: tuple[LRPair, ...] = (
    # IL1 family - primary target (Peng et al.)
    LRPair("IL1B", "IL1R1", "interleukin", literature_support=1.0,
           mechanism="Pro-inflammatory, macrophage-epithelial"),
    LRPair("IL1B", "IL1R2", "interleukin", literature_support=0.8,
           mechanism="Decoy receptor, inflammation modulation"),
    LRPair("IL1A", "IL1R1", "interleukin", literature_support=0.7,
           mechanism="Pro-inflammatory signaling"),

    # IL6 family - CAF-mediated
    LRPair("IL6", "IL6R", "interleukin", literature_support=0.9,
           mechanism="iCAF signaling, STAT3 activation"),
    LRPair("IL6", "IL6ST", "interleukin", literature_support=0.8,
           mechanism="gp130 co-receptor"),
    LRPair("LIF", "LIFR", "interleukin", literature_support=0.6,
           mechanism="Stem cell maintenance"),

    # CXCL family - immune recruitment
    LRPair("CXCL12", "CXCR4", "chemokine", literature_support=0.9,
           mechanism="iCAF chemotaxis, metastasis"),
    LRPair("CXCL12", "CXCR7", "chemokine", literature_support=0.7,
           mechanism="Alternative CXCL12 receptor"),
    LRPair("CXCL8", "CXCR1", "chemokine", literature_support=0.6,
           mechanism="Neutrophil recruitment"),
    LRPair("CXCL8", "CXCR2", "chemokine", literature_support=0.6,
           mechanism="Neutrophil recruitment"),

    # EGF family - epithelial proliferation
    LRPair("EGF", "EGFR", "growth_factor", literature_support=0.9,
           mechanism="Epithelial proliferation, KAC expansion"),
    LRPair("AREG", "EGFR", "growth_factor", literature_support=0.8,
           mechanism="Amphiregulin, wound healing"),
    LRPair("EREG", "EGFR", "growth_factor", literature_support=0.7,
           mechanism="Epiregulin signaling"),
    LRPair("HBEGF", "EGFR", "growth_factor", literature_support=0.7,
           mechanism="HB-EGF, angiogenesis"),

    # TGF-beta family - EMT, myCAF
    LRPair("TGFB1", "TGFBR1", "tgfb", literature_support=0.9,
           mechanism="EMT induction, myCAF differentiation"),
    LRPair("TGFB1", "TGFBR2", "tgfb", literature_support=0.9,
           mechanism="TGF-beta signaling"),
    LRPair("TGFB2", "TGFBR1", "tgfb", literature_support=0.6,
           mechanism="Alternative TGF-beta"),

    # WNT family - stemness, plasticity
    LRPair("WNT5A", "FZD5", "wnt", literature_support=0.7,
           mechanism="Non-canonical WNT, EMT"),
    LRPair("WNT3A", "FZD1", "wnt", literature_support=0.6,
           mechanism="Canonical WNT, stemness"),

    # Notch - cell fate
    LRPair("JAG1", "NOTCH1", "notch", literature_support=0.8,
           mechanism="Notch signaling, lineage plasticity"),
    LRPair("DLL4", "NOTCH1", "notch", literature_support=0.7,
           mechanism="Angiogenesis, tip cell selection"),

    # HGF/MET - invasiveness
    LRPair("HGF", "MET", "growth_factor", literature_support=0.8,
           mechanism="Scatter factor, invasion"),

    # FGF family
    LRPair("FGF2", "FGFR1", "growth_factor", literature_support=0.7,
           mechanism="Fibroblast growth, angiogenesis"),

    # Immune checkpoints
    LRPair("CD274", "PDCD1", "checkpoint", literature_support=0.9,
           mechanism="PD-L1/PD-1 immune evasion"),
)


def compute_attention_weighted_lr_scores(
    attention_weights: np.ndarray,
    cell_types: Sequence[str],
    expression_matrix: np.ndarray,
    gene_names: Sequence[str],
    lr_pairs: Sequence[LRPair] | None = None,
    min_expression: float = 0.1,
) -> list[LRScoreResult]:
    """Compute attention-weighted L-R interaction scores.

    The key insight: attention weights from the niche encoder tell us
    which neighbor cells the model finds most informative. By weighting
    L-R scores by attention, we identify communication axes that matter
    for stage prediction.

    Args:
        attention_weights: [N_cells, K_neighbors] attention from niche encoder
        cell_types: Cell type labels for each cell
        expression_matrix: [N_cells, N_genes] expression values
        gene_names: Gene names corresponding to columns
        lr_pairs: L-R pairs to score (default: all in database)
        min_expression: Minimum expression threshold

    Returns:
        List of LRScoreResult with attention-weighted scores
    """
    if lr_pairs is None:
        lr_pairs = LR_PAIRS_DATABASE

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    results = []

    for pair in lr_pairs:
        if pair.ligand not in gene_to_idx or pair.receptor not in gene_to_idx:
            continue

        lig_idx = gene_to_idx[pair.ligand]
        rec_idx = gene_to_idx[pair.receptor]

        lig_expr = expression_matrix[:, lig_idx]
        rec_expr = expression_matrix[:, rec_idx]

        lig_mask = lig_expr > min_expression
        rec_mask = rec_expr > min_expression

        if not (lig_mask.any() and rec_mask.any()):
            continue

        raw_score = float(np.sqrt(lig_expr.mean() * rec_expr.mean()))

        mean_attention = float(attention_weights.mean())

        lig_attention = attention_weights[lig_mask].mean() if lig_mask.any() else 0.0
        rec_attention = attention_weights[rec_mask].mean() if rec_mask.any() else 0.0
        attention_weight = float((lig_attention + rec_attention) / 2)

        weighted_score = raw_score * (1 + attention_weight)

        n_interacting = int((lig_mask & rec_mask).sum())
        confidence = min(1.0, n_interacting / 100) * pair.literature_support

        results.append(LRScoreResult(
            pair=pair,
            raw_score=raw_score,
            attention_weight=attention_weight,
            weighted_score=weighted_score,
            n_cells=n_interacting,
            confidence=confidence,
        ))

    return sorted(results, key=lambda x: x.weighted_score, reverse=True)


def aggregate_lr_scores_by_stage(
    scores_by_sample: dict[str, list[LRScoreResult]],
    stages: dict[str, str],
) -> pd.DataFrame:
    """Aggregate L-R scores by disease stage.

    Args:
        scores_by_sample: Sample ID -> list of LRScoreResult
        stages: Sample ID -> stage label

    Returns:
        DataFrame with mean scores per L-R pair per stage
    """
    records = []
    for sample_id, scores in scores_by_sample.items():
        stage = stages.get(sample_id, "unknown")
        for score in scores:
            records.append({
                "sample_id": sample_id,
                "stage": stage,
                "lr_pair": score.pair.name,
                "ligand": score.pair.ligand,
                "receptor": score.pair.receptor,
                "family": score.pair.family,
                "raw_score": score.raw_score,
                "attention_weight": score.attention_weight,
                "weighted_score": score.weighted_score,
                "n_cells": score.n_cells,
                "confidence": score.confidence,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    agg = df.groupby(["stage", "lr_pair"]).agg({
        "raw_score": "mean",
        "attention_weight": "mean",
        "weighted_score": "mean",
        "n_cells": "sum",
        "confidence": "mean",
        "ligand": "first",
        "receptor": "first",
        "family": "first",
    }).reset_index()

    return agg


def identify_stage_specific_interactions(
    aggregated_scores: pd.DataFrame,
    fold_change_threshold: float = 1.5,
    min_confidence: float = 0.3,
) -> pd.DataFrame:
    """Identify L-R interactions enriched in specific stages.

    Compares weighted scores across stages to find stage-specific axes.

    Args:
        aggregated_scores: Output from aggregate_lr_scores_by_stage
        fold_change_threshold: Minimum fold-change for enrichment
        min_confidence: Minimum confidence score

    Returns:
        DataFrame with stage-enriched L-R pairs and fold changes
    """
    if aggregated_scores.empty:
        return pd.DataFrame()

    stages = aggregated_scores["stage"].unique()
    if len(stages) < 2:
        return pd.DataFrame()

    results = []
    for lr_pair in aggregated_scores["lr_pair"].unique():
        pair_data = aggregated_scores[aggregated_scores["lr_pair"] == lr_pair]

        if pair_data["confidence"].mean() < min_confidence:
            continue

        for stage in stages:
            stage_score = pair_data[pair_data["stage"] == stage]["weighted_score"]
            other_scores = pair_data[pair_data["stage"] != stage]["weighted_score"]

            if stage_score.empty or other_scores.empty:
                continue

            stage_mean = stage_score.mean()
            other_mean = other_scores.mean()

            if other_mean > 0:
                fold_change = stage_mean / other_mean
            else:
                fold_change = float("inf") if stage_mean > 0 else 1.0

            if fold_change >= fold_change_threshold:
                row = pair_data[pair_data["stage"] == stage].iloc[0]
                results.append({
                    "lr_pair": lr_pair,
                    "enriched_stage": stage,
                    "fold_change": fold_change,
                    "stage_score": stage_mean,
                    "background_score": other_mean,
                    "ligand": row["ligand"],
                    "receptor": row["receptor"],
                    "family": row["family"],
                    "confidence": row["confidence"],
                })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values("fold_change", ascending=False)
