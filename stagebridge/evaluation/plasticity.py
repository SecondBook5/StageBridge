"""Plasticity scoring for hypothesis H1.3: Niche conditioning reduces fate uncertainty.

This module implements plasticity analysis to test whether niche context resolves
cell fate uncertainty. Key concepts from HYPOTHESIS.md:

- Plasticity = entropy of transition probability distribution per cell
- High plasticity = cell could go either way (repair vs tumor)
- The hypothesis: niche context REDUCES plasticity (resolves fate uncertainty)
- Specifically: IL1B-high niches commit plastic cells toward tumor

KACs/RPII represent a high-plasticity intermediate state at a bifurcation:
  - Path A: Differentiate to AT1 (lung repair/resolution)
  - Path B: Progress to tumor (malignant transformation)

References:
- H1.3 in docs/HYPOTHESIS.md (lines 44-86, 155-164)
- KACs Atlas (Nature 2024) - Han et al.
- Peng/Kadara (Cell 2025) - Multimodal spatial-omics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# Default stage names for LUAD progression
DEFAULT_STAGES: tuple[str, ...] = ("Normal", "AAH", "AIS", "MIA", "LUAD")

# Repair vs tumor fate labels
FATE_LABELS: tuple[str, ...] = ("repair", "tumor")


def compute_plasticity_score(
    transition_probs: Tensor | np.ndarray,
    *,
    epsilon: float = 1e-10,
) -> Tensor:
    """Compute plasticity as entropy of transition probability distribution.

    Plasticity measures fate uncertainty: cells with high plasticity could
    transition to multiple states with similar probability. Low plasticity
    means the cell is committed to a specific fate.

    Args:
        transition_probs: (n_cells, n_stages) probability of transitioning to each stage.
            Must sum to 1 along dim=-1 (i.e., valid probability distribution).
        epsilon: Small constant for numerical stability in log computation.

    Returns:
        (n_cells,) entropy values - higher = more plastic (uncertain fate).
        Range: [0, log(n_stages)] where max entropy occurs for uniform distribution.

    Example:
        >>> probs = torch.tensor([[0.1, 0.1, 0.2, 0.3, 0.3],  # high plasticity
        ...                       [0.01, 0.01, 0.01, 0.02, 0.95]])  # low plasticity (committed)
        >>> scores = compute_plasticity_score(probs)
        >>> assert scores[0] > scores[1]  # first cell is more plastic
    """
    if isinstance(transition_probs, np.ndarray):
        transition_probs = torch.from_numpy(transition_probs)

    # Ensure valid probability distribution
    probs = transition_probs.float()
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    # Clamp for numerical stability
    probs = probs.clamp(min=epsilon, max=1.0 - epsilon)

    # Renormalize after clamping
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Shannon entropy: H = -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)

    return entropy


def compute_normalized_plasticity(
    transition_probs: Tensor | np.ndarray,
    *,
    epsilon: float = 1e-10,
) -> Tensor:
    """Compute normalized plasticity score in [0, 1] range.

    Normalizes entropy by the maximum possible entropy (uniform distribution),
    making scores comparable across different numbers of stages.

    Args:
        transition_probs: (n_cells, n_stages) probability distribution.
        epsilon: Small constant for numerical stability.

    Returns:
        (n_cells,) normalized entropy in [0, 1].
        0 = fully committed (deterministic), 1 = maximum uncertainty (uniform).
    """
    if isinstance(transition_probs, np.ndarray):
        transition_probs = torch.from_numpy(transition_probs)

    probs = transition_probs.float()
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    n_stages = probs.shape[-1]
    max_entropy = np.log(n_stages)  # entropy of uniform distribution

    entropy = compute_plasticity_score(probs, epsilon=epsilon)
    normalized = entropy / max_entropy

    return normalized.clamp(0.0, 1.0)


def compute_niche_resolution(
    plasticity_with_niche: Tensor | np.ndarray,
    plasticity_without_niche: Tensor | np.ndarray,
) -> Tensor:
    """Compute how much niche conditioning reduces plasticity (resolves fate uncertainty).

    This is the core metric for testing H1.3.1: "Niche conditioning reduces fate
    uncertainty." A positive value means the niche helps resolve cell fate.

    Args:
        plasticity_with_niche: (n_cells,) entropy when model has niche context.
        plasticity_without_niche: (n_cells,) entropy when model lacks niche context.

    Returns:
        (n_cells,) reduction in entropy when niche is provided.
        Positive = niche resolves plasticity (supports H1.3.1).
        Negative = niche increases uncertainty (contradicts H1.3.1).
        Zero = niche has no effect on fate certainty.
    """
    if isinstance(plasticity_with_niche, np.ndarray):
        plasticity_with_niche = torch.from_numpy(plasticity_with_niche)
    if isinstance(plasticity_without_niche, np.ndarray):
        plasticity_without_niche = torch.from_numpy(plasticity_without_niche)

    resolution = plasticity_without_niche.float() - plasticity_with_niche.float()
    return resolution


def compute_niche_resolution_effect_size(
    plasticity_with_niche: Tensor | np.ndarray,
    plasticity_without_niche: Tensor | np.ndarray,
) -> dict[str, float]:
    """Compute effect size statistics for niche resolution.

    Provides multiple effect size measures to characterize how much the niche
    resolves fate uncertainty across the cell population.

    Args:
        plasticity_with_niche: (n_cells,) entropy with niche context.
        plasticity_without_niche: (n_cells,) entropy without niche context.

    Returns:
        Dictionary with:
        - mean_resolution: Mean entropy reduction (higher = more resolution)
        - median_resolution: Median entropy reduction (robust to outliers)
        - std_resolution: Standard deviation of resolution
        - fraction_resolved: Fraction of cells with positive resolution
        - cohens_d: Cohen's d effect size (paired)
        - mean_relative_resolution: Mean relative entropy reduction (%)
    """
    resolution = compute_niche_resolution(plasticity_with_niche, plasticity_without_niche)

    if isinstance(plasticity_with_niche, np.ndarray):
        plasticity_with_niche = torch.from_numpy(plasticity_with_niche)
    if isinstance(plasticity_without_niche, np.ndarray):
        plasticity_without_niche = torch.from_numpy(plasticity_without_niche)

    resolution_np = resolution.numpy() if isinstance(resolution, Tensor) else resolution
    without_np = (
        plasticity_without_niche.numpy()
        if isinstance(plasticity_without_niche, Tensor)
        else plasticity_without_niche
    )

    # Basic statistics
    mean_res = float(np.mean(resolution_np))
    median_res = float(np.median(resolution_np))
    std_res = float(np.std(resolution_np))

    # Fraction of cells where niche helps
    fraction_resolved = float(np.mean(resolution_np > 0))

    # Cohen's d for paired samples
    if std_res > 1e-10:
        cohens_d = mean_res / std_res
    else:
        cohens_d = 0.0 if abs(mean_res) < 1e-10 else float("inf") * np.sign(mean_res)

    # Relative resolution (% entropy reduction)
    # Avoid division by zero
    valid_mask = without_np > 1e-10
    if valid_mask.sum() > 0:
        relative_res = resolution_np[valid_mask] / without_np[valid_mask]
        mean_relative = float(np.mean(relative_res))
    else:
        mean_relative = 0.0

    return {
        "mean_resolution": mean_res,
        "median_resolution": median_res,
        "std_resolution": std_res,
        "fraction_resolved": fraction_resolved,
        "cohens_d": cohens_d,
        "mean_relative_resolution": mean_relative,
    }


@dataclass
class BifurcationCellResult:
    """Result of bifurcation cell identification."""

    indices: np.ndarray  # Indices of bifurcation cells
    plasticity_scores: np.ndarray  # Plasticity scores of bifurcation cells
    threshold: float  # Plasticity threshold used
    cell_type_distribution: dict[str, int] = field(default_factory=dict)
    n_total_cells: int = 0
    n_bifurcation_cells: int = 0


def identify_bifurcation_cells(
    plasticity_scores: Tensor | np.ndarray,
    cell_types: Sequence[str] | np.ndarray | None = None,
    *,
    threshold_percentile: float = 90.0,
    min_plasticity: float | None = None,
) -> BifurcationCellResult:
    """Identify cells at fate decision points (highest plasticity).

    Bifurcation cells are those with highest plasticity scores, indicating
    they are at a decision point where fate is not yet determined. These
    are the cells where niche context should have the most influence.

    For LUAD, we expect bifurcation cells to be enriched in KAC/RPII
    (alveolar progenitor) populations per the KACs Atlas (Han et al. 2024).

    Args:
        plasticity_scores: (n_cells,) plasticity scores from compute_plasticity_score.
        cell_types: Optional (n_cells,) cell type labels for distribution analysis.
        threshold_percentile: Percentile threshold for identifying high-plasticity cells.
            Default 90 means top 10% most plastic cells.
        min_plasticity: Optional minimum absolute plasticity score. If provided,
            cells must exceed both this threshold AND the percentile threshold.

    Returns:
        BifurcationCellResult with:
        - indices: Array of cell indices identified as bifurcation cells
        - plasticity_scores: Plasticity scores of identified cells
        - threshold: The plasticity threshold used
        - cell_type_distribution: Count of each cell type among bifurcation cells
        - n_total_cells: Total number of cells
        - n_bifurcation_cells: Number of bifurcation cells identified
    """
    if isinstance(plasticity_scores, Tensor):
        scores_np = plasticity_scores.numpy()
    else:
        scores_np = np.asarray(plasticity_scores)

    n_cells = len(scores_np)

    # Compute threshold
    percentile_threshold = float(np.percentile(scores_np, threshold_percentile))
    threshold = percentile_threshold

    # Apply minimum plasticity constraint if provided
    if min_plasticity is not None:
        threshold = max(threshold, min_plasticity)

    # Identify bifurcation cells
    bifurcation_mask = scores_np >= threshold
    bifurcation_indices = np.where(bifurcation_mask)[0]
    bifurcation_scores = scores_np[bifurcation_mask]

    # Compute cell type distribution if provided
    cell_type_dist: dict[str, int] = {}
    if cell_types is not None:
        cell_types_arr = np.asarray(cell_types)
        for ct in cell_types_arr[bifurcation_mask]:
            cell_type_dist[str(ct)] = cell_type_dist.get(str(ct), 0) + 1

    return BifurcationCellResult(
        indices=bifurcation_indices,
        plasticity_scores=bifurcation_scores,
        threshold=threshold,
        cell_type_distribution=cell_type_dist,
        n_total_cells=n_cells,
        n_bifurcation_cells=len(bifurcation_indices),
    )


def compute_bifurcation_enrichment(
    bifurcation_result: BifurcationCellResult,
    all_cell_types: Sequence[str] | np.ndarray,
    *,
    target_cell_types: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute enrichment of specific cell types among bifurcation cells.

    Tests whether certain cell types (e.g., KAC, RPII) are overrepresented
    among high-plasticity cells, as expected from the biological hypothesis.

    Args:
        bifurcation_result: Result from identify_bifurcation_cells.
        all_cell_types: (n_cells,) cell type labels for all cells.
        target_cell_types: Cell types to compute enrichment for.
            Default targets alveolar progenitors: ["KAC", "RPII", "AT2_transitional"].

    Returns:
        Dictionary mapping cell type to enrichment statistics:
        - {cell_type}_observed_fraction: Fraction in bifurcation cells
        - {cell_type}_expected_fraction: Fraction in all cells
        - {cell_type}_fold_enrichment: Ratio of observed/expected
        - {cell_type}_log2_enrichment: log2(fold enrichment)
    """
    if target_cell_types is None:
        target_cell_types = ["KAC", "RPII", "AT2_transitional", "Krt8_ADI"]

    all_types = np.asarray(all_cell_types)
    n_total = len(all_types)
    n_bifurcation = bifurcation_result.n_bifurcation_cells

    results: dict[str, float] = {}

    for ct in target_cell_types:
        # Count in all cells
        n_total_ct = int(np.sum(all_types == ct))
        expected_frac = n_total_ct / n_total if n_total > 0 else 0.0

        # Count in bifurcation cells
        observed_count = bifurcation_result.cell_type_distribution.get(ct, 0)
        observed_frac = observed_count / n_bifurcation if n_bifurcation > 0 else 0.0

        # Compute enrichment
        if expected_frac > 1e-10:
            fold_enrichment = observed_frac / expected_frac
            log2_enrichment = float(np.log2(fold_enrichment)) if fold_enrichment > 0 else float("-inf")
        else:
            fold_enrichment = float("inf") if observed_frac > 0 else 1.0
            log2_enrichment = float("inf") if observed_frac > 0 else 0.0

        results[f"{ct}_observed_fraction"] = observed_frac
        results[f"{ct}_expected_fraction"] = expected_frac
        results[f"{ct}_fold_enrichment"] = fold_enrichment
        results[f"{ct}_log2_enrichment"] = log2_enrichment

    return results


@dataclass
class FateCommitmentResult:
    """Result of fate commitment analysis."""

    correlation_tumor: float  # Correlation between niche feature and P(tumor)
    correlation_repair: float  # Correlation between niche feature and P(repair)
    mean_tumor_prob_high: float  # Mean P(tumor) in high-feature niches
    mean_tumor_prob_low: float  # Mean P(tumor) in low-feature niches
    mean_repair_prob_high: float  # Mean P(repair) in high-feature niches
    mean_repair_prob_low: float  # Mean P(repair) in low-feature niches
    odds_ratio: float  # Odds ratio for tumor vs repair in high vs low
    log_odds: float  # Log odds ratio
    feature_name: str = ""
    n_cells: int = 0
    threshold: float = 0.0


def analyze_fate_commitment(
    transition_probs: Tensor | np.ndarray,
    niche_features: Tensor | np.ndarray,
    stages: Sequence[str] | None = None,
    *,
    feature_name: str = "IL1B",
    tumor_stages: Sequence[str] | None = None,
    repair_stages: Sequence[str] | None = None,
    threshold_percentile: float = 50.0,
) -> FateCommitmentResult:
    """Analyze how niche features correlate with fate probabilities.

    Tests H1.3.2: Do cells in IL1B-high niches have higher P(tumor) vs P(repair)?

    The biological hypothesis predicts:
    - IL1B-high niches commit plastic cells toward tumor
    - IL1B-low niches permit repair (AT1 differentiation)

    Args:
        transition_probs: (n_cells, n_stages) probability distribution over stages.
        niche_features: (n_cells,) or (n_cells, 1) feature values (e.g., IL1B score).
        stages: Stage names. Default: ["Normal", "AAH", "AIS", "MIA", "LUAD"].
        feature_name: Name of the niche feature being analyzed.
        tumor_stages: Stages considered as tumor fate. Default: ["MIA", "LUAD"].
        repair_stages: Stages considered as repair fate. Default: ["Normal"].
        threshold_percentile: Percentile to split high/low feature groups.

    Returns:
        FateCommitmentResult with correlation and odds ratio statistics.
    """
    if isinstance(transition_probs, Tensor):
        probs = transition_probs.numpy()
    else:
        probs = np.asarray(transition_probs)

    if isinstance(niche_features, Tensor):
        features = niche_features.numpy()
    else:
        features = np.asarray(niche_features)

    # Flatten features if needed
    if features.ndim > 1:
        features = features.ravel()

    n_cells = probs.shape[0]

    # Default stages
    if stages is None:
        stages = list(DEFAULT_STAGES)
    stages = list(stages)
    n_stages = len(stages)

    # Ensure probs match stages
    if probs.shape[1] != n_stages:
        raise ValueError(
            f"transition_probs has {probs.shape[1]} stages but {n_stages} stage names provided"
        )

    # Default fate definitions
    if tumor_stages is None:
        tumor_stages = ["MIA", "LUAD"]
    if repair_stages is None:
        repair_stages = ["Normal"]

    # Map stage names to indices
    stage_to_idx = {s: i for i, s in enumerate(stages)}
    tumor_idx = [stage_to_idx[s] for s in tumor_stages if s in stage_to_idx]
    repair_idx = [stage_to_idx[s] for s in repair_stages if s in stage_to_idx]

    # Compute fate probabilities
    p_tumor = probs[:, tumor_idx].sum(axis=1) if tumor_idx else np.zeros(n_cells)
    p_repair = probs[:, repair_idx].sum(axis=1) if repair_idx else np.zeros(n_cells)

    # Correlations
    if np.std(features) > 1e-10 and np.std(p_tumor) > 1e-10:
        corr_tumor = float(np.corrcoef(features, p_tumor)[0, 1])
    else:
        corr_tumor = 0.0

    if np.std(features) > 1e-10 and np.std(p_repair) > 1e-10:
        corr_repair = float(np.corrcoef(features, p_repair)[0, 1])
    else:
        corr_repair = 0.0

    # Split by feature threshold
    threshold = float(np.percentile(features, threshold_percentile))
    high_mask = features >= threshold
    low_mask = features < threshold

    # Mean probabilities in each group
    mean_tumor_high = float(np.mean(p_tumor[high_mask])) if high_mask.sum() > 0 else 0.0
    mean_tumor_low = float(np.mean(p_tumor[low_mask])) if low_mask.sum() > 0 else 0.0
    mean_repair_high = float(np.mean(p_repair[high_mask])) if high_mask.sum() > 0 else 0.0
    mean_repair_low = float(np.mean(p_repair[low_mask])) if low_mask.sum() > 0 else 0.0

    # Odds ratio: (tumor_high / repair_high) / (tumor_low / repair_low)
    # Avoid division by zero
    eps = 1e-10
    odds_high = mean_tumor_high / max(mean_repair_high, eps)
    odds_low = mean_tumor_low / max(mean_repair_low, eps)
    odds_ratio = odds_high / max(odds_low, eps)
    log_odds = float(np.log(odds_ratio)) if odds_ratio > 0 else float("-inf")

    return FateCommitmentResult(
        correlation_tumor=corr_tumor,
        correlation_repair=corr_repair,
        mean_tumor_prob_high=mean_tumor_high,
        mean_tumor_prob_low=mean_tumor_low,
        mean_repair_prob_high=mean_repair_high,
        mean_repair_prob_low=mean_repair_low,
        odds_ratio=odds_ratio,
        log_odds=log_odds,
        feature_name=feature_name,
        n_cells=n_cells,
        threshold=threshold,
    )


def compute_plasticity_by_cell_type(
    plasticity_scores: Tensor | np.ndarray,
    cell_types: Sequence[str] | np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute plasticity statistics stratified by cell type.

    Tests whether KACs/RPII have highest plasticity among epithelial cells,
    as predicted by the biological hypothesis.

    Args:
        plasticity_scores: (n_cells,) plasticity scores.
        cell_types: (n_cells,) cell type labels.

    Returns:
        Dictionary mapping cell type to statistics:
        {cell_type: {mean, median, std, min, max, n_cells}}
    """
    if isinstance(plasticity_scores, Tensor):
        scores = plasticity_scores.numpy()
    else:
        scores = np.asarray(plasticity_scores)

    types = np.asarray(cell_types)
    unique_types = np.unique(types)

    results: dict[str, dict[str, float]] = {}

    for ct in unique_types:
        mask = types == ct
        ct_scores = scores[mask]

        if len(ct_scores) > 0:
            results[str(ct)] = {
                "mean": float(np.mean(ct_scores)),
                "median": float(np.median(ct_scores)),
                "std": float(np.std(ct_scores)),
                "min": float(np.min(ct_scores)),
                "max": float(np.max(ct_scores)),
                "n_cells": int(len(ct_scores)),
            }

    return results


def rank_cell_types_by_plasticity(
    plasticity_by_type: dict[str, dict[str, float]],
    *,
    metric: str = "mean",
) -> list[tuple[str, float]]:
    """Rank cell types by plasticity score.

    Args:
        plasticity_by_type: Output from compute_plasticity_by_cell_type.
        metric: Which statistic to rank by (mean, median).

    Returns:
        List of (cell_type, score) tuples sorted descending by score.
    """
    rankings = []
    for ct, stats in plasticity_by_type.items():
        if metric in stats:
            rankings.append((ct, stats[metric]))

    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def stage_logits_to_probs(
    stage_logits: Tensor | np.ndarray,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Convert stage logits to probability distribution.

    Utility function to convert raw model outputs (logits) to probabilities
    suitable for plasticity analysis.

    Args:
        stage_logits: (n_cells, n_stages) raw logits from model.
        temperature: Temperature for softmax. Higher = softer distribution.
            Use temperature > 1 to amplify uncertainty for sensitivity analysis.

    Returns:
        (n_cells, n_stages) probability distribution (sums to 1 along stages).
    """
    if isinstance(stage_logits, np.ndarray):
        stage_logits = torch.from_numpy(stage_logits)

    logits = stage_logits.float()

    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)
    return probs


def compute_plasticity_from_logits(
    stage_logits: Tensor | np.ndarray,
    *,
    temperature: float = 1.0,
    normalize: bool = False,
) -> Tensor:
    """Compute plasticity directly from stage logits.

    Convenience function combining logits-to-probs conversion and plasticity
    computation.

    Args:
        stage_logits: (n_cells, n_stages) raw logits from model.
        temperature: Softmax temperature.
        normalize: If True, return normalized plasticity in [0, 1].

    Returns:
        (n_cells,) plasticity scores.
    """
    probs = stage_logits_to_probs(stage_logits, temperature=temperature)

    if normalize:
        return compute_normalized_plasticity(probs)
    else:
        return compute_plasticity_score(probs)


@dataclass
class PlasticityReport:
    """Comprehensive plasticity analysis report."""

    # Overall statistics
    mean_plasticity: float
    median_plasticity: float
    std_plasticity: float

    # Niche resolution (if computed)
    niche_resolution: dict[str, float] | None = None

    # Bifurcation cells
    bifurcation_result: BifurcationCellResult | None = None
    bifurcation_enrichment: dict[str, float] | None = None

    # Fate commitment
    fate_commitment: FateCommitmentResult | None = None

    # Cell type breakdown
    plasticity_by_cell_type: dict[str, dict[str, float]] | None = None
    cell_type_ranking: list[tuple[str, float]] | None = None


def generate_plasticity_report(
    transition_probs: Tensor | np.ndarray,
    *,
    transition_probs_no_niche: Tensor | np.ndarray | None = None,
    cell_types: Sequence[str] | np.ndarray | None = None,
    niche_features: Tensor | np.ndarray | None = None,
    niche_feature_name: str = "IL1B",
    stages: Sequence[str] | None = None,
    bifurcation_percentile: float = 90.0,
) -> PlasticityReport:
    """Generate comprehensive plasticity analysis report.

    This is the main entry point for plasticity analysis. It computes all
    relevant metrics for testing H1.3 and produces a report suitable for
    inclusion in evaluation results.

    Args:
        transition_probs: (n_cells, n_stages) probability distribution.
        transition_probs_no_niche: Optional (n_cells, n_stages) probabilities
            from model without niche context, for computing niche resolution.
        cell_types: Optional cell type labels for stratified analysis.
        niche_features: Optional niche feature values (e.g., IL1B score)
            for fate commitment analysis.
        niche_feature_name: Name of the niche feature.
        stages: Stage names.
        bifurcation_percentile: Percentile for identifying bifurcation cells.

    Returns:
        PlasticityReport with all computed metrics.
    """
    # Compute plasticity scores
    plasticity = compute_plasticity_score(transition_probs)
    plasticity_np = plasticity.numpy()

    # Basic statistics
    report = PlasticityReport(
        mean_plasticity=float(np.mean(plasticity_np)),
        median_plasticity=float(np.median(plasticity_np)),
        std_plasticity=float(np.std(plasticity_np)),
    )

    # Niche resolution analysis
    if transition_probs_no_niche is not None:
        plasticity_no_niche = compute_plasticity_score(transition_probs_no_niche)
        report.niche_resolution = compute_niche_resolution_effect_size(
            plasticity, plasticity_no_niche
        )

    # Bifurcation cell analysis
    report.bifurcation_result = identify_bifurcation_cells(
        plasticity,
        cell_types=cell_types,
        threshold_percentile=bifurcation_percentile,
    )

    # Bifurcation enrichment
    if cell_types is not None:
        report.bifurcation_enrichment = compute_bifurcation_enrichment(
            report.bifurcation_result,
            cell_types,
        )

    # Fate commitment analysis
    if niche_features is not None:
        report.fate_commitment = analyze_fate_commitment(
            transition_probs,
            niche_features,
            stages=stages,
            feature_name=niche_feature_name,
        )

    # Cell type stratification
    if cell_types is not None:
        report.plasticity_by_cell_type = compute_plasticity_by_cell_type(
            plasticity, cell_types
        )
        report.cell_type_ranking = rank_cell_types_by_plasticity(
            report.plasticity_by_cell_type
        )

    return report


def report_to_dict(report: PlasticityReport) -> dict[str, Any]:
    """Convert PlasticityReport to dictionary for JSON serialization.

    Args:
        report: PlasticityReport to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    result: dict[str, Any] = {
        "mean_plasticity": report.mean_plasticity,
        "median_plasticity": report.median_plasticity,
        "std_plasticity": report.std_plasticity,
    }

    if report.niche_resolution is not None:
        result["niche_resolution"] = report.niche_resolution

    if report.bifurcation_result is not None:
        result["bifurcation"] = {
            "n_cells": report.bifurcation_result.n_bifurcation_cells,
            "n_total": report.bifurcation_result.n_total_cells,
            "fraction": (
                report.bifurcation_result.n_bifurcation_cells
                / report.bifurcation_result.n_total_cells
                if report.bifurcation_result.n_total_cells > 0
                else 0.0
            ),
            "threshold": report.bifurcation_result.threshold,
            "cell_type_distribution": report.bifurcation_result.cell_type_distribution,
        }

    if report.bifurcation_enrichment is not None:
        result["bifurcation_enrichment"] = report.bifurcation_enrichment

    if report.fate_commitment is not None:
        fc = report.fate_commitment
        result["fate_commitment"] = {
            "feature_name": fc.feature_name,
            "correlation_tumor": fc.correlation_tumor,
            "correlation_repair": fc.correlation_repair,
            "mean_tumor_prob_high": fc.mean_tumor_prob_high,
            "mean_tumor_prob_low": fc.mean_tumor_prob_low,
            "mean_repair_prob_high": fc.mean_repair_prob_high,
            "mean_repair_prob_low": fc.mean_repair_prob_low,
            "odds_ratio": fc.odds_ratio,
            "log_odds": fc.log_odds,
            "threshold": fc.threshold,
            "n_cells": fc.n_cells,
        }

    if report.plasticity_by_cell_type is not None:
        result["plasticity_by_cell_type"] = report.plasticity_by_cell_type

    if report.cell_type_ranking is not None:
        result["cell_type_ranking"] = [
            {"cell_type": ct, "plasticity": score}
            for ct, score in report.cell_type_ranking
        ]

    return result


__all__ = [
    # Core plasticity functions
    "compute_plasticity_score",
    "compute_normalized_plasticity",
    "compute_niche_resolution",
    "compute_niche_resolution_effect_size",
    # Bifurcation analysis
    "identify_bifurcation_cells",
    "compute_bifurcation_enrichment",
    "BifurcationCellResult",
    # Fate commitment analysis
    "analyze_fate_commitment",
    "FateCommitmentResult",
    # Cell type analysis
    "compute_plasticity_by_cell_type",
    "rank_cell_types_by_plasticity",
    # Utility functions
    "stage_logits_to_probs",
    "compute_plasticity_from_logits",
    # Report generation
    "generate_plasticity_report",
    "report_to_dict",
    "PlasticityReport",
    # Constants
    "DEFAULT_STAGES",
    "FATE_LABELS",
]
