"""Clonal validation metrics for H3 hypothesis testing.

This module implements validation of model predictions against clonal ground truth
from CNV inference. The key hypotheses from docs/HYPOTHESIS.md:

H3.1: Cells with high predicted transition probability are more likely to belong
      to shared clones (Pattern 1a/1b).
      - Test: Logistic regression AUC, Fisher exact test
      - Required: AUC > 0.6, OR > 1.5, p < 0.05

H3.2: Pattern 1a cases (direct lineage) show higher niche influence scores than
      Pattern 2 cases (independent origins).
      - Test: Mann-Whitney U
      - Required: Significantly higher in 1a (p < 0.05)

Pattern definitions (from Peng et al. 2025):
- Pattern 1a: Precursor clones present in LUAD + LUAD has additional subclones (direct lineage)
- Pattern 1b: Shared clones + stage-specific clones in both (branched evolution)
- Pattern 2: No shared clones between precursor and LUAD (independent origins)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# Patterns that indicate shared clones (evolutionary continuity)
SHARED_CLONE_PATTERNS = {"1a", "1b"}
# Patterns that indicate independent origins (no evolutionary continuity)
INDEPENDENT_PATTERNS = {"2"}
# All valid patterns for analysis
VALID_PATTERNS = {"1a", "1b", "2", "stable"}


@dataclass
class H3_1_Result:
    """Result of H3.1 hypothesis test: transition probability vs shared clones."""

    # Logistic regression metrics
    auc: float
    auc_ci_lower: float
    auc_ci_upper: float

    # Fisher exact test
    odds_ratio: float
    fisher_pvalue: float

    # Contingency table
    n_high_trans_shared: int  # High transition, shared clone
    n_high_trans_independent: int  # High transition, independent
    n_low_trans_shared: int  # Low transition, shared clone
    n_low_trans_independent: int  # Low transition, independent

    # Thresholds used
    transition_threshold: float
    transition_percentile: float

    # Sample sizes
    n_cells_total: int
    n_cells_with_pattern: int
    n_shared_clone_cells: int
    n_independent_cells: int

    # Hypothesis test result
    h3_1_supported: bool  # True if AUC > 0.6 AND OR > 1.5 AND p < 0.05

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "auc": self.auc,
            "auc_ci_lower": self.auc_ci_lower,
            "auc_ci_upper": self.auc_ci_upper,
            "odds_ratio": self.odds_ratio,
            "fisher_pvalue": self.fisher_pvalue,
            "contingency_table": {
                "high_trans_shared": self.n_high_trans_shared,
                "high_trans_independent": self.n_high_trans_independent,
                "low_trans_shared": self.n_low_trans_shared,
                "low_trans_independent": self.n_low_trans_independent,
            },
            "transition_threshold": self.transition_threshold,
            "transition_percentile": self.transition_percentile,
            "n_cells_total": self.n_cells_total,
            "n_cells_with_pattern": self.n_cells_with_pattern,
            "n_shared_clone_cells": self.n_shared_clone_cells,
            "n_independent_cells": self.n_independent_cells,
            "h3_1_supported": self.h3_1_supported,
        }


@dataclass
class H3_2_Result:
    """Result of H3.2 hypothesis test: niche influence by pattern."""

    # Mann-Whitney U test
    statistic: float
    pvalue: float

    # Effect size (rank-biserial correlation)
    effect_size: float

    # Group statistics
    mean_influence_1a: float
    mean_influence_1b: float
    mean_influence_2: float
    median_influence_1a: float
    median_influence_1b: float
    median_influence_2: float
    std_influence_1a: float
    std_influence_1b: float
    std_influence_2: float

    # Sample sizes
    n_cells_1a: int
    n_cells_1b: int
    n_cells_2: int

    # Pairwise comparisons
    pvalue_1a_vs_2: float
    pvalue_1b_vs_2: float
    pvalue_1a_vs_1b: float

    # Hypothesis test result
    h3_2_supported: bool  # True if 1a > 2 with p < 0.05

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mann_whitney_statistic": self.statistic,
            "mann_whitney_pvalue": self.pvalue,
            "effect_size_rank_biserial": self.effect_size,
            "mean_influence": {
                "1a": self.mean_influence_1a,
                "1b": self.mean_influence_1b,
                "2": self.mean_influence_2,
            },
            "median_influence": {
                "1a": self.median_influence_1a,
                "1b": self.median_influence_1b,
                "2": self.median_influence_2,
            },
            "std_influence": {
                "1a": self.std_influence_1a,
                "1b": self.std_influence_1b,
                "2": self.std_influence_2,
            },
            "n_cells": {
                "1a": self.n_cells_1a,
                "1b": self.n_cells_1b,
                "2": self.n_cells_2,
            },
            "pairwise_pvalues": {
                "1a_vs_2": self.pvalue_1a_vs_2,
                "1b_vs_2": self.pvalue_1b_vs_2,
                "1a_vs_1b": self.pvalue_1a_vs_1b,
            },
            "h3_2_supported": self.h3_2_supported,
        }


@dataclass
class ClonalValidationReport:
    """Complete clonal validation report for H3 hypotheses."""

    h3_1: H3_1_Result | None = None
    h3_2: H3_2_Result | None = None

    # Overall assessment
    h3_supported: bool = False  # True if both H3.1 and H3.2 supported

    # Metadata
    n_cells_analyzed: int = 0
    n_donors_analyzed: int = 0
    patterns_found: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "h3_1": self.h3_1.to_dict() if self.h3_1 else None,
            "h3_2": self.h3_2.to_dict() if self.h3_2 else None,
            "h3_supported": self.h3_supported,
            "n_cells_analyzed": self.n_cells_analyzed,
            "n_donors_analyzed": self.n_donors_analyzed,
            "patterns_found": self.patterns_found,
        }


def validate_h3_1(
    transition_probs: np.ndarray,
    clonal_patterns: Sequence[str],
    *,
    transition_percentile: float = 75.0,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> H3_1_Result:
    """Test H3.1: High transition probability cells belong to shared clones.

    The hypothesis predicts that cells with high predicted transition probability
    (toward more advanced stages) are more likely to belong to clones that are
    shared between precursor and invasive lesions (Pattern 1a/1b).

    Args:
        transition_probs: (n_cells,) predicted transition probabilities.
            Higher values = more likely to transition to advanced stage.
        clonal_patterns: (n_cells,) clonal pattern labels ("1a", "1b", "2", etc.)
        transition_percentile: Percentile threshold for "high transition" cells.
        n_bootstrap: Number of bootstrap samples for AUC confidence interval.
        random_state: Random seed for reproducibility.

    Returns:
        H3_1_Result with AUC, odds ratio, Fisher exact test, and hypothesis support.
    """
    rng = np.random.RandomState(random_state)

    # Convert to arrays
    trans_probs = np.asarray(transition_probs).ravel()
    patterns = np.asarray(clonal_patterns)

    n_total = len(trans_probs)

    # Filter to cells with valid patterns (1a, 1b, 2)
    valid_mask = np.isin(patterns, list(SHARED_CLONE_PATTERNS | INDEPENDENT_PATTERNS))
    trans_probs_valid = trans_probs[valid_mask]
    patterns_valid = patterns[valid_mask]

    n_valid = len(trans_probs_valid)
    if n_valid < 10:
        log.warning(f"Only {n_valid} cells with valid clonal patterns (need 1a/1b/2)")
        # Return empty result
        return H3_1_Result(
            auc=0.5,
            auc_ci_lower=0.5,
            auc_ci_upper=0.5,
            odds_ratio=1.0,
            fisher_pvalue=1.0,
            n_high_trans_shared=0,
            n_high_trans_independent=0,
            n_low_trans_shared=0,
            n_low_trans_independent=0,
            transition_threshold=0.0,
            transition_percentile=transition_percentile,
            n_cells_total=n_total,
            n_cells_with_pattern=n_valid,
            n_shared_clone_cells=0,
            n_independent_cells=0,
            h3_1_supported=False,
        )

    # Binary labels: 1 = shared clone (1a/1b), 0 = independent (2)
    is_shared = np.isin(patterns_valid, list(SHARED_CLONE_PATTERNS)).astype(int)
    n_shared = is_shared.sum()
    n_independent = len(is_shared) - n_shared

    # Compute AUC using logistic regression
    # Higher transition prob should predict shared clone status
    try:
        auc = roc_auc_score(is_shared, trans_probs_valid)
    except ValueError:
        # Only one class present
        auc = 0.5

    # Bootstrap confidence interval for AUC
    auc_bootstrap = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_valid, size=n_valid, replace=True)
        y_boot = is_shared[idx]
        x_boot = trans_probs_valid[idx]
        if len(np.unique(y_boot)) > 1:
            try:
                auc_boot = roc_auc_score(y_boot, x_boot)
                auc_bootstrap.append(auc_boot)
            except ValueError:
                pass

    if auc_bootstrap:
        auc_ci_lower = float(np.percentile(auc_bootstrap, 2.5))
        auc_ci_upper = float(np.percentile(auc_bootstrap, 97.5))
    else:
        auc_ci_lower = auc
        auc_ci_upper = auc

    # Compute threshold for high/low transition
    threshold = float(np.percentile(trans_probs_valid, transition_percentile))

    # Build contingency table
    high_trans = trans_probs_valid >= threshold
    low_trans = ~high_trans

    n_high_shared = int((high_trans & (is_shared == 1)).sum())
    n_high_indep = int((high_trans & (is_shared == 0)).sum())
    n_low_shared = int((low_trans & (is_shared == 1)).sum())
    n_low_indep = int((low_trans & (is_shared == 0)).sum())

    # Fisher exact test
    contingency = [[n_high_shared, n_high_indep], [n_low_shared, n_low_indep]]
    try:
        odds_ratio, fisher_pvalue = stats.fisher_exact(contingency)
    except ValueError:
        odds_ratio = 1.0
        fisher_pvalue = 1.0

    # Handle infinite odds ratio
    if np.isinf(odds_ratio):
        odds_ratio = 100.0  # Cap at large value

    # Determine if H3.1 is supported
    # Criteria: AUC > 0.6 AND OR > 1.5 AND p < 0.05
    h3_1_supported = (auc > 0.6) and (odds_ratio > 1.5) and (fisher_pvalue < 0.05)

    return H3_1_Result(
        auc=float(auc),
        auc_ci_lower=auc_ci_lower,
        auc_ci_upper=auc_ci_upper,
        odds_ratio=float(odds_ratio),
        fisher_pvalue=float(fisher_pvalue),
        n_high_trans_shared=n_high_shared,
        n_high_trans_independent=n_high_indep,
        n_low_trans_shared=n_low_shared,
        n_low_trans_independent=n_low_indep,
        transition_threshold=threshold,
        transition_percentile=transition_percentile,
        n_cells_total=n_total,
        n_cells_with_pattern=n_valid,
        n_shared_clone_cells=n_shared,
        n_independent_cells=n_independent,
        h3_1_supported=h3_1_supported,
    )


def validate_h3_2(
    niche_influence: np.ndarray,
    clonal_patterns: Sequence[str],
) -> H3_2_Result:
    """Test H3.2: Pattern 1a shows higher niche influence than Pattern 2.

    The hypothesis predicts that cells from patients with direct lineage evolution
    (Pattern 1a) show stronger niche influence on their transition dynamics than
    cells from patients with independent origins (Pattern 2).

    Args:
        niche_influence: (n_cells,) niche influence scores from the model.
            Higher values = stronger niche effect on transition.
        clonal_patterns: (n_cells,) clonal pattern labels ("1a", "1b", "2", etc.)

    Returns:
        H3_2_Result with Mann-Whitney test, effect size, and hypothesis support.
    """
    # Convert to arrays
    influence = np.asarray(niche_influence).ravel()
    patterns = np.asarray(clonal_patterns)

    # Separate by pattern
    mask_1a = patterns == "1a"
    mask_1b = patterns == "1b"
    mask_2 = patterns == "2"

    inf_1a = influence[mask_1a]
    inf_1b = influence[mask_1b]
    inf_2 = influence[mask_2]

    n_1a = len(inf_1a)
    n_1b = len(inf_1b)
    n_2 = len(inf_2)

    # Compute statistics
    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def safe_median(arr):
        return float(np.median(arr)) if len(arr) > 0 else 0.0

    def safe_std(arr):
        return float(np.std(arr)) if len(arr) > 0 else 0.0

    mean_1a = safe_mean(inf_1a)
    mean_1b = safe_mean(inf_1b)
    mean_2 = safe_mean(inf_2)
    median_1a = safe_median(inf_1a)
    median_1b = safe_median(inf_1b)
    median_2 = safe_median(inf_2)
    std_1a = safe_std(inf_1a)
    std_1b = safe_std(inf_1b)
    std_2 = safe_std(inf_2)

    # Mann-Whitney U tests (one-sided: 1a > 2)
    def safe_mannwhitneyu(x, y, alternative="greater"):
        if len(x) < 2 or len(y) < 2:
            return 0.0, 1.0
        try:
            stat, pval = stats.mannwhitneyu(x, y, alternative=alternative)
            return float(stat), float(pval)
        except ValueError:
            return 0.0, 1.0

    # Primary test: 1a vs 2
    stat_1a_2, pval_1a_2 = safe_mannwhitneyu(inf_1a, inf_2, "greater")

    # Secondary tests
    stat_1b_2, pval_1b_2 = safe_mannwhitneyu(inf_1b, inf_2, "greater")
    stat_1a_1b, pval_1a_1b = safe_mannwhitneyu(inf_1a, inf_1b, "two-sided")

    # Effect size: rank-biserial correlation
    # r = (2U)/(n1*n2) - 1
    # When U = n1*n2 (all x > y), r = 1
    # When U = 0 (all y > x), r = -1
    if n_1a > 0 and n_2 > 0:
        effect_size = (2 * stat_1a_2) / (n_1a * n_2) - 1
    else:
        effect_size = 0.0

    # Determine if H3.2 is supported
    # Criteria: 1a > 2 with p < 0.05
    h3_2_supported = (mean_1a > mean_2) and (pval_1a_2 < 0.05)

    return H3_2_Result(
        statistic=stat_1a_2,
        pvalue=pval_1a_2,
        effect_size=effect_size,
        mean_influence_1a=mean_1a,
        mean_influence_1b=mean_1b,
        mean_influence_2=mean_2,
        median_influence_1a=median_1a,
        median_influence_1b=median_1b,
        median_influence_2=median_2,
        std_influence_1a=std_1a,
        std_influence_1b=std_1b,
        std_influence_2=std_2,
        n_cells_1a=n_1a,
        n_cells_1b=n_1b,
        n_cells_2=n_2,
        pvalue_1a_vs_2=pval_1a_2,
        pvalue_1b_vs_2=pval_1b_2,
        pvalue_1a_vs_1b=pval_1a_1b,
        h3_2_supported=h3_2_supported,
    )


def run_clonal_validation(
    transition_probs: np.ndarray,
    niche_influence: np.ndarray,
    clonal_patterns: Sequence[str],
    donor_ids: Sequence[str] | None = None,
    *,
    transition_percentile: float = 75.0,
) -> ClonalValidationReport:
    """Run complete clonal validation for H3 hypotheses.

    This is the main entry point for validating model predictions against
    clonal ground truth. It tests both H3.1 (transition prob vs shared clones)
    and H3.2 (niche influence by pattern).

    Args:
        transition_probs: (n_cells,) predicted transition probabilities.
        niche_influence: (n_cells,) niche influence scores.
        clonal_patterns: (n_cells,) clonal pattern labels.
        donor_ids: Optional (n_cells,) donor IDs for counting unique patients.
        transition_percentile: Percentile for high/low transition split.

    Returns:
        ClonalValidationReport with H3.1 and H3.2 results.
    """
    patterns = np.asarray(clonal_patterns)
    n_cells = len(patterns)

    # Count unique patterns
    unique_patterns = [p for p in np.unique(patterns) if p in VALID_PATTERNS]

    # Count donors
    n_donors = 0
    if donor_ids is not None:
        donor_arr = np.asarray(donor_ids)
        valid_mask = np.isin(patterns, list(VALID_PATTERNS))
        n_donors = len(np.unique(donor_arr[valid_mask]))

    # Run H3.1 test
    log.info("Testing H3.1: Transition probability vs shared clones...")
    h3_1_result = validate_h3_1(
        transition_probs,
        clonal_patterns,
        transition_percentile=transition_percentile,
    )
    log.info(f"  AUC: {h3_1_result.auc:.3f} [{h3_1_result.auc_ci_lower:.3f}, {h3_1_result.auc_ci_upper:.3f}]")
    log.info(f"  Odds ratio: {h3_1_result.odds_ratio:.2f}, p={h3_1_result.fisher_pvalue:.4f}")
    log.info(f"  H3.1 supported: {h3_1_result.h3_1_supported}")

    # Run H3.2 test
    log.info("Testing H3.2: Niche influence by clonal pattern...")
    h3_2_result = validate_h3_2(niche_influence, clonal_patterns)
    log.info(f"  Mean influence - 1a: {h3_2_result.mean_influence_1a:.3f}, 2: {h3_2_result.mean_influence_2:.3f}")
    log.info(f"  Mann-Whitney p-value (1a > 2): {h3_2_result.pvalue_1a_vs_2:.4f}")
    log.info(f"  H3.2 supported: {h3_2_result.h3_2_supported}")

    # Overall H3 assessment
    h3_supported = h3_1_result.h3_1_supported and h3_2_result.h3_2_supported

    return ClonalValidationReport(
        h3_1=h3_1_result,
        h3_2=h3_2_result,
        h3_supported=h3_supported,
        n_cells_analyzed=n_cells,
        n_donors_analyzed=n_donors,
        patterns_found=unique_patterns,
    )


def compute_transition_probability(
    stage_logits: np.ndarray,
    current_stage_idx: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute transition probability from stage logits.

    Extracts the probability of transitioning to a MORE advanced stage
    (higher stage index) from the model's stage prediction logits.

    Args:
        stage_logits: (n_cells, n_stages) raw logits from model.
        current_stage_idx: (n_cells,) current stage index (0-4).
        temperature: Softmax temperature.

    Returns:
        (n_cells,) probability of transitioning to more advanced stage.
    """
    from scipy.special import softmax

    logits = np.asarray(stage_logits)
    current = np.asarray(current_stage_idx).astype(int)
    n_cells, n_stages = logits.shape

    # Apply temperature and softmax
    probs = softmax(logits / temperature, axis=-1)

    # For each cell, sum probability of stages > current
    transition_probs = np.zeros(n_cells)
    for i in range(n_cells):
        # Probability of being in a higher stage
        if current[i] < n_stages - 1:
            transition_probs[i] = probs[i, current[i] + 1 :].sum()
        else:
            # Already at highest stage
            transition_probs[i] = 0.0

    return transition_probs


def compute_niche_influence_from_attention(
    attention_weights: np.ndarray,
    receiver_idx: int = 0,
) -> np.ndarray:
    """Compute niche influence score from attention weights.

    Aggregates attention from niche tokens (rings, references) to the receiver,
    giving a scalar measure of how much the niche influences the cell.

    Args:
        attention_weights: (n_cells, n_heads, n_tokens, n_tokens) attention.
        receiver_idx: Index of receiver token (usually 0).

    Returns:
        (n_cells,) niche influence scores.
    """
    attn = np.asarray(attention_weights)

    # Average over heads
    if attn.ndim == 4:
        attn_mean = attn.mean(axis=1)  # (n_cells, n_tokens, n_tokens)
    else:
        attn_mean = attn

    # Sum attention FROM niche tokens TO receiver
    # Token layout: [receiver, ring1, ring2, ring3, ring4, hlca, luca, pathway, stats]
    # Niche tokens are indices 1-8 (everything except receiver)
    n_cells = attn_mean.shape[0]
    niche_influence = np.zeros(n_cells)

    for i in range(n_cells):
        # Attention to receiver from niche tokens
        # attn_mean[i, receiver_idx, 1:] = attention receiver pays to niche
        # We want: how much does niche influence receiver
        # This is attention FROM niche TO receiver: attn_mean[i, 1:, receiver_idx]
        niche_to_receiver = attn_mean[i, 1:, receiver_idx].sum()
        niche_influence[i] = niche_to_receiver

    return niche_influence


__all__ = [
    # Main entry point
    "run_clonal_validation",
    # Individual validation tests
    "validate_h3_1",
    "validate_h3_2",
    # Result classes
    "H3_1_Result",
    "H3_2_Result",
    "ClonalValidationReport",
    # Utility functions
    "compute_transition_probability",
    "compute_niche_influence_from_attention",
    # Constants
    "SHARED_CLONE_PATTERNS",
    "INDEPENDENT_PATTERNS",
    "VALID_PATTERNS",
]
