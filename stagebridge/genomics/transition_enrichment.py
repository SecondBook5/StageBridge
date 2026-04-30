"""Transition zone genomic enrichment analysis.

Tests whether high-transition niches are enriched for clinically
interpretable genomic features (driver mutations, actionable variants,
clonality states).

Statistical tests:
- Continuous features: Mann-Whitney U or permutation test
- Binary features: Fisher exact test
- Multi-category: Chi-square or permutation test
All p-values are FDR-corrected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from stagebridge.genomics.schemas import TransitionGenomicEnrichment

logger = logging.getLogger(__name__)


def load_transition_scores(path: str | Path) -> pd.DataFrame:
    """Load StageBridge transition scores.

    Args:
        path: Path to transition scores file

    Returns:
        DataFrame with transition scores
    """
    path = Path(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")

    col_mapping = {
        "cell_id": "barcode",
        "spot_id": "barcode",
        "transition": "transition_score",
        "gamma": "transition_score",
    }
    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["barcode", "transition_score"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def define_high_transition_group(
    transition_df: pd.DataFrame,
    quantile: float = 0.90,
    score_col: str = "transition_score",
) -> pd.DataFrame:
    """Define high-transition group by quantile threshold.

    Args:
        transition_df: DataFrame with transition scores
        quantile: Quantile threshold (default 90th percentile)
        score_col: Column name for transition score

    Returns:
        DataFrame with cells in high-transition group
    """
    threshold = transition_df[score_col].quantile(quantile)
    return transition_df[transition_df[score_col] >= threshold].copy()


def define_low_transition_group(
    transition_df: pd.DataFrame,
    quantile: float = 0.50,
    score_col: str = "transition_score",
) -> pd.DataFrame:
    """Define low-transition group by quantile threshold.

    Args:
        transition_df: DataFrame with transition scores
        quantile: Quantile threshold (default median)
        score_col: Column name for transition score

    Returns:
        DataFrame with cells in low-transition group
    """
    threshold = transition_df[score_col].quantile(quantile)
    return transition_df[transition_df[score_col] < threshold].copy()


def compute_continuous_enrichment(
    high_values: np.ndarray,
    low_values: np.ndarray,
    method: Literal["mann_whitney", "permutation"] = "mann_whitney",
    n_permutations: int = 10000,
) -> tuple[float, float, float]:
    """Compute enrichment for continuous feature.

    Args:
        high_values: Values in high-transition group
        low_values: Values in low-transition group
        method: Statistical test method
        n_permutations: Number of permutations for permutation test

    Returns:
        Tuple of (p_value, effect_size, test_statistic)
    """
    high_values = np.array(high_values)
    low_values = np.array(low_values)

    high_values = high_values[~np.isnan(high_values)]
    low_values = low_values[~np.isnan(low_values)]

    if len(high_values) < 3 or len(low_values) < 3:
        return 1.0, 0.0, 0.0

    high_mean = np.mean(high_values)
    low_mean = np.mean(low_values)

    pooled_std = np.sqrt(
        (np.var(high_values) * len(high_values) + np.var(low_values) * len(low_values))
        / (len(high_values) + len(low_values))
    )
    if pooled_std > 0:
        effect_size = (high_mean - low_mean) / pooled_std
    else:
        effect_size = 0.0

    if method == "mann_whitney":
        stat, p_value = stats.mannwhitneyu(
            high_values, low_values, alternative="two-sided"
        )
    else:
        observed_diff = high_mean - low_mean
        combined = np.concatenate([high_values, low_values])
        n_high = len(high_values)

        perm_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_high = combined[:n_high]
            perm_low = combined[n_high:]
            perm_diffs.append(np.mean(perm_high) - np.mean(perm_low))

        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        stat = observed_diff

    return p_value, effect_size, stat


def compute_binary_enrichment(
    high_positive: int,
    high_total: int,
    low_positive: int,
    low_total: int,
) -> tuple[float, float]:
    """Test enrichment for binary feature using Fisher exact test.

    Args:
        high_positive: Positive count in high-transition group
        high_total: Total count in high-transition group
        low_positive: Positive count in low-transition group
        low_total: Total count in low-transition group

    Returns:
        Tuple of (p_value, odds_ratio)
    """
    table = [
        [high_positive, high_total - high_positive],
        [low_positive, low_total - low_positive],
    ]

    odds_ratio, p_value = stats.fisher_exact(table, alternative="two-sided")

    return p_value, odds_ratio


def compute_feature_enrichment(
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    feature: str,
    feature_type: Literal["continuous", "binary", "categorical"] = "continuous",
) -> dict:
    """Test enrichment for a single feature.

    Args:
        high_df: High-transition group DataFrame
        low_df: Low-transition group DataFrame
        feature: Feature column name
        feature_type: Type of feature

    Returns:
        Dictionary with test results
    """
    if feature not in high_df.columns or feature not in low_df.columns:
        return {
            "feature": feature,
            "p_value": 1.0,
            "effect_size": 0.0,
            "high_mean": np.nan,
            "low_mean": np.nan,
            "n_high": 0,
            "n_low": 0,
            "test_method": "none",
            "error": f"Feature '{feature}' not found",
        }

    high_values = high_df[feature].dropna()
    low_values = low_df[feature].dropna()

    if feature_type == "binary":
        high_positive = high_values.sum()
        low_positive = low_values.sum()

        p_value, odds_ratio = compute_binary_enrichment(
            high_positive, len(high_values),
            low_positive, len(low_values),
        )

        return {
            "feature": feature,
            "p_value": p_value,
            "effect_size": np.log2(odds_ratio) if odds_ratio > 0 else 0.0,
            "high_mean": high_positive / len(high_values) if len(high_values) > 0 else 0,
            "low_mean": low_positive / len(low_values) if len(low_values) > 0 else 0,
            "n_high": len(high_values),
            "n_low": len(low_values),
            "test_method": "fisher_exact",
            "odds_ratio": odds_ratio,
        }

    elif feature_type == "continuous":
        p_value, effect_size, _ = compute_continuous_enrichment(
            high_values.values, low_values.values
        )

        return {
            "feature": feature,
            "p_value": p_value,
            "effect_size": effect_size,
            "high_mean": high_values.mean(),
            "low_mean": low_values.mean(),
            "n_high": len(high_values),
            "n_low": len(low_values),
            "test_method": "mann_whitney",
        }

    else:
        high_counts = high_values.value_counts()
        low_counts = low_values.value_counts()

        all_categories = set(high_counts.index) | set(low_counts.index)
        contingency = []
        for cat in all_categories:
            contingency.append([
                high_counts.get(cat, 0),
                low_counts.get(cat, 0),
            ])

        if len(contingency) >= 2:
            stat, p_value, _, _ = stats.chi2_contingency(
                np.array(contingency).T
            )
        else:
            p_value = 1.0

        return {
            "feature": feature,
            "p_value": p_value,
            "effect_size": 0.0,
            "high_mean": np.nan,
            "low_mean": np.nan,
            "n_high": len(high_values),
            "n_low": len(low_values),
            "test_method": "chi_square",
        }


def adjust_pvalues_bh(
    df: pd.DataFrame,
    p_col: str = "p_value",
    q_col: str = "q_value",
) -> pd.DataFrame:
    """Adjust p-values using Benjamini-Hochberg FDR correction.

    Args:
        df: DataFrame with p-values
        p_col: Column name for p-values
        q_col: Column name for adjusted p-values

    Returns:
        DataFrame with q-values added
    """
    df = df.copy()

    if p_col not in df.columns or len(df) == 0:
        df[q_col] = np.nan
        return df

    from scipy.stats import false_discovery_control

    p_values = df[p_col].values
    valid_mask = ~np.isnan(p_values)

    q_values = np.full(len(p_values), np.nan)
    if valid_mask.sum() > 0:
        q_values[valid_mask] = false_discovery_control(p_values[valid_mask])

    df[q_col] = q_values

    return df


def compute_variant_enrichment_by_transition_score(
    transition_df: pd.DataFrame,
    variant_evidence_df: pd.DataFrame,
    high_quantile: float = 0.90,
    low_quantile: float = 0.50,
) -> pd.DataFrame:
    """Test variant enrichment in high vs low transition zones.

    Args:
        transition_df: DataFrame with transition scores
        variant_evidence_df: DataFrame with variant evidence per cell
        high_quantile: Quantile for high-transition group
        low_quantile: Quantile for low-transition group

    Returns:
        DataFrame with enrichment results
    """
    df = transition_df.merge(
        variant_evidence_df,
        on="barcode",
        how="left",
    )

    df["has_alt_evidence"] = df["evidence_label"].isin([
        "alt_supported", "weak_alt_evidence"
    ])

    high_df = define_high_transition_group(df, high_quantile)
    low_df = define_low_transition_group(df, low_quantile)

    results = []

    result = compute_feature_enrichment(
        high_df, low_df, "has_alt_evidence", "binary"
    )
    result["comparison"] = "high_vs_low_transition"
    result["feature_name"] = "any_variant_expression"
    results.append(result)

    for variant_id in df["variant_id"].dropna().unique():
        var_mask = df["variant_id"] == variant_id
        if var_mask.sum() < 10:
            continue

        var_df = df[var_mask].copy()
        var_high = define_high_transition_group(var_df, high_quantile)
        var_low = define_low_transition_group(var_df, low_quantile)

        result = compute_feature_enrichment(
            var_high, var_low, "has_alt_evidence", "binary"
        )
        result["comparison"] = "high_vs_low_transition"
        result["feature_name"] = f"variant:{variant_id}"
        results.append(result)

    result_df = pd.DataFrame(results)
    result_df = adjust_pvalues_bh(result_df)

    return result_df


def compute_actionability_enrichment_by_transition_score(
    transition_df: pd.DataFrame,
    actionability_df: pd.DataFrame,
    high_quantile: float = 0.90,
    low_quantile: float = 0.50,
) -> pd.DataFrame:
    """Test actionability enrichment in transition zones.

    Args:
        transition_df: DataFrame with transition scores (spot-level)
        actionability_df: DataFrame with actionability (variant-level)
        high_quantile: Quantile for high-transition
        low_quantile: Quantile for low-transition

    Returns:
        DataFrame with enrichment results
    """
    sample_action = actionability_df.groupby("sample_id").agg({
        "oncogenicity": lambda x: (x.isin(["oncogenic", "likely_oncogenic"])).any(),
        "actionability_level": lambda x: (x.isin([
            "level_1", "level_2", "level_3A", "level_3B", "level_4"
        ])).any(),
    }).reset_index()

    sample_action = sample_action.rename(columns={
        "oncogenicity": "has_oncogenic_variant",
        "actionability_level": "has_actionable_variant",
    })

    df = transition_df.merge(sample_action, on="sample_id", how="left")
    df["has_oncogenic_variant"] = df["has_oncogenic_variant"].fillna(False)
    df["has_actionable_variant"] = df["has_actionable_variant"].fillna(False)

    high_df = define_high_transition_group(df, high_quantile)
    low_df = define_low_transition_group(df, low_quantile)

    results = []

    for feature in ["has_oncogenic_variant", "has_actionable_variant"]:
        result = compute_feature_enrichment(high_df, low_df, feature, "binary")
        result["comparison"] = "high_vs_low_transition"
        result["feature_name"] = feature
        results.append(result)

    result_df = pd.DataFrame(results)
    result_df = adjust_pvalues_bh(result_df)

    return result_df


def compute_clonality_enrichment_by_transition_score(
    transition_df: pd.DataFrame,
    clonality_df: pd.DataFrame,
    high_quantile: float = 0.90,
    low_quantile: float = 0.50,
) -> pd.DataFrame:
    """Test clonality enrichment in transition zones.

    Args:
        transition_df: DataFrame with transition scores
        clonality_df: DataFrame with clonality estimates
        high_quantile: Quantile for high-transition
        low_quantile: Quantile for low-transition

    Returns:
        DataFrame with enrichment results
    """
    sample_clonality = clonality_df.groupby("sample_id").agg({
        "tumor_vaf": "mean",
        "cancer_cell_fraction": "mean",
        "clonality_label": lambda x: (x.isin(["clonal", "clonal_like"])).mean(),
    }).reset_index()

    sample_clonality = sample_clonality.rename(columns={
        "clonality_label": "clonal_fraction",
    })

    df = transition_df.merge(sample_clonality, on="sample_id", how="left")

    high_df = define_high_transition_group(df, high_quantile)
    low_df = define_low_transition_group(df, low_quantile)

    results = []

    for feature in ["tumor_vaf", "cancer_cell_fraction", "clonal_fraction"]:
        if feature in df.columns:
            result = compute_feature_enrichment(high_df, low_df, feature, "continuous")
            result["comparison"] = "high_vs_low_transition"
            result["feature_name"] = feature
            results.append(result)

    result_df = pd.DataFrame(results)
    result_df = adjust_pvalues_bh(result_df)

    return result_df


def generate_transition_genomic_enrichment_table(
    transition_df: pd.DataFrame,
    variant_evidence_df: pd.DataFrame | None = None,
    actionability_df: pd.DataFrame | None = None,
    clonality_df: pd.DataFrame | None = None,
    high_quantile: float = 0.90,
    low_quantile: float = 0.50,
) -> pd.DataFrame:
    """Generate comprehensive genomic enrichment table.

    Args:
        transition_df: DataFrame with transition scores
        variant_evidence_df: Optional variant evidence DataFrame
        actionability_df: Optional actionability DataFrame
        clonality_df: Optional clonality DataFrame
        high_quantile: Quantile for high-transition
        low_quantile: Quantile for low-transition

    Returns:
        Combined enrichment results DataFrame
    """
    all_results = []

    if variant_evidence_df is not None:
        var_results = compute_variant_enrichment_by_transition_score(
            transition_df, variant_evidence_df, high_quantile, low_quantile
        )
        var_results["category"] = "variant_evidence"
        all_results.append(var_results)

    if actionability_df is not None:
        action_results = compute_actionability_enrichment_by_transition_score(
            transition_df, actionability_df, high_quantile, low_quantile
        )
        action_results["category"] = "actionability"
        all_results.append(action_results)

    if clonality_df is not None:
        clonal_results = compute_clonality_enrichment_by_transition_score(
            transition_df, clonality_df, high_quantile, low_quantile
        )
        clonal_results["category"] = "clonality"
        all_results.append(clonal_results)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    combined = adjust_pvalues_bh(combined)

    return combined
