"""
Abundance-stratified metrics for spatial deconvolution benchmarking.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


AbundanceCategory = Literal["abundant", "medium", "rare"]


@dataclass
class AbundanceStratification:
    """
    Cell type stratification by abundance.

    Following Sun et al. 2026:
    - Abundant: mean proportion >= 4%
    - Medium: 2% <= mean proportion < 4%
    - Rare: mean proportion < 2%
    """

    abundant: list[str]
    medium: list[str]
    rare: list[str]
    mean_proportions: pd.Series

    def get_category(self, cell_type: str) -> AbundanceCategory:
        """Get abundance category for a cell type."""
        if cell_type in self.abundant:
            return "abundant"
        elif cell_type in self.medium:
            return "medium"
        elif cell_type in self.rare:
            return "rare"
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame of stratification."""
        rows = []
        for ct in self.mean_proportions.index:
            rows.append({
                "cell_type": ct,
                "mean_proportion": self.mean_proportions[ct],
                "abundance_category": self.get_category(ct),
            })
        return pd.DataFrame(rows).sort_values("mean_proportion", ascending=False)


def stratify_by_abundance(
    cell_type_proportions: pd.DataFrame,
    abundant_threshold: float = 0.04,
    rare_threshold: float = 0.02,
) -> AbundanceStratification:
    """
    Stratify cell types by abundance following Sun et al. 2026.

    Args:
        cell_type_proportions: DataFrame of shape (n_spots, n_celltypes)
        abundant_threshold: Threshold for abundant cell types (default: 4%)
        rare_threshold: Threshold for rare cell types (default: 2%)

    Returns:
        AbundanceStratification with cell types grouped by abundance
    """
    mean_props = cell_type_proportions.mean(axis=0)

    abundant = mean_props[mean_props >= abundant_threshold].index.tolist()
    rare = mean_props[mean_props < rare_threshold].index.tolist()
    medium = mean_props[
        (mean_props >= rare_threshold) & (mean_props < abundant_threshold)
    ].index.tolist()

    return AbundanceStratification(
        abundant=abundant,
        medium=medium,
        rare=rare,
        mean_proportions=mean_props,
    )


def compute_correlation_by_abundance(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    stratification: AbundanceStratification | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    """
    Compute correlation between predicted and ground truth proportions,
    stratified by cell type abundance.

    This is the key metric from Sun et al. showing performance degrades
    for rare cell types.

    Args:
        predicted: Predicted cell type proportions (n_spots, n_celltypes)
        ground_truth: Ground truth proportions (n_spots, n_celltypes)
        stratification: Optional pre-computed stratification (else computed)
        method: "pearson" or "spearman"

    Returns:
        DataFrame with columns: cell_type, correlation, abundance_category
    """
    # Ensure same cell types
    common_types = predicted.columns.intersection(ground_truth.columns)
    predicted = predicted[common_types]
    ground_truth = ground_truth[common_types]

    # Stratify if not provided
    if stratification is None:
        stratification = stratify_by_abundance(ground_truth)

    # Compute correlation per cell type
    correlations = []
    for ct in common_types:
        pred_vals = predicted[ct].values
        true_vals = ground_truth[ct].values

        # Skip if no variation
        if pred_vals.std() == 0 or true_vals.std() == 0:
            corr = np.nan
        else:
            if method == "pearson":
                corr, _ = pearsonr(pred_vals, true_vals)
            else:
                corr, _ = spearmanr(pred_vals, true_vals)

        correlations.append({
            "cell_type": ct,
            "correlation": corr,
            "abundance_category": stratification.get_category(ct),
            "mean_proportion": stratification.mean_proportions[ct],
        })

    return pd.DataFrame(correlations).sort_values("mean_proportion", ascending=False)


def compute_abundance_summary(
    correlation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute summary statistics by abundance category.

    Args:
        correlation_df: Output from compute_correlation_by_abundance

    Returns:
        Summary DataFrame with median/mean/std per category
    """
    summary = correlation_df.groupby("abundance_category")["correlation"].agg([
        ("median", "median"),
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("count", "count"),
    ]).reset_index()

    # Reorder categories
    category_order = ["abundant", "medium", "rare"]
    summary["abundance_category"] = pd.Categorical(
        summary["abundance_category"],
        categories=category_order,
        ordered=True
    )
    summary = summary.sort_values("abundance_category")

    return summary


def compute_f1_score_by_abundance(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    stratification: AbundanceStratification | None = None,
    presence_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Compute F1 score for binary presence/absence detection by abundance.

    Following Sun et al.: for rare cell types, binary presence detection
    is often more informative than continuous proportions.

    Args:
        predicted: Predicted proportions (n_spots, n_celltypes)
        ground_truth: Ground truth proportions (n_spots, n_celltypes)
        stratification: Optional pre-computed stratification
        presence_threshold: Threshold for calling presence (default: 5%)

    Returns:
        DataFrame with F1, precision, recall by cell type and abundance
    """
    common_types = predicted.columns.intersection(ground_truth.columns)
    predicted = predicted[common_types]
    ground_truth = ground_truth[common_types]

    if stratification is None:
        stratification = stratify_by_abundance(ground_truth)

    results = []
    for ct in common_types:
        # Binarize
        pred_present = (predicted[ct] >= presence_threshold).astype(int)
        true_present = (ground_truth[ct] >= presence_threshold).astype(int)

        # Compute metrics
        tp = ((pred_present == 1) & (true_present == 1)).sum()
        fp = ((pred_present == 1) & (true_present == 0)).sum()
        fn = ((pred_present == 0) & (true_present == 1)).sum()
        tn = ((pred_present == 0) & (true_present == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "cell_type": ct,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "abundance_category": stratification.get_category(ct),
            "mean_proportion": stratification.mean_proportions[ct],
        })

    return pd.DataFrame(results).sort_values("mean_proportion", ascending=False)


def compute_progression_specific_metrics(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    stage_labels: pd.Series,
    stratification: AbundanceStratification | None = None,
) -> dict[str, pd.DataFrame]:
    """
    UNIQUE CONTRIBUTION: Compute abundance-stratified metrics per progression stage.

    This extends Sun et al. by asking: Do backends preserve stage-specific
    cell type distributions, especially for rare progression-relevant types?

    Args:
        predicted: Predicted proportions (n_spots, n_celltypes)
        ground_truth: Ground truth proportions (n_spots, n_celltypes)
        stage_labels: Stage label per spot (e.g., "AAH", "AIS", "MIA", "LUAD")
        stratification: Optional pre-computed stratification

    Returns:
        Dictionary mapping stage to correlation DataFrame
    """
    if stratification is None:
        stratification = stratify_by_abundance(ground_truth)

    results_by_stage = {}
    for stage in stage_labels.unique():
        stage_mask = stage_labels == stage

        pred_stage = predicted.loc[stage_mask]
        true_stage = ground_truth.loc[stage_mask]

        if len(pred_stage) > 0:
            corr_df = compute_correlation_by_abundance(
                pred_stage,
                true_stage,
                stratification=stratification
            )
            corr_df["stage"] = stage
            results_by_stage[stage] = corr_df

    return results_by_stage


def identify_progression_relevant_rare_types(
    ground_truth: pd.DataFrame,
    stage_labels: pd.Series,
    stratification: AbundanceStratification,
    stage_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    UNIQUE CONTRIBUTION: Identify rare cell types with stage-specific enrichment.

    These are the most challenging but biologically important types for
    progression modeling.

    Args:
        ground_truth: Ground truth proportions (n_spots, n_celltypes)
        stage_labels: Stage label per spot
        stratification: Pre-computed abundance stratification
        stage_order: Optional stage progression order (e.g., ["AAH", "AIS", "MIA", "LUAD"])

    Returns:
        DataFrame of rare cell types with stage-enrichment scores
    """
    rare_types = stratification.rare

    if stage_order is None:
        stage_order = sorted(stage_labels.unique())

    results = []
    for ct in rare_types:
        stage_means = []
        for stage in stage_order:
            stage_mask = stage_labels == stage
            stage_mean = ground_truth.loc[stage_mask, ct].mean()
            stage_means.append(stage_mean)

        # Compute enrichment metrics
        max_mean = max(stage_means)
        min_mean = min(stage_means)
        fold_change = max_mean / (min_mean + 1e-10)

        # Monotonic trend test (correlation with stage order)
        stage_numeric = list(range(len(stage_order)))
        if len(set(stage_means)) > 1:
            trend_corr, _ = spearmanr(stage_numeric, stage_means)
        else:
            trend_corr = 0.0

        results.append({
            "cell_type": ct,
            "overall_mean": stratification.mean_proportions[ct],
            "max_stage_mean": max_mean,
            "min_stage_mean": min_mean,
            "fold_change": fold_change,
            "trend_correlation": trend_corr,
            "stage_enriched": stage_order[np.argmax(stage_means)],
        })

    df = pd.DataFrame(results).sort_values("fold_change", ascending=False)
    return df


def compare_backends_by_abundance(
    backend_results: dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    stratification: AbundanceStratification | None = None,
) -> pd.DataFrame:
    """
    Compare multiple backends with abundance stratification.

    Produces a comparison table like Sun et al. Figure 1B.

    Args:
        backend_results: Dict mapping backend name to predicted proportions
        ground_truth: Ground truth proportions
        stratification: Optional pre-computed stratification

    Returns:
        Comparison DataFrame with backend × abundance category performance
    """
    if stratification is None:
        stratification = stratify_by_abundance(ground_truth)

    results = []
    for backend_name, predicted in backend_results.items():
        corr_df = compute_correlation_by_abundance(
            predicted,
            ground_truth,
            stratification=stratification
        )

        # Summarize by abundance
        summary = compute_abundance_summary(corr_df)
        summary["backend"] = backend_name
        results.append(summary)

    # Combine all backends
    comparison = pd.concat(results, ignore_index=True)

    # Pivot for easy comparison
    comparison_wide = comparison.pivot(
        index="backend",
        columns="abundance_category",
        values="median"
    )

    return comparison_wide


def generate_benchmark_report(
    backend_results: dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    stage_labels: pd.Series | None = None,
    output_path: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate comprehensive benchmark report with abundance stratification.

    Args:
        backend_results: Dict mapping backend name to predicted proportions
        ground_truth: Ground truth proportions
        stage_labels: Optional stage labels for progression-specific analysis
        output_path: Optional path to save report tables

    Returns:
        Dictionary of report tables
    """
    # Stratify by abundance
    stratification = stratify_by_abundance(ground_truth)

    report = {
        "stratification_summary": stratification.summary(),
        "backend_comparison": compare_backends_by_abundance(
            backend_results, ground_truth, stratification
        ),
    }

    # Per-backend detailed results
    for backend_name, predicted in backend_results.items():
        corr_df = compute_correlation_by_abundance(
            predicted, ground_truth, stratification
        )
        report[f"{backend_name}_correlations"] = corr_df
        report[f"{backend_name}_summary"] = compute_abundance_summary(corr_df)

        f1_df = compute_f1_score_by_abundance(
            predicted, ground_truth, stratification
        )
        report[f"{backend_name}_f1_scores"] = f1_df

    # Progression-specific analysis (if stage labels provided)
    if stage_labels is not None:
        # Identify progression-relevant rare types
        rare_prog_types = identify_progression_relevant_rare_types(
            ground_truth, stage_labels, stratification
        )
        report["progression_relevant_rare_types"] = rare_prog_types

        # Per-stage backend comparison
        for backend_name, predicted in backend_results.items():
            stage_results = compute_progression_specific_metrics(
                predicted, ground_truth, stage_labels, stratification
            )

            # Combine stage results
            stage_combined = pd.concat(stage_results.values(), ignore_index=True)
            report[f"{backend_name}_by_stage"] = stage_combined

    # Save if path provided
    if output_path:
        import os
        os.makedirs(output_path, exist_ok=True)
        for name, df in report.items():
            df.to_csv(f"{output_path}/{name}.csv", index=False)

    return report
