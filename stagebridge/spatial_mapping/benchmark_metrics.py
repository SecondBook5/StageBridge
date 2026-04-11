"""
Comprehensive benchmark metrics for spatial deconvolution evaluation.

Based on Sun et al. 2026 and Li et al. 2023 benchmarking frameworks.

Metrics implemented:
- Pearson correlation (stratified by cell type abundance)
- F1 score for rare cell type detection
- Precision, Recall, Jaccard index
- ROC AUC, PR AUC
- JSD (Jensen-Shannon Divergence)
- RMSE (per cell type and total)
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    roc_auc_score,
    average_precision_score,
)


@dataclass
class BenchmarkMetrics:
    """Container for all benchmark metrics."""

    # Correlation metrics (stratified)
    pearson_abundant: float
    pearson_medium: float
    pearson_rare: float
    pearson_overall: float

    # Detection metrics (for rare types)
    f1_abundant: float
    f1_medium: float
    f1_rare: float

    # Per-spot metrics
    jsd_median: float
    jsd_mean: float
    rmse_total: float

    # Per-celltype RMSE
    rmse_per_celltype: dict[str, float]

    # Additional metrics
    precision_rare: float
    recall_rare: float
    jaccard_rare: float

    # Optional AUC metrics (if ground truth available)
    roc_auc_mean: float | None = None
    pr_auc_mean: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pearson_abundant": self.pearson_abundant,
            "pearson_medium": self.pearson_medium,
            "pearson_rare": self.pearson_rare,
            "pearson_overall": self.pearson_overall,
            "f1_abundant": self.f1_abundant,
            "f1_medium": self.f1_medium,
            "f1_rare": self.f1_rare,
            "jsd_median": self.jsd_median,
            "jsd_mean": self.jsd_mean,
            "rmse_total": self.rmse_total,
            "rmse_per_celltype": self.rmse_per_celltype,
            "precision_rare": self.precision_rare,
            "recall_rare": self.recall_rare,
            "jaccard_rare": self.jaccard_rare,
            "roc_auc_mean": self.roc_auc_mean,
            "pr_auc_mean": self.pr_auc_mean,
        }


def stratify_celltypes_by_abundance(
    ground_truth: pd.DataFrame,
    abundant_threshold: float = 0.10,
    rare_threshold: float = 0.01,
) -> dict[str, list[str]]:
    """
    Stratify cell types by their abundance (fraction of spots where present).

    Following Sun et al. 2026:
    - Abundant: >10% of spots have this cell type
    - Medium: 1-10% of spots
    - Rare: <1% of spots

    Args:
        ground_truth: DataFrame of true proportions (spots x celltypes)
        abundant_threshold: Threshold for abundant (default 0.10)
        rare_threshold: Threshold for rare (default 0.01)

    Returns:
        Dict with keys 'abundant', 'medium', 'rare' containing cell type lists
    """
    # Calculate fraction of spots where each cell type is present (>5% proportion)
    presence_threshold = 0.05
    presence_fraction = (ground_truth > presence_threshold).mean()

    abundant = presence_fraction[presence_fraction > abundant_threshold].index.tolist()
    rare = presence_fraction[presence_fraction < rare_threshold].index.tolist()
    medium = [ct for ct in ground_truth.columns if ct not in abundant and ct not in rare]

    return {
        "abundant": abundant,
        "medium": medium,
        "rare": rare,
    }


def compute_pearson_correlation(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    celltypes: list[str] | None = None,
) -> float:
    """
    Compute Pearson correlation between predicted and ground truth.

    Args:
        predicted: Predicted proportions (spots x celltypes)
        ground_truth: True proportions (spots x celltypes)
        celltypes: Subset of cell types to evaluate (None = all)

    Returns:
        Median Pearson correlation across cell types
    """
    if celltypes is None:
        celltypes = list(set(predicted.columns) & set(ground_truth.columns))

    if len(celltypes) == 0:
        return np.nan

    correlations = []
    for ct in celltypes:
        if ct in predicted.columns and ct in ground_truth.columns:
            pred = predicted[ct].values
            true = ground_truth[ct].values

            # Skip if no variance
            if np.std(pred) < 1e-10 or np.std(true) < 1e-10:
                continue

            r, _ = stats.pearsonr(pred, true)
            if not np.isnan(r):
                correlations.append(r)

    return float(np.median(correlations)) if correlations else np.nan


def compute_f1_scores(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    celltypes: list[str] | None = None,
    presence_threshold: float = 0.05,
) -> float:
    """
    Compute F1 score for cell type presence detection.

    Following Sun et al.: binarize at 5% threshold for presence.

    Args:
        predicted: Predicted proportions
        ground_truth: True proportions
        celltypes: Cell types to evaluate
        presence_threshold: Threshold for "present" (default 0.05)

    Returns:
        Median F1 score across cell types
    """
    if celltypes is None:
        celltypes = list(set(predicted.columns) & set(ground_truth.columns))

    if len(celltypes) == 0:
        return np.nan

    f1_scores = []
    for ct in celltypes:
        if ct in predicted.columns and ct in ground_truth.columns:
            pred_binary = (predicted[ct].values > presence_threshold).astype(int)
            true_binary = (ground_truth[ct].values > presence_threshold).astype(int)

            # Skip if all zeros or all ones
            if len(np.unique(true_binary)) < 2:
                continue

            f1 = f1_score(true_binary, pred_binary, zero_division=0)
            f1_scores.append(f1)

    return float(np.median(f1_scores)) if f1_scores else np.nan


def compute_jsd_per_spot(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> np.ndarray:
    """
    Compute Jensen-Shannon Divergence per spot.

    Following Li et al. 2023 metrics.R implementation.

    Args:
        predicted: Predicted proportions (spots x celltypes)
        ground_truth: True proportions (spots x celltypes)

    Returns:
        Array of JSD values per spot
    """
    # Align columns
    common_cols = list(set(predicted.columns) & set(ground_truth.columns))
    pred = predicted[common_cols].values
    true = ground_truth[common_cols].values

    # Normalize rows to sum to 1
    pred = pred / (pred.sum(axis=1, keepdims=True) + 1e-10)
    true = true / (true.sum(axis=1, keepdims=True) + 1e-10)

    jsd_values = []
    for i in range(len(pred)):
        if np.sum(pred[i]) > 0 and np.sum(true[i]) > 0:
            jsd = jensenshannon(pred[i], true[i], base=2)
            jsd_values.append(jsd ** 2)  # JSD returns sqrt, square for divergence
        else:
            jsd_values.append(1.0)

    return np.array(jsd_values)


def compute_rmse(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    """
    Compute RMSE (total and per cell type).

    Following Li et al. 2023 metrics.R implementation.

    Args:
        predicted: Predicted proportions
        ground_truth: True proportions

    Returns:
        (total_rmse, dict of per-celltype RMSE)
    """
    # Align columns
    common_cols = list(set(predicted.columns) & set(ground_truth.columns))
    pred = predicted[common_cols]
    true = ground_truth[common_cols]

    # Per cell type RMSE
    rmse_per_ct = {}
    total_mse = 0.0
    n_total = 0

    for ct in common_cols:
        mse = np.mean((pred[ct].values - true[ct].values) ** 2)
        rmse_per_ct[ct] = float(np.sqrt(mse))
        total_mse += np.sum((pred[ct].values - true[ct].values) ** 2)
        n_total += len(pred)

    total_rmse = float(np.sqrt(total_mse / n_total)) if n_total > 0 else np.nan

    return total_rmse, rmse_per_ct


def compute_detection_metrics(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    celltypes: list[str] | None = None,
    presence_threshold: float = 0.05,
) -> dict[str, float]:
    """
    Compute precision, recall, and Jaccard for cell type detection.

    Args:
        predicted: Predicted proportions
        ground_truth: True proportions
        celltypes: Cell types to evaluate
        presence_threshold: Threshold for presence

    Returns:
        Dict with precision, recall, jaccard (median across cell types)
    """
    if celltypes is None:
        celltypes = list(set(predicted.columns) & set(ground_truth.columns))

    if len(celltypes) == 0:
        return {"precision": np.nan, "recall": np.nan, "jaccard": np.nan}

    precisions, recalls, jaccards = [], [], []

    for ct in celltypes:
        if ct in predicted.columns and ct in ground_truth.columns:
            pred_binary = (predicted[ct].values > presence_threshold).astype(int)
            true_binary = (ground_truth[ct].values > presence_threshold).astype(int)

            if len(np.unique(true_binary)) < 2:
                continue

            precisions.append(precision_score(true_binary, pred_binary, zero_division=0))
            recalls.append(recall_score(true_binary, pred_binary, zero_division=0))
            jaccards.append(jaccard_score(true_binary, pred_binary, zero_division=0))

    return {
        "precision": float(np.median(precisions)) if precisions else np.nan,
        "recall": float(np.median(recalls)) if recalls else np.nan,
        "jaccard": float(np.median(jaccards)) if jaccards else np.nan,
    }


def compute_auc_metrics(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    celltypes: list[str] | None = None,
    presence_threshold: float = 0.05,
) -> dict[str, float]:
    """
    Compute ROC AUC and PR AUC for cell type detection.

    Args:
        predicted: Predicted proportions (used as scores)
        ground_truth: True proportions (binarized as labels)
        celltypes: Cell types to evaluate
        presence_threshold: Threshold for ground truth binarization

    Returns:
        Dict with roc_auc and pr_auc (mean across cell types)
    """
    if celltypes is None:
        celltypes = list(set(predicted.columns) & set(ground_truth.columns))

    roc_aucs, pr_aucs = [], []

    for ct in celltypes:
        if ct in predicted.columns and ct in ground_truth.columns:
            scores = predicted[ct].values
            labels = (ground_truth[ct].values > presence_threshold).astype(int)

            # Need both classes present
            if len(np.unique(labels)) < 2:
                continue

            try:
                roc_aucs.append(roc_auc_score(labels, scores))
                pr_aucs.append(average_precision_score(labels, scores))
            except ValueError:
                continue

    return {
        "roc_auc": float(np.mean(roc_aucs)) if roc_aucs else np.nan,
        "pr_auc": float(np.mean(pr_aucs)) if pr_aucs else np.nan,
    }


def compute_all_metrics(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
    abundant_threshold: float = 0.10,
    rare_threshold: float = 0.01,
) -> BenchmarkMetrics:
    """
    Compute all benchmark metrics following Sun et al. and Li et al.

    Args:
        predicted: Predicted cell type proportions (spots x celltypes)
        ground_truth: Ground truth proportions (spots x celltypes)
        abundant_threshold: Threshold for abundant cell types
        rare_threshold: Threshold for rare cell types

    Returns:
        BenchmarkMetrics dataclass with all computed metrics
    """
    # Align indices
    common_spots = predicted.index.intersection(ground_truth.index)
    predicted = predicted.loc[common_spots]
    ground_truth = ground_truth.loc[common_spots]

    # Stratify cell types
    strata = stratify_celltypes_by_abundance(
        ground_truth, abundant_threshold, rare_threshold
    )

    # Pearson correlations (stratified)
    pearson_abundant = compute_pearson_correlation(predicted, ground_truth, strata["abundant"])
    pearson_medium = compute_pearson_correlation(predicted, ground_truth, strata["medium"])
    pearson_rare = compute_pearson_correlation(predicted, ground_truth, strata["rare"])
    pearson_overall = compute_pearson_correlation(predicted, ground_truth)

    # F1 scores (stratified)
    f1_abundant = compute_f1_scores(predicted, ground_truth, strata["abundant"])
    f1_medium = compute_f1_scores(predicted, ground_truth, strata["medium"])
    f1_rare = compute_f1_scores(predicted, ground_truth, strata["rare"])

    # JSD
    jsd_values = compute_jsd_per_spot(predicted, ground_truth)
    jsd_median = float(np.median(jsd_values))
    jsd_mean = float(np.mean(jsd_values))

    # RMSE
    rmse_total, rmse_per_ct = compute_rmse(predicted, ground_truth)

    # Detection metrics for rare types
    detection = compute_detection_metrics(predicted, ground_truth, strata["rare"])

    # AUC metrics
    auc = compute_auc_metrics(predicted, ground_truth)

    return BenchmarkMetrics(
        pearson_abundant=pearson_abundant,
        pearson_medium=pearson_medium,
        pearson_rare=pearson_rare,
        pearson_overall=pearson_overall,
        f1_abundant=f1_abundant,
        f1_medium=f1_medium,
        f1_rare=f1_rare,
        jsd_median=jsd_median,
        jsd_mean=jsd_mean,
        rmse_total=rmse_total,
        rmse_per_celltype=rmse_per_ct,
        precision_rare=detection["precision"],
        recall_rare=detection["recall"],
        jaccard_rare=detection["jaccard"],
        roc_auc_mean=auc["roc_auc"],
        pr_auc_mean=auc["pr_auc"],
    )


def format_metrics_table(metrics: BenchmarkMetrics, backend_name: str) -> pd.DataFrame:
    """
    Format metrics as a summary table row.

    Args:
        metrics: Computed benchmark metrics
        backend_name: Name of the backend

    Returns:
        Single-row DataFrame with metrics
    """
    return pd.DataFrame({
        "backend": [backend_name],
        "PCC_abundant": [metrics.pearson_abundant],
        "PCC_medium": [metrics.pearson_medium],
        "PCC_rare": [metrics.pearson_rare],
        "PCC_overall": [metrics.pearson_overall],
        "F1_abundant": [metrics.f1_abundant],
        "F1_medium": [metrics.f1_medium],
        "F1_rare": [metrics.f1_rare],
        "JSD_median": [metrics.jsd_median],
        "RMSE_total": [metrics.rmse_total],
        "Precision_rare": [metrics.precision_rare],
        "Recall_rare": [metrics.recall_rare],
        "ROC_AUC": [metrics.roc_auc_mean],
        "PR_AUC": [metrics.pr_auc_mean],
    })
