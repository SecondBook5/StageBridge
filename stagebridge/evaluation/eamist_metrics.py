"""Classification metrics and statistics for EA-MIST lesion benchmarks."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


CANONICAL_STAGE_LABELS: tuple[str, ...] = ("Normal", "AAH", "AIS", "MIA", "LUAD")

# Grouped ordinal labels for rescue study
GROUPED_STAGE_LABELS: tuple[str, ...] = ("early_like", "intermediate_like", "invasive_like")

# Binary labels: pre-invasive vs invasive
BINARY_STAGE_LABELS: tuple[str, ...] = ("pre_invasive", "invasive")


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute a small-dependency Spearman correlation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    xr = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(xr) <= 0.0 or np.std(yr) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def compute_stage_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
) -> dict[str, float]:
    """Compute multiclass stage metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    label_list = list(range(len(CANONICAL_STAGE_LABELS))) if labels is None else [int(value) for value in labels]
    if y_true.shape[0] == 0:
        return {
            "stage_accuracy": float("nan"),
            "stage_macro_f1": float("nan"),
            "stage_balanced_accuracy": float("nan"),
            "stage_central_recall_mean": float("nan"),
        }
    recalls = recall_score(y_true, y_pred, labels=label_list, average=None, zero_division=0)
    recall_by_label = {int(label): float(recalls[idx]) for idx, label in enumerate(label_list)}
    central_labels = [label for label in label_list if label in {1, 2, 3}]
    central_recall = [recall_by_label[label] for label in central_labels if label in recall_by_label]
    return {
        "stage_accuracy": float(np.mean(y_true == y_pred)),
        "stage_macro_f1": float(f1_score(y_true, y_pred, labels=label_list, average="macro", zero_division=0)),
        "stage_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "stage_central_recall_mean": float(np.mean(central_recall)) if central_recall else float("nan"),
        "stage_nonzero_central_recalls": float(np.sum(np.asarray(central_recall) > 0.0)),
    }


def stage_confusion_matrix_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    label_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Return a JSON-friendly stage confusion matrix payload."""
    label_list = list(range(len(CANONICAL_STAGE_LABELS))) if labels is None else [int(value) for value in labels]
    names = list(CANONICAL_STAGE_LABELS if label_names is None else label_names)
    matrix = confusion_matrix(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64), labels=label_list)
    return {"labels": label_list, "label_names": names, "matrix": matrix.astype(int).tolist()}


def stage_support_payload(
    y_true: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    label_names: Iterable[str] | None = None,
) -> dict[str, int]:
    """Return observed support per stage."""
    label_list = list(range(len(CANONICAL_STAGE_LABELS))) if labels is None else [int(value) for value in labels]
    names = list(CANONICAL_STAGE_LABELS if label_names is None else label_names)
    y_true = np.asarray(y_true, dtype=np.int64)
    return {
        str(names[idx]): int(np.sum(y_true == int(label)))
        for idx, label in enumerate(label_list)
    }


def compute_grouped_stage_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
) -> dict[str, float]:
    """Compute grouped ordinal classification metrics.

    Uses the 3-class grouped labels (early_like, intermediate_like, invasive_like).
    Includes weighted kappa for ordinal agreement.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    label_list = list(range(len(GROUPED_STAGE_LABELS))) if labels is None else [int(v) for v in labels]
    if y_true.shape[0] == 0:
        return {
            "grouped_macro_f1": float("nan"),
            "grouped_balanced_accuracy": float("nan"),
            "grouped_weighted_kappa": float("nan"),
            "grouped_accuracy": float("nan"),
        }
    macro_f1 = float(f1_score(y_true, y_pred, labels=label_list, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(np.mean(y_true == y_pred))
    # Weighted (linear) kappa for ordinal agreement
    kappa = _linear_weighted_kappa(y_true, y_pred, num_classes=len(label_list))
    return {
        "grouped_macro_f1": macro_f1,
        "grouped_balanced_accuracy": bal_acc,
        "grouped_weighted_kappa": kappa,
        "grouped_accuracy": acc,
    }


def _linear_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> float:
    """Compute linearly-weighted Cohen's kappa for ordinal labels."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape[0] == 0:
        return float("nan")
    # Build weight matrix: w_ij = |i - j| / (num_classes - 1)
    weight_mat = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            weight_mat[i, j] = abs(i - j) / max(num_classes - 1, 1)
    # Observed confusion matrix
    observed = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(np.float64)
    n = observed.sum()
    if n == 0:
        return float("nan")
    observed /= n
    # Expected under independence
    row_marginals = observed.sum(axis=1)
    col_marginals = observed.sum(axis=0)
    expected = np.outer(row_marginals, col_marginals)
    # Weighted kappa
    po = (weight_mat * observed).sum()
    pe = (weight_mat * expected).sum()
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return float(1.0 - po / pe)


def grouped_confusion_matrix_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    label_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Return a JSON-friendly grouped confusion matrix payload."""
    label_list = list(range(len(GROUPED_STAGE_LABELS))) if labels is None else [int(v) for v in labels]
    names = list(GROUPED_STAGE_LABELS if label_names is None else label_names)
    matrix = confusion_matrix(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64), labels=label_list)
    return {"labels": label_list, "label_names": names, "matrix": matrix.astype(int).tolist()}


def grouped_support_payload(
    y_true: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    label_names: Iterable[str] | None = None,
) -> dict[str, int]:
    """Return observed support per grouped stage."""
    label_list = list(range(len(GROUPED_STAGE_LABELS))) if labels is None else [int(v) for v in labels]
    names = list(GROUPED_STAGE_LABELS if label_names is None else label_names)
    y_true = np.asarray(y_true, dtype=np.int64)
    return {
        str(names[idx]): int(np.sum(y_true == int(label)))
        for idx, label in enumerate(label_list)
    }


def composite_selection_score_grouped(metrics: dict[str, float]) -> float:
    """Checkpoint-selection score for grouped ordinal classification.

    Prioritises ordinal metrics: displacement_spearman + weighted_kappa + balanced_accuracy.
    """
    spearman = float(metrics.get("displacement_spearman", float("nan")))
    kappa = float(metrics.get("grouped_weighted_kappa", float("nan")))
    bal_acc = float(metrics.get("grouped_balanced_accuracy", float("nan")))
    macro_f1 = float(metrics.get("grouped_macro_f1", float("nan")))
    if np.isnan(bal_acc):
        return -float("inf")
    score = 0.0
    if not np.isnan(spearman):
        score += 0.40 * max(spearman, 0.0)
    if not np.isnan(kappa):
        score += 0.30 * max(kappa, 0.0)
    if not np.isnan(bal_acc):
        score += 0.20 * bal_acc
    if not np.isnan(macro_f1):
        score += 0.10 * macro_f1
    return float(score)


def compute_displacement_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    stage_indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute weak stage-ordered displacement regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {
            "displacement_mae": float("nan"),
            "displacement_spearman": float("nan"),
            "displacement_stage_monotonicity": float("nan"),
        }
    metrics = {
        "displacement_mae": float(np.mean(np.abs(y_true[valid] - y_pred[valid]))),
        "displacement_spearman": _safe_spearman(y_true[valid], y_pred[valid]),
        "displacement_stage_monotonicity": float("nan"),
    }
    if stage_indices is not None:
        stage_indices = np.asarray(stage_indices, dtype=np.int64)
        stage_valid = valid & np.isfinite(stage_indices)
        if stage_valid.sum() > 0:
            stage_means = []
            for stage_idx in sorted(np.unique(stage_indices[stage_valid]).tolist()):
                mask = stage_valid & (stage_indices == int(stage_idx))
                if np.any(mask):
                    stage_means.append((int(stage_idx), float(np.mean(y_pred[mask]))))
            if len(stage_means) >= 2:
                diffs = np.diff(np.asarray([value for _stage, value in stage_means], dtype=np.float32))
                metrics["displacement_stage_monotonicity"] = float(np.mean(diffs >= -1e-6))
    return metrics


def compute_masked_edge_metrics(
    logits: np.ndarray | None,
    targets: np.ndarray | None,
    mask: np.ndarray | None,
    *,
    edge_labels: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute auxiliary edge metrics only for valid masked lesions."""
    if logits is None or targets is None or mask is None:
        return {}
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    labels = list(edge_labels or [f"edge_{idx}" for idx in range(logits.shape[1])])
    metrics: dict[str, float] = {}
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    for idx, edge_label in enumerate(labels):
        valid = mask[:, idx]
        if valid.sum() == 0:
            metrics[f"{edge_label}_auroc"] = float("nan")
            metrics[f"{edge_label}_auprc"] = float("nan")
            continue
        y_true = targets[valid, idx].astype(np.int64)
        y_prob = probabilities[valid, idx]
        metrics[f"{edge_label}_auroc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        metrics[f"{edge_label}_auprc"] = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return metrics


def composite_selection_score(metrics: dict[str, float]) -> float:
    """Compute the checkpoint-selection score with balanced-accuracy guardrails."""
    macro_f1 = float(metrics.get("stage_macro_f1", float("nan")))
    balanced = float(metrics.get("stage_balanced_accuracy", float("nan")))
    spearman = float(metrics.get("displacement_spearman", float("nan")))
    central_recall = float(metrics.get("stage_central_recall_mean", float("nan")))
    nonzero_central = float(metrics.get("stage_nonzero_central_recalls", float("nan")))
    if np.isnan(macro_f1):
        return -float("inf")
    score = macro_f1
    if not np.isnan(balanced):
        score += 0.25 * balanced
        if balanced < 0.20:
            score -= 0.50
    if not np.isnan(spearman):
        score += 0.10 * max(spearman, 0.0)
    if not np.isnan(central_recall):
        score += 0.05 * central_recall
    if not np.isnan(nonzero_central) and nonzero_central < 2.0:
        score -= 0.25
    return float(score)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> float:
    """Compute expected calibration error for binary probabilities."""
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= left) & (y_prob < right if right < 1.0 else y_prob <= right)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y_true[mask]))
        confidence = float(np.mean(y_prob[mask]))
        ece += (np.sum(mask) / max(y_true.shape[0], 1)) * abs(accuracy - confidence)
    return float(ece)


def temperature_scale_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature on validation logits using a simple grid search."""
    labels = np.asarray(labels, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    best_temp = 1.0
    best_loss = float("inf")
    for temperature in np.linspace(0.5, 3.0, 26):
        probs = 1.0 / (1.0 + np.exp(-logits / temperature))
        eps = 1e-6
        loss = -np.mean(labels * np.log(np.clip(probs, eps, 1.0 - eps)) + (1.0 - labels) * np.log(np.clip(1.0 - probs, eps, 1.0 - eps)))
        if loss < best_loss:
            best_loss = float(loss)
            best_temp = float(temperature)
    return best_temp


def threshold_from_validation(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Choose a validation threshold by balanced accuracy, with F1 as tie-break."""
    best_threshold = 0.5
    best_score = (-1.0, -1.0)
    for threshold in np.linspace(0.1, 0.9, 81):
        pred = (y_prob >= threshold).astype(int)
        bal = balanced_accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)
        score = (float(bal), float(f1))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Compute lesion-level binary metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
    }
    return metrics


def build_curve_frames(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ROC, PR, and calibration curves as DataFrames."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if len(np.unique(y_true)) > 1:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    else:
        fpr = np.asarray([0.0, 1.0], dtype=np.float32)
        tpr = np.asarray([0.0, 1.0], dtype=np.float32)
        roc_thresholds = np.asarray([1.0, 0.0], dtype=np.float32)
        precision = np.asarray([float(np.mean(y_true))], dtype=np.float32)
        recall = np.asarray([1.0], dtype=np.float32)
        pr_thresholds = np.asarray([], dtype=np.float32)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    bins = np.linspace(0.0, 1.0, 11)
    cal_rows: list[dict[str, float]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= left) & (y_prob < right if right < 1.0 else y_prob <= right)
        cal_rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": int(np.sum(mask)),
                "mean_probability": float(np.mean(y_prob[mask])) if np.any(mask) else float("nan"),
                "observed_frequency": float(np.mean(y_true[mask])) if np.any(mask) else float("nan"),
            }
        )
    cal_df = pd.DataFrame(cal_rows)
    return roc_df, pr_df, cal_df


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    num_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Estimate bootstrap confidence intervals for AUROC and AUPRC."""
    rng = np.random.default_rng(int(seed))
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(int(num_bootstrap)):
        rows = rng.choice(np.arange(y_true.shape[0]), size=y_true.shape[0], replace=True)
        sample_true = y_true[rows]
        sample_prob = y_prob[rows]
        if len(np.unique(sample_true)) < 2:
            continue
        aurocs.append(float(roc_auc_score(sample_true, sample_prob)))
        auprcs.append(float(average_precision_score(sample_true, sample_prob)))
    def _interval(values: list[float]) -> tuple[float, float]:
        if not values:
            return (float("nan"), float("nan"))
        return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
    return {"auroc_ci": _interval(aurocs), "auprc_ci": _interval(auprcs)}


def build_per_donor_metrics(frame: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Compute donor-level metrics from a lesion prediction table."""
    rows: list[dict[str, float | str]] = []
    for donor_id, donor_frame in frame.groupby("donor_id", sort=False):
        metrics = compute_binary_metrics(
            donor_frame["label"].to_numpy(dtype=np.int64),
            donor_frame["probability"].to_numpy(dtype=np.float32),
            threshold=threshold,
        )
        rows.append({"donor_id": str(donor_id), **metrics, "n_lesions": int(donor_frame.shape[0])})
    return pd.DataFrame(rows)


def confusion_matrix_payload(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float) -> dict[str, object]:
    """Return a JSON-friendly confusion matrix payload."""
    pred = (np.asarray(y_prob) >= float(threshold)).astype(int)
    matrix = confusion_matrix(np.asarray(y_true, dtype=np.int64), pred, labels=[0, 1])
    return {"labels": [0, 1], "matrix": matrix.astype(int).tolist(), "threshold": float(threshold)}
