"""Classification metrics and statistics for EA-MIST lesion benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
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
