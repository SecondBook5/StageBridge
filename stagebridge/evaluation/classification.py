"""Classification metrics and artifacts for communication-relay benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(slots=True, frozen=True)
class TemperatureScaler:
    """Scalar temperature fitted on validation logits."""

    temperature: float = 1.0

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) / max(float(self.temperature), 1e-6)


def fit_temperature_scaler(
    logits: np.ndarray, labels: np.ndarray, *, max_iter: int = 200
) -> TemperatureScaler:
    if np.asarray(logits).size == 0:
        return TemperatureScaler(temperature=1.0)
    if len(np.unique(np.asarray(labels))) < 2:
        return TemperatureScaler(temperature=1.0)
    logits_t = torch.tensor(np.asarray(logits, dtype=np.float32))
    labels_t = torch.tensor(np.asarray(labels, dtype=np.float32))
    temperature = torch.nn.Parameter(torch.ones((), dtype=torch.float32))
    optimizer = torch.optim.LBFGS(
        [temperature], lr=0.05, max_iter=int(max_iter), line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = logits_t / temperature.clamp_min(1e-3)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return TemperatureScaler(temperature=float(temperature.detach().clamp_min(1e-3).item()))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:], strict=False):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        acc = truth[mask].mean()
        conf = probs[mask].mean()
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def calibration_curve_table(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> pd.DataFrame:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, float]] = []
    for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:], strict=False)):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        rows.append(
            {
                "bin_id": float(idx),
                "bin_left": float(left),
                "bin_right": float(right),
                "count": float(mask.sum()),
                "mean_confidence": float(probs[mask].mean()) if np.any(mask) else float("nan"),
                "empirical_accuracy": float(truth[mask].mean()) if np.any(mask) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    best_threshold = 0.5
    best_pair = (-np.inf, -np.inf)
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (probs >= threshold).astype(np.int64)
        tn = float(np.sum((truth == 0) & (pred == 0)))
        fp = float(np.sum((truth == 0) & (pred == 1)))
        fn = float(np.sum((truth == 1) & (pred == 0)))
        tp = float(np.sum((truth == 1) & (pred == 1)))
        tpr = tp / max(tp + fn, 1.0)
        tnr = tn / max(tn + fp, 1.0)
        bal = 0.5 * (tpr + tnr)
        precision = tp / max(tp + fp, 1.0)
        recall = tpr
        f1 = (
            0.0
            if (precision + recall) <= 0.0
            else (2.0 * precision * recall) / (precision + recall)
        )
        pair = (float(bal), float(f1))
        if pair > best_pair:
            best_pair = pair
            best_threshold = float(threshold)
    return best_threshold


def binary_classification_metrics(
    probabilities: np.ndarray, labels: np.ndarray, *, threshold: float
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    pred = (probs >= float(threshold)).astype(np.int64)
    auroc = float("nan")
    auprc = float("nan")
    if len(np.unique(truth)) > 1:
        auroc = float(roc_auc_score(truth, probs))
        auprc = float(average_precision_score(truth, probs))
    tn = float(np.sum((truth == 0) & (pred == 0)))
    fp = float(np.sum((truth == 0) & (pred == 1)))
    fn = float(np.sum((truth == 1) & (pred == 0)))
    tp = float(np.sum((truth == 1) & (pred == 1)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    macro_f1_neg = (
        0.0
        if (specificity + (tn / max(tn + fn, 1.0))) <= 0.0
        else (2.0 * specificity * (tn / max(tn + fn, 1.0)))
        / max(specificity + (tn / max(tn + fn, 1.0)), 1e-12)
    )
    macro_f1_pos = (
        0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    )
    return {
        "auroc": auroc,
        "auprc": auprc,
        "macro_f1": float(0.5 * (macro_f1_neg + macro_f1_pos)),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float((pred == truth).mean()),
        "ece": expected_calibration_error(probs, truth),
        "threshold": float(threshold),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def curve_tables(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import precision_recall_curve, roc_curve

    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if len(np.unique(truth)) < 2:
        roc = pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "threshold": [1.0, 0.0]})
        pr = pd.DataFrame(
            {
                "precision": [truth.mean(), truth.mean()],
                "recall": [1.0, 0.0],
                "threshold": [1.0, 0.0],
            }
        )
        return roc, pr
    fpr, tpr, roc_thresholds = roc_curve(truth, probs)
    precision, recall, pr_thresholds = precision_recall_curve(truth, probs)
    roc = pd.DataFrame(
        {
            "fpr": fpr.astype(float),
            "tpr": tpr.astype(float),
            "threshold": roc_thresholds.astype(float),
        }
    )
    pr = pd.DataFrame(
        {
            "precision": precision.astype(float),
            "recall": recall.astype(float),
            "threshold": np.pad(
                pr_thresholds.astype(float),
                (0, max(0, precision.shape[0] - pr_thresholds.shape[0])),
                constant_values=np.nan,
            ),
        }
    )
    return roc, pr


def aggregate_predictions(
    query_logits: np.ndarray,
    bag_index: np.ndarray,
    sample_ids: list[str],
    donor_ids: list[str],
    labels: np.ndarray,
    *,
    scaler: TemperatureScaler | None = None,
) -> pd.DataFrame:
    logits = np.asarray(query_logits, dtype=np.float64)
    groups = np.asarray(bag_index, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    scaler = scaler or TemperatureScaler(temperature=1.0)
    scaled = scaler.apply(logits)
    probs = 1.0 / (1.0 + np.exp(-scaled))
    for bag_id, sample_id in enumerate(sample_ids):
        mask = groups == int(bag_id)
        sample_prob = float(probs[mask].mean()) if np.any(mask) else 0.5
        sample_logit = float(scaled[mask].mean()) if np.any(mask) else 0.0
        rows.append(
            {
                "bag_index": int(bag_id),
                "sample_id": str(sample_id),
                "donor_id": str(donor_ids[bag_id]),
                "label": float(labels[bag_id]),
                "bag_logit": sample_logit,
                "bag_probability": sample_prob,
                "num_queries": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "TemperatureScaler",
    "aggregate_predictions",
    "binary_classification_metrics",
    "calibration_curve_table",
    "choose_threshold",
    "curve_tables",
    "expected_calibration_error",
    "fit_temperature_scaler",
]
