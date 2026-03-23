"""
Evaluation metrics for semi-synthetic benchmark.

Computes recovery metrics for:
- Receiver state prediction
- Sender attribution
- Distance sensitivity
- Stage-aware behavior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ReceiverStateMetrics:
    """Metrics for receiver state recovery."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auroc: float = 0.0
    auprc: float = 0.0
    n_samples: int = 0


@dataclass
class SenderAttributionMetrics:
    """Metrics for sender attribution."""

    sender_group: str = ""
    attribution_correlation: float = 0.0
    attribution_auroc: float = 0.0
    radius_rank_correlation: float = 0.0


@dataclass
class DistanceSensitivityMetrics:
    """Metrics for distance sensitivity evaluation."""

    radii_tested: list[float] = field(default_factory=list)
    decay_correlation: float = 0.0
    radius_ordering_correct: bool = False
    effect_by_radius: dict[float, float] = field(default_factory=dict)


@dataclass
class BenchmarkMetrics:
    """Complete benchmark evaluation metrics."""

    receiver_state: ReceiverStateMetrics = field(default_factory=ReceiverStateMetrics)
    sender_attribution: list[SenderAttributionMetrics] = field(default_factory=list)
    distance_sensitivity: DistanceSensitivityMetrics = field(
        default_factory=DistanceSensitivityMetrics
    )
    stage_metrics: dict[str, ReceiverStateMetrics] = field(default_factory=dict)
    overall_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "receiver_state": {
                "accuracy": self.receiver_state.accuracy,
                "precision": self.receiver_state.precision,
                "recall": self.receiver_state.recall,
                "f1": self.receiver_state.f1,
                "auroc": self.receiver_state.auroc,
                "auprc": self.receiver_state.auprc,
                "n_samples": self.receiver_state.n_samples,
            },
            "sender_attribution": [
                {
                    "sender_group": s.sender_group,
                    "attribution_correlation": s.attribution_correlation,
                    "attribution_auroc": s.attribution_auroc,
                }
                for s in self.sender_attribution
            ],
            "distance_sensitivity": {
                "radii_tested": self.distance_sensitivity.radii_tested,
                "decay_correlation": self.distance_sensitivity.decay_correlation,
                "radius_ordering_correct": self.distance_sensitivity.radius_ordering_correct,
            },
            "stage_metrics": {
                stage: {"f1": m.f1, "auroc": m.auroc} for stage, m in self.stage_metrics.items()
            },
            "overall_score": self.overall_score,
        }


def evaluate_receiver_state_recovery(
    predictions: np.ndarray | pd.Series,
    ground_truth: np.ndarray | pd.Series,
    prediction_probs: np.ndarray | pd.Series | None = None,
) -> ReceiverStateMetrics:
    """Evaluate receiver state prediction.

    Args:
        predictions: Binary predictions (0/1 or False/True)
        ground_truth: Ground truth labels
        prediction_probs: Optional probability scores for AUROC/AUPRC

    Returns:
        ReceiverStateMetrics
    """
    predictions = np.asarray(predictions).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)

    n_samples = len(predictions)
    if n_samples == 0:
        return ReceiverStateMetrics()

    # Basic metrics
    tp = (predictions & ground_truth).sum()
    fp = (predictions & ~ground_truth).sum()
    fn = (~predictions & ground_truth).sum()
    tn = (~predictions & ~ground_truth).sum()

    accuracy = (tp + tn) / n_samples
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    # ROC/PR metrics
    auroc = 0.0
    auprc = 0.0

    if prediction_probs is not None:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            probs = np.asarray(prediction_probs)
            if len(np.unique(ground_truth)) > 1:
                auroc = roc_auc_score(ground_truth.astype(int), probs)
                auprc = average_precision_score(ground_truth.astype(int), probs)
        except Exception as e:
            log.warning("Could not compute AUROC/AUPRC: %s", e)

    return ReceiverStateMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auroc=float(auroc),
        auprc=float(auprc),
        n_samples=n_samples,
    )


def evaluate_sender_attribution(
    model_attribution: np.ndarray | pd.Series,
    ground_truth_counts: np.ndarray | pd.Series,
    sender_group: str,
) -> SenderAttributionMetrics:
    """Evaluate sender attribution accuracy.

    Args:
        model_attribution: Model's attribution scores for sender group
        ground_truth_counts: Ground truth sender counts
        sender_group: Name of the sender group

    Returns:
        SenderAttributionMetrics
    """
    model_attr = np.asarray(model_attribution).flatten()
    gt_counts = np.asarray(ground_truth_counts).flatten()

    if len(model_attr) != len(gt_counts):
        raise ValueError("Attribution and ground truth must have same length")

    # Correlation
    if np.std(model_attr) > 0 and np.std(gt_counts) > 0:
        correlation = np.corrcoef(model_attr, gt_counts)[0, 1]
    else:
        correlation = 0.0

    # AUROC for detecting presence of senders
    auroc = 0.0
    if len(np.unique(gt_counts > 0)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            auroc = roc_auc_score((gt_counts > 0).astype(int), model_attr)
        except Exception:
            pass

    return SenderAttributionMetrics(
        sender_group=sender_group,
        attribution_correlation=float(correlation) if not np.isnan(correlation) else 0.0,
        attribution_auroc=float(auroc),
    )


def evaluate_distance_sensitivity(
    model_effects: dict[float, np.ndarray],
    ground_truth_effects: dict[float, np.ndarray],
) -> DistanceSensitivityMetrics:
    """Evaluate distance sensitivity.

    Args:
        model_effects: Model effect predictions by radius {radius: effects}
        ground_truth_effects: Ground truth effects by radius

    Returns:
        DistanceSensitivityMetrics
    """
    radii = sorted(model_effects.keys())

    if not radii:
        return DistanceSensitivityMetrics()

    # Compute mean effect by radius
    model_means = [np.mean(model_effects[r]) for r in radii]
    gt_means = [np.mean(ground_truth_effects[r]) for r in radii]

    # Correlation of distance decay
    if len(radii) > 2 and np.std(model_means) > 0 and np.std(gt_means) > 0:
        decay_corr = np.corrcoef(model_means, gt_means)[0, 1]
    else:
        decay_corr = 0.0

    # Check radius ordering (smaller radius should have stronger effect)
    # Ground truth: effect should decay with radius
    ordering_correct = True
    for i in range(len(gt_means) - 1):
        if gt_means[i] < gt_means[i + 1]:
            # GT shows decay, check if model does too
            if model_means[i] < model_means[i + 1]:
                ordering_correct = False
                break

    return DistanceSensitivityMetrics(
        radii_tested=radii,
        decay_correlation=float(decay_corr) if not np.isnan(decay_corr) else 0.0,
        radius_ordering_correct=ordering_correct,
        effect_by_radius={r: float(model_means[i]) for i, r in enumerate(radii)},
    )


def compute_benchmark_metrics(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    sender_groups: list[str] | None = None,
    radii: list[float] | None = None,
) -> BenchmarkMetrics:
    """Compute complete benchmark evaluation metrics.

    Args:
        predictions: DataFrame with model predictions
        ground_truth: DataFrame with ground truth labels
        sender_groups: Optional list of sender groups for attribution
        radii: Optional list of radii for distance sensitivity

    Returns:
        BenchmarkMetrics
    """
    metrics = BenchmarkMetrics()

    # Receiver state recovery
    if "predicted_interacting" in predictions.columns and "is_interacting" in ground_truth.columns:
        pred_probs = predictions.get("predicted_prob", None)
        metrics.receiver_state = evaluate_receiver_state_recovery(
            predictions["predicted_interacting"],
            ground_truth["is_interacting"],
            pred_probs,
        )

    # Sender attribution
    if sender_groups:
        for sender in sender_groups:
            model_col = f"attribution_{sender}"
            gt_col = f"n_{sender}"

            if model_col in predictions.columns and gt_col in ground_truth.columns:
                attr_metrics = evaluate_sender_attribution(
                    predictions[model_col],
                    ground_truth[gt_col],
                    sender,
                )
                metrics.sender_attribution.append(attr_metrics)

    # Distance sensitivity
    if radii:
        model_effects = {}
        gt_effects = {}

        for radius in radii:
            model_col = f"effect_r{int(radius)}"
            gt_col = f"gt_effect_r{int(radius)}"

            if model_col in predictions.columns:
                model_effects[radius] = predictions[model_col].values
            if gt_col in ground_truth.columns:
                gt_effects[radius] = ground_truth[gt_col].values

        if model_effects and gt_effects:
            metrics.distance_sensitivity = evaluate_distance_sensitivity(model_effects, gt_effects)

    # Stage-specific metrics
    if "stage" in ground_truth.columns and "predicted_interacting" in predictions.columns:
        for stage in ground_truth["stage"].unique():
            stage_mask = ground_truth["stage"] == stage
            if stage_mask.sum() > 0:
                pred_probs = predictions.get("predicted_prob", None)
                stage_probs = pred_probs[stage_mask] if pred_probs is not None else None

                metrics.stage_metrics[str(stage)] = evaluate_receiver_state_recovery(
                    predictions.loc[stage_mask, "predicted_interacting"],
                    ground_truth.loc[stage_mask, "is_interacting"],
                    stage_probs,
                )

    # Overall score (weighted combination)
    score_components = []
    if metrics.receiver_state.auroc > 0:
        score_components.append(metrics.receiver_state.auroc)
    if metrics.receiver_state.f1 > 0:
        score_components.append(metrics.receiver_state.f1)
    if metrics.sender_attribution:
        avg_attr = np.mean([s.attribution_correlation for s in metrics.sender_attribution])
        if not np.isnan(avg_attr):
            score_components.append((avg_attr + 1) / 2)  # Scale to 0-1
    if metrics.distance_sensitivity.decay_correlation > 0:
        score_components.append((metrics.distance_sensitivity.decay_correlation + 1) / 2)

    if score_components:
        metrics.overall_score = float(np.mean(score_components))

    return metrics


def compute_latent_separation(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute latent space separation between interacting and non-interacting cells.

    Args:
        embeddings: Cell embeddings (n_cells, latent_dim)
        labels: Binary labels (n_cells,)

    Returns:
        Dictionary with separation metrics
    """
    labels = np.asarray(labels).astype(bool)
    pos_mask = labels
    neg_mask = ~labels

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return {"silhouette": 0.0, "centroid_distance": 0.0}

    pos_emb = embeddings[pos_mask]
    neg_emb = embeddings[neg_mask]

    # Centroid distance
    pos_centroid = pos_emb.mean(axis=0)
    neg_centroid = neg_emb.mean(axis=0)
    centroid_dist = np.linalg.norm(pos_centroid - neg_centroid)

    # Silhouette-like score
    try:
        from sklearn.metrics import silhouette_score

        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(embeddings, labels.astype(int))
        else:
            silhouette = 0.0
    except Exception:
        silhouette = 0.0

    return {
        "silhouette": float(silhouette),
        "centroid_distance": float(centroid_dist),
    }
