"""
Evaluation metrics for expression-aware semi-synthetic benchmarks.

Three core evaluation tasks for niche interaction recovery:
1. DE gene prediction - can the model recover interaction-induced gene changes?
2. Sender cell prediction - can the model identify which neighbors are true senders?
3. Receiver cell prediction - can the model identify interacting receivers?

All metrics use AUPRC (Area Under Precision-Recall Curve) for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class DEGenePredictionMetrics:
    """Metrics for DE gene prediction task."""

    auprc: float = 0.0
    auroc: float = 0.0
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    n_true_de: int = 0
    n_predicted_de: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "auprc": self.auprc,
            "auroc": self.auroc,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "n_true_de": self.n_true_de,
            "n_predicted_de": self.n_predicted_de,
        }


@dataclass
class SenderPredictionMetrics:
    """Metrics for sender cell prediction task."""

    auprc: float = 0.0
    auroc: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    n_true_senders: int = 0
    n_predicted_senders: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "auprc": self.auprc,
            "auroc": self.auroc,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "n_true_senders": self.n_true_senders,
            "n_predicted_senders": self.n_predicted_senders,
        }


@dataclass
class ReceiverPredictionMetrics:
    """Metrics for receiver cell prediction task."""

    auprc: float = 0.0
    auroc: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    n_true_interacting: int = 0
    n_predicted_interacting: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "auprc": self.auprc,
            "auroc": self.auroc,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "n_true_interacting": self.n_true_interacting,
            "n_predicted_interacting": self.n_predicted_interacting,
        }


@dataclass
class InteractionEvaluationReport:
    """Complete evaluation report for expression-aware benchmark."""

    de_gene_metrics: dict[str, DEGenePredictionMetrics] = field(default_factory=dict)
    sender_metrics: dict[str, SenderPredictionMetrics] = field(default_factory=dict)
    receiver_metrics: dict[str, ReceiverPredictionMetrics] = field(default_factory=dict)
    aggregate_auprc: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "de_gene_metrics": {k: v.to_dict() for k, v in self.de_gene_metrics.items()},
            "sender_metrics": {k: v.to_dict() for k, v in self.sender_metrics.items()},
            "receiver_metrics": {k: v.to_dict() for k, v in self.receiver_metrics.items()},
            "aggregate_auprc": self.aggregate_auprc,
        }


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_score))
    except ImportError:
        # Fallback implementation
        return _compute_auprc_manual(y_true, y_score)


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        return 0.0


def _compute_auprc_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Manual AUPRC computation without sklearn."""
    # Sort by score descending
    order = np.argsort(-y_score)
    y_true = y_true[order]

    # Compute precision at each threshold
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0

    tp_cumsum = np.cumsum(y_true)
    precision = tp_cumsum / np.arange(1, len(y_true) + 1)
    recall = tp_cumsum / n_pos

    # Compute area under PR curve
    # Add (0, 1) point
    recall = np.concatenate([[0], recall])
    precision = np.concatenate([[1], precision])

    # Compute area using trapezoidal rule
    auprc = np.sum(np.diff(recall) * precision[1:])
    return float(auprc)


def evaluate_de_gene_prediction(
    predicted_scores: dict[str, float],
    true_de_genes: list[str],
    all_genes: list[str],
    k_values: list[int] | None = None,
) -> DEGenePredictionMetrics:
    """Evaluate DE gene prediction.

    Args:
        predicted_scores: Gene -> importance score from model
        true_de_genes: Ground truth DE genes (up or downregulated)
        all_genes: All gene names
        k_values: K values for precision@k, recall@k

    Returns:
        DEGenePredictionMetrics
    """
    if k_values is None:
        k_values = [10, 25, 50, 100]

    # Convert to arrays
    n_genes = len(all_genes)
    y_true = np.array([1 if g in true_de_genes else 0 for g in all_genes])
    y_score = np.array([predicted_scores.get(g, 0.0) for g in all_genes])

    # Handle edge cases
    if y_true.sum() == 0:
        log.warning("No true DE genes found")
        return DEGenePredictionMetrics(n_true_de=0)

    if np.std(y_score) == 0:
        log.warning("All predicted scores are identical")
        return DEGenePredictionMetrics(n_true_de=int(y_true.sum()))

    # Compute metrics
    metrics = DEGenePredictionMetrics(
        auprc=compute_auprc(y_true, y_score),
        auroc=compute_auroc(y_true, y_score),
        n_true_de=int(y_true.sum()),
    )

    # Precision@k and Recall@k
    order = np.argsort(-y_score)
    for k in k_values:
        if k > n_genes:
            continue
        top_k = order[:k]
        tp_at_k = y_true[top_k].sum()
        metrics.precision_at_k[k] = float(tp_at_k / k)
        metrics.recall_at_k[k] = float(tp_at_k / max(1, y_true.sum()))

    # Count predicted DE genes (score > threshold)
    threshold = np.percentile(y_score, 95)  # Top 5%
    metrics.n_predicted_de = int((y_score > threshold).sum())

    return metrics


def evaluate_sender_prediction(
    predicted_sender_scores: np.ndarray,
    true_sender_labels: np.ndarray,
    threshold: float = 0.5,
) -> SenderPredictionMetrics:
    """Evaluate sender cell prediction.

    Args:
        predicted_sender_scores: Model's prediction of sender probability for each neighbor
        true_sender_labels: Ground truth (1 if sender within radius, 0 otherwise)
        threshold: Threshold for binary prediction

    Returns:
        SenderPredictionMetrics
    """
    # Handle edge cases
    if len(predicted_sender_scores) == 0:
        return SenderPredictionMetrics()

    if true_sender_labels.sum() == 0:
        return SenderPredictionMetrics(n_true_senders=0)

    # Binary predictions
    y_pred = (predicted_sender_scores > threshold).astype(int)
    y_true = true_sender_labels.astype(int)

    # Compute metrics
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return SenderPredictionMetrics(
        auprc=compute_auprc(y_true, predicted_sender_scores),
        auroc=compute_auroc(y_true, predicted_sender_scores),
        accuracy=float((y_pred == y_true).mean()),
        f1=float(f1),
        n_true_senders=int(y_true.sum()),
        n_predicted_senders=int(y_pred.sum()),
    )


def evaluate_receiver_prediction(
    predicted_interacting_scores: np.ndarray,
    true_interacting_labels: np.ndarray,
    threshold: float = 0.5,
) -> ReceiverPredictionMetrics:
    """Evaluate receiver cell prediction.

    Args:
        predicted_interacting_scores: Model's prediction of interacting probability
        true_interacting_labels: Ground truth (1 if receiver is interacting, 0 otherwise)
        threshold: Threshold for binary prediction

    Returns:
        ReceiverPredictionMetrics
    """
    # Handle edge cases
    if len(predicted_interacting_scores) == 0:
        return ReceiverPredictionMetrics()

    if true_interacting_labels.sum() == 0:
        return ReceiverPredictionMetrics(n_true_interacting=0)

    # Binary predictions
    y_pred = (predicted_interacting_scores > threshold).astype(int)
    y_true = true_interacting_labels.astype(int)

    # Compute metrics
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return ReceiverPredictionMetrics(
        auprc=compute_auprc(y_true, predicted_interacting_scores),
        auroc=compute_auroc(y_true, predicted_interacting_scores),
        accuracy=float((y_pred == y_true).mean()),
        f1=float(f1),
        n_true_interacting=int(y_true.sum()),
        n_predicted_interacting=int(y_pred.sum()),
    )


def evaluate_length_scale_recovery(
    predicted_length_scales: dict[str, float],
    true_length_scales: dict[str, float],
) -> dict[str, Any]:
    """Evaluate recovery of interaction length scales.

    Key capability for interpretable niche models: learning interaction radii.

    Args:
        predicted_length_scales: Interaction name -> predicted radius
        true_length_scales: Interaction name -> true radius

    Returns:
        Dict with correlation and per-interaction errors
    """
    if not predicted_length_scales or not true_length_scales:
        return {"correlation": 0.0, "mae": float("inf")}

    common_keys = set(predicted_length_scales.keys()) & set(true_length_scales.keys())
    if not common_keys:
        return {"correlation": 0.0, "mae": float("inf")}

    pred = np.array([predicted_length_scales[k] for k in common_keys])
    true = np.array([true_length_scales[k] for k in common_keys])

    # Correlation
    if len(pred) >= 2:
        corr, _ = stats.pearsonr(pred, true)
    else:
        corr = 1.0 if np.allclose(pred, true) else 0.0

    # MAE
    mae = np.abs(pred - true).mean()

    # Per-interaction
    per_interaction = {
        k: {
            "predicted": float(predicted_length_scales[k]),
            "true": float(true_length_scales[k]),
            "error": float(abs(predicted_length_scales[k] - true_length_scales[k])),
        }
        for k in common_keys
    }

    return {
        "correlation": float(corr),
        "mae": float(mae),
        "per_interaction": per_interaction,
    }


def evaluate_interaction_benchmark(
    model_outputs: dict[str, Any],
    ground_truth: dict[str, Any],
    gene_names: list[str],
) -> InteractionEvaluationReport:
    """Full evaluation against expression-aware benchmark.

    Args:
        model_outputs: Dict containing:
            - gene_importance: Dict[celltype, Dict[gene, score]]
            - sender_scores: Dict[interaction, np.ndarray]
            - receiver_scores: Dict[interaction, np.ndarray]
            - length_scales: Dict[interaction, float] (optional)
        ground_truth: Dict containing:
            - de_genes: Dict[celltype, list[str]]
            - sender_labels: Dict[interaction, np.ndarray]
            - receiver_labels: Dict[interaction, np.ndarray]
            - length_scales: Dict[interaction, float]
        gene_names: List of all gene names

    Returns:
        InteractionEvaluationReport
    """
    report = InteractionEvaluationReport()
    auprc_values = []

    # 1. DE gene prediction
    if "gene_importance" in model_outputs and "de_genes" in ground_truth:
        for celltype, true_de in ground_truth["de_genes"].items():
            if celltype in model_outputs["gene_importance"]:
                pred_scores = model_outputs["gene_importance"][celltype]
                metrics = evaluate_de_gene_prediction(pred_scores, true_de, gene_names)
                report.de_gene_metrics[celltype] = metrics
                auprc_values.append(metrics.auprc)

    # 2. Sender prediction
    if "sender_scores" in model_outputs and "sender_labels" in ground_truth:
        for interaction, true_labels in ground_truth["sender_labels"].items():
            if interaction in model_outputs["sender_scores"]:
                pred_scores = model_outputs["sender_scores"][interaction]
                metrics = evaluate_sender_prediction(pred_scores, true_labels)
                report.sender_metrics[interaction] = metrics
                auprc_values.append(metrics.auprc)

    # 3. Receiver prediction
    if "receiver_scores" in model_outputs and "receiver_labels" in ground_truth:
        for interaction, true_labels in ground_truth["receiver_labels"].items():
            if interaction in model_outputs["receiver_scores"]:
                pred_scores = model_outputs["receiver_scores"][interaction]
                metrics = evaluate_receiver_prediction(pred_scores, true_labels)
                report.receiver_metrics[interaction] = metrics
                auprc_values.append(metrics.auprc)

    # Aggregate AUPRC
    if auprc_values:
        report.aggregate_auprc = float(np.mean(auprc_values))

    return report


@dataclass
class PathwayPredictionMetrics:
    """Metrics for pathway activity prediction task."""

    correlation: float = 0.0  # Pearson correlation with ground truth
    mse: float = 0.0  # Mean squared error
    mae: float = 0.0  # Mean absolute error
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation": self.correlation,
            "mse": self.mse,
            "mae": self.mae,
            "n_samples": self.n_samples,
        }


@dataclass
class StageEffectMetrics:
    """Metrics for stage-modulated effect recovery."""

    correlation: float = 0.0  # Correlation between predicted and true stage effects
    rank_correlation: float = 0.0  # Spearman rank correlation
    effect_direction_accuracy: float = 0.0  # Fraction of stages with correct effect direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation": self.correlation,
            "rank_correlation": self.rank_correlation,
            "effect_direction_accuracy": self.effect_direction_accuracy,
        }


def evaluate_pathway_prediction(
    predicted_scores: np.ndarray,
    true_scores: np.ndarray,
) -> PathwayPredictionMetrics:
    """Evaluate pathway activity prediction.

    Args:
        predicted_scores: (n_cells,) model's pathway activity predictions
        true_scores: (n_cells,) ground truth pathway scores

    Returns:
        PathwayPredictionMetrics
    """
    if len(predicted_scores) == 0 or len(true_scores) == 0:
        return PathwayPredictionMetrics()

    if len(predicted_scores) != len(true_scores):
        return PathwayPredictionMetrics()

    # Correlation
    if np.std(predicted_scores) > 0 and np.std(true_scores) > 0:
        corr, _ = stats.pearsonr(predicted_scores, true_scores)
    else:
        corr = 0.0

    # MSE and MAE
    mse = np.mean((predicted_scores - true_scores) ** 2)
    mae = np.mean(np.abs(predicted_scores - true_scores))

    return PathwayPredictionMetrics(
        correlation=float(corr),
        mse=float(mse),
        mae=float(mae),
        n_samples=len(predicted_scores),
    )


def evaluate_stage_effect_recovery(
    predicted_effects: dict[str, float],
    true_effects: dict[str, float],
    stages: list[str],
) -> StageEffectMetrics:
    """Evaluate recovery of stage-modulated effects.

    Args:
        predicted_effects: stage -> predicted effect size
        true_effects: stage -> true effect size
        stages: list of stages in order

    Returns:
        StageEffectMetrics
    """
    common_stages = [s for s in stages if s in predicted_effects and s in true_effects]

    if len(common_stages) < 2:
        return StageEffectMetrics()

    pred = np.array([predicted_effects[s] for s in common_stages])
    true = np.array([true_effects[s] for s in common_stages])

    # Pearson correlation
    if np.std(pred) > 0 and np.std(true) > 0:
        corr, _ = stats.pearsonr(pred, true)
        rank_corr, _ = stats.spearmanr(pred, true)
    else:
        corr, rank_corr = 0.0, 0.0

    # Effect direction accuracy (is effect increasing/decreasing correctly?)
    pred_diffs = np.diff(pred)
    true_diffs = np.diff(true)
    direction_matches = np.sign(pred_diffs) == np.sign(true_diffs)
    direction_acc = np.mean(direction_matches)

    return StageEffectMetrics(
        correlation=float(corr),
        rank_correlation=float(rank_corr),
        effect_direction_accuracy=float(direction_acc),
    )


def evaluate_pathway_benchmark(
    model_outputs: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate pathway and stage-effect predictions.

    Args:
        model_outputs: Dict containing:
            - pathway_scores: Dict[pathway, np.ndarray] predicted pathway activities
            - stage_effects: Dict[interaction, Dict[stage, float]] predicted stage effects
        ground_truth: Dict containing:
            - pathway_scores: Dict[pathway, np.ndarray] true pathway activities
            - stage_effect_sizes: Dict[interaction, Dict[stage, float]]
            - stages: list[str]

    Returns:
        Dict with pathway and stage effect metrics
    """
    results = {
        "pathway_metrics": {},
        "stage_effect_metrics": {},
    }

    # Pathway prediction
    if "pathway_scores" in model_outputs and "pathway_scores" in ground_truth:
        for pathway, true_scores in ground_truth["pathway_scores"].items():
            if pathway in model_outputs["pathway_scores"]:
                pred_scores = model_outputs["pathway_scores"][pathway]
                metrics = evaluate_pathway_prediction(pred_scores, true_scores)
                results["pathway_metrics"][pathway] = metrics.to_dict()

    # Stage effect recovery
    if "stage_effects" in model_outputs and "stage_effect_sizes" in ground_truth:
        stages = ground_truth.get("stages", [])
        for interaction, true_effects in ground_truth["stage_effect_sizes"].items():
            if interaction in model_outputs["stage_effects"]:
                pred_effects = model_outputs["stage_effects"][interaction]
                metrics = evaluate_stage_effect_recovery(pred_effects, true_effects, stages)
                results["stage_effect_metrics"][interaction] = metrics.to_dict()

    return results
