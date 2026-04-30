"""Evaluation metrics for StageBridge predictions.

Metrics:
- Wasserstein distance (W2): Optimal transport distance between predicted and true distributions
- MMD: Maximum Mean Discrepancy with RBF kernel
- Mean displacement: Average L2 distance of predictions from ground truth
- Stage accuracy: Nearest-neighbor stage classification accuracy
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KNeighborsClassifier

from stagebridge.contracts import LATENT_DIM


def compute_wasserstein(
    predicted: np.ndarray,
    target: np.ndarray,
    method: Literal["sliced", "sinkhorn"] = "sliced",
    n_projections: int = 100,
) -> float:
    """Compute Wasserstein distance between predicted and target distributions.

    Args:
        predicted: [N, D] predicted embeddings
        target: [M, D] target embeddings
        method: "sliced" (fast, approximate) or "sinkhorn" (exact but slower)
        n_projections: Number of random projections for sliced W2

    Returns:
        Wasserstein-2 distance
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)

    if method == "sliced":
        return _sliced_wasserstein(predicted, target, n_projections)
    else:
        return _sinkhorn_wasserstein(predicted, target)


def _sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 100,
) -> float:
    """Sliced Wasserstein distance (fast approximation)."""
    d = x.shape[1]
    rng = np.random.default_rng(42)

    distances = []
    for _ in range(n_projections):
        theta = rng.standard_normal(d)
        theta /= np.linalg.norm(theta)

        x_proj = x @ theta
        y_proj = y @ theta

        x_sorted = np.sort(x_proj)
        y_sorted = np.sort(y_proj)

        # Interpolate to same size if needed
        n = min(len(x_sorted), len(y_sorted))
        x_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x_sorted)), x_sorted)
        y_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y_sorted)), y_sorted)

        distances.append(np.mean((x_interp - y_interp) ** 2))

    return np.sqrt(np.mean(distances))


def _sinkhorn_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    max_iters: int = 100,
) -> float:
    """Sinkhorn Wasserstein distance."""
    n, m = len(x), len(y)
    C = cdist(x, y, metric="sqeuclidean")

    K = np.exp(-C / epsilon)
    u = np.ones(n) / n
    v = np.ones(m) / m

    a = np.ones(n) / n
    b = np.ones(m) / m

    for _ in range(max_iters):
        u = a / (K @ v + 1e-10)
        v = b / (K.T @ u + 1e-10)

    pi = np.diag(u) @ K @ np.diag(v)
    return np.sqrt(np.sum(pi * C))


def compute_mmd(
    predicted: np.ndarray,
    target: np.ndarray,
    kernel: Literal["rbf", "linear"] = "rbf",
    bandwidth: float | None = None,
) -> float:
    """Compute Maximum Mean Discrepancy.

    Args:
        predicted: [N, D] predicted embeddings
        target: [M, D] target embeddings
        kernel: "rbf" or "linear"
        bandwidth: RBF bandwidth (None = median heuristic)

    Returns:
        MMD^2 value
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if kernel == "linear":
        return _linear_mmd(predicted, target)
    else:
        return _rbf_mmd(predicted, target, bandwidth)


def _linear_mmd(x: np.ndarray, y: np.ndarray) -> float:
    """Linear kernel MMD."""
    xx = np.mean(x @ x.T)
    yy = np.mean(y @ y.T)
    xy = np.mean(x @ y.T)
    return xx + yy - 2 * xy


def _rbf_mmd(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float | None = None,
) -> float:
    """RBF kernel MMD with median heuristic for bandwidth."""
    if bandwidth is None:
        combined = np.vstack([x, y])
        dists = cdist(combined, combined, metric="sqeuclidean")
        bandwidth = np.median(dists[dists > 0])
        bandwidth = max(bandwidth, 1e-6)

    def rbf_kernel(a, b):
        dists = cdist(a, b, metric="sqeuclidean")
        return np.exp(-dists / (2 * bandwidth))

    kxx = rbf_kernel(x, x)
    kyy = rbf_kernel(y, y)
    kxy = rbf_kernel(x, y)

    n, m = len(x), len(y)

    # Unbiased estimator
    mmd = (np.sum(kxx) - np.trace(kxx)) / (n * (n - 1))
    mmd += (np.sum(kyy) - np.trace(kyy)) / (m * (m - 1))
    mmd -= 2 * np.mean(kxy)

    return max(0, mmd)


def compute_displacement(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Compute displacement statistics between paired predictions and targets.

    Args:
        predicted: [N, D] predicted embeddings
        target: [N, D] target embeddings (must be same length, paired)

    Returns:
        Dict with mean, std, median, max displacement
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if len(predicted) != len(target):
        raise ValueError(f"Length mismatch: {len(predicted)} vs {len(target)}")

    displacements = np.linalg.norm(predicted - target, axis=1)

    return {
        "mean_displacement": float(np.mean(displacements)),
        "std_displacement": float(np.std(displacements)),
        "median_displacement": float(np.median(displacements)),
        "max_displacement": float(np.max(displacements)),
    }


def compute_stage_accuracy(
    predicted: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_stages: np.ndarray,
    k: int = 5,
) -> dict[str, float]:
    """Compute stage classification accuracy via k-NN.

    Args:
        predicted: [N, D] predicted embeddings
        reference_embeddings: [M, D] reference set with known stages
        reference_stages: [M] stage labels for reference
        k: Number of neighbors for k-NN

    Returns:
        Dict with accuracy, per-class accuracy
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference_embeddings = np.asarray(reference_embeddings, dtype=np.float64)
    reference_stages = np.asarray(reference_stages)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(reference_embeddings, reference_stages)

    predicted_stages = knn.predict(predicted)

    # Overall accuracy
    accuracy = float(np.mean(predicted_stages == reference_stages[:len(predicted)]))

    # Per-class accuracy
    unique_stages = np.unique(reference_stages)
    per_class = {}
    for stage in unique_stages:
        mask = reference_stages[:len(predicted)] == stage
        if mask.sum() > 0:
            per_class[str(stage)] = float(np.mean(predicted_stages[mask] == stage))

    return {
        "stage_accuracy": accuracy,
        "per_class_accuracy": per_class,
    }


def evaluate_predictions(
    predicted: np.ndarray,
    target: np.ndarray,
    reference_embeddings: np.ndarray | None = None,
    reference_stages: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        predicted: [N, D] predicted embeddings
        target: [N, D] target embeddings (paired with predicted)
        reference_embeddings: Optional [M, D] for stage accuracy
        reference_stages: Optional [M] stage labels

    Returns:
        Dict with all metrics
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if predicted.shape[1] != LATENT_DIM:
        raise ValueError(f"Expected dim {LATENT_DIM}, got {predicted.shape[1]}")

    metrics = {}

    # Distribution metrics
    metrics["wasserstein_distance"] = compute_wasserstein(predicted, target)
    metrics["mmd"] = compute_mmd(predicted, target)

    # Displacement metrics
    disp = compute_displacement(predicted, target)
    metrics.update(disp)

    # Stage accuracy (if reference provided)
    if reference_embeddings is not None and reference_stages is not None:
        stage_metrics = compute_stage_accuracy(
            predicted, reference_embeddings, reference_stages
        )
        metrics.update(stage_metrics)

    return metrics


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    import pandas as pd

    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-idx", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_parquet(args.predictions)
    ref_df = pd.read_parquet(args.reference)

    predicted = np.array(pred_df["predicted_z"].tolist())
    target = ref_df["receiver_z"].values[:len(predicted)]
    target = np.array([np.array(x) for x in target])

    metrics = evaluate_predictions(predicted, target)

    with open(args.output_dir / "evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation saved: {metrics}")
