"""Evaluation metrics and benchmark helpers for StageBridge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from stagebridge.training.losses import sinkhorn_distance


@dataclass(slots=True)
class TransitionEvalResult:
    """Metrics for one transition evaluation run."""

    sinkhorn: float
    mmd_rbf: float
    classifier_auc: float
    jsd_composition: float
    rank_consistency: float



def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def mmd_rbf(x: Tensor, y: Tensor, gamma: float | None = None) -> Tensor:
    """Compute RBF-kernel MMD^2 between two sets."""
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    def _kernel(a: Tensor, b: Tensor) -> Tensor:
        d2 = torch.cdist(a, b, p=2.0).pow(2)
        return torch.exp(-gamma * d2)

    k_xx = _kernel(x, x)
    k_yy = _kernel(y, y)
    k_xy = _kernel(x, y)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def classifier_two_sample_auc(
    x: Tensor,
    y: Tensor,
    max_points: int = 2000,
    seed: int = 42,
) -> float:
    """AUC of classifying source-vs-target points with logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    x_np = _to_numpy(x)
    y_np = _to_numpy(y)

    if x_np.shape[0] > max_points:
        idx = rng.choice(x_np.shape[0], size=max_points, replace=False)
        x_np = x_np[idx]
    if y_np.shape[0] > max_points:
        idx = rng.choice(y_np.shape[0], size=max_points, replace=False)
        y_np = y_np[idx]

    X = np.concatenate([x_np, y_np], axis=0)
    z = np.concatenate([np.zeros(x_np.shape[0]), np.ones(y_np.shape[0])], axis=0)

    X_tr, X_te, z_tr, z_te = train_test_split(X, z, test_size=0.3, random_state=seed, stratify=z)
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_tr, z_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(z_te, proba))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def composition_jsd(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Compute JSD between predicted and true label compositions."""
    labels = sorted(set(labels_pred.tolist()) | set(labels_true.tolist()))
    idx = {k: i for i, k in enumerate(labels)}
    p = np.zeros(len(labels), dtype=np.float64)
    q = np.zeros(len(labels), dtype=np.float64)

    for label in labels_pred:
        p[idx[label]] += 1
    for label in labels_true:
        q[idx[label]] += 1

    return js_divergence(p, q)


def stage_progression_rank_consistency(
    stage_centroids: dict[int, np.ndarray],
) -> float:
    """Measure monotonic increase in centroid distance from stage 0."""
    from scipy.stats import spearmanr

    if 0 not in stage_centroids:
        return float("nan")

    base = stage_centroids[0]
    stage_ids = sorted(stage_centroids)
    dists = [np.linalg.norm(stage_centroids[s] - base) for s in stage_ids]
    corr, _ = spearmanr(stage_ids, dists)
    return float(corr)


def predict_next_stage(
    model: Any,
    x_src: Tensor,
    stage_src: int,
    stage_tgt: int,
    num_steps: int = 8,
) -> Tensor:
    """Generate predicted next-stage samples by Euler integration."""
    with torch.no_grad():
        c_s = model.forward_set_context(x_src)
        stage_pair = model.encode_stage_pair_tensor(
            stage_src=stage_src,
            stage_tgt=stage_tgt,
            n=x_src.shape[0],
            device=x_src.device,
        )
        x_pred = model.integrate_euler(x0=x_src, c_s=c_s, stage_pair_id=stage_pair, num_steps=num_steps)
    return x_pred


def evaluate_transition(
    model: Any,
    x_src: Tensor,
    x_tgt: Tensor,
    stage_src: int,
    stage_tgt: int,
    num_steps: int = 8,
    ot_epsilon: float = 0.05,
    sinkhorn_iters: int = 80,
) -> TransitionEvalResult:
    """Compute main benchmark metrics for a transition."""
    x_pred = predict_next_stage(
        model=model,
        x_src=x_src,
        stage_src=stage_src,
        stage_tgt=stage_tgt,
        num_steps=num_steps,
    )

    n = min(x_pred.shape[0], x_tgt.shape[0])
    x_pred = x_pred[:n]
    x_tgt = x_tgt[:n]

    sink = float(
        sinkhorn_distance(
            x_src=x_pred,
            x_tgt=x_tgt,
            epsilon=ot_epsilon,
            n_iters=sinkhorn_iters,
        ).item()
    )
    mmd = float(mmd_rbf(x_pred, x_tgt).item())
    auc = classifier_two_sample_auc(x_pred, x_tgt)

    # Unsupervised fallback: compare k-means cluster composition.
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=min(10, n), random_state=42, batch_size=256)
    labels_true = km.fit_predict(_to_numpy(x_tgt))
    labels_pred = km.predict(_to_numpy(x_pred))
    jsd = composition_jsd(labels_pred=labels_pred, labels_true=labels_true)

    # Rank consistency on centroids between src, pred, tgt.
    centroids = {
        int(stage_src): _to_numpy(x_src).mean(axis=0),
        int(stage_tgt): _to_numpy(x_tgt).mean(axis=0),
        int(stage_tgt + 10): _to_numpy(x_pred).mean(axis=0),
    }
    rank = stage_progression_rank_consistency(stage_centroids=centroids)

    return TransitionEvalResult(
        sinkhorn=sink,
        mmd_rbf=mmd,
        classifier_auc=auc,
        jsd_composition=jsd,
        rank_consistency=rank,
    )


def context_generalization_delta(train_metric: float, test_metric: float) -> float:
    """Compute held-out context delta (lower is better for distances)."""
    return float(test_metric - train_metric)


def summarize_fold_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-fold metrics with mean/std summary keys."""
    if not metrics:
        return {}
    keys = sorted(metrics[0])
    out: dict[str, float] = {}
    for key in keys:
        vals = np.array([m[key] for m in metrics], dtype=np.float64)
        out[f"{key}_mean"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))
    return out
