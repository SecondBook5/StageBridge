"""Reference mapping confidence scoring.

Computes confidence scores for how well query cells map to each reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class ConfidenceResult:
    """Confidence scores for reference mapping.

    Attributes:
        hlca_confidence: [N] HLCA mapping confidence (0-1)
        luca_confidence: [N] LuCA mapping confidence (0-1)
        hlca_raw_distance: [N] Raw distance to HLCA neighbors
        luca_raw_distance: [N] Raw distance to LuCA neighbors
        combined_confidence: [N] Combined confidence score
    """

    hlca_confidence: np.ndarray
    luca_confidence: np.ndarray
    hlca_raw_distance: np.ndarray | None = None
    luca_raw_distance: np.ndarray | None = None
    combined_confidence: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        return len(self.hlca_confidence)


def compute_confidence(
    hlca_latent: np.ndarray,
    luca_latent: np.ndarray,
    hlca_reference: np.ndarray | None = None,
    luca_reference: np.ndarray | None = None,
    k: int = 30,
    method: str = "percentile",
) -> ConfidenceResult:
    """Compute confidence scores for reference mappings.

    Confidence is based on distance to k nearest neighbors in reference.
    Lower distance = higher confidence.

    Args:
        hlca_latent: [N, 30] Query HLCA embeddings
        luca_latent: [N, 10] Query LuCA embeddings
        hlca_reference: [M, 30] Reference HLCA embeddings (for k-NN)
        luca_reference: [M, 10] Reference LuCA embeddings (for k-NN)
        k: Number of neighbors for distance computation
        method: Confidence method ("percentile", "inverse", "softmax")

    Returns:
        ConfidenceResult with confidence scores
    """
    hlca_latent = np.asarray(hlca_latent, dtype=np.float32)
    luca_latent = np.asarray(luca_latent, dtype=np.float32)

    if hlca_reference is not None:
        hlca_dist = _compute_knn_distance(hlca_latent, hlca_reference, k)
        hlca_conf = _distance_to_confidence(hlca_dist, method)
    else:
        hlca_dist = _self_knn_distance(hlca_latent, k)
        hlca_conf = _distance_to_confidence(hlca_dist, method)

    if luca_reference is not None:
        luca_dist = _compute_knn_distance(luca_latent, luca_reference, k)
        luca_conf = _distance_to_confidence(luca_dist, method)
    else:
        luca_dist = _self_knn_distance(luca_latent, k)
        luca_conf = _distance_to_confidence(luca_dist, method)

    combined = (hlca_conf + luca_conf) / 2

    return ConfidenceResult(
        hlca_confidence=hlca_conf,
        luca_confidence=luca_conf,
        hlca_raw_distance=hlca_dist,
        luca_raw_distance=luca_dist,
        combined_confidence=combined,
    )


def _compute_knn_distance(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute mean distance to k nearest neighbors in reference.

    Args:
        query: [N, D] query embeddings
        reference: [M, D] reference embeddings
        k: number of neighbors

    Returns:
        [N] mean distance to k nearest neighbors
    """
    try:
        from sklearn.neighbors import NearestNeighbors

        k = min(k, len(reference))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(reference)
        distances, _ = nn.kneighbors(query)
        return distances.mean(axis=1).astype(np.float32)
    except ImportError:
        return _naive_knn_distance(query, reference, k)


def _self_knn_distance(
    embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute mean distance to k nearest neighbors within same set.

    Args:
        embeddings: [N, D] embeddings
        k: number of neighbors (excluding self)

    Returns:
        [N] mean distance to k nearest neighbors
    """
    try:
        from sklearn.neighbors import NearestNeighbors

        k = min(k + 1, len(embeddings))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        return distances[:, 1:].mean(axis=1).astype(np.float32)
    except ImportError:
        return _naive_knn_distance(embeddings, embeddings, k, exclude_self=True)


def _naive_knn_distance(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    exclude_self: bool = False,
) -> np.ndarray:
    """Naive k-NN without sklearn (for testing/fallback)."""
    n_query = len(query)
    distances = np.zeros(n_query, dtype=np.float32)

    for i in range(n_query):
        dists = np.linalg.norm(reference - query[i], axis=1)
        if exclude_self:
            dists[i] = np.inf
        topk = np.partition(dists, k)[:k]
        distances[i] = topk.mean()

    return distances


def _distance_to_confidence(
    distances: np.ndarray,
    method: str = "percentile",
) -> np.ndarray:
    """Convert distances to confidence scores.

    Args:
        distances: [N] raw distances
        method: Conversion method
            - "percentile": Rank-based (0-1 uniform)
            - "inverse": 1 / (1 + distance)
            - "softmax": softmax(-distance)

    Returns:
        [N] confidence scores in [0, 1]
    """
    if method == "percentile":
        ranks = np.argsort(np.argsort(distances))
        return 1.0 - (ranks / (len(distances) - 1 + 1e-8)).astype(np.float32)

    elif method == "inverse":
        return (1.0 / (1.0 + distances)).astype(np.float32)

    elif method == "softmax":
        exp_neg = np.exp(-distances - distances.max())
        return (exp_neg / exp_neg.sum()).astype(np.float32)

    else:
        raise ValueError(f"Unknown confidence method: {method}")


def filter_low_confidence(
    hlca_latent: np.ndarray,
    luca_latent: np.ndarray,
    confidence: ConfidenceResult,
    threshold: float = 0.2,
    which: str = "combined",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter cells with low mapping confidence.

    Args:
        hlca_latent: [N, 30] HLCA embeddings
        luca_latent: [N, 10] LuCA embeddings
        confidence: Confidence scores
        threshold: Minimum confidence to keep
        which: Which confidence to use ("hlca", "luca", "combined")

    Returns:
        (filtered_hlca, filtered_luca, keep_mask)
    """
    if which == "hlca":
        scores = confidence.hlca_confidence
    elif which == "luca":
        scores = confidence.luca_confidence
    elif which == "combined":
        scores = confidence.combined_confidence
        if scores is None:
            scores = (confidence.hlca_confidence + confidence.luca_confidence) / 2
    else:
        raise ValueError(f"Unknown confidence type: {which}")

    keep_mask = scores >= threshold

    return hlca_latent[keep_mask], luca_latent[keep_mask], keep_mask
