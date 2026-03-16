"""
Caching utilities for expensive plot computations

Provides LRU caching for dimensionality reduction algorithms to avoid
redundant computation when generating multiple plots from same data.
"""

import hashlib
import numpy as np
from functools import lru_cache
from typing import Tuple


def hash_array(arr: np.ndarray) -> str:
    """Fast hash for numpy arrays using md5 on bytes"""
    return hashlib.md5(arr.tobytes()).hexdigest()


@lru_cache(maxsize=8)
def compute_pca_cached(
    embeddings_hash: str,
    n_samples: int,
    n_features: int,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cached PCA computation

    Note: This is a cache key function. Actual computation happens in caller
    by reconstructing array from hash. Used to avoid redundant PCA calls.
    """
    # This function signature serves as cache key
    # Actual computation done externally
    pass


@lru_cache(maxsize=8)
def compute_tsne_cached(
    embeddings_hash: str,
    n_samples: int,
    n_features: int,
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
) -> str:
    """Cached t-SNE computation key"""
    pass


@lru_cache(maxsize=8)
def compute_umap_cached(
    embeddings_hash: str,
    n_samples: int,
    n_features: int,
    n_components: int = 2,
    random_state: int = 42,
) -> str:
    """Cached UMAP computation key"""
    pass


@lru_cache(maxsize=8)
def compute_phate_cached(
    embeddings_hash: str,
    n_samples: int,
    n_features: int,
    n_components: int = 2,
    random_state: int = 42,
) -> str:
    """Cached PHATE computation key"""
    pass


class DimensionalityReductionCache:
    """
    Cache manager for expensive dimensionality reduction computations

    Usage:
        cache = DimensionalityReductionCache()
        X_pca = cache.get_or_compute_pca(embeddings)
        X_tsne = cache.get_or_compute_tsne(embeddings)
    """

    def __init__(self):
        self._cache = {}

    def _make_key(self, method: str, embeddings: np.ndarray, **kwargs) -> str:
        """Generate cache key from method name, data hash, and parameters"""
        data_hash = hash_array(embeddings)
        param_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{method}_{data_hash}_{param_str}"

    def get_or_compute_pca(
        self, embeddings: np.ndarray, n_components: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get cached PCA or compute if not cached"""
        key = self._make_key("pca", embeddings, n_components=n_components)

        if key not in self._cache:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(embeddings)
            variance_ratio = pca.explained_variance_ratio_
            self._cache[key] = (X_reduced, variance_ratio)
            print("      [Cache MISS] Computed PCA")
        else:
            print("      [Cache HIT] Loaded PCA from cache")

        return self._cache[key]

    def get_or_compute_tsne(
        self, embeddings: np.ndarray, perplexity: int = 30, random_state: int = 42
    ) -> np.ndarray:
        """Get cached t-SNE or compute if not cached"""
        key = self._make_key("tsne", embeddings, perplexity=perplexity, random_state=random_state)

        if key not in self._cache:
            from sklearn.manifold import TSNE

            perplexity = min(perplexity, len(embeddings) // 4)
            tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
            X_reduced = tsne.fit_transform(embeddings)
            self._cache[key] = X_reduced
            print("      [Cache MISS] Computed t-SNE (~30s)")
        else:
            print("      [Cache HIT] Loaded t-SNE from cache")

        return self._cache[key]

    def get_or_compute_umap(self, embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
        """Get cached UMAP or compute if not cached"""
        key = self._make_key("umap", embeddings, random_state=random_state)

        if key not in self._cache:
            try:
                import umap

                reducer = umap.UMAP(random_state=random_state)
                X_reduced = reducer.fit_transform(embeddings)
                self._cache[key] = X_reduced
                print("      [Cache MISS] Computed UMAP (~20s)")
            except ImportError:
                print("      [SKIPPED] UMAP not available - pip install umap-learn")
                return None
        else:
            print("      [Cache HIT] Loaded UMAP from cache")

        return self._cache[key]

    def get_or_compute_phate(self, embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
        """Get cached PHATE or compute if not cached"""
        key = self._make_key("phate", embeddings, random_state=random_state)

        if key not in self._cache:
            try:
                import phate

                phate_op = phate.PHATE(random_state=random_state)
                X_reduced = phate_op.fit_transform(embeddings)
                self._cache[key] = X_reduced
                print("      [Cache MISS] Computed PHATE (~40s)")
            except ImportError:
                print("      [SKIPPED] PHATE not available - pip install phate")
                return None
        else:
            print("      [Cache HIT] Loaded PHATE from cache")

        return self._cache[key]

    def clear(self):
        """Clear all cached computations"""
        self._cache.clear()

    def size_mb(self) -> float:
        """Estimate cache size in MB"""
        total_bytes = sum(
            arr.nbytes
            if isinstance(arr, np.ndarray)
            else sum(a.nbytes for a in arr if isinstance(a, np.ndarray))
            for arr in self._cache.values()
        )
        return total_bytes / (1024 * 1024)


# Global cache instance
_global_cache = DimensionalityReductionCache()


def get_cache() -> DimensionalityReductionCache:
    """Get global cache instance"""
    return _global_cache


def clear_cache():
    """Clear global cache"""
    _global_cache.clear()
