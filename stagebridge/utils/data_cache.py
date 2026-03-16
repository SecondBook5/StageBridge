"""
Global data cache for expensive loading operations

Provides singleton cache to avoid redundant parquet/CSV loading across
multiple scripts and pipeline stages. Particularly useful for:
- cells.parquet (loaded by visualization, analysis, training scripts)
- neighborhoods.parquet (loaded by multiple analysis steps)
- Training results CSVs (loaded by reporting scripts)

Usage:
    from stagebridge.utils.data_cache import get_data_cache

    cache = get_data_cache()
    cells_df = cache.read_parquet("data/processed/synthetic/cells.parquet")
    # Second call is instant
    cells_df = cache.read_parquet("data/processed/synthetic/cells.parquet")
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any


class DataCache:
    """Singleton cache for expensive data loading operations."""

    _instance: Optional['DataCache'] = None
    _cache: dict[str, pd.DataFrame] = {}
    _verbose: bool = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            cls._instance._verbose = True
        return cls._instance

    def read_parquet(self, path: Path | str, columns: list | None = None,
                     use_cache: bool = True, **kwargs) -> pd.DataFrame:
        """
        Read parquet with caching.

        Args:
            path: Path to parquet file
            columns: Optional list of columns to load (memory optimization)
            use_cache: Whether to use cache (default True)
            **kwargs: Additional arguments passed to pd.read_parquet

        Returns:
            DataFrame (cached or freshly loaded)
        """
        path = Path(path).resolve()
        cache_key = f"parquet:{path}"

        if columns:
            cache_key += f":cols:{','.join(sorted(columns))}"

        if use_cache and cache_key in self._cache:
            if self._verbose:
                df = self._cache[cache_key]
                print(f"  [Cache HIT] {path.name} ({df.shape[0]:,} rows × {df.shape[1]} cols)")
            return self._cache[cache_key]

        # Load from disk
        if columns:
            df = pd.read_parquet(path, columns=columns, **kwargs)
        else:
            df = pd.read_parquet(path, **kwargs)

        if self._verbose:
            size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"  [Cache MISS] {path.name} ({df.shape[0]:,} rows × {df.shape[1]} cols, {size_mb:.1f} MB)")

        if use_cache:
            self._cache[cache_key] = df

        return df

    def read_csv(self, path: Path | str, use_cache: bool = True, **kwargs) -> pd.DataFrame:
        """
        Read CSV with caching.

        Args:
            path: Path to CSV file
            use_cache: Whether to use cache (default True)
            **kwargs: Additional arguments passed to pd.read_csv

        Returns:
            DataFrame (cached or freshly loaded)
        """
        path = Path(path).resolve()
        cache_key = f"csv:{path}"

        if use_cache and cache_key in self._cache:
            if self._verbose:
                df = self._cache[cache_key]
                print(f"  [Cache HIT] {path.name} ({df.shape[0]:,} rows)")
            return self._cache[cache_key]

        # Load from disk
        df = pd.read_csv(path, **kwargs)

        if self._verbose:
            size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"  [Cache MISS] {path.name} ({df.shape[0]:,} rows, {size_mb:.1f} MB)")

        if use_cache:
            self._cache[cache_key] = df

        return df

    def clear(self):
        """Clear all cached data."""
        n_items = len(self._cache)
        size_mb = self.size_mb()
        self._cache.clear()
        if self._verbose:
            print(f"  [Cache CLEAR] Freed {n_items} items ({size_mb:.1f} MB)")

    def size_mb(self) -> float:
        """Estimate total cache size in MB."""
        total_bytes = sum(
            df.memory_usage(deep=True).sum()
            for df in self._cache.values()
        )
        return total_bytes / (1024 * 1024)

    def info(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            'n_items': len(self._cache),
            'size_mb': self.size_mb(),
            'keys': list(self._cache.keys()),
        }

    def set_verbose(self, verbose: bool):
        """Enable/disable verbose cache logging."""
        self._verbose = verbose


# Global cache instance
_global_cache = DataCache()


def get_data_cache() -> DataCache:
    """Get global data cache singleton."""
    return _global_cache


def clear_data_cache():
    """Clear global data cache."""
    _global_cache.clear()


def cache_info() -> dict[str, Any]:
    """Get global cache info."""
    return _global_cache.info()
