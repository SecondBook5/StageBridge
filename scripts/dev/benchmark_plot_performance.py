#!/usr/bin/env python
"""
Benchmark plot generation performance

Compare original vs optimized implementations to measure speedup.
"""

import sys
import time
import numpy as np
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_data(n_samples=1000, n_features=32):
    """Generate test embeddings and labels"""
    np.random.seed(42)

    # 4 clear clusters
    embeddings = []
    labels = []
    for i in range(4):
        cluster = np.random.randn(n_samples // 4, n_features)
        cluster += np.array([i * 3, i * 2] + [0] * (n_features - 2))
        embeddings.append(cluster)
        labels.extend([i] * (n_samples // 4))

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    return embeddings, labels


def benchmark_original_plots(embeddings, labels, output_dir):
    """Benchmark original implementation (no caching)"""
    from stagebridge.viz.individual_plots import (
        plot_pca_with_variance,
        plot_tsne,
        plot_umap,
        plot_phate,
    )

    times = {}

    # PCA
    t0 = time.time()
    plot_pca_with_variance(embeddings, labels, output_dir / "pca.png", dpi=150)
    times['pca'] = time.time() - t0

    # t-SNE
    t0 = time.time()
    plot_tsne(embeddings, labels, output_dir / "tsne.png", dpi=150)
    times['tsne'] = time.time() - t0

    # UMAP
    t0 = time.time()
    plot_umap(embeddings, labels, output_dir / "umap.png", dpi=150)
    times['umap'] = time.time() - t0

    # PHATE
    t0 = time.time()
    plot_phate(embeddings, labels, output_dir / "phate.png", dpi=150)
    times['phate'] = time.time() - t0

    return times


def benchmark_optimized_plots(embeddings, labels, output_dir):
    """Benchmark optimized implementation (with caching)"""
    from stagebridge.viz.individual_plots_optimized import (
        plot_pca_with_variance,
        plot_tsne,
        plot_umap,
        plot_phate,
    )
    from stagebridge.viz.plot_cache import clear_cache

    # First run (cold cache)
    clear_cache()
    times_cold = {}

    t0 = time.time()
    plot_pca_with_variance(embeddings, labels, output_dir / "pca_opt.png", dpi=150)
    times_cold['pca'] = time.time() - t0

    t0 = time.time()
    plot_tsne(embeddings, labels, output_dir / "tsne_opt.png", dpi=150)
    times_cold['tsne'] = time.time() - t0

    t0 = time.time()
    plot_umap(embeddings, labels, output_dir / "umap_opt.png", dpi=150)
    times_cold['umap'] = time.time() - t0

    t0 = time.time()
    plot_phate(embeddings, labels, output_dir / "phate_opt.png", dpi=150)
    times_cold['phate'] = time.time() - t0

    # Second run (warm cache - same data)
    times_warm = {}

    t0 = time.time()
    plot_pca_with_variance(embeddings, labels, output_dir / "pca_opt2.png", dpi=150)
    times_warm['pca'] = time.time() - t0

    t0 = time.time()
    plot_tsne(embeddings, labels, output_dir / "tsne_opt2.png", dpi=150)
    times_warm['tsne'] = time.time() - t0

    t0 = time.time()
    plot_umap(embeddings, labels, output_dir / "umap_opt2.png", dpi=150)
    times_warm['umap'] = time.time() - t0

    t0 = time.time()
    plot_phate(embeddings, labels, output_dir / "phate_opt2.png", dpi=150)
    times_warm['phate'] = time.time() - t0

    return times_cold, times_warm


def main():
    print("=" * 80)
    print("PLOT GENERATION PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Generate test data
    print("\nGenerating test data (1000 samples, 32 features)...")
    embeddings, labels = generate_test_data(n_samples=1000, n_features=32)
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Labels: {labels.shape}")

    # Create temp output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Benchmark original
        print("\n" + "=" * 80)
        print("ORIGINAL IMPLEMENTATION (no caching)")
        print("=" * 80)
        print("Running...")
        t0_total = time.time()
        original_times = benchmark_original_plots(embeddings, labels, output_dir)
        total_original = time.time() - t0_total

        print("\nResults:")
        for method, t in original_times.items():
            print(f"  {method.upper():8s}: {t:6.2f}s")
        print(f"  {'TOTAL':8s}: {total_original:6.2f}s")

        # Benchmark optimized
        print("\n" + "=" * 80)
        print("OPTIMIZED IMPLEMENTATION (with caching)")
        print("=" * 80)
        print("Running (cold cache)...")
        t0_total = time.time()
        optimized_cold, optimized_warm = benchmark_optimized_plots(embeddings, labels, output_dir)
        total_optimized = time.time() - t0_total

        print("\nCold cache results:")
        for method, t in optimized_cold.items():
            print(f"  {method.upper():8s}: {t:6.2f}s")

        print("\nWarm cache results (2nd run with same data):")
        for method, t in optimized_warm.items():
            speedup = original_times[method] / t if t > 0 else float('inf')
            print(f"  {method.upper():8s}: {t:6.2f}s  (speedup: {speedup:5.1f}×)")

        # Summary
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)

        total_warm = sum(optimized_warm.values())
        overall_speedup = total_original / total_warm if total_warm > 0 else float('inf')

        print(f"\nOriginal total:  {total_original:6.2f}s")
        print(f"Optimized cold:  {sum(optimized_cold.values()):6.2f}s")
        print(f"Optimized warm:  {total_warm:6.2f}s")
        print(f"\nOverall speedup (warm cache): {overall_speedup:5.1f}×")

        # Memory estimate
        print("\n" + "=" * 80)
        print("MEMORY ESTIMATE")
        print("=" * 80)
        from stagebridge.viz.plot_cache import get_cache
        cache = get_cache()
        cache_size_mb = cache.size_mb()
        print(f"Cache size: {cache_size_mb:.1f} MB")

        # Calculate memory saved
        embedding_size_mb = embeddings.nbytes / (1024 * 1024)
        print(f"Embedding size: {embedding_size_mb:.1f} MB")
        print(f"Memory efficiency: {cache_size_mb / (embedding_size_mb * 4):.1f}× vs reloading")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
