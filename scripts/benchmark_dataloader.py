#!/usr/bin/env python
"""
Benchmark DataLoader Performance

Compares original vs optimized DataLoader implementations to measure
real-world training throughput improvements.

Measures:
- Data loading time
- __getitem__ throughput
- Epoch iteration time
- Memory usage

Expected improvements:
- 5-10× faster __getitem__
- 2-3× faster overall epoch time
- 30-50% memory reduction
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path
import psutil
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_original_loader(data_dir, n_epochs=3):
    """Benchmark original DataLoader."""
    from stagebridge.data.loaders import get_dataloader

    print("=" * 80)
    print("BENCHMARKING ORIGINAL DATALOADER")
    print("=" * 80)

    mem_before = get_memory_usage_mb()
    t0 = time.time()

    # Create loader
    print("\nInitializing loader...")
    t_init = time.time()
    loader = get_dataloader(
        data_dir=data_dir,
        fold=0,
        split="train",
        batch_size=32,
        latent_dim=32,
        shuffle=True,
    )
    init_time = time.time() - t_init
    print(f"  Initialization: {init_time:.2f}s")

    mem_after_init = get_memory_usage_mb()
    print(f"  Memory: {mem_after_init - mem_before:.1f} MB")

    # Benchmark epoch iteration
    print(f"\nRunning {n_epochs} epochs...")
    epoch_times = []

    for epoch in range(n_epochs):
        t_epoch = time.time()
        batch_count = 0

        for batch in loader:
            batch_count += 1
            # Simulate minimal training work
            _ = batch.z_source.mean()

        epoch_time = time.time() - t_epoch
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1}: {epoch_time:.2f}s ({batch_count} batches)")

    total_time = time.time() - t0
    mem_peak = get_memory_usage_mb()

    return {
        'init_time': init_time,
        'epoch_times': epoch_times,
        'mean_epoch_time': np.mean(epoch_times),
        'total_time': total_time,
        'memory_mb': mem_peak - mem_before,
        'batches_per_epoch': batch_count,
    }


def benchmark_optimized_loader(data_dir, n_epochs=3):
    """Benchmark optimized DataLoader."""
    from stagebridge.data.loaders_optimized import get_dataloader_optimized

    print("\n" + "=" * 80)
    print("BENCHMARKING OPTIMIZED DATALOADER")
    print("=" * 80)

    mem_before = get_memory_usage_mb()
    t0 = time.time()

    # Create loader
    print("\nInitializing optimized loader...")
    t_init = time.time()
    loader = get_dataloader_optimized(
        data_dir=data_dir,
        fold=0,
        split="train",
        batch_size=32,
        latent_dim=32,
        shuffle=True,
        use_cache=True,
    )
    init_time = time.time() - t_init
    print(f"  Initialization: {init_time:.2f}s")

    mem_after_init = get_memory_usage_mb()
    print(f"  Memory: {mem_after_init - mem_before:.1f} MB")

    # Benchmark epoch iteration
    print(f"\nRunning {n_epochs} epochs...")
    epoch_times = []

    for epoch in range(n_epochs):
        t_epoch = time.time()
        batch_count = 0

        for batch in loader:
            batch_count += 1
            # Simulate minimal training work
            _ = batch.z_source.mean()

        epoch_time = time.time() - t_epoch
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1}: {epoch_time:.2f}s ({batch_count} batches)")

    total_time = time.time() - t0
    mem_peak = get_memory_usage_mb()

    return {
        'init_time': init_time,
        'epoch_times': epoch_times,
        'mean_epoch_time': np.mean(epoch_times),
        'total_time': total_time,
        'memory_mb': mem_peak - mem_before,
        'batches_per_epoch': batch_count,
    }


def print_comparison(original, optimized):
    """Print detailed comparison."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print("\n1. Initialization Time")
    print("-" * 40)
    print(f"  Original:  {original['init_time']:6.2f}s")
    print(f"  Optimized: {optimized['init_time']:6.2f}s")
    if original['init_time'] > 0:
        speedup = original['init_time'] / optimized['init_time']
        print(f"  Speedup:   {speedup:6.2f}× {'(slower)' if speedup < 1 else ''}")

    print("\n2. Epoch Iteration Time")
    print("-" * 40)
    print(f"  Original:  {original['mean_epoch_time']:6.2f}s/epoch")
    print(f"  Optimized: {optimized['mean_epoch_time']:6.2f}s/epoch")
    if optimized['mean_epoch_time'] > 0:
        speedup = original['mean_epoch_time'] / optimized['mean_epoch_time']
        print(f"  Speedup:   {speedup:6.2f}×")

    print("\n3. Total Time")
    print("-" * 40)
    print(f"  Original:  {original['total_time']:6.2f}s")
    print(f"  Optimized: {optimized['total_time']:6.2f}s")
    if optimized['total_time'] > 0:
        speedup = original['total_time'] / optimized['total_time']
        print(f"  Speedup:   {speedup:6.2f}×")
        print(f"  Time saved: {original['total_time'] - optimized['total_time']:6.2f}s")

    print("\n4. Memory Usage")
    print("-" * 40)
    print(f"  Original:  {original['memory_mb']:6.1f} MB")
    print(f"  Optimized: {optimized['memory_mb']:6.1f} MB")
    diff_mb = original['memory_mb'] - optimized['memory_mb']
    print(f"  Reduction: {diff_mb:6.1f} MB ({diff_mb/original['memory_mb']*100:+.1f}%)")

    print("\n" + "=" * 80)
    print("PROJECTED IMPACT FOR FULL TRAINING")
    print("=" * 80)

    # Project to 50 epochs
    original_50_epochs = original['mean_epoch_time'] * 50
    optimized_50_epochs = optimized['mean_epoch_time'] * 50

    print(f"\n50-epoch training (synthetic data):")
    print(f"  Original:  {original_50_epochs/60:6.2f} minutes")
    print(f"  Optimized: {optimized_50_epochs/60:6.2f} minutes")
    print(f"  Saved:     {(original_50_epochs - optimized_50_epochs)/60:6.2f} minutes per run")

    # Project to full ablation suite (5 folds × 8 ablations × 50 epochs)
    n_runs = 5 * 8  # folds × ablations
    original_full = original_50_epochs * n_runs
    optimized_full = optimized_50_epochs * n_runs

    print(f"\nFull ablation suite (5 folds × 8 ablations):")
    print(f"  Original:  {original_full/3600:6.2f} hours")
    print(f"  Optimized: {optimized_full/3600:6.2f} hours")
    print(f"  Saved:     {(original_full - optimized_full)/3600:6.2f} hours")

    print("\n" + "=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark DataLoader performance")
    parser.add_argument("--data-dir", default="data/processed/synthetic",
                       help="Path to processed data")
    parser.add_argument("--n-epochs", type=int, default=3,
                       help="Number of epochs to benchmark")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not (data_dir / "cells.parquet").exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Generate synthetic data first:")
        print("  python -c 'from stagebridge.data.synthetic import generate_synthetic_dataset; generate_synthetic_dataset()'")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("DATALOADER PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Data: {data_dir}")
    print(f"Epochs: {args.n_epochs}")
    print("=" * 80)

    # Benchmark original
    try:
        original_results = benchmark_original_loader(data_dir, n_epochs=args.n_epochs)
    except Exception as e:
        print(f"\nOriginal loader failed: {e}")
        print("This is expected if the original implementation has issues.")
        original_results = None

    # Benchmark optimized
    try:
        optimized_results = benchmark_optimized_loader(data_dir, n_epochs=args.n_epochs)
    except Exception as e:
        print(f"\nOptimized loader failed: {e}")
        optimized_results = None
        import traceback
        traceback.print_exc()

    # Compare
    if original_results and optimized_results:
        print_comparison(original_results, optimized_results)
    elif optimized_results:
        print("\n" + "=" * 80)
        print("OPTIMIZED LOADER RESULTS (original unavailable)")
        print("=" * 80)
        print(f"Mean epoch time: {optimized_results['mean_epoch_time']:.2f}s")
        print(f"Memory usage: {optimized_results['memory_mb']:.1f} MB")
    else:
        print("\nBoth loaders failed - check data directory")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
