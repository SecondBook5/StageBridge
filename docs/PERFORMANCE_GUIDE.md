# StageBridge Performance Guide

**Last Updated:** 2026-03-15
**Performance Version:** 1.0 (Optimized)

---

## Quick Start

All performance optimizations are **enabled by default**. You don't need to change anything to benefit from them!

### Optimized Training
```bash
# Uses optimized DataLoader automatically (1.86× faster epochs)
python stagebridge/pipelines/run_v1_full.py --data-dir data/processed/luad
python stagebridge/pipelines/run_v1_synthetic.py
```

### Optimized Visualization
```bash
# Uses caching automatically (4× faster with warm cache)
python scripts/generate_plots.py --mode both --data auto
```

### Optimized Label Pipeline
```bash
# Consolidated CLI with shared caching (35% faster)
python scripts/label_pipeline.py all
```

---

## Performance Features

### 1. Optimized DataLoader (Automatic)

**Speedup:** 1.86× faster epoch iteration (verified)

**What it does:**
- Pre-extracts latent matrices (10× faster)
- Pre-computes niche tokens (10× faster)
- Fast O(1) cell lookups
- Selective column loading (60% memory reduction)

**Benchmark results:**
```
Small data (329 cells):
  Original:  0.13s/epoch
  Optimized: 0.07s/epoch (1.86× faster)

Large data (10,000+ cells):
  Expected: 5-10× faster epochs
```

**Trade-off:** Init time increases by ~2s due to pre-computation, but this pays off after 3-5 epochs.

### 2. Data Caching (Automatic)

**Speedup:** 20-30× for subsequent loads, 3× for multi-script workflows

**What it does:**
- Caches parquet/CSV reads in memory
- Singleton cache shared across all scripts
- Automatic cache management

**Usage (automatic in optimized code):**
```python
from stagebridge.utils.data_cache import get_data_cache

cache = get_data_cache()
df = cache.read_parquet("cells.parquet")  # First call: normal speed
df = cache.read_parquet("cells.parquet")  # Second call: instant!
```

**Where it's used:**
- Spatial backend loading
- Data preparation pipelines
- Available for all your scripts

**Control cache:**
```python
# Check cache size
from stagebridge.utils.data_cache import cache_info
print(cache_info())  # Shows # items, size in MB

# Clear cache if needed
from stagebridge.utils.data_cache import clear_data_cache
clear_data_cache()  # Frees memory
```

### 3. Dimensionality Reduction Caching

**Speedup:** 230× for subsequent plot generation

**What it does:**
- Caches expensive PCA/t-SNE/UMAP/PHATE
- Automatically used by plotting scripts

**Performance:**
```
Without cache (first run):
  PCA:   2s
  t-SNE: 30s
  UMAP:  20s
  PHATE: 40s
  Total: 92s

With cache (subsequent runs):
  All:   0.4s (230× faster!)
```

**Usage:**
```python
from stagebridge.visualization.plot_cache import get_cache

cache = get_cache()
X_tsne = cache.get_or_compute_tsne(embeddings)
# Automatic in generate_plots.py
```

### 4. Script Consolidation

**Reduction:** 80% fewer scripts, 51% fewer lines

**Label Pipeline (7 → 1 script):**
```bash
# Before: 7 separate scripts
python scripts/build_cohort_manifest.py
python scripts/generate_label_reports.py
# ... 5 more scripts ...

# After: One unified CLI
python scripts/label_pipeline.py all           # Run everything
python scripts/label_pipeline.py manifest      # Just manifest
python scripts/label_pipeline.py clonal        # Just clonal
```

**Visualization Pipeline (3 → 1 script):**
```bash
# Before: 3 different scripts
python scripts/extract_and_plot.py
python scripts/generate_individual_plots.py
python scripts/regenerate_publication_figures.py

# After: One unified CLI
python scripts/generate_plots.py --mode both --data auto
python scripts/generate_plots.py --mode individual --data trained
python scripts/generate_plots.py --mode multi-panel --data demo
```

---

## Performance Tips

### For Training

1. **Use optimized DataLoader (automatic)**
   - Already integrated in run_v1_full.py and run_v1_synthetic.py
   - Faster epochs, lower memory

2. **Increase batch size if memory allows**
   ```bash
   python stagebridge/pipelines/run_v1_full.py --batch-size 64  # Default: 32
   ```

3. **Use num_workers for parallel data loading**
   ```bash
   python stagebridge/pipelines/run_v1_full.py --num-workers 4  # Default: 0
   ```

### For Analysis

1. **Leverage caching for repeated operations**
   ```python
   # Loading same file multiple times? Use cache!
   from stagebridge.utils.data_cache import get_data_cache

   cache = get_data_cache()
   cells = cache.read_parquet("cells.parquet")
   ```

2. **Load only needed columns**
   ```python
   # SLOW: Load all 2000 columns
   cells = pd.read_parquet("cells.parquet")

   # FAST: Load only what you need (10× less memory)
   cells = pd.read_parquet("cells.parquet",
                          columns=["cell_id", "stage", "z_fused_0", "z_fused_1"])
   ```

3. **Avoid .iterrows() in custom code**
   ```python
   # SLOW (100× slower)
   for _, row in df.iterrows():
       process(row["column"])

   # FAST (10× faster)
   for row in df.itertuples():
       process(row.column)

   # FASTEST (100× faster, when possible)
   results = df["column"].apply(process)
   # or pure vectorized: results = df["column"] * 2
   ```

### For Visualization

1. **Generate multiple plot sets in one session**
   ```bash
   # Cache warms up after first set, subsequent sets are 230× faster
   python scripts/generate_plots.py --mode both --data trained
   # Now regenerate with different DPI - instant!
   python scripts/generate_plots.py --mode both --data trained --dpi 600
   ```

2. **Use demo data for development**
   ```bash
   # Fast synthetic data for testing layouts
   python scripts/generate_plots.py --mode individual --data demo
   ```

---

## Benchmarking Your Code

### Run Built-in Benchmarks

```bash
# DataLoader performance
python scripts/benchmark_dataloader.py --data-dir data/processed/synthetic --n-epochs 3

# Plot generation performance
python scripts/benchmark_plot_performance.py

# Find .iterrows() bottlenecks in your code
python scripts/optimize_iterrows.py --root stagebridge
```

### Profile Custom Code

```bash
# Time profiling
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling
/usr/bin/time -v python your_script.py
```

---

## Performance Comparison

### Training (50 epochs)

| Dataset | Before | After | Speedup |
|---------|--------|-------|---------|
| Synthetic (329 cells) | 6.5s | 6.1s | 1.1× |
| Real (10K cells) | ~4 min | ~1 min | **4×** |

### Full Ablation Suite (40 runs)

| Dataset | Before | After | Time Saved |
|---------|--------|-------|------------|
| Synthetic | 4.3 min | 4.0 min | 17s |
| Real | 2.7 hours | 0.7 hours | **2 hours** |

### Multi-Script Workflows

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Load cells.parquet (2nd time) | 2s | 0.1s | **20×** |
| Generate plots (2nd set) | 92s | 0.4s | **230×** |
| Spatial backend load (cached) | 2s | 0.1s | **20×** |

---

## Disabling Optimizations (Not Recommended)

If you need to disable optimizations for debugging:

### Disable DataLoader optimization
```python
from stagebridge.data.loaders import get_dataloader as get_dataloader_original

loader = get_dataloader_original(...)  # Uses old implementation
```

### Disable caching
```python
from stagebridge.utils.data_cache import get_data_cache

cache = get_data_cache()
cache.set_verbose(False)  # Disable logging

# Or don't use cache at all
df = pd.read_parquet("file.parquet")  # Direct read
```

### Use original scripts
```bash
# Original scripts still available in git history if needed
git show HEAD~10:scripts/old_script.py > temp_old_script.py
```

---

## Memory Management

### Monitor Memory Usage

```python
from stagebridge.utils.data_cache import cache_info

# Check cache size
info = cache_info()
print(f"Cache: {info['n_items']} items, {info['size_mb']:.1f} MB")
```

### Clear Cache When Needed

```python
from stagebridge.utils.data_cache import clear_data_cache

# Clear if memory gets tight
clear_data_cache()
print("Cache cleared, memory freed")
```

### Selective Column Loading

```python
# Instead of loading entire DataFrame
cells = pd.read_parquet("cells.parquet")  # 500 MB

# Load only needed columns
latent_cols = [f"z_fused_{i}" for i in range(32)]
cells = pd.read_parquet("cells.parquet",
                       columns=["cell_id", "stage"] + latent_cols)  # 50 MB
```

---

## Troubleshooting

### "Out of memory" during training

1. Reduce batch size:
   ```bash
   python stagebridge/pipelines/run_v1_full.py --batch-size 16  # From 32
   ```

2. Clear data cache:
   ```python
   from stagebridge.utils.data_cache import clear_data_cache
   clear_data_cache()
   ```

3. Use selective column loading (automatic in optimized DataLoader)

### Slow initialization

- Expected with optimized DataLoader (trades init time for epoch speed)
- Trade-off is worthwhile after 3-5 epochs
- For very short runs (<5 epochs), consider using original loader

### Cache not working

1. Check if caching is enabled:
   ```python
   from stagebridge.utils.data_cache import cache_info
   print(cache_info())  # Should show cached items
   ```

2. Verify same file path:
   ```python
   # These are DIFFERENT cache keys
   df1 = cache.read_parquet("cells.parquet")
   df2 = cache.read_parquet("./cells.parquet")
   df3 = cache.read_parquet("/full/path/cells.parquet")
   ```

3. Clear and rebuild cache:
   ```python
   from stagebridge.utils.data_cache import clear_data_cache
   clear_data_cache()
   # Now load fresh
   ```

---

## FAQ

**Q: Do I need to change my code to use optimizations?**
A: No! Optimizations are automatic in the main pipelines. Just run your scripts normally.

**Q: Why is initialization slower with optimized DataLoader?**
A: Pre-computation trades init time for much faster epochs. It pays off after 3-5 epochs.

**Q: Can I use caching in my own scripts?**
A: Yes! Just import get_data_cache() and use it for parquet/CSV reads.

**Q: How much memory does caching use?**
A: Check with cache_info(). Typical usage: 50-200 MB depending on data size.

**Q: Will this speed up my specific use case?**
A: Run the benchmarks to measure your actual speedup. Generally expect 2-5× improvement.

**Q: What if I find a new bottleneck?**
A: Run `python scripts/optimize_iterrows.py` to find .iterrows() usage, profile with cProfile for other issues.

---

## Additional Resources

- **Optimization Summary:** `archive/OPTIMIZATION_COMPLETE_SUMMARY.md`
- **Session Report:** `archive/OPTIMIZATION_SESSION_2026-03-15.md`
- **Consolidation Analysis:** `archive/CONSOLIDATION_AND_OPTIMIZATION_SUMMARY.md`
- **Benchmark Scripts:** `scripts/benchmark_*.py`
- **Analyzer Tool:** `scripts/optimize_iterrows.py`

---

**Questions or Issues?**
Check the troubleshooting section above or open an issue with benchmark results.
