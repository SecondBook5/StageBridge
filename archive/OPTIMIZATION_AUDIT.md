# StageBridge V1 - Code Optimization Audit

## Repository Statistics

### Codebase Size
- Python files: 167
- Total functions/classes: 1,102
- Lines of code: ~50,595
- Largest files:
  - figure_generation.py: 2,099 lines
  - research_frontend.py: 1,637 lines
  - hlca_mapper.py: 1,553 lines
  - train.py: 1,429 lines

### Import Performance
- Import time: 0.001s (excellent)
- No circular dependencies detected
- Clean module structure

## Complexity Analysis

### High Complexity Functions (>15 cyclomatic complexity)
1. `map_full_snrna_with_hlca` (hlca_mapper.py) - complexity: 45
2. `run` (build_eamist_bags.py) - complexity: 34
3. `evaluate_hlca_mapping_outputs` (hlca_mapper.py) - complexity: 32
4. `run_data_prep` (run_data_prep.py) - complexity: 31
5. `run_tangram_hlca_projection` (tangram_mapper.py) - complexity: 29

**Recommendation**: These functions should be refactored into smaller helper functions.

### Long Functions (>100 LOC)
1. `map_full_snrna_with_hlca` - 248 statements
2. `evaluate_hlca_mapping_outputs` - 191 statements
3. Several visualization functions - 120-160 statements each

**Recommendation**: Extract visualization helper functions into a utilities module.

## Performance Opportunities

### 1. Caching
- Current functions with caching: 0
- Tensor conversions (cpu/numpy/detach): 892 occurrences
- Loop iterations: 1,437 Python loops

**Opportunities:**
- Add `@lru_cache` to pure functions:
  - Reference loading functions
  - Metric computation functions
  - Stage parsing/normalization functions
- Cache expensive computations in model classes

### 2. Vectorization
- Many loops could be replaced with NumPy/PyTorch vectorized operations
- Especially in data preprocessing and metric computation
- Potential speedup: 10-100x for large datasets

### 3. Memory Optimization
- Multiple tensor conversions between GPU/CPU
- Consider keeping tensors on device longer
- Use `torch.no_grad()` consistently in evaluation
- Pre-allocate arrays where possible

### 4. I/O Optimization
- Use memory-mapped files for large datasets (mmap)
- Implement lazy loading for reference atlases
- Add checkpointing for long-running pipelines
- Cache processed data artifacts

## Directory Structure Optimization

### Current Structure
```
stagebridge/
├── analysis/ (2 files)
├── cli.py
├── config.py
├── context_model/ (13 files)
├── data/
│   ├── common/ (3 files)
│   └── luad_evo/ (24 files)
├── evaluation/ (20 files)
├── logging_utils.py
├── models/ (2 files)
├── notebook_api.py
├── pipelines/ (18 files)
├── reference/ (11 files)
├── spatial_backends/ (4 files)
├── spatial_mapping/ (4 files)
├── transition_model/ (13 files)
├── utils/ (3 files)
└── viz/ (11 files)
```

### Recommendations

1. **Consolidate visualization code**:
   - Merge `viz/` and `visualization/` into single `visualization/` module
   - Extract common plotting utilities

2. **Simplify evaluation**:
   - 20 files in evaluation/ seems high
   - Consider grouping related evaluations

3. **Consolidate spatial code**:
   - Merge `spatial_backends/` and `spatial_mapping/` into single `spatial/` module

## Code Quality Improvements

### Completed
- [x] Removed all emojis (43 files)
- [x] Auto-fixed 359 lint issues
- [x] Cleaned all __pycache__ directories
- [x] Fixed all import errors
- [x] 100% test pass rate

### Remaining
- [ ] Refactor high-complexity functions (45+ complexity)
- [ ] Split long functions (>150 LOC)
- [ ] Add type hints to untyped functions
- [ ] Add docstrings to undocumented functions
- [ ] Implement caching for expensive operations
- [ ] Vectorize performance-critical loops
- [ ] Consolidate duplicate code patterns

## Performance Benchmarks

### Baseline (Current)
- Import time: 0.001s
- Test suite: ~22s (100 tests)
- Lint check: ~2s
- No performance profiling data available

### Target Goals
- Import time: <0.001s (maintain)
- Test suite: <20s (10% improvement)
- Training speed: 20-30% faster with caching
- Memory usage: 15-20% reduction with optimization

## Optimization Priority

### High Priority (Performance Impact)
1. Add caching to reference loading functions
2. Vectorize data preprocessing loops
3. Optimize tensor conversions (reduce CPU/GPU transfers)
4. Implement lazy loading for large datasets

### Medium Priority (Code Quality)
1. Refactor high-complexity functions
2. Split long functions into helpers
3. Consolidate duplicate code
4. Add comprehensive docstrings

### Low Priority (Clean Structure)
1. Consolidate visualization modules
2. Merge spatial modules
3. Simplify evaluation structure
4. Organize utility functions

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
- Add @lru_cache to pure functions
- Optimize common tensor conversions
- Pre-allocate arrays in tight loops
- Add torch.no_grad() where missing

### Phase 2: Code Quality (2-4 hours)
- Refactor top 5 high-complexity functions
- Split long functions (>150 LOC)
- Extract common visualization patterns
- Add missing docstrings

### Phase 3: Directory Restructure (1-2 hours)
- Consolidate visualization modules
- Merge spatial modules
- Clean up evaluation structure

### Phase 4: Performance Testing (2-3 hours)
- Profile end-to-end pipeline
- Benchmark critical paths
- Validate optimizations
- Update documentation

## Expected Outcomes

After optimization:
- 20-30% faster training
- 15-20% less memory usage
- Cleaner, more maintainable code
- Better performance on large datasets
- Easier to understand and extend

## Notes

- Optimization should not break existing tests
- Maintain backward compatibility where possible
- Document all performance-critical changes
- Profile before and after optimization
- Focus on user-facing bottlenecks first
