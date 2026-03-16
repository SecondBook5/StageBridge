# StageBridge V1 Implementation To-Do List

**Last Updated:** 2026-03-15
**Purpose:** Complete checklist for implementing V1 codebase
**Priority:** Work top-to-bottom within each section

---

## Quick Status

| Category | Complete | In Progress | To Do | Total |
|----------|----------|-------------|-------|-------|
| **Data Pipeline** | 2 | 2 | 6 | 10 |
| **Model Layers** | 2 | 3 | 2 | 7 |
| **Training** | 0 | 0 | 8 | 8 |
| **Evaluation** | 0 | 0 | 7 | 7 |
| **Testing** | 0 | 0 | 6 | 6 |
| **Notebook** | 0 | 0 | 1 | 1 |
| **TOTAL** | **4** | **5** | **30** | **39** |

---

## 1. Data Pipeline (Step 0)

### 1.1 Core Pipeline  PARTIALLY COMPLETE

- [x] Extract tar archives
- [x] Basic QC filtering
- [ ] **Memory-efficient backed-mode QC** (HIGH PRIORITY)
  - File: `stagebridge/pipelines/run_data_prep.py`
  - Test on smaller dataset first
  - Verify memory usage < 64GB

- [ ] **Spatial backend integration** (BLOCKING)
  - File: `stagebridge/spatial_backends/tangram_wrapper.py`
  - File: `stagebridge/spatial_backends/destvi_wrapper.py`
  - File: `stagebridge/spatial_backends/tacco_wrapper.py`
  - Each must output standardized format

- [ ] **Canonical artifacts generation**
  - Generate `cells.parquet` from merged h5ads
  - Generate `neighborhoods.parquet` from spatial graphs
  - Generate `stage_edges.parquet` from stage definitions
  - Generate `split_manifest.json` for CV
  - Generate `feature_spec.yaml`

### 1.2 Spatial Backend Wrappers (BLOCKING FOR TRAINING)

**Priority: Complete all 3 before training**

```python
# File: stagebridge/spatial_backends/tangram_wrapper.py
def run_tangram(snrna_path, spatial_path, output_dir):
    """
    Run Tangram spatial mapping.

    Returns:
        - cell_type_proportions.parquet
        - mapping_confidence.parquet
        - upstream_metrics.json
        - backend_metadata.json
    """
    # TODO: Implement
    pass
```

**Tasks:**
- [ ] Implement `tangram_wrapper.py`
- [ ] Implement `destvi_wrapper.py`
- [ ] Implement `tacco_wrapper.py`
- [ ] Create base class `SpatialBackend` for standardization
- [ ] Add upstream metrics computation
- [ ] Test on small dataset subset

### 1.3 Data Loaders

```python
# File: stagebridge/data/loaders.py
class CellDataset(Dataset):
    """Load cells with optional neighborhoods and expression"""
    # TODO: Implement memory-mapped loading
    pass

class StageEdgeBatchLoader(DataLoader):
    """Sample source→target cell pairs for training"""
    # TODO: Implement stratified sampling
    pass
```

**Tasks:**
- [ ] Implement `CellDataset` with memory mapping
- [ ] Implement `StageEdgeBatchLoader` for transitions
- [ ] Add caching for frequently accessed data
- [ ] Test loading speed (target: <1s per batch)

---

## 2. Model Layers

### 2.1 Layer A: Dual-Reference Latent  IN PROGRESS

```python
# File: stagebridge/models/dual_reference.py
class DualReferenceLatentMapper(nn.Module):
    def __init__(self, hlca_path, luca_path, fusion_method='concat'):
        # TODO: Load pretrained scVI models
        pass

    def forward(self, expression):
        # TODO: Map to HLCA and LuCA spaces
        # TODO: Fuse embeddings
        pass
```

**Tasks:**
- [ ] Download HLCA reference atlas
- [ ] Download LuCA reference atlas
- [ ] Implement scVI alignment wrapper
- [ ] Implement fusion layer (concat or learned)
- [ ] Test on small cell subset
- [ ] Validate latent space quality (UMAP visualization)

### 2.2 Layer B: Local Niche Encoder  COMPLETE

- [x] `LocalNicheTransformerEncoder` implemented
- [x] 9-token tokenizer implemented
- [ ] **Add influence tensor extraction method**
  ```python
  def get_influence_tensor(self):
      """Extract attention weights for interpretability"""
      # TODO: Return (n_cells, n_neighbors) attention matrix
      pass
  ```

### 2.3 Layer C: Hierarchical Set Transformer  COMPLETE

- [x] ISAB, SAB, PMA blocks implemented
- [ ] **Add set membership tracking**
  ```python
  def track_set_membership(self, cell_ids, lesion_ids):
      """Track which cells belong to which lesions"""
      # TODO: For evaluation and visualization
      pass
  ```

### 2.4 Layer D: Flow Matching  IN PROGRESS (HIGH PRIORITY)

```python
# File: stagebridge/models/flow_matching.py
class OTCFMTransitionModel(nn.Module):
    def __init__(self, latent_dim, context_dim):
        # TODO: Implement conditional flow network
        self.velocity_net = MLP([latent_dim + context_dim + 1, 512, 512, latent_dim])
        self.diffusion_net = MLP([latent_dim + context_dim + 1, 256, 1])

    def compute_sinkhorn_coupling(self, z_src, z_tgt, epsilon=0.05, num_iters=100):
        """Compute OT coupling matrix via Sinkhorn"""
        # TODO: Implement
        pass

    def forward(self, z_src, z_tgt, context, t):
        """Compute flow matching loss"""
        # TODO:
        # 1. Compute coupling π
        # 2. Sample time t ~ U[0,1]
        # 3. Interpolate z(t)
        # 4. Predict velocity
        # 5. Compute MSE loss
        pass

    def sample_trajectory(self, z_src, context, num_steps=100):
        """Sample stochastic trajectory"""
        # TODO: Euler-Maruyama integration
        pass
```

**Tasks:**
- [ ] Implement Sinkhorn algorithm
- [ ] Implement interpolation with noise
- [ ] Implement velocity network
- [ ] Implement stochastic sampling
- [ ] Test on synthetic 2D data (ground truth available)
- [ ] Validate on one LUAD edge (AIS → MIA)

### 2.5 Layer F: Evolutionary Compatibility  IN PROGRESS

```python
# File: stagebridge/models/evolution_compat.py
class EvolutionaryCompatibilityModule(nn.Module):
    def __init__(self, wes_dim, latent_dim):
        self.compatibility_net = nn.Linear(wes_dim * 2, 1)

    def compute_compatibility(self, z_pred, wes_source, wes_target_pool, metadata):
        """
        Compute compatibility scores.

        Returns:
            matched_scores: compatibility with correct donor/stage
            wrong_donor_scores: compatibility with wrong donor
            wrong_stage_scores: compatibility with wrong stage
        """
        # TODO: Implement
        pass

    def compatibility_loss(self, matched, wrong_donor, wrong_stage, margin=0.3):
        """Contrastive loss"""
        # TODO: max(0, margin - matched + wrong_donor) + ...
        pass
```

**Tasks:**
- [ ] Implement compatibility scoring
- [ ] Implement contrastive loss
- [ ] Add negative sampling logic
- [ ] Test matched vs shuffled separation
- [ ] Validate effect size > 0.5

---

## 3. Training Infrastructure

### 3.1 Full Training Loop (HIGH PRIORITY)

```python
# File: stagebridge/training/trainer.py
class StageBridgeTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)

    def train_epoch(self):
        """Single training epoch"""
        # TODO:
        # 1. Iterate over batches
        # 2. Forward pass through all layers
        # 3. Compute composite loss
        # 4. Backward and optimize
        # 5. Log metrics
        pass

    def validate(self):
        """Validation epoch"""
        # TODO: Compute validation metrics
        pass

    def train(self):
        """Full training loop with early stopping"""
        # TODO:
        # for epoch in range(max_epochs):
        #     train_epoch()
        #     validate()
        #     checkpoint if best
        #     early stop if needed
        pass
```

**Tasks:**
- [ ] Implement `StageBridgeTrainer` class
- [ ] Implement composite loss (flow + compatibility + aux)
- [ ] Add gradient clipping
- [ ] Add learning rate scheduling
- [ ] Add early stopping logic
- [ ] Add checkpoint management
- [ ] Add comprehensive logging
- [ ] Test on small dataset (smoke test)

### 3.2 Configuration System

```yaml
# File: configs/luad_evo_v1.yaml
model:
  latent_dim: 256
  niche_embedding_dim: 256
  set_embedding_dim: 512
  n_attention_heads: 8
  n_inducing_points: 64

training:
  batch_size: 64
  learning_rate: 1.0e-4
  max_epochs: 100
  early_stopping_patience: 10
  grad_clip: 1.0

loss_weights:
  flow_matching: 1.0
  evolutionary_compatibility: 0.05
  auxiliary: 0.01
```

**Tasks:**
- [ ] Create V1 config file
- [ ] Add config validation
- [ ] Add config override system
- [ ] Test config loading

---

## 4. Evaluation Infrastructure

### 4.1 Metrics Implementation

```python
# File: stagebridge/evaluation/metrics.py
class MetricsComputer:
    @staticmethod
    def compute_wasserstein(pred, true):
        """Wasserstein distance"""
        # TODO: Implement per-dimension average
        pass

    @staticmethod
    def compute_mmd(pred, true, gamma=1.0):
        """Maximum Mean Discrepancy"""
        # TODO: Implement with RBF kernel
        pass

    @staticmethod
    def compute_ece(confidences, accuracies, n_bins=10):
        """Expected Calibration Error"""
        # TODO: Implement binned calibration
        pass

    @staticmethod
    def compute_coverage(pred, true, sigma, alpha=0.1):
        """Prediction interval coverage"""
        # TODO: Check if true falls in predicted intervals
        pass

    def compute_all(self, predictions, targets, uncertainties=None):
        """Compute full metric suite"""
        # TODO: Return dict with all metrics
        pass
```

**Tasks:**
- [ ] Implement all metric functions
- [ ] Add statistical testing utilities
- [ ] Add bootstrap confidence intervals
- [ ] Test on synthetic data with known metrics

### 4.2 Cross-Validation Orchestrator

```python
# File: stagebridge/evaluation/cv.py
class DonorHeldOutCV:
    def __init__(self, split_manifest, config):
        self.splits = split_manifest['splits']
        self.config = config

    def run_fold(self, fold_id):
        """Train and evaluate one fold"""
        # TODO:
        # 1. Create fold-specific data loaders
        # 2. Train model
        # 3. Evaluate on test donors
        # 4. Save results
        pass

    def run_all_folds(self, parallel=False):
        """Run all 5 folds"""
        # TODO: Support parallel execution
        pass

    def aggregate_results(self):
        """Aggregate metrics across folds"""
        # TODO: Compute mean ± std
        pass
```

**Tasks:**
- [ ] Implement `DonorHeldOutCV` class
- [ ] Add parallel fold execution
- [ ] Add results aggregation
- [ ] Test on small dataset

### 4.3 Ablation Runner

```python
# File: stagebridge/evaluation/ablations.py
def run_ablation(ablation_name, config):
    """
    Run single ablation experiment.

    ablation_name: 'no_niche', 'no_genomics', etc.
    """
    # TODO:
    # 1. Modify config based on ablation
    # 2. Train model
    # 3. Evaluate
    # 4. Return metrics
    pass

def run_all_tier1_ablations(config):
    """Run all Tier 1 ablations"""
    ablations = [
        'no_niche',
        'pooled_niche',
        'no_genomics',
        'genomics_as_feature',
        'deterministic',
        'flat_pooling',
        'hlca_only',
        'luca_only',
        'alt_backend',
    ]
    # TODO: Run all ablations, aggregate results
    pass
```

**Tasks:**
- [ ] Implement ablation configuration logic
- [ ] Implement `run_ablation` function
- [ ] Implement `run_all_tier1_ablations`
- [ ] Add statistical comparison utilities
- [ ] Generate ablation heatmap figure

---

## 5. Testing

### 5.1 Unit Tests

```python
# File: tests/test_models.py
def test_dual_reference_mapper():
    """Test Layer A"""
    # TODO: Test on small synthetic data
    pass

def test_niche_encoder():
    """Test Layer B"""
    # TODO: Test 9-token construction
    # TODO: Test attention mechanism
    pass

def test_flow_matching():
    """Test Layer D"""
    # TODO: Test on synthetic 2D trajectories
    # TODO: Verify coupling is valid
    pass

def test_compatibility_module():
    """Test Layer F"""
    # TODO: Test matched > shuffled
    pass
```

**Tasks:**
- [ ] Write unit tests for all layers
- [ ] Write tests for data loaders
- [ ] Write tests for metrics
- [ ] Achieve >80% code coverage

### 5.2 Integration Tests

```python
# File: tests/test_integration.py
def test_end_to_end_smoke():
    """Smoke test: full pipeline on tiny dataset"""
    # TODO:
    # 1. Create tiny synthetic dataset (100 cells)
    # 2. Run full training for 5 epochs
    # 3. Verify no crashes, finite loss
    pass

def test_data_pipeline():
    """Test Step 0 on small subset"""
    # TODO: Test QC, spatial backends, artifact generation
    pass
```

**Tasks:**
- [ ] Write end-to-end smoke test
- [ ] Write data pipeline integration test
- [ ] Set up CI/CD to run tests automatically

### 5.3 Synthetic Benchmarks

```python
# File: tests/test_synthetic.py
def generate_synthetic_progression(n_cells=1000):
    """
    Generate synthetic cell progression with known transitions.

    Returns:
        cells, neighborhoods, ground_truth_trajectories
    """
    # TODO: Implement
    pass

def test_synthetic_recovery():
    """Test that model recovers synthetic ground truth"""
    # TODO:
    # 1. Generate synthetic data
    # 2. Train model
    # 3. Check recovery accuracy > 0.7
    pass
```

**Tasks:**
- [ ] Implement synthetic data generator
- [ ] Test ground truth recovery
- [ ] Add to continuous testing

---

## 6. Notebook Update

### 6.1 Primary Notebook

**File:** `StageBridge.ipynb`

**Required sections:**
1. Setup and configuration
2. Part 1: Data preparation (Step 0)
   - Load canonical artifacts
   - Visualize dataset overview
3. Part 2: Layer A (Dual-reference latent)
   - Compute embeddings
   - Visualize UMAP
4. Part 3: Layer B (Local niche encoder)
   - Build neighborhood graphs
   - Visualize example neighborhoods
5. Part 4: Training (or load checkpoint)
6. Part 5: Evaluation
   - Compute all metrics
   - Visualize calibration, ablations
7. Part 6: Biological interpretation
   - Attention heatmaps
   - Trajectory plots
8. Summary and next steps

**Tasks:**
- [ ] Replace old notebook (backup created: `StageBridge.ipynb.backup`)
- [ ] Create end-to-end demo mode (synthetic data)
- [ ] Create full mode (real LUAD data)
- [ ] Add all visualizations
- [ ] Test notebook runs end-to-end

---

## 7. Priority Order (Implementation Sequence)

### Week 1: Data Infrastructure (BLOCKING)
**Goal:** Can load data for training

1. [ ] Complete backed-mode QC in `run_data_prep.py`
2. [ ] Implement spatial backend wrappers (Tangram/DestVI/TACCO)
3. [ ] Generate all canonical artifacts
4. [ ] Implement `CellDataset` and `StageEdgeBatchLoader`
5. [ ] Test data loading on small subset

**Success:** Can iterate over training batches efficiently

### Week 2: Complete Layer D (CRITICAL PATH)
**Goal:** Can train basic model

6. [ ] Implement Sinkhorn coupling
7. [ ] Implement flow matching forward pass
8. [ ] Implement stochastic sampling
9. [ ] Test on synthetic 2D data
10. [ ] Validate on one LUAD edge

**Success:** Loss converges on synthetic data

### Week 3: Training Infrastructure
**Goal:** Can train full model

11. [ ] Implement `StageBridgeTrainer` class
12. [ ] Integrate all layers (A→B→C→D→F)
13. [ ] Add composite loss
14. [ ] Add checkpoint management
15. [ ] Run smoke test on small dataset

**Success:** Can train for 100 epochs without crashes

### Week 4: Layer A and Full Integration
**Goal:** All layers working

16. [ ] Download and process reference atlases
17. [ ] Implement Layer A alignment
18. [ ] Complete Layer F compatibility module
19. [ ] Run full integration test
20. [ ] Train on small LUAD subset

**Success:** Model trains on real data

### Week 5-6: Evaluation Infrastructure
**Goal:** Can evaluate rigorously

21. [ ] Implement all metrics
22. [ ] Implement donor-held-out CV
23. [ ] Implement ablation runner
24. [ ] Run CV on small dataset
25. [ ] Validate metrics make sense

**Success:** Can compute all V1 metrics

### Week 7-8: Full Experiments (HPC Required)
**Goal:** Generate V1 results

26. [ ] Run full data prep on HPC
27. [ ] Train full model (5 folds × full data)
28. [ ] Run all Tier 1 ablations
29. [ ] Generate all evaluation metrics
30. [ ] Create all publication figures

**Success:** All metrics meet V1 targets

### Week 9-10: Testing and Refinement
**Goal:** Publication-ready code

31. [ ] Write all unit tests
32. [ ] Write integration tests
33. [ ] Achieve >80% code coverage
34. [ ] Add documentation strings
35. [ ] Update notebook to final version

**Success:** Code passes all tests

### Week 11-12: Paper Writing
**Goal:** Submit paper

36. [ ] Write Results section (with real data)
37. [ ] Finalize all figures
38. [ ] Write Discussion
39. [ ] Write Abstract (last)
40. [ ] Submit!

---

## 8. Critical Path Dependencies

```
Data Infrastructure → Layer D → Training Loop → Full Integration → Evaluation → Experiments
     (Week 1)        (Week 2)     (Week 3)         (Week 4)        (Week 5-6)   (Week 7-8)
```

**Cannot proceed to next stage without completing previous stage.**

### Blocking Items (DO FIRST)

1. **HPC Access** - Need for data prep
2. **Spatial Backend Wrappers** - Blocks artifact generation
3. **Layer D Flow Matching** - Blocks training
4. **Reference Atlases** - Blocks Layer A

---

## 9. Testing Checkpoints

After each major component, verify:

- [ ] **Data loaders:** Can load 1000 batches in <10 minutes
- [ ] **Layer D:** Loss converges on synthetic 2D data
- [ ] **Training:** Smoke test runs for 10 epochs without crash
- [ ] **Layer A:** UMAP shows stage structure
- [ ] **Full model:** Loss decreases on real data
- [ ] **CV:** 5 folds complete in reasonable time
- [ ] **Metrics:** All values in expected ranges
- [ ] **Ablations:** Effect sizes match expectations

---

## 10. Success Criteria

V1 implementation is complete when:

### Technical
- [ ] All 39 tasks above are complete
- [ ] All tests pass
- [ ] Notebook runs end-to-end
- [ ] Code is documented

### Scientific
- [ ] Wasserstein distance: 0.45 ± 0.05 
- [ ] ECE < 0.1 
- [ ] Compatibility gap > 0.3 
- [ ] Backend correlation > 0.7 
- [ ] All ablations show expected patterns

### Publication
- [ ] All figures generated
- [ ] All tables complete
- [ ] Results section written
- [ ] Code and data released

---

## 11. Quick Reference

### Most Important Files to Create/Modify

**High Priority:**
1. `stagebridge/models/flow_matching.py` (NEW)
2. `stagebridge/spatial_backends/*.py` (NEW)
3. `stagebridge/training/trainer.py` (NEW)
4. `stagebridge/evaluation/metrics.py` (NEW)
5. `stagebridge/data/loaders.py` (NEW)
6. `stagebridge/pipelines/run_data_prep.py` (MODIFY - backed mode)

**Medium Priority:**
7. `stagebridge/models/dual_reference.py` (NEW)
8. `stagebridge/models/evolution_compat.py` (MODIFY)
9. `stagebridge/evaluation/cv.py` (NEW)
10. `stagebridge/evaluation/ablations.py` (NEW)

**Lower Priority:**
11. `tests/*.py` (NEW)
12. `StageBridge.ipynb` (REPLACE)

---

## 12. Daily Development Template

```markdown
## Day [N] - [Date]

### Goals
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

### Progress
- Completed: ...
- In progress: ...
- Blocked by: ...

### Decisions Made
- ...

### Next Session
- Start with: ...
```

---

## 13. Resources Needed

### Computational
- [ ] HPC allocation requested (128GB RAM, 8 cores for data prep)
- [ ] GPU access (V100 or A100, 32GB+ VRAM)
- [ ] Storage allocation (~300GB)

### Data
- [ ] HLCA reference atlas downloaded
- [ ] LuCA reference atlas downloaded
- [ ] Raw LUAD data accessible

### Tools
- [ ] Tangram installed and tested
- [ ] DestVI (scvi-tools) installed
- [ ] TACCO installed

---

**End of Implementation To-Do List**

**Start with:** Week 1, Task 1 (backed-mode QC)
**Next review:** After completing Week 1 tasks
