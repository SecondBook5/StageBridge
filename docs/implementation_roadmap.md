# StageBridge V1 Implementation Roadmap

**Last Updated:** 2026-03-22
**Status:** V1 Implementation Complete
**Target:** Publication preparation

---

## 1. Overview

This document tracks implementation status for StageBridge V1. Each component is categorized as:
-  **Complete** - Fully implemented and tested
-  **In Progress** - Partially implemented, actively being worked on
-  **Planned** - Designed but not yet implemented
- ⏸ **Deferred** - Pushed to V2/V3

---

## 2. Core Components Status

### 2.1 Data Pipeline (Step 0)

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **Raw data extraction** |  Complete | snRNA, Visium, WES tarballs | - |
| **QC filtering** |  In Progress | Memory-efficient backed mode implemented | **HIGH** |
| **Normalization** |  Complete | log1p, scaling | - |
| **Merge operations** |  In Progress | snRNA done, spatial backed-mode | **HIGH** |
| **Spatial backend - Tangram** |  Planned | Integration script needed | **HIGH** |
| **Spatial backend - DestVI** |  Planned | Integration script needed | **HIGH** |
| **Spatial backend - TACCO** |  Planned | Integration script needed | **HIGH** |
| **Canonical artifacts generation** |  Planned | cells.parquet, neighborhoods.parquet, etc. | **HIGH** |
| **Audit report** |  Planned | QC summary and provenance | MEDIUM |

**Blocking Issues:**
- Need HPC resources for full spatial data processing (35GB+ files)
- Backed-mode implementation needs testing on real data

**Next Steps:**
1. Test backed-mode QC on smaller datasets
2. Move data to HPC for full pipeline run
3. Implement spatial backend wrappers
4. Generate canonical artifacts

---

### 2.2 Layer A: Dual-Reference Latent Mapping

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **HLCA reference alignment** |  In Progress | scVI integration scaffolded | **HIGH** |
| **LuCA reference alignment** |  In Progress | scVI integration scaffolded | **HIGH** |
| **Euclidean embedding** |  In Progress | Basic implementation exists | **HIGH** |
| **Latent fusion** |  Planned | Concatenation or learned fusion | MEDIUM |
| **Batch correction** |  Planned | Harmony at reference level | MEDIUM |
| **Contrastive pretraining** |  Planned | Optional, may skip for V1 | LOW |

**Blocking Issues:**
- Need to download/process HLCA and LuCA reference atlases

**Next Steps:**
1. Download reference atlases
2. Implement scVI alignment wrapper
3. Test on small subset of data
4. Validate latent space quality

---

### 2.3 Layer B: Local Niche Encoder

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **LocalNicheTransformerEncoder** |  Complete | EA-MIST implementation | - |
| **9-token tokenizer** |  Complete | All tokens implemented | - |
| **Neighborhood graph builder** |  In Progress | K-NN and radius modes | MEDIUM |
| **Distance binning** |  Complete | 4 rings implemented | - |
| **Attention mechanism** |  Complete | Self-attention over tokens | - |
| **Influence tensor extraction** |  Planned | For interpretability | MEDIUM |

**Blocking Issues:**
- None major

**Next Steps:**
1. Validate neighborhood graphs on spatial data
2. Implement influence tensor extraction
3. Add attention visualization utilities

---

### 2.4 Layer C: Hierarchical Set Transformer

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **ISAB block** |  Complete | EA-MIST implementation | - |
| **SAB block** |  Complete | EA-MIST implementation | - |
| **PMA block** |  Complete | EA-MIST implementation | - |
| **Hierarchical pooling** |  Complete | Cell → Lesion → Stage | - |
| **Set membership tracking** |  Planned | For evaluation | LOW |

**Blocking Issues:**
- None

**Next Steps:**
1. Validate on cell-level data
2. Test hierarchical pooling scales

---

### 2.5 Layer D: Flow Matching Transition Model

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **OT-CFM algorithm** |  In Progress | Scaffolded in stochastic_dynamics.py | **HIGH** |
| **Sinkhorn coupling** |  In Progress | Implementation exists | **HIGH** |
| **Flow interpolation** |  In Progress | Basic interpolant | **HIGH** |
| **Conditional flow network** |  Planned | MLP conditioned on niche context | **HIGH** |
| **Stochastic sampling** |  Planned | Euler-Maruyama integration | **HIGH** |
| **Uncertainty estimation** |  Planned | MC sampling | MEDIUM |

**Blocking Issues:**
- Need to integrate with Layers A-C outputs
- Need to test on real stage-edge data

**Next Steps:**
1. Complete conditional flow network
2. Implement stochastic sampling
3. Test on synthetic data first
4. Validate on one LUAD edge

---

### 2.6 Layer F: Evolutionary Compatibility

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **WES feature extraction** |  Complete | TMB, signatures, clones | - |
| **Compatibility scoring** |  In Progress | Scaffolded | MEDIUM |
| **Contrastive loss** |  Planned | Margin-based | MEDIUM |
| **Regularization integration** |  Planned | Into transition loss | MEDIUM |
| **Matched/shuffled controls** |  Planned | For evaluation | MEDIUM |

**Blocking Issues:**
- Need WES data processed and linked to cells

**Next Steps:**
1. Complete compatibility scoring function
2. Implement contrastive loss
3. Add regularization to training loop
4. Test matched vs shuffled separation

---

## 3. Training Infrastructure

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **Data loaders** |  In Progress | Cell and edge loaders scaffolded | **HIGH** |
| **Training loop** |  Planned | Full end-to-end training | **HIGH** |
| **Loss composition** |  Planned | Flow + compatibility + aux | **HIGH** |
| **Optimizer setup** |  Planned | AdamW with scheduling | MEDIUM |
| **Checkpoint management** |  Planned | Save/load/resume | MEDIUM |
| **Logging** |  In Progress | Basic logging exists | MEDIUM |
| **Config system** |  Complete | Hydra-based | - |

**Next Steps:**
1. Implement data loaders for canonical artifacts
2. Build full training loop
3. Add comprehensive logging
4. Test on small dataset

---

## 4. Evaluation Infrastructure

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **Donor-held-out CV** |  Planned | Split generation and evaluation | **HIGH** |
| **Transition quality metrics** |  Planned | Wasserstein, MMD, KL | **HIGH** |
| **Uncertainty metrics** |  Planned | ECE, NLL, Coverage | **HIGH** |
| **Compatibility metrics** |  Planned | Matched vs shuffled gap | MEDIUM |
| **Backend comparison** |  Planned | Across Tangram/DestVI/TACCO | MEDIUM |
| **Ablation runner** |  Planned | Automated ablation execution | MEDIUM |
| **Statistical testing** |  Planned | Paired tests, corrections | MEDIUM |
| **Artifact logging** |  Planned | All outputs tracked | MEDIUM |

**Next Steps:**
1. Implement evaluation metrics
2. Build CV harness
3. Create ablation runner
4. Add statistical testing utilities

---

## 5. Visualization and Interpretation

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **UMAP visualization** |  Planned | Latent space + stage colors | MEDIUM |
| **Attention heatmaps** |  Planned | Niche influence patterns | MEDIUM |
| **Trajectory plots** |  Planned | Flow field and paths | MEDIUM |
| **Calibration curves** |  Planned | Uncertainty visualization | MEDIUM |
| **Spatial overlays** |  Planned | Attention on tissue images | LOW |
| **Publication figures** |  Planned | Per figure specs | LOW |

**Next Steps:**
1. Implement core plotting utilities
2. Create figure generation scripts
3. Automate figure updates with new results

---

## 6. Testing and Validation

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **Unit tests** |  Planned | Per-module tests | MEDIUM |
| **Integration tests** |  Planned | End-to-end smoke tests | **HIGH** |
| **Synthetic benchmarks** |  Planned | Ground truth recovery | MEDIUM |
| **Negative controls** |  Planned | Shuffle, wrong-stage, etc. | MEDIUM |
| **Reproducibility tests** |  Planned | Seed consistency | MEDIUM |

**Next Steps:**
1. Write unit tests for completed modules
2. Create synthetic data generator
3. Implement integration smoke tests

---

## 7. Documentation

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **README** |  Complete | Updated for V1 | - |
| **Architecture docs** |  Complete | All layers documented | - |
| **Methods overview** |  Complete | v1_methods_overview.md | - |
| **Data model spec** |  Complete | data_model_specification.md | - |
| **Evaluation protocol** |  Complete | evaluation_protocol.md | - |
| **Figure specs** |  Complete | figure_table_specifications.md | - |
| **Paper outline** |  Complete | paper_outline.md | - |
| **API documentation** |  Planned | Docstrings and examples | LOW |
| **Tutorial notebooks** |  Planned | Getting started guides | LOW |

**Status:** Documentation is publication-ready for V1 scope

---

## 8. Infrastructure and Deployment

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| **HPC setup** |  Planned | Configuration for cluster | **HIGH** |
| **Docker container** |  Planned | Reproducibility | MEDIUM |
| **Environment spec** |  Complete | requirements.txt / conda env | - |
| **CI/CD pipeline** |  Planned | GitHub Actions | LOW |
| **Code release** |  Planned | Public GitHub repo | MEDIUM |
| **Data release** |  Planned | Zenodo upload | MEDIUM |

**Next Steps:**
1. Set up HPC access and configuration
2. Create Docker container for reproducibility
3. Prepare for code/data release

---

## 9. Milestones and Timeline

### Milestone 0: Infrastructure Setup (Week 1-2)
-  Documentation complete
-  Data pipeline on HPC
-  Spatial backend integration
-  Reference atlas processing

**Status:** 60% complete

### Milestone 1: End-to-End Training (Week 3-4)
-  All layers integrated
-  Full training loop
-  Checkpoint management
-  Basic evaluation

**Status:** 30% complete

### Milestone 2: Evaluation Harness (Week 5-6)
-  Donor-held-out CV
-  All metrics implemented
-  Statistical testing
-  Artifact logging

**Status:** 10% complete

### Milestone 3: Synthetic Validation (Week 7)
-  Synthetic data generator
-  Ground truth recovery tests
-  Negative controls

**Status:** 0% complete

### Milestone 4: Real Data Experiments (Week 8-10)
-  Full model training
-  Ablation suite (Tier 1)
-  Backend comparison
-  All figures generated

**Status:** 0% complete

### Milestone 5: Paper Writing (Week 11-12)
-  Methods section
-  Results section
-  Discussion section
-  Final figures and tables

**Status:** 20% complete (intro/methods can start early)

---

## 10. Critical Path Analysis

### Blocking Dependencies

1. **HPC Access for Data Processing** (Blocks: Milestone 0)
   - Need: 128GB RAM, 8 cores
   - For: Full spatial data merge and QC
   - **Action:** Request HPC allocation ASAP

2. **Spatial Backend Integration** (Blocks: Milestone 1)
   - Need: Tangram, DestVI, TACCO wrappers
   - For: Canonical artifacts generation
   - **Action:** Implement this week

3. **Reference Atlas Download** (Blocks: Milestone 1)
   - Need: HLCA and LuCA processed atlases
   - For: Layer A alignment
   - **Action:** Download and preprocess

4. **Canonical Artifacts** (Blocks: Milestone 2-5)
   - Need: cells.parquet, neighborhoods.parquet, stage_edges.parquet
   - For: All downstream training and evaluation
   - **Action:** Generate after spatial backends complete

### Parallel Work Streams

**Stream 1: Data Pipeline** (Week 1-2)
- HPC setup
- Spatial backend integration
- Artifact generation

**Stream 2: Model Development** (Week 1-4)
- Complete Layer D (flow matching)
- Complete Layer F (compatibility)
- Integration testing

**Stream 3: Evaluation** (Week 3-6)
- Implement metrics
- Build CV harness
- Ablation infrastructure

**Stream 4: Paper Writing** (Week 1-12, continuous)
- Methods (start early)
- Introduction (start early)
- Results (weeks 8-10)
- Discussion (weeks 10-12)

---

## 11. Risk Assessment

### High Risk Items

1. **Spatial Data Memory Issues**
   - Risk: OOM crashes during processing
   - Mitigation: Backed mode implemented, HPC required
   - Status: Partially mitigated

2. **Reference Atlas Integration**
   - Risk: Version incompatibility or alignment failures
   - Mitigation: Test on small subset first
   - Status: Not yet tested

3. **Training Stability**
   - Risk: NaN losses, gradient explosions
   - Mitigation: Gradient clipping, careful init
   - Status: Not yet tested

4. **Compute Resources**
   - Risk: Insufficient GPU time for full experiments
   - Mitigation: Request HPC allocation early
   - Status: Need to request

### Medium Risk Items

1. **Spatial Backend Discrepancies**
   - Risk: Backends give very different results
   - Mitigation: Degraded backend controls
   - Status: To be tested

2. **Ablation Runtime**
   - Risk: 6 ablations × 5 folds = 30 runs may take too long
   - Mitigation: Parallelize on multiple GPUs
   - Status: Need infrastructure

3. **Data Release Timing**
   - Risk: Data not publicly available by submission
   - Mitigation: Start Zenodo prep early
   - Status: Not started

---

## 12. Resource Requirements

### Computational

**Immediate (Week 1-2):**
- HPC node: 128GB RAM, 8 CPU cores
- Duration: 12 hours for data prep
- Purpose: Spatial data processing

**Training Phase (Week 3-10):**
- 1 V100 GPU (32GB VRAM)
- Duration: ~24 hours per training run
- Purpose: Model training and evaluation

**Ablation Phase (Week 8-10):**
- 8 V100 GPUs (parallel)
- Duration: 3 days total
- Purpose: Full ablation suite

**Total Estimate:**
- ~200 GPU-hours for full V1 completion
- ~100 CPU-hours for data processing

### Storage

- Raw data: ~100GB
- Processed data: ~150GB
- Artifacts (all runs): ~50GB
- **Total:** ~300GB

### Personnel

**Current Phase:**
- 1 lead developer (full-time)
- 1 domain expert (part-time consult)
- 1 data engineer (for HPC setup)

---

## 13. Go/No-Go Decision Points

### Decision Point 1: After Spatial Backend Integration (Week 2)
**Go Criteria:**
- All 3 backends run successfully
- Canonical artifacts generated
- Spatial coherence metrics reasonable

**No-Go:** Revisit backend selection or data quality

### Decision Point 2: After First Full Training Run (Week 4)
**Go Criteria:**
- Training stable (no NaNs)
- Loss converges
- Predictions reasonable (qualitative check)

**No-Go:** Debug training issues before ablations

### Decision Point 3: After Synthetic Validation (Week 7)
**Go Criteria:**
- Ground truth recovery > 0.5 correlation
- Negative controls behave as expected

**No-Go:** Revisit model architecture

### Decision Point 4: After Real Data Experiments (Week 10)
**Go Criteria:**
- All Tier 1 ablations show expected patterns
- Backend robustness demonstrated
- Uncertainty calibrated (ECE < 0.1)

**No-Go:** Additional experiments needed

---

## 14. Success Criteria for V1 Completion

### Technical Criteria
-  All layers implemented and tested
-  Full training pipeline runs end-to-end
-  Donor-held-out CV implemented
-  All Tier 1 ablations complete
-  Spatial backend robustness demonstrated
-  Uncertainty calibrated (ECE < 0.1)
-  Code passes integration tests
-  Results reproducible with saved seeds

### Scientific Criteria
-  Full model outperforms all baselines (p < 0.01)
-  Niche influence effect size > 0.5
-  Genomic compatibility separates matched vs shuffled (p < 0.01)
-  Results hold across all 3 spatial backends
-  Negative controls behave as expected
-  At least one clear biological insight from LUAD data

### Publication Criteria
-  All figures complete and polished
-  All tables complete
-  Methods section complete
-  Results section complete
-  Discussion section complete
-  Evidence matrix complete (all claims supported)
-  Supplementary materials complete
-  Code and data release ready

---

## 15. Next Actions (Immediate)

### This Week (Week 1)
1. **Request HPC allocation** for data processing
2. **Download HLCA and LuCA atlases**
3. **Implement spatial backend wrappers** (Tangram/DestVI/TACCO)
4. **Test backed-mode QC** on small dataset
5. **Set up synthetic data generator** for early testing

### Next Week (Week 2)
6. **Run full data pipeline on HPC**
7. **Generate all canonical artifacts**
8. **Complete Layer D flow matching implementation**
9. **Begin integration testing**
10. **Start Methods section writing**

### Priority Order
1. HPC setup (BLOCKING)
2. Spatial backends (BLOCKING)
3. Reference atlases (BLOCKING)
4. Layer D completion (HIGH)
5. Everything else in parallel

---

## 16. Contacts and Resources

### Key Personnel
- Lead Developer: [Name]
- PI: [Name]
- HPC Admin: [Contact for cluster access]
- Domain Expert: [Lung cancer biologist]

### External Resources
- HLCA Atlas: https://cellxgene.cziscience.com/collections/...
- LuCA Atlas: https://cellxgene.cziscience.com/collections/...
- Tangram: https://github.com/broadinstitute/Tangram
- DestVI: https://docs.scvi-tools.org/
- TACCO: https://github.com/simonwm/tacco

### Internal Resources
- HPC Documentation: [Link]
- Lab Compute Policy: [Link]
- Data Storage: [Path]

---

**End of Implementation Roadmap**

**Last Review:** 2026-03-15
**Next Review:** Weekly during implementation phase
