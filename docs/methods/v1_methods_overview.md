# StageBridge V1 Methods Overview

## Publication-Ready Technical Specification

**Last Updated:** 2026-03-15
**Status:** V1-Minimal Scope
**Target:** First publication

---

## 1. Overview

StageBridge is a multiscale stochastic transformer framework for learning cell-state transitions under spatial and multimodal constraints. Version 1 (V1-Minimal) implements the core architecture required for the first publication, focusing on cell-level transition modeling with evolutionary compatibility constraints.

### 1.1 Core Innovation

Cross-sectional stage transitions become more identifiable when modeled in:
- **Dual-reference geometry** (healthy + disease anchors)
- **Local niche influence** (spatial neighborhood context)
- **Stochastic dynamics** (flow matching with uncertainty)
- **Evolutionary constraints** (genomic compatibility)

### 1.2 V1 Scope

V1 consists of exactly these components:
- Raw data pipeline (Step 0)
- Spatial backend benchmark (Tangram/DestVI/TACCO)
- Dual-reference latent mapping (HLCA + LuCA, Euclidean)
- Local niche encoder (EA-MIST Layer B)
- Hierarchical set transformer (EA-MIST Layer C)
- Flow matching transition model (OT-CFM)
- Evolutionary compatibility regularizer
- Donor-held-out evaluation with uncertainty quantification
- Tier 1 ablation suite

### 1.3 V1 Explicit Non-Goals

Deferred to V2/V3:
- Non-Euclidean geometry (hyperbolic/spherical)
- Neural SDE backend
- Phase portrait / attractor decoder
- Cohort transport layer
- Destination-conditioned transitions

---

## 2. Architecture

### 2.1 Four-Layer Design

```
Input: Cell expression + spatial coordinates + genomics (optional)
  ↓
Layer A: Dual-Reference Latent Mapping (HLCA + LuCA)
  → Euclidean embeddings in healthy and disease space
  ↓
Layer B: Local Niche Encoder (9-token EA-MIST)
  → Receiver cell + 4 distance rings + HLCA + LuCA + Pathway + Stats
  ↓
Layer C: Hierarchical Set Transformer (ISAB/SAB/PMA)
  → Lesion-level and stage-level aggregation
  ↓
Layer D: Flow Matching Transition Model (OT-CFM)
  → Stochastic cell-state transitions with Sinkhorn coupling
  ↓
Layer F: Evolutionary Compatibility (WES regularizer)
  → Genomic constraints on transition plausibility
  ↓
Output: Target cell distributions + uncertainty + compatibility scores
```

### 2.2 Layer A: Dual-Reference Latent Mapping

**Purpose:** Map cells into structured latent space using healthy and disease references.

**V1 Implementation:** Euclidean embeddings

**Inputs:**
- Normalized gene expression (log1p, scaled)
- Cell type annotations (if available)

**References:**
- HLCA (Human Lung Cell Atlas) for healthy lung structure
- LuCA (Lung Cancer Atlas) for disease-specific patterns

**Outputs:**
- `z_healthy`: Euclidean embedding in HLCA space (dim: 64-128)
- `z_disease`: Euclidean embedding in LuCA space (dim: 64-128)
- `z_fused`: Concatenated or learned fusion (dim: 128-256)

**Technical Details:**
- Reference alignment via scVI or scANVI
- Euclidean distance metrics for V1
- Optional contrastive pretraining
- Batch correction at reference level

**V2 Upgrade Path:**
- Hyperspherical embedding for healthy manifold
- Hyperbolic embedding for disease branching
- Learned coordinate fusion with Riemannian geodesics

### 2.3 Layer B: Local Niche Encoder

**Purpose:** Encode spatial neighborhood context as 9-token representation.

**V1 Implementation:** EA-MIST `LocalNicheTransformerEncoder`

**9-Token Design:**
1. **Receiver token:** Target cell state
2-5. **Ring tokens:** 4 distance-binned neighborhood rings
6. **HLCA token:** Healthy reference similarity aggregate
7. **LuCA token:** Disease reference similarity aggregate
8. **Pathway token:** Ligand-receptor or pathway activity
9. **Stats token:** Neighborhood statistics (density, diversity, etc.)

**Architecture:**
- Self-attention over 9 tokens
- Positional encoding for spatial structure
- Optional prototype bottleneck for compression

**Inputs:**
- Cell latent states from Layer A
- Spatial coordinates or neighborhood graphs
- Reference similarity scores
- Optional pathway annotations

**Outputs:**
- Niche embedding per receiver cell (dim: 256-512)
- Attention weights (for interpretability)
- Optional influence tensor (sender → receiver attribution)

**Technical Details:**
- K-nearest neighbor graphs (k=50-200) or radius-based
- Distance-binned rings for multiscale context
- Permutation-invariant aggregation within rings
- Dropout and layer norm for stability

### 2.4 Layer C: Hierarchical Set Transformer

**Purpose:** Aggregate cell neighborhoods into lesion and stage representations.

**V1 Implementation:** EA-MIST set encoder (ISAB/SAB/PMA)

**Architecture Blocks:**
- **ISAB** (Induced Set Attention Block): Inducing-point attention for efficiency
- **SAB** (Set Attention Block): Full set attention
- **PMA** (Pooling by Multihead Attention): Learned pooling to fixed size

**Hierarchy:**
```
Cells (with niche context from Layer B)
  → ISAB (inducing points for efficiency)
  → SAB (self-attention over set)
  → PMA (pool to lesion representation)
  → [Optional] Second-level pooling to stage/donor representation
```

**Inputs:**
- Niche embeddings from Layer B (variable set size)
- Lesion/stage/donor metadata

**Outputs:**
- Lesion-level embedding (dim: 256-512)
- Optional stage-level embedding
- Set membership indicators

**Technical Details:**
- Permutation invariance by design
- Handles variable set sizes
- Inducing points reduce O(n²) to O(nm) complexity
- Number of inducing points: 32-128

**V1 Use Cases:**
- Hierarchical context for transition model
- Optional auxiliary lesion classification (not primary loss)
- Donor-level aggregation for evaluation

### 2.5 Layer D: Flow Matching Transition Model

**Purpose:** Model cell-state transitions as stochastic conditional flows.

**V1 Implementation:** Optimal Transport Conditional Flow Matching (OT-CFM)

**Mathematical Framework:**

Given source distribution X_src and target distribution X_tgt:

1. **Sinkhorn Coupling:**
   ```
   π = argmin_π <C, π> + ε H(π)
   where C_ij = ||x_src[i] - x_tgt[j]||²
   ```

2. **Flow Interpolation:**
   ```
   z(t) = (1-t)x_src + t x_tgt + σ(t)ε
   where t ∈ [0,1], ε ~ N(0,I)
   ```

3. **Conditional Flow:**
   ```
   dz/dt = v_θ(z(t), t, context)
   where context = niche embedding from Layers B/C
   ```

4. **Training Objective:**
   ```
   L = E_t,π [(v_θ(z(t), t, ctx) - (x_tgt - x_src))²]
   ```

**Inputs:**
- Source cell latent (from Layer A)
- Target cell latent or target stage distribution
- Niche context (from Layers B/C)
- Stage-edge condition (e.g., AIS → MIA)
- Optional genomic features

**Outputs:**
- Predicted target distribution
- Drift field v(z,t)
- Diffusion scale (uncertainty estimate)
- Transition probability or log-likelihood

**Technical Details:**
- Sinkhorn epsilon: 0.01-0.1
- Sinkhorn iterations: 50-100
- Time sampling: uniform t ~ U[0,1]
- Integration: Euler or Euler-Maruyama
- Number of stochastic passes for uncertainty: 10-100

**Stochastic Sampling:**
```python
def sample_trajectory(z_src, context, num_steps=100):
    trajectory = [z_src]
    z = z_src
    dt = 1.0 / num_steps
    for t in np.linspace(0, 1, num_steps):
        drift = model.predict_velocity(z, t, context)
        diffusion = model.predict_diffusion(z, t, context)
        z = z + drift * dt + diffusion * np.sqrt(dt) * randn()
        trajectory.append(z)
    return trajectory
```

**V2 Upgrade Path:**
- Neural SDE with state-dependent diffusion
- Score matching objective
- Full SDE integration with adaptive timesteps

### 2.6 Layer F: Evolutionary Compatibility Module

**Purpose:** Constrain transitions by genomic/clonal compatibility.

**V1 Implementation:** Existing WES regularizer

**Compatibility Scoring:**

For each predicted transition (cell_i in stage_s → stage_t):

1. **Matched Donor/Stage:**
   ```
   score_match = similarity(wes_i, wes_target_pool[stage_t, donor_i])
   ```

2. **Mismatched Negatives:**
   ```
   score_wrong_stage = similarity(wes_i, wes_target_pool[stage_other])
   score_wrong_donor = similarity(wes_i, wes_target_pool[donor_other])
   ```

3. **Compatibility Loss:**
   ```
   L_compat = max(0, margin - score_match + score_wrong_stage)
            + max(0, margin - score_match + score_wrong_donor)
   ```

**Inputs:**
- WES features (mutation burden, signature, clonality)
- Source cell state
- Predicted target state
- Target stage/donor metadata

**Outputs:**
- Compatibility score (higher = more compatible)
- Compatibility penalty (for training)
- Diagnostic matched vs mismatched statistics

**Technical Details:**
- WES features: TMB, signature weights, clone labels
- Similarity metric: cosine or learned MLP
- Margin: 0.1-0.5
- Regularization weight: 0.01-0.1
- Graceful no-op when genomics unavailable

**Required Controls:**
- Matched vs shuffled donor
- Matched vs shuffled stage
- With vs without genomics

---

## 3. Training Protocol

### 3.1 Staged Training (V1 Curriculum)

**Stage 0: Raw Data Pipeline (Blocking)**
- Extract and merge snRNA, spatial, WES
- QC filtering and normalization
- Spatial backend benchmark
- Generate canonical artifacts
- **Duration:** 1-2 days (HPC required for full data)

**Stage 1: Reference Alignment**
- Train HLCA and LuCA alignment
- Validate reference anchoring
- **Objective:** Stable reference embeddings
- **Duration:** 2-4 hours per reference

**Stage 2: Niche Encoder Pretraining (Optional)**
- Train Layer B on niche composition prediction
- Or use contrastive pretraining
- **Objective:** Meaningful niche representations
- **Duration:** 4-8 hours

**Stage 3: Transition Model Training**
- Full model: Layers A→B→C→D→F
- Train with flow matching + compatibility loss
- **Objective:** Stable transition learning
- **Duration:** 12-24 hours

**Stage 4: Ablations and Evaluation**
- Run Tier 1 ablations (6 required)
- Donor-held-out evaluation
- Uncertainty calibration
- **Duration:** 2-3 days

### 3.2 Hyperparameters (V1 Defaults)

**Data:**
- Min genes per cell: 200
- Min cells per gene: 3
- Max pct mitochondrial: 20%
- Min counts per cell: 500
- Neighborhood k: 50-200
- Distance bins: [0-50, 50-100, 100-200, 200+] μm

**Architecture:**
- Latent dim (Layer A): 128
- Niche embedding dim (Layer B): 256
- Set embedding dim (Layer C): 512
- Transition model hidden: [512, 512, 256]
- Number of inducing points (Layer C): 64
- Number of attention heads: 8

**Training:**
- Batch size: 64-256 cells or 32-64 lesions
- Learning rate: 1e-4 (with warmup)
- Weight decay: 1e-5
- Optimizer: AdamW
- Scheduler: Cosine annealing
- Max epochs: 100-200
- Early stopping: 10-20 epochs
- Gradient clipping: 1.0

**Loss Weights:**
- Flow matching: 1.0
- Evolutionary compatibility: 0.05-0.1
- Auxiliary lesion classification: 0.01 (if used)

**Regularization:**
- Dropout: 0.1-0.2
- Layer norm: everywhere
- Gradient clipping: 1.0
- Label smoothing: 0.1 (for classification)

### 3.3 Data Splits (Donor-Held-Out)

**Strategy:** Donor-level cross-validation

**Splits:**
- Train donors: 70% (e.g., 12 donors)
- Validation donors: 15% (e.g., 3 donors)
- Test donors: 15% (e.g., 3 donors)

**Constraints:**
- All stages represented in each split
- Balanced stage distribution where possible
- Stratified by major clinical covariates

**Evaluation Edges:**
- Test on all stage-to-stage edges seen in training
- Report per-edge metrics separately
- Aggregate with donor-level bootstrapping

---

## 4. Evaluation Metrics

### 4.1 Cell-Level Transition Quality

**Primary Metrics:**
- **Wasserstein distance** between predicted and true target distributions
- **MMD** (Maximum Mean Discrepancy) with RBF kernel
- **KL divergence** (if distributions are normalized)

**Secondary Metrics:**
- Cosine similarity in latent space
- Euclidean distance in latent space
- Classification accuracy (if discrete targets)

**Baselines:**
- Deterministic mapping (no flow matching)
- No-context baseline (no niche influence)
- Mean-target baseline (predict stage mean)

**Success Criterion:**
V1 model must outperform all baselines on held-out donors.

### 4.2 Niche Influence Quality

**Metrics:**
- **Influence recovery** on synthetic benchmarks (ground truth available)
- **Attention entropy** (high = diffuse influence, low = specific)
- **Shuffle sensitivity:** Metric degradation when neighborhoods shuffled

**Interpretability Outputs:**
- Sender → receiver attention maps
- Per-cell-type influence weights
- Spatial influence heatmaps

**Success Criterion:**
- Synthetic influence recovery > pooled-context baseline
- Real-data shuffle sensitivity effect size > 0.3 SD

### 4.3 Uncertainty Quality

**Metrics:**
- **Expected Calibration Error (ECE):** Binned calibration
- **Negative Log-Likelihood (NLL):** Predictive likelihood
- **Coverage:** Fraction of true targets in prediction intervals
- **Interval width:** Average prediction uncertainty

**Controls:**
- Uncertainty should be higher on:
  - Wrong-stage edges
  - Shuffled neighborhoods
  - Held-out donors
  - Low-data regions

**Success Criterion:**
- ECE < 0.1
- Coverage matches nominal level (e.g., 90% coverage for 90% intervals)
- Uncertainty increases on negative controls

### 4.4 Evolutionary Compatibility Quality

**Metrics:**
- **Matched vs shuffled separation:** Mean compatibility difference
- **Effect size:** Cohen's d or Cliff's delta
- **Regularization impact:** Reduction in implausible transitions

**Controls:**
- Shuffled donor genomics
- Shuffled stage genomics
- Random genomic features

**Success Criterion:**
- Matched compatibility > shuffled controls (p < 0.01)
- Effect size > 0.5 SD
- Regularizer reduces wrong-stage/donor scores

### 4.5 Spatial Backend Robustness

**Metrics:**
- **Upstream quality:**
  - Cell type proportion accuracy (vs ground truth where available)
  - Spatial coherence metrics
  - Mapping confidence distributions

- **Downstream utility:**
  - Transition quality under each backend
  - Niche influence consistency across backends
  - Ablation effect sizes under each backend

**Backends (V1 Required):**
- Tangram
- DestVI
- TACCO

**Success Criterion:**
- Final biological conclusions hold across all 3 backends
- Canonical backend justified by quantitative comparison
- No unique dependence on one backend

---

## 5. Ablation Suite (Tier 1)

### 5.1 Required Ablations (V1)

1. **Stochastic vs Deterministic**
   - Full model (flow matching) vs deterministic regression
   - Metric: Uncertainty quality, distribution matching

2. **Niche Context Variants**
   - No niche vs pooled niche vs full 9-token niche
   - Metric: Transition quality, influence interpretability

3. **Genomics Integration**
   - No genomics vs genomics-as-feature vs genomics-as-constraint
   - Metric: Compatibility separation, implausible transition rate

4. **Set Aggregation**
   - Flat pooling vs hierarchical set transformer
   - Metric: Lesion-level quality, computational efficiency

5. **Reference Design**
   - HLCA only vs LuCA only vs dual reference
   - Metric: Latent space quality, transition identifiability

6. **Spatial Backend**
   - Canonical backend vs alternative backend(s)
   - Metric: Robustness of conclusions, upstream/downstream quality

### 5.2 Reporting Standards

For each ablation, report:
- Mean ± std across donor-held-out folds
- Effect size relative to full model (Cohen's d)
- Compute time delta
- Key figures showing qualitative difference

### 5.3 Evidence Matrix

Maintain mapping: **Claim → [Figure, Table, Ablation, Statistics]**

Example:
| Claim | Evidence |
|-------|----------|
| "Niche context improves transition quality" | Fig 3B, Table 3 row 2, Ablation #2, p<0.001 |
| "Genomics as constraint outperforms as feature" | Fig 5C, Table 3 row 3, Ablation #3, ES=0.7 |

---

## 6. Reproducibility

### 6.1 Artifact Logging (Every Run)

**Required artifacts:**
- `resolved_config.yaml`: Full config with all defaults
- `git_commit.txt`: Exact code version
- `seed.txt`: Random seed
- `split_manifest.json`: Train/val/test donor IDs
- `metrics.csv`: All metrics per epoch
- `diagnostics.json`: Model-specific diagnostics
- `checkpoint.pt`: Model weights
- `artifact_manifest.json`: Paths to all outputs

### 6.2 Environment Specification

```yaml
python: 3.11
pytorch: 2.2
cuda: 11.8
packages:
  - scanpy==1.9
  - scvi-tools==1.0
  - squidpy==1.3
  - hydra-core==1.3
  - pot==0.9  # optimal transport
  - pandas==2.0
  - numpy==1.24
  - scikit-learn==1.3
```

### 6.3 Computational Requirements

**Minimum:**
- 1 GPU (16GB+ VRAM)
- 64GB RAM for preprocessing
- 500GB disk for data + artifacts

**Recommended:**
- Multi-GPU for parallel ablations
- 128GB+ RAM for full dataset
- 1TB+ disk for all experiments

**HPC Requirements:**
- Step 0 (data prep): 128GB RAM, 8 CPU cores, 6-12 hours
- Training: 1 GPU, 24-48 hours per run
- Full ablation suite: 4-8 GPUs, 3-5 days

---

## 7. Implementation Status

### 7.1 Completed Components

- ✅ Layer A scaffolding (reference alignment structure exists)
- ✅ Layer B implementation (`LocalNicheTransformerEncoder`)
- ✅ Layer C implementation (`ISAB`, `SAB`, `PMA`)
- ✅ Layer D scaffolding (`stochastic_dynamics.py`)
- ✅ Layer F scaffolding (WES regularizer exists)
- ✅ Config system (Hydra-based)
- ✅ Basic data loaders

### 7.2 In-Progress Components

- 🔄 Step 0 data pipeline (run_data_prep.py)
- 🔄 Spatial backend benchmark loop
- 🔄 Full training script integration
- 🔄 Donor-held-out evaluation harness

### 7.3 Required for V1 Completion

- ❌ Canonical artifacts generation (cells.parquet, neighborhoods.parquet, etc.)
- ❌ Spatial backend standardization layer
- ❌ Tier 1 ablation scripts
- ❌ Evaluation and plotting utilities
- ❌ Documentation of all modules
- ❌ Integration tests
- ❌ Benchmark on synthetic data
- ❌ Final publication figures

---

## 8. Next Steps for Paper Preparation

### 8.1 Immediate (Week 1-2)

1. Complete Step 0 data pipeline
2. Generate all canonical artifacts
3. Run spatial backend benchmark
4. Validate flow matching implementation
5. Create synthetic test datasets

### 8.2 Short-term (Week 3-6)

6. Full model training on real data
7. Donor-held-out evaluation
8. Tier 1 ablations
9. Uncertainty calibration
10. Draft figures 1-4

### 8.3 Medium-term (Week 7-12)

11. Evolutionary compatibility validation
12. Spatial backend robustness analysis
13. Final figures and tables
14. Methods writing
15. Results writing

### 8.4 Paper Writing Parallel Track

- **Introduction:** Start now (can write before results)
- **Methods:** Start with architecture description (stable)
- **Results:** Requires completed experiments
- **Discussion:** Can draft framework early
- **Figures:** Iterative with results

---

## 9. Publication Claim (V1)

**Core Thesis:**

> Cell-state transitions in cross-sectional spatial and single-cell data become more identifiable when modeled in dual-reference geometry, conditioned on local niche influence, constrained by evolutionary compatibility, and shown to be robust across spatial mapping backends.

**Supporting Claims:**

1. Dual-reference geometry (HLCA + LuCA) provides better transition structure than single-reference
2. Local niche influence (9-token encoder) improves transition quality over pooled or no context
3. Stochastic flow matching better captures uncertainty than deterministic mapping
4. Genomic compatibility as constraint outperforms genomic features concatenated
5. Hierarchical set transformer enables interpretable lesion-level aggregation
6. Results are robust to spatial backend choice (Tangram/DestVI/TACCO)

**Success Criteria:**

V1 publication is ready when:
- All 6 supporting claims have quantitative evidence
- Evidence matrix is complete
- Donor-held-out validation shows generalization
- Uncertainty is calibrated and reported
- Spatial backend robustness is demonstrated
- Code is reproducible with saved configs and seeds
- All Tier 1 ablations are complete

---

## 10. Differentiation from Related Work

### 10.1 vs CellOracle, Dynamo, scVelo

**StageBridge V1 advances:**
- Explicit spatial niche conditioning (not just k-NN cell similarity)
- Dual-reference geometry for progression structure
- Evolutionary compatibility constraints
- Stochastic dynamics with uncertainty
- Multi-backend spatial mapping validation

### 10.2 vs Optimal Transport Methods (TrajectoryNet, CellOT)

**StageBridge V1 advances:**
- Niche-conditioned transitions (not just cell-cell OT)
- Hierarchical context aggregation
- Genomic compatibility regularization
- Spatial backend robustness requirement

### 10.3 vs Spatial Analysis Tools (Squidpy, SPATA, Giotto)

**StageBridge V1 advances:**
- Transition modeling as primary objective (not just spatial pattern discovery)
- Stochastic dynamics for uncertainty quantification
- Multi-reference geometry integration
- Evolutionary constraints

### 10.4 vs EA-MIST (Own Prior Work)

**StageBridge V1 advances:**
- Cell-level learning (not lesion-level classification)
- Stochastic transition model (not static MIL)
- Dual-reference latent space
- Evolutionary compatibility
- Spatial backend benchmark requirement

---

## References

- HLCA: Sikkema et al., Nature Medicine 2023
- LuCA: Salcher et al., Nature Medicine 2022
- OT-CFM: Tong et al., ICML 2024
- EA-MIST: (Internal, Layer B+C architecture)
- Tangram: Biancalani et al., Nature Methods 2021
- DestVI: Lopez et al., Nature Methods 2022
- TACCO: Roden et al., Nature Biotechnology 2022

---

**End of V1 Methods Overview**
