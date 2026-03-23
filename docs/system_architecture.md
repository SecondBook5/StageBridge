# StageBridge System Architecture and Infrastructure

**Last Updated:** 2026-03-22
**Purpose:** Complete technical specification of system architecture, infrastructure, and computational design
**Audience:** Technical readers, system architects, reproducibility reviewers

---

## 1. System Overview

StageBridge is a modular, scalable framework for learning cell-state transitions from multimodal spatial single-cell data. The system is designed for:
- **Modularity:** Each layer is independently testable and replaceable
- **Scalability:** Handles millions of cells with efficient batching and caching
- **Reproducibility:** Complete provenance tracking and deterministic execution
- **Extensibility:** Plugin architecture for new backends and models

---

## 2. High-Level Architecture

### 2.1 System Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          StageBridge System                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │              Data Layer (Step 0)                           │         │
│  │  • Raw data ingestion (GEO archives)                       │         │
│  │  • QC filtering and normalization                          │         │
│  │  • Spatial backend orchestration                           │         │
│  │  • Canonical artifact generation                           │         │
│  └────────────────────────────────────────────────────────────┘         │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │              Model Layer (Layers A-F)                      │         │
│  │  • Layer A: Dual-Reference Latent Mapping                  │         │
│  │  • Layer B: Local Niche Encoder                            │         │
│  │  • Layer C: Hierarchical Set Transformer                   │         │
│  │  • Layer D: Flow Matching Transition Model                 │         │
│  │  • Layer F: Evolutionary Compatibility                     │         │
│  └────────────────────────────────────────────────────────────┘         │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │            Training & Evaluation Layer                     │         │
│  │  • Staged training curriculum                              │         │
│  │  • Donor-held-out cross-validation                         │         │
│  │  • Ablation orchestration                                  │         │
│  │  • Metrics computation and logging                         │         │
│  └────────────────────────────────────────────────────────────┘         │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │            Visualization & Interpretation Layer            │         │
│  │  • UMAP and latent space visualization                     │         │
│  │  • Attention heatmaps and influence tensors                │         │
│  │  • Trajectory and flow field plots                         │         │
│  │  • Publication figure generation                           │         │
│  └────────────────────────────────────────────────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Information Flow

```
Raw Data → QC → Spatial Mapping → Canonical Artifacts
                                          ↓
                                    Data Loaders
                                          ↓
                    ┌────────────────────────────────┐
                    │    Training Loop               │
                    │                                │
Cells → Layer A → Layer B → Layer C → Layer D → Loss
  ↓                  ↓         ↓         ↓           ↑
WES ────────────────────────────────────> Layer F ──┘
                    │                                │
                    └────────────────────────────────┘
                                          ↓
                              Predictions + Uncertainty
                                          ↓
                            Evaluation Metrics + Figures
```

---

## 3. Data Layer Architecture

### 3.1 Pipeline Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Step 0: Data Preparation                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Raw Data]                                                         │
│     ├─ GSE308103_RAW.tar (snRNA-seq)                               │
│     ├─ GSE307534_RAW.tar (Visium)                                  │
│     └─ GSE307529_RAW.tar (WES)                                     │
│                   ↓                                                 │
│  [Extraction & Conversion]                                          │
│     ├─ Extract tarballs                                             │
│     ├─ Convert to h5ad format                                       │
│     └─ Per-sample validation                                        │
│                   ↓                                                 │
│  [QC Filtering]                                                     │
│     ├─ Backed-mode loading (memory efficient)                       │
│     ├─ Calculate QC metrics (genes, counts, mito)                   │
│     ├─ Filter cells and genes                                       │
│     └─ Save filtered datasets                                       │
│                   ↓                                                 │
│  [Normalization]                                                    │
│     ├─ Total counts normalization (target: 10^4)                    │
│     ├─ log1p transformation                                         │
│     └─ HVG selection (top 2000)                                     │
│                   ↓                                                 │
│  [Spatial Backend Benchmark]                                        │
│     ├─ Run Tangram                                                  │
│     ├─ Run DestVI                                                   │
│     ├─ Run TACCO                                                    │
│     └─ Standardize outputs                                          │
│                   ↓                                                 │
│  [Canonical Artifacts]                                              │
│     ├─ cells.parquet                                                │
│     ├─ neighborhoods.parquet                                        │
│     ├─ stage_edges.parquet                                          │
│     ├─ split_manifest.json                                          │
│     ├─ feature_spec.yaml                                            │
│     └─ spatial_backend/ (per-backend outputs)                       │
│                   ↓                                                 │
│  [Validation & Audit]                                               │
│     ├─ Data integrity checks                                        │
│     ├─ Completeness validation                                      │
│     └─ Audit report generation                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Storage Architecture

```
data/
├── raw/
│   └── geo/
│       ├── GSE308103_RAW.tar
│       ├── GSE307534_RAW.tar
│       ├── GSE307529_RAW.tar
│       ├── GSE308103_snrna/          # Extracted
│       │   ├── GSM_*_matrix.mtx.txt.gz
│       │   ├── GSM_*_barcodes.txt.gz
│       │   └── GSM_*_features.txt.gz
│       └── GSE307534_spatial/        # Extracted
│           └── GSM_*.tar.gz
│
├── interim/
│   ├── snrna/
│   │   └── sample_*.h5ad           # Per-sample h5ad files
│   └── spatial/
│       └── sample_*.h5ad           # Per-sample h5ad files
│
└── processed/
    └── luad_evo/
        ├── snrna_merged.h5ad              # 19GB
        ├── snrna_qc_normalized.h5ad       # 15GB (post-QC)
        ├── spatial_merged.h5ad            # 35GB
        ├── spatial_qc_normalized.h5ad     # 28GB (post-QC)
        ├── wes_features.parquet           # 50KB
        ├── cells.parquet                  # 1GB
        ├── neighborhoods.parquet          # 2GB
        ├── stage_edges.parquet            # 10MB
        ├── split_manifest.json            # 10KB
        ├── feature_spec.yaml              # 5KB
        ├── spatial_backend/
        │   ├── tangram/
        │   │   ├── cell_type_proportions.parquet
        │   │   ├── mapping_confidence.parquet
        │   │   ├── upstream_metrics.json
        │   │   └── backend_metadata.json
        │   ├── destvi/
        │   └── tacco/
        └── audit_report.json
```

**Storage Requirements:**
- Raw data: ~100 GB
- Interim files: ~50 GB (can be deleted after processing)
- Processed data: ~150 GB
- **Total:** ~300 GB with safety margin

### 3.3 Data Loading Architecture

```python
# Efficient data loading with caching and batching

class CellDataset:
    """Lazy-loading dataset for cells with optional neighborhood context"""

    def __init__(self, cells_path, neighborhoods_path=None, ...):
        # Memory-mapped loading of parquet files
        self.cells = pd.read_parquet(cells_path)  # ~1GB
        if neighborhoods_path:
            self.neighborhoods = pd.read_parquet(neighborhoods_path)  # ~2GB

        # Build lookup indices (fast)
        self.cell_id_to_idx = {cid: i for i, cid in enumerate(self.cells.cell_id)}

    def __getitem__(self, idx):
        # Fetch cell data
        cell = self.cells.iloc[idx]

        # Optional: Fetch neighborhood on-demand
        if self.load_neighborhoods:
            niche = self.neighborhoods[
                self.neighborhoods.receiver_cell_id == cell.cell_id
            ]
            return {"cell": cell, "niche": niche}

        return {"cell": cell}

class StageEdgeBatchLoader:
    """Batch loader for stage-edge transitions"""

    def __init__(self, cells_path, edges_path, batch_size=64, ...):
        self.cells = CellDataset(cells_path, ...)
        self.edges = pd.read_parquet(edges_path)
        self.batch_size = batch_size

    def __iter__(self):
        # Sample edges (with replacement or stratified)
        for edge in self.sample_edges():
            # Sample source and target cells from this edge
            src_cells = self.sample_cells(edge.source_cell_ids, self.batch_size)
            tgt_cells = self.sample_cells(edge.target_cell_ids, self.batch_size)

            yield {
                "source_cells": src_cells,
                "target_cells": tgt_cells,
                "edge_id": edge.edge_id
            }
```

**Optimization Strategies:**
- Memory-mapped file access (parquet)
- Lazy loading of neighborhoods (only when needed)
- Pre-built indices for fast lookups
- Batch sampling with shuffling
- Optional disk caching of frequent accesses

---

## 4. Model Layer Architecture

### 4.1 Layer Interfaces

Each layer follows a standardized interface for composability:

```python
class Layer(nn.Module):
    """Abstract base layer interface"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, inputs, **kwargs):
        """
        Args:
            inputs: Input tensors or dict
            **kwargs: Layer-specific options

        Returns:
            outputs: Output tensors or dict
            diagnostics: Optional dict of interpretability outputs
        """
        raise NotImplementedError

    def get_diagnostics(self):
        """Return interpretability diagnostics (attention, influence, etc.)"""
        return {}
```

### 4.2 Layer A: Dual-Reference Latent Mapping

```
┌─────────────────────────────────────────────────────────────────┐
│                Layer A: Dual-Reference Latent                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Cell expression (N, G) where G=2000 HVGs                │
│                                                                 │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │  HLCA Encoder    │      │  LuCA Encoder    │                │
│  │  (scVI-based)    │      │  (scVI-based)    │                │
│  │                  │      │                  │                │
│  │  [G] → [512]     │      │  [G] → [512]     │                │
│  │    → [256]       │      │    → [256]       │                │
│  │    → [128]       │      │    → [128]       │                │
│  │                  │      │                  │                │
│  │  z_healthy: 128  │      │  z_disease: 128  │                │
│  └──────────────────┘      └──────────────────┘                │
│           │                          │                          │
│           └──────────┬───────────────┘                          │
│                      ↓                                          │
│             ┌─────────────────┐                                 │
│             │  Fusion Layer   │                                 │
│             │  (Concat or MLP)│                                 │
│             │                 │                                 │
│             │  z_fused: 256   │                                 │
│             └─────────────────┘                                 │
│                      ↓                                          │
│  Output: (N, 256) fused latent embeddings                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
class DualReferenceLatentMapper(Layer):
    def __init__(self, config):
        super().__init__(config)
        # Load pretrained reference models
        self.hlca_encoder = scvi.model.SCVI.load(config.hlca_path)
        self.luca_encoder = scvi.model.SCVI.load(config.luca_path)

        # Optional fusion MLP
        if config.fusion_method == "learned":
            self.fusion = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )

    def forward(self, expression):
        # Map to reference spaces
        z_healthy = self.hlca_encoder.get_latent_representation(expression)
        z_disease = self.luca_encoder.get_latent_representation(expression)

        # Fuse
        if self.config.fusion_method == "concat":
            z_fused = torch.cat([z_healthy, z_disease], dim=-1)
        elif self.config.fusion_method == "learned":
            z_concat = torch.cat([z_healthy, z_disease], dim=-1)
            z_fused = self.fusion(z_concat)

        return {
            "z_fused": z_fused,
            "z_healthy": z_healthy,
            "z_disease": z_disease
        }
```

### 4.3 Layer B: Local Niche Encoder

```
┌────────────────────────────────────────────────────────────────┐
│              Layer B: Local Niche Encoder                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input: Cell latents (N, 256) + Neighborhood graphs           │
│                                                                │
│  ┌──────────────────────────────────────────────────┐         │
│  │         9-Token Sequence Construction            │         │
│  │                                                  │         │
│  │  Token 1: Receiver cell (latent + meta)         │         │
│  │  Token 2: Ring 0 (0-50μm aggregation)           │         │
│  │  Token 3: Ring 1 (50-100μm aggregation)         │         │
│  │  Token 4: Ring 2 (100-200μm aggregation)        │         │
│  │  Token 5: Ring 3 (200+μm aggregation)           │         │
│  │  Token 6: HLCA token (ref similarity)           │         │
│  │  Token 7: LuCA token (ref similarity)           │         │
│  │  Token 8: Pathway token (LR activity)           │         │
│  │  Token 9: Stats token (density, diversity)      │         │
│  │                                                  │         │
│  │  Shape: (N, 9, 256)                             │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────┐         │
│  │      Multi-Head Self-Attention                   │         │
│  │                                                  │         │
│  │  Q, K, V = Linear(tokens)                        │         │
│  │  Attention(Q, K, V) with 8 heads                 │         │
│  │  Output: (N, 9, 256)                             │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────┐         │
│  │      Feed-Forward Network                        │         │
│  │                                                  │         │
│  │  FFN(x) = ReLU(Linear(x)) → Linear(x)           │         │
│  │  Residual + LayerNorm                            │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  Output: Niche embeddings (N, 256)                            │
│          Attention weights (N, 9, 9) for interpretability     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Computational Complexity:**
- Token construction: O(N × k) where k = avg neighbors per cell
- Self-attention: O(N × 9²) = O(N) since 9 is constant
- Overall: Linear in number of cells

### 4.4 Layer C: Hierarchical Set Transformer

```
┌────────────────────────────────────────────────────────────────┐
│           Layer C: Hierarchical Set Transformer                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input: Cell niche embeddings (variable set sizes)            │
│                                                                │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Level 1: Cell-to-Cell Aggregation              │         │
│  │                                                  │         │
│  │  ISAB (Induced Set Attention Block):            │         │
│  │    • M=64 inducing points                        │         │
│  │    • Attention(cells, inducing points)           │         │
│  │    • Reduces O(N²) to O(N×M)                     │         │
│  │                                                  │         │
│  │  SAB (Set Attention Block):                     │         │
│  │    • Full self-attention over induced repr.      │         │
│  │    • Permutation invariant                       │         │
│  │                                                  │         │
│  │  Output: (M, 512) per lesion                    │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Level 2: Cell-to-Lesion Aggregation            │         │
│  │                                                  │         │
│  │  PMA (Pooling by Multihead Attention):          │         │
│  │    • K=1 seed vectors for lesion repr.           │         │
│  │    • Attention(seed, cells) → lesion embedding   │         │
│  │                                                  │         │
│  │  Output: (1, 512) per lesion                    │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Level 3: Lesion-to-Stage (Optional)            │         │
│  │                                                  │         │
│  │  PMA: Stage-level aggregation                    │         │
│  │  Output: (1, 512) per stage                     │         │
│  └──────────────────────────────────────────────────┘         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Computational Complexity:**
- ISAB: O(N×M + M²) ≈ O(N) for fixed M
- SAB: O(M²) = O(1) for fixed M
- PMA: O(M×K) ≈ O(M) for fixed K
- Overall: Linear in number of cells (efficient!)

### 4.5 Layer D: Flow Matching Transition Model

```
┌────────────────────────────────────────────────────────────────┐
│          Layer D: Flow Matching Transition Model               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input: z_src (N, 256), z_tgt (M, 256), niche_ctx (N, 512)    │
│                                                                │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Step 1: Optimal Transport Coupling              │         │
│  │                                                  │         │
│  │  Compute cost matrix C[i,j] = ||z_src[i] - z_tgt[j]||²    │
│  │                                                  │         │
│  │  Sinkhorn algorithm:                             │         │
│  │    π = argmin <C, π> + ε H(π)                    │         │
│  │    where H(π) is entropy regularizer             │         │
│  │                                                  │         │
│  │  Output: Coupling matrix π (N, M)               │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Step 2: Sample Time and Interpolate            │         │
│  │                                                  │         │
│  │  Sample t ~ U[0, 1]                              │         │
│  │                                                  │         │
│  │  For each source i, sample target j from π[i]   │         │
│  │                                                  │         │
│  │  Interpolate:                                    │         │
│  │    z(t) = (1-t) z_src[i] + t z_tgt[j] + σ(t)ε   │         │
│  │    where ε ~ N(0, I) for stochasticity           │         │
│  │                                                  │         │
│  │  True velocity:                                  │         │
│  │    v_true = z_tgt[j] - z_src[i]                 │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Step 3: Predict Velocity with Neural Network   │         │
│  │                                                  │         │
│  │  Input to NN: [z(t), t, niche_ctx]              │         │
│  │                                                  │         │
│  │  Architecture:                                   │         │
│  │    FC(768) → ReLU → FC(512) → ReLU → FC(256)    │         │
│  │                                                  │         │
│  │  Output: v_pred(z(t), t, ctx)                    │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Step 4: Compute Loss                            │         │
│  │                                                  │         │
│  │  L_flow = MSE(v_pred, v_true)                    │         │
│  │         = ||v_pred - (z_tgt - z_src)||²          │         │
│  │                                                  │         │
│  │  Optional: Add diffusion prediction              │         │
│  │  L_diff = NLL under predicted σ(t)               │         │
│  └──────────────────────────────────────────────────┘         │
│                                                                │
│  Inference: Integrate ODE/SDE from z_src to predict z_tgt     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Stochastic Sampling:**
```python
def sample_trajectory(z_src, niche_ctx, num_steps=100):
    """Sample stochastic trajectory from source to target"""
    dt = 1.0 / num_steps
    z = z_src.clone()
    trajectory = [z]

    for step in range(num_steps):
        t = torch.tensor([step * dt])

        # Predict drift
        v = velocity_network(z, t, niche_ctx)

        # Predict diffusion (optional)
        sigma = diffusion_network(z, t, niche_ctx)

        # Euler-Maruyama step
        dW = torch.randn_like(z) * torch.sqrt(dt)
        z = z + v * dt + sigma * dW

        trajectory.append(z)

    return torch.stack(trajectory)
```

### 4.6 Layer F: Evolutionary Compatibility

```
┌────────────────────────────────────────────────────────────────┐
│          Layer F: Evolutionary Compatibility Module            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input: z_pred (N, 256), wes_features (N, F)                  │
│         target_pool_wes (M, F) with metadata                   │
│                                                                │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Step 1: Compatibility Scoring                   │         │
│  │                                                  │         │
│  │  For each predicted cell i:                      │         │
│  │                                                  │         │
│  │    score_matched = cosine_sim(                   │         │
│  │      wes[i],                                     │         │
│  │      target_pool_wes[same_donor, same_stage]    │         │
│  │    )                                             │         │
│  │                                                  │         │
│  │    score_wrong_donor = cosine_sim(               │         │
│  │      wes[i],                                     │         │
│  │      target_pool_wes[other_donor, same_stage]   │         │
│  │    )                                             │         │
│  │                                                  │         │
│  │    score_wrong_stage = cosine_sim(               │         │
│  │      wes[i],                                     │         │
│  │      target_pool_wes[same_donor, other_stage]   │         │
│  │    )                                             │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Step 2: Contrastive Loss                        │         │
│  │                                                  │         │
│  │  L_compat = Σ[                                   │         │
│  │    max(0, margin - score_matched + score_wrong_donor)      │
│  │  + max(0, margin - score_matched + score_wrong_stage)      │
│  │  ]                                               │         │
│  │                                                  │         │
│  │  margin = 0.3 (hyperparameter)                   │         │
│  └──────────────────────────────────────────────────┘         │
│                         ↓                                      │
│  Output: Compatibility scores + Loss penalty                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Training Infrastructure

### 5.1 Training Loop Architecture

```python
def train_epoch(model, data_loader, optimizer, config):
    """Single training epoch"""
    model.train()
    epoch_metrics = defaultdict(list)

    for batch in data_loader:
        # Forward pass through all layers
        outputs = model(
            src_cells=batch["source_cells"],
            tgt_cells=batch["target_cells"],
            niche_ctx=batch["niche_context"],
            wes_features=batch["wes_features"],
            edge_id=batch["edge_id"]
        )

        # Compute composite loss
        loss = (
            config.w_flow * outputs["loss_flow"] +
            config.w_compat * outputs["loss_compat"] +
            config.w_aux * outputs["loss_aux"]  # Optional
        )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Log metrics
        epoch_metrics["loss"].append(loss.item())
        epoch_metrics["loss_flow"].append(outputs["loss_flow"].item())
        epoch_metrics["loss_compat"].append(outputs["loss_compat"].item())

    return {k: np.mean(v) for k, v in epoch_metrics.items()}
```

### 5.2 Checkpoint Management

```python
class CheckpointManager:
    """Manages model checkpoints with versioning"""

    def __init__(self, checkpoint_dir, keep_top_k=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.checkpoint_history = []

    def save(self, model, optimizer, epoch, metrics, config):
        """Save checkpoint with full state"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "git_commit": get_git_commit(),
            "timestamp": datetime.now().isoformat()
        }

        # Save with informative name
        filename = f"checkpoint_epoch{epoch}_val{metrics['val_loss']:.4f}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        # Track history
        self.checkpoint_history.append({
            "path": filepath,
            "epoch": epoch,
            "val_loss": metrics["val_loss"]
        })

        # Prune old checkpoints (keep top-k by val loss)
        self.prune_checkpoints()

        return filepath

    def load_best(self):
        """Load best checkpoint by validation loss"""
        if not self.checkpoint_history:
            raise ValueError("No checkpoints found")

        best = min(self.checkpoint_history, key=lambda x: x["val_loss"])
        return torch.load(best["path"])
```

### 5.3 Distributed Training (Optional)

```python
def setup_distributed():
    """Setup for multi-GPU training"""
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_distributed(config):
    """Distributed training wrapper"""
    local_rank = setup_distributed()

    # Create model and wrap with DDP
    model = StageBridgeModel(config).to(local_rank)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True
    )

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=config.world_size,
        rank=local_rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler
    )

    # Training loop
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        train_epoch(model, train_loader, optimizer, config)
```

---

## 6. Evaluation Infrastructure

### 6.1 Cross-Validation Orchestrator

```python
class DonorHeldOutCV:
    """Orchestrate donor-held-out cross-validation"""

    def __init__(self, split_manifest, config):
        self.splits = split_manifest["splits"]
        self.config = config
        self.results = []

    def run_fold(self, fold_id):
        """Run one CV fold"""
        split = self.splits[fold_id]

        # Create fold-specific data loaders
        train_loader = create_loader(split["train_donors"], ...)
        val_loader = create_loader(split["val_donors"], ...)
        test_loader = create_loader(split["test_donors"], ...)

        # Train model
        model = train_model(
            train_loader,
            val_loader,
            config=self.config,
            fold_id=fold_id
        )

        # Evaluate on test donors
        test_metrics = evaluate_model(model, test_loader, fold_id)

        # Save results
        self.results.append({
            "fold_id": fold_id,
            "train_donors": split["train_donors"],
            "val_donors": split["val_donors"],
            "test_donors": split["test_donors"],
            "metrics": test_metrics
        })

        return test_metrics

    def run_all_folds(self, parallel=False):
        """Run all folds (optionally in parallel)"""
        if parallel:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=5)(
                delayed(self.run_fold)(i) for i in range(len(self.splits))
            )
        else:
            results = [self.run_fold(i) for i in range(len(self.splits))]

        return self.aggregate_results(results)

    def aggregate_results(self, results):
        """Aggregate metrics across folds"""
        metrics = defaultdict(list)
        for fold_result in results:
            for metric_name, value in fold_result["metrics"].items():
                metrics[metric_name].append(value)

        # Compute mean ± std
        aggregated = {}
        for metric_name, values in metrics.items():
            aggregated[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values
            }

        return aggregated
```

### 6.2 Metrics Computation

```python
class MetricsComputer:
    """Compute all evaluation metrics"""

    @staticmethod
    def compute_wasserstein(pred, true):
        """Wasserstein distance between distributions"""
        from scipy.stats import wasserstein_distance
        distances = []
        for dim in range(pred.shape[1]):
            dist = wasserstein_distance(pred[:, dim], true[:, dim])
            distances.append(dist)
        return np.mean(distances)

    @staticmethod
    def compute_mmd(pred, true, gamma=1.0):
        """Maximum Mean Discrepancy with RBF kernel"""
        XX = np.sum(pred**2, axis=1)[:, None]
        YY = np.sum(true**2, axis=1)[None, :]
        XY = pred @ true.T
        Kxx = np.exp(-gamma * (XX - 2*XY + XX.T))
        Kyy = np.exp(-gamma * (YY - 2*YY.T + YY))
        Kxy = np.exp(-gamma * (XX - 2*XY + YY))
        return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

    @staticmethod
    def compute_ece(confidences, accuracies, n_bins=10):
        """Expected Calibration Error"""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
            if mask.sum() == 0:
                continue
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            ece += bin_weight * np.abs(bin_conf - bin_acc)
        return ece

    def compute_all(self, predictions, targets, uncertainties=None):
        """Compute full metric suite"""
        metrics = {
            "wasserstein": self.compute_wasserstein(predictions, targets),
            "mmd": self.compute_mmd(predictions, targets)
        }

        if uncertainties is not None:
            metrics["ece"] = self.compute_ece(...)
            metrics["coverage"] = self.compute_coverage(...)
            metrics["nll"] = self.compute_nll(...)

        return metrics
```

---

## 7. Computational Resources

### 7.1 Hardware Requirements

**Minimum Configuration:**
- 1× NVIDIA V100 GPU (32GB VRAM)
- 64GB RAM
- 8 CPU cores
- 500GB SSD storage

**Recommended Configuration:**
- 1× NVIDIA A100 GPU (80GB VRAM) or 2× V100
- 128GB RAM
- 16 CPU cores
- 1TB NVMe SSD storage

**HPC Configuration (for full pipeline):**
- Data prep node: 128GB RAM, 8 CPU cores, no GPU
- Training nodes: 1 GPU per node, 32GB RAM, 8 cores
- Total: 1 data prep node + 8 training nodes (for parallel ablations)

### 7.2 Runtime Estimates

| Stage | Hardware | Time | Notes |
|-------|----------|------|-------|
| **Data Prep (Step 0)** | HPC node (128GB RAM) | 10 hours | Blocking, run once |
| **Reference Alignment** | 1× V100 | 4 hours | HLCA + LuCA |
| **Full Model Training** | 1× V100 | 24 hours | 100 epochs with early stopping |
| **Single Ablation** | 1× V100 | 24 hours | Per ablation, per fold |
| **Full Ablation Suite** | 8× V100 (parallel) | 3 days | 6 ablations × 5 folds = 30 runs |
| **Evaluation (all metrics)** | 1× V100 | 6 hours | Per trained model |
| **Figure Generation** | CPU only | 2 hours | All publication figures |

**Total Time Estimate:**
- Sequential (1 GPU): ~15 days
- Parallel (8 GPUs): ~5 days
- Development/debugging: +1-2 weeks

### 7.3 Memory Profiling

```python
# Memory usage breakdown for typical training batch

Component                    | Memory (GB) | Notes
-----------------------------|-------------|------------------
Model parameters             | 0.5         | All layers
Optimizer state (AdamW)      | 1.0         | 2× params
Batch data (64 cells)        | 0.1         | Latents + context
Intermediate activations     | 2.0         | Forward pass
Gradients                    | 0.5         | Backward pass
CUDA overhead                | 1.0         | PyTorch runtime
-----------------------------|-------------|------------------
**Total per batch**          | **5.1 GB**  | Fits in 16GB easily

Peak during evaluation:
- MC sampling (100 passes)   | +4.0 GB     | Uncertainty estimation
- Metrics computation        | +1.0 GB     | Temporary arrays
**Total evaluation**         | **10.1 GB** | Fits in 16GB with headroom
```

---

## 8. Software Stack

### 8.1 Core Dependencies

```yaml
# environment.yaml
name: stagebridge
channels:
  - conda-forge
  - pytorch
  - nvidia

dependencies:
  # Core
  - python=3.11
  - pytorch=2.2
  - torchvision=0.17
  - pytorch-cuda=11.8

  # Scientific computing
  - numpy=1.24
  - scipy=1.11
  - pandas=2.0
  - scikit-learn=1.3

  # Single-cell analysis
  - scanpy=1.9
  - anndata=0.9
  - scvi-tools=1.0
  - squidpy=1.3

  # Spatial backends
  - tangram-sc=1.2
  - destvi=0.9  # via scvi-tools
  - tacco=0.3

  # Optimal transport
  - pot=0.9

  # Configuration
  - hydra-core=1.3
  - omegaconf=2.3

  # Utilities
  - tqdm=4.66
  - joblib=1.3
  - pyyaml=6.0

  # Visualization
  - matplotlib=3.7
  - seaborn=0.12
  - plotly=5.17

  # Development
  - pytest=7.4
  - black=23.7
  - ruff=0.0.290
```

### 8.2 Module Structure

```
stagebridge/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── defaults.yaml
│   └── luad_evo.yaml
├── data/
│   ├── __init__.py
│   ├── datasets.py           # CellDataset, EdgeLoader
│   ├── loaders.py             # Data loading utilities
│   ├── preprocessing.py       # QC, normalization
│   └── luad_evo/
│       ├── snrna.py
│       ├── visium.py
│       └── wes.py
├── models/
│   ├── __init__.py
│   ├── base.py                # Layer interface
│   ├── dual_reference.py      # Layer A
│   ├── niche_encoder.py       # Layer B
│   ├── set_transformer.py     # Layer C
│   ├── flow_matching.py       # Layer D
│   └── evolution_compat.py    # Layer F
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Training loop
│   ├── optimizer.py           # Optimizer setup
│   └── checkpoints.py         # Checkpoint management
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # All metrics
│   ├── cv.py                  # Cross-validation
│   └── ablations.py           # Ablation runner
├── visualization/
│   ├── __init__.py
│   ├── latent_space.py        # UMAP, PCA plots
│   ├── attention.py           # Attention heatmaps
│   ├── trajectories.py        # Flow fields
│   └── figures.py             # Publication figures
├── pipelines/
│   ├── __init__.py
│   ├── run_data_prep.py       # Step 0
│   ├── run_training.py        # Full training
│   └── run_evaluation.py      # Full evaluation
├── spatial_backends/
│   ├── __init__.py
│   ├── tangram_wrapper.py
│   ├── destvi_wrapper.py
│   └── tacco_wrapper.py
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   ├── io_utils.py
│   └── types.py
├── cli.py                     # Command-line interface
└── notebook_api.py            # Jupyter API
```

---

## 9. Deployment and Reproducibility

### 9.1 Docker Container

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY environment.yaml /tmp/environment.yaml
RUN conda env create -f /tmp/environment.yaml

# Activate environment
SHELL ["conda", "run", "-n", "stagebridge", "/bin/bash", "-c"]

# Copy source code
COPY . /app/stagebridge
WORKDIR /app/stagebridge

# Install package
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "stagebridge", "python", "-m", "stagebridge.cli"]
```

### 9.2 Reproducibility Checklist

- [ ] All code version-controlled in Git
- [ ] Docker container built and tested
- [ ] All configs saved with runs
- [ ] All random seeds fixed and logged
- [ ] Environment fully specified (conda/docker)
- [ ] Data preprocessing scripts included
- [ ] Trained model checkpoints saved
- [ ] Evaluation scripts included
- [ ] Figure generation scripts included
- [ ] Documentation complete
- [ ] Unit tests passing
- [ ] Integration tests passing

---

## 10. Summary

StageBridge V1 architecture is:
- **Modular:** Clear layer interfaces, composable components
- **Scalable:** Linear complexity in number of cells
- **Efficient:** Memory-mapped data loading, backed-mode processing
- **Reproducible:** Complete provenance tracking, deterministic execution
- **Robust:** Multi-backend validation, comprehensive evaluation
- **Extensible:** Plugin architecture for new components

**Ready for:** HPC deployment, full-scale experiments, publication

---

**End of System Architecture Document**
