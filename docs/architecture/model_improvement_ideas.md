# Model Improvement Ideas from Literature Review

Consolidated insights from four cutting-edge papers to make StageBridge more impactful.

---

## 1. Latent Space Geometry (from scPhere)

### Problem with Standard VAE
- Gaussian prior leads to crowding near origin
- Poor preservation of hierarchical/branching structures

### Solution: Hyperspherical or Hyperbolic Latent Space

**Option A: von Mises-Fisher (vMF) prior on hypersphere**
- Uniform density on sphere surface prevents crowding
- Better batch correction while preserving biology
- Natural for data that lives on a manifold

**Option B: Hyperbolic (Lorentz model) latent space**
- Exponentially more volume at boundaries = good for hierarchies
- Better for branching trajectories (e.g., normal -> pre-malignant -> malignant)
- Preserves hierarchical cell type relationships

### Implementation for StageBridge
```python
# Instead of Gaussian latent:
# z ~ N(0, I)

# Use vMF latent:
# z ~ vMF(mu, kappa)  # kappa controls concentration
# Sample via rejection sampling or Householder transform

# Or hyperbolic (Lorentz):
# z in H^n (hyperboloid)
# Use exponential/logarithmic maps for encoder/decoder
```

**Priority: MEDIUM** - Significant architectural change, but could improve trajectory modeling.

---

## 2. Geometry-Preserving Transitions (from GeoBridge)

### Problem
- Transitions in expression space are nonlinear
- Linear interpolation in standard latent space doesn't follow true dynamics

### Solution: Invertible Neural Networks (INN)

**Key insight**: Use bijective mapping so that:
1. Forward: Expression -> Latent (isometric, preserves distances)
2. Inverse: Latent -> Expression (exact reconstruction)
3. Transitions become constant-velocity geodesics (straight lines) in latent space

### Implementation for StageBridge
```python
# Replace encoder/decoder with INN blocks (e.g., RealNVP, NICE, or Glow)
class InvertibleEncoder(nn.Module):
    def __init__(self):
        self.blocks = nn.ModuleList([
            CouplingBlock(...) for _ in range(n_blocks)
        ])
    
    def forward(self, x):
        z = x
        log_det = 0
        for block in self.blocks:
            z, ld = block(z)
            log_det += ld
        return z, log_det
    
    def inverse(self, z):
        x = z
        for block in reversed(self.blocks):
            x = block.inverse(x)
        return x
```

**Priority: HIGH** - Could significantly improve transition modeling (our core task).

---

## 3. Continual Learning for Reference Integration (from CRC paper)

### Problem
- Our dual-reference (HLCA/LuCA) integration is ad-hoc
- Adding new reference data could cause catastrophic forgetting

### Solution: Elastic Weight Consolidation (EWC) + Experience Replay

**EWC**: Penalize changes to weights important for previous tasks
```python
# EWC loss term
ewc_loss = 0
for name, param in model.named_parameters():
    fisher = fisher_information[name]  # Importance of each weight
    ewc_loss += (fisher * (param - old_params[name])**2).sum()

total_loss = task_loss + lambda_ewc * ewc_loss
```

**Experience Replay**: Keep buffer of representative cells from each reference
```python
# During training on new reference, also replay old reference cells
replay_buffer = select_representative_cells(
    reference_data, 
    n=buffer_size,
    strategy='bregman_information'  # Select high-uncertainty cells
)
```

### Implementation for StageBridge
- When mapping to HLCA first, then LuCA: use EWC to preserve HLCA structure
- Maintain replay buffer of healthy reference cells when training on cancer

**Priority: MEDIUM** - Addresses dual-reference fusion principled way.

---

## 4. Relative Representations (from CRC paper)

### Problem
- Absolute latent coordinates are not comparable across datasets
- Hard to link observational atlas to perturbation data

### Solution: Anchor-based Relative Coordinates

Instead of absolute z, compute similarity to anchor points (cell state archetypes):
```python
# Select anchor cells representing key states
anchors = select_archetypes(atlas, n_anchors=50)  # e.g., normal, pre-malignant, malignant

# For any cell, compute relative representation
def relative_repr(z, anchor_embeddings):
    """Compute cosine similarity to each anchor."""
    return F.cosine_similarity(z.unsqueeze(1), anchor_embeddings.unsqueeze(0), dim=-1)

# Transition quantification via Energy Distance
def energy_distance(P, Q):
    """Quantify shift between distributions."""
    return 2 * E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
```

### Implementation for StageBridge
- Define archetypes: Normal lung, AAH, AIS, MIA, Invasive LUAD
- Compute relative representations for all cells
- Track transitions as movements in archetype-relative space

**Priority: HIGH** - Directly enables stage transition quantification.

---

## 5. Pathway Regression Loss (from SpatialFusion) - IMPLEMENTED

### Problem
- Latent space may not capture biologically meaningful variation
- Similar cell type composition != similar functional state

### Solution: Auxiliary Pathway Prediction Loss

Add loss term predicting pathway activation (PROGENy scores):
```python
# Pathways relevant to lung cancer:
PATHWAYS = ['EGFR', 'MAPK', 'PI3K', 'TGFb', 'NFkB', 'TNFa', 'JAK-STAT', 'VEGF', 'Androgen', 'Estrogen']

class PathwayHead(nn.Module):
    def __init__(self, latent_dim, n_pathways=10):
        self.predictor = nn.Linear(latent_dim, n_pathways)
    
    def forward(self, z):
        return self.predictor(z)

# During training
pathway_scores = progeny.score(adata)  # Ground truth from PROGENy
predicted_pathways = pathway_head(latent_z)
pathway_loss = F.mse_loss(predicted_pathways, pathway_scores)

total_loss = reconstruction_loss + alpha * pathway_loss
```

### Implementation Status: COMPLETE

**Files:**
- `stagebridge/biology/pathway_targets.py` - PROGENy pathway scoring (14 pathways)
- `stagebridge/pipelines/complete_data_prep.py` - Pre-computes pathway_0..13 columns
- `stagebridge/pipelines/run_v1_ddp.py` - PathwayHead class, 5% weight in training loop
- `stagebridge/transition_model/relational_pretraining.py` - pathway_head in SSL

**Priority: COMPLETE** - Lightweight addition with strong biological grounding.

---

## 6. Niche Definition by Pathway Activation (from SpatialFusion)

### Key Insight
> "Spatial niches are defined by coordinated pathway activation across cell types, not just cell type composition."

Two neighborhoods with same cell type composition can have different functional states!

### Implementation for StageBridge
```python
# Current: Niche = aggregation of neighbor cell types
# Improved: Niche = aggregation of neighbor STATES (cell type + pathway activation)

def compute_niche_representation(center_cell, neighbors, pathway_scores):
    """Niche = cell type composition + pathway pattern."""
    cell_type_composition = aggregate_cell_types(neighbors)
    pathway_pattern = aggregate_pathways(neighbors, pathway_scores)
    return torch.cat([cell_type_composition, pathway_pattern], dim=-1)
```

**Priority: HIGH** - Directly improves our core niche modeling.

---

## 7. Frozen Foundation Model Embeddings (from SpatialFusion)

### Problem
- Training from scratch is expensive and may not generalize

### Solution: Use frozen scGPT/Geneformer embeddings as input

```python
# Instead of raw gene expression:
# cell_embedding = model(gene_expression)

# Use pretrained foundation model:
from scgpt import scGPTEncoder
foundation_encoder = scGPTEncoder.from_pretrained('scgpt-human')
foundation_encoder.eval()
foundation_encoder.requires_grad_(False)

# Get frozen embeddings
with torch.no_grad():
    cell_embeddings = foundation_encoder(gene_expression)

# Train lightweight downstream model
niche_model = NicheTransformer(input_dim=foundation_dim)
```

### Benefits
- Leverages billions of cells of pretraining
- Much lighter training (~300K params vs millions)
- Better generalization

**Priority: MEDIUM-HIGH** - Could significantly reduce training cost and improve generalization.

---

## 8. Graph Convolutional Masked Autoencoder (from SpatialFusion)

### Problem
- Current spatial encoding may not capture neighborhood structure well

### Solution: GCMAE for spatial context

```python
# Build k-NN graph from spatial coordinates
spatial_graph = knn_graph(coordinates, k=30)

# Use graph convolution + masking for self-supervised learning
class GCMAE(nn.Module):
    def __init__(self):
        self.encoder = GCNEncoder(...)
        self.decoder = GCNDecoder(...)
    
    def forward(self, x, edge_index, mask_ratio=0.15):
        # Mask random nodes
        masked_x, mask = random_mask(x, mask_ratio)
        # Encode with graph convolution
        z = self.encoder(masked_x, edge_index)
        # Reconstruct masked nodes
        x_recon = self.decoder(z, edge_index)
        return x_recon, mask
```

**Priority: MEDIUM** - Alternative to our current ring-based spatial encoding.

---

---

## 9. IL1B Pathway Supervision (from Peng/Kadara) - IMPLEMENTED

### Problem
- The Peng/Kadara hypothesis states IL1B+ macrophage niches drive AT2-to-LUAD progression
- Model should explicitly learn this inflammatory axis

### Solution: IL1B Auxiliary Head

```python
class IL1BHead(nn.Module):
    """Predict IL1B pathway activity from niche context."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
```

### Implementation Status: COMPLETE

**Files:**
- `stagebridge/pipelines/run_v1_ddp.py` - IL1BHead class, 5% weight in training
- Target: NFkB pathway index from PROGENy (best proxy for IL1B signaling)

**Priority: COMPLETE** - Direct test of the key biological hypothesis.

---

## 10. KAC Intermediate State Supervision (from Han et al. 2024) - IMPLEMENTED

### Problem
- KRT8+ Alveolar Intermediate Cells (KACs) are the critical precursor state
- Trajectory: Normal AT2 -> AIC -> KAC -> LUAD
- Model should explicitly supervise learning of this state

### Solution: KAC Auxiliary Head

```python
KAC_MARKERS = [
    "KRT8", "CLDN4", "CDKN1A", "CDKN2A", "PLAUR",  # Core markers
    "CEACAM5", "CEACAM6", "MUC1", "MSLN", "CD24",  # Extended
]

class KACHead(nn.Module):
    """Predict KAC (KRT8+ Alveolar Intermediate Cell) signature."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
```

### Implementation Status: COMPLETE

**Files:**
- `stagebridge/biology/pathway_targets.py` - KAC_MARKERS list, compute_kac_targets()
- `stagebridge/pipelines/run_v1_ddp.py` - KACHead class, 5% weight
- Target: p53 pathway (CDKN1A proxy) from PROGENy

**Priority: COMPLETE** - Supervises the key intermediate state from Han et al. 2024 Nature.

---

## 11. Ki67 Proliferation Prediction (from OSDR) - IMPLEMENTED

### Problem
- Niche encoder may not capture dynamically-relevant features
- Division rate is a direct readout of cellular fitness

### Solution: Auxiliary Proliferation Prediction

From OSDR (Nature 2026): If niche encoder can predict Ki67 well, it's learning features relevant to cell state dynamics.

```python
class ProliferationHead(nn.Module):
    def __init__(self, latent_dim):
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, z):
        return self.predictor(z)  # Binary classification

# Target: Ki67/MKI67 expression > 75th percentile
prolif_target = (mki67_expression > threshold).float()
prolif_loss = F.binary_cross_entropy_with_logits(pred, prolif_target)
```

### Implementation Status: COMPLETE

**Files:**
- `stagebridge/biology/pathway_targets.py` - `compute_proliferation_targets()` function
- `stagebridge/pipelines/complete_data_prep.py` - Pre-computes `proliferation_label` column
- `stagebridge/pipelines/run_v1_ddp.py` - ProliferationHead class, 5% weight in training loop

**Priority: COMPLETE** - Forces latent to encode division-relevant features.

---

## Summary: Prioritized Implementation Order

### Phase 1 (Immediate - Low Effort, High Impact) - COMPLETE
1. **Pathway regression loss** - DONE (PROGENy prediction head)
2. **Ki67 proliferation loss** - DONE (OSDR-inspired)
3. **IL1B head** - DONE (Peng/Kadara hypothesis test)
4. **KAC head** - DONE (Han et al. 2024 Nature - KRT8+ intermediate state)
5. **Stage-aware OT** - DONE (Adjacent stage pairing only)
6. **DestVI gamma integration** - DONE (Functional state in Token 7)
7. **B/plasma cell signatures** - DONE (Hao et al. 2022 - immunotherapy response)

### Phase 1b (Remaining - Medium Effort)
8. **Niche pathway encoding** - Include pathway scores in niche representation
9. **Relative representations** - Define stage archetypes, compute relative coords

### Phase 2 (Short-term - Medium Effort, High Impact)
5. **Frozen foundation embeddings** - Replace raw expression with scGPT embeddings
6. **Invertible encoder** - Replace VAE with INN for geometry preservation

### Phase 3 (Long-term - High Effort, Medium-High Impact)
7. **Hyperbolic latent space** - For better hierarchy/trajectory modeling
8. **EWC + Replay** - For principled dual-reference integration
9. **GCMAE spatial encoding** - Alternative to ring aggregation

### Implementation Status Summary

| Improvement | Source | Status | Location |
|-------------|--------|--------|----------|
| Pathway regression | SpatialFusion | DONE | `pathway_targets.py`, `run_v1_ddp.py` |
| Ki67 proliferation | OSDR | DONE | `pathway_targets.py`, `run_v1_ddp.py` |
| IL1B head | Peng et al. 2020 | DONE | `run_v1_ddp.py:IL1BHead` |
| KAC head | Han et al. 2024 Nature | DONE | `run_v1_ddp.py:KACHead`, `pathway_targets.py:KAC_MARKERS` |
| Stage-aware OT | OT-CFM | DONE | `run_v1_ddp.py` (adjacent stage pairing) |
| DestVI gamma | DestVI | DONE | `run_v1_ddp.py` (Token 7 enrichment) |
| B/plasma signatures | Hao et al. 2022 | DONE | `signatures.py` (plasma_cell, cxcl13_tls) |
| Sparse attention | Doctrine | DONE | `receiver_niche_encoder.py` (ENTROPY/TOPK/SPARSEMAX) |
| Hyperbolic geometry | scPhere | Code ready, not enabled | `reference_geometry/` |
| Geodesic bridges | GeoBridge | Pending | - |
| Foundation embeddings | SpatialFusion | Pending | - |
| EWC + Replay | CRC | Pending | - |

---

## Architecture Integration Sketch

```
Input: Single cell gene expression + spatial coordinates

[Foundation Encoder (frozen scGPT)]
         |
         v
    Cell embeddings (512-dim)
         |
    +----+----+
    |         |
    v         v
[Pathway    [Spatial GCMAE]
 Predictor]      |
    |            v
    |    Neighborhood embedding
    |            |
    +-----+------+
          |
          v
   [Invertible Fusion Block]
          |
          v
   Hyperbolic latent z (on H^n)
          |
    +-----+-----+
    |           |
    v           v
[Relative    [Stage 
 Repr Head]  Classifier]
    |           |
    v           v
Archetype   Progression
similarities   score
```

---

## References

1. scPhere - Hyperspherical/hyperbolic VAE for scRNA-seq
2. GeoBridge - Geodesic bridge via INN for trajectory inference  
3. CRC Continual Learning - EWC + relative representations for atlas comparison
4. SpatialFusion - Pathway-informed spatial niche mapping
