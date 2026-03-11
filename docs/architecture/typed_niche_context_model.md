# Architecture: EA-MIST Context Model

**Scientific layer:** 4 — Lesion-level context modeling via local niche aggregation
**Package location:** `stagebridge/context_model/`, `stagebridge/pipelines/train_lesion.py`

## Role in the System

The context model encodes local tissue microenvironments (niches) into lesion-level representations that predict disease stage and evolutionary displacement. Each lesion is treated as a **bag of neighborhoods**: the model must extract lesion-level signal from an unordered set of spatially grounded local niches.

## Data Contract

### Lesion Bags

Each lesion produces a `LesionBag` containing:

- **lesion_id, donor_id, patient_id** — Identifiers for stratified evaluation
- **stage** — Canonical stage label (Normal, AAH, AIS, MIA, LUAD)
- **neighborhoods** — List of `LocalNicheExample` instances (one per spatial niche)
- **stage_index** — Ordinal stage class (0–4 canonical, 0–2 grouped)
- **displacement_target** — Weak ordinal supervision target in [0, 1]
- **evolution_features** — Optional WES-derived lesion-level features
- **edge_targets, edge_target_mask** — Optional auxiliary binary edge labels

### Local Niche Example

Each neighborhood contains multi-perspective features for one spatial niche:

| Feature | Shape | Description |
|---------|-------|-------------|
| `receiver_embedding` | `(D_r,)` | Central cell latent vector from HLCA embedding |
| `receiver_state_id` | int | Discrete receiver cell-type identity |
| `ring_compositions` | `(num_rings, D_s)` | Ring-wise sender composition at increasing radii |
| `hlca_features` | `(13,)` | Cosine similarities to HLCA healthy reference states |
| `luca_features` | `(15,)` | Cosine similarities to LuCA cancer atlas states |
| `lr_pathway_summary` | `(D_lr,)` | Compact ligand-receptor and pathway summary |
| `neighborhood_stats` | `(D_stats,)` | Density, diversity, and uncertainty statistics |
| `flat_features` | `(D_flat,)` | Flattened feature vector for MLP ablations |
| `center_coord` | `(2,)` | Spatial tissue coordinate |

### Grouped Ordinal Labels

The canonical 5-class labels can be collapsed into 3 grouped ordinal labels:

| Grouped label | Original stages | Index | Displacement target |
|--------------|----------------|-------|-------------------|
| `early_like` | Normal, AAH | 0 | 0.0 |
| `intermediate_like` | AIS, MIA | 1 | 0.5 |
| `invasive_like` | LUAD | 2 | 1.0 |

Grouped mode is activated by `use_grouped_labels: true` in config. This changes `num_stage_classes` from 5 to 3 throughout the pipeline, remaps `stage_index` and `displacement_target` on all bags before fold creation, and switches to grouped-specific metrics (weighted kappa, grouped balanced accuracy).

## Architecture

### Local Niche Encoder

Each neighborhood is encoded independently into a fixed-size embedding by `LocalNicheTransformerEncoder`.

#### Token Construction

The encoder converts each niche into a sequence of typed tokens:

| Token type | ID | Count | Projection |
|-----------|-----|-------|-----------|
| Receiver | 0 | 1 | `Linear(D_r → model_dim) + StateEmb(state_id) + TypeEmb(0)` |
| Ring | 1 | `num_rings` | `Linear(D_s → model_dim) + RingEmb(ring_id) + TypeEmb(1)` |
| HLCA | 2 | 1 | `Linear(13 → model_dim) + TypeEmb(2)` |
| LuCA | 3 | 1 | `Linear(15 → model_dim) + TypeEmb(3)` |
| L/R pathway | 4 | 1 | `Linear(D_lr → model_dim) + TypeEmb(4)` |
| Statistics | 5 | 1 | `Linear(D_stats → model_dim) + TypeEmb(5)` |
| Atlas contrast | 6 | 0 or 1 | Contrast MLP (see below) `+ TypeEmb(6)` |

Default sequence length: `1 + num_rings + 4 = 9 tokens` (10 with contrast token).

#### Atlas Contrast Token

When `use_atlas_contrast_token: true` and both HLCA and LuCA features are available, an additional token captures cross-atlas relationships:

```
h = hlca_features[:, :min_dim]    # truncate to common dim
l = luca_features[:, :min_dim]
contrast_input = [hlca_features, luca_features, l-h, h*l, |l-h|]
```

Input dimension: `hlca_dim + luca_dim + 3 × min(hlca_dim, luca_dim)` = 67 for (13, 15).

Processed by: `Linear(67 → model_dim) → GELU → Linear(model_dim → model_dim)`.

#### Self-Attention

Token sequence is processed by `num_layers` SAB (Self-Attention Block) layers:

```
For each SAB:  MultiHeadAttn(Q=X, K=X, V=X) → Residual → LayerNorm → FFN → Residual → LayerNorm
```

FFN expands to 4× hidden dim: `Linear(model_dim → 4*model_dim) → GELU → Linear(4*model_dim → model_dim)`.

#### Pooling

PMA (Pooling by Multihead Attention) reduces the token sequence to a single embedding:

```
seed = learnable (1, num_pma_seeds, model_dim)
output = MultiHeadAttn(Q=seed, K=tokens, V=tokens) → Residual → FFN → LayerNorm
```

Output: `neighborhood_embedding (model_dim,)` per niche.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_dim` | 128 | Token and output embedding dimension |
| `num_heads` | 4 | Attention heads per SAB layer |
| `num_layers` | 2 | Number of SAB self-attention blocks |
| `num_receiver_states` | 32 | Vocabulary size for receiver state embedding |
| `num_rings` | 4 | Number of spatial distance rings |
| `dropout` | 0.1 | Dropout rate |
| `use_atlas_contrast_token` | false | Include 10th contrast token |

### EA-MIST Model (Lesion-Level)

`EAMISTModel` aggregates niche embeddings into a lesion-level representation.

#### Pipeline

```
1. encode_local(batch)  →  local_embeddings (B, N, hidden_dim)
2. [Optional] Prototype bottleneck  →  soft assignment to K prototypes
3. LesionSetTransformerBackbone(ISAB → SAB → PMA)  →  lesion_embedding (B, hidden_dim)
4. [Optional] Evolution branch fusion  →  gated/FiLM conditioning
5. [Optional] Distribution-aware pooling  →  7 statistics appended
6. LesionMultitaskHeads  →  stage_logits, displacement, edge_logits
```

#### Set Transformer Backbone

Processes the variable-length set of niche embeddings:

| Block | Description |
|-------|-------------|
| **ISAB** | Induced Set Attention Block with `M` inducing points: O(NM) complexity |
| **SAB** | Full self-attention refinement across niches |
| **PMA** | Pools to `K` fixed-size summary vectors via learned seeds |

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | Embedding dimension |
| `num_heads` | 4 | Attention heads |
| `num_layers` | 2 | Transformer blocks |
| `num_inducing_points` | 16 | ISAB inducing point count |
| `num_pma_seeds` | 1 | PMA seed vectors |
| `dropout` | 0.1 | Dropout rate |

#### Prototype Bottleneck (Optional)

When enabled, niche embeddings are soft-assigned to `K` learned prototypes before set-level aggregation:

- Assignment: `softmax(embeddings @ prototypes.T / sqrt(d))`
- Sparse mode available (top-k instead of full softmax)
- Regularized by diversity and entropy losses

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_prototypes` | true | Enable prototype bottleneck |
| `num_prototypes` | 16 | Number of learned niche motifs |
| `sparse_assignments` | false | Top-k (sparse) vs softmax (soft) |

#### Evolution Branch (Optional)

Conditions the lesion embedding on WES-derived evolutionary features:

**Gated mode** (default):
```
gate = σ(Linear([lesion_emb, evo_proj]))
fused = gate · lesion_emb + (1 - gate) · evo_proj
```

**FiLM mode**:
```
γ, β = Linear(evo_proj), Linear(evo_proj)
fused = lesion_emb · (1 + γ) + β
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evolution_dim` | None | Feature dimension; None disables |
| `evolution_mode` | "gated" | "gated" or "film" |

#### Distribution-Aware Pooling

When enabled, a per-niche transition score head produces scalar scores for each neighborhood, then computes 7 summary statistics that are concatenated with the lesion embedding before the task heads:

Score head: `Linear(hidden_dim → hidden_dim) → GELU → Dropout → Linear(hidden_dim → 1)`

Statistics: mean, std, min, max, q25, median, q75 (computed over valid niches only).

Head input: `[lesion_embedding, dist_stats]` → dimension `hidden_dim + 7`.

#### Multitask Heads

| Head | Architecture | Output |
|------|-------------|--------|
| **Stage** | `Linear → GELU → Dropout → Linear(→ num_classes)` | `(B, C)` logits |
| **Displacement** | `Linear → GELU → Dropout → Linear(→ 1)` | `(B,)` scalar |
| **Edge** (optional) | `Linear → GELU → Dropout → Linear(→ num_edges)` | `(B, E)` logits |

#### Reference Feature Modes

The atlas features can be selectively ablated at the model level:

| Mode | HLCA | LuCA | Contrast token | Description |
|------|------|------|---------------|-------------|
| `no_atlas` | Zeroed | Zeroed | No | Spatial-only baseline |
| `hlca_only` | Active | Zeroed | No | Healthy atlas only |
| `luca_only` | Zeroed | Active | No | Cancer atlas only |
| `hlca_luca` | Active | Active | No | Both atlases |
| `hlca_luca_contrast` | Active | Active | Yes | Both + contrast token |

### Baseline Models

`LesionAggregatorModel` uses the same local encoder but simpler lesion-level aggregation:

| Family | Aggregator | Description |
|--------|-----------|-------------|
| `pooled` | Mean pooling | Simplest bag-level baseline |
| `deep_sets` | DeepSets (φ→ρ) | Permutation-invariant, no attention |
| `lesion_set_transformer` | ISAB+SAB+PMA | Attention baseline without prototypes/evolution |

All baselines share the local encoder architecture and reference feature mode handling.

## Training

### Loss Function

Total loss is a weighted sum of five components:

$$L = w_s \cdot L_{stage} + w_d \cdot L_{disp} + w_e \cdot L_{edge} + w_o \cdot L_{ordinal} + w_t \cdot L_{transition} + L_{reg}$$

| Loss | Function | Default weight | Description |
|------|---------|----------------|-------------|
| $L_{stage}$ | Cross-entropy (class-weighted) | 1.0 | Main classification loss |
| $L_{disp}$ | SmoothL1 | 0.5 | Displacement regression |
| $L_{edge}$ | Binary cross-entropy (masked) | 0.25 | Auxiliary edge prediction |
| $L_{ordinal}$ | EMD (CDF distance) | 0.5 | Ordinal stage penalty |
| $L_{transition}$ | SmoothL1 (detached target) | 0.1 | Niche-lesion consistency |
| $L_{reg}$ | Diversity + entropy | (built-in) | Prototype regularization |

**Ordinal stage loss** (EMD): Compares cumulative distributions rather than point predictions. Penalizes predicting LUAD when the truth is Normal more than predicting AAH:

$$L_{ordinal} = \text{mean}(|CDF_{pred} - CDF_{target}|)$$

**Transition consistency loss**: Couples the lesion-level displacement prediction with the mean per-niche transition score. The niche scores are detached so gradients only flow into the displacement head.

### Optimizer

AdamW with gradient clipping:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.0005 | Base learning rate |
| `weight_decay` | 0.001 | L2 regularization |
| `grad_clip_norm` | 1.0 | Max gradient norm |
| `max_epochs` | 150 | Training epoch limit |
| `patience` | 35 | Early stopping patience |

### Hyperparameter Optimization

Optuna TPE sampler with median pruning:

| Search dimension | Values |
|-----------------|--------|
| `hidden_dim` | [32, 64, 128] |
| `dropout` | [0.2, 0.3, 0.4] |
| `learning_rate` | [0.0001, 0.0003, 0.0005, 0.001, 0.003] |
| `weight_decay` | [1e-4, 5e-4, 1e-3, 5e-3] |
| `num_layers` | [1, 2] (eamist only) |
| `num_prototypes` | [4, 8, 16] (eamist only) |
| `evolution_mode` | [gated, film] (eamist only) |

50 trials per model×mode×fold. Pruned trials check against the median of completed trials after `n_warmup_steps` epochs.

### Composite Selection Score

**Canonical (5-class):**
$$\text{score} = F_1^{macro} + 0.25 \cdot \text{bal\_acc} + 0.10 \cdot \max(\rho_s, 0) + 0.05 \cdot \text{central\_recall}$$

**Grouped (3-class):**
$$\text{score} = 0.40 \cdot \max(\rho_s, 0) + 0.30 \cdot \max(\kappa_w, 0) + 0.20 \cdot \text{bal\_acc} + 0.10 \cdot F_1^{macro}$$

The grouped score emphasizes ordinal metrics (Spearman displacement correlation + linear-weighted Cohen's kappa), reflecting the scientific priority of correctly ordering lesions along the progression axis.

## Evaluation Protocol

### Cross-Validation

Donor-held-out 3-fold cross-validation. Each fold contains train/val/test splits stratified by donor to prevent information leakage between related lesions.

### Negative Controls

Two permutation-based controls verify that the model uses atlas features meaningfully:

| Control | Transformation | Preserves | Destroys |
|---------|---------------|-----------|----------|
| `atlas_label_shuffle` | Shuffle HLCA/LuCA across all niches | Spatial structure | Atlas-stage correspondence |
| `within_lesion_niche_shuffle` | Shuffle neighborhood order per lesion | Per-lesion statistics | Spatial ordering |

Both create deep copies of bags and use the `hlca_luca` reference mode for model construction.

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `displacement_spearman` | Ordinal | Spearman rank correlation of predicted displacement vs target |
| `grouped_weighted_kappa` | Ordinal | Linear-weighted Cohen's κ for 3-class agreement |
| `grouped_balanced_accuracy` | Classification | Mean per-class recall |
| `grouped_macro_f1` | Classification | Macro-averaged F1 across classes |
| `displacement_mae` | Regression | Mean absolute error on displacement |

## Relationship to Other Layers

- **Upstream:** Spatial mapping produces neighborhood features; reference mapping provides HLCA/LuCA embeddings and cell-type labels
- **Downstream:** Evaluation layer computes metrics and ablation tables; results tracking persists artifacts
