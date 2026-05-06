# Methodology: How Key Numbers Were Calculated

**Source document**: `docs/stagebridge_preliminary.tex` (lines 331-333)
**Data source**: Peng et al. LUAD precursor dataset (GSE234047)
**Processed data**: `/data1/chaunzt1/stagebridge/processed/luad_evo/canonical/`

---

## 1. IL1B Expression Analysis

### 2.8-fold IL1B increase
**What it measures**: Ratio of mean IL1B expression in Invasive vs Normal tissue

**Calculation**:
```python
# From snRNA-seq or spatial data with IL1B gene expression
mean_il1b_normal = adata[adata.obs['stage'] == 'Normal', 'IL1B'].X.mean()
mean_il1b_invasive = adata[adata.obs['stage'] == 'Invasive', 'IL1B'].X.mean()
fold_change = mean_il1b_invasive / mean_il1b_normal  # = 2.8
```

### IL1B+ cell percentages (31%, 49%, 51%)
**What it measures**: Fraction of cells with IL1B expression above threshold per stage

**Calculation**:
```python
# Threshold typically log-normalized expression > 0 or > 0.5
threshold = 0.5  # or median non-zero expression
for stage in ['Normal', 'Preinvasive', 'Invasive']:
    stage_cells = adata[adata.obs['stage'] == stage]
    il1b_positive = (stage_cells[:, 'IL1B'].X > threshold).sum()
    pct = 100 * il1b_positive / len(stage_cells)
    # Normal: 31%, Preinvasive: 49%, Invasive: 51%
```

### IL1B-stage correlation (Spearman r = 0.336)
**Calculation**:
```python
from scipy.stats import spearmanr
stage_numeric = adata.obs['stage'].map({'Normal': 0, 'Preinvasive': 1, 'Invasive': 2})
il1b_expr = adata[:, 'IL1B'].X.flatten()
r, p = spearmanr(stage_numeric, il1b_expr)  # r = 0.336, p < 1e-16
```

### Transition zone IL1B (0.35 vs 0.28)
**What it measures**: Mean IL1B expression in spots at stage boundaries vs interior

**Calculation**:
```python
# Spatial data with stage annotations and coordinates
# Transition zone = spots within X microns of a stage boundary
transition_mask = identify_transition_zones(adata_spatial, distance_threshold=100)
mean_il1b_transition = adata_spatial[transition_mask, 'IL1B'].X.mean()  # 0.35
mean_il1b_nontransition = adata_spatial[~transition_mask, 'IL1B'].X.mean()  # 0.28
```

---

## 2. Cell Type Composition Analysis

### T-cell percentages (15.9%, 5.7%, 10.9%)
**What it measures**: Fraction of cells annotated as T cells per stage

**Calculation**:
```python
# Using cell type annotations (from HLCA/LuCA mapping or manual annotation)
for stage in ['Normal', 'Preinvasive', 'Invasive']:
    stage_cells = adata[adata.obs['stage'] == stage]
    t_cells = (stage_cells.obs['cell_type'].str.contains('T cell')).sum()
    pct = 100 * t_cells / len(stage_cells)
    # Normal: 15.9%, Preinvasive: 5.7%, Invasive: 10.9%
```

**Note**: "Preinvasive" combines AAH + AIS + MIA stages. The 57% reduction is:
```python
reduction = (15.9 - 5.7) / 15.9 * 100  # = 64% (paper says 57%, may use different grouping)
```

### Macrophages (11.0% → 5.0%)
**Same methodology** as T-cells, filtering for macrophage annotations.

### Fibroblasts (24.8% → 10.7%)
**Same methodology**, filtering for fibroblast annotations.

---

## 3. Proliferation Analysis

### 3.6-fold proliferation increase (7.1% → 25.7%)
**What it measures**: Fraction of cells in proliferative state (S/G2M phase or high proliferation score)

**Calculation**:
```python
# Option A: Cell cycle phase
for stage in ['Normal', 'Invasive']:
    stage_cells = adata[adata.obs['stage'] == stage]
    proliferating = stage_cells.obs['phase'].isin(['S', 'G2M']).sum()
    pct = 100 * proliferating / len(stage_cells)
    # Normal: 7.1%, Invasive: 25.7%

# Option B: Proliferation score threshold
# Using scanpy.tl.score_genes_cell_cycle or similar
threshold = adata.obs['proliferation_score'].median()
proliferating = (stage_cells.obs['proliferation_score'] > threshold).sum()
```

**Fold change**:
```python
fold = 25.7 / 7.1  # = 3.62 ≈ 3.6
```

---

## 4. Genomic Alteration Analysis

### TP53 mutations (25% → 48%)
**What it measures**: Percentage of donors/samples with TP53 mutation per stage

**Data source**: Whole-exome sequencing (WES) data from Peng et al.

**Calculation**:
```python
# WES data linked to samples
for stage in ['Normal', 'Invasive']:
    stage_samples = wes_df[wes_df['stage'] == stage]
    tp53_mutated = (stage_samples['TP53_mutation'] == True).sum()
    pct = 100 * tp53_mutated / len(stage_samples)
    # Normal: 25%, Invasive: 48%
```

### EGFR mutations (7% → 29%)
**Same methodology** for EGFR gene.

---

## 5. Context Encoding Correlations

### gamma_4 ~ TGFbeta (r = 0.18)
**What it measures**: Correlation between learned context dimension and pathway activity

**Calculation**:
```python
# After running StageBridge inference to get context embeddings
# And computing pathway scores (e.g., via PROGENy or AUCell)
from scipy.stats import pearsonr

gamma_4 = context_embeddings[:, 4]  # 5th dimension (0-indexed)
tgfb_score = adata.obs['TGFb_pathway_score']  # from decoupler/PROGENy
r, p = pearsonr(gamma_4, tgfb_score)  # r = 0.18
```

### gamma_4 stage correlation (Spearman r = 0.364)
```python
stage_numeric = adata.obs['stage'].map({'Normal': 0, 'AAH': 1, 'AIS': 2, 'MIA': 3, 'LUAD': 4})
r, p = spearmanr(gamma_4, stage_numeric)  # r = 0.364, p < 1e-16
```

### Transition zone t-test (t = -8.35, p = 7e-17)
```python
from scipy.stats import ttest_ind
gamma4_transition = gamma_4[transition_mask]
gamma4_nontransition = gamma_4[~transition_mask]
t, p = ttest_ind(gamma4_transition, gamma4_nontransition)  # t = -8.35
```

---

## 6. Model Performance Metrics

### Drift alignment (0.964 +/- 0.008)
**What it measures**: Cosine similarity between predicted velocity and OT-derived transport direction

**Calculation**:
```python
# For each cell pair (source in Normal, target in Preinvasive)
# 1. Get OT coupling from Sinkhorn
# 2. Compute transport direction: target_z - source_z
# 3. Predict velocity with model: v = model.predict_drift(source_z, niche_context)
# 4. Compute alignment: cosine_similarity(v, transport_direction)

from sklearn.metrics.pairwise import cosine_similarity
alignments = []
for source, target in ot_coupled_pairs:
    transport_dir = target - source
    predicted_v = model.drift_head(source, context)
    align = cosine_similarity(predicted_v, transport_dir)
    alignments.append(align)
    
mean_alignment = np.mean(alignments)  # 0.964
std_alignment = np.std(alignments)    # 0.008
```

### Ablation delta percentages
```python
full_loss = 0.003555
ablation_loss = 0.003958  # e.g., no_gate
delta_pct = 100 * (ablation_loss - full_loss) / full_loss  # +11.3%
```

---

## Data Files Required to Reproduce

1. **snRNA-seq data**: `cells.parquet` with columns:
   - `stage`: Normal, AAH, AIS, MIA, LUAD
   - `cell_type`: from HLCA/LuCA mapping
   - `phase`: cell cycle phase (G1, S, G2M)
   - `z_fused_*`: embedding dimensions
   - Gene expression columns or link to `expression.parquet`

2. **Spatial data**: `spots.parquet` or spatial AnnData with:
   - Coordinates (x, y)
   - Stage annotations
   - Deconvolved cell type proportions

3. **WES data**: Mutation calls per sample/donor

4. **Model outputs**: 
   - Context embeddings (gamma)
   - Predicted velocities
   - Attention weights

---

## Scripts That Should Exist (To Be Created/Located)

1. `scripts/compute_cell_composition_by_stage.py`
2. `scripts/compute_il1b_statistics.py`
3. `scripts/compute_proliferation_by_stage.py`
4. `scripts/compute_mutation_frequencies.py`
5. `scripts/compute_context_correlations.py`
6. `scripts/identify_transition_zones.py`

---

## Verification Checklist

Before presenting these numbers:

- [ ] Re-run IL1B fold change calculation on current processed data
- [ ] Verify T-cell percentage calculation matches cell type annotations used
- [ ] Confirm proliferation scoring method (phase vs score threshold)
- [ ] Cross-check mutation frequencies with Peng et al. supplementary tables
- [ ] Ensure "Preinvasive" grouping is consistent (AAH+AIS+MIA vs just AIS)

---

## Citation

Original data: Peng et al. "Single-cell and spatial transcriptomics reveal proinflammatory niches in lung adenocarcinoma precursor lesions" (2025), GSE234047
