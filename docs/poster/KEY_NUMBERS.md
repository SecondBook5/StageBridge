# StageBridge Poster - Key Numbers

## From stagebridge_preliminary.tex (verified)

### Model Performance
- Normal-to-Preinvasive drift alignment: **0.964 +/- 0.008**
- Overall drift alignment: 0.309 +/- 0.081
- IL1B correlation: r = 0.14
- KAC correlation: r = 0.26

### Ablation Results (from comparison_report.json)
- Full model val loss: 0.00356
- no_gate: +11.3% (most critical component)
- gw_barycentric: +7.7%
- hlca_only: +7.2%
- luca_only: +5.3%
- no_niche: +2.2%

### Baseline Comparison
- StageBridge: 0.0036 MSE
- GraphSAGE: 0.058 MSE (16x worse)
- Pooling/DeepSets/SetTransformer: ~0.081 MSE (22x worse)

---

## Biological Numbers (from stagebridge_preliminary.tex lines 331-333)

### IL1B Expression
- Normal: 31% IL1B+ cells
- Preinvasive: 49% IL1B+ cells
- Invasive: 51% IL1B+ cells
- **2.8-fold increase** across progression
- Spearman r = 0.336, p < 10^-16
- Transition zones: 0.35 vs 0.28 (non-transition)

### T-Cell Depletion
- Normal: **15.9%** T cells
- Preinvasive: **5.7%** T cells (lowest!)
- Invasive: 10.9% T cells
- **57% reduction** at preinvasive stage
- Immunosuppression PRECEDES frank invasion

### Macrophages
- Normal: 11.0%
- Preinvasive: 5.0%
- 55% reduction

### Fibroblasts
- Normal: 24.8%
- Preinvasive: 10.7%
- 57% reduction

### Proliferation
- Normal: **7.1%** proliferating cells
- Invasive: **25.7%** proliferating cells
- **3.6-fold increase**

### Driver Mutations
- TP53: 25% (Normal) to 48% (Invasive)
- EGFR: 7% (Normal) to 29% (Invasive)
- KRAS: rising across stages

### Context Encoding
- gamma_4 correlates with TGFbeta (r = 0.18)
- gamma_6 correlates with TNFalpha/NFkB (r = 0.12)
- gamma_4 stage correlation: Spearman r = 0.364, p < 10^-16
- Transition zone difference: t = -8.35, p = 7e-17

---

## Dataset
- 1,437,916 cells
- 639,816 spatial spots
- 25 donors
- 5 stages: Normal, AAH, AIS, MIA, LUAD

---

## Key Story Points for Poster

1. **Transitions are recoverable** (0.964 alignment for early stages)
2. **Gating mechanism is critical** (+11.3% when removed)
3. **Niche is modulatory, not driving** (+2.2% = cell-autonomous transitions)
4. **IL1B marks progression window** (2.8-fold increase, peaks in preinvasive)
5. **T-cell depletion precedes invasion** (57% drop at preinvasive)
6. **Proliferation explodes in invasive** (3.6-fold increase)

The biological insight: Epithelial transitions are largely cell-autonomous, but occur preferentially in inflammatory-stromal niches that provide permissive conditions.
