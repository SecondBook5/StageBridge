# StageBridge Paper Outline

## 1. Abstract (complete)
- Problem: modeling tumor stage progression as context-aware rather than cell-intrinsic
- Method: Set Transformer over typed niche tokens + OT-based transition model
- Key result: Set Transformer context outperforms pooled and RNA-only baselines on invasive transition
- Significance: demonstrates that local tissue context improves stage-transition modeling

## 2. Introduction (complete)
- Biological motivation: LUAD progression Normal->AAH->AIS->MIA->LUAD
- ML problem: learning context-aware stage transitions
- Why transformers: permutation invariance, attention-based niche weighting
- Hypotheses (3)
- Project scope

## 3. Related Work (complete)
- Set Transformer (Lee et al. 2019)
- Graph Transformers (Dwivedi & Bresson 2020, Ying et al. 2021)
- Spatial deconvolution (Tangram, TACCO, DestVI)
- Single-cell trajectory/transport (TrajectoryNet, CellOT, OT-CFM, Waddington-OT)
- StageBridge positioning

## 4. Research Project Problem (complete)
- Inputs, outputs, formal problem statement
- Three hypotheses
- Why transformer-based niche encoder is central

## 5. Method (complete)
- 5.1 Data representation (snRNA-seq, Visium, HLCA latent)
- 5.2 Typed niche token construction
- 5.3 Set Transformer context encoder (ISAB, SAB, PMA)
- 5.4 Optional Graph Transformer context encoder
- 5.5 Transition model (OT coupling, Schrodinger bridge interpolant, drift network)
- 5.6 Loss and training objective
- 5.7 Evaluation (Sinkhorn divergence, classifier AUC, calibration)

## 6. Experimental Plan (draft)
- Matched comparisons: rna_only, pooled, set_only, graph_of_sets
- Donor-holdout evaluation
- Per-edge analysis

## 7. Preliminary Results (draft)
- Table 1: Core mode comparison
- Figure 2: Mode comparison by edge
- Figure 3: Context ablation decision matrix
- Key findings

## 8. Planned Statistical Evaluation (draft)
- Bootstrap confidence intervals
- Multiple edge replication
- Ablation plan

## 9. Conclusion (draft)
- Summary of findings
- Limitations
- Future work

## 10. References (complete)
