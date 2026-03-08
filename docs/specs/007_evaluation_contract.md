# 007 — Evaluation Contract

Interpretation and evaluation code lives under `stagebridge/evaluation/`. Training logic does not own tissue-level interpretation.

## Principle

Evaluation is broader than metrics. A transition model that achieves low loss but reveals nothing about tissue biology is incomplete.

## Components

### 1. Held-Out Performance

Donor-held-out cross-validation with: Sinkhorn divergence, MMD, Wasserstein distance, per-cell-type transport accuracy. Within-donor evaluation is informative but insufficient.

### 2. Calibration

Predicted uncertainty should match observed uncertainty. For stochastic models, trajectory spread should correlate with actual target variability. Produce calibration plots per disease edge.

### 3. Ablations

| Comparison | Question |
|------------|----------|
| Set-only vs RNA-only | Does spatial context improve transitions? |
| GoST vs Set-only | Does inter-set graph attention add value? |
| WES-regularized vs unregularized | Does evolutionary constraint help? |
| Learned vs Gaussian-SB | Does learning improve over the Gaussian prior? |
| OT-coupled vs random pairs | Do OT couplings improve training? |

Same data splits, evaluation metrics, and hyperparameter budget for each comparison. No cherry-picking.

### 4. Context Sensitivity (Niche Shuffling)

Shuffle niche compositions across patients within the same stage. Measure prediction change. A context-sensitive model produces different trajectories for different niches. If shuffling does not change predictions, the context model is not contributing.

### 5. Trajectory Analysis

Do trajectories from the same source niche cluster? Do different edges produce qualitatively different trajectory shapes? Evidence of bifurcation? How does structure change across ablation conditions?

### 6. Fixed Points

Points where drift is near zero. Do they correspond to biologically meaningful states? How does WES regularization affect the fixed-point landscape?

### 7. Niche Regimes

Cluster niches by composition, compare transition dynamics across clusters. Identify compositions that accelerate, decelerate, or redirect transitions. Primary output relevant to the biological question.

### 8. Pseudotime-Like Structure

Project cells onto learned trajectories for pseudotime-like coordinates. Compare with independent methods (diffusion pseudotime, CellRank). Consistency increases confidence.

### 9. Gene/Program Attribution

Gradient-based attribution from drift network. Which genes drive velocity at key transitions? Do attributed genes match known LUAD biology?

### 10. Tissue-Level Reporting

Per-edge summary (dominant drift, trajectory duration, niche dependence). Per-stage summary (dynamic vs stable populations). Cross-edge comparison (how dynamics change with progression).

## Why Tissue Interpretation Matters

The goal is not just predicting cell positions. It is understanding how tissue microenvironment structure influences cancer initiation dynamics. Without this bridge between model outputs and biological claims, the model is a statistical exercise, not a scientific tool.
