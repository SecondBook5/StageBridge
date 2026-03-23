# StageBridge V1 Evaluation Protocol

**Last Updated:** 2026-03-22
**Status:** V1 Canonical Evaluation Specification

---

## 1. Overview

This document specifies the complete evaluation protocol for StageBridge V1. All results must pass these standards to be publication-ready.

### 1.1 Evaluation Principles

1. **Donor-held-out:** Primary evaluation unit is the donor
2. **Cross-validated:** Report mean ± std across folds
3. **Multi-metric:** Use complementary metrics per evaluation axis
4. **Negative controls:** Mandatory for all major claims
5. **Uncertainty aware:** Report calibration and coverage
6. **Backend robust:** Validate across multiple spatial backends

### 1.2 Five Evaluation Axes

1. Cell-level transition quality
2. Niche influence quality
3. Uncertainty quality
4. Evolutionary compatibility quality
5. Spatial backend robustness

---

## 2. Donor-Held-Out Cross-Validation

### 2.1 Split Strategy

**Method:** Stratified K-fold donor-level cross-validation

**Parameters:**
- K = 5 folds
- Stratification variables: stage distribution, smoking status
- Random seed: 42 (fixed for reproducibility)

**Split Sizes:**
- Train: 12 donors (70%)
- Validation: 3 donors (15%)
- Test: 3 donors (15%)

**Constraints:**
- All stages must appear in each split
- Balanced stage distribution where possible
- Genomics availability balanced across splits

### 2.2 Evaluation Procedure

For each fold:
1. Train on train donors
2. Select hyperparameters on validation donors
3. Evaluate on test donors
4. Save all metrics and predictions

**Aggregation:**
- Report mean ± std across 5 folds
- Bootstrap confidence intervals (1000 iterations)
- Statistical significance via paired t-test or Wilcoxon

### 2.3 Independence Unit

**Critical:** The donor is the independence unit, not the cell.

**Correct:**
```python
# Compute metric per donor, then aggregate
donor_metrics = []
for donor in test_donors:
    cells = dataset[dataset.donor_id == donor]
    metric = compute_metric(cells)
    donor_metrics.append(metric)
mean_metric = np.mean(donor_metrics)
std_metric = np.std(donor_metrics)
```

**Incorrect:**
```python
# DO NOT pool all cells and compute metric
all_cells = dataset[dataset.split == "test"]
metric = compute_metric(all_cells)  # PSEUDO-REPLICATION!
```

---

## 3. Cell-Level Transition Quality

### 3.1 Primary Metrics

**Metric 1: Wasserstein Distance**

```python
from scipy.stats import wasserstein_distance

def eval_wasserstein(predicted_latents, target_latents):
    """
    predicted_latents: (N, D) array of predicted cell states
    target_latents: (M, D) array of true target cell states
    """
    # Compute per-dimension Wasserstein, then average
    distances = []
    for d in range(predicted_latents.shape[1]):
        dist = wasserstein_distance(
            predicted_latents[:, d],
            target_latents[:, d]
        )
        distances.append(dist)
    return np.mean(distances)
```

**Interpretation:**
- Lower is better
- Units: Latent space distance
- Sensitive to distribution shape

**Metric 2: Maximum Mean Discrepancy (MMD)**

```python
def rbf_kernel(X, Y, gamma=1.0):
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    XY = X @ Y.T
    K = np.exp(-gamma * (XX - 2*XY + YY))
    return K

def mmd(X, Y, gamma=1.0):
    """MMD with RBF kernel"""
    Kxx = rbf_kernel(X, X, gamma).mean()
    Kyy = rbf_kernel(Y, Y, gamma).mean()
    Kxy = rbf_kernel(X, Y, gamma).mean()
    return Kxx + Kyy - 2 * Kxy
```

**Interpretation:**
- Lower is better
- Scale-free (depends on gamma)
- Robust to outliers

**Metric 3: KL Divergence (if normalized distributions)**

```python
from scipy.stats import entropy

def kl_divergence(p_pred, p_true, bins=50):
    """Estimate KL divergence via histograms"""
    # Compute histograms over latent space
    range_min = min(p_pred.min(), p_true.min())
    range_max = max(p_pred.max(), p_true.max())

    hist_pred, _ = np.histogram(p_pred, bins=bins, range=(range_min, range_max), density=True)
    hist_true, _ = np.histogram(p_true, bins=bins, range=(range_min, range_max), density=True)

    # Add small constant to avoid log(0)
    hist_pred = hist_pred + 1e-10
    hist_true = hist_true + 1e-10

    return entropy(hist_true, hist_pred)
```

### 3.2 Secondary Metrics

**Metric 4: Cosine Similarity**

```python
from sklearn.metrics.pairwise import cosine_similarity

def mean_cosine_similarity(pred, true):
    """Average cosine similarity between predicted and true"""
    # Match each predicted cell to nearest true cell
    similarities = cosine_similarity(pred, true)
    # Max similarity per predicted cell
    return similarities.max(axis=1).mean()
```

**Metric 5: Euclidean Distance**

```python
from scipy.spatial.distance import cdist

def nearest_neighbor_distance(pred, true):
    """Mean distance to nearest true cell"""
    distances = cdist(pred, true, metric='euclidean')
    return distances.min(axis=1).mean()
```

### 3.3 Baselines

**Baseline 1: Mean Target**
- Predict the mean of target distribution for all source cells
- Simplest baseline, no learning

**Baseline 2: Deterministic Regression**
- Train deterministic MLP: z_src → z_tgt
- No flow matching, no uncertainty

**Baseline 3: No Context**
- Flow matching without niche context
- Tests value of spatial information

**Baseline 4: Pooled Context**
- Flow matching with simple mean-pooled neighborhood
- Tests value of structured 9-token niche

### 3.4 Per-Edge Evaluation

Report metrics separately for each edge:
- Normal → AIS
- AIS → MIA
- MIA → Invasive
- Normal → Invasive (skip connection)

**Rationale:** Different edges have different difficulty and biological importance.

### 3.5 Success Criteria

**V1 passes if:**
- Full model significantly outperforms all baselines on test donors (p < 0.01)
- Improvement holds across all major edges
- Effect size (Cohen's d) > 0.5 for at least 2 baselines

---

## 4. Niche Influence Quality

### 4.1 Synthetic Benchmark (Ground Truth Available)

**Metric 1: Influence Recovery Accuracy**

Given synthetic data with known sender → receiver influences:

```python
def influence_recovery(true_influence, predicted_influence):
    """
    true_influence: (N_receivers, N_cell_types) ground truth weights
    predicted_influence: (N_receivers, N_cell_types) predicted weights
    """
    # Correlation per receiver
    correlations = []
    for i in range(len(true_influence)):
        corr = np.corrcoef(true_influence[i], predicted_influence[i])[0, 1]
        correlations.append(corr)
    return np.mean(correlations)
```

**Success Criterion:** Correlation > 0.5 on synthetic data

### 4.2 Real Data: Attention Analysis

**Metric 2: Attention Entropy**

```python
def attention_entropy(attention_weights):
    """
    attention_weights: (N_receivers, N_neighbors) attention matrix
    """
    # Normalize to probabilities
    probs = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    # Compute entropy per receiver
    entropies = -(probs * np.log(probs + 1e-10)).sum(axis=1)
    return np.mean(entropies)
```

**Interpretation:**
- High entropy: diffuse attention (many neighbors important)
- Low entropy: focused attention (few neighbors dominate)
- Expected: intermediate entropy, varies by cell type and stage

**Metric 3: Top-K Sender Attribution**

For each receiver, identify top-K most influential sender cell types:

```python
def top_k_sender_types(attention_weights, neighbor_cell_types, k=5):
    """Identify most influential sender cell types"""
    # Aggregate attention by cell type
    influence_by_type = {}
    for cell_type in np.unique(neighbor_cell_types):
        mask = (neighbor_cell_types == cell_type)
        influence_by_type[cell_type] = attention_weights[:, mask].sum(axis=1).mean()

    # Sort by influence
    sorted_types = sorted(influence_by_type.items(), key=lambda x: x[1], reverse=True)
    return sorted_types[:k]
```

### 4.3 Shuffle Sensitivity Test

**Metric 4: Shuffle Degradation**

```python
def shuffle_sensitivity(model, data, metric_fn, n_shuffles=10):
    """Measure metric degradation under neighborhood shuffling"""
    # Original metric
    original_metric = metric_fn(model.predict(data))

    # Shuffled metrics
    shuffled_metrics = []
    for _ in range(n_shuffles):
        # Shuffle neighborhood assignments
        shuffled_data = shuffle_neighborhoods(data)
        shuffled_metric = metric_fn(model.predict(shuffled_data))
        shuffled_metrics.append(shuffled_metric)

    # Return degradation
    degradation = original_metric - np.mean(shuffled_metrics)
    return degradation, np.std(shuffled_metrics)
```

**Success Criterion:**
- Degradation > 0 (metric worsens with shuffling)
- Effect size > 0.3 SD
- p < 0.01 (paired test)

### 4.4 Biological Plausibility

**Qualitative Checks:**
- Do epithelial cells attend to fibroblast/immune cells?
- Do immune cells attend to other immune cells?
- Do spatial distance constraints hold (nearby cells have higher influence)?
- Are cell-type-specific influence patterns interpretable?

**Generate for paper:**
- Sender → receiver heatmaps per cell type pair
- Spatial influence maps overlaid on tissue images
- Top-K sender tables per receiver type and stage

---

## 5. Uncertainty Quality

### 5.1 Calibration Metrics

**Metric 1: Expected Calibration Error (ECE)**

```python
def expected_calibration_error(confidences, accuracies, n_bins=10):
    """Compute ECE over binned predictions"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Find predictions in this bin
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if mask.sum() == 0:
            continue

        # Average confidence and accuracy in bin
        bin_confidence = confidences[mask].mean()
        bin_accuracy = accuracies[mask].mean()
        bin_weight = mask.sum() / len(confidences)

        # Weighted absolute difference
        ece += bin_weight * np.abs(bin_confidence - bin_accuracy)

    return ece
```

**Success Criterion:** ECE < 0.1

**Metric 2: Negative Log-Likelihood (NLL)**

```python
def negative_log_likelihood(predictions, targets, sigmas):
    """
    Gaussian NLL: -log p(target | prediction, sigma)
    """
    mse = ((predictions - targets) ** 2).sum(axis=1)
    log_sigmas_sq = 2 * np.log(sigmas + 1e-10)
    nll = 0.5 * (log_sigmas_sq + mse / (sigmas**2 + 1e-10))
    return nll.mean()
```

**Lower is better**

**Metric 3: Coverage**

For 90% prediction intervals, what fraction of true targets fall within?

```python
def coverage(predictions, targets, sigmas, alpha=0.1):
    """
    Compute empirical coverage of (1-alpha) prediction intervals
    """
    from scipy.stats import norm
    z_score = norm.ppf(1 - alpha/2)  # e.g., 1.96 for 95%

    # Compute intervals
    lower = predictions - z_score * sigmas
    upper = predictions + z_score * sigmas

    # Check if targets in interval
    in_interval = (targets >= lower) & (targets <= upper)
    return in_interval.mean()
```

**Success Criterion:** Coverage ≈ (1 - alpha) within ±5%

**Metric 4: Interval Width**

```python
def mean_interval_width(sigmas, alpha=0.1):
    """Average width of prediction intervals"""
    from scipy.stats import norm
    z_score = norm.ppf(1 - alpha/2)
    widths = 2 * z_score * sigmas
    return widths.mean()
```

**Should be:** As narrow as possible while maintaining coverage

### 5.2 Uncertainty Control Tests

**Test 1: Wrong-Stage Edges**

Predict cells on edges not seen in training (e.g., Invasive → Normal).

**Expected:** Higher uncertainty than training edges

**Test 2: Shuffled Neighborhoods**

Predict with randomly shuffled neighborhood contexts.

**Expected:** Higher uncertainty than true neighborhoods

**Test 3: Held-Out Donors**

Uncertainty should be higher on test donors than validation donors.

**Test 4: Low-Data Regions**

Rare cell types or rare transitions should have higher uncertainty.

### 5.3 Monte Carlo Uncertainty Estimation

```python
def mc_uncertainty_estimate(model, x, context, n_samples=100):
    """Estimate uncertainty via repeated stochastic forward passes"""
    predictions = []

    for _ in range(n_samples):
        # Stochastic forward pass (with dropout or flow noise)
        pred = model.predict_stochastic(x, context)
        predictions.append(pred)

    predictions = np.stack(predictions)  # (n_samples, batch_size, latent_dim)

    # Mean prediction
    mean_pred = predictions.mean(axis=0)

    # Uncertainty: standard deviation across samples
    std_pred = predictions.std(axis=0)

    return mean_pred, std_pred
```

### 5.4 Success Criteria

**V1 passes if:**
- ECE < 0.1 on test donors
- Coverage matches nominal level (within ±5%)
- Uncertainty increases on all negative controls
- NLL is finite and better than deterministic baseline

---

## 6. Evolutionary Compatibility Quality

### 6.1 Matched vs Mismatched Separation

**Primary Metric: Compatibility Score Gap**

```python
def compatibility_gap(model, data):
    """
    Compute gap between matched and mismatched compatibility scores
    """
    # Matched: same donor, same stage
    matched_scores = model.compute_compatibility(
        data.source_cells,
        data.target_cells_matched,
        data.wes_features
    )

    # Wrong donor
    wrong_donor_scores = model.compute_compatibility(
        data.source_cells,
        data.target_cells_wrong_donor,
        data.wes_features_shuffled_donor
    )

    # Wrong stage
    wrong_stage_scores = model.compute_compatibility(
        data.source_cells,
        data.target_cells_wrong_stage,
        data.wes_features_shuffled_stage
    )

    gap_donor = matched_scores.mean() - wrong_donor_scores.mean()
    gap_stage = matched_scores.mean() - wrong_stage_scores.mean()

    return gap_donor, gap_stage
```

**Success Criterion:**
- gap_donor > 0 with p < 0.01
- gap_stage > 0 with p < 0.01
- Effect size (Cohen's d) > 0.5

### 6.2 Effect Size

```python
def cohens_d(group1, group2):
    """Cohen's d effect size"""
    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(), group2.std()
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / pooled_std
```

### 6.3 Regularization Impact

**Metric: Implausible Transition Rate**

```python
def implausible_transition_rate(predictions, wes_features, threshold=0.3):
    """
    Fraction of predictions with compatibility < threshold
    """
    compatibility_scores = compute_compatibility(predictions, wes_features)
    implausible = (compatibility_scores < threshold).mean()
    return implausible
```

**Compare:**
- Model with genomic regularizer
- Model without genomic regularizer

**Expected:** Regularizer reduces implausible transition rate

### 6.4 Diagnostic Outputs

**For each test donor:**
- Distribution of matched compatibility scores
- Distribution of wrong-donor compatibility scores
- Distribution of wrong-stage compatibility scores
- Example high-compatibility transitions
- Example low-compatibility transitions (filtered by regularizer)

---

## 7. Spatial Backend Robustness

### 7.1 Upstream Quality Evaluation

**For each backend (Tangram, DestVI, TACCO):**

**Metric 1: Spatial Coherence**

```python
import squidpy as sq

def spatial_coherence(adata_spatial, cell_type_key="cell_type"):
    """Moran's I for spatial autocorrelation"""
    sq.gr.spatial_neighbors(adata_spatial)
    sq.gr.spatial_autocorr(
        adata_spatial,
        mode="moran",
        genes=None,
        n_perms=100
    )
    moran_i = adata_spatial.uns["moranI"]["I"].mean()
    return moran_i
```

**Higher = more spatially coherent**

**Metric 2: Proportion Quality**

```python
def proportion_entropy(proportions):
    """Entropy of cell type proportions per spot"""
    # proportions: (n_spots, n_cell_types)
    entropies = -(proportions * np.log(proportions + 1e-10)).sum(axis=1)
    return entropies.mean()
```

**Metric 3: Mapping Confidence**

```python
def confidence_stats(confidence_scores):
    """Summary statistics of mapping confidence"""
    return {
        "mean": confidence_scores.mean(),
        "median": np.median(confidence_scores),
        "q25": np.percentile(confidence_scores, 25),
        "q75": np.percentile(confidence_scores, 75),
        "low_confidence_frac": (confidence_scores < 0.5).mean()
    }
```

### 7.2 Downstream Utility Evaluation

**For each backend:**

**Metric 1: Transition Quality with Backend**

Run full StageBridge model using cells mapped by this backend.

```python
results = {}
for backend in ["tangram", "destvi", "tacco"]:
    model = train_stagebridge(backend=backend, ...)
    metrics = evaluate(model, test_data)
    results[backend] = metrics
```

**Compare:** Wasserstein distance, MMD, calibration across backends

**Metric 2: Niche Influence Consistency**

```python
def influence_consistency_across_backends(model_tangram, model_destvi, model_tacco):
    """
    Compute correlation of influence patterns across backends
    """
    influence_tangram = model_tangram.get_influence_tensor()
    influence_destvi = model_destvi.get_influence_tensor()
    influence_tacco = model_tacco.get_influence_tensor()

    corr_td = np.corrcoef(influence_tangram.flatten(), influence_destvi.flatten())[0,1]
    corr_tt = np.corrcoef(influence_tangram.flatten(), influence_tacco.flatten())[0,1]
    corr_dt = np.corrcoef(influence_destvi.flatten(), influence_tacco.flatten())[0,1]

    return {"tangram_destvi": corr_td, "tangram_tacco": corr_tt, "destvi_tacco": corr_dt}
```

**Success Criterion:** Correlations > 0.7

**Metric 3: Ablation Effect Sizes Across Backends**

Run Tier 1 ablations with each backend.

```python
ablation_effects = {}
for backend in backends:
    for ablation in ablations:
        effect_size = run_ablation(ablation, backend=backend)
        ablation_effects[(ablation, backend)] = effect_size
```

**Check:** Do ablation conclusions hold across backends?

### 7.3 Canonical Backend Selection

**Weighted Score:**

```
backend_score = w1 * upstream_quality
              + w2 * downstream_utility
              + w3 * robustness
              + w4 * practicality
```

**Weights (suggested):**
- w1 = 0.3 (upstream quality)
- w2 = 0.4 (downstream utility)
- w3 = 0.2 (robustness)
- w4 = 0.1 (runtime, ease of use)

**Select:** Backend with highest weighted score

**Document:** Rationale for selection with quantitative justification

### 7.4 Success Criteria

**V1 passes if:**
- All 3 backends run successfully
- Final biological conclusions hold across all 3 backends
- Canonical backend outperforms or matches alternatives on weighted score
- Backend choice is justified quantitatively

---

## 8. Statistical Testing

### 8.1 Paired Tests (Across Folds)

For comparing two models (e.g., full vs ablation):

```python
from scipy.stats import ttest_rel, wilcoxon

def compare_models(metrics_model_a, metrics_model_b):
    """
    metrics_model_a: (n_folds,) array
    metrics_model_b: (n_folds,) array
    """
    # Paired t-test (parametric)
    t_stat, p_value_t = ttest_rel(metrics_model_a, metrics_model_b)

    # Wilcoxon signed-rank test (non-parametric)
    w_stat, p_value_w = wilcoxon(metrics_model_a, metrics_model_b)

    # Effect size
    effect_size = cohens_d(metrics_model_a, metrics_model_b)

    return {
        "t_statistic": t_stat,
        "p_value_parametric": p_value_t,
        "p_value_nonparametric": p_value_w,
        "effect_size": effect_size
    }
```

### 8.2 Bootstrap Confidence Intervals

```python
from scipy.stats import bootstrap

def bootstrap_ci(data, statistic_fn, n_resamples=1000, confidence_level=0.95):
    """Compute bootstrap confidence interval"""
    result = bootstrap(
        (data,),
        statistic_fn,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method='percentile'
    )
    return result.confidence_interval
```

### 8.3 Multiple Comparisons Correction

When running multiple ablations:

```python
from statsmodels.stats.multitest import multipletests

def correct_pvalues(p_values, method='holm'):
    """
    Apply multiple comparisons correction
    method: 'bonferroni', 'holm', 'fdr_bh'
    """
    reject, p_corrected, _, _ = multipletests(p_values, method=method)
    return p_corrected, reject
```

### 8.4 Reporting Standards

For every comparison, report:
- Mean ± std for each group
- Test statistic (t or W)
- p-value (corrected if multiple comparisons)
- Effect size (Cohen's d or Cliff's delta)
- Confidence intervals

**Example Table:**

| Comparison | Model A | Model B | Δ | p-value | Effect Size |
|------------|---------|---------|---|---------|-------------|
| Full vs No-Context | 0.45±0.05 | 0.62±0.07 | -0.17 | <0.001 | 1.2 |

---

## 9. Negative Controls

### 9.1 Required Controls

**Control 1: Shuffled Neighborhoods**

Randomly reassign neighborhood contexts to receiver cells.

**Expected:** Transition quality degrades, uncertainty increases

**Control 2: Shuffled Donor Genomics**

Randomly reassign WES features across donors.

**Expected:** Compatibility gap disappears

**Control 3: Wrong-Stage Edges**

Evaluate on edges not in training graph (e.g., Invasive → Normal).

**Expected:** High uncertainty, low quality

**Control 4: Reference Ablation**

Remove HLCA or LuCA reference, use random embeddings.

**Expected:** Transition quality degrades

**Control 5: Degraded Spatial Backend**

Intentionally corrupt spatial backend outputs (add noise, shuffle proportions).

**Expected:** Transition quality degrades proportionally to corruption level

### 9.2 Positive Controls

**Control 1: Synthetic Data with Ground Truth**

Generate synthetic progression with known dynamics.

**Expected:** Model recovers ground truth transitions and influences

**Control 2: Within-Stage Transitions**

Predict Stage A → Stage A (no progression).

**Expected:** Near-identity map, very low Wasserstein distance

---

## 10. Artifact Generation

### 10.1 Per-Run Artifacts

Save for every training run:
- `config.yaml`: Resolved configuration
- `metrics.csv`: All metrics per epoch
- `diagnostics.json`: Model-specific diagnostics
- `predictions_test.pkl`: Test set predictions
- `uncertainty_test.pkl`: Test set uncertainties
- `checkpoint_best.pt`: Best model weights
- `git_commit.txt`: Code version
- `seed.txt`: Random seed

### 10.2 Per-Ablation Artifacts

Save for every ablation:
- `ablation_results.csv`: Metrics across all folds
- `ablation_summary.json`: Statistical test results
- `ablation_figures.pdf`: Visual comparisons

### 10.3 Final Publication Artifacts

Save for paper:
- `evidence_matrix.csv`: Claim → Evidence mapping
- `main_results_table.csv`: Table 3 for paper
- `ablation_heatmap.pdf`: Figure 7 for paper
- `backend_comparison.csv`: Table 5 for paper

---

## 11. Evaluation Script Template

```python
#!/usr/bin/env python
"""StageBridge V1 Evaluation Script"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from stagebridge.evaluation import (
    evaluate_transition_quality,
    evaluate_niche_influence,
    evaluate_uncertainty,
    evaluate_compatibility,
    evaluate_backend_robustness
)

def main():
    # Load configuration
    config = load_config("config.yaml")

    # Load trained model
    model = load_model("checkpoint_best.pt")

    # Load test data
    test_data = load_test_data(config)

    results = {}

    # 1. Transition quality
    print("Evaluating transition quality...")
    results["transition"] = evaluate_transition_quality(
        model, test_data,
        metrics=["wasserstein", "mmd", "kl", "cosine"]
    )

    # 2. Niche influence
    print("Evaluating niche influence...")
    results["niche"] = evaluate_niche_influence(
        model, test_data,
        shuffle_test=True,
        n_shuffles=10
    )

    # 3. Uncertainty
    print("Evaluating uncertainty...")
    results["uncertainty"] = evaluate_uncertainty(
        model, test_data,
        n_mc_samples=100,
        alpha=0.1
    )

    # 4. Compatibility
    print("Evaluating evolutionary compatibility...")
    results["compatibility"] = evaluate_compatibility(
        model, test_data,
        negative_controls=["wrong_donor", "wrong_stage"]
    )

    # 5. Backend robustness
    print("Evaluating spatial backend robustness...")
    results["backend"] = evaluate_backend_robustness(
        config,
        test_data,
        backends=["tangram", "destvi", "tacco"]
    )

    # Save results
    output_path = Path("evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    # Generate summary report
    generate_summary_report(results, "evaluation_report.pdf")

if __name__ == "__main__":
    main()
```

---

## 12. Success Criteria Summary

V1 evaluation is complete and publication-ready when:

-  All 5 evaluation axes show positive results
-  All baselines are outperformed significantly (p < 0.01)
-  Effect sizes > 0.5 for key comparisons
-  Uncertainty is calibrated (ECE < 0.1, coverage correct)
-  Evolutionary compatibility shows matched > shuffled (p < 0.01)
-  Results hold across all 3 spatial backends
-  All negative controls behave as expected
-  Statistical tests are properly corrected
-  All artifacts are saved and version-controlled
-  Evidence matrix is complete (every claim has evidence)

---

**End of Evaluation Protocol**
