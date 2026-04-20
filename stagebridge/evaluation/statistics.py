"""Statistical comparison utilities for StageBridge.

Computes:
- Effect sizes (Cohen's d, Cliff's delta)
- Statistical tests (paired t-test, Wilcoxon signed-rank)
- Bootstrap confidence intervals
- Multiple comparison corrections (Bonferroni, FDR)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size.

    Args:
        x: First sample
        y: Second sample

    Returns:
        Cohen's d value
    """
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / (pooled_std + 1e-10))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta (non-parametric effect size).

    Args:
        x: First sample
        y: Second sample

    Returns:
        Cliff's delta value in [-1, 1]
    """
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return float((more - less) / (n_x * n_y))


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Sample values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (0-1)
        seed: Random seed

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return (float(np.mean(values)), np.nan, np.nan)

    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci
    return (
        float(np.mean(values)),
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude.

    Args:
        d: Cohen's d value

    Returns:
        Interpretation string
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def paired_comparison(
    baseline_values: np.ndarray,
    model_values: np.ndarray,
    metric_name: str,
) -> dict[str, Any]:
    """Comprehensive paired comparison between baseline and model.

    Args:
        baseline_values: Baseline metric values
        model_values: Model metric values
        metric_name: Name of the metric

    Returns:
        Dictionary with comparison statistics
    """
    baseline = np.asarray(baseline_values)
    model = np.asarray(model_values)

    mask = ~(np.isnan(baseline) | np.isnan(model))
    baseline = baseline[mask]
    model = model[mask]

    if len(baseline) < 3:
        return {"error": "insufficient_samples", "n": len(baseline)}

    diff = model - baseline
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    t_stat, t_pvalue = stats.ttest_rel(model, baseline)

    try:
        w_stat, w_pvalue = stats.wilcoxon(model, baseline, alternative="two-sided")
    except ValueError:
        w_stat, w_pvalue = np.nan, np.nan

    d = cohens_d(model, baseline)
    delta = cliffs_delta(model, baseline)

    baseline_ci = bootstrap_ci(baseline)
    model_ci = bootstrap_ci(model)
    diff_ci = bootstrap_ci(diff)

    return {
        "metric": metric_name,
        "n_pairs": len(baseline),
        "baseline_mean": float(np.mean(baseline)),
        "baseline_std": float(np.std(baseline, ddof=1)),
        "baseline_ci_lower": baseline_ci[1],
        "baseline_ci_upper": baseline_ci[2],
        "model_mean": float(np.mean(model)),
        "model_std": float(np.std(model, ddof=1)),
        "model_ci_lower": model_ci[1],
        "model_ci_upper": model_ci[2],
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
        "diff_ci_lower": diff_ci[1],
        "diff_ci_upper": diff_ci[2],
        "relative_improvement_pct": float(mean_diff / (np.abs(np.mean(baseline)) + 1e-10) * 100),
        "paired_ttest_statistic": float(t_stat),
        "paired_ttest_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_pvalue": float(w_pvalue) if not np.isnan(w_pvalue) else None,
        "cohens_d": d,
        "cliffs_delta": delta,
        "effect_size_interpretation": interpret_effect_size(d),
    }


def apply_multiple_comparison_correction(
    pvalues: list[float],
    method: str = "fdr_bh",
) -> tuple[list[float], list[bool]]:
    """Apply multiple comparison correction.

    Args:
        pvalues: List of p-values
        method: Correction method ("bonferroni" or "fdr_bh")

    Returns:
        Tuple of (corrected p-values, rejection decisions)
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)

    if method == "bonferroni":
        corrected = np.minimum(pvalues * n, 1.0)
        reject = corrected < 0.05
    elif method == "fdr_bh":
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        corrected = np.zeros(n)
        for i, (idx, p) in enumerate(zip(sorted_idx, sorted_pvals)):
            corrected[idx] = p * n / (i + 1)
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        corrected = np.minimum(corrected, 1.0)
        reject = corrected < 0.05
    else:
        corrected = pvalues
        reject = pvalues < 0.05

    return corrected.tolist(), reject.tolist()


def generate_latex_comparison_table(
    comparisons: list[dict],
    output_path: Path,
) -> None:
    """Generate LaTeX table for publication.

    Args:
        comparisons: List of comparison dictionaries
        output_path: Output file path
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical Comparison: StageBridge vs Baselines}",
        r"\label{tab:statistical_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Metric & Baseline & StageBridge & $\Delta$ & $p$-value & Effect \\",
        r"\midrule",
    ]

    for comp in comparisons:
        if "error" in comp:
            continue

        baseline = f"${comp['baseline_mean']:.3f} \\pm {comp['baseline_std']:.3f}$"
        model = f"${comp['model_mean']:.3f} \\pm {comp['model_std']:.3f}$"
        delta = f"${comp['mean_difference']:+.3f}$"

        p = comp["paired_ttest_pvalue"]
        if p < 0.001:
            p_str = "$<0.001$"
        elif p < 0.01:
            p_str = f"${p:.3f}$"
        else:
            p_str = f"${p:.2f}$"

        effect = comp["effect_size_interpretation"]

        lines.append(f"{comp['metric']} & {baseline} & {model} & {delta} & {p_str} & {effect} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2mm}",
        r"\footnotesize{Values are mean $\pm$ std over 5-fold CV $\times$ 3 seeds. ",
        r"$p$-values from paired $t$-test. Effect sizes: Cohen's $d$.}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines))
