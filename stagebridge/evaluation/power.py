"""Power analysis and sample size utilities for StageBridge.

Addresses reviewer concern: "Is N donors sufficient given donor-level clustering?"

Computes:
- Effective sample size accounting for intra-donor correlation
- Design effect (DEFF) from ICC
- Retrospective power analysis
- Minimum detectable effect size
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PowerAnalysisResult:
    """Results from power analysis."""

    n_donors: int
    n_cells: int
    cells_per_donor_mean: float
    cells_per_donor_std: float
    icc: float
    design_effect: float
    effective_sample_size: float
    observed_effect_size: float
    power: float
    min_detectable_effect: float
    interpretation: str


def compute_icc(
    values: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Compute intraclass correlation coefficient (ICC).

    ICC measures how much variance is between-group vs within-group.
    High ICC = strong donor effect = reduced effective sample size.

    Uses ICC(1,1) - one-way random effects, single measurement.

    Args:
        values: Outcome values (e.g., metric per cell)
        groups: Group labels (e.g., donor_id per cell)

    Returns:
        ICC value in [0, 1]. 0 = no clustering, 1 = perfect clustering.
    """
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    if k < 2:
        return 0.0

    # Compute group means and sizes
    group_means = []
    group_sizes = []
    for g in unique_groups:
        mask = groups == g
        group_means.append(np.mean(values[mask]))
        group_sizes.append(np.sum(mask))

    group_means = np.array(group_means)
    group_sizes = np.array(group_sizes)
    grand_mean = np.mean(values)
    n_total = len(values)

    # Between-group sum of squares
    ss_between = np.sum(group_sizes * (group_means - grand_mean) ** 2)

    # Within-group sum of squares
    ss_within = 0.0
    for g, gm in zip(unique_groups, group_means):
        mask = groups == g
        ss_within += np.sum((values[mask] - gm) ** 2)

    # Mean squares
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)

    # Average group size (harmonic mean for unbalanced)
    n_0 = (n_total - np.sum(group_sizes ** 2) / n_total) / (k - 1)

    # ICC(1,1)
    icc = (ms_between - ms_within) / (ms_between + (n_0 - 1) * ms_within)

    return max(0.0, min(1.0, icc))


def compute_design_effect(
    icc: float,
    avg_cluster_size: float,
) -> float:
    """Compute design effect (DEFF) for clustered data.

    DEFF = 1 + (m - 1) * ICC
    where m = average cluster size

    Args:
        icc: Intraclass correlation coefficient
        avg_cluster_size: Average number of observations per cluster

    Returns:
        Design effect. DEFF > 1 means clustering reduces effective N.
    """
    return 1.0 + (avg_cluster_size - 1) * icc


def compute_effective_sample_size(
    n_total: int,
    design_effect: float,
) -> float:
    """Compute effective sample size accounting for clustering.

    Args:
        n_total: Total number of observations
        design_effect: Design effect from compute_design_effect

    Returns:
        Effective sample size (always <= n_total)
    """
    return n_total / design_effect


def compute_power(
    effect_size: float,
    n_effective: float,
    alpha: float = 0.05,
) -> float:
    """Compute statistical power for two-sample comparison.

    Uses normal approximation for large samples.

    Args:
        effect_size: Cohen's d or standardized effect
        n_effective: Effective sample size per group
        alpha: Significance level

    Returns:
        Power in [0, 1]
    """
    if n_effective < 2 or effect_size == 0:
        return 0.0

    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n_effective / 2)

    # Critical value
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Power
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)

    return float(power)


def compute_min_detectable_effect(
    n_effective: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Compute minimum detectable effect size.

    Given sample size and desired power, what's the smallest effect
    we can reliably detect?

    Args:
        n_effective: Effective sample size per group
        alpha: Significance level
        power: Desired power

    Returns:
        Minimum detectable Cohen's d
    """
    if n_effective < 2:
        return np.inf

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    mde = (z_alpha + z_beta) * np.sqrt(2 / n_effective)

    return float(mde)


def run_power_analysis(
    metric_values: np.ndarray,
    donor_ids: np.ndarray,
    observed_effect: float | None = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> PowerAnalysisResult:
    """Run complete power analysis for donor-clustered data.

    Args:
        metric_values: Outcome values (one per cell/observation)
        donor_ids: Donor ID for each observation
        observed_effect: Observed effect size (Cohen's d). If None, computed from data.
        alpha: Significance level
        target_power: Target power for MDE calculation

    Returns:
        PowerAnalysisResult with all computed values
    """
    metric_values = np.asarray(metric_values)
    donor_ids = np.asarray(donor_ids)

    # Remove NaNs
    valid = ~np.isnan(metric_values)
    metric_values = metric_values[valid]
    donor_ids = donor_ids[valid]

    n_cells = len(metric_values)
    unique_donors = np.unique(donor_ids)
    n_donors = len(unique_donors)

    # Cells per donor stats
    cells_per_donor = [np.sum(donor_ids == d) for d in unique_donors]
    cells_per_donor_mean = np.mean(cells_per_donor)
    cells_per_donor_std = np.std(cells_per_donor)

    # ICC
    icc = compute_icc(metric_values, donor_ids)

    # Design effect
    deff = compute_design_effect(icc, cells_per_donor_mean)

    # Effective sample size
    n_eff = compute_effective_sample_size(n_cells, deff)

    # Effect size (if not provided, use standardized mean)
    if observed_effect is None:
        observed_effect = np.mean(metric_values) / (np.std(metric_values) + 1e-10)

    # Power
    power = compute_power(observed_effect, n_eff / 2, alpha)

    # Minimum detectable effect
    mde = compute_min_detectable_effect(n_eff / 2, alpha, target_power)

    # Interpretation
    if icc < 0.05:
        cluster_interp = "negligible donor clustering (ICC < 0.05)"
    elif icc < 0.15:
        cluster_interp = "mild donor clustering (0.05 <= ICC < 0.15)"
    elif icc < 0.30:
        cluster_interp = "moderate donor clustering (0.15 <= ICC < 0.30)"
    else:
        cluster_interp = "strong donor clustering (ICC >= 0.30)"

    if power >= 0.80:
        power_interp = f"adequately powered ({power:.0%})"
    elif power >= 0.50:
        power_interp = f"underpowered ({power:.0%}), consider larger sample"
    else:
        power_interp = f"severely underpowered ({power:.0%})"

    interpretation = f"{cluster_interp}; {power_interp}; MDE = {mde:.2f}"

    return PowerAnalysisResult(
        n_donors=n_donors,
        n_cells=n_cells,
        cells_per_donor_mean=float(cells_per_donor_mean),
        cells_per_donor_std=float(cells_per_donor_std),
        icc=float(icc),
        design_effect=float(deff),
        effective_sample_size=float(n_eff),
        observed_effect_size=float(observed_effect),
        power=float(power),
        min_detectable_effect=float(mde),
        interpretation=interpretation,
    )


def power_analysis_from_cv_results(
    cv_results: pd.DataFrame,
    metric_col: str,
    fold_col: str = "fold",
) -> dict[str, Any]:
    """Run power analysis treating CV folds as clusters.

    For StageBridge, each fold uses different test donors, so fold-level
    variance approximates donor-level variance.

    Args:
        cv_results: DataFrame with CV results
        metric_col: Column name for metric to analyze
        fold_col: Column name for fold identifier

    Returns:
        Dictionary with power analysis results
    """
    values = cv_results[metric_col].values
    folds = cv_results[fold_col].values

    result = run_power_analysis(values, folds)

    # Additional CV-specific stats
    fold_means = cv_results.groupby(fold_col)[metric_col].mean()
    fold_variance = fold_means.var()
    overall_variance = cv_results[metric_col].var()

    return {
        "n_folds": len(fold_means),
        "n_runs": len(cv_results),
        "fold_means": fold_means.to_dict(),
        "fold_variance": float(fold_variance),
        "overall_variance": float(overall_variance),
        "variance_ratio": float(fold_variance / (overall_variance + 1e-10)),
        "power_analysis": {
            "icc": result.icc,
            "design_effect": result.design_effect,
            "effective_n": result.effective_sample_size,
            "power": result.power,
            "min_detectable_effect": result.min_detectable_effect,
            "interpretation": result.interpretation,
        }
    }


def generate_power_report(
    results: list[PowerAnalysisResult],
    metric_names: list[str],
) -> str:
    """Generate human-readable power analysis report.

    Args:
        results: List of PowerAnalysisResult objects
        metric_names: Names for each result

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "POWER ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Sample: {results[0].n_donors} donors, {results[0].n_cells:,} cells",
        f"Cells per donor: {results[0].cells_per_donor_mean:.0f} +/- {results[0].cells_per_donor_std:.0f}",
        "",
        "-" * 70,
        f"{'Metric':<25} {'ICC':>8} {'DEFF':>8} {'Eff.N':>10} {'Power':>8} {'MDE':>8}",
        "-" * 70,
    ]

    for name, r in zip(metric_names, results):
        lines.append(
            f"{name:<25} {r.icc:>8.3f} {r.design_effect:>8.1f} "
            f"{r.effective_sample_size:>10.0f} {r.power:>8.1%} {r.min_detectable_effect:>8.2f}"
        )

    lines.extend([
        "-" * 70,
        "",
        "Interpretation:",
    ])

    for name, r in zip(metric_names, results):
        lines.append(f"  {name}: {r.interpretation}")

    lines.extend([
        "",
        "Notes:",
        "  - ICC: Intraclass correlation (0=no clustering, 1=perfect clustering)",
        "  - DEFF: Design effect (effective sample inflation factor)",
        "  - Eff.N: Effective sample size after accounting for clustering",
        "  - Power: Statistical power at observed effect size (alpha=0.05)",
        "  - MDE: Minimum detectable effect for 80% power",
        "=" * 70,
    ])

    return "\n".join(lines)
