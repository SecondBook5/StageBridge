"""Granger causality analysis for niche-cell state relationships.

Inspired by Cflows (Sun et al., 2025) - applies Granger causality to
predicted trajectories to identify causal niche signals.

Tests whether niche signals (L-R, pathway activities) "Granger-cause"
cell state transitions - i.e., past niche predicts future state better
than past state alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from scipy import stats


def granger_causality_test(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 3,
) -> Dict[str, float]:
    """Test if x Granger-causes y.

    Tests whether past values of x help predict y beyond
    what past values of y alone provide.

    Args:
        x: Potential cause (niche signal), shape (n_samples,)
        y: Effect (cell state), shape (n_samples,)
        max_lag: Maximum lag to test

    Returns:
        Dict with F-statistic, p-value, and optimal lag
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    n = len(y)
    best_result = {"f_stat": 0, "p_value": 1, "lag": 1}

    for lag in range(1, max_lag + 1):
        if n <= 2 * lag + 2:
            continue

        # Restricted model: y ~ y_lagged only
        Y = y[lag:]
        Y_lagged = np.column_stack([y[lag-i-1:-i-1] for i in range(lag)])

        X_restricted = add_constant(Y_lagged)
        model_restricted = OLS(Y, X_restricted).fit()
        rss_restricted = model_restricted.ssr

        # Unrestricted model: y ~ y_lagged + x_lagged
        X_lagged = np.column_stack([x[lag-i-1:-i-1] for i in range(lag)])
        X_unrestricted = add_constant(np.column_stack([Y_lagged, X_lagged]))
        model_unrestricted = OLS(Y, X_unrestricted).fit()
        rss_unrestricted = model_unrestricted.ssr

        # F-test
        df1 = lag  # Number of restrictions
        df2 = len(Y) - 2 * lag - 1  # Residual df

        if df2 > 0 and rss_unrestricted > 0:
            f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)

            if f_stat > best_result["f_stat"]:
                best_result = {"f_stat": f_stat, "p_value": p_value, "lag": lag}

    return best_result


def test_niche_granger_causes_state(
    niche_features: pd.DataFrame,
    cell_states: np.ndarray,
    pseudotime: np.ndarray,
    n_bins: int = 50,
    max_lag: int = 3,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test which niche features Granger-cause cell state changes.

    Bins cells by pseudotime and tests Granger causality of
    niche signals on cell state trajectory.

    Args:
        niche_features: DataFrame with niche signals (L-R scores, pathway activities)
        cell_states: Cell state values (e.g., first PC of latent, EMT score)
        pseudotime: Pseudotime or transition probability ordering
        n_bins: Number of pseudotime bins
        max_lag: Maximum lag for Granger test
        alpha: Significance threshold

    Returns:
        DataFrame with Granger causality results per feature
    """
    # Bin by pseudotime
    bins = np.linspace(pseudotime.min(), pseudotime.max(), n_bins + 1)
    bin_indices = np.digitize(pseudotime, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute binned means
    binned_state = np.array([
        cell_states[bin_indices == i].mean() if (bin_indices == i).sum() > 0 else np.nan
        for i in range(n_bins)
    ])

    results = []
    for col in niche_features.columns:
        binned_niche = np.array([
            niche_features[col].values[bin_indices == i].mean()
            if (bin_indices == i).sum() > 0 else np.nan
            for i in range(n_bins)
        ])

        # Remove NaN bins
        valid = ~(np.isnan(binned_state) | np.isnan(binned_niche))
        if valid.sum() < 10:
            continue

        gc_result = granger_causality_test(
            binned_niche[valid],
            binned_state[valid],
            max_lag=max_lag
        )

        results.append({
            "feature": col,
            "f_stat": gc_result["f_stat"],
            "p_value": gc_result["p_value"],
            "optimal_lag": gc_result["lag"],
            "significant": gc_result["p_value"] < alpha,
        })

    df = pd.DataFrame(results)

    # Multiple testing correction (Benjamini-Hochberg)
    if len(df) > 0:
        from statsmodels.stats.multitest import multipletests
        _, df["p_value_adj"], _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["significant_adj"] = df["p_value_adj"] < alpha
        df = df.sort_values("p_value")

    return df


def test_lr_granger_causality(
    adata,
    lr_scores: pd.DataFrame,
    latent: np.ndarray,
    pseudotime: np.ndarray,
    n_top_pairs: int = 50,
) -> pd.DataFrame:
    """Test which L-R pairs Granger-cause cell state transitions.

    Args:
        adata: AnnData with cell info
        lr_scores: DataFrame of L-R interaction scores
        latent: Latent representation (n_cells, n_dims)
        pseudotime: Pseudotime ordering
        n_top_pairs: Number of top variable L-R pairs to test

    Returns:
        DataFrame with Granger causality results
    """
    # Use first PC of latent as cell state summary
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    cell_state_1d = pca.fit_transform(latent).flatten()

    # Select most variable L-R pairs
    lr_var = lr_scores.var()
    top_pairs = lr_var.nlargest(n_top_pairs).index.tolist()

    print(f"Testing Granger causality for {len(top_pairs)} L-R pairs...")

    results = test_niche_granger_causes_state(
        niche_features=lr_scores[top_pairs],
        cell_states=cell_state_1d,
        pseudotime=pseudotime,
    )

    return results


def test_pathway_granger_causality(
    pathway_scores: pd.DataFrame,
    latent: np.ndarray,
    pseudotime: np.ndarray,
) -> pd.DataFrame:
    """Test which pathways Granger-cause cell state transitions.

    Args:
        pathway_scores: DataFrame of pathway activity scores (e.g., PROGENy)
        latent: Latent representation
        pseudotime: Pseudotime ordering

    Returns:
        DataFrame with Granger causality results
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    cell_state_1d = pca.fit_transform(latent).flatten()

    print(f"Testing Granger causality for {len(pathway_scores.columns)} pathways...")

    results = test_niche_granger_causes_state(
        niche_features=pathway_scores,
        cell_states=cell_state_1d,
        pseudotime=pseudotime,
    )

    return results


def plot_granger_results(
    results: pd.DataFrame,
    title: str = "Granger Causality: Niche → Cell State",
    top_n: int = 20,
    save_path: Optional[str] = None,
):
    """Plot Granger causality results.

    Args:
        results: DataFrame from granger causality tests
        title: Plot title
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    df = results.head(top_n).copy()
    df = df.iloc[::-1]  # Reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(df) + 2))

    colors = ['#d62728' if sig else '#1f77b4'
              for sig in df["significant_adj"]]

    ax.barh(df["feature"], -np.log10(df["p_value_adj"]), color=colors)
    ax.axvline(-np.log10(0.05), color='gray', linestyle='--', label='p=0.05')
    ax.set_xlabel('-log10(adjusted p-value)')
    ax.set_title(title)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Significant'),
        Patch(facecolor='#1f77b4', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def evaluate_causal_niche_signals(
    adata,
    latent: np.ndarray,
    pseudotime: np.ndarray,
    lr_scores: Optional[pd.DataFrame] = None,
    pathway_scores: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Full Granger causality evaluation of niche signals.

    Args:
        adata: AnnData
        latent: Cell latent representation
        pseudotime: Transition probability / pseudotime
        lr_scores: L-R interaction scores (optional)
        pathway_scores: Pathway activity scores (optional)
        output_dir: Directory to save results

    Returns:
        Dict with 'lr' and 'pathway' DataFrames of results
    """
    from pathlib import Path

    results = {}

    if lr_scores is not None:
        print("\n=== L-R Granger Causality ===")
        lr_results = test_lr_granger_causality(
            adata, lr_scores, latent, pseudotime
        )
        results["lr"] = lr_results

        n_sig = lr_results["significant_adj"].sum()
        print(f"  Significant L-R pairs: {n_sig}/{len(lr_results)}")
        if n_sig > 0:
            print(f"  Top causal pairs:")
            for _, row in lr_results[lr_results["significant_adj"]].head(5).iterrows():
                print(f"    {row['feature']}: F={row['f_stat']:.2f}, p={row['p_value_adj']:.2e}")

    if pathway_scores is not None:
        print("\n=== Pathway Granger Causality ===")
        pathway_results = test_pathway_granger_causality(
            pathway_scores, latent, pseudotime
        )
        results["pathway"] = pathway_results

        n_sig = pathway_results["significant_adj"].sum()
        print(f"  Significant pathways: {n_sig}/{len(pathway_results)}")
        if n_sig > 0:
            print(f"  Top causal pathways:")
            for _, row in pathway_results[pathway_results["significant_adj"]].head(5).iterrows():
                print(f"    {row['feature']}: F={row['f_stat']:.2f}, p={row['p_value_adj']:.2e}")

    # Save and plot
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if "lr" in results:
            results["lr"].to_csv(output_dir / "granger_lr_results.csv", index=False)
            plot_granger_results(
                results["lr"],
                title="L-R Pairs Granger-Causing Cell State",
                save_path=str(output_dir / "granger_lr_plot.png")
            )

        if "pathway" in results:
            results["pathway"].to_csv(output_dir / "granger_pathway_results.csv", index=False)
            plot_granger_results(
                results["pathway"],
                title="Pathways Granger-Causing Cell State",
                save_path=str(output_dir / "granger_pathway_plot.png")
            )

    return results
