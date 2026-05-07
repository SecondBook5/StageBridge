"""EMT (Epithelial-Mesenchymal Transition) scoring for StageBridge evaluation.

Evaluates whether predicted cell state transitions correlate with EMT signatures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# Core EMT markers
EMT_GENES = {
    "epithelial": ["CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "CLDN1", "OCLN"],
    "mesenchymal": ["VIM", "CDH2", "FN1", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2"],
    "partial_emt": ["SNAI1", "ZEB1", "CDH1"],  # Co-expression indicates partial EMT
}

# Extended EMT signature from literature (Tan et al., Hallmarks of EMT)
EMT_SIGNATURE_UP = [
    "VIM", "CDH2", "FN1", "SNAI1", "SNAI2", "ZEB1", "ZEB2",
    "TWIST1", "TWIST2", "MMP2", "MMP9", "SERPINE1", "ACTA2",
    "COL1A1", "COL3A1", "TGFB1", "TGFB2", "WNT5A",
]

EMT_SIGNATURE_DOWN = [
    "CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "CLDN1", "CLDN4",
    "OCLN", "DSP", "PKP3", "MUC1",
]


def score_emt(
    adata,
    layer: Optional[str] = None,
    use_raw: bool = True,
) -> pd.DataFrame:
    """Compute EMT scores for each cell.

    Args:
        adata: AnnData with expression data
        layer: Layer to use (None = X)
        use_raw: Use adata.raw if available

    Returns:
        DataFrame with columns: epithelial_score, mesenchymal_score, emt_score, partial_emt
    """
    import scanpy as sc

    # Get expression matrix
    if use_raw and adata.raw is not None:
        expr_adata = adata.raw.to_adata()
    else:
        expr_adata = adata

    if layer is not None:
        X = expr_adata.layers[layer]
    else:
        X = expr_adata.X

    var_names = list(expr_adata.var_names)
    var_names_upper = [g.upper() for g in var_names]

    def get_gene_expr(gene_list):
        """Get mean expression of genes that exist in data."""
        indices = []
        found = []
        for gene in gene_list:
            gene_upper = gene.upper()
            if gene_upper in var_names_upper:
                indices.append(var_names_upper.index(gene_upper))
                found.append(gene)

        if not indices:
            return np.zeros(X.shape[0]), []

        if hasattr(X, 'toarray'):
            expr = X[:, indices].toarray()
        else:
            expr = X[:, indices]

        return np.mean(expr, axis=1), found

    # Score epithelial markers
    epi_score, epi_found = get_gene_expr(EMT_GENES["epithelial"])
    print(f"  Epithelial markers found: {len(epi_found)}/{len(EMT_GENES['epithelial'])}")

    # Score mesenchymal markers
    mes_score, mes_found = get_gene_expr(EMT_GENES["mesenchymal"])
    print(f"  Mesenchymal markers found: {len(mes_found)}/{len(EMT_GENES['mesenchymal'])}")

    # EMT score: mesenchymal - epithelial (higher = more mesenchymal)
    emt_score = mes_score - epi_score

    # Partial EMT: high SNAI1/ZEB1 but still CDH1+
    partial_genes = EMT_GENES["partial_emt"]
    partial_expr, partial_found = get_gene_expr(partial_genes)

    # Partial EMT flag: cells with intermediate EMT score
    emt_25 = np.percentile(emt_score, 25)
    emt_75 = np.percentile(emt_score, 75)
    partial_emt = (emt_score > emt_25) & (emt_score < emt_75)

    return pd.DataFrame({
        "epithelial_score": epi_score,
        "mesenchymal_score": mes_score,
        "emt_score": emt_score,
        "partial_emt": partial_emt,
    }, index=adata.obs_names)


def score_emt_extended(
    adata,
    layer: Optional[str] = None,
    use_raw: bool = True,
) -> pd.DataFrame:
    """Compute extended EMT signature score (up - down genes)."""
    import scanpy as sc

    if use_raw and adata.raw is not None:
        expr_adata = adata.raw.to_adata()
    else:
        expr_adata = adata

    if layer is not None:
        X = expr_adata.layers[layer]
    else:
        X = expr_adata.X

    var_names = list(expr_adata.var_names)
    var_names_upper = [g.upper() for g in var_names]

    def get_mean_expr(gene_list):
        indices = []
        for gene in gene_list:
            if gene.upper() in var_names_upper:
                indices.append(var_names_upper.index(gene.upper()))
        if not indices:
            return np.zeros(X.shape[0])
        if hasattr(X, 'toarray'):
            return np.mean(X[:, indices].toarray(), axis=1)
        return np.mean(X[:, indices], axis=1)

    up_score = get_mean_expr(EMT_SIGNATURE_UP)
    down_score = get_mean_expr(EMT_SIGNATURE_DOWN)

    return pd.DataFrame({
        "emt_up_score": up_score,
        "emt_down_score": down_score,
        "emt_signature_score": up_score - down_score,
    }, index=adata.obs_names)


def evaluate_emt_along_trajectory(
    adata,
    transition_probs: np.ndarray,
    stage_col: str = "stage",
) -> dict:
    """Evaluate if EMT increases along predicted transition trajectories.

    Args:
        adata: AnnData with expression and stage info
        transition_probs: Per-cell transition probability from model
        stage_col: Column containing stage labels

    Returns:
        Dict with correlation stats and per-stage EMT scores
    """
    from scipy.stats import spearmanr, pearsonr

    # Get EMT scores
    emt_df = score_emt(adata)

    # Correlation between transition probability and EMT
    mask = ~np.isnan(transition_probs)

    spearman_r, spearman_p = spearmanr(transition_probs[mask], emt_df["emt_score"].values[mask])
    pearson_r, pearson_p = pearsonr(transition_probs[mask], emt_df["emt_score"].values[mask])

    # Per-stage EMT scores
    stage_emt = {}
    if stage_col in adata.obs.columns:
        for stage in adata.obs[stage_col].unique():
            mask_stage = adata.obs[stage_col] == stage
            stage_emt[stage] = {
                "mean_emt": emt_df.loc[mask_stage, "emt_score"].mean(),
                "std_emt": emt_df.loc[mask_stage, "emt_score"].std(),
                "partial_emt_frac": emt_df.loc[mask_stage, "partial_emt"].mean(),
            }

    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "stage_emt": stage_emt,
        "interpretation": (
            "Positive correlation suggests transitions capture EMT-like changes. "
            "Partial EMT in preinvasive stages indicates plasticity window."
        ),
    }


def plot_emt_by_stage(
    adata,
    stage_col: str = "stage",
    save_path: Optional[str] = None,
):
    """Plot EMT scores stratified by disease stage."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    emt_df = score_emt(adata)
    emt_df["stage"] = adata.obs[stage_col].values

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # EMT score by stage
    stage_order = ["Normal", "Preinvasive", "Invasive"]
    stage_order = [s for s in stage_order if s in emt_df["stage"].unique()]

    sns.boxplot(data=emt_df, x="stage", y="emt_score", order=stage_order, ax=axes[0])
    axes[0].set_title("EMT Score by Stage")
    axes[0].set_ylabel("EMT Score (Mes - Epi)")

    # Epithelial vs Mesenchymal scores
    sns.boxplot(data=emt_df, x="stage", y="epithelial_score", order=stage_order, ax=axes[1])
    axes[1].set_title("Epithelial Markers by Stage")

    sns.boxplot(data=emt_df, x="stage", y="mesenchymal_score", order=stage_order, ax=axes[2])
    axes[2].set_title("Mesenchymal Markers by Stage")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_emt_along_pseudotime(
    adata,
    pseudotime: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot EMT score along pseudotime/transition trajectory."""
    import matplotlib.pyplot as plt

    emt_df = score_emt(adata)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bin pseudotime and compute mean EMT
    n_bins = 50
    bins = np.linspace(pseudotime.min(), pseudotime.max(), n_bins + 1)
    bin_indices = np.digitize(pseudotime, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = []
    bin_emt_mean = []
    bin_emt_std = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 10:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_emt_mean.append(emt_df.loc[mask, "emt_score"].mean())
            bin_emt_std.append(emt_df.loc[mask, "emt_score"].std())

    bin_centers = np.array(bin_centers)
    bin_emt_mean = np.array(bin_emt_mean)
    bin_emt_std = np.array(bin_emt_std)

    ax.fill_between(bin_centers, bin_emt_mean - bin_emt_std, bin_emt_mean + bin_emt_std, alpha=0.3)
    ax.plot(bin_centers, bin_emt_mean, linewidth=2)

    ax.set_xlabel("Pseudotime / Transition Probability")
    ax.set_ylabel("EMT Score")
    ax.set_title("EMT Along Predicted Trajectory")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# Key EMT regulators from Cflows (Sun et al., 2025) and literature
EMT_REGULATORS = {
    "transcription_factors": [
        "ESRRA",  # Key driver in Cflows - regulates CDH1, SNAI1
        "SNAI1", "SNAI2",  # SNAIL family
        "ZEB1", "ZEB2",  # ZEB family
        "TWIST1", "TWIST2",  # TWIST family
        "AHR",  # Aryl hydrocarbon receptor - in Cflows network
    ],
    "signaling": [
        "TGFB1", "TGFB2", "TGFB3",  # TGF-beta - major EMT inducer
        "WNT5A", "WNT3A",  # WNT signaling
        "NOTCH1", "NOTCH2",  # Notch signaling
        "IL6", "IL1B",  # Inflammatory - relevant to our niche hypothesis
    ],
}


def test_emt_granger_causality(
    adata,
    niche_features: pd.DataFrame,
    pseudotime: np.ndarray,
    use_raw: bool = True,
    output_dir: Optional[str] = None,
) -> dict:
    """Test which niche signals Granger-cause EMT.

    Inspired by Cflows approach - tests causal relationships between
    niche signals and EMT progression.

    Args:
        adata: AnnData with expression
        niche_features: DataFrame with niche signals (L-R scores, etc.)
        pseudotime: Pseudotime / transition probability ordering
        use_raw: Use raw counts for EMT scoring
        output_dir: Directory to save results

    Returns:
        Dict with Granger causality results and EMT scores
    """
    from pathlib import Path

    # Import granger module
    from stagebridge.biology.granger_causality import test_niche_granger_causes_state, plot_granger_results

    # Get EMT scores
    emt_df = score_emt(adata, use_raw=use_raw)
    emt_score = emt_df["emt_score"].values

    print("=== EMT Granger Causality Analysis ===")
    print(f"  Testing {len(niche_features.columns)} niche features")

    # Test Granger causality: niche → EMT
    results = test_niche_granger_causes_state(
        niche_features=niche_features,
        cell_states=emt_score,
        pseudotime=pseudotime,
    )

    n_sig = results["significant_adj"].sum() if len(results) > 0 else 0
    print(f"  Significant niche → EMT: {n_sig}/{len(results)}")

    if n_sig > 0:
        print(f"\n  Top causal signals for EMT:")
        for _, row in results[results["significant_adj"]].head(5).iterrows():
            print(f"    {row['feature']}: F={row['f_stat']:.2f}, p={row['p_value_adj']:.2e}, lag={row['optimal_lag']}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results.to_csv(output_dir / "emt_granger_results.csv", index=False)

        if len(results) > 0:
            plot_granger_results(
                results,
                title="Niche Signals Granger-Causing EMT",
                save_path=str(output_dir / "emt_granger_plot.png")
            )

    return {
        "granger_results": results,
        "emt_scores": emt_df,
    }


def analyze_emt_regulators(
    adata,
    pseudotime: np.ndarray,
    use_raw: bool = True,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Analyze expression of known EMT regulators along trajectory.

    Args:
        adata: AnnData with expression
        pseudotime: Pseudotime ordering
        use_raw: Use raw counts
        output_dir: Directory to save results

    Returns:
        DataFrame with regulator dynamics
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    if use_raw and adata.raw is not None:
        expr_adata = adata.raw.to_adata()
    else:
        expr_adata = adata

    var_names = list(expr_adata.var_names)
    var_names_upper = [g.upper() for g in var_names]

    # Collect all regulators
    all_regulators = EMT_REGULATORS["transcription_factors"] + EMT_REGULATORS["signaling"]

    # Get expression for available regulators
    results = []
    available = []

    for gene in all_regulators:
        if gene.upper() in var_names_upper:
            idx = var_names_upper.index(gene.upper())
            if hasattr(expr_adata.X, 'toarray'):
                expr = expr_adata.X[:, idx].toarray().flatten()
            else:
                expr = expr_adata.X[:, idx].flatten()

            # Correlation with pseudotime
            from scipy.stats import spearmanr
            r, p = spearmanr(pseudotime, expr)

            results.append({
                "gene": gene,
                "category": "TF" if gene in EMT_REGULATORS["transcription_factors"] else "signaling",
                "mean_expr": expr.mean(),
                "spearman_r": r,
                "p_value": p,
                "direction": "up" if r > 0 else "down",
            })
            available.append((gene, expr))

    df = pd.DataFrame(results)
    if len(df) > 0:
        from statsmodels.stats.multitest import multipletests
        _, df["p_value_adj"], _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["significant"] = df["p_value_adj"] < 0.05
        df = df.sort_values("spearman_r", ascending=False)

    print(f"\n=== EMT Regulator Dynamics ===")
    print(f"  Found {len(available)}/{len(all_regulators)} regulators")

    if len(df) > 0:
        up = df[(df["direction"] == "up") & df["significant"]]
        down = df[(df["direction"] == "down") & df["significant"]]
        print(f"  Upregulated along trajectory: {len(up)}")
        print(f"  Downregulated along trajectory: {len(down)}")

        if len(up) > 0:
            print(f"\n  Top upregulated:")
            for _, row in up.head(5).iterrows():
                print(f"    {row['gene']} ({row['category']}): r={row['spearman_r']:.3f}")

        if len(down) > 0:
            print(f"\n  Top downregulated:")
            for _, row in down.head(5).iterrows():
                print(f"    {row['gene']} ({row['category']}): r={row['spearman_r']:.3f}")

    # Plot
    if output_dir and len(available) > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_dir / "emt_regulator_dynamics.csv", index=False)

        # Plot top regulators along pseudotime
        n_plot = min(6, len(available))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        # Sort by absolute correlation
        sorted_available = sorted(available, key=lambda x: abs(
            spearmanr(pseudotime, x[1])[0]
        ), reverse=True)

        for i, (gene, expr) in enumerate(sorted_available[:n_plot]):
            ax = axes[i]

            # Bin and plot
            n_bins = 30
            bins = np.linspace(pseudotime.min(), pseudotime.max(), n_bins + 1)
            bin_idx = np.digitize(pseudotime, bins) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            bin_centers = [(bins[j] + bins[j+1])/2 for j in range(n_bins)]
            bin_means = [expr[bin_idx == j].mean() for j in range(n_bins)]
            bin_stds = [expr[bin_idx == j].std() for j in range(n_bins)]

            ax.fill_between(bin_centers,
                           np.array(bin_means) - np.array(bin_stds),
                           np.array(bin_means) + np.array(bin_stds),
                           alpha=0.3)
            ax.plot(bin_centers, bin_means, linewidth=2)
            ax.set_xlabel("Pseudotime")
            ax.set_ylabel("Expression")
            ax.set_title(gene)

        # Hide unused axes
        for i in range(n_plot, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / "emt_regulator_trajectories.png", dpi=150, bbox_inches="tight")
        print(f"\n  Saved: {output_dir / 'emt_regulator_trajectories.png'}")

    return df
