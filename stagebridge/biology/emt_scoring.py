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
