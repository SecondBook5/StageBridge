"""
Pathway activity analysis for StageBridge.

Computes pathway-level activity scores and identifies stage-specific pathways.
"""

from typing import Any
import pandas as pd
import logging

from .signatures import GENE_SIGNATURES, score_all_signatures

log = logging.getLogger(__name__)


def compute_pathway_activity(
    adata: Any,
    pathways: list[str] | None = None,
    layer: str | None = None,
    method: str = "mean_zscore",
) -> pd.DataFrame:
    """
    Compute pathway activity scores for all cells.

    Parameters
    ----------
    adata : AnnData
        Gene expression data
    pathways : list, optional
        List of pathway names (default: all in GENE_SIGNATURES)
    layer : str, optional
        Layer to use for expression
    method : str
        Scoring method: "mean_zscore" (default) or "gsva" (if installed)

    Returns
    -------
    DataFrame
        Pathway activity scores (n_cells, n_pathways)
    """
    if pathways is None:
        pathways = list(GENE_SIGNATURES.keys())

    # Filter to available pathways
    available = [p for p in pathways if p in GENE_SIGNATURES]
    if len(available) < len(pathways):
        missing = set(pathways) - set(available)
        log.warning(f"Pathways not found: {missing}")

    signatures = {k: GENE_SIGNATURES[k] for k in available}
    return score_all_signatures(adata, signatures, layer=layer, add_to_obs=False)


def run_enrichment_analysis(
    gene_list: list[str],
    background: list[str] | None = None,
    gene_sets: dict[str, list[str]] | None = None,
    method: str = "hypergeometric",
) -> pd.DataFrame:
    """
    Run gene set enrichment analysis on a gene list.

    Parameters
    ----------
    gene_list : list
        Genes of interest (e.g., top attention genes)
    background : list, optional
        Background gene universe (default: all genes in signatures)
    gene_sets : dict, optional
        Gene sets to test (default: GENE_SIGNATURES)
    method : str
        Method: "hypergeometric" or "gsea"

    Returns
    -------
    DataFrame
        Enrichment results with p-values
    """
    from scipy import stats

    if gene_sets is None:
        gene_sets = {k: v["genes"] for k, v in GENE_SIGNATURES.items()}

    if background is None:
        # Use all genes from signatures
        background = list(set(g for gs in gene_sets.values() for g in gs))

    gene_set = set(gene_list)
    background_set = set(background)
    n_background = len(background_set)
    n_query = len(gene_set & background_set)

    results = []
    for name, pathway_genes in gene_sets.items():
        pathway_set = set(pathway_genes) & background_set
        n_pathway = len(pathway_set)
        n_overlap = len(gene_set & pathway_set)

        # Hypergeometric test
        # P(X >= n_overlap) where X ~ Hypergeometric(N, K, n)
        # N = background, K = pathway genes, n = query genes
        pval = stats.hypergeom.sf(n_overlap - 1, n_background, n_pathway, n_query)

        # Fold enrichment
        expected = n_query * n_pathway / n_background if n_background > 0 else 0
        fold_enrichment = n_overlap / expected if expected > 0 else 0

        results.append(
            {
                "pathway": name,
                "overlap": n_overlap,
                "pathway_size": n_pathway,
                "query_size": n_query,
                "fold_enrichment": fold_enrichment,
                "pvalue": pval,
                "overlap_genes": list(gene_set & pathway_set),
            }
        )

    df = pd.DataFrame(results)

    # Multiple testing correction (Benjamini-Hochberg)
    df = df.sort_values("pvalue")
    n_tests = len(df)
    df["rank"] = range(1, n_tests + 1)
    df["qvalue"] = df["pvalue"] * n_tests / df["rank"]
    df["qvalue"] = df["qvalue"].clip(upper=1.0)
    df = df.drop("rank", axis=1)

    return df.sort_values("pvalue")


def compare_pathway_activity_by_stage(
    adata: Any,
    stage_col: str = "stage",
    pathways: list[str] | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """
    Compare pathway activity across progression stages.

    Parameters
    ----------
    adata : AnnData
        Gene expression data with stage annotation
    stage_col : str
        Column name for stage labels
    pathways : list, optional
        Pathways to analyze

    Returns
    -------
    DataFrame
        Mean activity per stage with statistics
    """
    from scipy import stats

    activity = compute_pathway_activity(adata, pathways, layer)
    activity["stage"] = adata.obs[stage_col].values

    # Stage order for LUAD progression
    stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    available_stages = [s for s in stage_order if s in activity["stage"].unique()]

    results = []
    for pathway in activity.columns:
        if pathway == "stage":
            continue

        # Mean per stage
        stage_means = activity.groupby("stage")[pathway].mean()

        # Kruskal-Wallis test for difference across stages
        groups = [activity[activity["stage"] == s][pathway].values for s in available_stages]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            stat, pval = stats.kruskal(*groups)
        else:
            stat, pval = 0, 1.0

        # Spearman correlation with stage order (monotonic trend)
        stage_numeric = activity["stage"].map({s: i for i, s in enumerate(stage_order)})
        valid = ~stage_numeric.isna()
        if valid.sum() > 10:
            rho, rho_pval = stats.spearmanr(stage_numeric[valid], activity.loc[valid, pathway])
        else:
            rho, rho_pval = 0, 1.0

        result = {
            "pathway": pathway,
            "kruskal_stat": stat,
            "kruskal_pval": pval,
            "spearman_rho": rho,
            "spearman_pval": rho_pval,
            "trend": "increasing" if rho > 0.1 else ("decreasing" if rho < -0.1 else "flat"),
        }

        for stage in available_stages:
            if stage in stage_means.index:
                result[f"mean_{stage}"] = stage_means[stage]

        results.append(result)

    return pd.DataFrame(results).sort_values("kruskal_pval")


def identify_stage_specific_pathways(
    adata: Any,
    stage_col: str = "stage",
    pathways: list[str] | None = None,
    pval_threshold: float = 0.05,
    fold_change_threshold: float = 1.5,
) -> dict[str, pd.DataFrame]:
    """
    Identify pathways that are specifically enriched in each stage.

    Parameters
    ----------
    adata : AnnData
        Gene expression data
    stage_col : str
        Stage column
    pathways : list, optional
        Pathways to analyze
    pval_threshold : float
        P-value threshold for significance
    fold_change_threshold : float
        Minimum fold change vs other stages

    Returns
    -------
    dict
        Stage -> DataFrame of enriched pathways
    """
    from scipy import stats

    activity = compute_pathway_activity(adata, pathways)
    activity["stage"] = adata.obs[stage_col].values

    stages = activity["stage"].unique()
    stage_specific = {}

    for stage in stages:
        results = []
        stage_mask = activity["stage"] == stage
        other_mask = ~stage_mask

        for pathway in activity.columns:
            if pathway == "stage":
                continue

            stage_vals = activity.loc[stage_mask, pathway]
            other_vals = activity.loc[other_mask, pathway]

            # Mann-Whitney U test
            stat, pval = stats.mannwhitneyu(stage_vals, other_vals, alternative="greater")

            # Fold change
            stage_mean = stage_vals.mean()
            other_mean = other_vals.mean()
            fc = stage_mean / (other_mean + 1e-10)

            if pval < pval_threshold and fc >= fold_change_threshold:
                results.append(
                    {
                        "pathway": pathway,
                        "stage_mean": stage_mean,
                        "other_mean": other_mean,
                        "fold_change": fc,
                        "pvalue": pval,
                    }
                )

        if results:
            stage_specific[stage] = pd.DataFrame(results).sort_values("pvalue")

    return stage_specific
