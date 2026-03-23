"""
Niche-biology associations for StageBridge.

Links model outputs (niche influence, attention) to biological processes.
This is the core biological interpretation layer.
"""

from typing import Any
import numpy as np
import pandas as pd
import logging

from .pathway_analysis import compute_pathway_activity

log = logging.getLogger(__name__)


def correlate_niche_influence_with_biology(
    influence_df: pd.DataFrame,
    adata: Any,
    pathways: list[str] | None = None,
    cell_id_col: str = "cell_id",
    influence_col: str = "ring_influence",
) -> pd.DataFrame:
    """
    Correlate niche influence scores with biological pathway activity.

    This directly tests: "Do cells with high niche influence show
    distinct biological signatures?"

    Parameters
    ----------
    influence_df : DataFrame
        Niche influence scores from model (must have cell_id column)
    adata : AnnData
        Gene expression data (indexed by cell_id)
    pathways : list, optional
        Pathways to test
    cell_id_col : str
        Column in influence_df with cell IDs
    influence_col : str
        Column with influence scores

    Returns
    -------
    DataFrame
        Correlation results per pathway
    """
    from scipy import stats

    # Compute pathway activity
    activity = compute_pathway_activity(adata, pathways)

    # Align cells
    common_cells = list(set(influence_df[cell_id_col]) & set(activity.index))
    if len(common_cells) == 0:
        raise ValueError("No common cells between influence_df and adata")

    log.info(f"Analyzing {len(common_cells)} cells with both influence and expression")

    # Get aligned data
    influence = influence_df.set_index(cell_id_col).loc[common_cells, influence_col].values
    activity_aligned = activity.loc[common_cells]

    results = []
    for pathway in activity_aligned.columns:
        pathway_scores = activity_aligned[pathway].values

        # Pearson correlation
        r, pval = stats.pearsonr(influence, pathway_scores)

        # Spearman correlation (more robust)
        rho, rho_pval = stats.spearmanr(influence, pathway_scores)

        results.append(
            {
                "pathway": pathway,
                "pearson_r": r,
                "pearson_pval": pval,
                "spearman_rho": rho,
                "spearman_pval": rho_pval,
                "direction": "positive" if r > 0 else "negative",
                "interpretation": _interpret_correlation(pathway, r),
            }
        )

    df = pd.DataFrame(results)

    # Multiple testing correction
    df = df.sort_values("spearman_pval")
    n = len(df)
    df["rank"] = range(1, n + 1)
    df["qvalue"] = df["spearman_pval"] * n / df["rank"]
    df["qvalue"] = df["qvalue"].clip(upper=1.0)
    df = df.drop("rank", axis=1)

    return df.sort_values("spearman_pval")


def _interpret_correlation(pathway: str, r: float) -> str:
    """Generate biological interpretation of pathway-influence correlation."""
    if abs(r) < 0.1:
        return "No significant association"

    direction = "positively" if r > 0 else "negatively"
    strength = "strongly" if abs(r) > 0.3 else "moderately"

    # Pathway-specific interpretations
    interpretations = {
        "emt_hallmark": f"EMT {direction} associated with niche influence - suggests mesenchymal transition driven by microenvironment",
        "caf_general": f"CAF signature {direction} associated - fibroblast remodeling {'increases' if r > 0 else 'decreases'} with niche effects",
        "macrophage_m2": f"M2 macrophages {direction} associated - tumor-promoting immune microenvironment",
        "t_cell_exhaustion": f"T cell exhaustion {direction} correlated - immune escape mechanism",
        "il1b_macrophage": f"IL1B+ macrophages {direction} associated - inflammatory niche signaling",
        "proliferation": f"Proliferation {direction} correlated - growth influenced by niche",
        "nfkb_pathway": f"NF-kB {direction} associated - inflammatory signaling axis",
    }

    if pathway in interpretations:
        return f"{strength.capitalize()} ({r:.2f}): {interpretations[pathway]}"
    else:
        return f"{strength.capitalize()} {direction} association (r={r:.2f})"


def identify_biological_drivers(
    attention_weights: np.ndarray,
    sender_types: np.ndarray,
    adata: Any,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Identify which sender cell types drive biological processes.

    Parameters
    ----------
    attention_weights : ndarray
        Attention from receivers to senders (n_receivers, n_senders)
    sender_types : ndarray
        Cell type labels for senders
    adata : AnnData
        Expression data for senders
    top_n : int
        Number of top associations to return

    Returns
    -------
    DataFrame
        Sender type -> pathway associations
    """
    from scipy import stats

    # Get unique sender types
    unique_types = np.unique(sender_types)
    unique_types = [t for t in unique_types if t and t != "Unknown"]

    # Compute pathway activity for senders
    activity = compute_pathway_activity(adata)

    results = []
    for sender_type in unique_types:
        # Get attention to this sender type
        type_mask = sender_types == sender_type
        if type_mask.sum() == 0:
            continue

        # Mean attention to this type
        attention_to_type = attention_weights[:, type_mask].mean(axis=1)

        # Correlate with pathway activity of receivers
        for pathway in activity.columns:
            if len(attention_to_type) != len(activity):
                continue

            r, pval = stats.pearsonr(attention_to_type, activity[pathway].values)

            if pval < 0.05:  # Only significant
                results.append(
                    {
                        "sender_type": sender_type,
                        "pathway": pathway,
                        "correlation": r,
                        "pvalue": pval,
                        "biological_meaning": _interpret_sender_pathway(sender_type, pathway, r),
                    }
                )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("pvalue")
    return df.head(top_n * len(unique_types))


def _interpret_sender_pathway(sender_type: str, pathway: str, r: float) -> str:
    """Generate interpretation for sender-pathway associations."""
    direction = "promotes" if r > 0 else "suppresses"

    # Known biology
    known_associations = {
        (
            "Fibroblast",
            "emt_hallmark",
        ): f"CAFs {direction} EMT in receivers - classic stromal-epithelial crosstalk",
        (
            "Macrophage",
            "il1b_macrophage",
        ): f"Macrophages {direction} inflammatory signaling - IL1B autocrine loop",
        (
            "T_cell",
            "t_cell_exhaustion",
        ): f"T cells {direction} exhaustion phenotype - potential paracrine signaling",
        (
            "Endothelial",
            "wnt_pathway",
        ): f"Endothelial cells {direction} WNT - vascular niche signaling",
    }

    key = (sender_type, pathway)
    if key in known_associations:
        return known_associations[key]

    return f"Attention to {sender_type} {direction} {pathway} activity"


def compute_niche_pathway_associations(
    influence_df: pd.DataFrame,
    adata: Any,
    stage_col: str = "stage",
) -> dict[str, pd.DataFrame]:
    """
    Compute niche-pathway associations stratified by stage.

    Parameters
    ----------
    influence_df : DataFrame
        Niche influence scores
    adata : AnnData
        Expression data
    stage_col : str
        Stage column

    Returns
    -------
    dict
        Stage -> association DataFrame
    """
    stages = adata.obs[stage_col].unique()
    results = {}

    for stage in stages:
        stage_mask = adata.obs[stage_col] == stage
        stage_adata = adata[stage_mask].copy()

        stage_influence = influence_df[influence_df["cell_id"].isin(stage_adata.obs_names)]

        if len(stage_influence) < 50:
            log.warning(f"Skipping {stage}: too few cells ({len(stage_influence)})")
            continue

        try:
            results[stage] = correlate_niche_influence_with_biology(stage_influence, stage_adata)
        except Exception as e:
            log.warning(f"Failed for stage {stage}: {e}")

    return results


def generate_biological_hypotheses(
    niche_biology_df: pd.DataFrame,
    min_correlation: float = 0.2,
    max_pvalue: float = 0.01,
) -> list[dict[str, Any]]:
    """
    Generate testable biological hypotheses from niche-biology associations.

    Returns structured hypotheses that are clearly labeled as model-generated
    predictions requiring experimental validation.

    Parameters
    ----------
    niche_biology_df : DataFrame
        Output from correlate_niche_influence_with_biology
    min_correlation : float
        Minimum correlation magnitude
    max_pvalue : float
        Maximum p-value for significance

    Returns
    -------
    list of dict
        Structured hypotheses with evidence and validation suggestions
    """
    hypotheses = []

    # Filter significant associations
    sig_df = niche_biology_df[
        (niche_biology_df["spearman_pval"] < max_pvalue)
        & (niche_biology_df["spearman_rho"].abs() >= min_correlation)
    ]

    for _, row in sig_df.iterrows():
        pathway = row["pathway"]
        rho = row["spearman_rho"]
        direction = "increased" if rho > 0 else "decreased"

        # Generate hypothesis
        hypothesis = {
            "id": f"H_{pathway[:10]}_{direction[0]}",
            "statement": _generate_hypothesis_statement(pathway, direction, rho),
            "pathway": pathway,
            "correlation": rho,
            "pvalue": row["spearman_pval"],
            "confidence": _assess_hypothesis_confidence(row),
            "validation_approaches": _suggest_validation(pathway),
            "caveats": [
                "Model-generated hypothesis - requires experimental validation",
                "Association does not imply causation",
                "May be confounded by cell type composition",
            ],
            "status": "HYPOTHESIS - NOT VALIDATED",
        }

        hypotheses.append(hypothesis)

    return hypotheses


def _generate_hypothesis_statement(pathway: str, direction: str, rho: float) -> str:
    """Generate natural language hypothesis statement."""
    pathway_names = {
        "emt_hallmark": "epithelial-mesenchymal transition",
        "caf_general": "cancer-associated fibroblast activity",
        "t_cell_exhaustion": "T cell exhaustion",
        "macrophage_m2": "M2 macrophage polarization",
        "il1b_macrophage": "IL1B+ inflammatory macrophage presence",
        "proliferation": "cell proliferation",
        "nfkb_pathway": "NF-kB pathway activation",
    }

    pathway_name = pathway_names.get(pathway, pathway.replace("_", " "))

    return (
        f"Cells with higher niche influence show {direction} {pathway_name} "
        f"(rho={rho:.2f}), suggesting that local microenvironment context "
        f"{'promotes' if direction == 'increased' else 'suppresses'} this program."
    )


def _assess_hypothesis_confidence(row: pd.Series) -> str:
    """Assess confidence level of hypothesis."""
    rho = abs(row["spearman_rho"])
    pval = row["spearman_pval"]

    if rho > 0.4 and pval < 0.001:
        return "HIGH - Strong effect, highly significant"
    elif rho > 0.2 and pval < 0.01:
        return "MEDIUM - Moderate effect, significant"
    else:
        return "LOW - Weak effect or borderline significance"


def _suggest_validation(pathway: str) -> list[str]:
    """Suggest validation approaches for a pathway hypothesis."""
    common = [
        "Validate with independent cohort",
        "Spatial transcriptomics co-localization analysis",
    ]

    pathway_specific = {
        "emt_hallmark": [
            "IHC for vimentin/E-cadherin ratio",
            "Single-cell trajectory analysis of EMT states",
        ],
        "caf_general": [
            "Co-culture assays with isolated CAFs",
            "Spatial proteomics for CAF markers",
        ],
        "t_cell_exhaustion": [
            "Flow cytometry for PD-1/TIM-3/LAG-3",
            "TCR sequencing for clonal exhaustion",
        ],
        "macrophage_m2": [
            "CD163/CD206 IHC staining",
            "Cytokine profiling of tumor lysates",
        ],
    }

    return common + pathway_specific.get(pathway, ["Pathway-specific assays"])
