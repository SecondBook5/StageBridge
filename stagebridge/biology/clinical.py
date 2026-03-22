"""
Clinical relevance tools for StageBridge.

Connects model outputs to clinical outcomes and provides
risk stratification based on niche phenotypes.
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import logging

from .signatures import score_all_signatures

log = logging.getLogger(__name__)


def compute_risk_scores(
    adata: Any,
    influence_df: pd.DataFrame | None = None,
    method: str = "composite",
) -> pd.DataFrame:
    """
    Compute clinical risk scores based on biological signatures.

    Risk is assessed using:
    - EMT score (high = worse prognosis)
    - CAF infiltration (high = worse)
    - M2 macrophage polarization (high = worse)
    - T cell exhaustion (high = worse)
    - Proliferation (high = worse)
    - Niche influence (context-dependent)

    Parameters
    ----------
    adata : AnnData
        Expression data with or without pre-computed signatures
    influence_df : DataFrame, optional
        Niche influence scores
    method : str
        Scoring method: "composite" (weighted sum) or "individual"

    Returns
    -------
    DataFrame
        Risk scores per cell
    """
    # Ensure signatures are computed
    sig_cols = [c for c in adata.obs.columns if c.startswith("sig_")]
    if not sig_cols:
        log.info("Computing signature scores...")
        score_all_signatures(adata)
        sig_cols = [c for c in adata.obs.columns if c.startswith("sig_")]

    # Risk-associated signatures and weights
    # Positive weight = higher score = higher risk
    risk_weights = {
        "sig_emt_hallmark": 0.20,  # EMT associated with worse outcomes
        "sig_caf_general": 0.15,   # CAF infiltration
        "sig_macrophage_m2": 0.15, # Tumor-promoting macrophages
        "sig_t_cell_exhaustion": 0.15,  # Immune escape
        "sig_proliferation": 0.10,  # High proliferation
        "sig_il1b_macrophage": 0.10,  # Inflammatory niche
        "sig_nfkb_pathway": 0.05,   # NF-kB activation
    }

    # Protective signatures (negative weight)
    protective_weights = {
        "sig_t_cell_cytotoxic": -0.10,  # Active anti-tumor immunity
        "sig_at2_markers": -0.05,       # AT2 identity retention
    }

    all_weights = {**risk_weights, **protective_weights}

    # Filter to available signatures
    available_weights = {k: v for k, v in all_weights.items() if k in sig_cols}

    if method == "composite":
        # Compute weighted composite score
        scores = np.zeros(adata.n_obs)
        total_weight = 0

        for sig, weight in available_weights.items():
            # Z-score normalize
            sig_values = adata.obs[sig].values
            sig_z = (sig_values - sig_values.mean()) / (sig_values.std() + 1e-10)
            scores += weight * sig_z
            total_weight += abs(weight)

        # Normalize to 0-1 range
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        risk_df = pd.DataFrame({
            "cell_id": adata.obs_names,
            "risk_score": scores,
            "risk_category": pd.cut(scores, bins=[0, 0.33, 0.67, 1.0],
                                    labels=["low", "intermediate", "high"]),
        })

    else:  # individual
        risk_df = pd.DataFrame({"cell_id": adata.obs_names})

        for sig in available_weights:
            sig_values = adata.obs[sig].values
            sig_z = (sig_values - sig_values.mean()) / (sig_values.std() + 1e-10)
            risk_df[sig.replace("sig_", "risk_")] = sig_z

    # Add niche influence if available
    if influence_df is not None and "ring_influence" in influence_df.columns:
        cell_influence = influence_df.set_index("cell_id")["ring_influence"]
        risk_df["niche_influence"] = risk_df["cell_id"].map(cell_influence)

    return risk_df


def stratify_by_niche_phenotype(
    adata: Any,
    influence_df: pd.DataFrame,
    n_phenotypes: int = 4,
    method: str = "kmeans",
) -> pd.DataFrame:
    """
    Stratify cells into niche phenotypes based on biological signatures.

    Identifies distinct niche-biology patterns that may have clinical relevance.

    Parameters
    ----------
    adata : AnnData
        Expression data with signatures
    influence_df : DataFrame
        Niche influence scores
    n_phenotypes : int
        Number of phenotypes to identify
    method : str
        Clustering method: "kmeans" or "hierarchical"

    Returns
    -------
    DataFrame
        Phenotype assignments with characterization
    """
    from sklearn.preprocessing import StandardScaler

    # Get key signatures for phenotyping
    phenotype_sigs = [
        "sig_emt_hallmark",
        "sig_caf_general",
        "sig_macrophage_m2",
        "sig_t_cell_exhaustion",
        "sig_proliferation",
    ]

    available_sigs = [s for s in phenotype_sigs if s in adata.obs.columns]
    if len(available_sigs) < 3:
        log.warning(f"Only {len(available_sigs)} signatures available for phenotyping")

    # Prepare feature matrix
    X = adata.obs[available_sigs].values
    X = StandardScaler().fit_transform(X)

    # Add niche influence
    if influence_df is not None:
        cell_influence = influence_df.set_index("cell_id")["ring_influence"]
        influence_values = adata.obs_names.map(cell_influence).values.reshape(-1, 1)
        influence_values = np.nan_to_num(influence_values, nan=0)
        influence_values = StandardScaler().fit_transform(influence_values)
        X = np.hstack([X, influence_values])

    # Cluster
    if method == "kmeans":
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=n_phenotypes, random_state=42, n_init=10)
    else:
        from sklearn.cluster import AgglomerativeClustering
        clusterer = AgglomerativeClustering(n_clusters=n_phenotypes)

    labels = clusterer.fit_predict(X)

    # Create result DataFrame
    result = pd.DataFrame({
        "cell_id": adata.obs_names,
        "phenotype": labels,
    })

    # Characterize each phenotype
    characterizations = []
    for pheno in range(n_phenotypes):
        mask = labels == pheno
        char = {"phenotype": pheno, "n_cells": mask.sum()}

        for sig in available_sigs:
            sig_name = sig.replace("sig_", "")
            char[f"mean_{sig_name}"] = adata.obs.loc[mask, sig].mean()

        characterizations.append(char)

    char_df = pd.DataFrame(characterizations)

    # Name phenotypes based on dominant features
    phenotype_names = []
    for _, row in char_df.iterrows():
        # Find highest and lowest signatures
        sig_cols = [c for c in char_df.columns if c.startswith("mean_")]
        sig_values = row[sig_cols]
        top_sig = sig_values.idxmax().replace("mean_", "")
        phenotype_names.append(f"Phenotype_{int(row['phenotype'])}_high_{top_sig[:8]}")

    char_df["phenotype_name"] = phenotype_names

    # Add names to result
    name_map = dict(zip(char_df["phenotype"], char_df["phenotype_name"]))
    result["phenotype_name"] = result["phenotype"].map(name_map)

    log.info(f"Identified {n_phenotypes} niche phenotypes")
    log.info(f"Phenotype sizes: {result['phenotype'].value_counts().to_dict()}")

    return result, char_df


def generate_clinical_summary(
    adata: Any,
    risk_df: pd.DataFrame,
    phenotype_df: pd.DataFrame | None = None,
    stage_col: str = "stage",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Generate comprehensive clinical summary report.

    Parameters
    ----------
    adata : AnnData
        Expression data
    risk_df : DataFrame
        Risk scores from compute_risk_scores
    phenotype_df : DataFrame, optional
        Phenotype assignments
    stage_col : str
        Stage column
    output_dir : Path, optional
        Output directory for report

    Returns
    -------
    dict
        Summary statistics and findings
    """
    summary = {
        "n_cells": len(risk_df),
        "risk_distribution": {},
        "stage_risk": {},
        "key_findings": [],
    }

    # Risk distribution
    if "risk_category" in risk_df.columns:
        summary["risk_distribution"] = risk_df["risk_category"].value_counts().to_dict()

    # Risk by stage
    if stage_col in adata.obs.columns and "risk_score" in risk_df.columns:
        risk_df_merged = risk_df.copy()
        risk_df_merged["stage"] = adata.obs[stage_col].values

        stage_risk = risk_df_merged.groupby("stage")["risk_score"].agg(["mean", "std", "count"])
        summary["stage_risk"] = stage_risk.to_dict()

        # Find high-risk stages
        high_risk_stages = stage_risk[stage_risk["mean"] > stage_risk["mean"].median()].index.tolist()
        if high_risk_stages:
            summary["key_findings"].append(
                f"High-risk stages: {', '.join(high_risk_stages)}"
            )

    # Phenotype analysis
    if phenotype_df is not None and len(phenotype_df) == 2:
        cell_pheno, char_df = phenotype_df
        summary["phenotypes"] = char_df.to_dict("records")

        # Find highest risk phenotype
        if "risk_score" in risk_df.columns:
            merged = risk_df.merge(cell_pheno, on="cell_id")
            pheno_risk = merged.groupby("phenotype_name")["risk_score"].mean()
            highest_risk = pheno_risk.idxmax()
            summary["key_findings"].append(
                f"Highest risk phenotype: {highest_risk} (mean={pheno_risk.max():.3f})"
            )

    # Clinical interpretation
    interpretations = []

    if "risk_distribution" in summary and summary["risk_distribution"]:
        high_pct = summary["risk_distribution"].get("high", 0) / summary["n_cells"] * 100
        if high_pct > 30:
            interpretations.append(
                f"WARNING: {high_pct:.1f}% of cells classified as high-risk"
            )
        elif high_pct < 10:
            interpretations.append(
                f"Favorable: Only {high_pct:.1f}% of cells classified as high-risk"
            )

    summary["clinical_interpretation"] = interpretations

    # Save report if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        import json
        with open(output_dir / "clinical_summary.json", "w") as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            json.dump(summary, f, indent=2, default=convert)

        # Save risk scores
        risk_df.to_csv(output_dir / "cell_risk_scores.csv", index=False)

        log.info(f"Saved clinical summary to {output_dir}")

    return summary
