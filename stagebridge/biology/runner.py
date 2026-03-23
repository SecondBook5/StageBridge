"""
Biology analysis runner for StageBridge.

Provides a simple interface to run comprehensive biological interpretation
after model training.

Usage:
    from stagebridge.biology.runner import run_biological_analysis

    results = run_biological_analysis(
        adata=adata,
        influence_df=influence_df,  # From model
        output_dir=Path("results/biology"),
    )
"""

from pathlib import Path
from typing import Any
import logging
import json

import pandas as pd

from .signatures import score_all_signatures, GENE_SIGNATURES
from .pathway_analysis import (
    compare_pathway_activity_by_stage,
    identify_stage_specific_pathways,
)
from .niche_biology import (
    correlate_niche_influence_with_biology,
    compute_niche_pathway_associations,
    generate_biological_hypotheses,
)
from .clinical import (
    compute_risk_scores,
    stratify_by_niche_phenotype,
    generate_clinical_summary,
)
from .plots import (
    plot_signature_scores_by_stage,
    plot_niche_biology_heatmap,
    plot_pathway_activity_ridge,
    plot_emt_caf_immune_triangle,
    plot_biological_summary_panel,
)

log = logging.getLogger(__name__)


def run_biological_analysis(
    adata: Any,
    influence_df: pd.DataFrame | None = None,
    output_dir: Path | str = "biology_results",
    stage_col: str = "stage",
    generate_plots: bool = True,
    generate_clinical: bool = True,
) -> dict[str, Any]:
    """
    Run comprehensive biological analysis pipeline.

    This function:
    1. Scores all gene signatures
    2. Analyzes pathway activity by stage
    3. Correlates niche influence with biology
    4. Generates biological hypotheses
    5. Computes clinical risk scores
    6. Creates publication-quality plots

    Parameters
    ----------
    adata : AnnData
        Gene expression data with stage annotations
    influence_df : DataFrame, optional
        Niche influence scores from trained model.
        Must have columns: cell_id, ring_influence, stage
    output_dir : Path or str
        Directory for outputs
    stage_col : str
        Column name for stage labels
    generate_plots : bool
        Whether to generate plots
    generate_clinical : bool
        Whether to compute clinical scores

    Returns
    -------
    dict
        Comprehensive results including:
        - signature_scores: DataFrame of all signature scores
        - pathway_comparison: Stage-wise pathway comparison
        - niche_biology: Niche-biology correlations
        - hypotheses: Generated biological hypotheses
        - clinical: Risk scores and phenotypes (if generate_clinical)
        - output_files: List of generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "output_dir": str(output_dir),
        "output_files": [],
    }

    log.info("=" * 60)
    log.info("StageBridge Biological Analysis Pipeline")
    log.info("=" * 60)

    # Step 1: Score all signatures
    log.info("\n[1/6] Computing gene signature scores...")
    signature_scores = score_all_signatures(adata, add_to_obs=True)
    signature_scores.to_parquet(output_dir / "signature_scores.parquet")
    results["signature_scores"] = signature_scores
    results["output_files"].append("signature_scores.parquet")
    log.info(
        f"  Scored {len(signature_scores.columns)} signatures for {len(signature_scores)} cells"
    )

    # Step 2: Pathway activity by stage
    log.info("\n[2/6] Analyzing pathway activity by stage...")
    pathway_comparison = compare_pathway_activity_by_stage(adata, stage_col=stage_col)
    pathway_comparison.to_csv(output_dir / "pathway_stage_comparison.csv", index=False)
    results["pathway_comparison"] = pathway_comparison
    results["output_files"].append("pathway_stage_comparison.csv")

    # Find significant trends
    sig_trends = pathway_comparison[pathway_comparison["kruskal_pval"] < 0.05]
    log.info(f"  Found {len(sig_trends)} pathways with significant stage differences")

    # Step 3: Stage-specific pathways
    log.info("\n[3/6] Identifying stage-specific pathways...")
    stage_specific = identify_stage_specific_pathways(adata, stage_col=stage_col)
    for stage, df in stage_specific.items():
        df.to_csv(output_dir / f"stage_specific_{stage}.csv", index=False)
        results["output_files"].append(f"stage_specific_{stage}.csv")
    results["stage_specific_pathways"] = stage_specific
    log.info(f"  Identified stage-specific pathways for {len(stage_specific)} stages")

    # Step 4: Niche-biology correlations
    if influence_df is not None:
        log.info("\n[4/6] Correlating niche influence with biology...")
        niche_biology = correlate_niche_influence_with_biology(
            influence_df, adata, cell_id_col="cell_id", influence_col="ring_influence"
        )
        niche_biology.to_csv(output_dir / "niche_biology_correlations.csv", index=False)
        results["niche_biology"] = niche_biology
        results["output_files"].append("niche_biology_correlations.csv")

        # Top associations
        top_pos = niche_biology[niche_biology["spearman_rho"] > 0].head(3)
        top_neg = niche_biology[niche_biology["spearman_rho"] < 0].head(3)
        log.info("  Top positive associations:")
        for _, row in top_pos.iterrows():
            log.info(f"    {row['pathway']}: rho={row['spearman_rho']:.3f}")
        log.info("  Top negative associations:")
        for _, row in top_neg.iterrows():
            log.info(f"    {row['pathway']}: rho={row['spearman_rho']:.3f}")

        # Stage-stratified associations
        stage_associations = compute_niche_pathway_associations(
            influence_df, adata, stage_col=stage_col
        )
        for stage, df in stage_associations.items():
            df.to_csv(output_dir / f"niche_biology_{stage}.csv", index=False)
            results["output_files"].append(f"niche_biology_{stage}.csv")
        results["stage_niche_biology"] = stage_associations
    else:
        log.info("\n[4/6] Skipping niche-biology (no influence data)")
        results["niche_biology"] = None

    # Step 5: Generate hypotheses
    if influence_df is not None and "niche_biology" in results:
        log.info("\n[5/6] Generating biological hypotheses...")
        hypotheses = generate_biological_hypotheses(results["niche_biology"])
        results["hypotheses"] = hypotheses

        # Save hypotheses
        with open(output_dir / "biological_hypotheses.json", "w") as f:
            json.dump(hypotheses, f, indent=2, default=str)
        results["output_files"].append("biological_hypotheses.json")

        log.info(f"  Generated {len(hypotheses)} testable hypotheses")
        for h in hypotheses[:3]:
            log.info(f"    [{h['confidence']}] {h['statement'][:80]}...")
    else:
        log.info("\n[5/6] Skipping hypothesis generation (no niche data)")
        results["hypotheses"] = []

    # Step 6: Clinical analysis
    if generate_clinical:
        log.info("\n[6/6] Computing clinical relevance...")
        risk_df = compute_risk_scores(adata, influence_df)
        risk_df.to_csv(output_dir / "cell_risk_scores.csv", index=False)
        results["output_files"].append("cell_risk_scores.csv")

        phenotype_result = stratify_by_niche_phenotype(adata, influence_df or pd.DataFrame())
        phenotype_df, char_df = phenotype_result
        phenotype_df.to_csv(output_dir / "cell_phenotypes.csv", index=False)
        char_df.to_csv(output_dir / "phenotype_characteristics.csv", index=False)
        results["output_files"].extend(["cell_phenotypes.csv", "phenotype_characteristics.csv"])

        clinical_summary = generate_clinical_summary(
            adata, risk_df, phenotype_result, stage_col=stage_col, output_dir=output_dir
        )
        results["clinical"] = {
            "risk_scores": risk_df,
            "phenotypes": phenotype_df,
            "summary": clinical_summary,
        }
        log.info(f"  Risk distribution: {clinical_summary.get('risk_distribution', {})}")
    else:
        log.info("\n[6/6] Skipping clinical analysis")
        results["clinical"] = None

    # Generate plots
    if generate_plots:
        log.info("\n" + "=" * 60)
        log.info("Generating publication-quality plots...")
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        try:
            # Signature scores by stage
            key_sigs = [
                "emt_hallmark",
                "caf_general",
                "macrophage_m2",
                "t_cell_exhaustion",
                "proliferation",
            ]
            available_sigs = [s for s in key_sigs if f"sig_{s}" in adata.obs.columns]
            if available_sigs:
                plot_signature_scores_by_stage(
                    adata,
                    available_sigs,
                    stage_col=stage_col,
                    save_path=plots_dir / "signature_scores_by_stage.png",
                )
                results["output_files"].append("plots/signature_scores_by_stage.png")

            # Ridge plot
            if len(available_sigs) >= 3:
                plot_pathway_activity_ridge(
                    adata,
                    available_sigs[:5],
                    stage_col=stage_col,
                    save_path=plots_dir / "pathway_ridge.png",
                )
                results["output_files"].append("plots/pathway_ridge.png")

            # EMT/CAF/Immune triangle
            plot_emt_caf_immune_triangle(
                adata, stage_col=stage_col, save_path=plots_dir / "emt_caf_immune_triangle.png"
            )
            results["output_files"].append("plots/emt_caf_immune_triangle.png")

            # Niche-biology heatmap
            if results.get("niche_biology") is not None:
                plot_niche_biology_heatmap(
                    results["niche_biology"], save_path=plots_dir / "niche_biology_heatmap.png"
                )
                results["output_files"].append("plots/niche_biology_heatmap.png")

                # Summary panel
                plot_biological_summary_panel(
                    adata,
                    influence_df,
                    results["niche_biology"],
                    stage_col=stage_col,
                    save_path=plots_dir / "biological_summary_panel.png",
                )
                results["output_files"].append("plots/biological_summary_panel.png")

            log.info(f"  Saved plots to {plots_dir}")

        except Exception as e:
            log.warning(f"Plot generation failed: {e}")

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("BIOLOGICAL ANALYSIS COMPLETE")
    log.info("=" * 60)
    log.info(f"Output directory: {output_dir}")
    log.info(f"Generated {len(results['output_files'])} files")

    # Save manifest
    with open(output_dir / "analysis_manifest.json", "w") as f:
        manifest = {
            "n_cells": adata.n_obs,
            "n_signatures": len(GENE_SIGNATURES),
            "stages": list(adata.obs[stage_col].unique()),
            "has_niche_data": influence_df is not None,
            "n_hypotheses": len(results.get("hypotheses", [])),
            "output_files": results["output_files"],
        }
        json.dump(manifest, f, indent=2)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log.info("Biology runner module loaded.")
    log.info("Usage: from stagebridge.biology.runner import run_biological_analysis")
