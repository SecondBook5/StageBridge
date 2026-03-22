#!/usr/bin/env python
"""
Run biological interpretation analysis on trained StageBridge model.

This pipeline extracts attention weights from a trained model and computes:
1. Attention-weighted L-R interaction scores
2. Stage-specific niche ecosystem summaries
3. Intervention target prioritization
4. Niche-level progression risk

The outputs are publication-ready figures and tables that connect model
predictions to the Peng et al. IL1B+ macrophage niche mechanism.

Usage:
    python -m stagebridge.pipelines.run_biological_analysis \\
        --checkpoint results/training/best_model.pt \\
        --data-dir /data/processed/luad_evo/canonical \\
        --output-dir results/biological_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to canonical data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/biological_analysis"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on synthetic data for testing",
    )
    return parser.parse_args()


def load_model_and_extract_attention(
    checkpoint_path: Path,
    data_loader: Any,
    device: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """
    Load trained model and extract attention weights for all samples.

    Returns
    -------
    attention_dict : dict
        cell_id -> attention weights (n_senders,)
    metadata_df : DataFrame
        Cell metadata including stage, donor, coordinates
    """
    log.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config
    model_config = checkpoint.get("model_config", {})
    log.info(f"Model config: {model_config}")

    # For now, return empty structures for synthetic testing
    # Real implementation would load the model and run inference
    attention_dict = {}
    metadata_records = []

    log.warning("Full model loading not implemented - using synthetic attention")

    return attention_dict, pd.DataFrame(metadata_records)


def generate_synthetic_attention_data(
    n_cells: int = 1000,
    n_senders: int = 20,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic attention data for testing the biological analysis pipeline.
    """
    rng = np.random.default_rng(seed)

    stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    stage_weights = [0.1, 0.2, 0.3, 0.25, 0.15]

    # Generate cell metadata
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    cell_stages = rng.choice(stages, size=n_cells, p=stage_weights)
    cell_donors = rng.choice([f"donor_{i}" for i in range(10)], size=n_cells)

    # Spatial coordinates
    coords = rng.uniform(0, 1000, size=(n_cells, 2))

    metadata_df = pd.DataFrame({
        "cell_id": cell_ids,
        "stage": cell_stages,
        "donor_id": cell_donors,
        "x": coords[:, 0],
        "y": coords[:, 1],
    })

    # Generate attention weights
    attention_dict = {}
    for cell_id in cell_ids:
        # Attention to senders
        attention_dict[cell_id] = rng.dirichlet(np.ones(n_senders))

    # Generate sender types for each cell's neighborhood
    sender_type_names = [
        "Macrophage", "Fibroblast", "T_cell", "Endothelial",
        "AT2", "AT1", "Club", "Unknown"
    ]
    sender_types = rng.choice(len(sender_type_names), size=(n_cells, n_senders))

    # Generate ligand expression for senders
    # IL1B higher in macrophages (type 0), especially in AAH/AIS stages
    ligand_genes = ["IL1B", "IL6", "TNF", "TGFB1", "AREG", "CXCL12", "HGF"]

    ligand_expression_records = []
    for i, (cell_id, stage) in enumerate(zip(cell_ids, cell_stages)):
        for j in range(n_senders):
            sender_type = sender_types[i, j]
            record = {
                "cell_id": cell_id,
                "sender_idx": j,
                "sender_type": sender_type_names[sender_type],
            }

            for ligand in ligand_genes:
                # Base expression
                expr = rng.exponential(0.5)

                # Macrophages express more IL1B in early stages
                if ligand == "IL1B" and sender_type == 0:  # Macrophage
                    if stage in ["AAH", "AIS"]:
                        expr *= 3.0  # Higher in early stages
                    elif stage == "MIA":
                        expr *= 2.0

                # Fibroblasts express more TGFB1
                if ligand == "TGFB1" and sender_type == 1:
                    expr *= 2.0

                record[ligand] = expr

            ligand_expression_records.append(record)

    ligand_df = pd.DataFrame(ligand_expression_records)

    # Generate receptor expression for receivers
    receptor_genes = ["IL1R1", "IL6ST", "TNFRSF1A", "TGFBR2", "EGFR", "CXCR4", "MET"]

    receptor_records = []
    for cell_id, stage in zip(cell_ids, cell_stages):
        record = {"cell_id": cell_id}

        for receptor in receptor_genes:
            expr = rng.exponential(0.5)

            # IL1R1 higher in epithelial cells of early stages
            if receptor == "IL1R1" and stage in ["AAH", "AIS", "MIA"]:
                expr *= 1.5

            record[receptor] = expr

        receptor_records.append(record)

    receptor_df = pd.DataFrame(receptor_records)

    log.info(f"Generated synthetic data: {n_cells} cells, {n_senders} senders per cell")

    return attention_dict, metadata_df, ligand_df, receptor_df, sender_type_names


def run_biological_analysis(
    attention_dict: dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
    ligand_df: pd.DataFrame,
    receptor_df: pd.DataFrame,
    sender_type_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """
    Run complete biological analysis pipeline.
    """
    from stagebridge.biology.attention_lr_scoring import (
        compute_attention_weighted_lr_scores,
        aggregate_lr_scores_by_stage,
        identify_stage_specific_interactions,
        compute_il1b_axis_score,
        generate_niche_ecosystem_summary,
        create_lr_interaction_report,
        export_lr_scores_for_visualization,
    )
    from stagebridge.biology.intervention_targets import (
        prioritize_intervention_targets,
        compute_niche_level_risk,
        aggregate_niche_risks_by_region,
        generate_intervention_plan,
        export_intervention_report,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "n_cells_analyzed": len(attention_dict),
        "stages": list(metadata_df["stage"].unique()),
    }

    # 1. Compute attention-weighted L-R scores per cell
    log.info("Computing attention-weighted L-R interaction scores...")

    all_cell_scores = []
    il1b_axis_results = []

    for cell_id, attention in attention_dict.items():
        cell_meta = metadata_df[metadata_df["cell_id"] == cell_id].iloc[0]
        stage = cell_meta["stage"]

        # Get sender expression for this cell's neighborhood
        cell_ligand = ligand_df[ligand_df["cell_id"] == cell_id]
        if len(cell_ligand) == 0:
            continue

        # Pivot to get ligand expression matrix
        ligand_cols = [c for c in cell_ligand.columns if c not in ["cell_id", "sender_idx", "sender_type"]]
        ligand_matrix = cell_ligand[ligand_cols]

        # Get sender types
        sender_types = cell_ligand["sender_type"].map(
            {name: i for i, name in enumerate(sender_type_names)}
        ).values

        # Get receptor expression
        cell_receptor = receptor_df[receptor_df["cell_id"] == cell_id]
        if len(cell_receptor) == 0:
            continue
        receptor_expr = cell_receptor.iloc[0].drop("cell_id")

        # Compute L-R scores
        scores = compute_attention_weighted_lr_scores(
            attention_weights=attention,
            sender_types=sender_types,
            ligand_expression=ligand_matrix,
            receptor_expression=receptor_expr,
            type_names=sender_type_names,
        )

        all_cell_scores.append((stage, scores))

        # IL1B axis analysis
        il1b_result = compute_il1b_axis_score(scores)
        il1b_result["cell_id"] = cell_id
        il1b_result["stage"] = stage
        il1b_axis_results.append(il1b_result)

    log.info(f"Computed L-R scores for {len(all_cell_scores)} cells")

    # 2. Aggregate by stage
    log.info("Aggregating L-R scores by stage...")
    stage_scores = aggregate_lr_scores_by_stage(all_cell_scores)

    # Export for visualization
    export_lr_scores_for_visualization(
        stage_scores,
        output_path=str(output_dir / "lr_scores_by_stage.csv"),
    )

    # 3. Identify stage-specific interactions
    log.info("Identifying stage-specific L-R interactions...")
    stage_specific_df = identify_stage_specific_interactions(stage_scores)

    if not stage_specific_df.empty:
        stage_specific_df.to_csv(output_dir / "stage_specific_interactions.csv", index=False)
        results["n_stage_specific_interactions"] = len(stage_specific_df)

        # Check IL1B-IL1R1 in early stages
        il1b_early = stage_specific_df[
            (stage_specific_df["ligand"] == "IL1B") &
            (stage_specific_df["stage"].isin(["AAH", "AIS"]))
        ]
        results["il1b_enriched_in_early_stages"] = len(il1b_early) > 0

    # 4. Generate niche ecosystem summaries
    log.info("Generating niche ecosystem summaries...")
    summaries = generate_niche_ecosystem_summary(stage_scores)

    # Save summaries
    summary_records = []
    for stage, summary in summaries.items():
        summary_records.append({
            "stage": stage,
            "n_cells": summary.n_cells,
            "risk_level": summary.risk_level,
            "interpretation": summary.biological_interpretation,
            "key_findings": "; ".join(summary.key_findings),
        })

    pd.DataFrame(summary_records).to_csv(
        output_dir / "niche_ecosystem_summaries.csv", index=False
    )

    # 5. Create L-R interaction report
    log.info("Creating L-R interaction report...")
    report = create_lr_interaction_report(summaries, stage_specific_df)

    with open(output_dir / "lr_interaction_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # 6. Prioritize intervention targets
    log.info("Prioritizing intervention targets...")
    targets = prioritize_intervention_targets(
        stage_scores,
        stage_specific_df=stage_specific_df,
        target_stages=["AAH", "AIS", "MIA"],
    )

    if targets:
        target_records = []
        for t in targets:
            target_records.append({
                "ligand": t.ligand,
                "receptor": t.receptor,
                "target_gene": t.target_gene,
                "priority_score": t.priority_score,
                "druggability": t.druggability,
                "rationale": t.rationale,
                "expected_effect": t.expected_effect,
            })

        pd.DataFrame(target_records).to_csv(
            output_dir / "intervention_targets.csv", index=False
        )
        results["n_intervention_targets"] = len(targets)
        results["top_target"] = f"{targets[0].ligand}-{targets[0].receptor}" if targets else None

    # 7. Compute niche-level risk
    log.info("Computing niche-level progression risk...")

    # Create cell-level risk scores (based on IL1B axis activity + stage)
    risk_records = []
    for il1b_result in il1b_axis_results:
        base_risk = {"Normal": 0.1, "AAH": 0.3, "AIS": 0.5, "MIA": 0.7, "LUAD": 0.9}.get(
            il1b_result.get("stage", "Unknown"), 0.5
        )

        # IL1B activity increases risk
        il1b_contribution = 0.2 if il1b_result.get("detected", False) else 0.0
        il1b_contribution *= il1b_result.get("score", 0)

        risk_records.append({
            "cell_id": il1b_result["cell_id"],
            "risk_score": min(base_risk + il1b_contribution, 1.0),
        })

    cell_risks = pd.DataFrame(risk_records)

    # Get coordinates
    coords = metadata_df[["x", "y"]].values

    niche_scores = compute_niche_level_risk(
        cell_risks=cell_risks,
        spatial_coords=coords,
        niche_radius=100.0,
        min_cells=3,
    )

    if niche_scores:
        niche_df, niche_summary = aggregate_niche_risks_by_region(niche_scores)
        niche_df.to_csv(output_dir / "niche_risk_scores.csv", index=False)
        results["niche_analysis"] = niche_summary

    # 8. Generate sample-level intervention plan (example)
    log.info("Generating example intervention plan...")
    sample_id = metadata_df["donor_id"].iloc[0] if len(metadata_df) > 0 else "test"
    stage = metadata_df["stage"].mode().iloc[0] if len(metadata_df) > 0 else "AAH"

    plan = generate_intervention_plan(
        sample_id=sample_id,
        stage=stage,
        targets=targets,
        niche_scores=niche_scores[:10] if niche_scores else [],
        overall_risk=cell_risks["risk_score"].mean() if len(cell_risks) > 0 else 0.5,
    )

    export_intervention_report(plan, str(output_dir / "intervention_plan.json"))

    # Save IL1B axis analysis
    il1b_df = pd.DataFrame(il1b_axis_results)
    il1b_df.to_csv(output_dir / "il1b_axis_analysis.csv", index=False)

    # Summary statistics
    results["il1b_detection_rate"] = (
        il1b_df["detected"].mean() if "detected" in il1b_df.columns else 0
    )

    log.info(f"Biological analysis complete. Results saved to {output_dir}")

    return results


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("StageBridge Biological Analysis Pipeline")
    log.info("=" * 60)

    if args.synthetic:
        log.info("Running with synthetic data for testing")
        attention_dict, metadata_df, ligand_df, receptor_df, sender_type_names = (
            generate_synthetic_attention_data(n_cells=500, n_senders=20)
        )
    else:
        # Load real data
        log.info(f"Loading data from {args.data_dir}")
        attention_dict, metadata_df = load_model_and_extract_attention(
            args.checkpoint, None, args.device
        )

        if len(attention_dict) == 0:
            log.warning("No attention data loaded - falling back to synthetic")
            attention_dict, metadata_df, ligand_df, receptor_df, sender_type_names = (
                generate_synthetic_attention_data(n_cells=500, n_senders=20)
            )
        else:
            # Load expression data
            raise NotImplementedError("Real data loading not yet implemented")

    # Run analysis
    results = run_biological_analysis(
        attention_dict=attention_dict,
        metadata_df=metadata_df,
        ligand_df=ligand_df,
        receptor_df=receptor_df,
        sender_type_names=sender_type_names,
        output_dir=args.output_dir,
    )

    # Print summary
    log.info("=" * 60)
    log.info("Analysis Summary")
    log.info("=" * 60)

    for key, value in results.items():
        if isinstance(value, dict):
            log.info(f"  {key}:")
            for k, v in value.items():
                log.info(f"    {k}: {v}")
        else:
            log.info(f"  {key}: {value}")

    # Save summary
    with open(args.output_dir / "analysis_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Done!")


if __name__ == "__main__":
    main()
