#!/usr/bin/env python3
"""Run H1.3 (plasticity) hypothesis validation after training.

This script validates that:
  - H1.3.0: Niche conditioning reduces fate uncertainty (entropy reduction)
  - H1.3.1: KACs/RPII have highest plasticity among epithelial cells
  - H1.3.2: IL1B-high niches commit plastic cells toward tumor

Uses REAL model outputs - no placeholders.

Usage:
    python -m stagebridge.pipelines.run_plasticity_validation \
        --checkpoint /path/to/best_checkpoint.pt \
        --checkpoint_no_niche /path/to/no_niche_ablation.pt \
        --data_dir /path/to/canonical/ \
        --output_dir /path/to/plasticity_results/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="H1.3 Plasticity Validation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained full model checkpoint")
    parser.add_argument("--checkpoint_no_niche", type=str, default=None,
                        help="Path to no-niche ablation checkpoint (for niche resolution)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to canonical data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for plasticity results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cells.parquet
    cells_path = Path(args.data_dir) / "cells.parquet"
    if not cells_path.exists():
        logger.error(f"cells.parquet not found at {cells_path}")
        return 1

    cells_df = pd.read_parquet(cells_path)
    logger.info(f"Loaded {len(cells_df)} cells from {cells_path}")

    # Import utilities
    from stagebridge.evaluation.model_inference import (
        load_checkpoint,
        build_model_from_checkpoint,
        prepare_inference_data,
        run_inference,
    )
    from stagebridge.evaluation.plasticity import (
        generate_plasticity_report,
        report_to_dict,
    )

    # Load full model and run inference
    logger.info(f"Loading full model from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    model = build_model_from_checkpoint(checkpoint, device=args.device)

    dataloader, cell_ids, current_stage = prepare_inference_data(
        cells_df, batch_size=args.batch_size
    )

    logger.info("Running inference with full model...")
    outputs_full = run_inference(
        model=model,
        dataloader=dataloader,
        cell_ids=cell_ids,
        current_stage=current_stage,
        device=args.device,
    )

    # Optionally run no-niche model for niche resolution comparison
    outputs_no_niche = None
    if args.checkpoint_no_niche and Path(args.checkpoint_no_niche).exists():
        logger.info(f"Loading no-niche ablation from {args.checkpoint_no_niche}")
        checkpoint_no_niche = load_checkpoint(args.checkpoint_no_niche, device=args.device)
        model_no_niche = build_model_from_checkpoint(checkpoint_no_niche, device=args.device)

        logger.info("Running inference with no-niche model...")
        outputs_no_niche = run_inference(
            model=model_no_niche,
            dataloader=dataloader,
            cell_ids=cell_ids,
            current_stage=current_stage,
            device=args.device,
        )
    else:
        logger.warning("No no-niche checkpoint provided, skipping niche resolution analysis")

    # Prepare cell types and IL1B niche features
    cell_types = cells_df['cell_type'].values if 'cell_type' in cells_df.columns else None

    # Extract IL1B-related niche features if available
    # Look for pathway scores or explicit IL1B columns
    il1b_cols = [c for c in cells_df.columns if 'il1b' in c.lower() or 'pathway' in c.lower()]
    if il1b_cols:
        # Use first matching column as proxy for IL1B niche score
        niche_features = cells_df[il1b_cols[0]].values.astype(np.float32)
        niche_feature_name = il1b_cols[0]
        logger.info(f"Using '{niche_feature_name}' as IL1B niche feature proxy")
    else:
        # Fall back to niche_influence from attention weights
        niche_features = outputs_full.niche_influence
        niche_feature_name = "niche_influence"
        logger.info("Using attention-derived niche_influence as IL1B proxy")

    # Generate plasticity report
    logger.info("Generating plasticity report...")
    report = generate_plasticity_report(
        transition_probs=torch.from_numpy(outputs_full.stage_probs),
        transition_probs_no_niche=(
            torch.from_numpy(outputs_no_niche.stage_probs)
            if outputs_no_niche is not None else None
        ),
        cell_types=cell_types,
        niche_features=niche_features,
        niche_feature_name=niche_feature_name,
        stages=["Normal", "AAH", "AIS", "MIA", "LUAD"],
        bifurcation_percentile=90.0,
    )

    # Convert to dict and save
    report_dict = report_to_dict(report)

    # Add metadata
    report_dict["metadata"] = {
        "n_cells": len(cells_df),
        "checkpoint": str(args.checkpoint),
        "checkpoint_no_niche": str(args.checkpoint_no_niche) if args.checkpoint_no_niche else None,
        "niche_feature": niche_feature_name,
    }

    # Save results
    results_path = output_dir / "plasticity_validation.json"
    with open(results_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Saved plasticity validation to {results_path}")

    # Save inference outputs
    outputs_full.to_dataframe().to_parquet(
        output_dir / "full_model_inference.parquet", index=False
    )
    if outputs_no_niche is not None:
        outputs_no_niche.to_dataframe().to_parquet(
            output_dir / "no_niche_inference.parquet", index=False
        )

    # Print summary
    logger.info("=" * 60)
    logger.info("H1.3 PLASTICITY VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mean plasticity: {report.mean_plasticity:.4f}")
    logger.info(f"Bifurcation cells: {report.bifurcation_result.n_bifurcation_cells:,} "
                f"({100 * report.bifurcation_result.n_bifurcation_cells / report.bifurcation_result.n_total_cells:.1f}%)")

    if report.niche_resolution:
        logger.info(f"Niche resolution (mean): {report.niche_resolution['mean_resolution']:.4f}")
        logger.info(f"Niche resolution (Cohen's d): {report.niche_resolution['cohens_d']:.4f}")
        h1_3_1_supported = report.niche_resolution['cohens_d'] > 0.2
        logger.info(f"H1.3.1 (niche reduces uncertainty): {'SUPPORTED' if h1_3_1_supported else 'NOT SUPPORTED'}")

    if report.fate_commitment:
        fc = report.fate_commitment
        logger.info(f"IL1B-tumor correlation: {fc.correlation_tumor:.4f}")
        logger.info(f"IL1B odds ratio (tumor vs repair): {fc.odds_ratio:.4f}")
        h1_3_2_supported = fc.correlation_tumor > 0.1 and fc.odds_ratio > 1.0
        logger.info(f"H1.3.2 (IL1B commits to tumor): {'SUPPORTED' if h1_3_2_supported else 'NOT SUPPORTED'}")

    if report.cell_type_ranking:
        logger.info("\nCell type plasticity ranking (top 5):")
        for ct, score in report.cell_type_ranking[:5]:
            logger.info(f"  {ct}: {score:.4f}")

    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
