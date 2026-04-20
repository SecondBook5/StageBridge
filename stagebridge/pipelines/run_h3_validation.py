#!/usr/bin/env python3
"""Run H3 (clonal evolution) hypothesis validation after training.

This script evaluates the trained StageBridge model's ability to predict
clonal evolution patterns, validating the H3 hypothesis:
  - H3.1: Transition probability correlates with shared clones across stages
  - H3.2: Niche influence differs by clonal pattern (1a > 1b > 2)

Uses REAL model outputs - no placeholders.

Usage:
    python -m stagebridge.pipelines.run_h3_validation \
        --checkpoint /path/to/best_checkpoint.pt \
        --data_dir /path/to/canonical/ \
        --output_dir /path/to/h3_results/
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
    parser = argparse.ArgumentParser(description="H3 Hypothesis Validation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to canonical data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for H3 results")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cells.parquet with clonal patterns
    cells_path = Path(args.data_dir) / "cells.parquet"
    if not cells_path.exists():
        logger.error(f"cells.parquet not found at {cells_path}")
        return 1

    cells_df = pd.read_parquet(cells_path)
    logger.info(f"Loaded {len(cells_df)} cells from {cells_path}")

    # Load neighborhoods.parquet for proper niche context (optional but recommended)
    neighborhoods_path = Path(args.data_dir) / "neighborhoods.parquet"
    if neighborhoods_path.exists():
        neighborhoods_df = pd.read_parquet(neighborhoods_path)
        logger.info(f"Loaded {len(neighborhoods_df)} neighborhoods from {neighborhoods_path}")
    else:
        logger.warning(f"neighborhoods.parquet not found at {neighborhoods_path}")
        logger.warning("Using receiver-only inference (degraded niche influence accuracy)")
        neighborhoods_df = None

    # Check for clonal patterns
    if "clonal_pattern" not in cells_df.columns:
        logger.error("clonal_pattern column not found in cells.parquet")
        logger.error("Run clonal extraction first: python -m stagebridge.pipelines.run_clonal_extraction")
        return 1

    # Filter to cells with known clonal patterns
    # Canonical patterns: 1a (direct lineage), 1b (branched), 2 (independent)
    valid_patterns = {"1a", "1b", "2"}
    mask = cells_df["clonal_pattern"].isin(valid_patterns)
    n_valid = mask.sum()

    # Pattern distribution
    pattern_counts = cells_df["clonal_pattern"].value_counts()
    logger.info(f"Pattern distribution:\n{pattern_counts}")

    # If no canonical patterns found, check if CNV extraction was run
    if n_valid == 0:
        logger.error("=" * 60)
        logger.error("NO CANONICAL CLONAL PATTERNS FOUND (1a, 1b, 2)")
        logger.error("=" * 60)
        logger.error("H3 validation requires clonal patterns from CNV inference.")
        logger.error("Run clonal extraction first:")
        logger.error("  python -m stagebridge.pipelines.run_clonal_extraction \\")
        logger.error("      --spatial-h5ad <path_to_spatial.h5ad> \\")
        logger.error("      --output-dir <output_dir>")
        logger.error("")
        logger.error("Then rerun data prep with --clonal_patterns flag.")

        # Save report indicating H3 validation skipped
        skip_report = {
            "h3_supported": None,
            "status": "SKIPPED",
            "reason": "No canonical clonal patterns (1a/1b/2) found in data",
            "pattern_distribution": pattern_counts.to_dict(),
            "recommendation": "Run clonal extraction pipeline (CNV inference) first",
        }
        results_path = output_dir / "h3_validation.json"
        with open(results_path, "w") as f:
            json.dump(skip_report, f, indent=2)
        logger.info(f"Saved skip report to {results_path}")
        return 1

    logger.info(f"Cells with valid clonal patterns: {n_valid:,} / {len(cells_df):,}")

    if n_valid < 100:
        logger.warning(f"Only {n_valid} cells with valid patterns - H3 validation may be unreliable")

    # Load checkpoint and run inference
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    logger.info(f"Running model inference from {checkpoint_path}")

    # Import inference utilities
    from stagebridge.evaluation.model_inference import (
        load_checkpoint,
        build_model_from_checkpoint,
        prepare_inference_data,
        run_inference,
    )

    # Load model
    checkpoint = load_checkpoint(checkpoint_path, device=args.device)
    model = build_model_from_checkpoint(checkpoint, device=args.device)

    # Prepare data (with neighborhood context if available)
    dataloader, cell_ids, current_stage = prepare_inference_data(
        cells_df,
        neighborhoods_df=neighborhoods_df,
        batch_size=args.batch_size,
    )

    # Run inference to get REAL outputs
    logger.info("Running model inference...")
    outputs = run_inference(
        model=model,
        dataloader=dataloader,
        cell_ids=cell_ids,
        current_stage=current_stage,
        device=args.device,
    )

    logger.info(f"Inference complete: {len(outputs.transition_probs)} cells")
    logger.info(f"  Transition prob range: [{outputs.transition_probs.min():.3f}, {outputs.transition_probs.max():.3f}]")
    logger.info(f"  Niche influence range: [{outputs.niche_influence.min():.3f}, {outputs.niche_influence.max():.3f}]")

    # Map outputs to DataFrame
    cell_id_to_idx = {cid: i for i, cid in enumerate(outputs.cell_ids)}

    # Get transition_probs and niche_influence for cells with clonal patterns
    transition_probs = np.array([
        outputs.transition_probs[cell_id_to_idx[cid]]
        for cid in cells_df['cell_id'].values
        if cid in cell_id_to_idx
    ])
    niche_influence = np.array([
        outputs.niche_influence[cell_id_to_idx[cid]]
        for cid in cells_df['cell_id'].values
        if cid in cell_id_to_idx
    ])
    patterns = cells_df["clonal_pattern"].values

    # Run H3 validation
    from stagebridge.evaluation.clonal_validation import (
        run_clonal_validation,
    )

    donor_ids = cells_df["donor_id"].values if "donor_id" in cells_df.columns else None

    report = run_clonal_validation(
        transition_probs=transition_probs,
        niche_influence=niche_influence,
        clonal_patterns=patterns,
        donor_ids=donor_ids,
    )

    # Save inference outputs for downstream analysis
    inference_df = outputs.to_dataframe()
    inference_df.to_parquet(output_dir / "model_inference.parquet", index=False)
    logger.info(f"Saved model inference to {output_dir / 'model_inference.parquet'}")

    # Save H3 validation results
    results_path = output_dir / "h3_validation.json"
    with open(results_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Saved H3 validation results to {results_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("H3 VALIDATION SUMMARY (REAL MODEL OUTPUTS)")
    logger.info("=" * 60)
    logger.info(f"H3.1 (transition ~ shared clones): AUC={report.h3_1.auc:.3f}, supported={report.h3_1.h3_1_supported}")
    logger.info(f"H3.2 (niche influence by pattern): p={report.h3_2.pvalue_1a_vs_2:.4f}, supported={report.h3_2.h3_2_supported}")
    logger.info(f"Overall H3 supported: {report.h3_supported}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
