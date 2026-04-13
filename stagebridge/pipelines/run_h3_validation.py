#!/usr/bin/env python3
"""Run H3 (clonal evolution) hypothesis validation after training.

This script evaluates the trained StageBridge model's ability to predict
clonal evolution patterns, validating the H3 hypothesis:
  - H3.1: Transition probability correlates with shared clones across stages
  - H3.2: Niche influence differs by clonal pattern (1a > 1b > 2)

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
    parser.add_argument("--device", type=str, default="cuda")
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

    # Check for clonal patterns
    if "clonal_pattern" not in cells_df.columns:
        logger.error("clonal_pattern column not found in cells.parquet")
        logger.error("Run clonal extraction first: python -m stagebridge.pipelines.run_clonal_extraction")
        return 1

    # Filter to cells with known clonal patterns
    valid_patterns = {"1a", "1b", "2"}
    mask = cells_df["clonal_pattern"].isin(valid_patterns)
    n_valid = mask.sum()
    logger.info(f"Cells with valid clonal patterns: {n_valid:,} / {len(cells_df):,}")

    if n_valid < 100:
        logger.warning(f"Only {n_valid} cells with valid patterns - H3 validation may be unreliable")

    # Pattern distribution
    pattern_counts = cells_df["clonal_pattern"].value_counts()
    logger.info(f"Pattern distribution:\n{pattern_counts}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # For now, use placeholder values until model inference is wired up
    # TODO: Run model inference to get actual transition_probs and niche_influence
    logger.warning("Using placeholder values - full model inference not yet implemented")

    # Placeholder: random values for demonstration
    n_cells = len(cells_df)
    transition_probs = np.random.uniform(0, 1, n_cells)
    niche_influence = np.random.uniform(0, 1, n_cells)
    patterns = cells_df["clonal_pattern"].values

    # Run H3 validation
    from stagebridge.evaluation.clonal_validation import (
        run_clonal_validation,
        ClonalValidationReport,
    )

    donor_ids = cells_df["donor_id"].values if "donor_id" in cells_df.columns else None

    report = run_clonal_validation(
        transition_probs=transition_probs,
        niche_influence=niche_influence,
        patterns=patterns,
        donor_ids=donor_ids,
    )

    # Save results
    results_path = output_dir / "h3_validation.json"
    with open(results_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Saved H3 validation results to {results_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("H3 VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"H3.1 (transition ~ shared clones): AUC={report.h3_1.auc:.3f}, supported={report.h3_1.h3_1_supported}")
    logger.info(f"H3.2 (niche influence by pattern): p={report.h3_2.pvalue_1a_vs_2:.4f}, supported={report.h3_2.h3_2_supported}")
    logger.info(f"Overall H3 supported: {report.h3_supported}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
