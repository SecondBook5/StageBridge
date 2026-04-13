#!/usr/bin/env python3
"""Extract clonal evolution patterns from spatial transcriptomics data.

This pipeline runs CNV inference and classifies patients into clonal
evolution patterns (1a, 1b, 2) following Peng et al. 2025 methodology.

Usage:
    python -m stagebridge.pipelines.run_clonal_extraction \
        --spatial-h5ad /path/to/spatial_merged.h5ad \
        --output-dir results/clonal/

Requirements:
    pip install infercnvpy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import scanpy as sc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clonal evolution patterns from spatial data"
    )
    parser.add_argument(
        "--spatial-h5ad",
        type=Path,
        required=True,
        help="Path to spatial AnnData (e.g., spatial_merged.h5ad)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/clonal"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--patient-key",
        default="donor_id",
        help="Column with patient IDs (default: donor_id)",
    )
    parser.add_argument(
        "--stage-key",
        default="stage",
        help="Column with stage annotations (default: stage)",
    )
    parser.add_argument(
        "--cell-type-key",
        default="cell_type",
        help="Column with cell type annotations (default: cell_type)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Window size for CNV smoothing (default: 100)",
    )
    parser.add_argument(
        "--min-clone-cells",
        type=int,
        default=5,
        help="Minimum cells per clone (default: 5)",
    )
    parser.add_argument(
        "--reference-types",
        nargs="+",
        default=["AT2"],
        help="Cell types to use as diploid reference (default: AT2)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Check dependencies
    try:
        import infercnvpy
    except ImportError:
        logger.error(
            "infercnvpy is required. Install with: pip install infercnvpy"
        )
        return 1

    # Load spatial data
    logger.info(f"Loading spatial data from {args.spatial_h5ad}")
    if not args.spatial_h5ad.exists():
        logger.error(f"File not found: {args.spatial_h5ad}")
        return 1

    adata = sc.read_h5ad(args.spatial_h5ad)
    logger.info(f"Loaded {adata.n_obs} spots x {adata.n_vars} genes")

    # Check required columns
    for col, name in [
        (args.patient_key, "patient"),
        (args.stage_key, "stage"),
        (args.cell_type_key, "cell type"),
    ]:
        if col not in adata.obs.columns:
            logger.error(f"Missing {name} column: '{col}' not in adata.obs")
            logger.info(f"Available columns: {list(adata.obs.columns)}")
            return 1

    # Log data summary
    logger.info(f"Patients: {adata.obs[args.patient_key].nunique()}")
    logger.info(f"Stages: {adata.obs[args.stage_key].value_counts().to_dict()}")
    logger.info(f"Cell types: {adata.obs[args.cell_type_key].nunique()}")

    # Import after checking infercnvpy is available
    from stagebridge.clonal import extract_clonal_patterns, ClonalPattern
    from stagebridge.clonal.extract import add_clonal_patterns_to_adata

    # Run extraction
    logger.info("Starting clonal pattern extraction...")
    patterns = extract_clonal_patterns(
        adata,
        output_dir=args.output_dir,
        patient_key=args.patient_key,
        stage_key=args.stage_key,
        cell_type_key=args.cell_type_key,
        reference_types=args.reference_types,
        window_size=args.window_size,
        min_clone_cells=args.min_clone_cells,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("CLONAL PATTERN EXTRACTION COMPLETE")
    logger.info("=" * 60)
    for patient, pattern in sorted(patterns.items()):
        logger.info(f"  {patient}: Pattern {pattern}")

    # Count patterns
    from collections import Counter
    counts = Counter(patterns.values())
    logger.info("-" * 40)
    logger.info("Pattern distribution:")
    for pattern in ["1a", "1b", "2", "stable", "uncategorized"]:
        if pattern in counts:
            logger.info(f"  Pattern {pattern}: {counts[pattern]} patients")

    logger.info("-" * 40)
    logger.info(f"Results saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
