#!/usr/bin/env python
"""
Run the full snRNA-seq conversion pipeline end-to-end.

Steps:
  1. Build manifest from raw extracted files.
  2. Convert each sample .raw_counts.mtx.txt.gz → .h5ad
  3. Merge all per-sample h5ad → snrna_merged.h5ad

Usage:
    python scripts/run_snrna_pipeline.py [--dry-run]

Set STAGEBRIDGE_DATA_ROOT before running if not using the default
/mnt/e/StageBridge_data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stagebridge.logging_utils import configure_root_logger, get_logger
configure_root_logger()
log = get_logger(__name__)

from stagebridge import config
from stagebridge.io.geo_snrna import (
    build_snrna_manifest,
    convert_snrna_dense_counts_to_h5ad,
    merge_snrna_h5ad,
)


def main(dry_run: bool = False) -> None:
    # ── Paths ────────────────────────────────────────────────────────────────
    raw_dir     = config.snrna_extracted_dir()
    interim_dir = config.interim_snrna_dir()
    manifest    = config.snrna_manifest_csv()
    merged      = config.snrna_merged_h5ad()

    config.ensure_dir(interim_dir)
    config.ensure_dir(merged.parent)

    log.info("Data root  : %s", config.get_data_root())
    log.info("Raw dir    : %s", raw_dir)
    log.info("Interim dir: %s", interim_dir)
    log.info("Manifest   : %s", manifest)
    log.info("Merged h5ad: %s", merged)

    if dry_run:
        log.info("[DRY RUN] No files will be written.")

    # ── Step 1: Manifest ─────────────────────────────────────────────────────
    log.info("Step 1/3 — Building snRNA manifest ...")
    if not dry_run:
        build_snrna_manifest(raw_dir, manifest)

    # ── Step 2: Per-sample conversion ────────────────────────────────────────
    import pandas as pd
    if not dry_run:
        df = pd.read_csv(manifest)
    else:
        import glob
        files = sorted(Path(raw_dir).glob("*.raw_counts.mtx.txt.gz"))
        df = pd.DataFrame({"sample_id": [f.stem.split(".")[0] for f in files],
                           "input_path": [str(f) for f in files]})

    log.info("Step 2/3 — Converting %d samples to h5ad ...", len(df))
    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        out_h5ad = interim_dir / f"{sample_id}.h5ad"
        if out_h5ad.exists():
            log.info("  Skip (exists): %s", out_h5ad)
            continue
        log.info("  Converting: %s", sample_id)
        if not dry_run:
            convert_snrna_dense_counts_to_h5ad(Path(row["input_path"]), out_h5ad)

    # ── Step 3: Merge ────────────────────────────────────────────────────────
    log.info("Step 3/3 — Merging into %s ...", merged)
    if not dry_run:
        merge_snrna_h5ad(manifest, merged)

    log.info("snRNA pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StageBridge snRNA pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
