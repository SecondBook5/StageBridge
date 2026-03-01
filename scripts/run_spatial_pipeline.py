#!/usr/bin/env python
"""
Run the full spatial Visium conversion pipeline end-to-end.

Steps:
  1. Expand all GSM*.tar.gz from the extracted dir into per-sample dirs.
  2. Load each sample and write a per-sample h5ad.
  3. Build a manifest CSV.
  4. Merge all per-sample h5ad → spatial_merged.h5ad

Usage:
    python scripts/run_spatial_pipeline.py [--dry-run]

Set STAGEBRIDGE_DATA_ROOT before running if not using the default
/mnt/e/StageBridge_data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stagebridge.logging_utils import configure_root_logger, get_logger
configure_root_logger()
log = get_logger(__name__)

from stagebridge import config
from stagebridge.io.geo_spatial import (
    expand_spatial_tarballs,
    write_spatial_h5ad,
    build_spatial_manifest,
    merge_spatial_h5ad,
)


def main(dry_run: bool = False) -> None:
    extracted_dir = config.spatial_extracted_dir()
    samples_dir   = config.spatial_samples_dir()
    interim_dir   = config.interim_spatial_dir()
    manifest      = config.spatial_manifest_csv()
    merged        = config.spatial_merged_h5ad()

    config.ensure_dir(samples_dir)
    config.ensure_dir(interim_dir)
    config.ensure_dir(merged.parent)

    log.info("Data root    : %s", config.get_data_root())
    log.info("Extracted dir: %s", extracted_dir)
    log.info("Samples dir  : %s", samples_dir)
    log.info("Interim dir  : %s", interim_dir)

    if dry_run:
        log.info("[DRY RUN] No files will be written.")

    # ── Step 1: Expand tarballs ───────────────────────────────────────────────
    log.info("Step 1/4 — Expanding spatial tarballs ...")
    if not dry_run:
        expand_spatial_tarballs(extracted_dir, samples_dir)

    # ── Step 2: Per-sample h5ad ───────────────────────────────────────────────
    sample_dirs = sorted(
        p for p in samples_dir.iterdir()
        if p.is_dir() and p.name.startswith("GSM")
    ) if samples_dir.exists() else []

    log.info("Step 2/4 — Converting %d samples to h5ad ...", len(sample_dirs))
    for sd in sample_dirs:
        out_h5ad = interim_dir / f"{sd.name}.h5ad"
        if out_h5ad.exists():
            log.info("  Skip (exists): %s", out_h5ad)
            continue
        log.info("  Converting: %s", sd.name)
        if not dry_run:
            write_spatial_h5ad(sd, out_h5ad)

    # ── Step 3: Manifest ─────────────────────────────────────────────────────
    log.info("Step 3/4 — Building spatial manifest ...")
    if not dry_run:
        build_spatial_manifest(samples_dir, manifest)

    # ── Step 4: Merge ────────────────────────────────────────────────────────
    log.info("Step 4/4 — Merging into %s ...", merged)
    if not dry_run:
        merge_spatial_h5ad(manifest, merged)

    log.info("Spatial pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StageBridge spatial pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
