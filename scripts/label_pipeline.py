#!/usr/bin/env python
"""
Unified Label Repair Pipeline

Consolidates 7 separate wrapper scripts into one CLI with subcommands.
Provides efficient manifest caching and clear pipeline orchestration.

Usage:
    python scripts/label_pipeline.py manifest           # Build manifest only
    python scripts/label_pipeline.py repair             # Full repair workflow
    python scripts/label_pipeline.py support            # Evaluate support
    python scripts/label_pipeline.py refine             # Refine labels
    python scripts/label_pipeline.py clonal             # Run clonal backend
    python scripts/label_pipeline.py cna                # Run CNA backend
    python scripts/label_pipeline.py phylogeny          # Run phylogeny backend
    python scripts/label_pipeline.py all                # Run complete pipeline

Replaces:
    - build_cohort_manifest.py
    - generate_label_reports.py
    - evaluate_label_support.py
    - refine_labels.py
    - run_clonal_backend.py
    - run_cna_backend.py
    - run_phylogeny_backend.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stagebridge.labels.cohort_manifest import build_cleaned_cohort_manifest
from stagebridge.notebook_api import compose_config
from stagebridge.pipelines.run_label_repair import (
    run_label_cna,
    run_label_clonal,
    run_label_manifest,
    run_label_phylogeny,
    run_label_refinement,
    run_label_repair,
    run_label_support,
)


def main():
    parser = argparse.ArgumentParser(
        description="Unified label repair pipeline with subcommands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s manifest           Build cohort manifest
  %(prog)s all                Run complete pipeline
  %(prog)s clonal             Run clonal analysis only
        """
    )

    subparsers = parser.add_subparsers(dest='command', required=True, help='Pipeline command')

    # Subcommand definitions
    subparsers.add_parser('manifest', help='Build cleaned cohort manifest')
    subparsers.add_parser('repair', help='Run full label repair workflow')
    subparsers.add_parser('support', help='Evaluate donor-held-out target viability')
    subparsers.add_parser('refine', help='Derive refined labels and risk scores')
    subparsers.add_parser('clonal', help='Run clonal backend or parse summaries')
    subparsers.add_parser('cna', help='Run CNA backend or parse summaries')
    subparsers.add_parser('phylogeny', help='Run phylogeny backend or parse summaries')
    subparsers.add_parser('all', help='Run complete label repair pipeline')

    # Global options
    parser.add_argument('--config-overrides', nargs='+', default=["labels=repair"],
                       help='Config overrides (default: labels=repair)')

    args = parser.parse_args()

    # Compose config once
    print(f"Loading configuration (overrides: {args.config_overrides})...")
    cfg = compose_config(overrides=args.config_overrides)

    # Build manifest once if needed by downstream commands
    manifest_cache = None
    if args.command in ['support', 'refine', 'clonal', 'cna', 'phylogeny', 'all']:
        print("\nBuilding cleaned cohort manifest (shared cache)...")
        manifest_cache = build_cleaned_cohort_manifest(cfg)
        print(f"  Manifest cached for downstream steps")

    # Execute command
    print(f"\nExecuting: {args.command}")
    print("=" * 80)

    if args.command == 'manifest':
        run_label_manifest(cfg)

    elif args.command == 'repair':
        run_label_repair(cfg)

    elif args.command == 'support':
        run_label_support(cfg, cached=manifest_cache)

    elif args.command == 'refine':
        run_label_refinement(cfg, cached=manifest_cache)

    elif args.command == 'clonal':
        run_label_clonal(cfg, manifest=manifest_cache["cleaned_manifest"])

    elif args.command == 'cna':
        run_label_cna(cfg, manifest=manifest_cache["cleaned_manifest"])

    elif args.command == 'phylogeny':
        run_label_phylogeny(cfg, manifest=manifest_cache["cleaned_manifest"])

    elif args.command == 'all':
        # Run complete pipeline with shared caching
        print("\n[1/7] Manifest...")
        run_label_manifest(cfg)

        print("\n[2/7] Label repair...")
        run_label_repair(cfg)

        print("\n[3/7] Label support...")
        run_label_support(cfg, cached=manifest_cache)

        print("\n[4/7] Label refinement...")
        run_label_refinement(cfg, cached=manifest_cache)

        print("\n[5/7] Clonal backend...")
        run_label_clonal(cfg, manifest=manifest_cache["cleaned_manifest"])

        print("\n[6/7] CNA backend...")
        run_label_cna(cfg, manifest=manifest_cache["cleaned_manifest"])

        print("\n[7/7] Phylogeny backend...")
        run_label_phylogeny(cfg, manifest=manifest_cache["cleaned_manifest"])

        print("\n" + "=" * 80)
        print("COMPLETE LABEL REPAIR PIPELINE FINISHED")
        print("=" * 80)

    print(f"\n✓ Command '{args.command}' completed successfully")


if __name__ == "__main__":
    main()
