#!/usr/bin/env python3
"""Check completeness of publication figures.

Verifies:
- All required figures exist in all formats (PNG, PDF, SVG)
- Source data exists for figures
- Manifest is up-to-date

Usage:
    python scripts/check_figure_completeness.py
    python scripts/check_figure_completeness.py --fix-manifest
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_mark(status: bool) -> str:
    """Return colored check mark or X."""
    return f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"


def warning_mark() -> str:
    """Return colored warning mark."""
    return f"{YELLOW}⚠{RESET}"


REQUIRED_MAIN_FIGURES = [
    "fig01_reference_geometry",
    "fig02_training_curves",
    "fig03_spatial_backends",
    "fig04_embeddings",
    "fig05_ablation_heatmap",
    "fig06_attention",
    "fig07_biology",
]

REQUIRED_SUPP_FIGURES = [
    "figS1_data_qc",
    "figS2_reference_diagnostics",
    "figS3_hpo_history",
]

REQUIRED_FORMATS = ["png", "pdf", "svg"]


def check_figure_exists(base_path: Path) -> tuple[bool, list[str]]:
    """Check if figure exists in all required formats.

    Returns:
        (all_exist, missing_formats)
    """
    missing = []
    for fmt in REQUIRED_FORMATS:
        if not (base_path.parent / f"{base_path.name}.{fmt}").exists():
            missing.append(fmt)
    return len(missing) == 0, missing


def check_source_data(data_root: Path) -> dict[str, bool]:
    """Check if required source data exists."""
    checks = {
        "Reference geometry": (data_root / "processed/luad_evo/reference_geometry/fused_embedding.parquet").exists(),
        "Training results": (data_root / "runs/v1_complete/results.json").exists(),
        "Spatial benchmark": (data_root / "processed/luad_evo/spatial_benchmark/backend_comparison.json").exists(),
        "Canonical data": (data_root / "processed/luad_evo/canonical/snrna_canonical.h5ad").exists(),
        "Ablation results": (data_root / "runs/ablations").exists(),
    }
    return checks


def regenerate_manifest(figures_root: Path) -> None:
    """Regenerate figure manifest from actual files."""
    from datetime import datetime

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "main_figures": {},
        "supplementary_figures": {},
        "formats": REQUIRED_FORMATS,
        "dpi": 300,
    }

    # Scan main figures
    main_dir = figures_root / "main"
    if main_dir.exists():
        for png_file in sorted(main_dir.glob("*.png")):
            base = png_file.stem
            manifest["main_figures"][base] = {
                "png": str(png_file.relative_to(figures_root)),
                "pdf": str(png_file.with_suffix(".pdf").relative_to(figures_root)) if png_file.with_suffix(".pdf").exists() else None,
                "svg": str(png_file.with_suffix(".svg").relative_to(figures_root)) if png_file.with_suffix(".svg").exists() else None,
            }

    # Scan supplementary figures
    supp_dir = figures_root / "supplementary"
    if supp_dir.exists():
        for png_file in sorted(supp_dir.glob("*.png")):
            base = png_file.stem
            manifest["supplementary_figures"][base] = {
                "png": str(png_file.relative_to(figures_root)),
                "pdf": str(png_file.with_suffix(".pdf").relative_to(figures_root)) if png_file.with_suffix(".pdf").exists() else None,
                "svg": str(png_file.with_suffix(".svg").relative_to(figures_root)) if png_file.with_suffix(".svg").exists() else None,
            }

    # Save manifest
    manifest_path = figures_root / "manifests" / "figure_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"{GREEN}✓{RESET} Regenerated manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Check publication figure completeness")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/scratch/chaunzt1/stagebridge"),
        help="Root data directory (default: /scratch/chaunzt1/stagebridge)"
    )
    parser.add_argument(
        "--fix-manifest",
        action="store_true",
        help="Regenerate figure manifest from existing files"
    )
    args = parser.parse_args()

    data_root = args.data_root
    figures_root = data_root / "runs/publication_figures"

    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Publication Figure Completeness Check{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    # Check source data
    print(f"{BLUE}Source Data:{RESET}")
    source_checks = check_source_data(data_root)
    all_sources_ok = True
    for name, exists in source_checks.items():
        print(f"  {check_mark(exists)} {name}")
        if not exists:
            all_sources_ok = False
    print()

    if not all_sources_ok:
        print(f"{warning_mark()} {YELLOW}Some source data is missing. Run prerequisite pipeline stages first.{RESET}\n")

    # Check main figures
    print(f"{BLUE}Main Figures:{RESET}")
    main_dir = figures_root / "main"
    main_complete = 0
    main_partial = 0
    main_missing = 0

    for fig_id in REQUIRED_MAIN_FIGURES:
        base_path = main_dir / fig_id
        exists, missing = check_figure_exists(base_path)

        if exists:
            print(f"  {check_mark(True)} {fig_id}")
            main_complete += 1
        elif missing == REQUIRED_FORMATS:
            print(f"  {check_mark(False)} {fig_id} (missing: all formats)")
            main_missing += 1
        else:
            print(f"  {warning_mark()} {fig_id} (missing: {', '.join(missing)})")
            main_partial += 1

    print(f"  {'-'*60}")
    print(f"  Complete: {main_complete}/{len(REQUIRED_MAIN_FIGURES)}")
    if main_partial > 0:
        print(f"  Partial: {main_partial}")
    if main_missing > 0:
        print(f"  Missing: {main_missing}")
    print()

    # Check supplementary figures
    print(f"{BLUE}Supplementary Figures:{RESET}")
    supp_dir = figures_root / "supplementary"
    supp_complete = 0
    supp_partial = 0
    supp_missing = 0

    for fig_id in REQUIRED_SUPP_FIGURES:
        base_path = supp_dir / fig_id
        exists, missing = check_figure_exists(base_path)

        if exists:
            print(f"  {check_mark(True)} {fig_id}")
            supp_complete += 1
        elif missing == REQUIRED_FORMATS:
            print(f"  {check_mark(False)} {fig_id} (missing: all formats)")
            supp_missing += 1
        else:
            print(f"  {warning_mark()} {fig_id} (missing: {', '.join(missing)})")
            supp_partial += 1

    print(f"  {'-'*60}")
    print(f"  Complete: {supp_complete}/{len(REQUIRED_SUPP_FIGURES)}")
    if supp_partial > 0:
        print(f"  Partial: {supp_partial}")
    if supp_missing > 0:
        print(f"  Missing: {supp_missing}")
    print()

    # Check manifest
    print(f"{BLUE}Manifest:{RESET}")
    manifest_path = figures_root / "manifests" / "figure_manifest.json"
    manifest_exists = manifest_path.exists()
    print(f"  {check_mark(manifest_exists)} figure_manifest.json")

    if manifest_exists:
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  Generated: {manifest.get('generated_at', 'unknown')}")
        print(f"  Main figures listed: {len(manifest.get('main_figures', {}))}")
        print(f"  Supplementary figures listed: {len(manifest.get('supplementary_figures', {}))}")
    print()

    # Fix manifest if requested
    if args.fix_manifest:
        print(f"{BLUE}Regenerating manifest...{RESET}")
        regenerate_manifest(figures_root)
        print()

    # Summary
    total_required = len(REQUIRED_MAIN_FIGURES) + len(REQUIRED_SUPP_FIGURES)
    total_complete = main_complete + supp_complete
    total_partial = main_partial + supp_partial
    total_missing = main_missing + supp_missing

    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Summary:{RESET}")
    print(f"  Complete: {total_complete}/{total_required} ({100*total_complete/total_required:.1f}%)")
    if total_partial > 0:
        print(f"  {warning_mark()} Partial: {total_partial} (missing some formats)")
    if total_missing > 0:
        print(f"  {check_mark(False)} Missing: {total_missing} (all formats missing)")
    print(f"{BLUE}{'='*70}{RESET}\n")

    # Recommendations
    if total_missing > 0 or total_partial > 0:
        print(f"{BLUE}Recommendations:{RESET}")
        if not all_sources_ok:
            print("  1. Run prerequisite pipeline stages to generate missing source data")
        if total_missing > 0:
            print("  2. Run figure generation: snakemake publication_figures --profile workflow/slurm")
        if total_partial > 0:
            print("  3. For partial figures, delete all formats and regenerate:")
            print("     rm $DATA/runs/publication_figures/main/<figure_id>.*")
        if not manifest_exists or args.fix_manifest:
            print("  4. Update manifest: python scripts/check_figure_completeness.py --fix-manifest")
        print()
    else:
        print(f"{GREEN}✓ All required figures are complete!{RESET}\n")


if __name__ == "__main__":
    main()
