"""Reference-layer pipeline entrypoint.

Maps query cells to HLCA (healthy) reference space using scANVI surgery,
producing latent embeddings and cell-type labels for each cell.

Uses the proper scANVI query-to-reference workflow:
1. Load pretrained HLCA scANVI model from Hugging Face Hub
2. Perform gene set alignment (surgery)
3. Fine-tune query model on subset
4. Infer latent embeddings for all query cells
5. Transfer cell-type labels

Usage:
    python -m stagebridge.pipelines.run_reference \
        --data-root /path/to/stagebridge/data

Output:
    Creates:
    - snrna_hlca_latent.h5ad - Query data with HLCA latent embeddings
    - hlca_labels.parquet - Cell-type label transfer results
    - hlca_mapping_report.json - Quality metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import uuid


def find_hlca_cache_dir(data_root: Path) -> Path | None:
    """Find the HLCA HubModel cache directory."""
    candidates = [
        data_root / "references/hlca/hub_cache",
        data_root / "references/hlca",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Map query cells to HLCA reference space using scANVI surgery"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing processed/ and references/",
    )
    parser.add_argument(
        "--snrna",
        type=str,
        default=None,
        help="Path to snRNA h5ad (default: {data-root}/processed/luad_evo/snrna_qc_normalized.h5ad)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {data-root}/processed/luad_evo/)",
    )
    parser.add_argument(
        "--surgery-epochs",
        type=int,
        default=500,
        help="Number of epochs for scANVI query surgery training",
    )
    parser.add_argument(
        "--train-max-cells",
        type=int,
        default=200000,
        help="Max cells to use for query model training",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for tracking (default: auto-generated)",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    snrna_path = Path(args.snrna) if args.snrna else data_root / "processed/luad_evo/snrna_qc_normalized.h5ad"
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "processed/luad_evo"
    run_id = args.run_id or f"hlca_map_{uuid.uuid4().hex[:8]}"

    # Find HLCA cache
    hlca_cache = find_hlca_cache_dir(data_root)
    if hlca_cache is None:
        print("ERROR: HLCA model cache not found. Run download_references.py first.")
        print(f"Expected at: {data_root / 'references/hlca/hub_cache'}")
        return 1

    print("=" * 60)
    print("HLCA Reference Mapping Pipeline (scANVI Surgery)")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  Query data: {snrna_path}")
    print(f"  HLCA cache: {hlca_cache}")
    print(f"  Output dir: {output_dir}")
    print(f"  Surgery epochs: {args.surgery_epochs}")
    print(f"  Training cells: {args.train_max_cells:,}")
    print()

    if not snrna_path.exists():
        print(f"ERROR: snRNA file not found: {snrna_path}")
        print("Run run_data_prep.py first.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    output_latent_h5ad = output_dir / "snrna_hlca_latent.h5ad"
    output_labels_parquet = output_dir / "hlca_labels.parquet"
    mapping_report_path = output_dir / "hlca_mapping_report.json"
    gene_report_path = output_dir / "hlca_gene_report.json"
    progress_path = output_dir / "hlca_mapping_progress.json"
    processed_hlca_dir = output_dir / "hlca_processed"

    # Build config for map_full_snrna_with_hlca
    hlca_cfg = {
        "hub_repo_id": "scvi-tools/human-lung-cell-atlas-scanvi",
        "model_cache_dir": str(hlca_cache),
        "query_model_dir": str(processed_hlca_dir / "query_model"),
        "surgery_epochs": args.surgery_epochs,
        "train_max_cells": args.train_max_cells,
        "batch_size_infer": 1024,
        "inference_chunk_size": 8192,
        "early_stopping": True,
        "early_stopping_patience": 10,
        "show_progress": True,
        "resume": True,
        "export_probs": False,
        "knn_label_transfer_levels": False,
    }

    try:
        from stagebridge.reference.hlca_mapper import map_full_snrna_with_hlca

        print("\nStarting scANVI query mapping...")
        print("This may take 30-60 minutes for large datasets.\n")

        result = map_full_snrna_with_hlca(
            run_id=run_id,
            snrna_h5ad_path=snrna_path,
            output_latent_h5ad_path=output_latent_h5ad,
            output_labels_parquet_path=output_labels_parquet,
            mapping_report_path=mapping_report_path,
            gene_report_path=gene_report_path,
            progress_path=progress_path,
            processed_hlca_dir=processed_hlca_dir,
            hlca_cfg=hlca_cfg,
        )

        print()
        print("=" * 60)
        print("Reference Mapping Complete")
        print("=" * 60)
        print(f"  Run ID: {result.run_id}")
        print(f"  Latent shape: {result.latent_shape}")
        print(f"  Gene overlap: {result.overlap_percent:.1f}%")
        print(f"  Peak memory: {result.peak_rss_mb:.0f} MB")
        print(f"  Wall time: {result.wall_time_seconds:.1f}s")
        print()
        print("Top 10 cell types:")
        for label, count in result.top10_labels:
            print(f"    {label}: {count:,}")
        print()
        print("Outputs:")
        print(f"  Latent h5ad: {result.latent_h5ad_path}")
        print(f"  Labels: {result.labels_parquet_path}")
        print(f"  Report: {result.mapping_report_path}")
        print()
        print("Next step: run_spatial_benchmark.py")

        return 0

    except Exception as e:
        print(f"\nERROR: Reference mapping failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
