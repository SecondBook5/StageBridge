"""Dual-reference mapping pipeline entrypoint.

Maps query cells to BOTH HLCA (healthy) and LuCA (cancer) reference spaces,
producing separate and fused latent embeddings for each cell.

Dual-reference design:
- HLCA = healthy lung anchor
- LuCA = disease-aware / malignant-progressive anchor
- Fused = comparative coordinate system for progression-relevant cells

Supports three modes:
1. HLCA-only (--hlca-only)
2. LuCA-only (--luca-only)
3. HLCA+LuCA fused (default)

Usage:
    python -m stagebridge.pipelines.run_reference \
        --data-root /path/to/stagebridge/data

Output:
    Creates reference_geometry/ directory with:
    - hlca_embedding.parquet
    - luca_embedding.parquet
    - fused_embedding.parquet
    - reference_confidence.parquet
    - reference_manifest.json
    - feature_overlap_report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal
import uuid


def find_reference_paths(data_root: Path) -> dict[str, Path | None]:
    """Find HLCA and LuCA reference paths."""
    results = {"hlca": None, "luca": None, "hlca_hub_cache": None}

    # HLCA: Check for HubModel cache or h5ad
    hlca_candidates = [
        data_root / "references/hlca/hlca_reference.h5ad",
        data_root / "references/hlca/hlca_core.h5ad",
    ]
    for candidate in hlca_candidates:
        if candidate.exists():
            results["hlca"] = candidate
            break

    # HLCA HubModel cache (for scANVI surgery)
    hub_cache = data_root / "references/hlca/hub_cache"
    if hub_cache.exists():
        results["hlca_hub_cache"] = hub_cache

    # LuCA: Check for h5ad
    luca_candidates = [
        data_root / "references/luca/luca_reference.h5ad",
        data_root / "references/luca/luca_luad.h5ad",
    ]
    for candidate in luca_candidates:
        if candidate.exists():
            results["luca"] = candidate
            break

    return results


def extract_hlca_reference_from_hub(hub_cache: Path, output_path: Path) -> Path:
    """Extract HLCA reference h5ad from HubModel cache."""
    from scvi.hub import HubModel

    print("Loading HLCA reference from HubModel cache...")
    hubmodel = HubModel.pull_from_huggingface_hub(
        "scvi-tools/human-lung-cell-atlas-scanvi",
        cache_dir=hub_cache,
    )

    ref_adata = hubmodel.adata
    print(f"  Reference cells: {ref_adata.n_obs:,}")
    print(f"  Reference genes: {ref_adata.n_vars:,}")

    # Ensure latent embedding exists
    if "X_scanvi_emb" not in ref_adata.obsm:
        print("  Computing latent embeddings...")
        ref_latent = hubmodel.model.get_latent_representation(ref_adata)
        ref_adata.obsm["X_scanvi_emb"] = ref_latent

    # Reindex to gene symbols if feature_name column exists
    if "feature_name" in ref_adata.var.columns:
        print("  Reindexing to gene symbols (feature_name)...")
        # Store original ENSG IDs
        ref_adata.var["ensembl_id"] = ref_adata.var_names.copy()
        # Use feature_name as var_names
        ref_adata.var_names = ref_adata.var["feature_name"].astype(str)
        # Handle duplicates by making unique
        ref_adata.var_names_make_unique()
        print(f"  Gene names: {list(ref_adata.var_names[:5])}")

    print(f"  Saving reference to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref_adata.write_h5ad(output_path)

    return output_path


def reindex_reference_to_symbols(ref_path: Path, max_size_gb: float = 5.0) -> Path:
    """Reindex a reference h5ad to use gene symbols instead of ENSG IDs.

    For large files (> max_size_gb), creates a gene mapping instead of rewriting.
    """
    import anndata

    file_size_gb = ref_path.stat().st_size / (1024**3)
    print(f"Checking gene format in {ref_path.name} ({file_size_gb:.1f} GB)...")

    # For large files, just check and warn - don't rewrite
    if file_size_gb > max_size_gb:
        print(f"  Large file - checking with backed mode...")
        adata = anndata.read_h5ad(ref_path, backed='r')
        first_gene = str(adata.var_names[0])
        if first_gene.startswith("ENSG") and "feature_name" in adata.var.columns:
            # Create a mapping file instead of rewriting
            mapping_path = ref_path.parent / f"{ref_path.stem}_gene_mapping.parquet"
            if not mapping_path.exists():
                import pandas as pd
                gene_map = pd.DataFrame({
                    "ensembl_id": adata.var_names.astype(str),
                    "gene_symbol": adata.var["feature_name"].astype(str),
                })
                gene_map.to_parquet(mapping_path)
                print(f"  Created gene mapping: {mapping_path}")
            print(f"  NOTE: Large file uses ENSG IDs. Pipeline will use feature_name for matching.")
        adata.file.close()
        return ref_path

    adata = anndata.read_h5ad(ref_path)

    # Check if already using symbols (not ENSG)
    first_gene = str(adata.var_names[0])
    if not first_gene.startswith("ENSG"):
        print(f"  Already using gene symbols: {first_gene}")
        return ref_path

    # Check for feature_name column
    if "feature_name" not in adata.var.columns:
        print(f"  WARNING: No feature_name column, keeping ENSG IDs")
        return ref_path

    print(f"  Reindexing from ENSG to gene symbols...")
    adata.var["ensembl_id"] = adata.var_names.copy()
    adata.var_names = adata.var["feature_name"].astype(str)
    adata.var_names_make_unique()

    # Save back
    adata.write_h5ad(ref_path)
    print(f"  Done. Gene names: {list(adata.var_names[:5])}")

    return ref_path


def run_dual_reference_mapping(
    query_path: Path,
    hlca_path: Path | None,
    luca_path: Path | None,
    output_dir: Path,
    *,
    mode: Literal["both", "hlca_only", "luca_only"] = "both",
    mapping_method: str = "knn_projection",
    fusion_method: str = "concat",
    k_neighbors: int = 50,
    hlca_latent_key: str = "X_scanvi_emb",
    luca_latent_key: str = "X_scVI",
    smoke_mode: bool = False,
    run_id: str | None = None,
) -> int:
    """Run dual-reference mapping pipeline."""
    from stagebridge.reference.pipeline import (
        ReferenceGeometryConfig,
        run_reference_pipeline,
    )

    run_id = run_id or f"ref_geo_{uuid.uuid4().hex[:8]}"

    # Determine which references to use
    use_hlca = mode in ("both", "hlca_only") and hlca_path is not None
    use_luca = mode in ("both", "luca_only") and luca_path is not None

    if not use_hlca and not use_luca:
        print("ERROR: No references available for the selected mode.")
        return 1

    print()
    print("=" * 60)
    print("Dual-Reference Mapping Pipeline")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  Mode: {mode}")
    print(f"  Query: {query_path}")
    print(f"  HLCA: {hlca_path if use_hlca else 'disabled'}")
    print(f"  LuCA: {luca_path if use_luca else 'disabled'}")
    print(f"  Output: {output_dir}")
    print(f"  Mapping method: {mapping_method}")
    print(f"  Fusion method: {fusion_method}")
    print(f"  k-neighbors: {k_neighbors}")
    if smoke_mode:
        print("  SMOKE MODE: Using 1000 cells only")
    print()

    config = ReferenceGeometryConfig(
        hlca_reference_path=str(hlca_path) if use_hlca else None,
        luca_reference_path=str(luca_path) if use_luca else None,
        query_data_path=str(query_path),
        mapping_method=mapping_method,
        k_neighbors=k_neighbors,
        hlca_latent_key=hlca_latent_key,
        luca_latent_key=luca_latent_key,
        fusion_method=fusion_method,
        normalize_fused=True,
        smoke_mode=smoke_mode,
        smoke_n_cells=1000,
    )

    def progress_callback(step: str, pct: float) -> None:
        print(f"  [{pct*100:5.1f}%] {step}")

    result = run_reference_pipeline(
        config,
        run_dir=output_dir,
        run_id=run_id,
        progress_callback=progress_callback,
    )

    print()
    print("=" * 60)
    if result.success:
        print("Dual-Reference Mapping Complete")
    else:
        print("Dual-Reference Mapping FAILED")
    print("=" * 60)
    print(f"  Run ID: {result.run_id}")
    print(f"  Cells: {result.n_cells:,}")
    print(f"  HLCA dim: {result.hlca_dim}")
    print(f"  LuCA dim: {result.luca_dim}")
    print(f"  Fused dim: {result.fused_dim}")
    print(f"  Wall time: {result.wall_time_seconds:.1f}s")
    print(f"  Validation: {result.validation_status}")

    if result.errors:
        print("\nErrors:")
        for err in result.errors:
            print(f"  - {err}")

    if result.warnings:
        print("\nWarnings:")
        for warn in result.warnings:
            print(f"  - {warn}")

    print(f"\nOutputs saved to: {result.output_dir}")
    print("  - hlca_embedding.parquet")
    print("  - luca_embedding.parquet")
    print("  - fused_embedding.parquet")
    print("  - reference_confidence.parquet")
    print("  - reference_manifest.json")

    if result.success:
        print("\nNext step: run_spatial_benchmark.py")
        return 0
    else:
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Map query cells to HLCA and LuCA reference spaces"
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
        "--hlca",
        type=str,
        default=None,
        help="Path to HLCA reference h5ad (auto-detected if not specified)",
    )
    parser.add_argument(
        "--luca",
        type=str,
        default=None,
        help="Path to LuCA reference h5ad (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {data-root}/processed/luad_evo/reference_geometry/)",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--hlca-only",
        action="store_true",
        help="Map to HLCA only (no LuCA)",
    )
    mode_group.add_argument(
        "--luca-only",
        action="store_true",
        help="Map to LuCA only (no HLCA)",
    )

    # Latent keys (in case references use different names)
    parser.add_argument(
        "--hlca-latent-key",
        type=str,
        default="X_scanvi_emb",
        help="Key in HLCA obsm containing latent embeddings",
    )
    parser.add_argument(
        "--luca-latent-key",
        type=str,
        default="X_scVI",
        help="Key in LuCA obsm containing latent embeddings",
    )

    # Mapping parameters
    parser.add_argument(
        "--mapping-method",
        type=str,
        choices=["knn_projection", "pca_projection"],
        default="knn_projection",
        help="Method for mapping query to references",
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        choices=["concat", "average", "weighted"],
        default="concat",
        help="Method for fusing HLCA and LuCA embeddings",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=50,
        help="Number of neighbors for k-NN projection",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode (1000 cells only)",
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
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "processed/luad_evo/reference_geometry"

    # Determine mode
    if args.hlca_only:
        mode = "hlca_only"
    elif args.luca_only:
        mode = "luca_only"
    else:
        mode = "both"

    # Find or use specified reference paths
    if args.hlca:
        hlca_path = Path(args.hlca)
    elif args.luca_only:
        hlca_path = None
    else:
        ref_paths = find_reference_paths(data_root)
        hlca_path = ref_paths["hlca"]

        # If no h5ad but HubModel cache exists, extract reference
        if hlca_path is None and ref_paths["hlca_hub_cache"] is not None:
            print("HLCA h5ad not found, extracting from HubModel cache...")
            hlca_path = data_root / "references/hlca/hlca_reference.h5ad"
            try:
                extract_hlca_reference_from_hub(ref_paths["hlca_hub_cache"], hlca_path)
            except Exception as e:
                print(f"ERROR: Failed to extract HLCA reference: {e}")
                if mode != "luca_only":
                    return 1

    if args.luca:
        luca_path = Path(args.luca)
    elif args.hlca_only:
        luca_path = None
    else:
        ref_paths = find_reference_paths(data_root)
        luca_path = ref_paths["luca"]

    # Validate inputs
    if not snrna_path.exists():
        print(f"ERROR: snRNA file not found: {snrna_path}")
        print("Run run_data_prep.py first.")
        return 1

    if mode == "both":
        if hlca_path is None:
            print("WARNING: HLCA reference not found, falling back to HLCA-only mode")
            print("  Download with: python -m stagebridge.pipelines.download_references --download_hlca")
        if luca_path is None:
            print("WARNING: LuCA reference not found")
            print("  If LuCA is not available, use --hlca-only mode")
            if hlca_path is not None:
                print("  Proceeding with HLCA-only...")
                mode = "hlca_only"
            else:
                return 1

    if hlca_path is not None and not hlca_path.exists():
        print(f"ERROR: HLCA reference not found: {hlca_path}")
        return 1

    if luca_path is not None and not luca_path.exists():
        print(f"ERROR: LuCA reference not found: {luca_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure references use gene symbols (not ENSG IDs) to match query
    if hlca_path is not None:
        reindex_reference_to_symbols(hlca_path)
    if luca_path is not None:
        reindex_reference_to_symbols(luca_path)

    return run_dual_reference_mapping(
        query_path=snrna_path,
        hlca_path=hlca_path,
        luca_path=luca_path,
        output_dir=output_dir,
        mode=mode,
        mapping_method=args.mapping_method,
        fusion_method=args.fusion_method,
        k_neighbors=args.k_neighbors,
        hlca_latent_key=args.hlca_latent_key,
        luca_latent_key=args.luca_latent_key,
        smoke_mode=args.smoke,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
