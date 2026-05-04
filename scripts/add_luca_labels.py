#!/usr/bin/env python3
"""Add LuCA cell type labels to snRNA data and re-run LIANA with 33 cell types.

Run on HPC with GPU:
    srun --partition=gpu --gres=gpu:1 --mem=128G --time=4:00:00 --account=chaunzt1 --pty bash
    python scripts/add_luca_labels.py
"""

import scanpy as sc
import pandas as pd
from pathlib import Path

# Paths - adjust as needed
DATA = Path("/data1/chaunzt1/stagebridge/processed/luad_evo")
SNRNA = DATA / "snrna_with_celltypes.h5ad"
LUCA_MODEL = Path("/data1/chaunzt1/stagebridge/references/luca/model")
LUCA_REF = Path("/data1/chaunzt1/stagebridge/references/luca/luca_core.h5ad")
OUTPUT = DATA / "canonical"


def main():
    print("=" * 60)
    print("Adding LuCA cell type labels")
    print("=" * 60)

    # Load data
    print(f"\nLoading {SNRNA}...")
    adata = sc.read_h5ad(SNRNA)
    print(f"  {adata.n_obs:,} cells")
    print(f"  Existing columns: {list(adata.obs.columns)}")

    # Check if LuCA labels already exist
    luca_cols = [c for c in adata.obs.columns if 'luca' in c.lower()]
    if luca_cols:
        print(f"\n  Found existing LuCA columns: {luca_cols}")
        print("  Using existing labels, skipping mapping...")
        cell_type_col = luca_cols[0]
    else:
        # Run LuCA label transfer
        print("\nRunning LuCA label transfer...")
        from stagebridge.reference.mapper import ReferenceMapper

        if not LUCA_MODEL.exists():
            print(f"  ERROR: LuCA model not found at {LUCA_MODEL}")
            print("  Please download or specify correct path")
            return

        mapper = ReferenceMapper(
            luca_model_dir=LUCA_MODEL,
            luca_ref_path=LUCA_REF if LUCA_REF.exists() else None,
            surgery_epochs=200,
        )

        result = mapper.map_to_luca(adata, return_labels=True, return_probs=True)

        # Add labels to adata
        adata.obs["cell_type_luca"] = result.labels
        if result.confidence is not None:
            adata.obs["cell_type_luca_confidence"] = result.confidence
        if result.entropy is not None:
            adata.obs["cell_type_luca_entropy"] = result.entropy

        # Save updated h5ad
        out_path = DATA / "snrna_with_celltypes_luca.h5ad"
        print(f"\nSaving to {out_path}...")
        adata.write_h5ad(out_path)

        cell_type_col = "cell_type_luca"

    # Show cell type distribution
    print(f"\nLuCA cell types ({adata.obs[cell_type_col].nunique()} unique):")
    print(adata.obs[cell_type_col].value_counts().head(20))

    # Re-run LIANA with LuCA labels
    print("\n" + "=" * 60)
    print("Running LIANA with LuCA cell types...")
    print("=" * 60)

    from stagebridge.biology.features import run_liana

    lr_results = run_liana(
        adata,
        cell_type_col=cell_type_col,
        resource="consensus",
        expr_prop=0.1,
        n_perms=100,
        verbose=True,
    )

    print(f"\n  {len(lr_results):,} interactions")
    print(f"  {lr_results['source'].nunique()} source cell types")
    print(f"  {lr_results['target'].nunique()} target cell types")

    # Save
    out_liana = OUTPUT / "liana_interactions_luca.parquet"
    lr_results.to_parquet(out_liana)
    print(f"\nSaved to {out_liana}")

    # Also copy to data/ for figures
    local_copy = Path("/home/booka/projects/StageBridge/data/liana_interactions_luca.parquet")
    lr_results.to_parquet(local_copy)
    print(f"Saved local copy to {local_copy}")

    print("\nDone!")


if __name__ == "__main__":
    main()
