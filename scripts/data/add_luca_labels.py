#!/usr/bin/env python3
"""Add LuCA cell type labels to snRNA data and re-run LIANA with 33 cell types.

Uses stagebridge.reference.transfer_labels and stagebridge.biology.run_liana APIs.

Run on HPC with GPU:
    srun --partition=gpu --gres=gpu:1 --mem=128G --time=4:00:00 --account=chaunzt1 --pty bash
    python scripts/add_luca_labels.py --input /path/to/snrna.h5ad --output-dir /path/to/output
"""

import argparse
from pathlib import Path

import scanpy as sc

from stagebridge.reference import transfer_labels
from stagebridge.biology import run_liana, extract_il1b_interactions


def main():
    parser = argparse.ArgumentParser(description="Add LuCA labels and run LIANA")
    parser.add_argument("--input", type=Path, required=True, help="Input h5ad file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--luca-model", type=Path, default=None, help="LuCA model path")
    parser.add_argument("--skip-liana", action="store_true", help="Skip LIANA analysis")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Adding LuCA cell type labels")
    print("=" * 60)

    print(f"\nLoading {args.input}...")
    adata = sc.read_h5ad(args.input)
    print(f"  {adata.n_obs:,} cells")

    # Check if LuCA labels already exist
    luca_cols = [c for c in adata.obs.columns if "luca" in c.lower()]
    if luca_cols:
        print(f"  Found existing LuCA columns: {luca_cols}")
        cell_type_col = luca_cols[0]
    else:
        print("\nTransferring LuCA labels...")
        adata = transfer_labels(
            adata,
            reference="luca",
            model_dir=args.luca_model,
            inplace=True,
        )
        cell_type_col = "cell_type_luca"

        # Save updated h5ad
        out_path = args.output_dir / "snrna_with_luca_labels.h5ad"
        print(f"Saving to {out_path}...")
        adata.write_h5ad(out_path)

    print(f"\nLuCA cell types ({adata.obs[cell_type_col].nunique()} unique):")
    print(adata.obs[cell_type_col].value_counts().head(20))

    if not args.skip_liana:
        print("\n" + "=" * 60)
        print("Running LIANA with LuCA cell types...")
        print("=" * 60)

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

        # Extract IL1B interactions
        il1b = extract_il1b_interactions(lr_results)
        print(f"  {len(il1b)} IL1B-related interactions")

        # Save
        out_liana = args.output_dir / "liana_interactions_luca.parquet"
        lr_results.to_parquet(out_liana)
        print(f"\nSaved LIANA results to {out_liana}")

        if len(il1b) > 0:
            out_il1b = args.output_dir / "il1b_interactions.parquet"
            il1b.to_parquet(out_il1b)
            print(f"Saved IL1B interactions to {out_il1b}")

    print("\nDone!")


if __name__ == "__main__":
    main()
