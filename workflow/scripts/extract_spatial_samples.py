#!/usr/bin/env python3
"""Extract sample IDs from spatial h5ad and write to manifest file."""

import argparse
import json
from pathlib import Path

import anndata as ad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", required=True, help="Path to spatial h5ad")
    parser.add_argument("--output", required=True, help="Output manifest JSON")
    parser.add_argument("--sample-col", default="sample_id", help="Sample column name")
    args = parser.parse_args()

    print(f"Loading spatial data from {args.spatial}...")
    spatial = ad.read_h5ad(args.spatial)

    if args.sample_col not in spatial.obs.columns:
        raise ValueError(f"Column '{args.sample_col}' not found. Available: {spatial.obs.columns.tolist()}")

    samples = sorted(spatial.obs[args.sample_col].unique().tolist())
    print(f"Found {len(samples)} samples")

    # Write manifest
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "sample_col": args.sample_col,
        "n_samples": len(samples),
        "samples": samples,
    }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest to {output_path}")


if __name__ == "__main__":
    main()
