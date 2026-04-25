#!/usr/bin/env python3
"""Relabel cells.parquet from 5-stage to 3-stage for training.

3-stage mapping:
  - Normal -> Normal
  - AAH, AIS, MIA -> Preinvasive
  - LUAD -> Invasive
"""
import argparse
from pathlib import Path
import pandas as pd
import json


STAGE_MAP = {
    'Normal': 'Normal',
    'AAH': 'Preinvasive',
    'AIS': 'Preinvasive',
    'MIA': 'Preinvasive',
    'LUAD': 'Invasive',
}

STAGE_3_ORDER = ['Normal', 'Preinvasive', 'Invasive']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cells.parquet...")
    cells_df = pd.read_parquet(input_dir / "cells.parquet")

    # Keep original 5-stage as backup
    cells_df['stage_5'] = cells_df['stage']

    # Relabel to 3-stage
    cells_df['stage'] = cells_df['stage_5'].map(STAGE_MAP)
    cells_df['stage_idx'] = cells_df['stage'].map({s: i for i, s in enumerate(STAGE_3_ORDER)})

    print(f"Stage distribution:")
    print(cells_df['stage'].value_counts())

    # Save
    cells_df.to_parquet(output_dir / "cells.parquet", index=False)
    print(f"Saved {len(cells_df)} cells to {output_dir / 'cells.parquet'}")

    # Copy neighborhoods unchanged
    print("Copying neighborhoods.parquet...")
    neighborhoods_df = pd.read_parquet(input_dir / "neighborhoods.parquet")
    neighborhoods_df.to_parquet(output_dir / "neighborhoods.parquet", index=False)

    # Update split manifest
    print("Copying split_manifest.json...")
    with open(input_dir / "split_manifest.json") as f:
        splits = json.load(f)

    # Update metadata
    splits['stages'] = STAGE_3_ORDER
    splits['n_stages'] = 3
    splits['stage_mapping'] = STAGE_MAP

    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Copy other files if they exist
    for fname in ["stage_edges.parquet", "data_manifest.json"]:
        src = input_dir / fname
        if src.exists():
            if fname == "data_manifest.json":
                with open(src) as f:
                    manifest = json.load(f)
                manifest['n_stages'] = 3
                manifest['stages'] = STAGE_3_ORDER
                with open(output_dir / fname, "w") as f:
                    json.dump(manifest, f, indent=2)
            else:
                import shutil
                shutil.copy(src, output_dir / fname)
            print(f"Copied {fname}")

    print("\nDone. 3-stage canonical data ready.")
    print(f"  Normal: {(cells_df['stage'] == 'Normal').sum()}")
    print(f"  Preinvasive: {(cells_df['stage'] == 'Preinvasive').sum()}")
    print(f"  Invasive: {(cells_df['stage'] == 'Invasive').sum()}")


if __name__ == "__main__":
    main()
