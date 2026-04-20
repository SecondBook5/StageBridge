#!/usr/bin/env python3
"""Validate donor-held-out splits - Snakemake wrapper.

Core logic in stagebridge.validation.splits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stagebridge.validation.splits import validate_splits_from_files


def main():
    parser = argparse.ArgumentParser(description="Validate donor-held-out splits")
    parser.add_argument("--cells", type=str, required=True, help="cells.parquet path")
    parser.add_argument("--splits", type=str, required=True, help="split_manifest.json path")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Validating splits...")
    results = validate_splits_from_files(
        cells_path=Path(args.cells),
        splits_path=Path(args.splits),
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Validation report saved to: {output_path}")

    if results["valid"]:
        print("[PASS] All splits validated - no donor leakage detected")
    else:
        print(f"[FAIL] Validation failed with {len(results['issues'])} issues")
        for issue in results["issues"]:
            print(f"  - {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
