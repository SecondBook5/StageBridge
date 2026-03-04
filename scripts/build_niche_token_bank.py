#!/usr/bin/env python
"""# path: scripts/build_niche_token_bank.py
Build a Zarr token bank for fast niche-token sampling during training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stagebridge.io.niche_tokens import (  # noqa: E402
    NicheTokenBank,
    build_zarr_token_bank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--niche_tokens_parquet",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/features/niche_tokens_full.parquet"),
    )
    parser.add_argument(
        "--out_zarr",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/features/niche_token_bank.zarr"),
    )
    parser.add_argument("--group_key", type=str, default="sample_id")
    parser.add_argument("--token_prefix", type=str, default="tok_smooth_")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = build_zarr_token_bank(
        niche_tokens_parquet=args.niche_tokens_parquet,
        output_zarr=args.out_zarr,
        group_key=args.group_key,
        token_prefix=args.token_prefix,
    )

    bank = NicheTokenBank(args.out_zarr)
    first_sample = result.sample_ids[0] if result.sample_ids else None
    sample_read_seconds = None
    sample_shape = None
    if first_sample is not None:
        meta = bank.sample_meta[first_sample]
        donor = str(meta.get("donor_id", "unknown_donor"))
        stage = str(meta.get("stage", "Unknown"))
        t0 = time.perf_counter()
        arr, _ = bank.sample_tokens(
            donor_id=donor,
            stage=stage,
            m_niche=64,
            strategy="random_m",
            fallback="donor",
        )
        sample_read_seconds = float(time.perf_counter() - t0)
        sample_shape = [int(arr.shape[0]), int(arr.shape[1])]

    audit = dict(result.audit)
    audit["sample_read_benchmark_seconds"] = sample_read_seconds
    audit["sample_read_shape"] = sample_shape
    audit_path = args.out_zarr.parent / f"{args.out_zarr.stem}.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    summary = {
        "ok": True,
        "out_zarr": str(result.output_path),
        "audit_json": str(audit_path),
        "n_groups": int(result.n_groups),
        "n_rows": int(result.n_rows),
        "token_dim": int(len(result.token_columns)),
        "sample_read_seconds": sample_read_seconds,
    }
    if args.json_output:
        print(json.dumps(summary))
    else:
        print(f"out_zarr={result.output_path}")
        print(f"audit_json={audit_path}")
        print(f"n_groups={result.n_groups} n_rows={result.n_rows} token_dim={len(result.token_columns)}")
        if sample_read_seconds is not None:
            print(f"sample_read_seconds={sample_read_seconds:.6f}")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
