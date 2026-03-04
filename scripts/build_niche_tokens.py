#!/usr/bin/env python
"""# path: scripts/build_niche_tokens.py
Build niche-token features from Tangram spatial scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stagebridge.io.niche_tokens import (  # noqa: E402
    build_niche_tokens_dataframe,
    write_niche_token_outputs,
    write_tokens_to_h5ad,
)


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spatial_h5ad",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/tangram/spatial_tangram_full.h5ad"),
    )
    parser.add_argument(
        "--scores_parquet",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/tangram/spatial_tangram_celltype_scores.parquet"),
    )
    parser.add_argument(
        "--out_parquet",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/features/niche_tokens_full.parquet"),
    )
    parser.add_argument(
        "--out_h5ad",
        type=Path,
        default=None,
        help="Optional output h5ad with niche-token annotations.",
    )
    parser.add_argument("--k_neighbors", type=int, default=6)
    parser.add_argument("--group_key", type=str, default="sample_id")
    parser.add_argument("--smooth", type=_str2bool, default=True)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = build_niche_tokens_dataframe(
        spatial_h5ad=args.spatial_h5ad,
        scores_parquet=args.scores_parquet,
        group_key=args.group_key,
        smooth=bool(args.smooth),
        k_neighbors=int(args.k_neighbors),
        eps=float(args.eps),
        show_progress=(not bool(args.no_progress)),
    )

    audit_json = args.out_parquet.with_suffix(".audit.json")
    write_niche_token_outputs(
        result=result,
        out_parquet=args.out_parquet,
        out_audit_json=audit_json,
    )

    out_h5ad = args.out_h5ad
    if out_h5ad is not None:
        token_cols = result.smooth_token_columns if result.smooth_token_columns else result.token_columns
        write_tokens_to_h5ad(
            spatial_h5ad=args.spatial_h5ad,
            out_h5ad=out_h5ad,
            tokens_df=result.df,
            token_columns=token_cols,
        )

    token_sum = result.df[result.token_columns].sum(axis=1).to_numpy()
    token_close_fraction = float((abs(token_sum - 1.0) <= 1e-3).mean())
    entropy_q = result.audit["entropy_quantiles"]
    confidence_q = result.audit["confidence_quantiles"]

    summary = {
        "ok": True,
        "out_parquet": str(args.out_parquet),
        "out_audit_json": str(audit_json),
        "out_h5ad": str(out_h5ad) if out_h5ad is not None else None,
        "n_rows": int(result.df.shape[0]),
        "n_cols": int(result.df.shape[1]),
        "n_token_cols": int(len(result.token_columns)),
        "n_smooth_token_cols": int(len(result.smooth_token_columns)),
        "token_sum_within_1pm1e-3_fraction": token_close_fraction,
        "entropy_quantile_spread_q95_q05": float(entropy_q["q95"] - entropy_q["q05"]),
        "confidence_q95": float(confidence_q["q95"]),
    }

    if args.json_output:
        print(json.dumps(summary))
    else:
        print(f"niche_tokens_parquet={args.out_parquet}")
        print(f"audit_json={audit_json}")
        if out_h5ad is not None:
            print(f"out_h5ad={out_h5ad}")
        print(f"rows={summary['n_rows']} cols={summary['n_cols']}")
        print(f"token_sum_within_1pm1e-3_fraction={summary['token_sum_within_1pm1e-3_fraction']:.6f}")
        print(f"entropy_spread_q95_q05={summary['entropy_quantile_spread_q95_q05']:.6f}")
        print(f"confidence_q95={summary['confidence_q95']:.6f}")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
