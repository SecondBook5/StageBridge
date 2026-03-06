#!/usr/bin/env python
"""# path: scripts/run_full_step_after_tangram.py
One-command post-Tangram pipeline smoke: tokens -> bank -> QC -> training smoke.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_config", type=str, default="local")
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
        "--out_tokens_parquet",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/features/niche_tokens_full.parquet"),
    )
    parser.add_argument(
        "--out_tokens_h5ad",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/anndata/spatial/spatial_tangram_with_tokens_full.h5ad"),
    )
    parser.add_argument("--write_tokens_h5ad", action="store_true")
    parser.add_argument(
        "--out_token_bank",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/features/niche_token_bank.zarr"),
    )
    parser.add_argument("--qc_sample_id", type=str, default="")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/post_tangram_smoke"))
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path) -> list[str]:
    print("$ " + " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")
    return lines


def _last_json_line(lines: list[str]) -> dict | None:
    for raw in reversed(lines):
        text = raw.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    return None


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent.parent
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict | None] = {}

    build_tokens_cmd = [
        sys.executable,
        "scripts/build_niche_tokens.py",
        "--spatial_h5ad",
        str(args.spatial_h5ad),
        "--scores_parquet",
        str(args.scores_parquet),
        "--out_parquet",
        str(args.out_tokens_parquet),
        "--smooth",
        "true",
        "--k_neighbors",
        "6",
        "--json",
    ]
    if args.write_tokens_h5ad:
        build_tokens_cmd.extend(["--out_h5ad", str(args.out_tokens_h5ad)])
    lines = _run(build_tokens_cmd, cwd=repo_root)
    summaries["build_niche_tokens"] = _last_json_line(lines)

    lines = _run(
        [
            sys.executable,
            "scripts/build_niche_token_bank.py",
            "--niche_tokens_parquet",
            str(args.out_tokens_parquet),
            "--out_zarr",
            str(args.out_token_bank),
            "--json",
        ],
        cwd=repo_root,
    )
    summaries["build_niche_token_bank"] = _last_json_line(lines)

    qc_sample = args.qc_sample_id.strip()
    qc_cmd = [
        sys.executable,
        "scripts/qc_niche_tokens.py",
        "--niche_tokens_parquet",
        str(args.out_tokens_parquet),
        "--spatial_h5ad",
        str(args.spatial_h5ad),
        "--out_dir",
        str(repo_root / "outputs" / "figures" / "niche_tokens"),
        "--json",
    ]
    if qc_sample:
        qc_cmd.extend(["--sample_id", qc_sample])
    lines = _run(qc_cmd, cwd=repo_root)
    qc_summary = _last_json_line(lines)
    summaries["qc_niche_tokens"] = qc_summary

    lines = _run(
        [
            sys.executable,
            "scripts/train_stagebridge.py",
            f"data={args.data_config}",
            "run_name=post_tangram_smoke",
            f"output_dir={out_dir}",
            "data.max_cells=50000",
            "training.max_epochs=1",
            "training.steps_per_epoch=10",
            "training.val_steps=1",
            "training.batch_cells=128",
            "training.num_ot_pairs=128",
            "training.device=cpu",
            "training.mixed_precision=false",
            "splits.n_folds=2",
            "experiment.baseline_models=[deepsets]",
            "experiment.ablations=[no_context]",
            "training.transition_src=AIS",
            "training.transition_tgt=MIA",
            "training.use_niche_tokens=true",
            f"training.niche_token_bank_path={args.out_token_bank}",
            "training.m_niche=64",
            "training.niche_sampling_strategy=random_m",
        ],
        cwd=repo_root,
    )
    summaries["train_stagebridge_smoke"] = _last_json_line(lines)

    summary = {
        "ok": True,
        "steps": summaries,
        "out_tokens_parquet": str(args.out_tokens_parquet),
        "out_token_bank": str(args.out_token_bank),
        "training_output_dir": str(out_dir),
    }
    if args.json_output:
        print(json.dumps(summary))
    else:
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
