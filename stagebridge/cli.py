"""Unified package CLI for the rebuilt StageBridge pipeline surface."""

from __future__ import annotations

import argparse
import json
from typing import Any

from stagebridge.notebook_api import available_steps, compose_config, run_pipeline, run_step


def _default_entrypoint_for_step(step: str) -> str:
    if step in {"transition_model", "train"}:
        return "default"
    if step in {"evaluation", "evaluate"}:
        return "default"
    return "default"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StageBridge unified CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_step = sub.add_parser("step", help="Run one workflow step")
    p_step.add_argument("name", choices=available_steps())
    p_step.add_argument("-c", "--config", dest="entrypoint", default=None)
    p_step.add_argument("-o", "--override", action="append", default=[])

    p_pipe = sub.add_parser("pipeline", help="Run multiple workflow steps")
    p_pipe.add_argument(
        "--steps",
        default="reference,spatial_mapping,context_model,transition_model,evaluation",
    )
    p_pipe.add_argument("-c", "--config", dest="entrypoint", default="default")
    p_pipe.add_argument("-o", "--override", action="append", default=[])

    p_train = sub.add_parser("train", help="Run training workflow")
    p_train.add_argument("-o", "--override", action="append", default=[])

    p_eval = sub.add_parser("evaluate", help="Run evaluation workflow")
    p_eval.add_argument("-o", "--override", action="append", default=[])

    p_data = sub.add_parser("data-prep", help="Run raw data preparation (Step 0)")
    p_data.add_argument(
        "--data-root", type=str, default=None, help="Override STAGEBRIDGE_DATA_ROOT"
    )
    p_data.add_argument("--force", action="store_true", help="Force re-processing")
    p_data.add_argument("--skip-qc", action="store_true", help="Skip QC filtering")
    p_data.add_argument("--skip-normalization", action="store_true", help="Skip normalization")

    return parser


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "step":
        entrypoint = args.entrypoint or _default_entrypoint_for_step(args.name)
        cfg = compose_config(entrypoint=entrypoint, overrides=args.override)
        _print(run_step(args.name, cfg))
        return 0

    if args.command == "pipeline":
        steps = [s.strip() for s in str(args.steps).split(",") if s.strip()]
        cfg = compose_config(entrypoint=args.entrypoint, overrides=args.override)
        _print(run_pipeline(cfg, steps=steps))
        return 0

    if args.command == "train":
        cfg = compose_config(entrypoint="default", overrides=args.override)
        _print(run_step("transition_model", cfg))
        return 0

    if args.command == "evaluate":
        cfg = compose_config(entrypoint="default", overrides=args.override)
        _print(run_step("evaluation", cfg))
        return 0

    if args.command == "data-prep":
        from stagebridge.pipelines.run_data_prep import run_data_prep

        result = run_data_prep(
            data_root=args.data_root,
            force=args.force,
            skip_qc=args.skip_qc,
            skip_normalization=args.skip_normalization,
        )
        _print(result)
        return 0 if result.get("ok") else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
