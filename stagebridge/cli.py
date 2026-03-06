"""Unified package CLI for StageBridge workflows."""
from __future__ import annotations

import argparse
import json
from typing import Any

from stagebridge.notebook_api import available_steps, compose_config, run_pipeline, run_step


def _default_entrypoint_for_step(step: str) -> str:
    if step in {"train", "train_stagebridge"}:
        return "train"
    if step in {"evaluate", "eval_stagebridge"}:
        return "eval"
    return "config"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StageBridge unified CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_step = sub.add_parser("step", help="Run one workflow step")
    p_step.add_argument("name", choices=available_steps())
    p_step.add_argument("-c", "--config", dest="entrypoint", default=None)
    p_step.add_argument("-o", "--override", action="append", default=[])

    p_pipe = sub.add_parser("pipeline", help="Run multiple workflow steps")
    p_pipe.add_argument("--steps", default="build_snrna,build_spatial,map_hlca,run_tangram,train")
    p_pipe.add_argument("-c", "--config", dest="entrypoint", default="config")
    p_pipe.add_argument("-o", "--override", action="append", default=[])

    p_train = sub.add_parser("train", help="Run training workflow")
    p_train.add_argument("-o", "--override", action="append", default=[])

    p_eval = sub.add_parser("evaluate", help="Run evaluation workflow")
    p_eval.add_argument("-o", "--override", action="append", default=[])

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
        cfg = compose_config(entrypoint="train", overrides=args.override)
        _print(run_step("train", cfg))
        return 0

    if args.command == "evaluate":
        cfg = compose_config(entrypoint="eval", overrides=args.override)
        _print(run_step("evaluate", cfg))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
