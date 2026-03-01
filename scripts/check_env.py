#!/usr/bin/env python
"""Environment validation for StageBridge reproducibility gates.

Usage:
    python scripts/check_env.py                  # human-readable output (default)
    python scripts/check_env.py --json           # machine-readable JSON
    python scripts/check_env.py --no-gpu         # skip CUDA checks (CPU CI)
    python scripts/check_env.py --output <path>  # save JSON report to path
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REQUIRED_MODULES = [
    "torch",
    "hydra",
    "omegaconf",
    "anndata",
    "scanpy",
    "squidpy",
    "harmonypy",
    "scipy",
    "numpy",
    "pandas",
    "matplotlib",
    "tqdm",
]

OPTIONAL_MODULES = [
    "scvi",
    "tangram",
    "umap",
    "sklearn",
]


def _module_check(name: str) -> dict[str, object]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        return {"name": name, "ok": True, "version": str(version)}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


def _torch_gpu_check() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": f"torch import failed: {exc}"}

    available = bool(torch.cuda.is_available())
    payload: dict[str, object] = {
        "ok": True,
        "cuda_available": available,
        "cuda_device_count": int(torch.cuda.device_count()) if available else 0,
        "torch_version": torch.__version__,
    }
    if available:
        payload["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        payload["cuda_version"] = torch.version.cuda
    return payload


def _stagebridge_check() -> dict[str, object]:
    """Verify stagebridge package imports and stage ontology is lung cancer."""
    try:
        from stagebridge.models.stagebridge import StageBridgeModel
        from stagebridge.training.trainer import StageBridgeTrainer
        from stagebridge.training.losses import build_sinkhorn_coupling
        from stagebridge.preprocessing.stage_ontology import CANONICAL_STAGE_ORDER
        stages = list(CANONICAL_STAGE_ORDER)
        expected = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
        if stages != expected:
            return {
                "ok": False,
                "error": f"Stage ontology mismatch: got {stages}, expected {expected}",
            }
        return {"ok": True, "canonical_stages": stages}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _smoke_forward_check() -> dict[str, object]:
    """Minimal CPU forward pass to catch shape/import errors early."""
    try:
        import torch
        from stagebridge.models.stagebridge import StageBridgeModel

        model = StageBridgeModel(
            input_dim=16, hidden_dim=32, vector_field_hidden_dim=64,
            num_heads=2, num_inducing_points=4, num_seed_vectors=1,
            num_stages=5, time_embedding_dim=16, stage_embedding_dim=16, dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            src = torch.randn(1, 8, 16)
            x_t = torch.randn(4, 16)
            t = torch.rand(4)
            stage_pair = torch.zeros(1, dtype=torch.long)
            out = model(src_set=src, x_t=x_t, t=t, stage_pair_id=stage_pair)
        if tuple(out.shape) != (4, 16):
            return {"ok": False, "error": f"Output shape {tuple(out.shape)} != (4, 16)"}
        return {"ok": True, "output_shape": list(out.shape)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _config_check(config_dir: Path) -> dict[str, object]:
    expected = [
        config_dir / "train.yaml",
        config_dir / "eval.yaml",
        config_dir / "model" / "stagebridge.yaml",
        config_dir / "training" / "default.yaml",
        config_dir / "experiment" / "full_benchmark.yaml",
        config_dir / "splits" / "donor_holdout.yaml",
    ]
    missing = [str(p) for p in expected if not p.exists()]

    parse_ok = True
    parse_errors: list[str] = []
    try:
        from omegaconf import OmegaConf
        for p in expected:
            if p.exists():
                try:
                    OmegaConf.load(p)
                except Exception as exc:
                    parse_ok = False
                    parse_errors.append(f"{p.name}: {exc}")
    except Exception as exc:
        parse_ok = False
        parse_errors.append(f"OmegaConf unavailable: {exc}")

    return {
        "ok": len(missing) == 0 and parse_ok,
        "missing": missing,
        "parse_errors": parse_errors,
    }


def _print_human(report: dict) -> None:
    """Print a clear human-readable summary."""
    ok_sym = "✓"
    fail_sym = "✗"
    skip_sym = "○"
    line = "=" * 60

    print(f"\n{line}")
    print("  StageBridge Environment Check")
    print(line)
    print(f"  Python : {report['python']['version'].split()[0]}")
    print(f"  Platform: {report['python']['platform']}")

    def _check_row(check: dict, label: str | None = None) -> None:
        ok = check.get("ok", False)
        sym = ok_sym if ok else fail_sym
        name = label or check.get("name", "?")
        ver = check.get("version", "")
        err = check.get("error", "")
        tail = f"  [{ver}]" if ver else (f"  — {err}" if err else "")
        print(f"  {sym}  {name}{tail}")

    print("\n  [Required modules]")
    for c in report["checks"]["modules"]:
        _check_row(c)

    print("\n  [StageBridge package]")
    sb = report["checks"]["stagebridge"]
    if sb.get("ok"):
        stages = sb.get("canonical_stages", [])
        print(f"  {ok_sym}  stagebridge  stages={stages}")
    else:
        print(f"  {fail_sym}  stagebridge  — {sb.get('error', '')}")

    print("\n  [Smoke forward pass]")
    sf = report["checks"]["smoke_forward"]
    if sf.get("ok"):
        print(f"  {ok_sym}  CPU forward pass  shape={sf.get('output_shape')}")
    else:
        print(f"  {fail_sym}  CPU forward pass  — {sf.get('error', '')}")

    if "gpu" in report["checks"]:
        print("\n  [GPU / CUDA]")
        g = report["checks"]["gpu"]
        if g.get("ok") and g.get("cuda_available"):
            devices = g.get("cuda_devices", [])
            print(f"  {ok_sym}  CUDA {g.get('cuda_version', '?')}  —  {devices}")
        elif g.get("ok"):
            print(f"  {skip_sym}  CUDA not available (CPU-only build)")
        else:
            print(f"  {fail_sym}  {g.get('error', 'CUDA check failed')}")

    print("\n  [Config files]")
    cfg = report["checks"]["configs"]
    if cfg.get("ok"):
        print(f"  {ok_sym}  All required config files present and parseable")
    else:
        for m in cfg.get("missing", []):
            print(f"  {fail_sym}  missing: {m}")
        for e in cfg.get("parse_errors", []):
            print(f"  {fail_sym}  parse error: {e}")

    if "optional" in report["checks"]:
        print("\n  [Optional modules]")
        for c in report["checks"]["optional"]:
            sym = ok_sym if c.get("ok") else skip_sym
            ver = c.get("version", "")
            err = c.get("error", "not installed")
            tail = f"  [{ver}]" if ver else f"  — {err}"
            print(f"  {sym}  {c['name']}{tail}")

    print(f"\n{line}")
    if report["ok"]:
        print("  RESULT: ALL REQUIRED CHECKS PASSED")
    else:
        print("  RESULT: FAILED — see above for details")
    print(f"{line}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate StageBridge environment and config integrity")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON (default: human-readable)")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU/CUDA checks")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/env_check.json"),
        help="Save JSON report to this path.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_dir = repo_root / "configs"

    module_checks = [_module_check(name) for name in REQUIRED_MODULES]
    optional_checks = [_module_check(name) for name in OPTIONAL_MODULES]
    stagebridge_check = _stagebridge_check()
    smoke_check = _smoke_forward_check()
    cfg_check = _config_check(config_dir)

    checks: dict[str, object] = {
        "modules": module_checks,
        "stagebridge": stagebridge_check,
        "smoke_forward": smoke_check,
        "configs": cfg_check,
        "optional": optional_checks,
    }
    if not args.no_gpu:
        checks["gpu"] = _torch_gpu_check()

    ok = (
        all(c["ok"] for c in module_checks)
        and bool(stagebridge_check.get("ok"))
        and bool(smoke_check.get("ok"))
        and bool(cfg_check.get("ok"))
    )

    report = {
        "ok": ok,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "checks": checks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    if args.json:
        print(json.dumps({"ok": ok, "output": str(args.output)}, indent=2))
    else:
        _print_human(report)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
