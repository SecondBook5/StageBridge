#!/usr/bin/env python
"""
Environment validation for StageBridge reproducibility gates.

Design principles
- Fail fast on missing *core* runtime modules.
- Treat nice-to-have modules as optional (report but do not fail).
- Verify CUDA when requested (skip-able for CPU CI).
- Verify the repo imports as a package (prefer editable install but tolerate sys.path injection).
- Verify the stage ontology is what the project claims (Normal → AAH → AIS → MIA → LUAD).
- Keep the smoke forward pass resilient to minor API drift by:
    (1) instantiating from Hydra configs first
    (2) inspecting model.forward signature and supplying args via a synonym map

Usage:
    python scripts/check_env.py                  # human-readable output (default)
    python scripts/check_env.py --json           # machine-readable JSON summary
    python scripts/check_env.py --no-gpu         # skip CUDA checks (CPU CI)
    python scripts/check_env.py --output <path>  # save JSON report to path
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo root to sys.path so this script can run even without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Define modules that must exist for the core pipeline.
# Note: module import names differ from package names (e.g., POT imports as 'ot').
REQUIRED_MODULES: List[str] = [
    # Core ML runtime.
    "torch",
    # Config stack.
    "hydra",
    "omegaconf",
    "yaml",
    # Scientific stack.
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    # Single-cell + spatial.
    "anndata",
    "scanpy",
    "squidpy",
    # OT utilities used by training/eval.
    "ot",
    "geomloss",
    # Embeddings + nearest neighbors (UMAP plots + manifolds).
    "umap",
    "pynndescent",
    "numba",
    # Poster/figure export (Sankey/interactive + static export).
    "plotly",
    "kaleido",
    # IO used in workflows (manifests, tables, big artifacts).
    "h5py",
    "pyarrow",
    "zarr",
    "numcodecs",
    # HLCA reference mapping via scvi-tools hub.
    "scvi",
    "huggingface_hub",
    # snRNA → spatial mapping.
    "tangram",
]

# Define modules that are useful but not required for the core to run.
OPTIONAL_MODULES: List[str] = [
    # Batch correction (optional if HLCA/scANVI is the main alignment tool).
    "harmonypy",
    # Image stack (only required if you run Squidpy image features/QC).
    "skimage",
    "PIL",
    "cv2",
    # Extra QC / trajectory tools (nice, not required).
    "scrublet",
    "cellrank",
    "scvelo",
]


def _module_check(name: str) -> Dict[str, Any]:
    """
    Import a module by name and return a structured status dict.

    Why this is written this way:
    - Many scientific Python packages are deprecating module.__version__.
    - The most stable way to read versions is importlib.metadata.version(<dist-name>).
    - Import names often differ from distribution names (e.g., yaml -> PyYAML).
    """
    try:
        # Import the module (this is the true availability check).
        importlib.import_module(name)

        # Map import names to distribution names when they differ.
        dist_name_map: Dict[str, str] = {
            # YAML
            "yaml": "PyYAML",
            # POT
            "ot": "POT",
            # scikit-learn
            "sklearn": "scikit-learn",
            # Pillow
            "PIL": "Pillow",
            # OpenCV (conda-forge)
            "cv2": "opencv",
            # Kaleido (conda-forge binding name)
            "kaleido": "python-kaleido",
            # Hydra import is "hydra" but distribution is hydra-core
            "hydra": "hydra-core",
            # UMAP import is "umap" but distribution is umap-learn
            "umap": "umap-learn",
            # scvi import is "scvi" but distribution is scvi-tools
            "scvi": "scvi-tools",
            # tangram import is "tangram" but distribution is tangram-sc
            "tangram": "tangram-sc",
        }

        # Resolve distribution name for metadata lookup.
        dist_name = dist_name_map.get(name, name)

        # Try to read distribution version.
        try:
            version = metadata.version(dist_name)
        except Exception:
            version = "unknown"

        return {"name": name, "ok": True, "version": str(version)}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


def _torch_gpu_check() -> Dict[str, Any]:
    """Validate torch CUDA availability and enumerate devices if present."""
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": f"torch import failed: {exc}"}

    available = bool(torch.cuda.is_available())
    payload: Dict[str, Any] = {
        "ok": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": available,
        "cuda_device_count": int(torch.cuda.device_count()) if available else 0,
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if available:
        payload["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return payload


def _stagebridge_import_check(repo_root: Path) -> Dict[str, Any]:
    """Verify stagebridge imports and that the import resolves to this repo."""
    try:
        import stagebridge as sb  # noqa: F401

        sb_path = Path(getattr(sb, "__file__", "")).resolve()
        in_repo = str(sb_path).startswith(str((repo_root / "stagebridge").resolve()))

        return {
            "ok": True,
            "stagebridge_file": str(sb_path),
            "resolves_to_repo": bool(in_repo),
            "note": (
                "stagebridge resolves to repo source (good)"
                if in_repo
                else "stagebridge imported, but not from repo source — run `pip install -e .`"
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _notebook_check(repo_root: Path) -> Dict[str, Any]:
    """Verify that the canonical notebook entrypoint exists."""
    nb_path = repo_root / "StageBridge.ipynb"
    if not nb_path.exists():
        return {"ok": False, "error": f"Missing notebook entrypoint: {nb_path}"}
    if nb_path.stat().st_size < 200:
        return {"ok": False, "error": f"Notebook exists but looks too small: {nb_path} ({nb_path.stat().st_size} bytes)"}
    return {"ok": True, "path": str(nb_path)}


def _stage_ontology_check() -> Dict[str, Any]:
    """Verify that the stage ontology matches the intended lung progression ordering."""
    expected = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    try:
        from stagebridge.preprocessing.stage_ontology import CANONICAL_STAGE_ORDER  # type: ignore

        stages = list(CANONICAL_STAGE_ORDER)
        if stages != expected:
            return {"ok": False, "error": f"Stage ontology mismatch: got {stages}, expected {expected}"}
        return {"ok": True, "canonical_stages": stages}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _find_first_target(cfg: Any) -> Optional[Dict[str, Any]]:
    """Recursively search a nested dict/list config for the first dict containing a Hydra '_target_'."""
    if isinstance(cfg, dict):
        if "_target_" in cfg:
            return cfg
        for v in cfg.values():
            found = _find_first_target(v)
            if found is not None:
                return found
    if isinstance(cfg, list):
        for v in cfg:
            found = _find_first_target(v)
            if found is not None:
                return found
    return None


def _instantiate_model_from_cfg(repo_root: Path) -> Tuple[bool, Any, str]:
    """
    Instantiate the StageBridge model from Hydra configs (source of truth).

    Strategy:
    - Search likely config files for a '_target_' dict:
        1) configs/model/stagebridge.yaml
        2) configs/experiment/smoke.yaml
        3) configs/experiment/full_benchmark.yaml
    - Instantiate the first found target dict via hydra.utils.instantiate().
    """
    try:
        from omegaconf import OmegaConf
    except Exception as exc:
        return False, None, f"OmegaConf import failed: {exc}"

    try:
        from hydra.utils import instantiate
    except Exception as exc:
        return False, None, f"hydra.utils.instantiate import failed: {exc}"

    config_candidates: List[Path] = [
        repo_root / "configs" / "model" / "stagebridge.yaml",
        repo_root / "configs" / "experiment" / "smoke.yaml",
        repo_root / "configs" / "experiment" / "full_benchmark.yaml",
    ]

    errors: List[str] = []

    for cfg_path in config_candidates:
        if not cfg_path.exists():
            errors.append(f"missing: {cfg_path}")
            continue

        try:
            cfg = OmegaConf.load(cfg_path)
        except Exception as exc:
            errors.append(f"parse failed: {cfg_path} — {exc}")
            continue

        try:
            cfg_container = OmegaConf.to_container(cfg, resolve=True)
        except Exception as exc:
            errors.append(f"to_container failed: {cfg_path} — {exc}")
            continue

        target_dict = _find_first_target(cfg_container)
        if target_dict is None:
            errors.append(f"no _target_ found in: {cfg_path}")
            continue

        try:
            model = instantiate(target_dict)
            return True, model, f"Instantiated from {cfg_path} with _target_={target_dict.get('_target_')}"
        except Exception as exc:
            errors.append(f"instantiate failed: {cfg_path} — {exc}")
            continue

    return False, None, " | ".join(errors) if errors else "No config candidates found."


def _instantiate_stagebridge_model() -> Tuple[bool, Any, str]:
    """
    Instantiate the StageBridgeModel in a resilient way.

    Priority:
    1) Instantiate from Hydra configs (source of truth).
    2) Fall back to direct constructor guessing only if configs do not contain targets.
    """
    repo_root = Path(__file__).resolve().parent.parent

    ok_cfg, model, msg_cfg = _instantiate_model_from_cfg(repo_root)
    if ok_cfg and model is not None:
        try:
            model.eval()
        except Exception:
            pass
        return True, model, msg_cfg

    try:
        from stagebridge.models.stagebridge import StageBridgeModel  # type: ignore
    except Exception as exc:
        return False, None, f"Import failed and config instantiation failed: {msg_cfg} | import error: {exc}"

    # Heuristic fallbacks (kept for emergencies).
    candidates: List[Dict[str, Any]] = []
    candidates.append(
        {
            "input_dim": 16,
            "hidden_dim": 32,
            "vector_field_hidden_dim": 64,
            "num_heads": 2,
            "num_inducing_points": 4,
            "num_seed_vectors": 1,
            "num_stages": 5,
            "time_embedding_dim": 16,
            "stage_embedding_dim": 16,
            "dropout": 0.0,
        }
    )

    for kwargs in candidates:
        try:
            model = StageBridgeModel(**kwargs)
            model.eval()
            return True, model, f"Instantiated StageBridgeModel with kwargs keys={sorted(list(kwargs.keys()))}"
        except Exception:
            continue

    return False, None, f"Could not instantiate StageBridgeModel. Config instantiation error: {msg_cfg}"


def _smoke_forward_check() -> Dict[str, Any]:
    """
    Minimal CPU forward pass to catch shape/import errors early.

    Your model.forward signature (observed) is:
        (x_set, x_t, t, stage_src, stage_tgt, mask=None) -> Tensor

    This smoke test:
    - Inspects model.forward signature
    - Provides x_set/x_t/t tensors
    - Provides stage_src/stage_tgt as ints
    - Provides mask as None (or a boolean tensor if required)
    """
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": f"torch import failed: {exc}"}

    ok_model, model, msg = _instantiate_stagebridge_model()
    if not ok_model or model is None:
        return {"ok": False, "error": msg}

    try:
        model.eval()
    except Exception:
        pass

    # Infer input dimension from model.config.input_dim if available.
    d_in = 16
    if hasattr(model, "config"):
        try:
            cfg = getattr(model, "config")
            if hasattr(cfg, "input_dim"):
                d_in = int(getattr(cfg, "input_dim"))
        except Exception:
            pass

    # Build minimal inputs.
    # Note: x_set is (B,N,D); x_t is (M,D) or (B,M,D) depending on implementation.
    x_set = torch.randn(1, 8, d_in)
    x_t_unbatched = torch.randn(4, d_in)
    t_unbatched = torch.rand(4)
    x_t_batched = torch.randn(1, 4, d_in)
    t_batched = torch.rand(1, 4)

    # Use simple stage indices (Normal=0 -> AAH=1 by default).
    stage_src = 0
    stage_tgt = 1

    # Default mask (optional).
    mask_none = None
    mask_unbatched = torch.ones(4, dtype=torch.bool)
    mask_batched = torch.ones(1, 4, dtype=torch.bool)

    # Normalize parameter names for matching.
    def _norm(s: str) -> str:
        return s.replace("_", "").lower()

    # Map normalized forward parameter names to available values.
    # This is tuned to your observed signature.
    # If you rename args later, add aliases here.
    value_by_name_unbatched: Dict[str, Any] = {
        "xset": x_set,
        "x_set": x_set,
        "set": x_set,
        "context": x_set,
        "xt": x_t_unbatched,
        "x_t": x_t_unbatched,
        "t": t_unbatched,
        "time": t_unbatched,
        "stagesrc": stage_src,
        "stagetgt": stage_tgt,
        "mask": mask_none,  # start with None; we retry with tensor if needed
    }

    value_by_name_batched: Dict[str, Any] = {
        "xset": x_set,
        "x_set": x_set,
        "set": x_set,
        "context": x_set,
        "xt": x_t_batched,
        "x_t": x_t_batched,
        "t": t_batched,
        "time": t_batched,
        "stagesrc": stage_src,
        "stagetgt": stage_tgt,
        "mask": mask_none,  # start with None; we retry with tensor if needed
    }

    # Inspect forward signature.
    try:
        sig = inspect.signature(model.forward)
    except Exception as exc:
        return {"ok": False, "error": f"Could not inspect model.forward signature: {exc} ({msg})"}

    params = [p for p in sig.parameters.values() if p.name != "self"]

    # Build positional args from a given value map.
    def _build_args(value_map: Dict[str, Any]) -> Tuple[bool, List[Any], str]:
        args: List[Any] = []
        for p in params:
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            key = _norm(p.name)
            if key in value_map:
                args.append(value_map[key])
                continue
            if p.default is not inspect._empty:
                args.append(p.default)
                continue
            return False, [], f"Unsatisfied required forward arg '{p.name}' (signature: {sig})"
        return True, args, "ok"

    last_error: Optional[str] = None

    # Try unbatched then batched.
    for attempt_name, value_map, mask_tensor in [
        ("unbatched", value_by_name_unbatched, mask_unbatched),
        ("batched", value_by_name_batched, mask_batched),
    ]:
        ok_args, args, why = _build_args(value_map)
        if not ok_args:
            last_error = why
            continue

        # Attempt 1: mask=None
        try:
            with torch.no_grad():
                out = model(*args)
            if not hasattr(out, "shape"):
                return {"ok": False, "error": f"Forward output not tensor-like on {attempt_name}. ({msg})"}
            if int(out.shape[-1]) != d_in:
                return {"ok": False, "error": f"Unexpected output shape {tuple(out.shape)} on {attempt_name}. ({msg})"}
            return {"ok": True, "output_shape": list(out.shape), "details": f"{msg} | forward_call={attempt_name} | signature={sig}"}
        except Exception as exc:
            # Retry with an explicit boolean mask if the model expects one.
            try:
                # Replace the mask value in the args list if 'mask' is in the signature.
                if any(_norm(p.name) == "mask" for p in params):
                    # Rebuild value_map with mask tensor.
                    value_map_retry = dict(value_map)
                    value_map_retry["mask"] = mask_tensor
                    ok_args2, args2, why2 = _build_args(value_map_retry)
                    if ok_args2:
                        with torch.no_grad():
                            out2 = model(*args2)
                        if not hasattr(out2, "shape"):
                            last_error = f"{exc}"
                            continue
                        if int(out2.shape[-1]) != d_in:
                            last_error = f"{exc}"
                            continue
                        return {
                            "ok": True,
                            "output_shape": list(out2.shape),
                            "details": f"{msg} | forward_call={attempt_name} (mask tensor) | signature={sig}",
                        }
                last_error = str(exc)
            except Exception as exc2:
                last_error = f"{exc} | retry failed: {exc2}"
                continue

    return {
        "ok": False,
        "error": f"Smoke forward failed. Last error: {last_error} ({msg}). Forward signature: {sig}",
    }


def _config_check(config_dir: Path) -> Dict[str, Any]:
    expected = [
        config_dir / "train.yaml",
        config_dir / "eval.yaml",
        config_dir / "model" / "stagebridge.yaml",
        config_dir / "training" / "default.yaml",
        config_dir / "experiment" / "smoke.yaml",
        config_dir / "splits" / "donor_holdout.yaml",
    ]
    missing = [str(p) for p in expected if not p.exists()]

    parse_ok = True
    parse_errors: List[str] = []
    try:
        from omegaconf import OmegaConf

        for p in expected:
            if p.exists():
                try:
                    OmegaConf.load(p)
                except Exception as exc:
                    parse_ok = False
                    parse_errors.append(f"{p}: {exc}")
    except Exception as exc:
        parse_ok = False
        parse_errors.append(f"OmegaConf unavailable: {exc}")

    return {"ok": len(missing) == 0 and parse_ok, "missing": missing, "parse_errors": parse_errors}


def _print_human(report: Dict[str, Any]) -> None:
    ok_sym = "✓"
    fail_sym = "✗"
    skip_sym = "○"
    line = "=" * 70

    print(f"\n{line}")
    print("  StageBridge Environment Check")
    print(line)
    print(f"  Python   : {report['python']['version'].split()[0]}")
    print(f"  Exec     : {report['python']['executable']}")
    print(f"  Platform : {report['python']['platform']}")

    print("\n  [Required modules]")
    for c in report["checks"]["modules"]:
        sym = ok_sym if c.get("ok") else fail_sym
        tail = f"  [{c.get('version', '')}]" if c.get("ok") else f"  — {c.get('error', '')}"
        print(f"  {sym}  {c['name']}{tail}")

    print("\n  [StageBridge import]")
    sb = report["checks"]["stagebridge_import"]
    if sb.get("ok"):
        print(f"  {ok_sym}  stagebridge  file={sb.get('stagebridge_file')}  — {sb.get('note', '')}")
    else:
        print(f"  {fail_sym}  stagebridge  — {sb.get('error', '')}")

    print("\n  [Notebook entrypoint]")
    nb = report["checks"]["notebook"]
    if nb.get("ok"):
        print(f"  {ok_sym}  StageBridge.ipynb  path={nb.get('path')}")
    else:
        print(f"  {fail_sym}  StageBridge.ipynb  — {nb.get('error', '')}")

    print("\n  [Stage ontology]")
    so = report["checks"]["stage_ontology"]
    if so.get("ok"):
        print(f"  {ok_sym}  stages={so.get('canonical_stages')}")
    else:
        print(f"  {fail_sym}  stage ontology  — {so.get('error', '')}")

    print("\n  [Smoke forward pass]")
    sf = report["checks"]["smoke_forward"]
    if sf.get("ok"):
        print(f"  {ok_sym}  CPU forward pass  shape={sf.get('output_shape')}  — {sf.get('details', '')}")
    else:
        print(f"  {fail_sym}  CPU forward pass  — {sf.get('error', '')}")

    if "gpu" in report["checks"]:
        print("\n  [GPU / CUDA]")
        g = report["checks"]["gpu"]
        if g.get("ok") and g.get("cuda_available"):
            print(f"  {ok_sym}  CUDA {g.get('cuda_version')}  devices={g.get('cuda_devices')}")
        elif g.get("ok"):
            print(f"  {skip_sym}  CUDA not available (CPU-only build)")
        else:
            print(f"  {fail_sym}  CUDA check failed — {g.get('error', '')}")

    print("\n  [Config files]")
    cfg = report["checks"]["configs"]
    if cfg.get("ok"):
        print("  ✓  Required config files present and parseable")
    else:
        for m in cfg.get("missing", []):
            print(f"  {fail_sym}  missing: {m}")
        for e in cfg.get("parse_errors", []):
            print(f"  {fail_sym}  parse error: {e}")

    print("\n  [Optional modules]")
    for c in report["checks"]["optional"]:
        sym = ok_sym if c.get("ok") else skip_sym
        tail = f"  [{c.get('version', '')}]" if c.get("ok") else f"  — {c.get('error', 'not installed')}"
        print(f"  {sym}  {c['name']}{tail}")

    print(f"\n{line}")
    print("  RESULT: ALL REQUIRED CHECKS PASSED" if report["ok"] else "  RESULT: FAILED — see above")
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
    stagebridge_import = _stagebridge_import_check(repo_root)
    notebook_check = _notebook_check(repo_root)
    stage_ontology = _stage_ontology_check()
    smoke_check = _smoke_forward_check()
    cfg_check = _config_check(config_dir)

    checks: Dict[str, Any] = {
        "modules": module_checks,
        "optional": optional_checks,
        "stagebridge_import": stagebridge_import,
        "notebook": notebook_check,
        "stage_ontology": stage_ontology,
        "smoke_forward": smoke_check,
        "configs": cfg_check,
    }
    if not args.no_gpu:
        checks["gpu"] = _torch_gpu_check()

    ok = (
        all(bool(c.get("ok")) for c in module_checks)
        and bool(stagebridge_import.get("ok"))
        and bool(notebook_check.get("ok"))
        and bool(stage_ontology.get("ok"))
        and bool(smoke_check.get("ok"))
        and bool(cfg_check.get("ok"))
    )

    report: Dict[str, Any] = {
        "ok": bool(ok),
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
        print(json.dumps({"ok": bool(ok), "output": str(args.output)}, indent=2))
    else:
        _print_human(report)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()