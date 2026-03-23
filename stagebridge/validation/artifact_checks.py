"""
Artifact validation checks for StageBridge runs.

Ensures all required outputs exist and are properly formatted before
treating a run as valid.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# =============================================================================
# Artifact Contracts
# =============================================================================

REQUIRED_ARTIFACTS = {
    "reference_geometry": [
        "hlca_embedding.parquet",
        "luca_embedding.parquet",
        "fused_embedding.parquet",
        "reference_confidence.parquet",
        "reference_manifest.json",
    ],
    "spatial_backend": [
        "backend_outputs.parquet",
        "backend_metadata.json",
    ],
    "training": [
        "model.pt",
        "training_history.json",
        "config.yaml",
    ],
    "evaluation": [
        "metrics.json",
        "predictions.parquet",
    ],
    "ablation": [
        "ablation_summary.csv",
        "ablation_results.json",
    ],
}

PARQUET_SCHEMAS = {
    "hlca_embedding.parquet": {"required_cols": ["cell_id"], "min_rows": 1},
    "luca_embedding.parquet": {"required_cols": ["cell_id"], "min_rows": 1},
    "fused_embedding.parquet": {"required_cols": ["cell_id"], "min_rows": 1},
    "reference_confidence.parquet": {
        "required_cols": ["cell_id", "hlca_confidence", "luca_confidence"],
        "min_rows": 1,
    },
}


# =============================================================================
# Check Functions
# =============================================================================


def check_artifact_exists(artifact_path: Path) -> dict[str, Any]:
    """Check if artifact file exists."""
    exists = artifact_path.exists()
    size = artifact_path.stat().st_size if exists else 0

    return {
        "path": str(artifact_path),
        "exists": exists,
        "size_bytes": size,
        "valid": exists and size > 0,
    }


def check_parquet_schema(
    parquet_path: Path,
    required_cols: list[str],
    min_rows: int = 1,
) -> dict[str, Any]:
    """Validate parquet file schema and content."""
    result = {
        "path": str(parquet_path),
        "valid": False,
        "issues": [],
    }

    if not parquet_path.exists():
        result["issues"].append("File does not exist")
        return result

    try:
        df = pd.read_parquet(parquet_path)
        result["n_rows"] = len(df)
        result["n_cols"] = len(df.columns)
        result["columns"] = list(df.columns)

        # Check required columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            result["issues"].append(f"Missing columns: {missing_cols}")

        # Check minimum rows
        if len(df) < min_rows:
            result["issues"].append(f"Too few rows: {len(df)} < {min_rows}")

        # Check for NaN in required columns
        for col in required_cols:
            if col in df.columns and df[col].isna().any():
                nan_count = df[col].isna().sum()
                result["issues"].append(f"Column '{col}' has {nan_count} NaN values")

        result["valid"] = len(result["issues"]) == 0

    except Exception as e:
        result["issues"].append(f"Failed to read: {e}")

    return result


def check_json_artifact(json_path: Path, required_keys: list[str] | None = None) -> dict[str, Any]:
    """Validate JSON artifact."""
    result = {
        "path": str(json_path),
        "valid": False,
        "issues": [],
    }

    if not json_path.exists():
        result["issues"].append("File does not exist")
        return result

    try:
        with open(json_path) as f:
            data = json.load(f)

        result["keys"] = list(data.keys()) if isinstance(data, dict) else "not_dict"

        if required_keys and isinstance(data, dict):
            missing = set(required_keys) - set(data.keys())
            if missing:
                result["issues"].append(f"Missing keys: {missing}")

        result["valid"] = len(result["issues"]) == 0

    except json.JSONDecodeError as e:
        result["issues"].append(f"Invalid JSON: {e}")
    except Exception as e:
        result["issues"].append(f"Failed to read: {e}")

    return result


def check_model_checkpoint(model_path: Path) -> dict[str, Any]:
    """Validate PyTorch model checkpoint."""
    result = {
        "path": str(model_path),
        "valid": False,
        "issues": [],
    }

    if not model_path.exists():
        result["issues"].append("File does not exist")
        return result

    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict):
            result["keys"] = list(checkpoint.keys())
            if "model_state_dict" not in checkpoint and "state_dict" not in checkpoint:
                # Might be raw state dict
                result["type"] = "state_dict"
            else:
                result["type"] = "checkpoint"
        else:
            result["type"] = type(checkpoint).__name__

        result["size_mb"] = model_path.stat().st_size / (1024 * 1024)
        result["valid"] = True

    except Exception as e:
        result["issues"].append(f"Failed to load: {e}")

    return result


# =============================================================================
# Aggregate Validation
# =============================================================================


def validate_run_artifacts(
    run_dir: Path,
    stage: str = "all",
) -> dict[str, Any]:
    """
    Validate all artifacts for a run.

    Parameters
    ----------
    run_dir : Path
        Directory containing run outputs
    stage : str
        Which stage to validate ("reference_geometry", "training", etc.) or "all"

    Returns
    -------
    dict
        Validation report
    """
    report = {
        "run_dir": str(run_dir),
        "valid": True,
        "stages": {},
        "issues": [],
    }

    stages_to_check = [stage] if stage != "all" else list(REQUIRED_ARTIFACTS.keys())

    for stage_name in stages_to_check:
        if stage_name not in REQUIRED_ARTIFACTS:
            continue

        stage_report = {"artifacts": {}, "valid": True}

        for artifact in REQUIRED_ARTIFACTS[stage_name]:
            artifact_path = run_dir / artifact

            if artifact.endswith(".parquet"):
                schema = PARQUET_SCHEMAS.get(artifact, {"required_cols": [], "min_rows": 1})
                check = check_parquet_schema(artifact_path, **schema)
            elif artifact.endswith(".json"):
                check = check_json_artifact(artifact_path)
            elif artifact.endswith(".pt"):
                check = check_model_checkpoint(artifact_path)
            else:
                check = check_artifact_exists(artifact_path)

            stage_report["artifacts"][artifact] = check

            if not check["valid"]:
                stage_report["valid"] = False
                report["issues"].extend(check.get("issues", [f"{artifact} invalid"]))

        report["stages"][stage_name] = stage_report

        if not stage_report["valid"]:
            report["valid"] = False

    return report


def save_validation_report(report: dict[str, Any], output_dir: Path) -> Path:
    """Save validation report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = output_dir / "artifact_validation.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary
    md_path = output_dir / "artifact_validation.md"
    with open(md_path, "w") as f:
        f.write("# Artifact Validation Report\n\n")
        f.write(f"**Run Directory:** `{report['run_dir']}`\n\n")
        f.write(f"**Overall Valid:** {'✅ Yes' if report['valid'] else '❌ No'}\n\n")

        if report["issues"]:
            f.write("## Issues\n\n")
            for issue in report["issues"]:
                f.write(f"- {issue}\n")
            f.write("\n")

        f.write("## Stage Details\n\n")
        for stage, stage_report in report.get("stages", {}).items():
            status = "✅" if stage_report["valid"] else "❌"
            f.write(f"### {status} {stage}\n\n")
            for artifact, check in stage_report.get("artifacts", {}).items():
                artifact_status = "✅" if check["valid"] else "❌"
                f.write(f"- {artifact_status} `{artifact}`\n")
            f.write("\n")

    return json_path
