#!/usr/bin/env python
"""Evaluate StageBridge predictions.

Usage:
    python scripts/evaluate.py predictions=runs/exp1/predictions.parquet
    python scripts/evaluate.py predictions=runs/exp1/predictions.parquet reference=data/cells.parquet
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.contracts import (
    LATENT_DIM,
    EvaluationOutputContract,
    InferenceOutputContract,
    assert_contract,
)
from stagebridge.evaluation import evaluate_predictions

log = logging.getLogger(__name__)


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate predictions."""
    df = pd.read_parquet(path)
    errors = InferenceOutputContract.validate(df)
    assert_contract(errors, "predictions")
    return df


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load reference embeddings and stages."""
    df = pd.read_parquet(path)

    # Extract fused embeddings
    fused_cols = [f"z_fused_{i}" for i in range(LATENT_DIM)]
    if all(c in df.columns for c in fused_cols):
        embeddings = df[fused_cols].values
    elif "embedding" in df.columns:
        embeddings = np.stack(df["embedding"].values)
    else:
        raise ValueError("No embeddings found in reference")

    stages = df["stage"].values
    return embeddings, stages


def parse_embeddings(series: pd.Series) -> np.ndarray:
    """Parse embedding column to numpy array."""
    if isinstance(series.iloc[0], list):
        return np.stack(series.values)
    elif isinstance(series.iloc[0], np.ndarray):
        return np.stack(series.values)
    else:
        raise ValueError(f"Unknown embedding format: {type(series.iloc[0])}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> dict:
    """Evaluate predictions."""
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Load predictions
    predictions_path = Path(cfg.get("predictions", ""))
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    log.info(f"Loading predictions from {predictions_path}")
    predictions_df = load_predictions(predictions_path)

    # Parse embeddings
    predicted = parse_embeddings(predictions_df["predicted_embedding"])
    source = parse_embeddings(predictions_df["source_embedding"])

    log.info(f"Evaluating {len(predicted)} predictions")

    # Load reference if provided (for stage accuracy)
    reference_path = cfg.get("reference", None)
    reference_embeddings = None
    reference_stages = None

    if reference_path and Path(reference_path).exists():
        log.info(f"Loading reference from {reference_path}")
        reference_embeddings, reference_stages = load_reference(Path(reference_path))

    # Compute metrics
    log.info("Computing metrics...")

    # Overall metrics
    overall_metrics = evaluate_predictions(
        predicted=predicted,
        target=source,  # Compare to source for displacement
        reference_embeddings=reference_embeddings,
        reference_stages=reference_stages,
    )

    # Per-transition metrics
    per_transition = {}
    for (src, tgt), group in predictions_df.groupby(["source_stage", "target_stage"]):
        pred = parse_embeddings(group["predicted_embedding"])
        src_emb = parse_embeddings(group["source_embedding"])

        transition_name = f"{src}->{tgt}"
        per_transition[transition_name] = {
            "n_samples": len(group),
            "mean_displacement": float(np.mean(np.linalg.norm(pred - src_emb, axis=1))),
            "mean_gate": float(group["context_gate"].mean()),
        }

    # Per-donor metrics
    per_donor = {}
    for donor, group in predictions_df.groupby("donor_id"):
        pred = parse_embeddings(group["predicted_embedding"])
        src_emb = parse_embeddings(group["source_embedding"])

        per_donor[str(donor)] = {
            "n_samples": len(group),
            "mean_displacement": float(np.mean(np.linalg.norm(pred - src_emb, axis=1))),
            "mean_gate": float(group["context_gate"].mean()),
        }

    # Assemble results
    results = {
        **overall_metrics,
        "per_transition": per_transition,
        "per_donor": per_donor,
        "metadata": {
            "model_name": "stagebridge",  # TODO: extract from predictions
            "checkpoint_path": str(predictions_path.parent / "checkpoints" / "best.pt"),
            "fold_idx": cfg.experiment.fold_idx,
            "n_samples": len(predictions_df),
            "evaluated_at": datetime.now().isoformat(),
        },
    }

    # Validate contract
    errors = EvaluationOutputContract.validate(results)
    if errors:
        log.warning(f"Evaluation contract warnings: {errors}")

    # Save results
    output_dir = Path(cfg.paths.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"Evaluation saved to {output_path}")

    # Log summary
    log.info("=== Evaluation Summary ===")
    log.info(f"Wasserstein distance: {results['wasserstein_distance']:.4f}")
    log.info(f"MMD: {results['mmd']:.4f}")
    log.info(f"Mean displacement: {results['mean_displacement']:.4f}")
    if "stage_accuracy" in results:
        log.info(f"Stage accuracy: {results['stage_accuracy']:.4f}")

    return results


if __name__ == "__main__":
    main()
