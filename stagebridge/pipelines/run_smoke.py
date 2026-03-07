"""AAH -> AIS smoke pipeline: end-to-end proof of StageBridge on synthetic data.

This pipeline creates synthetic single-cell data for two stages (AAH, AIS),
runs through every core component (HLCA placeholder, tokenization, graph,
Sinkhorn OT, flow-matching training), and writes artifacts to disk.

Usage::

    python -m stagebridge.pipelines.run_smoke [--config configs/topoflow/smoke.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n_cells: int = 50,
    n_genes: int = 100,
    input_dim: int = 16,
    seed: int = 42,
) -> Any:
    """Create a minimal AnnData with AAH + AIS cells from 3 donors.

    Returns an AnnData with:
      - X: (2*n_cells, n_genes) random counts
      - obsm["X_hlca"]: (2*n_cells, input_dim) latent embeddings
      - obs columns: stage, patient_id
    """
    import anndata

    rng = np.random.default_rng(seed)

    # Two clusters in latent space separated by a shift
    latent_aah = rng.standard_normal((n_cells, input_dim)).astype(np.float32)
    latent_ais = rng.standard_normal((n_cells, input_dim)).astype(np.float32) + 1.5

    latent = np.vstack([latent_aah, latent_ais])
    X = np.abs(rng.poisson(lam=2, size=(2 * n_cells, n_genes))).astype(np.float32)

    donors = ["donor_A", "donor_B", "donor_C"]
    donor_labels = rng.choice(donors, size=2 * n_cells)
    stage_labels = np.array(["AAH"] * n_cells + ["AIS"] * n_cells)

    import pandas as pd
    obs = pd.DataFrame({
        "stage": pd.Categorical(stage_labels, categories=["AAH", "AIS"]),
        "patient_id": donor_labels,
    })

    adata = anndata.AnnData(
        X=X,
        obs=obs,
    )
    adata.obsm["X_hlca"] = latent
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    log.info(
        "Synthetic data: %d cells, %d genes, %d latent dims, stages=%s, donors=%s",
        adata.n_obs, adata.n_vars, input_dim,
        list(adata.obs["stage"].cat.categories),
        donors,
    )
    return adata


# ---------------------------------------------------------------------------
# 2. HLCA placeholder (copies PCA / already-set latent)
# ---------------------------------------------------------------------------

def ensure_hlca_latent(adata: Any, input_dim: int) -> None:
    """Ensure ``adata.obsm['X_hlca']`` exists.

    In smoke mode the synthetic data already has this key.
    For real data, this would call ``stagebridge.preprocessing.harmonize``.
    """
    if "X_hlca" in adata.obsm:
        lat = adata.obsm["X_hlca"]
        if lat.shape[1] != input_dim:
            # Truncate or pad to match model input_dim
            if lat.shape[1] > input_dim:
                adata.obsm["X_hlca"] = lat[:, :input_dim]
            else:
                pad = np.zeros((lat.shape[0], input_dim - lat.shape[1]), dtype=np.float32)
                adata.obsm["X_hlca"] = np.hstack([lat, pad])
        log.info("HLCA latent present: shape=%s", adata.obsm["X_hlca"].shape)
        return

    # Fallback: compute PCA
    from sklearn.decomposition import PCA
    n_comp = min(input_dim, adata.n_vars, adata.n_obs - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    lat = pca.fit_transform(adata.X).astype(np.float32)
    if lat.shape[1] < input_dim:
        pad = np.zeros((lat.shape[0], input_dim - lat.shape[1]), dtype=np.float32)
        lat = np.hstack([lat, pad])
    adata.obsm["X_hlca"] = lat
    log.info("Computed PCA fallback latent: shape=%s", adata.obsm["X_hlca"].shape)


# ---------------------------------------------------------------------------
# 3. Sinkhorn OT coupling
# ---------------------------------------------------------------------------

def compute_sinkhorn_coupling(
    x_src: np.ndarray,
    x_tgt: np.ndarray,
    epsilon: float = 0.1,
    n_iters: int = 30,
) -> np.ndarray:
    """Compute entropic OT coupling between two cell populations."""
    from stagebridge.training.losses import build_sinkhorn_coupling

    src_t = torch.from_numpy(x_src)
    tgt_t = torch.from_numpy(x_tgt)
    coupling = build_sinkhorn_coupling(src_t, tgt_t, epsilon=epsilon, n_iters=n_iters)
    log.info(
        "Sinkhorn coupling: shape=%s, mass=%.4f",
        coupling.shape, float(coupling.sum()),
    )
    return coupling.numpy()


# ---------------------------------------------------------------------------
# 4. Build model config
# ---------------------------------------------------------------------------

def build_config_from_dict(cfg: dict[str, Any]) -> "StageBridgeConfig":
    """Build a StageBridgeConfig from a plain dict (YAML-loaded config)."""
    from stagebridge.utils.types import StageBridgeConfig

    model = cfg.get("model", {})
    bridge = cfg.get("bridge", {})
    training = cfg.get("training", {})

    return StageBridgeConfig(
        input_dim=int(model.get("input_dim", 16)),
        hidden_dim=int(model.get("hidden_dim", 32)),
        vector_field_hidden_dim=int(model.get("vector_field_hidden_dim", 64)),
        num_heads=int(model.get("num_heads", 4)),
        num_inducing_points=int(model.get("num_inducing_points", 8)),
        num_seed_vectors=int(model.get("num_seed_vectors", 1)),
        num_stages=int(model.get("num_stages", 2)),
        time_embedding_dim=int(model.get("time_embedding_dim", 16)),
        stage_embedding_dim=int(model.get("stage_embedding_dim", 16)),
        dropout=float(model.get("dropout", 0.1)),
        use_ot=bool(model.get("use_ot", True)),
        use_stage_embedding=bool(model.get("use_stage_embedding", True)),
        use_cross_attn_drift=bool(model.get("use_cross_attn_drift", False)),
        use_spatial_rpe=bool(model.get("use_spatial_rpe", False)),
        use_graph_transformer=bool(model.get("use_graph_transformer", False)),
        graph_num_layers=int(model.get("graph_num_layers", 2)),
        graph_num_heads=int(model.get("graph_num_heads", 4)),
        ot_epsilon=float(bridge.get("ot_epsilon", 0.1)),
        sinkhorn_iters=int(bridge.get("sinkhorn_iters", 30)),
        num_ot_pairs=int(bridge.get("num_ot_pairs", 32)),
        sigma=float(bridge.get("sigma", 0.0)),
        use_stochastic_bridge=bool(bridge.get("use_stochastic_bridge", False)),
        context_consistency_weight=float(bridge.get("context_consistency_weight", 0.0)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        max_epochs=int(training.get("max_epochs", 2)),
        steps_per_epoch=int(training.get("steps_per_epoch", 3)),
        val_steps=int(training.get("val_steps", 2)),
        patience=int(training.get("patience", 5)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=bool(training.get("mixed_precision", False)),
        device=str(training.get("device", "cpu")),
        seed=int(training.get("seed", 42)),
    )


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------

def run_smoke_training(
    adata: Any,
    sb_config: "StageBridgeConfig",
    stage_order: tuple[str, ...] = ("AAH", "AIS"),
    output_dir: str | Path = "outputs/runs/smoke_aah_ais",
) -> dict[str, Any]:
    """Train StageBridgeModel on synthetic data with donor-holdout CV.

    Returns dict with training history and benchmark metrics.
    """
    from stagebridge.models.stagebridge import StageBridgeModel
    from stagebridge.training.trainer import (
        StageBridgeTrainer,
        build_donor_holdout_splits,
        build_samplers_from_anndata,
    )
    from stagebridge.utils.seeds import set_global_seed

    set_global_seed(sb_config.seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Donor split — 3 donors, 3-fold is too fine, use 2-fold
    donors = sorted(adata.obs["patient_id"].unique().tolist())
    n_folds = min(2, len(donors))
    splits = build_donor_holdout_splits(donors, n_folds=n_folds, seed=sb_config.seed)

    all_results: list[dict[str, Any]] = []

    for fold_idx, split in enumerate(splits):
        log.info("=== Fold %d/%d ===", fold_idx + 1, len(splits))
        log.info("  train=%s  val=%s  test=%s", split.train_donors, split.val_donors, split.test_donors)

        train_sampler, val_sampler, test_sampler = build_samplers_from_anndata(
            adata,
            split=split,
            latent_key="X_hlca",
            stage_col="stage",
            donor_col="patient_id",
            batch_cells=min(32, adata.n_obs // 4),
            device=sb_config.resolved_device(),
        )

        model = StageBridgeModel(sb_config)
        trainer = StageBridgeTrainer(model, sb_config)

        result = trainer.fit(
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            test_sampler=test_sampler,
            output_dir=output_dir,
            run_name=f"smoke_fold{fold_idx}",
        )

        fold_result = {
            "fold": fold_idx,
            "best_val_loss": result.best_val_loss,
            "n_epochs": len(result.history),
            "benchmark": result.benchmark_metrics,
            "checkpoint": str(result.best_checkpoint),
        }
        all_results.append(fold_result)
        log.info("Fold %d complete: val_loss=%.6f", fold_idx, result.best_val_loss)

    return {
        "folds": all_results,
        "n_folds": len(splits),
        "stage_order": list(stage_order),
    }


# ---------------------------------------------------------------------------
# 6. Report generation
# ---------------------------------------------------------------------------

def generate_smoke_report(
    training_results: dict[str, Any],
    coupling_shape: tuple[int, ...],
    n_cells: int,
    input_dim: int,
    report_dir: str | Path,
) -> dict[str, Any]:
    """Generate and save a JSON smoke test report."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    val_losses = [f["best_val_loss"] for f in training_results["folds"]]
    all_finite = all(np.isfinite(v) for v in val_losses)

    report = {
        "status": "PASS" if all_finite else "FAIL",
        "pipeline": "smoke_aah_ais",
        "stages": training_results["stage_order"],
        "n_cells": n_cells,
        "input_dim": input_dim,
        "coupling_shape": list(coupling_shape),
        "training": {
            "n_folds": training_results["n_folds"],
            "val_losses": val_losses,
            "all_finite": all_finite,
        },
        "benchmarks": [f.get("benchmark", {}) for f in training_results["folds"]],
    }

    report_path = report_dir / "smoke_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Smoke report written to %s (status=%s)", report_path, report["status"])
    return report


# ---------------------------------------------------------------------------
# 7. Main orchestrator
# ---------------------------------------------------------------------------

def run_smoke_pipeline(
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full AAH -> AIS smoke pipeline.

    Parameters
    ----------
    config_path : path to YAML config (default: uses built-in smoke defaults)
    output_dir : override output directory

    Returns
    -------
    dict with report, artifacts path, and status
    """
    from stagebridge.utils.artifacts import RunArtifacts

    # --- Load config ---
    if config_path is not None:
        from stagebridge.utils.config_loader import load_yaml_config
        cfg = load_yaml_config(config_path, expand_env=False)
    else:
        cfg = _default_smoke_config()

    if output_dir is None:
        output_dir = Path(cfg.get("output", {}).get("output_dir", "outputs/runs")) / "smoke_aah_ais"
    output_dir = Path(output_dir)

    stages_cfg = cfg.get("stages", {})
    stage_order = tuple(stages_cfg.get("order", ["AAH", "AIS"]))
    data_cfg = cfg.get("data", {})
    n_cells = int(data_cfg.get("n_cells", 50))
    n_genes = int(data_cfg.get("n_genes", 100))
    input_dim = int(cfg.get("model", {}).get("input_dim", 16))

    log.info("--- Smoke Pipeline: %s ---", " -> ".join(stage_order))

    # Step 1: Synthetic data
    adata = make_synthetic_data(
        n_cells=n_cells, n_genes=n_genes, input_dim=input_dim,
        seed=int(cfg.get("training", {}).get("seed", 42)),
    )

    # Step 2: Ensure HLCA latent
    ensure_hlca_latent(adata, input_dim=input_dim)

    # Step 3: Sinkhorn coupling demo
    aah_mask = adata.obs["stage"] == "AAH"
    ais_mask = adata.obs["stage"] == "AIS"
    coupling = compute_sinkhorn_coupling(
        adata.obsm["X_hlca"][aah_mask],
        adata.obsm["X_hlca"][ais_mask],
        epsilon=float(cfg.get("bridge", {}).get("ot_epsilon", 0.1)),
        n_iters=int(cfg.get("bridge", {}).get("sinkhorn_iters", 30)),
    )

    # Step 4: Build model config and train
    sb_config = build_config_from_dict(cfg)
    training_results = run_smoke_training(
        adata, sb_config,
        stage_order=stage_order,
        output_dir=output_dir,
    )

    # Step 5: Generate report
    report = generate_smoke_report(
        training_results=training_results,
        coupling_shape=coupling.shape,
        n_cells=n_cells * 2,
        input_dim=input_dim,
        report_dir=output_dir / "report",
    )

    # Step 6: Save artifacts
    artifacts = RunArtifacts(
        run_id="smoke_aah_ais",
        config=cfg,
        metrics=report,
        status=report["status"],
    )
    artifacts.save(output_dir / "artifacts")

    log.info("--- Smoke Pipeline Complete: %s ---", report["status"])
    return {
        "status": report["status"],
        "report": report,
        "artifacts_dir": str(output_dir / "artifacts"),
        "output_dir": str(output_dir),
    }


def _default_smoke_config() -> dict[str, Any]:
    """Return built-in smoke config when no YAML path is provided."""
    return {
        "data": {"synthetic": True, "n_cells": 50, "n_genes": 100},
        "model": {
            "input_dim": 16, "hidden_dim": 32, "vector_field_hidden_dim": 64,
            "num_heads": 4, "num_inducing_points": 8, "num_seed_vectors": 1,
            "num_stages": 2, "time_embedding_dim": 16, "stage_embedding_dim": 16,
            "dropout": 0.1, "use_ot": True, "use_stage_embedding": True,
        },
        "bridge": {
            "ot_epsilon": 0.1, "sinkhorn_iters": 30, "num_ot_pairs": 32,
            "sigma": 0.0, "use_stochastic_bridge": False,
            "context_consistency_weight": 0.0,
        },
        "training": {
            "learning_rate": 1e-3, "weight_decay": 0.0, "grad_clip_norm": 1.0,
            "max_epochs": 2, "steps_per_epoch": 3, "val_steps": 2,
            "patience": 5, "gradient_accumulation_steps": 1,
            "mixed_precision": False, "device": "cpu", "seed": 42,
        },
        "stages": {"order": ["AAH", "AIS"]},
        "output": {"output_dir": "outputs/runs"},
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="StageBridge smoke pipeline (AAH -> AIS)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    result = run_smoke_pipeline(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
