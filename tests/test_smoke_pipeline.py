"""Tests for the AAH → AIS smoke pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def test_make_synthetic_data():
    from stagebridge.pipelines.run_smoke import make_synthetic_data

    adata = make_synthetic_data(n_cells=20, n_genes=50, input_dim=8, seed=0)

    assert adata.n_obs == 40  # 20 AAH + 20 AIS
    assert adata.n_vars == 50
    assert "X_hlca" in adata.obsm
    assert adata.obsm["X_hlca"].shape == (40, 8)
    assert set(adata.obs["stage"].cat.categories) == {"AAH", "AIS"}
    assert len(adata.obs["patient_id"].unique()) == 3


def test_synthetic_data_clusters_separable():
    from stagebridge.pipelines.run_smoke import make_synthetic_data

    adata = make_synthetic_data(n_cells=100, n_genes=50, input_dim=16, seed=42)
    latent = adata.obsm["X_hlca"]
    aah = latent[adata.obs["stage"] == "AAH"]
    ais = latent[adata.obs["stage"] == "AIS"]

    # Clusters should have different means (shift of 1.5)
    mean_diff = np.linalg.norm(aah.mean(axis=0) - ais.mean(axis=0))
    assert mean_diff > 1.0, f"Clusters too close: diff={mean_diff:.3f}"


# ---------------------------------------------------------------------------
# HLCA latent
# ---------------------------------------------------------------------------

def test_ensure_hlca_latent_already_present():
    from stagebridge.pipelines.run_smoke import make_synthetic_data, ensure_hlca_latent

    adata = make_synthetic_data(n_cells=10, n_genes=20, input_dim=8)
    ensure_hlca_latent(adata, input_dim=8)
    assert adata.obsm["X_hlca"].shape == (20, 8)


def test_ensure_hlca_latent_truncates():
    from stagebridge.pipelines.run_smoke import make_synthetic_data, ensure_hlca_latent

    adata = make_synthetic_data(n_cells=10, n_genes=20, input_dim=16)
    ensure_hlca_latent(adata, input_dim=8)
    assert adata.obsm["X_hlca"].shape == (20, 8)


# ---------------------------------------------------------------------------
# Sinkhorn coupling
# ---------------------------------------------------------------------------

def test_sinkhorn_coupling_valid():
    from stagebridge.pipelines.run_smoke import compute_sinkhorn_coupling

    rng = np.random.default_rng(42)
    x_src = rng.standard_normal((30, 8)).astype(np.float32)
    x_tgt = rng.standard_normal((30, 8)).astype(np.float32) + 1.0

    coupling = compute_sinkhorn_coupling(x_src, x_tgt, epsilon=0.1, n_iters=30)
    assert coupling.shape == (30, 30)
    assert abs(coupling.sum() - 1.0) < 0.01, f"Mass={coupling.sum():.4f}"
    assert (coupling >= 0).all()


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def test_build_config_from_dict():
    from stagebridge.pipelines.run_smoke import build_config_from_dict, _default_smoke_config

    cfg = _default_smoke_config()
    sb_config = build_config_from_dict(cfg)

    assert sb_config.input_dim == 16
    assert sb_config.hidden_dim == 32
    assert sb_config.num_stages == 2
    assert sb_config.max_epochs == 2
    assert sb_config.device == "cpu"


# ---------------------------------------------------------------------------
# Full smoke pipeline (end-to-end)
# ---------------------------------------------------------------------------

def test_full_smoke_pipeline(tmp_path):
    from stagebridge.pipelines.run_smoke import run_smoke_pipeline

    result = run_smoke_pipeline(output_dir=tmp_path / "smoke_test")

    assert result["status"] == "PASS"

    # Check artifacts on disk
    artifacts_dir = Path(result["artifacts_dir"])
    assert (artifacts_dir / "pipeline_artifacts.json").exists()
    assert (artifacts_dir / "config.yaml").exists()
    assert (artifacts_dir / "metrics.json").exists()

    # Check report
    report_path = Path(result["output_dir"]) / "report" / "smoke_report.json"
    assert report_path.exists()
    with report_path.open() as f:
        report = json.load(f)
    assert report["status"] == "PASS"
    assert report["stages"] == ["AAH", "AIS"]
    assert report["training"]["all_finite"]
