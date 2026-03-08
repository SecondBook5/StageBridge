"""Mission 3 tests for the active reference latent branch."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from stagebridge.reference.hlca_mapper import run_active_reference_latent


def _write_reference_latent_h5ad(path: Path) -> None:
    obs = pd.DataFrame(
        {
            "stage": ["AAH", "AAH", "AIS", "AIS", "MIA", "LUAD"],
            "donor_id": ["P1", "P2", "P1", "P2", "P3", "P4"],
            "sample_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "hlca_label": ["AT2", "AT2", "AT2", "Basal", "Basal", "Secretory"],
        }
    )
    X = np.asarray(
        [
            [0.0, 0.0, 0.1, 0.2],
            [0.1, 0.0, 0.0, 0.1],
            [1.0, 1.1, 0.9, 1.0],
            [1.2, 1.0, 0.8, 1.1],
            [2.0, 2.1, 1.9, 2.0],
            [3.0, 3.2, 2.9, 3.1],
        ],
        dtype=np.float32,
    )
    ad.AnnData(X=X, obs=obs).write_h5ad(path)


def test_reference_latent_interface_reports_diagnostics(tmp_path: Path) -> None:
    latent_path = tmp_path / "snrna_latent.h5ad"
    _write_reference_latent_h5ad(latent_path)

    cfg = {
        "data": {
            "data_root": str(tmp_path),
            "snrna_h5ad": str(latent_path),
            "snrna_latent_h5ad": str(latent_path),
            "spatial_h5ad": str(tmp_path / "missing_spatial.h5ad"),
            "spatial_tangram_h5ad": str(tmp_path / "missing_tangram.h5ad"),
            "tangram_scores_parquet": str(tmp_path / "missing_scores.parquet"),
            "niche_token_bank_zarr": str(tmp_path / "missing_tokens.zarr"),
            "wes_features_path": str(tmp_path / "missing_wes.parquet"),
        },
        "reference": {
            "reference_h5ad": str(tmp_path / "missing_hlca.h5ad"),
        },
    }

    result = run_active_reference_latent(
        cfg,
        stages=["AAH", "AIS", "MIA"],
        max_cells_per_stage=2,
    )

    assert result.cohort.latent.shape == (5, 4)
    assert result.latent_store.summary()["shape"] == [5, 4]
    assert result.summary()["backend_name"] == "hlca"
    assert result.summary()["provenance"]["mode"] == "loaded"
    assert result.diagnostics["stage_preservation"]["n_stages"] == 3
    assert "AAH->AIS" in result.diagnostics["stage_preservation"]["centroid_distances"]
    assert "probe" in result.diagnostics["stage_preservation"]
    assert result.diagnostics["donor_leakage"]["n_donors"] >= 2
    assert "gene_overlap" in result.diagnostics
    assert "label_neighborhood" in result.diagnostics
    assert "stage_label_alignment" in result.diagnostics
    assert "alignment_gate" in result.diagnostics
    assert result.label_transfer["ok"] is True
    assert result.label_transfer["label_col"] == "hlca_label"


def test_reference_pca_backend_reports_fit_provenance(tmp_path: Path) -> None:
    raw_path = tmp_path / "snrna_raw.h5ad"
    _write_reference_latent_h5ad(raw_path)

    cfg = {
        "data": {
            "data_root": str(tmp_path),
            "snrna_h5ad": str(raw_path),
            "snrna_latent_h5ad": str(tmp_path / "missing_latent.h5ad"),
            "spatial_h5ad": str(tmp_path / "missing_spatial.h5ad"),
            "spatial_tangram_h5ad": str(tmp_path / "missing_tangram.h5ad"),
            "tangram_scores_parquet": str(tmp_path / "missing_scores.parquet"),
            "niche_token_bank_zarr": str(tmp_path / "missing_tokens.zarr"),
            "wes_features_path": str(tmp_path / "missing_wes.parquet"),
        },
        "reference": {
            "latent_backend": "pca",
            "n_components": 3,
            "reference_h5ad": str(tmp_path / "missing_hlca.h5ad"),
        },
    }

    result = run_active_reference_latent(
        cfg,
        stages=["AAH", "AIS", "MIA"],
        max_cells_per_stage=2,
    )

    assert result.cohort.latent.shape == (5, 3)
    assert result.summary()["backend_name"] == "pca"
    assert result.summary()["latent_key"] == "X_pca"
    assert result.summary()["provenance"]["mode"] == "fit"
    assert result.summary()["provenance"]["fit_source"] == str(raw_path)
    assert result.diagnostics["stage_preservation"]["n_stages"] == 3
    assert "probe" in result.diagnostics["stage_preservation"]
    assert result.diagnostics["alignment_gate"]["status"] in {"fail", "weak_pass", "pass"}
