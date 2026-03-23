"""Mission 3 tests for the spatial mapping branch."""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

# Skip entire module if scvi is not available (optional dependency)
scvi = pytest.importorskip("scvi", reason="scvi required for spatial mapping tests")

from stagebridge.pipelines.run_spatial_mapping import run_spatial_mapping
from stagebridge.spatial_mapping.destvi_mapper import run_destvi
from stagebridge.spatial_mapping.tacco_mapper import run_tacco
from stagebridge.spatial_mapping.tangram_mapper import load_active_tangram_mapping, run_tangram


def _write_spatial_h5ad(path: Path) -> None:
    obs = pd.DataFrame(
        {
            "stage": ["AAH", "AIS", "MIA", "LUAD"],
            "patient_id": ["P1", "P2", "P2", "P3"],
            "sample_id": ["V1", "V2", "V3", "V4"],
            "spot_id": ["spot1", "spot2", "spot3", "spot4"],
        }
    )
    adata = ad.AnnData(X=np.zeros((4, 3), dtype=np.float32), obs=obs)
    adata.obsm["X_tangram_ct"] = np.asarray(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.3, 0.3],
            [0.1, 0.1, 0.4, 0.4],
            [0.1, 0.2, 0.2, 0.5],
        ],
        dtype=np.float32,
    )
    adata.obsm["spatial"] = np.asarray(
        [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
        dtype=np.float32,
    )
    adata.uns["tangram_ct_columns"] = ["AT2", "Fibroblast lineage", "Macrophages", "Capillary"]
    adata.write_h5ad(path)


def _write_raw_snrna_h5ad(path: Path, *, include_labels: bool = True) -> None:
    obs = pd.DataFrame(
        {
            "stage": ["AAH", "AAH", "AIS", "AIS"],
            "donor_id": ["P1", "P1", "P2", "P2"],
            "sample_id": ["S1", "S1", "S2", "S2"],
        },
        index=[f"cell{i}" for i in range(4)],
    )
    if include_labels:
        obs["hlca_label"] = ["AT2", "AT2", "Fibroblast lineage", "Fibroblast lineage"]
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    X = np.asarray(
        [
            [5.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 5.0, 1.0],
        ],
        dtype=np.float32,
    )
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(path)


def _write_hlca_labels_parquet(path: Path) -> None:
    labels = pd.DataFrame(
        {
            "hlca_label": ["AT2", "AT2", "Fibroblast lineage", "Fibroblast lineage"],
        },
        index=pd.Index([f"cell{i}" for i in range(4)], name="cell_id"),
    )
    labels.to_parquet(path, index=True, engine="pyarrow")


def _write_raw_spatial_h5ad(path: Path) -> None:
    obs = pd.DataFrame(
        {
            "stage": ["AAH", "AIS"],
            "patient_id": ["P1", "P2"],
            "sample_id": ["V1", "V2"],
        },
        index=["spot1", "spot2"],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    X = np.asarray(
        [
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
        ],
        dtype=np.float32,
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    adata.write_h5ad(path)


def test_tangram_mapping_contract_and_interfaces(tmp_path: Path) -> None:
    spatial_path = tmp_path / "spatial_tangram.h5ad"
    _write_spatial_h5ad(spatial_path)
    cfg = {
        "seed": 42,
        "data": {
            "data_root": str(tmp_path),
            "spatial_h5ad": str(spatial_path),
            "spatial_tangram_h5ad": str(spatial_path),
        },
        "spatial_mapping": {
            "method": "tangram",
            "max_spots_per_stage": 2,
        },
    }

    tangram = load_active_tangram_mapping(cfg, stages=["AAH", "AIS", "MIA"], max_spots_per_stage=2)
    assert tangram.method == "tangram"
    assert tangram.status == "complete"
    assert tangram.compositions.shape == (3, 4)
    assert tangram.coords.shape == (3, 2)
    assert tangram.feature_names == ("AT2", "Fibroblast lineage", "Macrophages", "Capillary")
    assert tangram.qc is not None

    tacco = run_tacco(cfg)
    destvi = run_destvi(cfg)
    assert tacco.status in {"missing_inputs", "complete"}
    assert destvi.status in {"missing_inputs", "complete"}

    pipeline_output = run_spatial_mapping(cfg)
    assert pipeline_output["ok"] is True
    assert pipeline_output["status"] == "complete"
    assert (
        pipeline_output["spatial_mapping"]["n_spots"] == 4
        or pipeline_output["spatial_mapping"]["n_spots"] == 3
    )


def test_tangram_rebuild_and_tacco_raw_provider_paths(tmp_path: Path) -> None:
    raw_snrna = tmp_path / "snrna_raw.h5ad"
    raw_spatial = tmp_path / "spatial_raw.h5ad"
    labels_parquet = tmp_path / "snrna_labels.parquet"
    _write_raw_snrna_h5ad(raw_snrna, include_labels=False)
    _write_raw_spatial_h5ad(raw_spatial)
    _write_hlca_labels_parquet(labels_parquet)

    tangram_cfg = {
        "seed": 42,
        "data": {
            "data_root": str(tmp_path),
            "snrna_h5ad": str(raw_snrna),
            "snrna_latent_h5ad": str(tmp_path / "missing_latent.h5ad"),
            "spatial_h5ad": str(raw_spatial),
            "spatial_tangram_h5ad": str(tmp_path / "missing_tangram.h5ad"),
            "tangram_scores_parquet": str(tmp_path / "missing_scores.parquet"),
            "niche_token_bank_zarr": str(tmp_path / "missing_tokens.zarr"),
            "wes_features_path": str(tmp_path / "missing_wes.parquet"),
        },
        "reference": {
            "labels_parquet": str(labels_parquet),
        },
        "spatial_mapping": {
            "method": "tangram",
            "execution_mode": "rebuild_cached",
            "label_col": "hlca_label",
            "device": "cpu",
            "num_epochs": 2,
            "learning_rate": 0.05,
            "aggregate_profiles": True,
            "max_training_genes": 3,
            "min_shared_genes": 2,
            "min_cells_per_label": 1,
            "max_reference_cells_per_label": 2,
            "max_spots_per_stage": 2,
            "verbose": False,
        },
    }
    tangram = run_tangram(tangram_cfg, stages=["AAH", "AIS"], max_spots_per_stage=2, seed=42)
    assert tangram.status == "complete"
    assert tangram.execution_mode == "rebuild_cached"
    assert tangram.provenance is not None
    assert tangram.provenance["mode"] in {"rebuilt", "cached"}
    assert tangram.provenance["label_metadata"]["label_source"]["source"] == "labels_parquet"
    assert tangram.compositions is not None
    assert tangram.compositions.shape[0] == 2

    tacco_cfg = {
        "seed": 42,
        "data": tangram_cfg["data"],
        "reference": tangram_cfg["reference"],
        "spatial_mapping": {
            "method": "tacco",
            "execution_mode": "rebuild_cached",
            "label_col": "hlca_label",
            "annotation_method": "OT",
            "fallback_annotation_method": "nnls",
            "max_reference_cells_per_label": 2,
            "verbose": 0,
            "max_spots_per_stage": 2,
        },
    }
    tacco = run_tacco(tacco_cfg, stages=["AAH", "AIS"], max_spots_per_stage=2, seed=42)
    assert tacco.status == "complete"
    assert tacco.execution_mode == "rebuild_cached"
    assert tacco.provenance is not None
    assert tacco.provenance["annotation_method_used"] in {"OT", "nnls"}
    assert (
        tacco.provenance["reference_subset_metadata"]["label_source"]["source"] == "labels_parquet"
    )
    assert tacco.compositions is not None
    assert tacco.compositions.shape[0] == 2

    destvi_cfg = {
        "seed": 42,
        "data": tangram_cfg["data"],
        "reference": tangram_cfg["reference"],
        "spatial_mapping": {
            "method": "destvi",
            "execution_mode": "rebuild_cached",
            "label_col": "hlca_label",
            "batch_size": 2,
            "max_reference_cells_per_label": 2,
            "max_training_genes": 3,
            "min_shared_genes": 2,
            "show_progress": False,
            "condscvi": {
                "max_epochs": 1,
                "n_hidden": 16,
                "n_latent": 3,
                "n_layers": 1,
                "dropout_rate": 0.05,
                "lr": 1e-3,
                "prior": "mog",
                "num_classes_mog": 2,
            },
            "destvi": {
                "max_epochs": 1,
                "lr": 1e-3,
                "vamp_prior_p": 2,
            },
        },
    }
    destvi = run_destvi(destvi_cfg, stages=["AAH", "AIS"], max_spots_per_stage=2, seed=42)
    assert destvi.status == "complete"
    assert destvi.execution_mode == "rebuild_cached"
    assert destvi.provenance is not None
    assert destvi.provenance["training_report"]["n_labels"] == 2
    assert destvi.compositions is not None
    assert destvi.compositions.shape[0] == 2
