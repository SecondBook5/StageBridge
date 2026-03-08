"""Mission 3 data-contract tests for the active LUAD evolution path."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.data.luad_evo.snrna import load_luad_evo_snrna_latent
from stagebridge.data.luad_evo.visium import load_luad_evo_spatial_mapping
from stagebridge.data.luad_evo.wes import build_wes_feature_lookup, load_luad_evo_wes_features


def _write_latent_h5ad(path: Path) -> None:
    obs = pd.DataFrame(
        {
            "stage": ["AAH", "ais", "MIA", "LUAD"],
            "donor_id": ["P1", "P2", "P2", "P3"],
            "sample_id": ["S1", "S2", "S3", "S4"],
            "hlca_label": ["AT2", "AT2", "Basal", "Secretory"],
        }
    )
    adata = ad.AnnData(X=np.arange(24, dtype=np.float32).reshape(4, 6), obs=obs)
    adata.write_h5ad(path)


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


def _write_wes_parquet(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3"],
            "stage": ["AAH", "AIS", "LUAD"],
            "tmb": [1.0, 2.0, 3.0],
            "kras_mut": [0.0, 1.0, 0.0],
            "egfr_mut": [0.0, 0.0, 1.0],
            "tp53_mut": [1.0, 0.0, 1.0],
            "stk11_mut": [0.0, 0.0, 0.0],
            "keap1_mut": [0.0, 1.0, 0.0],
            "smad4_mut": [0.0, 0.0, 1.0],
            "braf_mut": [0.0, 0.0, 0.0],
        }
    )
    frame.to_parquet(path, index=False)


def test_luad_evo_data_contract_loaders(tmp_path: Path) -> None:
    latent_path = tmp_path / "snrna_latent.h5ad"
    spatial_path = tmp_path / "spatial_tangram.h5ad"
    wes_path = tmp_path / "wes.parquet"
    token_bank_path = tmp_path / "token_bank.zarr"
    token_bank_path.mkdir()
    _write_latent_h5ad(latent_path)
    _write_spatial_h5ad(spatial_path)
    _write_wes_parquet(wes_path)

    cfg = {
        "data": {
            "data_root": str(tmp_path),
            "snrna_h5ad": str(latent_path),
            "snrna_latent_h5ad": str(latent_path),
            "spatial_h5ad": str(spatial_path),
            "spatial_tangram_h5ad": str(spatial_path),
            "tangram_scores_parquet": str(tmp_path / "scores.parquet"),
            "niche_token_bank_zarr": str(token_bank_path),
            "wes_features_path": str(wes_path),
        },
        "reference": {
            "reference_h5ad": str(tmp_path / "missing_hlca.h5ad"),
        },
    }

    paths = resolve_luad_evo_paths(cfg)
    assert paths.snrna_latent_h5ad == latent_path.resolve()
    assert paths.spatial_tangram_h5ad == spatial_path.resolve()
    assert paths.wes_features_path == wes_path.resolve()

    snrna = load_luad_evo_snrna_latent(cfg, stages=["AAH", "AIS", "MIA"])
    assert snrna.latent.shape == (3, 6)
    assert snrna.obs["stage"].tolist() == ["AAH", "AIS", "MIA"]
    assert "patient_id" in snrna.obs.columns
    assert "donor_id" in snrna.obs.columns

    spatial = load_luad_evo_spatial_mapping(cfg, stages=["AAH", "AIS", "MIA"])
    assert spatial.compositions.shape == (3, 4)
    assert spatial.coords.shape == (3, 2)
    assert spatial.feature_names == ("AT2", "Fibroblast lineage", "Macrophages", "Capillary")
    assert "donor_id" in spatial.obs.columns
    assert "patient_id" in spatial.obs.columns

    wes = load_luad_evo_wes_features(cfg, stages=["AAH", "AIS"])
    lookup = build_wes_feature_lookup(wes)
    assert wes.frame.shape[0] == 2
    assert lookup[("P1", "AAH")].shape == (8,)
    assert lookup[("P2", "AIS")][0] == 2.0
