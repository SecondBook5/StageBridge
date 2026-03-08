"""Metadata helpers for the LUAD evolution cohort."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from stagebridge.data.common.schema import LuadEvoPaths

from .stages import CANONICAL_STAGE_ORDER


@dataclass(slots=True, frozen=True)
class LuadEvoDataset:
    """High-level dataset contract for the active v1 cohort."""

    name: str = "luad_evo"
    stages: tuple[str, ...] = CANONICAL_STAGE_ORDER
    modalities: tuple[str, ...] = ("snRNA-seq", "Visium", "WES")


def _cfg_get(cfg: DictConfig | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, key)
        return default if value is None else value
    current: Any = cfg
    for part in key.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def resolve_luad_evo_paths(cfg: DictConfig | dict[str, Any]) -> LuadEvoPaths:
    """Resolve the active LUAD evolution assets with real-file fallbacks."""
    data_root = Path(str(_cfg_get(cfg, "data.data_root", "/mnt/e/StageBridge_data"))).resolve()
    data_cfg = {
        "snrna_h5ad": Path(str(_cfg_get(cfg, "data.snrna_h5ad", data_root / "interim/anndata/snrna/snrna_full.h5ad"))),
        "snrna_latent_h5ad": Path(str(_cfg_get(cfg, "data.snrna_latent_h5ad", data_root / "processed/anndata/snrna_hlca_latent_full.h5ad"))),
        "hlca_labels_parquet": Path(str(_cfg_get(cfg, "reference.labels_parquet", data_root / "processed/hlca/snrna_full_hlca_labels.parquet"))),
        "spatial_h5ad": Path(str(_cfg_get(cfg, "data.spatial_h5ad", data_root / "interim/anndata/spatial/spatial_full.h5ad"))),
        "spatial_tangram_h5ad": Path(str(_cfg_get(cfg, "data.spatial_tangram_h5ad", data_root / "processed/tangram/spatial_tangram_full.h5ad"))),
        "tangram_scores_parquet": Path(str(_cfg_get(cfg, "data.tangram_scores_parquet", data_root / "processed/tangram/spatial_tangram_celltype_scores.parquet"))),
        "niche_token_bank_zarr": Path(str(_cfg_get(cfg, "data.niche_token_bank_zarr", data_root / "processed/features/niche_token_bank.zarr"))),
        "wes_features_path": Path(str(_cfg_get(cfg, "data.wes_features_path", data_root / "processed/features/wes_features.parquet"))),
    }
    hlca_raw = _cfg_get(cfg, "reference.reference_h5ad", _cfg_get(cfg, "data.hlca_h5ad"))
    hlca_path = Path(str(hlca_raw)).resolve() if hlca_raw else None
    return LuadEvoPaths(
        data_root=data_root,
        snrna_h5ad=data_cfg["snrna_h5ad"].resolve(),
        snrna_latent_h5ad=data_cfg["snrna_latent_h5ad"].resolve(),
        hlca_labels_parquet=data_cfg["hlca_labels_parquet"].resolve(),
        spatial_h5ad=data_cfg["spatial_h5ad"].resolve(),
        spatial_tangram_h5ad=data_cfg["spatial_tangram_h5ad"].resolve(),
        tangram_scores_parquet=data_cfg["tangram_scores_parquet"].resolve(),
        niche_token_bank_zarr=data_cfg["niche_token_bank_zarr"].resolve(),
        wes_features_path=data_cfg["wes_features_path"].resolve(),
        hlca_h5ad=hlca_path,
    )
