"""Typed runtime settings extracted from Hydra/OmegaConf config trees."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class DataSettings:
    """Core data paths and column names required by workflows."""

    snrna_h5ad: Path
    spatial_h5ad: Path
    hlca_h5ad: Path
    stage_col: str
    donor_col: str
    latent_key: str


@dataclass(slots=True)
class RuntimeSettings:
    """Top-level runtime settings consumed by notebook and CLI entrypoints."""

    run_name: str
    output_dir: Path
    seed: int
    data_root: Path | None
    data: DataSettings


@dataclass(slots=True)
class WorkflowRequest:
    """Validated request object for notebook/CLI workflow dispatch."""

    step: str
    settings: RuntimeSettings


class ConfigValidationError(ValueError):
    """Raised when required config fields are missing or invalid."""


def _require(cfg: DictConfig, key: str) -> object:
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ConfigValidationError(f"Missing required config key: {key}")
    return value


def runtime_settings_from_cfg(cfg: DictConfig) -> RuntimeSettings:
    """Build and validate typed runtime settings from a DictConfig."""
    run_name = str(_require(cfg, "run_name"))
    output_dir = Path(str(_require(cfg, "output_dir")))
    seed_raw = OmegaConf.select(cfg, "seed")
    if seed_raw is None:
        seed_raw = OmegaConf.select(cfg, "training.seed")
    seed = int(seed_raw) if seed_raw is not None else 42

    data_root_raw = OmegaConf.select(cfg, "data.data_root")
    data_root = Path(str(data_root_raw)) if data_root_raw is not None else None

    data = DataSettings(
        snrna_h5ad=Path(str(_require(cfg, "data.snrna_h5ad"))),
        spatial_h5ad=Path(str(_require(cfg, "data.spatial_h5ad"))),
        hlca_h5ad=Path(str(_require(cfg, "data.hlca_h5ad"))),
        stage_col=str(_require(cfg, "data.stage_col")),
        donor_col=str(_require(cfg, "data.donor_col")),
        latent_key=str(_require(cfg, "data.latent_key")),
    )

    return RuntimeSettings(
        run_name=run_name,
        output_dir=output_dir,
        seed=seed,
        data_root=data_root,
        data=data,
    )


def workflow_request_from_cfg(step: str, cfg: DictConfig) -> WorkflowRequest:
    """Create a typed workflow request for `step`."""
    return WorkflowRequest(step=step, settings=runtime_settings_from_cfg(cfg))
