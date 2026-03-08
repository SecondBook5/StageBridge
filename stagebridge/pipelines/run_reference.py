"""Reference-layer pipeline entrypoint."""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def run_reference(cfg: DictConfig) -> dict[str, Any]:
    """Report the active reference-layer configuration.

    Mission 1 keeps orchestration importable while the scientific internals are
    relocated. Full execution behavior is intentionally deferred.
    """
    data_cfg = OmegaConf.to_container(cfg.get("data", {}), resolve=True)
    reference_cfg = OmegaConf.to_container(cfg.get("reference", {}), resolve=True)
    return {
        "ok": True,
        "pipeline": "reference",
        "status": "structural_stub",
        "data": data_cfg,
        "reference": reference_cfg,
    }
