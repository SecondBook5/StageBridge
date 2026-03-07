"""Shared configuration helpers used across training and evaluation workflows."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from stagebridge.logging_utils import get_logger
from stagebridge.utils.types import StageBridgeConfig

log = get_logger(__name__)


def stable_hash(payload: object) -> str:
    """Deterministic SHA-256 hash of a JSON-serializable payload."""
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def model_value(cfg: DictConfig, key: str) -> object:
    """Read a model field from either cfg.model.config.<key> or cfg.model.<key>."""
    value = OmegaConf.select(cfg, f"model.config.{key}")
    if value is None:
        value = OmegaConf.select(cfg, f"model.{key}")
    if value is None:
        raise KeyError(f"Missing model field '{key}' in cfg.model or cfg.model.config.")
    return value


def model_value_or_default(cfg: DictConfig, key: str, default: object) -> object:
    """Read a model field with fallback to *default*."""
    value = OmegaConf.select(cfg, f"model.config.{key}")
    if value is None:
        value = OmegaConf.select(cfg, f"model.{key}")
    return default if value is None else value


def build_stagebridge_config_from_cfg(
    cfg: DictConfig,
    *,
    input_dim: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> StageBridgeConfig:
    """Build ``StageBridgeConfig`` from Hydra model/training fields.

    This is the canonical config-construction path shared by train/eval
    workflows to avoid drift in default handling.
    """
    defaults = asdict(StageBridgeConfig())

    def _train_bool(path: str, default: bool) -> bool:
        raw = OmegaConf.select(cfg, path)
        return bool(default if raw is None else raw)

    def _train_int(path: str, default: int) -> int:
        raw = OmegaConf.select(cfg, path)
        return int(default if raw is None else raw)

    def _train_float(path: str, default: float) -> float:
        raw = OmegaConf.select(cfg, path)
        return float(default if raw is None else raw)

    payload: dict[str, Any] = {
        "input_dim": int(model_value(cfg, "input_dim") if input_dim is None else input_dim),
        "hidden_dim": int(model_value(cfg, "hidden_dim")),
        "vector_field_hidden_dim": int(model_value(cfg, "vector_field_hidden_dim")),
        "num_heads": int(model_value(cfg, "num_heads")),
        "num_inducing_points": int(model_value(cfg, "num_inducing_points")),
        "num_seed_vectors": int(model_value(cfg, "num_seed_vectors")),
        "num_stages": int(model_value(cfg, "num_stages")),
        "time_embedding_dim": int(model_value(cfg, "time_embedding_dim")),
        "stage_embedding_dim": int(model_value(cfg, "stage_embedding_dim")),
        "dropout": float(model_value(cfg, "dropout")),
        "ot_epsilon": float(cfg.training.ot_epsilon),
        "sinkhorn_iters": int(cfg.training.sinkhorn_iters),
        "num_ot_pairs": int(cfg.training.num_ot_pairs),
        "context_consistency_weight": float(cfg.training.context_consistency_weight),
        "learning_rate": float(cfg.training.learning_rate),
        "weight_decay": float(cfg.training.weight_decay),
        "grad_clip_norm": float(cfg.training.grad_clip_norm),
        "max_epochs": int(cfg.training.max_epochs),
        "steps_per_epoch": int(cfg.training.steps_per_epoch),
        "val_steps": int(cfg.training.val_steps),
        "patience": int(cfg.training.patience),
        "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
        "mixed_precision": bool(cfg.training.mixed_precision),
        "device": str(cfg.training.device),
        "seed": int(cfg.training.seed),
        "use_ot": bool(cfg.training.use_ot),
        "use_stage_embedding": bool(cfg.training.use_stage_embedding),
        "sigma": _train_float("training.sigma", float(defaults["sigma"])),
        "use_stochastic_bridge": _train_bool(
            "training.use_stochastic_bridge", bool(defaults["use_stochastic_bridge"])
        ),
        "use_cross_attn_drift": _train_bool(
            "training.use_cross_attn_drift", bool(defaults["use_cross_attn_drift"])
        ),
        "use_wes_features": _train_bool("training.use_wes_features", bool(defaults["use_wes_features"])),
        "use_spatial_rpe": bool(model_value_or_default(cfg, "use_spatial_rpe", defaults["use_spatial_rpe"])),
        "rpe_hidden_dim": int(model_value_or_default(cfg, "rpe_hidden_dim", defaults["rpe_hidden_dim"])),
        "wes_feature_dim": int(model_value_or_default(cfg, "wes_feature_dim", defaults["wes_feature_dim"])),
        "wes_hidden_dim": int(model_value_or_default(cfg, "wes_hidden_dim", defaults["wes_hidden_dim"])),
        "use_lr_features": _train_bool("training.use_lr_features", bool(defaults["use_lr_features"])),
        "lr_feature_dim": int(model_value_or_default(cfg, "lr_feature_dim", defaults["lr_feature_dim"])),
        "lr_hidden_dim": int(model_value_or_default(cfg, "lr_hidden_dim", defaults["lr_hidden_dim"])),
        "use_spatial_niche": _train_bool("training.use_spatial_niche", bool(defaults["use_spatial_niche"])),
        "spatial_niche_dim": int(
            model_value_or_default(cfg, "spatial_niche_dim", defaults["spatial_niche_dim"])
        ),
        "spatial_niche_hidden": int(
            model_value_or_default(cfg, "spatial_niche_hidden", defaults["spatial_niche_hidden"])
        ),
        "use_multihop_consistency": _train_bool(
            "training.use_multihop_consistency",
            bool(model_value_or_default(cfg, "use_multihop_consistency", defaults["use_multihop_consistency"])),
        ),
        "multihop_consistency_weight": float(
            model_value_or_default(cfg, "multihop_consistency_weight", defaults["multihop_consistency_weight"])
        ),
        "use_dirichlet_head": _train_bool(
            "training.use_dirichlet_head",
            bool(model_value_or_default(cfg, "use_dirichlet_head", defaults["use_dirichlet_head"])),
        ),
        "dirichlet_hidden_dim": int(
            model_value_or_default(cfg, "dirichlet_hidden_dim", defaults["dirichlet_hidden_dim"])
        ),
        "dirichlet_loss_weight": float(
            model_value_or_default(cfg, "dirichlet_loss_weight", defaults["dirichlet_loss_weight"])
        ),
        "profile_train_steps": _train_int(
            "training.profile_train_steps", int(defaults["profile_train_steps"])
        ),
        "use_graph_transformer": bool(
            model_value_or_default(cfg, "use_graph_transformer", defaults["use_graph_transformer"])
        ),
        "graph_num_layers": int(
            model_value_or_default(cfg, "graph_num_layers", defaults["graph_num_layers"])
        ),
        "graph_num_heads": int(
            model_value_or_default(cfg, "graph_num_heads", defaults["graph_num_heads"])
        ),
    }

    if overrides:
        payload.update(overrides)

    return StageBridgeConfig(**payload)


def ensure_latent(
    adata: Any,
    cfg: DictConfig,
    *,
    allow_hlca_reference: bool = True,
) -> None:
    """Ensure ``adata.obsm[latent_key]`` exists, computing it if needed.

    Parameters
    ----------
    adata : anndata.AnnData
    cfg : DictConfig
    allow_hlca_reference : bool
        When True (default, used by training), attempts to load the HLCA
        reference for alignment.  When False (used by evaluation), raises
        if the fallback key is also missing.
    """
    target_key = str(cfg.data.latent_key)
    fallback_key = str(cfg.data.fallback_latent_key)

    if target_key in adata.obsm:
        return

    # Support latent-only AnnData where X already stores latent vectors.
    var_names = adata.var_names.astype(str) if adata.var_names is not None else np.array([], dtype=object)
    if adata.X is not None and adata.n_vars > 0 and all(str(v).startswith("latent_") for v in var_names):
        adata.obsm[target_key] = np.asarray(adata.X, dtype=np.float32)
        log.info("Attached latent from adata.X -> adata.obsm['%s'] (latent-only h5ad).", target_key)
        return

    if fallback_key not in adata.obsm:
        if not allow_hlca_reference:
            raise KeyError(
                f"Missing both '{target_key}' and '{fallback_key}' in obsm."
            )
        X = adata.layers["log1p"] if "log1p" in adata.layers else adata.X
        from sklearn.decomposition import TruncatedSVD

        n_comp = min(int(model_value(cfg, "input_dim")), min(X.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=int(cfg.seed))
        adata.obsm[fallback_key] = svd.fit_transform(X).astype(np.float32)
        log.info("Computed fallback latent '%s' with %d components.", fallback_key, n_comp)

    reference = None
    if allow_hlca_reference and bool(cfg.data.use_hlca_reference):
        from stagebridge.io.hlca import load_hlca_reference

        hlca_path = Path(str(cfg.data.hlca_h5ad))
        if hlca_path.exists():
            reference = load_hlca_reference(h5ad_path=hlca_path, backed=None)
        else:
            log.warning("HLCA reference not found at %s; using fallback latent alignment.", hlca_path)

    from stagebridge.preprocessing.harmonize import add_hlca_latent

    add_hlca_latent(
        adata=adata,
        reference=reference,
        source_key=fallback_key,
        output_key=target_key,
        n_components=int(model_value(cfg, "input_dim")),
    )


def maybe_load_wes_lookup(
    cfg: DictConfig,
    *,
    enabled: bool | None = None,
) -> dict[tuple[str, str], np.ndarray] | None:
    """Load WES features parquet and return a (patient_id, stage) -> np.ndarray lookup."""
    use_wes = bool(enabled) if enabled is not None else bool(OmegaConf.select(cfg, "training.use_wes_features") or False)
    if not use_wes:
        return None
    from stagebridge.io.paths import StageBridgePaths

    paths = StageBridgePaths.from_cfg(cfg)
    wes_path = Path(paths.data_root) / "processed" / "features" / "wes_features.parquet"
    if not wes_path.exists():
        log.warning(
            "use_wes_features=true but wes_features.parquet not found at %s. "
            "Run scripts/run_wes_pipeline.py first. Proceeding without WES features.",
            wes_path,
        )
        return None
    from stagebridge.io.geo_wes import WES_FEATURE_COLS, load_wes_features

    df = load_wes_features(wes_path)
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for _, row in df.iterrows():
        key = (str(row["patient_id"]), str(row["stage"]))
        lookup[key] = row[WES_FEATURE_COLS].values.astype(np.float32)
    log.info("Loaded WES features for %d (patient, stage) pairs", len(lookup))
    return lookup


def maybe_load_lr_lookup(
    cfg: DictConfig,
    *,
    enabled: bool | None = None,
) -> dict[tuple[str, str], np.ndarray] | None:
    """Load LR features parquet and return a (patient_id, stage) -> np.ndarray lookup."""
    use_lr = bool(enabled) if enabled is not None else bool(OmegaConf.select(cfg, "training.use_lr_features") or False)
    if not use_lr:
        return None
    lr_path = Path(str(OmegaConf.select(cfg, "training.lr_features_path") or ""))
    if not lr_path.exists():
        log.warning(
            "use_lr_features=true but lr_features parquet not found at %s. "
            "Proceeding without LR features.",
            lr_path,
        )
        return None
    from stagebridge.io.lr_signaling import load_lr_features

    lookup = load_lr_features(lr_path)
    log.info("Loaded LR features for %d (patient, stage) pairs", len(lookup))
    return lookup
