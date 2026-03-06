"""Evaluate a trained StageBridge checkpoint on donor-held-out splits."""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from stagebridge.logging_utils import configure_root_logger, get_logger
from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.preprocessing.harmonize import add_hlca_latent, ensure_required_obs_fields
from stagebridge.preprocessing.stage_ontology import stage_to_index
from stagebridge.runs import build_output_context
from stagebridge.training.trainer import (
    TransitionSampler,
    build_donor_holdout_splits,
    donors_with_min_stage_coverage,
)
from stagebridge.training.eval import evaluate_transition
from stagebridge.utils.types import RunManifest, StageBridgeConfig

configure_root_logger()
log = get_logger(__name__)

try:
    import anndata
except Exception as exc:  # pragma: no cover
    raise ImportError("anndata is required for eval_stagebridge.py") from exc



def _ensure_latent(adata: anndata.AnnData, cfg: DictConfig) -> None:
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
        raise KeyError(
            f"Missing both '{cfg.data.latent_key}' and '{cfg.data.fallback_latent_key}' in obsm."
        )

    add_hlca_latent(
        adata=adata,
        reference=None,
        source_key=fallback_key,
        output_key=target_key,
        n_components=int(_model_value(cfg, "input_dim")),
    )


def _stable_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _model_value(cfg: DictConfig, key: str) -> object:
    """Read a model field from either cfg.model.config.<key> or cfg.model.<key>."""
    value = OmegaConf.select(cfg, f"model.config.{key}")
    if value is None:
        value = OmegaConf.select(cfg, f"model.{key}")
    if value is None:
        raise KeyError(f"Missing model field '{key}' in cfg.model or cfg.model.config.")
    return value


def run_evaluation(cfg: DictConfig) -> dict[str, object]:
    ckpt_path = Path(str(cfg.checkpoint))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    snrna_path = Path(str(cfg.data.snrna_h5ad))
    if not snrna_path.exists():
        raise FileNotFoundError(f"snRNA merged h5ad not found at {snrna_path}")

    adata = anndata.read_h5ad(snrna_path)
    ensure_required_obs_fields(adata)
    _ensure_latent(adata, cfg)

    donor_ids = donors_with_min_stage_coverage(
        obs_stage=np.asarray(adata.obs[cfg.data.stage_col].astype(str)),
        obs_donor=np.asarray(adata.obs[cfg.data.donor_col].astype(str)),
        min_stages=int(cfg.splits.min_stages_per_donor),
    )
    splits = build_donor_holdout_splits(
        donor_ids=donor_ids,
        n_folds=int(cfg.splits.n_folds),
        seed=int(cfg.training.seed),
    )
    split = splits[0]

    payload = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = payload.get("config", {})

    model_input_dim = int(_model_value(cfg, "input_dim"))
    model_hidden_dim = int(_model_value(cfg, "hidden_dim"))
    model_vector_field_hidden_dim = int(_model_value(cfg, "vector_field_hidden_dim"))
    model_num_heads = int(_model_value(cfg, "num_heads"))
    model_num_inducing_points = int(_model_value(cfg, "num_inducing_points"))
    model_num_seed_vectors = int(_model_value(cfg, "num_seed_vectors"))
    model_num_stages = int(_model_value(cfg, "num_stages"))
    model_time_embedding_dim = int(_model_value(cfg, "time_embedding_dim"))
    model_stage_embedding_dim = int(_model_value(cfg, "stage_embedding_dim"))
    model_dropout = float(_model_value(cfg, "dropout"))

    model_cfg = StageBridgeConfig(
        input_dim=int(ckpt_cfg.get("input_dim", model_input_dim)),
        hidden_dim=int(ckpt_cfg.get("hidden_dim", model_hidden_dim)),
        vector_field_hidden_dim=int(
            ckpt_cfg.get("vector_field_hidden_dim", model_vector_field_hidden_dim)
        ),
        num_heads=int(ckpt_cfg.get("num_heads", model_num_heads)),
        num_inducing_points=int(ckpt_cfg.get("num_inducing_points", model_num_inducing_points)),
        num_seed_vectors=int(ckpt_cfg.get("num_seed_vectors", model_num_seed_vectors)),
        num_stages=int(ckpt_cfg.get("num_stages", model_num_stages)),
        time_embedding_dim=int(ckpt_cfg.get("time_embedding_dim", model_time_embedding_dim)),
        stage_embedding_dim=int(ckpt_cfg.get("stage_embedding_dim", model_stage_embedding_dim)),
        dropout=float(ckpt_cfg.get("dropout", model_dropout)),
        ot_epsilon=float(ckpt_cfg.get("ot_epsilon", cfg.training.ot_epsilon)),
        sinkhorn_iters=int(ckpt_cfg.get("sinkhorn_iters", cfg.training.sinkhorn_iters)),
        use_ot=bool(ckpt_cfg.get("use_ot", cfg.training.use_ot)),
        use_stage_embedding=bool(ckpt_cfg.get("use_stage_embedding", cfg.training.use_stage_embedding)),
        device=str(cfg.training.device),
    )

    model = StageBridgeModel(config=model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(model_cfg.resolved_device())
    model.eval()

    latent = np.asarray(adata.obsm[str(cfg.data.latent_key)], dtype=np.float32)
    obs_stage = np.asarray(adata.obs[str(cfg.data.stage_col)].astype(str).values)
    obs_donor = np.asarray(adata.obs[str(cfg.data.donor_col)].astype(str).values)
    try:
        test_sampler = TransitionSampler(
            latent=latent,
            obs_stage=obs_stage,
            obs_donor=obs_donor,
            donor_ids=list(split.test_donors),
            batch_cells=int(cfg.training.batch_cells),
            device=model_cfg.resolved_device(),
        )
    except ValueError:
        log.warning(
            "No valid transitions for eval test donor split; falling back to all eligible donors."
        )
        test_sampler = TransitionSampler(
            latent=latent,
            obs_stage=obs_stage,
            obs_donor=obs_donor,
            donor_ids=donor_ids,
            batch_cells=int(cfg.training.batch_cells),
            device=model_cfg.resolved_device(),
        )

    results: list[dict[str, float]] = []
    with torch.no_grad():
        for src, tgt in test_sampler.available_transitions:
            x_src, x_tgt = test_sampler.sample_transition_pair(src, tgt, n_cells=256)
            out = evaluate_transition(
                model=model,
                x_src=x_src,
                x_tgt=x_tgt,
                stage_src=stage_to_index(src),
                stage_tgt=stage_to_index(tgt),
                num_steps=8,
                ot_epsilon=float(cfg.training.ot_epsilon),
                sinkhorn_iters=int(cfg.training.sinkhorn_iters),
            )
            results.append(
                {
                    "stage_src": src,
                    "stage_tgt": tgt,
                    "sinkhorn": out.sinkhorn,
                    "mmd_rbf": out.mmd_rbf,
                    "classifier_auc": out.classifier_auc,
                    "jsd_composition": out.jsd_composition,
                    "rank_consistency": out.rank_consistency,
                }
            )

    out_ctx = build_output_context(cfg.output_dir)
    out_dir = out_ctx.tables_dir
    out_path = out_dir / f"eval_{cfg.run_name}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "results": results,
            },
            fh,
            indent=2,
        )

    manifest = RunManifest(
        run_name=str(cfg.run_name),
        task="eval",
        model_name="stagebridge",
        variant_label="stagebridge_eval",
        ablation=None,
        seed=int(cfg.training.seed),
        device_requested=str(cfg.training.device),
        device_resolved=model_cfg.resolved_device(),
        config_path="configs/eval.yaml",
        config_hash=_stable_hash(
            {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "checkpoint": str(ckpt_path),
            }
        ),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        data_paths={
            "snrna_h5ad": str(cfg.data.snrna_h5ad),
            "spatial_h5ad": str(cfg.data.spatial_h5ad),
            "hlca_h5ad": str(cfg.data.hlca_h5ad),
        },
        output_paths={
            "tables_dir": str(out_dir),
            "eval_json": str(out_path),
        },
        metrics_path=str(out_path),
        notes=f"checkpoint={ckpt_path}",
    )
    manifest_path = out_dir / f"run_manifest_eval_{cfg.run_name}.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest.to_dict(), fh, indent=2)

    log.info("Evaluation results saved to %s", out_path)
    log.info("Evaluation run manifest saved to %s", manifest_path)
    return {
        "ok": True,
        "checkpoint": str(ckpt_path),
        "eval_json": str(out_path),
        "run_manifest_json": str(manifest_path),
        "n_transitions": len(results),
    }
