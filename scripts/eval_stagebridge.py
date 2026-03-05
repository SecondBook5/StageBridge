#!/usr/bin/env python
"""Evaluate a trained StageBridge checkpoint on donor-held-out splits."""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stagebridge.logging_utils import configure_root_logger, get_logger
from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.preprocessing.harmonize import add_hlca_latent, ensure_required_obs_fields
from stagebridge.preprocessing.stage_ontology import stage_to_index
from stagebridge.training.trainer import (
    build_donor_holdout_splits,
    build_samplers_from_anndata,
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

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except Exception as exc:  # pragma: no cover
    raise ImportError("hydra-core and omegaconf are required for eval_stagebridge.py") from exc



def _ensure_latent(adata: anndata.AnnData, cfg: DictConfig) -> None:
    if cfg.data.latent_key in adata.obsm:
        return
    if cfg.data.fallback_latent_key not in adata.obsm:
        raise KeyError(
            f"Missing both '{cfg.data.latent_key}' and '{cfg.data.fallback_latent_key}' in obsm."
        )
    add_hlca_latent(
        adata=adata,
        reference=None,
        source_key=str(cfg.data.fallback_latent_key),
        output_key=str(cfg.data.latent_key),
        n_components=int(cfg.model.config.input_dim),
    )


def _stable_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
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

    model_cfg = StageBridgeConfig(
        input_dim=int(ckpt_cfg.get("input_dim", cfg.model.config.input_dim)),
        hidden_dim=int(ckpt_cfg.get("hidden_dim", cfg.model.config.hidden_dim)),
        vector_field_hidden_dim=int(
            ckpt_cfg.get("vector_field_hidden_dim", cfg.model.config.vector_field_hidden_dim)
        ),
        num_heads=int(ckpt_cfg.get("num_heads", cfg.model.config.num_heads)),
        num_inducing_points=int(ckpt_cfg.get("num_inducing_points", cfg.model.config.num_inducing_points)),
        num_seed_vectors=int(ckpt_cfg.get("num_seed_vectors", cfg.model.config.num_seed_vectors)),
        num_stages=int(ckpt_cfg.get("num_stages", cfg.model.config.num_stages)),
        time_embedding_dim=int(ckpt_cfg.get("time_embedding_dim", cfg.model.config.time_embedding_dim)),
        stage_embedding_dim=int(ckpt_cfg.get("stage_embedding_dim", cfg.model.config.stage_embedding_dim)),
        dropout=float(ckpt_cfg.get("dropout", cfg.model.config.dropout)),
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

    _, _, test_sampler = build_samplers_from_anndata(
        adata=adata,
        split=split,
        latent_key=str(cfg.data.latent_key),
        stage_col=str(cfg.data.stage_col),
        donor_col=str(cfg.data.donor_col),
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

    out_dir = Path(str(cfg.output_dir)) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
