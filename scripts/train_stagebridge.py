#!/usr/bin/env python
"""Train StageBridge and benchmark variants with donor-held-out evaluation."""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stagebridge.logging_utils import configure_root_logger, get_logger
from stagebridge.models.baselines import DeepSetsFlowModel, LinearTransitionBaseline, NoContextFlowModel
from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.preprocessing.harmonize import add_hlca_latent, ensure_required_obs_fields
from stagebridge.preprocessing.stage_ontology import normalize_stage_series, stage_to_index
from stagebridge.training.eval import evaluate_transition
from stagebridge.training.trainer import (
    StageBridgeTrainer,
    build_donor_holdout_splits,
    build_samplers_from_anndata,
    donors_with_min_stage_coverage,
)
from stagebridge.utils.seeds import set_global_seed
from stagebridge.utils.types import RunManifest, StageBridgeConfig

configure_root_logger()
log = get_logger(__name__)

try:
    import anndata
except Exception as exc:  # pragma: no cover
    raise ImportError("anndata is required for train_stagebridge.py") from exc

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "hydra-core and omegaconf are required. Install updated environment.yml dependencies."
    ) from exc


def _set_torch_seed(seed: int) -> None:
    set_global_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stable_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _ensure_latent(adata: anndata.AnnData, cfg: DictConfig) -> None:
    target_key = str(cfg.data.latent_key)
    fallback_key = str(cfg.data.fallback_latent_key)

    if target_key in adata.obsm:
        return

    if fallback_key not in adata.obsm:
        X = adata.layers["log1p"] if "log1p" in adata.layers else adata.X
        from sklearn.decomposition import TruncatedSVD

        n_comp = min(int(cfg.model.input_dim), min(X.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=int(cfg.seed))
        adata.obsm[fallback_key] = svd.fit_transform(X).astype(np.float32)
        log.info("Computed fallback latent '%s' with %d components.", fallback_key, n_comp)

    reference = None
    if bool(cfg.data.use_hlca_reference):
        from stagebridge.io.hlca import load_hlca_reference

        hlca_path = Path(str(cfg.data.hlca_h5ad))
        if hlca_path.exists():
            reference = load_hlca_reference(h5ad_path=hlca_path, backed=None)
        else:
            log.warning("HLCA reference not found at %s; using fallback latent alignment.", hlca_path)

    add_hlca_latent(
        adata=adata,
        reference=reference,
        source_key=fallback_key,
        output_key=target_key,
        n_components=int(cfg.model.input_dim),
    )


def _load_training_adata(cfg: DictConfig) -> anndata.AnnData:
    snrna_path = Path(str(cfg.data.snrna_h5ad))
    if not snrna_path.exists():
        raise FileNotFoundError(f"snRNA merged h5ad not found at {snrna_path}")

    adata_snrna = anndata.read_h5ad(snrna_path)
    ensure_required_obs_fields(adata_snrna)
    adata_snrna.obs[str(cfg.data.stage_col)] = normalize_stage_series(adata_snrna.obs[str(cfg.data.stage_col)])
    _ensure_latent(adata=adata_snrna, cfg=cfg)
    adata_snrna.obs["modality"] = "snrna"

    if not bool(cfg.data.use_spatial):
        return adata_snrna

    spatial_path = Path(str(cfg.data.spatial_h5ad))
    if not spatial_path.exists():
        raise FileNotFoundError(
            f"Spatial merged h5ad not found at {spatial_path} while use_spatial=true"
        )

    adata_spatial = anndata.read_h5ad(spatial_path)
    ensure_required_obs_fields(adata_spatial)
    adata_spatial.obs[str(cfg.data.stage_col)] = normalize_stage_series(adata_spatial.obs[str(cfg.data.stage_col)])
    _ensure_latent(adata=adata_spatial, cfg=cfg)
    adata_spatial.obs["modality"] = "spatial"

    # Merge modalities while preserving the shared latent representation used by training.
    latent_key = str(cfg.data.latent_key)
    adata = anndata.concat([adata_snrna, adata_spatial], join="outer", merge="same")
    adata.obs_names_make_unique()

    # Concatenate latent coordinates manually because AnnData concat does not merge obsm arrays by default.
    X_latent = np.vstack(
        [
            np.asarray(adata_snrna.obsm[latent_key], dtype=np.float32),
            np.asarray(adata_spatial.obsm[latent_key], dtype=np.float32),
        ]
    )
    adata.obsm[latent_key] = X_latent
    return adata


def _build_model(model_name: str, sb_cfg: StageBridgeConfig) -> torch.nn.Module:
    if model_name == "stagebridge":
        return StageBridgeModel(config=sb_cfg)
    if model_name == "deepsets":
        return DeepSetsFlowModel(config=sb_cfg)
    if model_name == "no_context":
        return NoContextFlowModel(config=sb_cfg)
    raise ValueError(f"Unsupported model '{model_name}'.")


def _run_linear_baseline_fold(
    adata: anndata.AnnData,
    split,
    cfg: DictConfig,
    device: str,
) -> dict[str, float]:
    train_sampler, _, test_sampler = build_samplers_from_anndata(
        adata=adata,
        split=split,
        latent_key=str(cfg.data.latent_key),
        stage_col=str(cfg.data.stage_col),
        donor_col=str(cfg.data.donor_col),
        batch_cells=int(cfg.training.batch_cells),
        device=device,
    )

    src_stage, tgt_stage = train_sampler.available_transitions[0]
    x_src_fit, x_tgt_fit = train_sampler.sample_transition_pair(src_stage, tgt_stage, n_cells=1024)
    linear = LinearTransitionBaseline()
    linear.fit(x_src_fit.detach().cpu().numpy(), x_tgt_fit.detach().cpu().numpy())

    class _LinearWrapper(torch.nn.Module):
        def __init__(self, linear_model: LinearTransitionBaseline) -> None:
            super().__init__()
            self.linear_model = linear_model

        def encode_stage_pair_tensor(self, stage_src: int, stage_tgt: int, n: int, device: torch.device):
            return torch.zeros((n,), dtype=torch.long, device=device)

        def forward_set_context(self, x_set: torch.Tensor, mask=None):
            return torch.zeros((1, x_set.shape[1]), device=x_set.device, dtype=x_set.dtype)

        def forward_vector_field(self, x_t, t, c_s, stage_pair_id):
            pred = self.integrate_euler(x_t, c_s, stage_pair_id)
            return pred - x_t

        def integrate_euler(self, x0, c_s, stage_pair_id, num_steps: int = 1):
            pred = self.linear_model.predict(x0.detach().cpu().numpy()).astype(np.float32)
            return torch.tensor(pred, device=x0.device, dtype=x0.dtype)

    wrapper = _LinearWrapper(linear)
    metrics = []
    for src, tgt in test_sampler.available_transitions:
        x_src, x_tgt = test_sampler.sample_transition_pair(src, tgt, n_cells=256)
        out = evaluate_transition(
            model=wrapper,
            x_src=x_src,
            x_tgt=x_tgt,
            stage_src=stage_to_index(src),
            stage_tgt=stage_to_index(tgt),
            num_steps=1,
            ot_epsilon=float(cfg.training.ot_epsilon),
            sinkhorn_iters=int(cfg.training.sinkhorn_iters),
        )
        metrics.append(
            {
                "sinkhorn": out.sinkhorn,
                "mmd_rbf": out.mmd_rbf,
                "classifier_auc": out.classifier_auc,
                "jsd_composition": out.jsd_composition,
                "rank_consistency": out.rank_consistency,
            }
        )

    agg: dict[str, float] = {}
    for key in metrics[0]:
        vals = np.array([m[key] for m in metrics], dtype=float)
        agg[f"{key}_mean"] = float(np.nanmean(vals))
        agg[f"{key}_std"] = float(np.nanstd(vals))
    return agg


def _build_variant_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    primary = str(cfg.experiment.primary_model)
    baselines = [str(m) for m in cfg.experiment.baseline_models]
    ablations = [str(a) for a in cfg.experiment.ablations]

    specs: list[dict[str, Any]] = []

    def _append(label: str, model_name: str, ablation: str | None, overrides: dict[str, Any]) -> None:
        specs.append(
            {
                "label": label,
                "model_name": model_name,
                "ablation": ablation,
                "overrides": overrides,
            }
        )

    _append(label=primary, model_name=primary, ablation=None, overrides={})
    for model_name in baselines:
        _append(label=model_name, model_name=model_name, ablation=None, overrides={})

    for ablation in ablations:
        if ablation == "no_ot":
            _append(
                label=f"{primary}__ablation_no_ot",
                model_name=primary,
                ablation="no_ot",
                overrides={"use_ot": False},
            )
        elif ablation == "no_context":
            _append(
                label=f"{primary}__ablation_no_context",
                model_name="no_context",
                ablation="no_context",
                overrides={},
            )
        elif ablation == "no_stage_embedding":
            _append(
                label=f"{primary}__ablation_no_stage_embedding",
                model_name=primary,
                ablation="no_stage_embedding",
                overrides={"use_stage_embedding": False},
            )
        elif ablation == "no_context_consistency":
            _append(
                label=f"{primary}__ablation_no_context_consistency",
                model_name=primary,
                ablation="no_context_consistency",
                overrides={"context_consistency_weight": 0.0},
            )

    # Deduplicate by label while preserving order.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        if spec["label"] in seen:
            continue
        if spec["model_name"] not in {"stagebridge", "deepsets", "no_context", "linear"}:
            continue
        deduped.append(spec)
        seen.add(spec["label"])
    return deduped


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    _set_torch_seed(int(cfg.seed))

    adata = _load_training_adata(cfg)

    if adata.n_obs > int(cfg.data.max_cells):
        rng = np.random.default_rng(int(cfg.seed))
        idx = rng.choice(adata.n_obs, size=int(cfg.data.max_cells), replace=False)
        adata = adata[idx].copy()
        log.info("Subsampled to %d cells for training.", adata.n_obs)

    donor_ids = donors_with_min_stage_coverage(
        obs_stage=np.asarray(adata.obs[str(cfg.data.stage_col)].astype(str)),
        obs_donor=np.asarray(adata.obs[str(cfg.data.donor_col)].astype(str)),
        min_stages=int(cfg.splits.min_stages_per_donor),
    )
    if len(donor_ids) < int(cfg.splits.n_folds):
        raise ValueError(
            f"Not enough donors ({len(donor_ids)}) for {cfg.splits.n_folds}-fold CV."
        )

    splits = build_donor_holdout_splits(
        donor_ids=donor_ids,
        n_folds=int(cfg.splits.n_folds),
        seed=int(cfg.seed),
    )

    run_dir = Path(str(cfg.output_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    base_cfg = StageBridgeConfig(
        input_dim=int(cfg.model.input_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        vector_field_hidden_dim=int(cfg.model.vector_field_hidden_dim),
        num_heads=int(cfg.model.num_heads),
        num_inducing_points=int(cfg.model.num_inducing_points),
        num_seed_vectors=int(cfg.model.num_seed_vectors),
        num_stages=int(cfg.model.num_stages),
        time_embedding_dim=int(cfg.model.time_embedding_dim),
        stage_embedding_dim=int(cfg.model.stage_embedding_dim),
        dropout=float(cfg.model.dropout),
        ot_epsilon=float(cfg.training.ot_epsilon),
        sinkhorn_iters=int(cfg.training.sinkhorn_iters),
        num_ot_pairs=int(cfg.training.num_ot_pairs),
        context_consistency_weight=float(cfg.training.context_consistency_weight),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        grad_clip_norm=float(cfg.training.grad_clip_norm),
        max_epochs=int(cfg.training.max_epochs),
        steps_per_epoch=int(cfg.training.steps_per_epoch),
        val_steps=int(cfg.training.val_steps),
        patience=int(cfg.training.patience),
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        mixed_precision=bool(cfg.training.mixed_precision),
        device=str(cfg.training.device),
        seed=int(cfg.training.seed),
        use_ot=bool(cfg.training.use_ot),
        use_stage_embedding=bool(cfg.training.use_stage_embedding),
    )

    variant_specs = _build_variant_specs(cfg)
    summary: dict[str, dict[str, object]] = {}
    run_manifests: list[dict[str, object]] = []
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)

    for spec in variant_specs:
        label = str(spec["label"])
        model_name = str(spec["model_name"])
        overrides = dict(spec["overrides"])
        ablation = spec["ablation"]

        fold_outputs: list[dict[str, float]] = []
        checkpoints: list[str] = []

        variant_cfg = replace(base_cfg)
        for key, value in overrides.items():
            setattr(variant_cfg, key, value)

        for fold_idx, split in enumerate(splits):
            fold_name = f"{cfg.run_name}_{label}_fold{fold_idx}"

            if model_name == "linear":
                fold_metrics = _run_linear_baseline_fold(
                    adata=adata,
                    split=split,
                    cfg=cfg,
                    device=variant_cfg.resolved_device(),
                )
                fold_outputs.append(fold_metrics)
                checkpoints.append("linear_baseline")
            else:
                model = _build_model(model_name=model_name, sb_cfg=variant_cfg)
                trainer = StageBridgeTrainer(model=model, config=variant_cfg)

                train_sampler, val_sampler, test_sampler = build_samplers_from_anndata(
                    adata=adata,
                    split=split,
                    latent_key=str(cfg.data.latent_key),
                    stage_col=str(cfg.data.stage_col),
                    donor_col=str(cfg.data.donor_col),
                    batch_cells=int(cfg.training.batch_cells),
                    device=variant_cfg.resolved_device(),
                )

                out = trainer.fit(
                    train_sampler=train_sampler,
                    val_sampler=val_sampler,
                    test_sampler=test_sampler,
                    output_dir=run_dir,
                    run_name=fold_name,
                )

                fold_metrics = {"best_val_loss": out.best_val_loss}
                fold_metrics.update(out.benchmark_metrics)
                fold_outputs.append(fold_metrics)
                checkpoints.append(str(out.best_checkpoint))

            run_manifest = RunManifest(
                run_name=fold_name,
                task="train",
                model_name=model_name,
                variant_label=label,
                ablation=ablation,
                seed=int(cfg.seed),
                device_requested=str(cfg.training.device),
                device_resolved=variant_cfg.resolved_device(),
                config_path="configs/train.yaml",
                config_hash=_stable_hash(
                    {
                        "config": resolved_cfg,
                        "variant": spec,
                        "fold_index": fold_idx,
                    }
                ),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                data_paths={
                    "snrna_h5ad": str(cfg.data.snrna_h5ad),
                    "spatial_h5ad": str(cfg.data.spatial_h5ad),
                    "hlca_h5ad": str(cfg.data.hlca_h5ad),
                },
                output_paths={
                    "output_dir": str(run_dir),
                    "tables_dir": str(run_dir / "tables"),
                    "figures_dir": str(run_dir / "figures"),
                },
                notes=f"fold={fold_idx}",
            )
            run_manifests.append(run_manifest.to_dict())

        aggregate: dict[str, float] = {}
        if fold_outputs:
            keys = sorted(fold_outputs[0])
            for key in keys:
                vals = np.array([f[key] for f in fold_outputs], dtype=float)
                aggregate[f"{key}_mean"] = float(np.nanmean(vals))
                aggregate[f"{key}_std"] = float(np.nanstd(vals))

        summary[label] = {
            "model_name": model_name,
            "ablation": ablation,
            "overrides": overrides,
            "folds": fold_outputs,
            "aggregate": aggregate,
            "checkpoints": checkpoints,
        }

    payload = {
        "run_name": str(cfg.run_name),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "results": summary,
    }
    metrics_path = run_dir / "tables" / f"metrics_{cfg.run_name}.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    cfg_snapshot = run_dir / "tables" / f"config_{cfg.run_name}.json"
    with cfg_snapshot.open("w", encoding="utf-8") as fh:
        json.dump(OmegaConf.to_container(cfg, resolve=True), fh, indent=2)

    run_manifest_path = run_dir / "tables" / f"run_manifest_{cfg.run_name}.json"
    with run_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump({"run_name": str(cfg.run_name), "entries": run_manifests}, fh, indent=2)

    log.info("Training complete. Metrics saved to %s", metrics_path)
    log.info("Run manifest saved to %s", run_manifest_path)


if __name__ == "__main__":
    main()
