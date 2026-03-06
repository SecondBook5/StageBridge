"""Train StageBridge and benchmark variants with donor-held-out evaluation."""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any
import shutil

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from stagebridge.logging_utils import configure_root_logger, get_logger
from stagebridge.io.niche_tokens import NicheTokenBank
from stagebridge.models.baselines import DeepSetsFlowModel, LinearTransitionBaseline, NoContextFlowModel
from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.preprocessing.harmonize import add_hlca_latent, ensure_required_obs_fields
from stagebridge.preprocessing.stage_ontology import normalize_stage_series, stage_to_index
from stagebridge.runs import build_output_context
from stagebridge.training.eval import evaluate_transition
from stagebridge.training.trainer import (
    StageBridgeTrainer,
    build_donor_holdout_splits,
    build_samplers_from_anndata,
    donors_with_min_stage_coverage,
)
from stagebridge.utils.seeds import set_global_seed
from stagebridge.utils.types import RunManifest, StageBridgeConfig
from stagebridge.viz.curves import plot_training_curves
from stagebridge.viz.embeddings import plot_umap_with_trajectories
from stagebridge.viz.flows import compute_macroflow_matrix, plot_macroflow_sankey

configure_root_logger()
log = get_logger(__name__)

try:
    import anndata
except Exception as exc:  # pragma: no cover
    raise ImportError("anndata is required for train_stagebridge.py") from exc


def _set_torch_seed(seed: int) -> None:
    set_global_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _ensure_latent(adata: anndata.AnnData, cfg: DictConfig) -> None:
    target_key = str(cfg.data.latent_key)
    fallback_key = str(cfg.data.fallback_latent_key)

    if target_key in adata.obsm:
        return

    # Support latent-only AnnData where X already stores latent vectors
    # (e.g., snrna_hlca_latent_full.h5ad with var_names latent_*).
    var_names = adata.var_names.astype(str) if adata.var_names is not None else np.array([], dtype=object)
    if adata.X is not None and adata.n_vars > 0 and all(str(v).startswith("latent_") for v in var_names):
        adata.obsm[target_key] = np.asarray(adata.X, dtype=np.float32)
        log.info("Attached latent from adata.X -> adata.obsm['%s'] (latent-only h5ad).", target_key)
        return

    if fallback_key not in adata.obsm:
        X = adata.layers["log1p"] if "log1p" in adata.layers else adata.X
        from sklearn.decomposition import TruncatedSVD

        n_comp = min(int(_model_value(cfg, "input_dim")), min(X.shape) - 1)
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
        n_components=int(_model_value(cfg, "input_dim")),
    )


# HLCA cell-type labels that constitute the epithelial lineage.
# Derived from the HLCA level-3 annotation scheme.
_EPITHELIAL_HLCA_LABELS: frozenset[str] = frozenset({
    "AT1", "AT2", "Basal", "Secretory", "Ciliated",
    "KRT5- KRT17+ epithelial", "Goblet", "Neuroendocrine",
    "Squamous", "Alveolar epithelial cells",
    "Club", "Club-like", "Proliferating epithelial",
})


def _apply_lineage_filter(adata: anndata.AnnData, cfg: DictConfig) -> anndata.AnnData:
    """Filter cells to a specific lineage when ``training.lineage_filter`` is set.

    Supported values:
    * ``null`` / not set — keep all cells (default)
    * ``"epithelial"``   — keep only HLCA epithelial cell types

    Filtering requires ``adata.obs["hlca_label"]`` to be present.
    If the column is absent the filter is skipped with a warning.
    """
    lineage = OmegaConf.select(cfg, "training.lineage_filter") or None
    if lineage is None:
        return adata

    lineage = str(lineage).lower()
    if "hlca_label" not in adata.obs.columns:
        log.warning(
            "lineage_filter=%r requested but 'hlca_label' column not found in adata.obs; "
            "skipping lineage filter.",
            lineage,
        )
        return adata

    if lineage == "epithelial":
        mask = adata.obs["hlca_label"].isin(_EPITHELIAL_HLCA_LABELS)
    else:
        raise ValueError(
            f"Unknown lineage_filter={lineage!r}. Supported values: null, 'epithelial'."
        )

    n_before = adata.n_obs
    adata = adata[mask].copy()
    log.info(
        "Lineage filter '%s': %d → %d cells (%.1f%% retained).",
        lineage,
        n_before,
        adata.n_obs,
        100.0 * adata.n_obs / max(n_before, 1),
    )
    return adata


def _load_training_adata(cfg: DictConfig) -> anndata.AnnData:
    snrna_path = Path(str(cfg.data.snrna_h5ad))
    if not snrna_path.exists():
        raise FileNotFoundError(f"snRNA merged h5ad not found at {snrna_path}")

    adata_snrna = anndata.read_h5ad(snrna_path)
    ensure_required_obs_fields(adata_snrna)
    if "donor_id" in adata_snrna.obs.columns:
        donor_vals = adata_snrna.obs["donor_id"].astype(str)
        if "patient_id" not in adata_snrna.obs.columns:
            adata_snrna.obs["patient_id"] = donor_vals
        else:
            patient_vals = adata_snrna.obs["patient_id"].astype(str)
            unknown_mask = patient_vals.str.startswith("unknown_")
            if bool(unknown_mask.any()):
                adata_snrna.obs.loc[unknown_mask, "patient_id"] = donor_vals.loc[unknown_mask]
    adata_snrna.obs[str(cfg.data.stage_col)] = normalize_stage_series(adata_snrna.obs[str(cfg.data.stage_col)])
    _ensure_latent(adata=adata_snrna, cfg=cfg)
    adata_snrna.obs["modality"] = "snrna"

    if not bool(cfg.data.use_spatial):
        return _apply_lineage_filter(adata_snrna, cfg)

    spatial_path = Path(str(cfg.data.spatial_h5ad))
    if not spatial_path.exists():
        raise FileNotFoundError(
            f"Spatial merged h5ad not found at {spatial_path} while use_spatial=true"
        )

    adata_spatial = anndata.read_h5ad(spatial_path)
    ensure_required_obs_fields(adata_spatial)
    if "donor_id" in adata_spatial.obs.columns:
        donor_vals = adata_spatial.obs["donor_id"].astype(str)
        if "patient_id" not in adata_spatial.obs.columns:
            adata_spatial.obs["patient_id"] = donor_vals
        else:
            patient_vals = adata_spatial.obs["patient_id"].astype(str)
            unknown_mask = patient_vals.str.startswith("unknown_")
            if bool(unknown_mask.any()):
                adata_spatial.obs.loc[unknown_mask, "patient_id"] = donor_vals.loc[unknown_mask]
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
    return _apply_lineage_filter(adata, cfg)


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
    transition_src: str | None = None,
    transition_tgt: str | None = None,
) -> dict[str, float]:
    train_sampler, _, test_sampler = build_samplers_from_anndata(
        adata=adata,
        split=split,
        latent_key=str(cfg.data.latent_key),
        stage_col=str(cfg.data.stage_col),
        donor_col=str(cfg.data.donor_col),
        batch_cells=int(cfg.training.batch_cells),
        device=device,
        transition_src=transition_src,
        transition_tgt=transition_tgt,
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
    variants = [str(v) for v in (OmegaConf.select(cfg, "experiment.variants") or [])]

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

    for variant in variants:
        if variant == "stagebridge_sb":
            _append(
                label="stagebridge_sb",
                model_name="stagebridge",
                ablation=None,
                overrides={"use_stochastic_bridge": True},
            )
        elif variant == "stagebridge_xattn_sb":
            # Cross-attention drift + SB-CFM: the headline "impressive" variant.
            # num_seed_vectors=4 gives the drift 4 niche context tokens to attend over.
            _append(
                label="stagebridge_xattn_sb",
                model_name="stagebridge",
                ablation=None,
                overrides={
                    "use_cross_attn_drift": True,
                    "use_stochastic_bridge": True,
                    "num_seed_vectors": 4,
                },
            )
        elif variant == "stagebridge_wes":
            # WES-conditioned StageBridge: somatic driver mutation features
            # (TMB, KRAS, EGFR, TP53, STK11, KEAP1, SMAD4, BRAF) are projected
            # and concatenated to the stage embedding, conditioning the drift on
            # the patient's mutational landscape.
            _append(
                label="stagebridge_wes",
                model_name="stagebridge",
                ablation=None,
                overrides={
                    "use_wes_features": True,
                    "use_stochastic_bridge": True,
                },
            )
        elif variant == "stagebridge_xattn_wes":
            # Full model: cross-attention drift + WES features + SB-CFM.
            _append(
                label="stagebridge_xattn_wes",
                model_name="stagebridge",
                ablation=None,
                overrides={
                    "use_cross_attn_drift": True,
                    "use_stochastic_bridge": True,
                    "use_wes_features": True,
                    "num_seed_vectors": 4,
                },
            )

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
        _known_variants = {"stagebridge_sb", "stagebridge_xattn_sb", "stagebridge_wes", "stagebridge_xattn_wes"}
        if spec["model_name"] not in {"stagebridge", "deepsets", "no_context", "linear"} and spec["label"] not in _known_variants:
            continue
        deduped.append(spec)
        seen.add(spec["label"])
    return deduped


def _resolve_transition_filter(cfg: DictConfig) -> tuple[str | None, str | None]:
    src = OmegaConf.select(cfg, "training.transition_src")
    tgt = OmegaConf.select(cfg, "training.transition_tgt")
    src_val = str(src) if src is not None and str(src).strip() else None
    tgt_val = str(tgt) if tgt is not None and str(tgt).strip() else None
    if (src_val is None) != (tgt_val is None):
        raise ValueError("training.transition_src and training.transition_tgt must both be set or both be null.")
    return src_val, tgt_val


def _maybe_load_wes_lookup(cfg: DictConfig) -> "dict[tuple[str, str], np.ndarray] | None":
    """Load WES features parquet and return a (patient_id, stage) → np.ndarray lookup."""
    use_wes = bool(OmegaConf.select(cfg, "training.use_wes_features") or False)
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


def _maybe_load_niche_token_bank(cfg: DictConfig) -> NicheTokenBank | None:
    use_tokens = bool(OmegaConf.select(cfg, "training.use_niche_tokens") or False)
    if not use_tokens:
        return None
    bank_path = Path(str(OmegaConf.select(cfg, "training.niche_token_bank_path")))
    if not bank_path.exists():
        raise FileNotFoundError(
            f"training.use_niche_tokens=true but token bank not found: {bank_path}\n"
            "Run scripts/build_niche_token_bank.py first."
        )
    seed = int(OmegaConf.select(cfg, "training.niche_random_seed") or cfg.seed)
    return NicheTokenBank(bank_path, random_seed=seed)


def _plot_variant_preview(
    *,
    model: torch.nn.Module,
    adata: anndata.AnnData,
    test_sampler,
    run_dir: Path,
    run_name: str,
    label: str,
    latent_key: str,
    transition_src: str | None,
    transition_tgt: str | None,
    seed: int,
) -> dict[str, str]:
    transitions = test_sampler.available_transitions
    if not transitions:
        return {}

    if transition_src is not None and transition_tgt is not None:
        if (transition_src, transition_tgt) in transitions:
            src, tgt = transition_src, transition_tgt
        else:
            src, tgt = transitions[0]
    else:
        src, tgt = transitions[0]

    x_src, x_tgt = test_sampler.sample_transition_pair(src, tgt, n_cells=512)
    with torch.no_grad():
        c_s = model.forward_set_context(x_src)
        stage_pair_id = model.encode_stage_pair_tensor(
            stage_src=stage_to_index(src),
            stage_tgt=stage_to_index(tgt),
            n=x_src.shape[0],
            device=x_src.device,
        )
        x_pred = model.integrate_euler(x0=x_src, c_s=c_s, stage_pair_id=stage_pair_id, num_steps=8)

    x_src_np = np.asarray(x_src.detach().cpu(), dtype=np.float32)
    x_pred_np = np.asarray(x_pred.detach().cpu(), dtype=np.float32)

    if "X_pca" not in adata.obsm:
        lat = np.asarray(adata.obsm[latent_key], dtype=np.float32)
        if lat.shape[1] == 1:
            lat = np.hstack([lat, np.zeros((lat.shape[0], 1), dtype=np.float32)])
        adata.obsm["X_pca"] = lat[:, :2].astype(np.float32, copy=False)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    umap_path = fig_dir / f"{run_name}_{label}_umap_pushforward.png"
    plot_umap_with_trajectories(
        adata=adata,
        uv0=x_src_np[:, :2],
        uv1_pred=x_pred_np[:, :2],
        output_path=umap_path,
        title=f"{label}: {src}->{tgt} pushforward",
    )

    flow_matrix, src_labels, tgt_labels = compute_macroflow_matrix(
        x_src=x_src_np,
        x_tgt_pred=x_pred_np,
        n_clusters=9,
        random_state=int(seed),
    )
    sankey_path = fig_dir / f"{run_name}_{label}_sankey_macroflow.png"
    plot_macroflow_sankey(
        flow_matrix=flow_matrix,
        source_labels=src_labels,
        target_labels=tgt_labels,
        output_path=sankey_path,
        title=f"{label}: {src}->{tgt} macro coupling",
    )
    return {
        "transition_src": src,
        "transition_tgt": tgt,
        "umap_pushforward_png": str(umap_path),
        "sankey_macroflow_png": str(sankey_path),
    }


def run_training(cfg: DictConfig) -> dict[str, object]:
    _set_torch_seed(int(cfg.seed))

    adata = _load_training_adata(cfg)

    # Determine actual latent dim from loaded data; may be < model.config.input_dim
    # when data has fewer features than the configured target dimension.
    _latent_key = str(cfg.data.latent_key)
    _actual_input_dim = int(adata.obsm[_latent_key].shape[1])

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

    out_ctx = build_output_context(cfg.output_dir)
    run_dir = out_ctx.output_dir

    model_hidden_dim = int(_model_value(cfg, "hidden_dim"))
    model_vector_field_hidden_dim = int(_model_value(cfg, "vector_field_hidden_dim"))
    model_num_heads = int(_model_value(cfg, "num_heads"))
    model_num_inducing_points = int(_model_value(cfg, "num_inducing_points"))
    model_num_seed_vectors = int(_model_value(cfg, "num_seed_vectors"))
    model_num_stages = int(_model_value(cfg, "num_stages"))
    model_time_embedding_dim = int(_model_value(cfg, "time_embedding_dim"))
    model_stage_embedding_dim = int(_model_value(cfg, "stage_embedding_dim"))
    model_dropout = float(_model_value(cfg, "dropout"))

    base_cfg = StageBridgeConfig(
        input_dim=_actual_input_dim,
        hidden_dim=model_hidden_dim,
        vector_field_hidden_dim=model_vector_field_hidden_dim,
        num_heads=model_num_heads,
        num_inducing_points=model_num_inducing_points,
        num_seed_vectors=model_num_seed_vectors,
        num_stages=model_num_stages,
        time_embedding_dim=model_time_embedding_dim,
        stage_embedding_dim=model_stage_embedding_dim,
        dropout=model_dropout,
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
        sigma=float(OmegaConf.select(cfg, "training.sigma") or 0.0),
        use_stochastic_bridge=bool(OmegaConf.select(cfg, "training.use_stochastic_bridge") or False),
        use_cross_attn_drift=bool(OmegaConf.select(cfg, "training.use_cross_attn_drift") or False),
        use_wes_features=bool(OmegaConf.select(cfg, "training.use_wes_features") or False),
    )

    variant_specs = _build_variant_specs(cfg)
    summary: dict[str, dict[str, object]] = {}
    run_manifests: list[dict[str, object]] = []
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    transition_src, transition_tgt = _resolve_transition_filter(cfg)
    niche_bank = _maybe_load_niche_token_bank(cfg)
    wes_lookup = _maybe_load_wes_lookup(cfg)
    use_niche_cfg = bool(OmegaConf.select(cfg, "training.use_niche_tokens") or False)
    m_niche = int(OmegaConf.select(cfg, "training.m_niche") or 64)
    niche_sampling_strategy = str(OmegaConf.select(cfg, "training.niche_sampling_strategy") or "random_m")
    niche_fallback = str(OmegaConf.select(cfg, "training.niche_fallback") or "donor")
    niche_random_seed = int(OmegaConf.select(cfg, "training.niche_random_seed") or cfg.seed)

    stagebridge_alias_paths: dict[str, str] = {}

    for spec in variant_specs:
        label = str(spec["label"])
        model_name = str(spec["model_name"])
        overrides = dict(spec["overrides"])
        ablation = spec["ablation"]

        fold_outputs: list[dict[str, float]] = []
        checkpoints: list[str] = []
        history_payloads: list[dict[str, object]] = []
        preview_paths: dict[str, str] = {}

        variant_cfg = replace(base_cfg)
        for key, value in overrides.items():
            setattr(variant_cfg, key, value)

        variant_use_niche = bool(
            use_niche_cfg
            and (niche_bank is not None)
            and (model_name in {"stagebridge", "deepsets"})
        )

        for fold_idx, split in enumerate(splits):
            fold_name = f"{cfg.run_name}_{label}_fold{fold_idx}"

            if model_name == "linear":
                fold_metrics = _run_linear_baseline_fold(
                    adata=adata,
                    split=split,
                    cfg=cfg,
                    device=variant_cfg.resolved_device(),
                    transition_src=transition_src,
                    transition_tgt=transition_tgt,
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
                    transition_src=transition_src,
                    transition_tgt=transition_tgt,
                    niche_token_bank=niche_bank,
                    use_niche_tokens=variant_use_niche,
                    m_niche=m_niche,
                    niche_sampling_strategy=niche_sampling_strategy,
                    niche_fallback=niche_fallback,
                    niche_random_seed=niche_random_seed,
                    wes_lookup=wes_lookup if variant_cfg.use_wes_features else None,
                )

                out = trainer.fit(
                    train_sampler=train_sampler,
                    val_sampler=val_sampler,
                    test_sampler=test_sampler,
                    output_dir=run_dir,
                    run_name=fold_name,
                )

                history_payloads.append({"name": f"fold{fold_idx}", "history": out.history})
                fold_metrics = {"best_val_loss": out.best_val_loss}
                fold_metrics.update(out.benchmark_metrics)
                fold_outputs.append(fold_metrics)
                checkpoints.append(str(out.best_checkpoint))

                if fold_idx == 0:
                    preview_paths = _plot_variant_preview(
                        model=model,
                        adata=adata,
                        test_sampler=test_sampler,
                        run_dir=run_dir,
                        run_name=str(cfg.run_name),
                        label=label,
                        latent_key=str(cfg.data.latent_key),
                        transition_src=transition_src,
                        transition_tgt=transition_tgt,
                        seed=int(cfg.seed),
                    )

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
                    "niche_token_bank": str(OmegaConf.select(cfg, "training.niche_token_bank_path") or ""),
                },
                output_paths={
                    "output_dir": str(run_dir),
                    "tables_dir": str(run_dir / "tables"),
                    "figures_dir": str(run_dir / "figures"),
                },
                notes=f"fold={fold_idx}; transition={transition_src}->{transition_tgt}",
            )
            run_manifests.append(run_manifest.to_dict())

        aggregate: dict[str, float] = {}
        if fold_outputs:
            keys = sorted(fold_outputs[0])
            for key in keys:
                vals = np.array([f[key] for f in fold_outputs], dtype=float)
                aggregate[f"{key}_mean"] = float(np.nanmean(vals))
                aggregate[f"{key}_std"] = float(np.nanstd(vals))

        if history_payloads:
            loss_curve_path = run_dir / "figures" / f"{cfg.run_name}_{label}_loss_curve.png"
            plot_training_curves(history_payloads=history_payloads, output_path=loss_curve_path)
            preview_paths["loss_curve_png"] = str(loss_curve_path)

        summary[label] = {
            "model_name": model_name,
            "ablation": ablation,
            "overrides": overrides,
            "folds": fold_outputs,
            "aggregate": aggregate,
            "checkpoints": checkpoints,
            "use_niche_tokens": variant_use_niche,
            "preview_paths": preview_paths,
        }

        if label == "stagebridge":
            fig_dir = run_dir / "figures"
            alias_map = {
                "loss_curve_png": fig_dir / f"{cfg.run_name}_loss_curve.png",
                "umap_pushforward_png": fig_dir / f"{cfg.run_name}_umap_pushforward.png",
                "sankey_macroflow_png": fig_dir / f"{cfg.run_name}_sankey_macroflow.png",
            }
            for key, alias in alias_map.items():
                src = preview_paths.get(key)
                if src and Path(src).exists():
                    shutil.copyfile(src, alias)
                    stagebridge_alias_paths[key] = str(alias)

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

    ablation_rows: list[dict[str, object]] = []
    for label, block in summary.items():
        row: dict[str, object] = {
            "label": label,
            "model_name": block.get("model_name"),
            "ablation": block.get("ablation"),
            "use_niche_tokens": bool(block.get("use_niche_tokens", False)),
        }
        for key, value in block.get("aggregate", {}).items():
            row[str(key)] = value
        for metric in ("sinkhorn", "mmd_rbf", "classifier_auc", "jsd_composition", "rank_consistency"):
            mean_key = f"{metric}_mean_mean"
            std_key = f"{metric}_mean_std"
            if mean_key in row:
                row[f"{metric}_mean"] = row[mean_key]
            if std_key in row:
                row[f"{metric}_std"] = row[std_key]
        ablation_rows.append(row)
    ablation_df = pd.DataFrame(ablation_rows).sort_values("label").reset_index(drop=True)
    ablation_csv = run_dir / "tables" / "ablation_matrix.csv"
    ablation_df.to_csv(ablation_csv, index=False)

    summary_json = {
        "ok": True,
        "run_name": str(cfg.run_name),
        "metrics_json": str(metrics_path),
        "run_manifest_json": str(run_manifest_path),
        "ablation_matrix_csv": str(ablation_csv),
        "stagebridge_figures": stagebridge_alias_paths,
    }

    log.info("Training complete. Metrics saved to %s", metrics_path)
    log.info("Run manifest saved to %s", run_manifest_path)
    log.info("Ablation matrix saved to %s", ablation_csv)
    return summary_json
