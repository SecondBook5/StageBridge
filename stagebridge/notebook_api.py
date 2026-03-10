"""Notebook-facing orchestration API for the rebuilt StageBridge layout."""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Callable
from collections import Counter

import anndata
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.utils.h5ad_io import (
    read_h5ad_obs_frame as _read_h5ad_obs_frame_shared,
    read_h5ad_spatial_coords,
    read_h5ad_n_vars,
)

from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.data.luad_evo.stages import normalize_stage_label
from stagebridge.data.luad_evo.wes import WES_FEATURE_COLS, load_luad_evo_wes_features
from stagebridge.evaluation.provider_benchmark import render_provider_benchmark_md, summarize_provider_benchmark
from stagebridge.utils.config_loader import load_yaml_config

_CONFIG_DIR = (Path(__file__).resolve().parent.parent / "configs").resolve()
_DEFAULT_CONFIG = _CONFIG_DIR / "default.yaml"
_COMPONENT_DIRS = {
    "data": _CONFIG_DIR / "data",
    "labels": _CONFIG_DIR / "labels",
    "spatial_mapping": _CONFIG_DIR / "spatial_mapping",
    "context_model": _CONFIG_DIR / "context_model",
    "splits": _CONFIG_DIR / "splits",
    "train": _CONFIG_DIR / "train",
    "evaluation": _CONFIG_DIR / "evaluation",
}
_TRANSITION_COMPONENTS = [
    _CONFIG_DIR / "transition_model" / "disease_edges.yaml",
    _CONFIG_DIR / "transition_model" / "gaussian_init.yaml",
    _CONFIG_DIR / "transition_model" / "stochastic_dynamics.yaml",
    _CONFIG_DIR / "transition_model" / "schrodinger_bridge.yaml",
    _CONFIG_DIR / "transition_model" / "wes_regularizer.yaml",
]
_ENTRYPOINT_ALIASES = {
    "config": "default",
    "train": "default",
    "eval": "default",
}

StepFn = Callable[..., dict[str, Any]]
StepSpec = tuple[str, str]


def _load_component(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing config component: {path}")
    return OmegaConf.create(load_yaml_config(path))


def _selected_profiles(base_cfg: DictConfig) -> dict[str, str]:
    defaults = {
        "data": "luad_evo",
        "labels": "repair",
        "spatial_mapping": "tangram",
        "context_model": "set_only",
        "splits": "donor_holdout",
        "train": "full_v1",
        "evaluation": "baseline",
    }
    profiles = OmegaConf.to_container(base_cfg.get("profiles", {}), resolve=False) or {}
    return {key: str(profiles.get(key, value)) for key, value in defaults.items()}


def compose_config(entrypoint: str = "default", overrides: list[str] | None = None) -> DictConfig:
    """Compose a config tree from the normalized repo layout."""
    entrypoint = _ENTRYPOINT_ALIASES.get(entrypoint, entrypoint)
    if entrypoint != "default":
        raise ValueError(f"Unsupported config entrypoint '{entrypoint}'. Use 'default'.")

    overrides = list(overrides or [])
    cfg = _load_component(_DEFAULT_CONFIG)
    selected = _selected_profiles(cfg)

    if entrypoint == "default":
        pass

    if "train=smoke" not in overrides and "evaluation=baseline" not in overrides:
        if entrypoint == "train":
            selected["train"] = "full_v1"
        if entrypoint == "eval":
            selected["evaluation"] = "baseline"

    dotlist_overrides: list[str] = []
    for override in overrides:
        if "=" not in override:
            dotlist_overrides.append(override)
            continue
        key, value = override.split("=", 1)
        if "." not in key and key in _COMPONENT_DIRS:
            selected[key] = value
            continue
        dotlist_overrides.append(override)

    for group, profile in selected.items():
        cfg = OmegaConf.merge(cfg, _load_component(_COMPONENT_DIRS[group] / f"{profile}.yaml"))
    for component_path in _TRANSITION_COMPONENTS:
        cfg = OmegaConf.merge(cfg, _load_component(component_path))
    if dotlist_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist_overrides))
    return cfg


def clone_config(cfg: DictConfig) -> DictConfig:
    """Create a mutable clone of a composed config."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _progress_iter(iterable: list[str], *, desc: str, enabled: bool) -> Any:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, unit="run")
    except Exception:
        return iterable


_STEP_REGISTRY: dict[str, StepSpec] = {
    "label_repair": ("stagebridge.pipelines.run_label_repair", "run_label_repair"),
    "pretrain_local": ("stagebridge.pipelines.pretrain_local", "run_pretrain_local"),
    "train_lesion": ("stagebridge.pipelines.train_lesion", "run_train_lesion"),
    "evaluate_lesion": ("stagebridge.pipelines.evaluate_lesion", "run_evaluate_lesion"),
    "eamist_report": ("stagebridge.pipelines.run_eamist_reporting", "run_eamist_reporting"),
    "reference": ("stagebridge.pipelines.run_reference", "run_reference"),
    "spatial_mapping": ("stagebridge.pipelines.run_spatial_mapping", "run_spatial_mapping"),
    "context_model": ("stagebridge.pipelines.run_context_model", "run_context_model"),
    "transition_model": ("stagebridge.pipelines.run_transition_model", "run_transition_model"),
    "evaluation": ("stagebridge.pipelines.run_evaluation", "run_evaluation"),
    "full": ("stagebridge.pipelines.run_full", "run_full"),
    # compatibility aliases retained at the API boundary only
    "build_snrna": ("stagebridge.pipelines.run_reference", "run_reference"),
    "build_spatial": ("stagebridge.pipelines.run_reference", "run_reference"),
    "map_hlca": ("stagebridge.pipelines.run_reference", "run_reference"),
    "run_tangram": ("stagebridge.pipelines.run_spatial_mapping", "run_spatial_mapping"),
    "train": ("stagebridge.pipelines.run_transition_model", "run_transition_model"),
    "evaluate": ("stagebridge.pipelines.run_evaluation", "run_evaluation"),
}


def _resolve_step_fn(step: str) -> StepFn:
    spec = _STEP_REGISTRY.get(step)
    if spec is None:
        valid = ", ".join(sorted(_STEP_REGISTRY))
        raise ValueError(f"Unknown step '{step}'. Valid steps: {valid}")
    module_name, fn_name = spec
    module = import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


def run_label_repair(*args, **kwargs):
    return _resolve_step_fn("label_repair")(*args, **kwargs)


def run_pretrain_local(*args, **kwargs):
    return _resolve_step_fn("pretrain_local")(*args, **kwargs)


def run_train_lesion(*args, **kwargs):
    return _resolve_step_fn("train_lesion")(*args, **kwargs)


def run_evaluate_lesion(*args, **kwargs):
    return _resolve_step_fn("evaluate_lesion")(*args, **kwargs)


def run_eamist_reporting(*args, **kwargs):
    return _resolve_step_fn("eamist_report")(*args, **kwargs)


def run_reference(*args, **kwargs):
    return _resolve_step_fn("reference")(*args, **kwargs)


def run_spatial_mapping(*args, **kwargs):
    return _resolve_step_fn("spatial_mapping")(*args, **kwargs)


def run_context_model(*args, **kwargs):
    return _resolve_step_fn("context_model")(*args, **kwargs)


def run_transition_model(*args, **kwargs):
    return _resolve_step_fn("transition_model")(*args, **kwargs)


def run_evaluation(*args, **kwargs):
    return _resolve_step_fn("evaluation")(*args, **kwargs)


def run_full(*args, **kwargs):
    return _resolve_step_fn("full")(*args, **kwargs)


def run_step(step: str, cfg: DictConfig) -> dict[str, Any]:
    """Run one pipeline step from the rebuilt pipeline namespace."""
    fn = _resolve_step_fn(step)
    return fn(cfg)


def run_pipeline(cfg: DictConfig, steps: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Run the default StageBridge v1 orchestration order."""
    ordered_steps = steps or [
        "reference",
        "spatial_mapping",
        "context_model",
        "transition_model",
        "evaluation",
    ]
    outputs: dict[str, dict[str, Any]] = {}
    for step in ordered_steps:
        if step == "reference":
            outputs[step] = _resolve_step_fn("reference")(cfg)
        elif step == "spatial_mapping":
            outputs[step] = _resolve_step_fn("spatial_mapping")(cfg, reference_output=outputs.get("reference"))
        elif step == "context_model":
            outputs[step] = _resolve_step_fn("context_model")(cfg, spatial_output=outputs.get("spatial_mapping"))
        elif step == "transition_model":
            outputs[step] = _resolve_step_fn("transition_model")(
                cfg,
                reference_output=outputs.get("reference"),
                spatial_output=outputs.get("spatial_mapping"),
                context_output=outputs.get("context_model"),
            )
        elif step == "evaluation":
            outputs[step] = _resolve_step_fn("evaluation")(
                cfg,
                transition_output=outputs.get("transition_model"),
                context_output=outputs.get("context_model"),
            )
        else:
            outputs[step] = run_step(step, cfg)
    return outputs


def build_step_status_table(pipeline_output: dict[str, Any]) -> pd.DataFrame:
    """Build a compact step-status table for notebook display."""
    rows: list[dict[str, Any]] = []
    for step_name, payload in pipeline_output.get("steps", pipeline_output).items():
        rows.append(
            {
                "step": step_name,
                "ok": bool(payload.get("ok", False)),
                "status": payload.get("status", "n/a"),
                "pipeline": payload.get("pipeline", step_name),
            }
        )
    return pd.DataFrame(rows)


def _normalize_dataset_obs(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    if "patient_id" not in out.columns and "donor_id" in out.columns:
        out["patient_id"] = out["donor_id"].astype(str)
    if "donor_id" not in out.columns and "patient_id" in out.columns:
        out["donor_id"] = out["patient_id"].astype(str)
    if "sample_id" not in out.columns:
        out["sample_id"] = out.index.astype(str)
    if "stage" in out.columns:
        out["stage"] = out["stage"].astype(str).map(normalize_stage_label)
    out["donor_id"] = out["donor_id"].astype(str)
    out["patient_id"] = out["patient_id"].astype(str)
    out["sample_id"] = out["sample_id"].astype(str)
    return out


def _read_h5ad_obs_frame(path: Path, *, columns: list[str]) -> pd.DataFrame:
    return _read_h5ad_obs_frame_shared(path, columns=columns)


_read_h5ad_spatial_coords = read_h5ad_spatial_coords
_read_h5ad_n_vars = read_h5ad_n_vars


def _sample_rows_by_stage(
    obs: pd.DataFrame,
    *,
    stages: list[str] | None,
    max_rows_per_stage: int,
    seed: int,
) -> np.ndarray:
    mask = np.ones(obs.shape[0], dtype=bool)
    if stages:
        wanted = {normalize_stage_label(stage) for stage in stages}
        mask &= obs["stage"].isin(wanted).to_numpy()
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return positions
    if max_rows_per_stage <= 0:
        return positions
    rng = np.random.default_rng(int(seed))
    chosen = np.zeros(obs.shape[0], dtype=bool)
    masked_stages = obs.iloc[positions]["stage"].to_numpy()
    for stage_name in pd.unique(masked_stages):
        stage_rows = positions[masked_stages == stage_name]
        if stage_rows.shape[0] <= max_rows_per_stage:
            chosen[stage_rows] = True
            continue
        keep = rng.choice(stage_rows, size=int(max_rows_per_stage), replace=False)
        chosen[keep] = True
    return np.flatnonzero(mask & chosen)


def _two_dimensional_embedding(matrix: Any, *, seed: int) -> np.ndarray:
    arr = matrix
    try:
        import scipy.sparse as sp
        from sklearn.decomposition import PCA, TruncatedSVD

        if sp.issparse(arr):
            arr = arr.tocsr().astype(np.float32, copy=False)
            arr = arr.copy()
            arr.data = np.log1p(arr.data)
            n_eff = max(2, min(8, int(arr.shape[0]) - 1, int(arr.shape[1]) - 1))
            emb = TruncatedSVD(n_components=n_eff, random_state=int(seed)).fit_transform(arr).astype(np.float32)
        else:
            arr = np.asarray(arr, dtype=np.float32)
            arr = np.log1p(np.clip(arr, 0.0, None))
            n_eff = max(2, min(8, int(arr.shape[0]) - 1, int(arr.shape[1]) - 1))
            emb = PCA(n_components=n_eff, random_state=int(seed)).fit_transform(arr).astype(np.float32)
    except Exception:
        arr = np.asarray(arr, dtype=np.float32)
        emb = arr[:, : min(2, arr.shape[1])]

    if emb.shape[1] >= 2:
        return emb[:, :2]
    padded = np.zeros((emb.shape[0], 2), dtype=np.float32)
    padded[:, : emb.shape[1]] = emb
    return padded


def _umap_embedding(matrix: Any, *, seed: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}.")
    if arr.shape[0] < 3:
        return _two_dimensional_embedding(arr, seed=seed)
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(20, max(2, arr.shape[0] - 1)),
            min_dist=0.15,
            random_state=int(seed),
        )
        emb = reducer.fit_transform(arr).astype(np.float32)
    except Exception:
        emb = _two_dimensional_embedding(arr, seed=seed)
    if emb.shape[1] >= 2:
        return emb[:, :2]
    padded = np.zeros((emb.shape[0], 2), dtype=np.float32)
    padded[:, : emb.shape[1]] = emb
    return padded


def _select_spatial_panel_genes(var_names: pd.Index) -> tuple[list[str], dict[str, str], bool]:
    var_lookup = {str(name).upper(): str(name) for name in var_names.astype(str)}
    requested = {
        "epithelial": ["EPCAM", "KRT19", "KRT8", "EPCAM"],
        "stromal": ["COL1A1", "COL1A2", "DCN", "COL3A1"],
        "immune": ["PTPRC", "LYZ", "CD3D", "HLA-DRA"],
        "vascular_program": ["VWF", "EMCN", "KDR", "PECAM1"],
    }
    chosen: list[str] = []
    roles: dict[str, str] = {}
    used_proxy = False
    for role, candidates in requested.items():
        selected = None
        for gene in candidates:
            selected = var_lookup.get(gene.upper())
            if selected is not None:
                break
        if selected is None:
            used_proxy = True
            continue
        chosen.append(selected)
        roles[selected] = role
        if selected.upper() != candidates[0].upper():
            used_proxy = True
    return chosen, roles, used_proxy


def run_data_preprocessing_overview(
    cfg: DictConfig,
    *,
    max_cells_per_stage: int = 256,
    max_spots_per_stage: int = 256,
) -> dict[str, Any]:
    """Load notebook-friendly data previews for snRNA, Visium, and WES."""
    paths = resolve_luad_evo_paths(cfg)
    stages = list(cfg.get("data", {}).get("stages", [])) or None
    seed = int(cfg.get("seed", 42))

    snrna_adata = anndata.read_h5ad(paths.snrna_latent_h5ad, backed="r")
    try:
        snrna_obs = _normalize_dataset_obs(snrna_adata.obs.copy())
        snrna_rows = _sample_rows_by_stage(
            snrna_obs,
            stages=stages,
            max_rows_per_stage=int(max_cells_per_stage),
            seed=seed,
        )
        snrna_obs_view = snrna_obs.iloc[snrna_rows].reset_index(drop=True)
        snrna_matrix = np.asarray(snrna_adata.X[snrna_rows], dtype=np.float32)
        snrna_pca = _two_dimensional_embedding(snrna_matrix, seed=seed)
        snrna_umap = _umap_embedding(snrna_matrix, seed=seed)
        snrna_label_col = "hlca_label" if "hlca_label" in snrna_obs_view.columns else None
        snrna_top_labels: list[tuple[str, int]] = []
        if snrna_label_col is not None:
            snrna_top_labels = list(
                snrna_obs_view[snrna_label_col].astype(str).value_counts().head(8).items()
            )
        snrna_n_genes = int(snrna_adata.shape[1])
    finally:
        try:
            snrna_adata.file.close()
        except Exception:
            pass

    spatial_obs = _normalize_dataset_obs(
        _read_h5ad_obs_frame(
            paths.spatial_h5ad,
            columns=["spot_id", "barcode", "donor_id", "patient_id", "stage", "sample_id"],
        )
    )
    spatial_rows = _sample_rows_by_stage(
        spatial_obs,
        stages=stages,
        max_rows_per_stage=int(max_spots_per_stage),
        seed=seed,
    )
    spatial_obs_view = spatial_obs.iloc[spatial_rows].reset_index(drop=True)
    spatial_coords = _read_h5ad_spatial_coords(paths.spatial_h5ad, spatial_rows)
    spatial_adata = anndata.read_h5ad(paths.spatial_h5ad, backed="r")
    try:
        spatial_var_names = pd.Index(spatial_adata.var_names.astype(str))
        panel_genes, panel_roles, used_proxy = _select_spatial_panel_genes(spatial_var_names)
        if panel_genes:
            gene_indices = [int(spatial_var_names.get_loc(gene)) for gene in panel_genes]
            panel_matrix = spatial_adata.X[spatial_rows][:, gene_indices]
            if hasattr(panel_matrix, "toarray"):
                panel_matrix = panel_matrix.toarray()
            panel_frame = pd.DataFrame(
                np.asarray(panel_matrix, dtype=np.float32),
                columns=panel_genes,
            )
        else:
            panel_frame = pd.DataFrame(index=np.arange(spatial_rows.shape[0]))
    finally:
        try:
            spatial_adata.file.close()
        except Exception:
            pass

    wes = load_luad_evo_wes_features(cfg, stages=stages)
    wes_frame = wes.frame.copy()

    return {
        "ok": True,
        "status": "complete",
        "snrna": {
            "obs": snrna_obs_view,
            "latent": snrna_matrix,
            "pca_embedding": snrna_pca,
            "umap_embedding": snrna_umap,
            "source_path": str(paths.snrna_latent_h5ad),
            "n_cells": int(snrna_obs_view.shape[0]),
            "n_genes": snrna_n_genes,
            "n_donors": int(snrna_obs_view["donor_id"].nunique()),
            "n_samples": int(snrna_obs_view["sample_id"].nunique()),
            "stage_counts": snrna_obs_view["stage"].value_counts().to_dict(),
            "sample_stage_counts": pd.crosstab(
                snrna_obs_view["sample_id"].astype(str),
                snrna_obs_view["stage"].astype(str),
            ),
            "top_labels": snrna_top_labels,
        },
        "spatial": {
            "obs": spatial_obs_view,
            "coords": spatial_coords,
            "source_path": str(paths.spatial_h5ad),
            "n_spots": int(spatial_obs_view.shape[0]),
            "n_genes": _read_h5ad_n_vars(paths.spatial_h5ad),
            "n_donors": int(spatial_obs_view["donor_id"].nunique()),
            "n_samples": int(spatial_obs_view["sample_id"].nunique()),
            "stage_counts": spatial_obs_view["stage"].value_counts().to_dict(),
            "feature_panel": panel_frame,
            "feature_panel_genes": panel_genes,
            "feature_panel_roles": panel_roles,
            "feature_panel_uses_proxy_genes": used_proxy,
        },
        "wes": {
            "frame": wes_frame,
            "feature_columns": list(wes.feature_columns),
            "source_path": str(wes.source_path),
            "n_rows": int(wes_frame.shape[0]),
            "n_donors": int(wes_frame["patient_id"].nunique()) if not wes_frame.empty else 0,
            "n_stages": int(wes_frame["stage"].nunique()) if not wes_frame.empty else 0,
            "stage_counts": wes_frame["stage"].value_counts().to_dict(),
            "tmb_mean": float(wes_frame["tmb"].mean()) if not wes_frame.empty else float("nan"),
            "mutation_prevalence": {
                column: float(wes_frame[column].mean()) if column in wes_frame.columns and not wes_frame.empty else float("nan")
                for column in WES_FEATURE_COLS
            },
        },
    }


def build_dataset_preprocessing_table(data_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize the notebook-driven dataset preprocessing preview."""
    snrna = data_output.get("snrna", {})
    spatial = data_output.get("spatial", {})
    wes = data_output.get("wes", {})
    rows = [
        {
            "modality": "snRNA-seq",
            "n_obs": snrna.get("n_cells", 0),
            "n_features": snrna.get("n_genes", 0),
            "n_donors": snrna.get("n_donors", 0),
            "n_samples": snrna.get("n_samples", 0),
            "n_stage_groups": len(snrna.get("stage_counts", {})),
            "source_path": snrna.get("source_path", "n/a"),
        },
        {
            "modality": "Visium",
            "n_obs": spatial.get("n_spots", 0),
            "n_features": spatial.get("n_genes", 0),
            "n_donors": spatial.get("n_donors", 0),
            "n_samples": spatial.get("n_samples", 0),
            "n_stage_groups": len(spatial.get("stage_counts", {})),
            "source_path": spatial.get("source_path", "n/a"),
        },
        {
            "modality": "WES",
            "n_obs": wes.get("n_rows", 0),
            "n_features": len(wes.get("feature_columns", [])),
            "n_donors": wes.get("n_donors", 0),
            "n_samples": 0,
            "n_stage_groups": wes.get("n_stages", 0),
            "source_path": wes.get("source_path", "n/a"),
        },
    ]
    return pd.DataFrame(rows)


def build_reference_summary_table(reference_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize the active reference-latent branch for notebook display."""
    reference = reference_output.get("reference", {})
    diagnostics = reference.get("diagnostics", {})
    donor = diagnostics.get("donor_leakage", {})
    stage_probe = diagnostics.get("stage_preservation", {}).get("probe", {})
    gene_overlap = diagnostics.get("gene_overlap", {})
    label_neighborhood = diagnostics.get("label_neighborhood", {})
    gate = diagnostics.get("alignment_gate", {})
    label_transfer = reference.get("label_transfer", {})
    shape = tuple(reference.get("latent_shape", [0, 0]))
    rows = [
        {"metric": "latent_backend", "value": reference.get("backend_name", "n/a")},
        {"metric": "provenance_mode", "value": reference.get("provenance", {}).get("mode", "n/a")},
        {"metric": "reference_source", "value": reference.get("source_path", "n/a")},
        {"metric": "latent_n_cells", "value": int(shape[0]) if len(shape) > 0 else 0},
        {"metric": "latent_dim", "value": int(shape[1]) if len(shape) > 1 else 0},
        {"metric": "stage_count", "value": diagnostics.get("stage_preservation", {}).get("n_stages", 0)},
        {"metric": "stage_probe_accuracy", "value": stage_probe.get("logreg_accuracy", float("nan"))},
        {"metric": "stage_probe_balanced_accuracy", "value": stage_probe.get("balanced_accuracy", float("nan"))},
        {"metric": "donor_leakage_accuracy", "value": donor.get("logreg_accuracy", float("nan"))},
        {"metric": "donor_chance_accuracy", "value": donor.get("chance_accuracy", float("nan"))},
        {"metric": "label_coverage", "value": label_transfer.get("coverage", 0.0)},
        {"metric": "gene_overlap_fraction", "value": gene_overlap.get("reference_query_overlap_fraction", float("nan"))},
        {"metric": "missing_gene_fraction", "value": gene_overlap.get("missing_gene_fraction", float("nan"))},
        {
            "metric": "neighbor_label_agreement",
            "value": label_neighborhood.get("mean_neighbor_label_agreement", float("nan")),
        },
        {"metric": "alignment_gate_status", "value": gate.get("status", "n/a")},
        {"metric": "alignment_gate_action", "value": gate.get("recommended_action", "n/a")},
    ]
    return pd.DataFrame(rows)


def build_reference_label_table(reference_output: dict[str, Any]) -> pd.DataFrame:
    """Expose the top transferred reference labels."""
    label_transfer = reference_output.get("reference", {}).get("label_transfer", {})
    top_labels = label_transfer.get("top_labels", [])
    if not top_labels:
        return pd.DataFrame(columns=["label", "count"])
    return pd.DataFrame(top_labels, columns=["label", "count"])


def build_reference_evaluation_table(reference_output: dict[str, Any]) -> pd.DataFrame:
    """Expose notebook-friendly HLCA reference quality metrics."""
    reference = reference_output.get("reference", {})
    diagnostics = reference.get("diagnostics", {})
    stage = diagnostics.get("stage_preservation", {})
    probe = stage.get("probe", {})
    donor = diagnostics.get("donor_leakage", {})
    gene_overlap = diagnostics.get("gene_overlap", {})
    label_neighborhood = diagnostics.get("label_neighborhood", {})
    label_alignment = diagnostics.get("stage_label_alignment", {})
    gate = diagnostics.get("alignment_gate", {})
    label_transfer = reference.get("label_transfer", {})
    centroid_distances = [float(value) for value in stage.get("centroid_distances", {}).values()]
    rows = [
        {"metric": "stage_probe_accuracy", "value": probe.get("logreg_accuracy", float("nan"))},
        {"metric": "stage_probe_balanced_accuracy", "value": probe.get("balanced_accuracy", float("nan"))},
        {"metric": "stage_probe_chance", "value": probe.get("chance_accuracy", float("nan"))},
        {"metric": "donor_leakage_accuracy", "value": donor.get("logreg_accuracy", float("nan"))},
        {"metric": "donor_leakage_chance", "value": donor.get("chance_accuracy", float("nan"))},
        {
            "metric": "mean_stage_centroid_distance",
            "value": float(np.mean(centroid_distances)) if centroid_distances else float("nan"),
        },
        {
            "metric": "min_stage_centroid_distance",
            "value": float(np.min(centroid_distances)) if centroid_distances else float("nan"),
        },
        {"metric": "label_coverage", "value": label_transfer.get("coverage", float("nan"))},
        {"metric": "gene_overlap_fraction", "value": gene_overlap.get("reference_query_overlap_fraction", float("nan"))},
        {"metric": "missing_gene_fraction", "value": gene_overlap.get("missing_gene_fraction", float("nan"))},
        {
            "metric": "nearest_neighbor_label_agreement",
            "value": label_neighborhood.get("mean_neighbor_label_agreement", float("nan")),
        },
        {
            "metric": "n_labeled_cells_for_neighbor_check",
            "value": label_neighborhood.get("n_labeled_cells", 0),
        },
        {
            "metric": "stage_label_alignment_rows",
            "value": len(label_alignment.get("rows", [])),
        },
        {
            "metric": "stage_label_alignment_cols",
            "value": len(label_alignment.get("cols", [])),
        },
        {"metric": "alignment_gate_status", "value": gate.get("status", "n/a")},
        {"metric": "alignment_gate_action", "value": gate.get("recommended_action", "n/a")},
    ]
    return pd.DataFrame(rows)


def build_spatial_summary_table(spatial_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize the active spatial mapping provider for notebook display."""
    summary = spatial_output.get("spatial_mapping", {})
    qc = summary.get("qc", {})
    rows = [
        {"metric": "method", "value": summary.get("method", "n/a")},
        {"metric": "status", "value": summary.get("status", "n/a")},
        {"metric": "n_spots", "value": summary.get("n_spots", 0)},
        {"metric": "n_features", "value": summary.get("n_features", 0)},
        {"metric": "mean_row_sum", "value": qc.get("mean_row_sum", float("nan"))},
        {"metric": "mean_max_assignment", "value": qc.get("mean_max_assignment", float("nan"))},
        {"metric": "source_path", "value": summary.get("source_path", "n/a")},
    ]
    return pd.DataFrame(rows)


def run_spatial_provider_ladder(
    cfg: DictConfig,
    *,
    methods: list[str] | None = None,
    reference_output: dict[str, Any] | None = None,
    use_tqdm: bool = True,
    execution_mode: str = "force_rebuild",
) -> dict[str, dict[str, Any]]:
    """Run Tangram/TACCO/DestVI through the active pipeline surface with notebook progress."""
    methods = methods or ["tangram", "tacco", "destvi"]
    outputs: dict[str, dict[str, Any]] = {}
    iterator = _progress_iter(methods, desc="Spatial providers", enabled=use_tqdm)
    for method in iterator:
        cfg_method = clone_config(cfg)
        cfg_method = OmegaConf.merge(cfg_method, _load_component(_COMPONENT_DIRS["spatial_mapping"] / f"{method}.yaml"))
        if not hasattr(cfg_method, "profiles") or cfg_method.profiles is None:
            cfg_method.profiles = OmegaConf.create({})
        cfg_method.profiles.spatial_mapping = method
        cfg_method.spatial_mapping.method = method
        cfg_method.spatial_mapping.execution_mode = execution_mode
        if use_tqdm:
            cfg_method.spatial_mapping.show_progress = True
        outputs[method] = run_spatial_mapping(cfg_method, reference_output=reference_output)
        if hasattr(iterator, "set_postfix_str"):
            iterator.set_postfix_str(f"{method}:{outputs[method].get('status', 'n/a')}")
    return outputs


def build_spatial_provider_table(provider_outputs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Summarize notebook-driven provider runs for quick inspection."""
    rows: list[dict[str, Any]] = []
    for method, payload in provider_outputs.items():
        summary = payload.get("spatial_mapping", {})
        rows.append(
            {
                "method": method,
                "status": payload.get("status", summary.get("status", "n/a")),
                "provider_version": summary.get("provider_version", "n/a"),
                "execution_mode": summary.get("execution_mode", "n/a"),
                "n_spots": summary.get("n_spots", 0),
                "n_features": summary.get("n_features", 0),
                "source_path": summary.get("source_path", "n/a"),
                "notes": summary.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _provider_matrix_and_columns(payload: dict[str, Any]) -> tuple[np.ndarray | None, list[str], pd.Index | None]:
    mapping = payload.get("mapping_result")
    if mapping is None or mapping.compositions is None:
        return None, [], None
    matrix = np.asarray(mapping.compositions, dtype=np.float32)
    columns = [str(value) for value in getattr(mapping, "feature_names", ()) or ()]
    obs = getattr(mapping, "obs", None)
    index = None if obs is None else pd.Index(obs.index.astype(str))
    return matrix, columns, index


def _normalized_provider_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    row_sums = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums > 0)


def build_spatial_provider_metric_table(provider_outputs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build comparable QC metrics for live provider runs.

    This is an internal quality screen, not a ground-truth accuracy claim.
    """
    rows: list[dict[str, Any]] = []
    for method, payload in provider_outputs.items():
        summary = payload.get("spatial_mapping", {})
        matrix, columns, _ = _provider_matrix_and_columns(payload)
        row: dict[str, Any] = {
            "method": method,
            "status": payload.get("status", summary.get("status", "n/a")),
            "execution_mode": summary.get("execution_mode", "n/a"),
            "n_spots": summary.get("n_spots", 0),
            "n_features": summary.get("n_features", 0),
            "provider_version": summary.get("provider_version", "n/a"),
        }
        if matrix is None or matrix.size == 0:
            rows.append(row)
            continue

        probs = _normalized_provider_matrix(matrix)
        row_sums = matrix.sum(axis=1)
        max_assignment = probs.max(axis=1)
        entropy = -(np.clip(probs, 1e-8, 1.0) * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
        norm_entropy = entropy / np.log(max(2, probs.shape[1]))
        winners = np.argmax(probs, axis=1)
        top_feature = columns[int(pd.Series(winners).mode().iloc[0])] if columns else "n/a"
        row_sum_closeness = float(np.mean(np.clip(1.0 - np.abs(row_sums - 1.0), 0.0, 1.0)))
        heuristic_score = (
            0.45 * row_sum_closeness
            + 0.35 * float(max_assignment.mean())
            + 0.20 * float((1.0 - norm_entropy).mean())
        )
        row.update(
            {
                "mean_row_sum": float(row_sums.mean()),
                "std_row_sum": float(row_sums.std()),
                "rows_close_to_one_frac": float(np.mean(np.abs(row_sums - 1.0) <= 0.05)),
                "mean_max_assignment": float(max_assignment.mean()),
                "mean_normalized_entropy": float(norm_entropy.mean()),
                "winner_diversity": int(np.unique(winners).size),
                "dominant_feature": top_feature,
                "qc_heuristic_score": heuristic_score,
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    if "qc_heuristic_score" in table.columns:
        table = table.sort_values(["status", "qc_heuristic_score"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return table


def build_spatial_provider_agreement_table(provider_outputs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Compare provider outputs on overlapping spots and shared feature columns."""
    rows: list[dict[str, Any]] = []
    methods = list(provider_outputs.keys())
    for idx, left_method in enumerate(methods):
        left_matrix, left_columns, left_index = _provider_matrix_and_columns(provider_outputs[left_method])
        if left_matrix is None or left_index is None:
            continue
        left_df = pd.DataFrame(_normalized_provider_matrix(left_matrix), index=left_index, columns=left_columns)
        for right_method in methods[idx + 1 :]:
            right_matrix, right_columns, right_index = _provider_matrix_and_columns(provider_outputs[right_method])
            if right_matrix is None or right_index is None:
                continue
            right_df = pd.DataFrame(_normalized_provider_matrix(right_matrix), index=right_index, columns=right_columns)
            shared_spots = left_df.index.intersection(right_df.index)
            shared_features = [feature for feature in left_df.columns if feature in right_df.columns]
            row = {
                "left_method": left_method,
                "right_method": right_method,
                "n_shared_spots": int(shared_spots.size),
                "n_shared_features": int(len(shared_features)),
            }
            if shared_spots.empty or not shared_features:
                rows.append(row)
                continue
            left_aligned = left_df.loc[shared_spots, shared_features].to_numpy(dtype=np.float32)
            right_aligned = right_df.loc[shared_spots, shared_features].to_numpy(dtype=np.float32)
            left_winners = np.argmax(left_aligned, axis=1)
            right_winners = np.argmax(right_aligned, axis=1)
            row.update(
                {
                    "winner_agreement": float(np.mean(left_winners == right_winners)),
                    "mean_abs_diff": float(np.mean(np.abs(left_aligned - right_aligned))),
                    "mean_cosine_similarity": float(
                        np.mean(
                            np.sum(left_aligned * right_aligned, axis=1)
                            / np.clip(
                                np.linalg.norm(left_aligned, axis=1) * np.linalg.norm(right_aligned, axis=1),
                                1e-8,
                                None,
                            )
                        )
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_provider_benchmark(
    cfg: DictConfig,
    *,
    methods: list[str] | None = None,
    modes: list[str] | None = None,
    edges: list[str] | None = None,
    seeds: list[int] | None = None,
    execution_mode: str = "force_rebuild",
    use_tqdm: bool = True,
    reference_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a matched provider benchmark with raw provider rebuilds and downstream scoring."""
    methods = methods or ["tangram", "tacco", "destvi"]
    modes = modes or ["pooled", "deep_sets"]
    edges = edges or ["AAH->AIS", "AIS->MIA"]
    seeds = seeds or [7, 13, 29]

    base_cfg = clone_config(cfg)
    base_cfg = OmegaConf.merge(base_cfg, _load_component(_COMPONENT_DIRS["train"] / "medium.yaml"))
    base_cfg.train.profile = "medium"
    if hasattr(base_cfg, "profiles") and base_cfg.profiles is not None:
        base_cfg.profiles.train = "medium"
    base_cfg.transition_model.wes_regularizer.enabled = False
    base_cfg.transition_model.schrodinger_bridge.sigma = 0.0

    reference = reference_output or run_reference(base_cfg)
    reference_gate = (reference.get("reference", {}).get("diagnostics", {}) or {}).get("alignment_gate", {})

    provider_outputs_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    provider_metric_rows: list[pd.DataFrame] = []
    agreement_rows: list[pd.DataFrame] = []
    downstream_rows: list[dict[str, Any]] = []

    seed_iter = _progress_iter([str(seed) for seed in seeds], desc="Provider benchmark seeds", enabled=use_tqdm)
    for seed_label in seed_iter:
        seed = int(seed_label)
        seed_outputs: dict[str, dict[str, Any]] = {}
        for method in methods:
            cfg_run = clone_config(base_cfg)
            cfg_run.seed = seed
            cfg_run.train.seed = seed
            cfg_run.spatial_mapping.method = method
            cfg_run.spatial_mapping.execution_mode = execution_mode
            if use_tqdm:
                cfg_run.spatial_mapping.show_progress = True
            if hasattr(cfg_run, "profiles") and cfg_run.profiles is not None:
                cfg_run.profiles.spatial_mapping = method

            spatial_output = run_spatial_mapping(cfg_run, reference_output=reference)
            seed_outputs[method] = spatial_output

            metric_table = build_spatial_provider_metric_table({method: spatial_output})
            if not metric_table.empty:
                metric_table["seed"] = seed
                provider_metric_rows.append(metric_table)

            if spatial_output.get("status") != "complete":
                continue

            for edge in edges:
                src, tgt = [part.strip() for part in str(edge).split("->", 1)]
                for mode in modes:
                    cfg_eval = clone_config(cfg_run)
                    cfg_eval.context_model.mode = mode
                    cfg_eval.context_model.graph_enabled = bool(mode == "graph_of_sets")
                    cfg_eval.transition_model.active_edge = [src, tgt]
                    context_output = run_context_model(cfg_eval, spatial_output=spatial_output)
                    transition_output = run_transition_model(
                        cfg_eval,
                        reference_output=reference,
                        spatial_output=spatial_output,
                        context_output=context_output,
                    )
                    evaluation_output = run_evaluation(
                        cfg_eval,
                        transition_output=transition_output,
                        context_output=context_output,
                    )
                    biology = evaluation_output.get("biology_summary") or {}
                    downstream_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "edge": edge,
                            "mode": mode,
                            "sinkhorn": float(evaluation_output["heldout_metrics"]["sinkhorn"]),
                            "calibration_error": float(evaluation_output["calibration"]["mean_abs_shift_error"]),
                            "dominant_increase_group": biology.get("dominant_increase_group"),
                            "dominant_decrease_group": biology.get("dominant_decrease_group"),
                            "status": evaluation_output.get("status", "complete"),
                        }
                    )

        provider_outputs_by_seed[seed] = seed_outputs
        agreement = build_spatial_provider_agreement_table(seed_outputs)
        if not agreement.empty:
            agreement["seed"] = seed
            agreement_rows.append(agreement)

    provider_metric_table = (
        pd.concat(provider_metric_rows, ignore_index=True)
        if provider_metric_rows
        else pd.DataFrame(columns=["method", "seed"])
    )
    agreement_table = (
        pd.concat(agreement_rows, ignore_index=True)
        if agreement_rows
        else pd.DataFrame(columns=["left_method", "right_method", "seed"])
    )
    downstream_table = pd.DataFrame(downstream_rows)
    benchmark = summarize_provider_benchmark(
        provider_metric_table=provider_metric_table,
        downstream_table=downstream_table,
        agreement_table=agreement_table,
        reference_gate=reference_gate,
    )
    benchmark_md = render_provider_benchmark_md(benchmark)
    return {
        "reference_gate": reference_gate,
        "provider_outputs_by_seed": provider_outputs_by_seed,
        "provider_metric_table": provider_metric_table,
        "provider_agreement_table": agreement_table,
        "provider_downstream_table": downstream_table,
        "benchmark": benchmark,
        "benchmark_md": benchmark_md,
    }


def build_provider_benchmark_table(benchmark_output: dict[str, Any]) -> pd.DataFrame:
    rows = benchmark_output.get("benchmark", {}).get("provider_scores", [])
    if not rows:
        return pd.DataFrame(columns=["method", "hybrid_rank_score"])
    return pd.DataFrame(rows).sort_values("hybrid_rank_score").reset_index(drop=True)


def apply_selected_provider(cfg: DictConfig, benchmark_output: dict[str, Any]) -> DictConfig:
    """Clone config and apply the benchmark-selected provider as the downstream default."""
    selected = (benchmark_output.get("benchmark") or {}).get("selected_provider")
    selection_status = (benchmark_output.get("benchmark") or {}).get("selection_status", "inconclusive")
    selection_reason = (benchmark_output.get("benchmark") or {}).get("selection_reason", "selection_not_run")
    if not selected:
        return clone_config(cfg)
    cfg_selected = clone_config(cfg)
    cfg_selected.spatial_mapping.method = str(selected)
    cfg_selected.spatial_mapping.selected_provider = str(selected)
    cfg_selected.spatial_mapping.selection_status = str(selection_status)
    cfg_selected.spatial_mapping.selection_reason = str(selection_reason)
    if hasattr(cfg_selected, "profiles") and cfg_selected.profiles is not None:
        cfg_selected.profiles.spatial_mapping = str(selected)
    return cfg_selected


def build_context_summary_table(context_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize the typed context branch for notebook display."""
    summary = context_output.get("context_model", {})
    token_summary = summary.get("typed_token_summary", {})
    rows = [
        {"metric": "mode", "value": summary.get("mode", "n/a")},
        {"metric": "spatial_mapping_method", "value": summary.get("spatial_mapping_method", "n/a")},
        {"metric": "n_token_rows", "value": token_summary.get("n_tokens", 0)},
        {"metric": "token_dim", "value": token_summary.get("token_dim", 0)},
    ]
    if "example_context_norm" in summary:
        rows.append({"metric": "context_norm", "value": summary.get("example_context_norm", float("nan"))})
        rows.append({"metric": "context_dim", "value": summary.get("example_context_dim", 0)})
    if "mean_token_confidence" in summary:
        rows.append({"metric": "mean_token_confidence", "value": summary.get("mean_token_confidence", float("nan"))})
    if "example_context_tokens" in summary:
        rows.append({"metric": "example_context_tokens", "value": summary.get("example_context_tokens", 0)})
    if "dataset_name" in summary:
        rows.append({"metric": "dataset_name", "value": summary.get("dataset_name", "n/a")})
        rows.append({"metric": "dataset_embedding_enabled", "value": summary.get("dataset_embedding_enabled", False)})
    if "graph_context_norm" in summary:
        rows.append({"metric": "graph_context_norm", "value": summary.get("graph_context_norm", float("nan"))})
        rows.append({"metric": "graph_num_nodes", "value": summary.get("graph_num_nodes", 0)})
        rows.append({"metric": "graph_num_edges", "value": summary.get("graph_num_edges", 0)})
    return pd.DataFrame(rows)


def build_transition_summary_table(
    transition_output: dict[str, Any],
    evaluation_output: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Summarize one transition run and its evaluation diagnostics."""
    split = transition_output.get("split_summary", {})
    wes = transition_output.get("wes_diagnostics", {})
    calibration = {} if evaluation_output is None else evaluation_output.get("calibration", {})
    heldout = {} if evaluation_output is None else evaluation_output.get("heldout_metrics", {})
    rows = [
        {"metric": "edge", "value": transition_output.get("edge", "n/a")},
        {"metric": "mode", "value": transition_output.get("mode", "n/a")},
        {"metric": "sigma", "value": transition_output.get("sigma", float("nan"))},
        {"metric": "diffusion_weight", "value": transition_output.get("diffusion_weight", float("nan"))},
        {"metric": "split_strategy", "value": split.get("split_strategy", "n/a")},
        {"metric": "same_donor_overlap", "value": len(split.get("overlap_donors", []))},
        {"metric": "wes_enabled", "value": wes.get("enabled", False)},
        {"metric": "wes_penalty_mean", "value": wes.get("regularizer_mean_penalty", float("nan"))},
        {"metric": "heldout_sinkhorn", "value": heldout.get("sinkhorn", float("nan"))},
        {"metric": "heldout_auc", "value": heldout.get("classifier_auc", float("nan"))},
        {"metric": "calibration_error", "value": calibration.get("mean_abs_shift_error", float("nan"))},
    ]
    if "encoder_parameter_delta" in transition_output:
        rows.append({"metric": "encoder_parameter_delta", "value": transition_output.get("encoder_parameter_delta", 0.0)})
    pretraining = transition_output.get("pretraining_summary") or {}
    if pretraining:
        metrics = pretraining.get("metrics", {}) or {}
        rows.append({"metric": "pretraining_encoder_delta", "value": pretraining.get("encoder_parameter_delta", float("nan"))})
        rows.append({"metric": "pretraining_loss_total", "value": metrics.get("loss_total", float("nan"))})
        rows.append({"metric": "pretraining_ranking_accuracy", "value": metrics.get("ranking_accuracy", float("nan"))})
        rows.append({"metric": "pretraining_provider_cosine", "value": metrics.get("provider_consistency_cosine", float("nan"))})
        rows.append({"metric": "pretraining_coordinate_accuracy", "value": metrics.get("coordinate_corruption_accuracy", float("nan"))})
        rows.append({"metric": "pretraining_group_relation_accuracy", "value": metrics.get("group_relation_accuracy", float("nan"))})
    aux = transition_output.get("auxiliary_context_shuffle_metrics") or {}
    if aux:
        rows.append({"metric": "context_auxiliary_task", "value": aux.get("task", "context_shuffle")})
        rows.append({"metric": "context_shuffle_loss", "value": aux.get("loss", float("nan"))})
        rows.append({"metric": "context_shuffle_accuracy", "value": aux.get("accuracy", float("nan"))})
        rows.append({"metric": "context_separation_score", "value": aux.get("separation_score", float("nan"))})
        rows.append({"metric": "context_auxiliary_margin", "value": aux.get("margin", float("nan"))})
        rows.append({"metric": "context_positive_score", "value": aux.get("positive_score", float("nan"))})
        rows.append({"metric": "drift_context_gate", "value": aux.get("drift_context_gate", float("nan"))})
        rows.append({"metric": "drift_context_attention_entropy", "value": aux.get("drift_context_attention_entropy", float("nan"))})
        rows.append({"metric": "provider_consistency_cosine", "value": aux.get("provider_consistency_cosine", float("nan"))})
        rows.append({"metric": "group_relation_accuracy", "value": aux.get("group_relation_accuracy", float("nan"))})
        negative_scores = aux.get("negative_control_scores", {}) or {}
        if negative_scores:
            rows.append(
                {
                    "metric": "negative_control_scores",
                    "value": ", ".join(f"{key}={float(value):.3f}" for key, value in negative_scores.items()),
                }
            )
    attention = transition_output.get("attention_summary") or {}
    if attention:
        rows.append({"metric": "attention_maps", "value": ", ".join(attention.get("available_maps", []))})
        rows.append({"metric": "top_attention_token_types", "value": ", ".join(attention.get("top_token_types", []))})
        rows.append({"metric": "attention_entropy", "value": attention.get("pma_attention_entropy", float("nan"))})
        rows.append({"metric": "confidence_weighted_attention_entropy", "value": attention.get("confidence_weighted_attention_entropy", float("nan"))})
        if attention.get("group_attention_scores"):
            rows.append(
                {
                    "metric": "group_attention_scores",
                    "value": ", ".join(
                        f"{key}={float(value):.3f}" for key, value in attention["group_attention_scores"].items()
                    ),
                }
            )
        if attention.get("relation_attention_scores"):
            rows.append(
                {
                    "metric": "relation_attention_scores",
                    "value": ", ".join(
                        f"{key}={float(value):.3f}" for key, value in attention["relation_attention_scores"].items()
                    ),
                }
            )
    transfer = transition_output.get("dataset_transfer_diagnostics") or {}
    if transfer:
        rows.append({"metric": "source_dataset", "value": transfer.get("source_dataset", "n/a")})
        rows.append({"metric": "transfer_dataset", "value": transfer.get("transfer_dataset", "n/a")})
        provider_views = transfer.get("provider_views_used", []) or []
        if provider_views:
            rows.append({"metric": "provider_views_used", "value": ", ".join(str(view) for view in provider_views)})
        rows.append({"metric": "cross_dataset_negatives_used", "value": transfer.get("cross_dataset_negatives_used", 0)})
        labels = transfer.get("negative_control_labels", []) or []
        if labels:
            rows.append({"metric": "negative_control_labels", "value": ", ".join(str(label) for label in labels)})
    return pd.DataFrame(rows)


def build_biology_summary_table(evaluation_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize edge-level biological insight signals for notebook display."""
    biology = evaluation_output.get("biology_summary") or {}
    if not biology:
        return pd.DataFrame(columns=["metric", "value"])
    rows = [
        {"metric": "edge", "value": biology.get("edge", "n/a")},
        {"metric": "dominant_increase_group", "value": biology.get("dominant_increase_group", "n/a")},
        {"metric": "dominant_decrease_group", "value": biology.get("dominant_decrease_group", "n/a")},
        {"metric": "split_strategy", "value": biology.get("split_strategy", "n/a")},
        {"metric": "n_overlap_donors", "value": len(biology.get("overlap_donors", []))},
        {"metric": "context_sensitivity_delta", "value": biology.get("context_sensitivity_delta", float("nan"))},
    ]
    return pd.DataFrame(rows)


def build_gate_ready_table(evaluation_output: dict[str, Any]) -> pd.DataFrame:
    """Expose the current evaluation outputs that feed scientific gates."""
    rows = [
        {"signal": "sinkhorn", "value": evaluation_output.get("heldout_metrics", {}).get("sinkhorn", float("nan"))},
        {
            "signal": "context_sensitivity_delta",
            "value": (evaluation_output.get("context_sensitivity", {}) or {}).get("context_sensitivity_delta", float("nan")),
        },
        {
            "signal": "mean_diffusion_scale",
            "value": evaluation_output.get("diffusion_diagnostics", {}).get("mean_diffusion_scale", float("nan")),
        },
        {
            "signal": "pseudotime_alignment",
            "value": evaluation_output.get("pseudotime_structure", {}).get("pseudotime_correlation", float("nan")),
        },
    ]
    return pd.DataFrame(rows)


def run_mode_ladder(
    cfg: DictConfig,
    *,
    modes: list[str] | None = None,
    edge: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run matched context-mode comparisons while reusing upstream steps."""
    modes = modes or [
        "rna_only",
        "pooled",
        "deep_sets",
        "set_only",
        "typed_hierarchical_transformer",
        "deep_sets_transformer_hybrid",
        "graph_of_sets",
    ]
    cfg_base = clone_config(cfg)
    if edge is not None:
        src, tgt = [part.strip() for part in str(edge).split("->", 1)]
        cfg_base.transition_model.active_edge = [src, tgt]

    reference = run_reference(cfg_base)
    spatial = run_spatial_mapping(cfg_base, reference_output=reference)
    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        cfg_mode = clone_config(cfg_base)
        cfg_mode.context_model.mode = mode
        cfg_mode.context_model.graph_enabled = bool(mode == "graph_of_sets")
        context = run_context_model(cfg_mode, spatial_output=spatial)
        transition = run_transition_model(
            cfg_mode,
            reference_output=reference,
            spatial_output=spatial,
            context_output=context,
        )
        evaluation = run_evaluation(cfg_mode, transition_output=transition, context_output=context)
        results[mode] = {
            "context_model": context,
            "transition_model": transition,
            "evaluation": evaluation,
        }
    return results


def build_mode_comparison_table(mode_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert a matched mode ladder run into a compact comparison table."""
    rows: list[dict[str, Any]] = []
    for mode, payload in mode_results.items():
        evaluation = payload["evaluation"]
        rows.append(
            {
                "mode": mode,
                "sinkhorn": evaluation["heldout_metrics"]["sinkhorn"],
                "sinkhorn_delta": evaluation["heldout_metrics"]["sinkhorn_delta"],
                "classifier_auc": evaluation["heldout_metrics"]["classifier_auc"],
                "calibration_error": evaluation["calibration"]["mean_abs_shift_error"],
                "context_sensitivity_delta": (evaluation.get("context_sensitivity") or {}).get("context_sensitivity_delta"),
                "dominant_increase_group": (evaluation.get("biology_summary") or {}).get("dominant_increase_group"),
                "dominant_decrease_group": (evaluation.get("biology_summary") or {}).get("dominant_decrease_group"),
                "split_strategy": payload["transition_model"]["split_summary"]["split_strategy"],
            }
        )
    return pd.DataFrame(rows).sort_values("sinkhorn").reset_index(drop=True)


def run_seeded_mode_ladder(
    cfg: DictConfig,
    *,
    seeds: list[int],
    modes: list[str] | None = None,
    edge: str | None = None,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Run a matched mode ladder over a deterministic seed grid."""
    seeded_results: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        cfg_seed = clone_config(cfg)
        cfg_seed.seed = int(seed)
        cfg_seed.train.seed = int(seed)
        seeded_results[int(seed)] = run_mode_ladder(cfg_seed, modes=modes, edge=edge)
    return seeded_results


def build_seeded_mode_summary_table(
    seeded_results: dict[int, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """Aggregate matched seed-grid mode runs into one summary table."""
    rows: list[dict[str, Any]] = []
    all_modes = sorted({mode for payload in seeded_results.values() for mode in payload})
    for mode in all_modes:
        sinkhorn: list[float] = []
        calibration: list[float] = []
        context_delta: list[float] = []
        increases: list[str] = []
        decreases: list[str] = []
        split_strategies: list[str] = []
        for _, payload in seeded_results.items():
            if mode not in payload:
                continue
            evaluation = payload[mode]["evaluation"]
            transition = payload[mode]["transition_model"]
            biology = evaluation.get("biology_summary") or {}
            sinkhorn.append(float(evaluation["heldout_metrics"]["sinkhorn"]))
            calibration.append(float(evaluation["calibration"]["mean_abs_shift_error"]))
            delta = (evaluation.get("context_sensitivity") or {}).get("context_sensitivity_delta")
            if delta is not None:
                context_delta.append(float(delta))
            if biology.get("dominant_increase_group") is not None:
                increases.append(str(biology["dominant_increase_group"]))
            if biology.get("dominant_decrease_group") is not None:
                decreases.append(str(biology["dominant_decrease_group"]))
            split_strategies.append(str(transition["split_summary"]["split_strategy"]))
        if not sinkhorn:
            continue
        increase_mode = Counter(increases).most_common(1)[0][0] if increases else None
        decrease_mode = Counter(decreases).most_common(1)[0][0] if decreases else None
        split_mode = Counter(split_strategies).most_common(1)[0][0] if split_strategies else None
        rows.append(
            {
                "mode": mode,
                "n_seeds": len(sinkhorn),
                "sinkhorn_mean": sum(sinkhorn) / len(sinkhorn),
                "sinkhorn_std": pd.Series(sinkhorn).std(ddof=0),
                "calibration_mean": sum(calibration) / len(calibration),
                "calibration_std": pd.Series(calibration).std(ddof=0),
                "context_delta_mean": None if not context_delta else sum(context_delta) / len(context_delta),
                "context_delta_std": None if not context_delta else pd.Series(context_delta).std(ddof=0),
                "dominant_increase_group": increase_mode,
                "dominant_decrease_group": decrease_mode,
                "split_strategy": split_mode,
            }
        )
    return pd.DataFrame(rows).sort_values(["sinkhorn_mean", "calibration_mean"]).reset_index(drop=True)


def run_latent_backend_compare(
    cfg: DictConfig,
    *,
    backends: list[str] | None = None,
    edge: str | None = None,
    mode: str = "set_only",
) -> dict[str, dict[str, Any]]:
    """Run matched latent-backend comparisons for one edge/mode pair."""
    backends = backends or ["hlca", "pca"]
    cfg_base = clone_config(cfg)
    cfg_base.context_model.mode = mode
    cfg_base.context_model.graph_enabled = bool(mode == "graph_of_sets")
    if edge is not None:
        src, tgt = [part.strip() for part in str(edge).split("->", 1)]
        cfg_base.transition_model.active_edge = [src, tgt]

    results: dict[str, dict[str, Any]] = {}
    for backend in backends:
        cfg_backend = clone_config(cfg_base)
        cfg_backend.reference.latent_backend = backend
        if backend == "pca":
            cfg_backend.reference.n_components = int(getattr(cfg_backend.reference, "n_components", 32))
        pipeline = run_full(cfg_backend)
        results[backend] = pipeline["steps"]
    return results


def build_latent_comparison_table(backend_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert matched latent-backend runs into a compact comparison table."""
    rows: list[dict[str, Any]] = []
    for backend, steps in backend_results.items():
        reference = steps["reference"]["reference"]
        evaluation = steps["evaluation"]
        rows.append(
            {
                "backend": backend,
                "sinkhorn": evaluation["heldout_metrics"]["sinkhorn"],
                "calibration_error": evaluation["calibration"]["mean_abs_shift_error"],
                "dominant_increase_group": (evaluation.get("biology_summary") or {}).get("dominant_increase_group"),
                "dominant_decrease_group": (evaluation.get("biology_summary") or {}).get("dominant_decrease_group"),
                "latent_dim": reference["latent_shape"][1],
                "provenance_mode": reference.get("provenance", {}).get("mode"),
                "source_path": reference.get("source_path"),
            }
        )
    return pd.DataFrame(rows).sort_values("sinkhorn").reset_index(drop=True)


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load JSON/YAML artifacts from a run-like directory for notebook inspection."""
    run_dir = Path(run_dir)
    payload: dict[str, Any] = {"run_dir": str(run_dir), "files": {}}
    if not run_dir.exists():
        payload["error"] = "missing_run_dir"
        return payload

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".yaml", ".yml", ".md"}:
            continue
        rel = path.relative_to(run_dir).as_posix()
        if path.suffix == ".json":
            try:
                payload["files"][rel] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload["files"][rel] = {"_error": "failed_to_parse_json"}
        else:
            payload["files"][rel] = path.read_text(encoding="utf-8")
    return payload


def available_steps() -> list[str]:
    return sorted(_STEP_REGISTRY)


__all__ = [
    "available_steps",
    "apply_selected_provider",
    "build_biology_summary_table",
    "build_context_summary_table",
    "build_dataset_preprocessing_table",
    "build_gate_ready_table",
    "build_latent_comparison_table",
    "build_provider_benchmark_table",
    "build_reference_evaluation_table",
    "build_mode_comparison_table",
    "build_spatial_provider_agreement_table",
    "build_spatial_provider_metric_table",
    "build_spatial_provider_table",
    "build_seeded_mode_summary_table",
    "build_reference_label_table",
    "build_reference_summary_table",
    "build_spatial_summary_table",
    "build_step_status_table",
    "build_transition_summary_table",
    "clone_config",
    "compose_config",
    "load_run",
    "run_data_preprocessing_overview",
    "run_latent_backend_compare",
    "run_mode_ladder",
    "run_provider_benchmark",
    "run_seeded_mode_ladder",
    "run_spatial_provider_ladder",
    "run_pipeline",
    "run_step",
]
