"""Transition-model pipeline entrypoint."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from stagebridge.context_model.graph_builder import build_spatial_knn_graph
from stagebridge.context_model.graph_encoder import GraphOfSetsContextEncoder
from stagebridge.context_model.hierarchical_transformer import TypedHierarchicalTransformerEncoder, dataset_name_to_id
from stagebridge.context_model.set_encoder import (
    DeepSetsContextEncoder,
    DeepSetsTransformerHybridEncoder,
    PooledContextEncoder,
    TypedSetContextEncoder,
)
from stagebridge.context_model.token_builder import build_typed_spot_tokens
from stagebridge.data.common.schema import LatentCohort
from stagebridge.data.luad_evo.wes import build_wes_feature_lookup, load_luad_evo_wes_features
from stagebridge.pipelines.run_reference import run_reference
from stagebridge.pipelines.run_spatial_mapping import run_spatial_mapping
from stagebridge.transition_model.disease_edges import edge_id_map, edge_label, resolve_disease_edge
from stagebridge.transition_model.stochastic_dynamics import EdgeWiseStochasticDynamics
from stagebridge.transition_model.relational_pretraining import RelationalPretrainingConfig
from stagebridge.transition_model.train import (
    build_stagewise_edge_split,
    pretrain_relational_transformer,
    train_edgewise_transition_model,
    train_edgewise_transition_model_with_context_encoder,
)
from stagebridge.transition_model.wes_regularizer import lookup_wes_vectors, pairwise_wes_penalty


def _build_shuffled_control_tokens(
    tokens: np.ndarray,
    coords: np.ndarray,
    confidence: np.ndarray,
    token_type_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    shuffled_tokens = np.asarray(tokens, dtype=np.float32).copy()
    for col_idx in range(shuffled_tokens.shape[1]):
        shuffled_tokens[:, col_idx] = shuffled_tokens[rng.permutation(shuffled_tokens.shape[0]), col_idx]
    shuffled_coords = np.asarray(coords, dtype=np.float32)[rng.permutation(coords.shape[0])].copy()
    shuffled_confidence = np.asarray(confidence, dtype=np.float32)[rng.permutation(confidence.shape[0])].copy()
    shuffled_token_type_ids = np.asarray(token_type_ids, dtype=np.int64)[rng.permutation(token_type_ids.shape[0])].copy()
    return shuffled_tokens, shuffled_coords, shuffled_confidence, shuffled_token_type_ids


def _select_context_rows(
    typed: Any,
    *,
    donor_candidates: list[str],
    stage: str,
    max_spots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Any, str, dict[str, Any], np.ndarray]:
    obs = typed.obs
    chosen_rows: np.ndarray | None = None
    chosen_donor = ""
    fallback = False
    for donor_id in donor_candidates:
        mask = (obs["donor_id"].astype(str) == str(donor_id)) & (obs["stage"].astype(str) == str(stage))
        rows = np.flatnonzero(mask.to_numpy())
        if rows.size > 0:
            chosen_rows = rows
            chosen_donor = str(donor_id)
            break

    if chosen_rows is None:
        mask = obs["stage"].astype(str) == str(stage)
        chosen_rows = np.flatnonzero(mask.to_numpy())
        if chosen_rows.size == 0:
            raise ValueError(f"No spatial rows available for source stage '{stage}'.")
        chosen_donor = str(obs.iloc[chosen_rows[0]]["donor_id"])
        fallback = True

    if chosen_rows.size > max_spots > 0:
        rng = np.random.default_rng(int(seed))
        chosen_rows = np.sort(rng.choice(chosen_rows, size=int(max_spots), replace=False))

    diagnostics = {
        "source_context_donor": chosen_donor,
        "source_context_stage": stage,
        "source_context_fallback": fallback,
        "n_source_context_spots": int(chosen_rows.size),
    }
    return (
        typed.tokens[chosen_rows],
        typed.coords[chosen_rows],
        obs.iloc[chosen_rows].reset_index(drop=True),
        chosen_donor,
        diagnostics,
        chosen_rows,
    )


def _build_optional_negative_control(
    typed: Any,
    *,
    stage: str,
    max_spots: int,
    seed: int,
    donor_candidates: list[str] | None = None,
    exclude_donors: list[str] | None = None,
    fallback_stage: str | None = None,
) -> dict[str, Any] | None:
    """Construct one biologically mismatched context control when possible."""
    obs = typed.obs
    donor_candidates = [str(item) for item in (donor_candidates or [])]
    exclude = {str(item) for item in (exclude_donors or [])}
    rows = np.array([], dtype=int)
    chosen_donor = None

    for donor_id in donor_candidates:
        if donor_id in exclude:
            continue
        mask = (obs["donor_id"].astype(str) == donor_id) & (obs["stage"].astype(str) == str(stage))
        rows = np.flatnonzero(mask.to_numpy())
        if rows.size > 0:
            chosen_donor = donor_id
            break

    if rows.size == 0:
        mask = obs["stage"].astype(str) == str(stage)
        if exclude:
            mask &= ~obs["donor_id"].astype(str).isin(sorted(exclude))
        rows = np.flatnonzero(mask.to_numpy())
    if rows.size == 0 and fallback_stage is not None:
        mask = obs["stage"].astype(str) == str(fallback_stage)
        if exclude:
            mask &= ~obs["donor_id"].astype(str).isin(sorted(exclude))
        rows = np.flatnonzero(mask.to_numpy())
    if rows.size == 0:
        return None

    if rows.size > max_spots > 0:
        rng = np.random.default_rng(int(seed))
        rows = np.sort(rng.choice(rows, size=int(max_spots), replace=False))

    return {
        "tokens": torch.tensor(typed.tokens[rows], dtype=torch.float32),
        "coords": torch.tensor(typed.coords[rows], dtype=torch.float32),
        "confidence": torch.tensor(typed.token_confidence[rows], dtype=torch.float32),
        "token_type_ids": torch.tensor(typed.token_type_ids[rows], dtype=torch.long),
        "dataset_ids": None,
        "note": "biological_mismatch",
        "stage": str(stage),
        "donor_id": None if chosen_donor is None else str(chosen_donor),
    }


def _spot_alignment_keys(obs: Any) -> list[str]:
    sample_ids = obs["sample_id"].astype(str) if "sample_id" in obs.columns else obs.index.astype(str)
    if "spot_id" in obs.columns:
        spot_ids = obs["spot_id"].astype(str)
    elif "barcode" in obs.columns:
        spot_ids = obs["barcode"].astype(str)
    else:
        spot_ids = obs.index.astype(str)
    stages = obs["stage"].astype(str) if "stage" in obs.columns else ["unknown"] * len(obs)
    return [
        f"{sample_id}|{spot_id}|{stage}"
        for sample_id, spot_id, stage in zip(sample_ids.tolist(), spot_ids.tolist(), list(stages), strict=False)
    ]


def _build_cross_provider_views(
    cfg: DictConfig,
    *,
    typed: Any,
    chosen_rows: np.ndarray,
    edge_id: int,
    selected_method: str | None,
) -> list[dict[str, Any]]:
    methods = list(cfg.get("context_model", {}).get("pretraining", {}).get("provider_methods", ["tangram", "tacco", "destvi"]))
    if not methods:
        return []
    reference_obs = typed.obs.iloc[chosen_rows].reset_index(drop=True)
    reference_keys = _spot_alignment_keys(reference_obs)
    dataset_ids = torch.tensor(
        [dataset_name_to_id(str(cfg.get("data", {}).get("dataset", "luad_evo")))],
        dtype=torch.long,
    )
    edge_ids = torch.tensor([int(edge_id)], dtype=torch.long)
    provider_views: list[dict[str, Any]] = []
    for method in methods:
        method_name = str(method)
        if selected_method is not None and method_name == str(selected_method):
            continue
        cfg_alt = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_alt.spatial_mapping.method = method_name
        try:
            alt_spatial = run_spatial_mapping(cfg_alt)
        except Exception:
            continue
        alt_result = alt_spatial.get("mapping_result")
        if alt_result is None or alt_result.compositions is None or alt_result.coords is None or alt_result.obs is None:
            continue
        alt_typed = build_typed_spot_tokens(
            alt_result.compositions,
            alt_result.coords,
            alt_result.obs,
            alt_result.feature_names,
        )
        alt_keys = _spot_alignment_keys(alt_typed.obs)
        alt_index = {key: idx for idx, key in enumerate(alt_keys)}
        matched_rows = [alt_index[key] for key in reference_keys if key in alt_index]
        if len(matched_rows) != len(reference_keys):
            continue
        matched = np.asarray(matched_rows, dtype=np.int64)
        provider_views.append(
            {
                "method": method_name,
                "tokens": torch.tensor(alt_typed.tokens[matched], dtype=torch.float32),
                "coords": torch.tensor(alt_typed.coords[matched], dtype=torch.float32),
                "confidence": torch.tensor(alt_typed.token_confidence[matched], dtype=torch.float32),
                "token_type_ids": torch.tensor(alt_typed.token_type_ids[matched], dtype=torch.long),
                "dataset_ids": dataset_ids.clone(),
                "edge_ids": edge_ids.clone(),
                "notes": "cross_provider_view",
            }
        )
    return provider_views


def _build_context_bundle(
    cfg: DictConfig,
    *,
    mode: str,
    stage_src: str,
    stage_tgt: str,
    edge_id: int,
    donor_candidates: list[str],
    typed: Any | None = None,
    spatial_method: str | None = None,
) -> dict[str, Any]:
    hidden_dim = int(cfg.get("context_model", {}).get("hidden_dim", 128))
    if mode == "rna_only":
        zero = torch.zeros(hidden_dim, dtype=torch.float32)
        return {
            "context": zero,
            "shuffled_context": zero.clone(),
            "context_encoder": None,
            "context_tokens": None,
            "shuffled_context_tokens": None,
            "context_coords": None,
            "shuffled_context_coords": None,
            "context_confidence": None,
            "shuffled_context_confidence": None,
            "context_missing_mask": None,
            "context_graph": None,
            "token_type_ids": None,
            "shuffled_token_type_ids": None,
            "dataset_ids": None,
            "provider_views": [],
            "context_negative_controls": [],
            "diagnostics": {"mode": "rna_only", "context_norm": 0.0},
            "typed_subset_tokens": None,
            "typed_feature_names": None,
        }

    if typed is None:
        raise ValueError("Typed spatial tokens are required for non-rna_only transition modes.")
    node_tokens, node_coords, node_obs, _, diagnostics, chosen_rows = _select_context_rows(
        typed,
        donor_candidates=donor_candidates,
        stage=stage_src,
        max_spots=int(cfg.get("context_model", {}).get("max_context_spots", 128)),
        seed=int(cfg.get("seed", 42)),
    )
    token_type_ids = typed.token_type_ids[chosen_rows].astype(np.int64, copy=False)
    token_confidence = typed.token_confidence[chosen_rows].astype(np.float32, copy=False)
    token_missing_mask = typed.token_missing_mask[chosen_rows].astype(bool, copy=False)
    shuffled_tokens, shuffled_coords, shuffled_confidence, shuffled_token_type_ids = _build_shuffled_control_tokens(
        node_tokens,
        node_coords,
        token_confidence,
        token_type_ids,
        seed=int(cfg.get("seed", 42)),
    )
    diagnostics.update(
        {
            "mean_token_confidence": float(token_confidence.mean()) if token_confidence.size else 0.0,
            "missing_token_fraction": float(token_missing_mask.mean()) if token_missing_mask.size else 0.0,
            "token_group_means": {
                str(name): float(node_tokens[:, idx].mean())
                for idx, name in enumerate(typed.schema.typed_feature_names)
            },
        }
    )
    dataset_name = str(cfg.get("data", {}).get("dataset", "luad_evo"))
    transfer_dataset = cfg.get("context_model", {}).get("transfer_dataset")
    dataset_ids = torch.tensor([dataset_name_to_id(dataset_name)], dtype=torch.long)
    context_negative_controls: list[dict[str, Any]] = []
    wrong_stage_control = _build_optional_negative_control(
        typed,
        stage=stage_tgt,
        max_spots=int(cfg.get("context_model", {}).get("max_context_spots", 128)),
        seed=int(cfg.get("seed", 42)) + 101,
        donor_candidates=donor_candidates,
        exclude_donors=[],
        fallback_stage=stage_src,
    )
    wrong_donor_candidates = [
        donor
        for donor in sorted(set(typed.obs["donor_id"].astype(str).tolist()))
        if str(donor) not in {str(item) for item in donor_candidates}
    ]
    wrong_donor_control = _build_optional_negative_control(
        typed,
        stage=stage_src,
        max_spots=int(cfg.get("context_model", {}).get("max_context_spots", 128)),
        seed=int(cfg.get("seed", 42)) + 202,
        donor_candidates=wrong_donor_candidates,
        exclude_donors=donor_candidates,
    )
    for control in (wrong_stage_control, wrong_donor_control):
        if control is not None:
            control["dataset_ids"] = dataset_ids.clone()
            control["label"] = str(control.get("note", "biological_mismatch"))
            context_negative_controls.append(control)
    if transfer_dataset and str(transfer_dataset) != dataset_name:
        context_negative_controls.append(
            {
                "tokens": torch.tensor(node_tokens, dtype=torch.float32),
                "coords": torch.tensor(node_coords, dtype=torch.float32),
                "confidence": torch.tensor(token_confidence, dtype=torch.float32),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "dataset_ids": torch.tensor([dataset_name_to_id(str(transfer_dataset))], dtype=torch.long),
                "label": "dataset_id_mismatch",
                "note": "cross_dataset_id_control",
                "stage": str(stage_src),
                "donor_id": diagnostics.get("source_context_donor"),
            }
        )

    if mode in {"pooled", "deep_sets", "set_only", "typed_hierarchical_transformer", "deep_sets_transformer_hybrid"}:
        if mode == "pooled":
            encoder = PooledContextEncoder(
                input_dim=node_tokens.shape[1],
                hidden_dim=hidden_dim,
            )
        elif mode == "deep_sets":
            encoder = DeepSetsContextEncoder(
                input_dim=node_tokens.shape[1],
                hidden_dim=hidden_dim,
                dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
            )
        elif mode == "set_only":
            encoder = TypedSetContextEncoder(
                input_dim=node_tokens.shape[1],
                hidden_dim=hidden_dim,
                num_heads=int(cfg.get("context_model", {}).get("num_heads", 4)),
                num_inducing_points=int(cfg.get("context_model", {}).get("num_inducing_points", 16)),
                dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
                num_token_types=len(typed.schema.typed_feature_names),
                use_spatial_rpe=bool(cfg.get("context_model", {}).get("use_spatial_rpe", True)),
                token_dropout_rate=float(cfg.get("context_model", {}).get("token_dropout_rate", 0.05)),
                use_confidence_gate=bool(cfg.get("context_model", {}).get("use_confidence_gate", True)),
            )
        elif mode == "deep_sets_transformer_hybrid":
            encoder = DeepSetsTransformerHybridEncoder(
                input_dim=node_tokens.shape[1],
                hidden_dim=hidden_dim,
                num_heads=int(cfg.get("context_model", {}).get("num_heads", 4)),
                num_inducing_points=int(cfg.get("context_model", {}).get("num_inducing_points", 16)),
                num_seed_vectors=int(cfg.get("context_model", {}).get("num_seed_vectors", 2)),
                dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
                num_token_types=len(typed.schema.typed_feature_names),
                use_spatial_rpe=bool(cfg.get("context_model", {}).get("use_spatial_rpe", True)),
                token_dropout_rate=float(cfg.get("context_model", {}).get("token_dropout_rate", 0.05)),
                use_confidence_gate=bool(cfg.get("context_model", {}).get("use_confidence_gate", True)),
            )
        else:
            encoder = TypedHierarchicalTransformerEncoder(
                input_dim=node_tokens.shape[1],
                hidden_dim=hidden_dim,
                num_heads=int(cfg.get("context_model", {}).get("num_heads", 4)),
                num_inducing_points=int(cfg.get("context_model", {}).get("num_inducing_points", 16)),
                dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
                num_token_types=len(typed.schema.typed_feature_names),
                num_group_summary_tokens=int(cfg.get("context_model", {}).get("num_group_summary_tokens", 2)),
                num_fusion_queries=int(cfg.get("context_model", {}).get("num_fusion_queries", 7)),
                dataset_embedding_dim=int(cfg.get("context_model", {}).get("dataset_embedding_dim", 16)),
                num_datasets=4,
                num_edges=max(8, len(edge_id_map())),
                use_spatial_rpe=bool(cfg.get("context_model", {}).get("use_spatial_rpe", True)),
                use_confidence_gate=bool(cfg.get("context_model", {}).get("use_confidence_gate", True)),
                token_dropout_rate=float(cfg.get("context_model", {}).get("token_dropout_rate", 0.05)),
                use_relation_tokens=bool(cfg.get("context_model", {}).get("use_relation_tokens", True)),
                group_names=list(typed.schema.typed_feature_names),
            )
        diagnostics.update(
            {
                "mode": mode,
                "spatial_mapping_method": spatial_method,
                "trainable_context_encoder": True,
                "context_training_mode": "joint_transition_optimization",
                "dataset_name": dataset_name,
            }
        )
        provider_views = []
        if mode == "typed_hierarchical_transformer" and bool(cfg.get("context_model", {}).get("pretraining", {}).get("provider_consistency_enabled", True)):
            provider_views = _build_cross_provider_views(
                cfg,
                typed=typed,
                chosen_rows=chosen_rows,
                edge_id=edge_id,
                selected_method=spatial_method,
            )
            diagnostics["provider_views_available"] = [str(view["method"]) for view in provider_views]
        return {
            "context": None,
            "shuffled_context": None,
            "context_encoder": encoder,
            "context_tokens": torch.tensor(node_tokens, dtype=torch.float32),
            "shuffled_context_tokens": torch.tensor(shuffled_tokens, dtype=torch.float32),
            "context_coords": torch.tensor(node_coords, dtype=torch.float32),
            "shuffled_context_coords": torch.tensor(shuffled_coords, dtype=torch.float32),
            "context_confidence": torch.tensor(token_confidence, dtype=torch.float32),
            "shuffled_context_confidence": torch.tensor(shuffled_confidence, dtype=torch.float32),
            "context_missing_mask": torch.tensor(token_missing_mask, dtype=torch.bool),
            "context_graph": None,
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "shuffled_token_type_ids": torch.tensor(shuffled_token_type_ids, dtype=torch.long),
            "dataset_ids": dataset_ids,
            "edge_ids": torch.tensor([int(edge_id)], dtype=torch.long),
            "provider_views": provider_views,
            "context_negative_controls": context_negative_controls,
            "diagnostics": diagnostics,
            "typed_subset_tokens": node_tokens,
            "typed_feature_names": typed.schema.typed_feature_names,
        }

    if mode != "graph_of_sets":
        raise ValueError(f"Unsupported context_model.mode '{mode}'.")

    graph = build_spatial_knn_graph(
        coords=node_coords,
        patient_ids=node_obs["donor_id"].astype(str).tolist(),
        stage_indices=[0] * node_coords.shape[0],
        dataset_ids=["luad_evo"] * node_coords.shape[0],
        k=min(6, max(node_coords.shape[0] - 1, 1)),
        include_cross_patient=False,
        include_cross_stage=False,
    )
    encoder = GraphOfSetsContextEncoder(
        input_dim=node_tokens.shape[1],
        hidden_dim=hidden_dim,
        num_graph_layers=int(cfg.get("context_model", {}).get("graph_num_layers", 2)),
        num_heads=int(cfg.get("context_model", {}).get("graph_num_heads", 4)),
        dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
    )
    diagnostics.update(
        {
            "mode": mode,
            "spatial_mapping_method": spatial_method,
            "graph_num_nodes": int(graph.num_nodes),
            "graph_num_edges": int(graph.edge_index.shape[1]),
            "trainable_context_encoder": True,
            "context_training_mode": "joint_transition_optimization",
        }
    )
    return {
        "context": None,
        "shuffled_context": None,
        "context_encoder": encoder,
        "context_tokens": torch.tensor(node_tokens, dtype=torch.float32),
        "shuffled_context_tokens": torch.tensor(shuffled_tokens, dtype=torch.float32),
        "context_coords": torch.tensor(node_coords, dtype=torch.float32),
        "shuffled_context_coords": torch.tensor(shuffled_coords, dtype=torch.float32),
        "context_confidence": torch.tensor(token_confidence, dtype=torch.float32),
        "shuffled_context_confidence": torch.tensor(shuffled_confidence, dtype=torch.float32),
        "context_missing_mask": torch.tensor(token_missing_mask, dtype=torch.bool),
        "context_graph": graph,
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "shuffled_token_type_ids": torch.tensor(shuffled_token_type_ids, dtype=torch.long),
        "dataset_ids": dataset_ids,
        "edge_ids": torch.tensor([int(edge_id)], dtype=torch.long),
        "provider_views": [],
        "context_negative_controls": [],
        "diagnostics": diagnostics,
        "typed_subset_tokens": node_tokens,
        "typed_feature_names": typed.schema.typed_feature_names,
    }


def _summarize_attention_maps(
    attention_maps: dict[str, torch.Tensor] | None,
    *,
    token_type_ids: torch.Tensor | None,
    typed_feature_names: list[str] | tuple[str, ...] | None,
    token_coords: torch.Tensor | None = None,
    token_confidence: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    if not attention_maps:
        return None
    summary: dict[str, Any] = {"available_maps": sorted(attention_maps.keys())}
    if token_type_ids is not None and typed_feature_names is not None:
        token_type_names = list(typed_feature_names)
        counts: dict[str, int] = {}
        for token_type in token_type_ids.detach().cpu().tolist():
            name = token_type_names[int(token_type)] if 0 <= int(token_type) < len(token_type_names) else str(token_type)
            counts[name] = counts.get(name, 0) + 1
        summary["token_type_distribution"] = counts
    pma_attention = attention_maps.get("pma_seed_attention")
    if pma_attention is not None:
        weights = pma_attention.detach().float().cpu()
        while weights.ndim > 1 and weights.shape[0] == 1:
            weights = weights[0]
        token_scores = weights.mean(dim=0).mean(dim=0) if weights.ndim == 3 else weights.reshape(-1)
        top_k = min(5, int(token_scores.shape[-1]))
        top_values, top_indices = torch.topk(token_scores, k=top_k)
        summary["top_token_indices"] = [int(idx) for idx in top_indices.tolist()]
        summary["top_token_attention"] = [float(value) for value in top_values.tolist()]
        if token_type_ids is not None and typed_feature_names is not None:
            token_names = []
            ids_cpu = token_type_ids.detach().cpu()
            feature_names = list(typed_feature_names)
            for idx in top_indices.tolist():
                token_type = int(ids_cpu[int(idx)].item())
                token_names.append(
                    feature_names[token_type] if 0 <= token_type < len(feature_names) else str(token_type)
                )
            summary["top_token_types"] = token_names
        if token_coords is not None:
            coords_cpu = token_coords.detach().float().cpu()
            dists = torch.linalg.norm(coords_cpu, dim=-1)
            summary["top_token_coordinates"] = [
                [float(value) for value in coords_cpu[int(idx)].tolist()]
                for idx in top_indices.tolist()
            ]
            summary["top_token_distance_bins"] = [float(dists[int(idx)].item()) for idx in top_indices.tolist()]
        if token_confidence is not None:
            confidence_cpu = token_confidence.detach().float().cpu()
            summary["top_token_confidence"] = [float(confidence_cpu[int(idx)].item()) for idx in top_indices.tolist()]
            weighted = token_scores * confidence_cpu
            weighted = weighted / weighted.sum().clamp_min(1e-8)
            summary["confidence_weighted_attention_entropy"] = float(
                (-(weighted * weighted.clamp_min(1e-8).log()).sum()).item()
            )
            summary["mean_confidence_weighted_attention"] = float(weighted.mean().item())
        normalized = token_scores / token_scores.sum().clamp_min(1e-8)
        summary["pma_attention_entropy"] = float(
            (-(normalized * normalized.clamp_min(1e-8).log()).sum()).item()
        )
    return summary


def _tensor_subset(latent: np.ndarray, obs: Any, donors: list[str], *, device: str) -> tuple[torch.Tensor, Any]:
    mask = obs["donor_id"].astype(str).isin([str(donor) for donor in donors]).to_numpy()
    if mask.sum() == 0:
        return torch.zeros((0, latent.shape[1]), dtype=torch.float32, device=device), obs.iloc[0:0].copy()
    return torch.tensor(latent[mask], dtype=torch.float32, device=device), obs.loc[mask].reset_index(drop=True)


def _subset_reference_cohort(
    cohort: LatentCohort,
    *,
    stages: list[str],
    max_cells_per_stage: int | None,
    seed: int,
) -> LatentCohort:
    obs = cohort.obs.reset_index(drop=True).copy()
    mask = obs["stage"].astype(str).isin([str(stage) for stage in stages]).to_numpy()
    if max_cells_per_stage is not None and max_cells_per_stage > 0:
        rng = np.random.default_rng(int(seed))
        chosen = np.zeros(obs.shape[0], dtype=bool)
        masked_positions = np.flatnonzero(mask)
        masked_stages = obs.iloc[masked_positions]["stage"].to_numpy()
        for stage_name in np.unique(masked_stages):
            stage_rows = masked_positions[masked_stages == stage_name]
            if stage_rows.shape[0] <= max_cells_per_stage:
                chosen[stage_rows] = True
                continue
            keep = rng.choice(stage_rows, size=int(max_cells_per_stage), replace=False)
            chosen[keep] = True
        mask &= chosen
    return LatentCohort(
        latent=np.asarray(cohort.latent[mask], dtype=np.float32),
        obs=obs.loc[mask].reset_index(drop=True),
        feature_names=cohort.feature_names,
        source_path=cohort.source_path,
        latent_key=cohort.latent_key,
    )


def _resolve_reference_payload(
    cfg: DictConfig,
    *,
    edge: Any,
    reference_output: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], LatentCohort]:
    payload = reference_output or run_reference(cfg)
    cohort = payload.get("cohort")
    if cohort is None:
        raise ValueError("Reference pipeline did not return a usable cohort.")
    subset = _subset_reference_cohort(
        cohort,
        stages=[edge.stage_src, edge.stage_tgt],
        max_cells_per_stage=int(cfg.get("transition_model", {}).get("max_cells_per_stage", 128)),
        seed=int(cfg.get("seed", 42)),
    )
    return payload, subset


def _resolve_typed_tokens(
    cfg: DictConfig,
    *,
    mode: str,
    spatial_output: dict[str, Any] | None = None,
    context_output: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any] | None, dict[str, Any] | None]:
    if mode == "rna_only":
        return None, spatial_output, context_output

    if context_output is not None and context_output.get("typed_tokens") is not None:
        return context_output["typed_tokens"], spatial_output, context_output

    spatial_payload = spatial_output or run_spatial_mapping(cfg)
    spatial_result = spatial_payload.get("mapping_result")
    if spatial_result is None or spatial_result.compositions is None or spatial_result.coords is None or spatial_result.obs is None:
        method = str(cfg.get("spatial_mapping", {}).get("method", "tangram"))
        status = None if spatial_result is None else spatial_result.status
        raise ValueError(
            f"Spatial mapping method '{method}' is not runnable for transition conditioning "
            f"(status={status!r})."
        )
    typed = build_typed_spot_tokens(
        spatial_result.compositions,
        spatial_result.coords,
        spatial_result.obs,
        spatial_result.feature_names,
    )
    return typed, spatial_payload, context_output


def run_transition_model(
    cfg: DictConfig,
    *,
    reference_output: dict[str, Any] | None = None,
    spatial_output: dict[str, Any] | None = None,
    context_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = str(cfg.get("context_model", {}).get("mode", "set_only"))
    device = str(cfg.get("train", {}).get("device", "cpu"))
    device = "cpu" if device != "cuda" or not torch.cuda.is_available() else device
    edge = resolve_disease_edge(cfg.get("transition_model", {}).get("active_edge"))
    edge_name = edge_label(edge)
    edge_id = edge_id_map()[edge_name]

    reference_payload, cohort = _resolve_reference_payload(
        cfg,
        edge=edge,
        reference_output=reference_output,
    )
    src_obs = cohort.obs[cohort.obs["stage"] == edge.stage_src].reset_index(drop=True)
    tgt_obs = cohort.obs[cohort.obs["stage"] == edge.stage_tgt].reset_index(drop=True)
    x_src_full = cohort.latent[cohort.obs["stage"] == edge.stage_src]
    x_tgt_full = cohort.latent[cohort.obs["stage"] == edge.stage_tgt]

    split = build_stagewise_edge_split(
        src_obs,
        tgt_obs,
        donor_col="donor_id",
        holdout_fraction=0.25,
        stage_src=edge.stage_src,
        stage_tgt=edge.stage_tgt,
    )
    x_src_train, src_train_obs = _tensor_subset(x_src_full, src_obs, split.source_train_donors, device=device)
    x_tgt_train, tgt_train_obs = _tensor_subset(x_tgt_full, tgt_obs, split.target_train_donors, device=device)
    x_src_test, src_test_obs = _tensor_subset(x_src_full, src_obs, split.source_test_donors, device=device)
    x_tgt_test, tgt_test_obs = _tensor_subset(x_tgt_full, tgt_obs, split.target_test_donors, device=device)

    evaluation_notes: list[str] = []
    if x_src_test.shape[0] == 0 or x_tgt_test.shape[0] == 0:
        evaluation_notes.append("Held-out edge cells unavailable for one stage; using training split for evaluation fallback.")
        if x_src_test.shape[0] == 0:
            x_src_test, src_test_obs = x_src_train, src_train_obs
        if x_tgt_test.shape[0] == 0:
            x_tgt_test, tgt_test_obs = x_tgt_train, tgt_train_obs

    typed, spatial_payload, resolved_context_output = _resolve_typed_tokens(
        cfg,
        mode=mode,
        spatial_output=spatial_output,
        context_output=context_output,
    )
    spatial_method = None
    if spatial_payload is not None:
        spatial_mapping = spatial_payload.get("spatial_mapping", {})
        if isinstance(spatial_mapping, dict):
            spatial_method = spatial_mapping.get("method")

    context_bundle = _build_context_bundle(
        cfg,
        mode=mode,
        stage_src=edge.stage_src,
        stage_tgt=edge.stage_tgt,
        edge_id=edge_id,
        donor_candidates=split.source_train_donors,
        typed=typed,
        spatial_method=spatial_method,
    )

    wes_cfg = cfg.get("transition_model", {}).get("wes_regularizer", {})
    wes_enabled = bool(wes_cfg.get("enabled", False))
    extra_cost: torch.Tensor | None = None
    wes_diagnostics = {"enabled": wes_enabled, "regularizer_mean_penalty": 0.0}
    if wes_enabled:
        wes = load_luad_evo_wes_features(cfg, stages=[edge.stage_src, edge.stage_tgt])
        lookup = build_wes_feature_lookup(wes)
        src_wes = lookup_wes_vectors(src_train_obs, lookup).to(device)
        tgt_wes = lookup_wes_vectors(tgt_train_obs, lookup).to(device)
        extra_cost = pairwise_wes_penalty(
            src_wes,
            tgt_wes,
            penalty_scale=float(wes_cfg.get("strength", 0.1)),
            normalize=True,
        ).to(device)
        wes_diagnostics = {
            "enabled": True,
            "regularizer_mean_penalty": float(extra_cost.mean().item()),
            "missing_src_wes_rows": int((src_wes.abs().sum(dim=1) == 0).sum().item()),
            "missing_tgt_wes_rows": int((tgt_wes.abs().sum(dim=1) == 0).sum().item()),
        }

    sigma = float(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("sigma", 0.1))
    diffusion_weight = 0.2 if sigma > 0.0 else 0.0
    context_dim = int(cfg.get("context_model", {}).get("hidden_dim", 128))
    model = EdgeWiseStochasticDynamics(
        input_dim=int(x_src_train.shape[1]),
        context_dim=context_dim,
        hidden_dim=int(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("hidden_dim", 128)),
        time_dim=int(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("time_embedding_dim", 32)),
        edge_dim=16,
        num_edges=len(edge_id_map()),
        dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
        min_diffusion_scale=1e-3,
        state_dependent_diffusion=bool(
            cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("state_dependent_diffusion", True)
        ),
        use_cross_attention_drift=bool(
            mode in {"set_only", "typed_hierarchical_transformer", "deep_sets_transformer_hybrid"}
            and cfg.get("context_model", {}).get("use_cross_attention_drift", True)
        ),
        use_ude=bool(
            cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("use_ude", False)
        ),
        ude_gate_init=float(
            cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("ude_gate_init", 0.0)
        ),
    ).to(device)

    learning_rate = float(cfg.get("train", {}).get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("train", {}).get("weight_decay", 1e-4))
    max_epochs = int(cfg.get("train", {}).get("max_epochs", 2))
    steps_per_epoch = int(cfg.get("train", {}).get("steps_per_epoch", 2))
    batch_cells = int(cfg.get("train", {}).get("batch_cells", 32))
    epsilon = float(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("ot_epsilon", 0.05))
    sinkhorn_iters = int(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("sinkhorn_iters", 80))
    num_ot_pairs = int(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("num_ot_pairs", 128))
    seed = int(cfg.get("seed", 42))

    attention_summary = None
    encoder_parameter_delta = 0.0
    trained_context_encoder = context_bundle.get("context_encoder")
    trained_context_tokens = None
    shuffled_context_tokens = None
    pretraining_summary = None
    if trained_context_encoder is None:
        history = train_edgewise_transition_model(
            model=model,
            x_src_train=x_src_train,
            x_tgt_train=x_tgt_train,
            context=context_bundle["context"].to(device),
            context_tokens=None,
            edge_id=edge_id,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            batch_cells=batch_cells,
            sigma=sigma,
            diffusion_weight=diffusion_weight,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            num_ot_pairs=num_ot_pairs,
            seed=seed,
            extra_cost=extra_cost,
        )
        trained_context = context_bundle["context"].to(device)
        shuffled_context = context_bundle["shuffled_context"].to(device)
    else:
        context_graph = context_bundle.get("context_graph")
        if context_graph is not None and hasattr(context_graph, "to"):
            context_graph = context_graph.to(device)
        trained_context_encoder = trained_context_encoder.to(device)
        pretraining_heads = None
        pretraining_config = None
        if mode == "typed_hierarchical_transformer" and bool(cfg.get("context_model", {}).get("pretraining", {}).get("enabled", True)):
            pre_cfg = cfg.get("context_model", {}).get("pretraining", {})
            pretraining_config = RelationalPretrainingConfig(
                mask_fraction=float(pre_cfg.get("mask_fraction", 0.15)),
                masked_token_weight=float(pre_cfg.get("masked_token_weight", 0.35)),
                ranking_weight=float(pre_cfg.get("ranking_weight", 0.35)),
                provider_consistency_weight=float(pre_cfg.get("provider_consistency_weight", 0.15)),
                coordinate_corruption_weight=float(pre_cfg.get("coordinate_corruption_weight", 0.10)),
                group_relation_weight=float(pre_cfg.get("group_relation_weight", 0.05)),
                ranking_margin=float(pre_cfg.get("ranking_margin", 0.2)),
                max_epochs=int(pre_cfg.get("max_epochs", 2)),
                steps_per_epoch=int(pre_cfg.get("steps_per_epoch", 4)),
                learning_rate=float(pre_cfg.get("learning_rate", learning_rate)),
                weight_decay=float(pre_cfg.get("weight_decay", weight_decay)),
                seed=seed,
            )
            pretraining = pretrain_relational_transformer(
                context_encoder=trained_context_encoder,
                context_tokens=context_bundle["context_tokens"].to(device),
                token_type_ids=context_bundle["token_type_ids"].to(device),
                token_coords=None if context_bundle["context_coords"] is None else context_bundle["context_coords"].to(device),
                token_confidence=None if context_bundle["context_confidence"] is None else context_bundle["context_confidence"].to(device),
                dataset_ids=None if context_bundle["dataset_ids"] is None else context_bundle["dataset_ids"].to(device),
                edge_ids=None if context_bundle.get("edge_ids") is None else context_bundle["edge_ids"].to(device),
                negative_controls=[
                    {
                        **control,
                        "tokens": control["tokens"].to(device),
                        "coords": None if control.get("coords") is None else control["coords"].to(device),
                        "confidence": None if control.get("confidence") is None else control["confidence"].to(device),
                        "token_type_ids": None if control.get("token_type_ids") is None else control["token_type_ids"].to(device),
                        "dataset_ids": None if control.get("dataset_ids") is None else control["dataset_ids"].to(device),
                    }
                    for control in context_bundle.get("context_negative_controls", [])
                ],
                provider_views=[
                    {
                        **view,
                        "tokens": view["tokens"].to(device),
                        "coords": None if view.get("coords") is None else view["coords"].to(device),
                        "confidence": None if view.get("confidence") is None else view["confidence"].to(device),
                        "token_type_ids": None if view.get("token_type_ids") is None else view["token_type_ids"].to(device),
                        "dataset_ids": None if view.get("dataset_ids") is None else view["dataset_ids"].to(device),
                    }
                    for view in context_bundle.get("provider_views", [])
                ],
                config=pretraining_config,
            )
            pretraining_heads = pretraining["heads"]
            pretraining_summary = {
                "history": pretraining["history"],
                "metrics": pretraining["metrics"],
                "encoder_parameter_delta": float(pretraining["encoder_parameter_delta"]),
            }
            context_bundle["diagnostics"]["pretraining_summary"] = pretraining_summary
        joint = train_edgewise_transition_model_with_context_encoder(
            model=model,
            context_encoder=trained_context_encoder,
            context_tokens=context_bundle["context_tokens"].to(device),
            shuffled_context_tokens=context_bundle["shuffled_context_tokens"].to(device),
            context_coords=None if context_bundle["context_coords"] is None else context_bundle["context_coords"].to(device),
            shuffled_context_coords=None
            if context_bundle["shuffled_context_coords"] is None
            else context_bundle["shuffled_context_coords"].to(device),
            context_confidence=None
            if context_bundle["context_confidence"] is None
            else context_bundle["context_confidence"].to(device),
            shuffled_context_confidence=None
            if context_bundle["shuffled_context_confidence"] is None
            else context_bundle["shuffled_context_confidence"].to(device),
            x_src_train=x_src_train,
            x_tgt_train=x_tgt_train,
            edge_id=edge_id,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            batch_cells=batch_cells,
            sigma=sigma,
            diffusion_weight=diffusion_weight,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            num_ot_pairs=num_ot_pairs,
            seed=seed,
            extra_cost=extra_cost,
            context_graph=context_graph,
            token_type_ids=None if context_bundle["token_type_ids"] is None else context_bundle["token_type_ids"].to(device),
            shuffled_token_type_ids=None
            if context_bundle["shuffled_token_type_ids"] is None
            else context_bundle["shuffled_token_type_ids"].to(device),
            dataset_ids=None if context_bundle["dataset_ids"] is None else context_bundle["dataset_ids"].to(device),
            edge_ids=None if context_bundle.get("edge_ids") is None else context_bundle["edge_ids"].to(device),
            context_negative_controls=[
                {
                    **control,
                    "tokens": control["tokens"].to(device),
                    "coords": None if control.get("coords") is None else control["coords"].to(device),
                    "confidence": None if control.get("confidence") is None else control["confidence"].to(device),
                    "token_type_ids": None if control.get("token_type_ids") is None else control["token_type_ids"].to(device),
                    "dataset_ids": None if control.get("dataset_ids") is None else control["dataset_ids"].to(device),
                }
                for control in context_bundle.get("context_negative_controls", [])
            ],
            provider_views=[
                {
                    **view,
                    "tokens": view["tokens"].to(device),
                    "coords": None if view.get("coords") is None else view["coords"].to(device),
                    "confidence": None if view.get("confidence") is None else view["confidence"].to(device),
                    "token_type_ids": None if view.get("token_type_ids") is None else view["token_type_ids"].to(device),
                    "dataset_ids": None if view.get("dataset_ids") is None else view["dataset_ids"].to(device),
                }
                for view in context_bundle.get("provider_views", [])
            ],
            capture_attention=bool(mode in {"set_only", "typed_hierarchical_transformer", "deep_sets_transformer_hybrid"}),
            auxiliary_loss_weight=float(cfg.get("context_model", {}).get("auxiliary_context_shuffle_weight", 0.1)),
            pretraining_config=None
            if pretraining_config is None
            else RelationalPretrainingConfig(
                mask_fraction=float(pretraining_config.mask_fraction),
                masked_token_weight=float(cfg.get("context_model", {}).get("finetune", {}).get("masked_token_weight", 0.05)),
                ranking_weight=float(cfg.get("context_model", {}).get("finetune", {}).get("ranking_weight", 0.15)),
                provider_consistency_weight=float(cfg.get("context_model", {}).get("finetune", {}).get("provider_consistency_weight", 0.05)),
                coordinate_corruption_weight=0.0,
                group_relation_weight=0.0,
                ranking_margin=float(pretraining_config.ranking_margin),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                seed=seed,
            ),
            pretraining_heads=pretraining_heads,
        )
        history = joint["history"]
        trained_summary = joint["context_summary"]
        shuffled_summary = joint["shuffled_context_summary"]
        trained_context = trained_summary.pooled_context.to(device)
        trained_context_tokens = None if getattr(trained_summary, "context_tokens", None) is None else trained_summary.context_tokens.to(device)
        shuffled_context = shuffled_summary.pooled_context.to(device)
        shuffled_context_tokens = None if getattr(shuffled_summary, "context_tokens", None) is None else shuffled_summary.context_tokens.to(device)
        encoder_parameter_delta = float(joint["encoder_parameter_delta"])
        context_bundle["diagnostics"]["context_norm"] = float(trained_context.norm().item())
        context_bundle["diagnostics"]["encoder_parameter_delta"] = encoder_parameter_delta
        context_bundle["diagnostics"]["encoder_internal"] = getattr(trained_summary, "diagnostics", {})
        context_bundle["diagnostics"]["auxiliary_context_shuffle_loss"] = float(joint["auxiliary_metrics"]["loss"])
        context_bundle["diagnostics"]["auxiliary_context_shuffle_accuracy"] = float(joint["auxiliary_metrics"]["accuracy"])
        context_bundle["diagnostics"]["context_separation_score"] = float(joint["auxiliary_metrics"]["separation_score"])
        context_bundle["diagnostics"]["auxiliary_task"] = str(joint["auxiliary_metrics"].get("task", "context_match_ranking"))
        context_bundle["diagnostics"]["auxiliary_margin"] = float(joint["auxiliary_metrics"].get("margin", 0.0))
        context_bundle["diagnostics"]["auxiliary_positive_score"] = float(joint["auxiliary_metrics"].get("positive_score", 0.0))
        context_bundle["diagnostics"]["drift_context_gate"] = float(joint["auxiliary_metrics"].get("drift_context_gate", 0.0))
        context_bundle["diagnostics"]["drift_context_attention_entropy"] = float(
            joint["auxiliary_metrics"].get("drift_context_attention_entropy", 0.0)
        )
        context_bundle["diagnostics"]["negative_control_scores"] = {
            str(key): float(value)
            for key, value in (joint["auxiliary_metrics"].get("negative_control_scores", {}) or {}).items()
        }
        context_bundle["diagnostics"]["auxiliary_loss_components"] = {
            str(key): float(value)
            for key, value in ((joint["auxiliary_metrics"].get("loss_components", {}) or {}).items())
        }
        if mode == "graph_of_sets":
            context_bundle["diagnostics"]["graph_num_nodes"] = int(getattr(trained_summary, "num_nodes", 0))
            context_bundle["diagnostics"]["graph_num_edges"] = int(getattr(trained_summary, "num_edges", 0))
        attention_summary = _summarize_attention_maps(
            getattr(trained_summary, "attention_maps", None),
            token_type_ids=getattr(trained_summary, "token_type_ids", None),
            typed_feature_names=context_bundle["typed_feature_names"],
            token_coords=getattr(trained_summary, "token_coords", None),
            token_confidence=getattr(trained_summary, "token_confidence", None),
        )
        if mode == "typed_hierarchical_transformer":
            hierarchical_diag = (getattr(trained_summary, "diagnostics", {}) or {}).get("group_diagnostics", [])
            if hierarchical_diag:
                first_diag = hierarchical_diag[0]
                group_scores = first_diag.get("fusion_attention_by_group", {})
                relation_scores = first_diag.get("fusion_attention_by_relation", {})
                query_scores = first_diag.get("query_attention_by_group", {})
                ranked_groups = sorted(group_scores.items(), key=lambda item: item[1], reverse=True)
                attention_summary = {
                    "available_maps": sorted((getattr(trained_summary, "attention_maps", {}) or {}).keys()),
                    "top_token_types": [str(name) for name, _ in ranked_groups[:4]],
                    "top_token_attention": [float(score) for _, score in ranked_groups[:4]],
                    "pma_attention_entropy": float(np.nan),
                    "group_attention_scores": {str(key): float(value) for key, value in group_scores.items()},
                    "relation_attention_scores": {str(key): float(value) for key, value in relation_scores.items()},
                    "query_attention_by_group": query_scores,
                }
        elif mode == "deep_sets_transformer_hybrid" and attention_summary is not None:
            attention_summary.update(
                {
                    "hybrid_gate_mean": float(getattr(trained_summary, "diagnostics", {}).get("hybrid_gate_mean", 0.0)),
                    "deep_sets_context_norm": float(
                        getattr(trained_summary, "diagnostics", {}).get("deep_sets_context_norm", 0.0)
                    ),
                    "transformer_refinement_norm": float(
                        getattr(trained_summary, "diagnostics", {}).get("transformer_refinement_norm", 0.0)
                    ),
                }
            )
        if attention_summary is not None:
            context_bundle["diagnostics"]["attention_summary"] = attention_summary

    return {
        "ok": True,
        "pipeline": "transition_model",
        "status": "complete",
        "mode": mode,
        "edge": edge_name,
        "sigma": sigma,
        "diffusion_weight": diffusion_weight,
        "reference": reference_payload.get("reference"),
        "spatial_mapping": None if spatial_payload is None else spatial_payload.get("spatial_mapping"),
        "context_model": None if resolved_context_output is None else resolved_context_output.get("context_model"),
        "split_summary": split.to_dict(),
        "context_diagnostics": context_bundle["diagnostics"],
        "wes_diagnostics": wes_diagnostics,
        "training_history": history,
        "evaluation_notes": evaluation_notes,
        "model": model,
        "context": trained_context,
        "context_tokens": trained_context_tokens,
        "shuffled_context": shuffled_context,
        "shuffled_context_tokens": shuffled_context_tokens,
        "trained_context_encoder": trained_context_encoder,
        "attention_summary": attention_summary,
        "encoder_parameter_delta": encoder_parameter_delta,
        "auxiliary_context_shuffle_metrics": None if trained_context_encoder is None else joint["auxiliary_metrics"],
        "pretraining_summary": pretraining_summary,
        "edge_id": edge_id,
        "x_src_test": x_src_test,
        "x_tgt_test": x_tgt_test,
        "typed_subset_tokens": context_bundle["typed_subset_tokens"],
        "typed_feature_names": context_bundle["typed_feature_names"],
        "token_package": {
            "token_values": None if context_bundle["context_tokens"] is None else context_bundle["context_tokens"].detach().cpu(),
            "token_coords": None if context_bundle["context_coords"] is None else context_bundle["context_coords"].detach().cpu(),
            "token_confidence": None if context_bundle["context_confidence"] is None else context_bundle["context_confidence"].detach().cpu(),
            "token_type_ids": None if context_bundle["token_type_ids"] is None else context_bundle["token_type_ids"].detach().cpu(),
            "token_missing_mask": None if context_bundle["context_missing_mask"] is None else context_bundle["context_missing_mask"].detach().cpu(),
            "dataset_ids": None if context_bundle.get("dataset_ids") is None else context_bundle["dataset_ids"].detach().cpu(),
        },
        "dataset_transfer_diagnostics": {
            "source_dataset": str(cfg.get("data", {}).get("dataset", "luad_evo")),
            "transfer_dataset": cfg.get("context_model", {}).get("transfer_dataset"),
            "dataset_embedding_enabled": bool(mode == "typed_hierarchical_transformer"),
            "provider_views_used": [str(view.get("method", "unknown")) for view in context_bundle.get("provider_views", [])],
            "cross_dataset_negatives_used": int(
                sum(
                    1
                    for control in context_bundle.get("context_negative_controls", [])
                    if control.get("dataset_ids") is not None
                    and int(control["dataset_ids"][0].item()) != int(context_bundle["dataset_ids"][0].item())
                )
            ),
            "negative_control_labels": [str(control.get("label", "negative")) for control in context_bundle.get("context_negative_controls", [])],
        },
    }
