"""Transition-model pipeline entrypoint."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from stagebridge.context_model.graph_builder import build_spatial_knn_graph
from stagebridge.context_model.graph_encoder import GraphOfSetsContextEncoder
from stagebridge.context_model.set_encoder import DeepSetsContextEncoder, PooledContextEncoder, TypedSetContextEncoder
from stagebridge.context_model.token_builder import build_typed_spot_tokens
from stagebridge.data.common.schema import LatentCohort
from stagebridge.data.luad_evo.wes import build_wes_feature_lookup, load_luad_evo_wes_features
from stagebridge.pipelines.run_context_model import run_context_model
from stagebridge.pipelines.run_reference import run_reference
from stagebridge.pipelines.run_spatial_mapping import run_spatial_mapping
from stagebridge.transition_model.disease_edges import edge_id_map, edge_label, resolve_disease_edge
from stagebridge.transition_model.stochastic_dynamics import EdgeWiseStochasticDynamics
from stagebridge.transition_model.train import build_stagewise_edge_split, train_edgewise_transition_model
from stagebridge.transition_model.wes_regularizer import lookup_wes_vectors, pairwise_wes_penalty


def _select_context_rows(
    typed: Any,
    *,
    donor_candidates: list[str],
    stage: str,
    max_spots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Any, str, dict[str, Any]]:
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
    )


def _build_context_bundle(
    cfg: DictConfig,
    *,
    mode: str,
    stage_src: str,
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
            "diagnostics": {"mode": "rna_only", "context_norm": 0.0},
            "typed_subset_tokens": None,
            "typed_feature_names": None,
        }

    if typed is None:
        raise ValueError("Typed spatial tokens are required for non-rna_only transition modes.")
    node_tokens, node_coords, node_obs, _, diagnostics = _select_context_rows(
        typed,
        donor_candidates=donor_candidates,
        stage=stage_src,
        max_spots=int(cfg.get("context_model", {}).get("max_context_spots", 128)),
        seed=int(cfg.get("seed", 42)),
    )
    shuffled_tokens = node_tokens.copy()
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    rng.shuffle(shuffled_tokens, axis=0)

    if mode in {"pooled", "deep_sets", "set_only"}:
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
        else:
            encoder = TypedSetContextEncoder(
                input_dim=node_tokens.shape[1],
                hidden_dim=hidden_dim,
                num_heads=int(cfg.get("context_model", {}).get("num_heads", 4)),
                num_inducing_points=int(cfg.get("context_model", {}).get("num_inducing_points", 16)),
                dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
            )
        with torch.no_grad():
            context = encoder(torch.tensor(node_tokens, dtype=torch.float32)).pooled_context
            shuffled = encoder(torch.tensor(shuffled_tokens, dtype=torch.float32)).pooled_context
        diagnostics.update(
            {
                "mode": mode,
                "context_norm": float(context.norm().item()),
                "spatial_mapping_method": spatial_method,
            }
        )
        return {
            "context": context,
            "shuffled_context": shuffled,
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
    with torch.no_grad():
        context_summary = encoder(torch.tensor(node_tokens, dtype=torch.float32), graph)
        shuffled_summary = encoder(torch.tensor(shuffled_tokens, dtype=torch.float32), graph)
    diagnostics.update(
        {
            "mode": mode,
            "context_norm": float(context_summary.pooled_context.norm().item()),
            "graph_num_nodes": int(context_summary.num_nodes),
            "graph_num_edges": int(context_summary.num_edges),
            "spatial_mapping_method": spatial_method,
        }
    )
    return {
        "context": context_summary.pooled_context,
        "shuffled_context": shuffled_summary.pooled_context,
        "diagnostics": diagnostics,
        "typed_subset_tokens": node_tokens,
        "typed_feature_names": typed.schema.typed_feature_names,
    }


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
    model = EdgeWiseStochasticDynamics(
        input_dim=int(x_src_train.shape[1]),
        context_dim=int(context_bundle["context"].shape[0]),
        hidden_dim=int(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("hidden_dim", 128)),
        time_dim=int(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("time_embedding_dim", 32)),
        edge_dim=16,
        num_edges=len(edge_id_map()),
        dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
        min_diffusion_scale=1e-3,
        state_dependent_diffusion=bool(
            cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("state_dependent_diffusion", True)
        ),
    ).to(device)

    history = train_edgewise_transition_model(
        model=model,
        x_src_train=x_src_train,
        x_tgt_train=x_tgt_train,
        context=context_bundle["context"].to(device),
        edge_id=edge_id,
        learning_rate=float(cfg.get("train", {}).get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("train", {}).get("weight_decay", 1e-4)),
        max_epochs=int(cfg.get("train", {}).get("max_epochs", 2)),
        steps_per_epoch=int(cfg.get("train", {}).get("steps_per_epoch", 2)),
        batch_cells=int(cfg.get("train", {}).get("batch_cells", 32)),
        sigma=sigma,
        diffusion_weight=diffusion_weight,
        epsilon=float(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("ot_epsilon", 0.05)),
        sinkhorn_iters=int(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("sinkhorn_iters", 80)),
        num_ot_pairs=int(cfg.get("transition_model", {}).get("schrodinger_bridge", {}).get("num_ot_pairs", 128)),
        seed=int(cfg.get("seed", 42)),
        extra_cost=extra_cost,
    )

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
        "context": context_bundle["context"].to(device),
        "shuffled_context": context_bundle["shuffled_context"].to(device),
        "edge_id": edge_id,
        "x_src_test": x_src_test,
        "x_tgt_test": x_tgt_test,
        "typed_subset_tokens": context_bundle["typed_subset_tokens"],
        "typed_feature_names": context_bundle["typed_feature_names"],
    }
