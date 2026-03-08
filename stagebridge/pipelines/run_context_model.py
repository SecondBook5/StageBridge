"""Context-model pipeline entrypoint."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from stagebridge.context_model.cell_to_spot_assignment import select_stage_donor_token_context
from stagebridge.context_model.graph_builder import build_spatial_knn_graph
from stagebridge.context_model.graph_encoder import GraphOfSetsContextEncoder
from stagebridge.context_model.hierarchical_transformer import TypedHierarchicalTransformerEncoder, dataset_name_to_id
from stagebridge.context_model.set_encoder import DeepSetsContextEncoder, PooledContextEncoder, TypedSetContextEncoder
from stagebridge.context_model.token_builder import build_typed_spot_tokens
from stagebridge.transition_model.disease_edges import edge_id_map, edge_label, resolve_disease_edge
from stagebridge.pipelines.run_spatial_mapping import run_spatial_mapping


def _sample_stage_donor_rows(
    typed: Any,
    *,
    donor_id: str,
    stage: str,
    max_spots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    obs = typed.obs
    mask = (obs["donor_id"].astype(str) == str(donor_id)) & (obs["stage"].astype(str) == str(stage))
    rows = np.flatnonzero(mask.to_numpy())
    if rows.size == 0:
        raise ValueError(f"No typed-token rows found for donor={donor_id}, stage={stage}.")
    if rows.size > max_spots > 0:
        rng = np.random.default_rng(int(seed))
        rows = np.sort(rng.choice(rows, size=int(max_spots), replace=False))
    return typed.tokens[rows], typed.coords[rows], obs.iloc[rows].reset_index(drop=True)


def _resolve_mapping_result(
    cfg: DictConfig,
    *,
    spatial_output: dict[str, Any] | None = None,
) -> Any:
    output = spatial_output or run_spatial_mapping(cfg)
    result = output.get("mapping_result")
    if result is None or result.compositions is None or result.coords is None or result.obs is None:
        method = str(cfg.get("spatial_mapping", {}).get("method", "tangram"))
        status = None if result is None else result.status
        raise ValueError(
            f"Spatial mapping method '{method}' is not runnable for context construction "
            f"(status={status!r})."
        )
    return output


def run_context_model(
    cfg: DictConfig,
    *,
    spatial_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = str(cfg.get("context_model", {}).get("mode", "set_only"))
    if mode == "rna_only":
        return {
            "ok": True,
            "pipeline": "context_model",
            "status": "complete",
            "context_model": {"mode": "rna_only", "notes": "No spatial context encoding applied."},
        }

    spatial_payload = _resolve_mapping_result(cfg, spatial_output=spatial_output)
    spatial = spatial_payload["mapping_result"]
    typed = build_typed_spot_tokens(
        spatial.compositions,
        spatial.coords,
        spatial.obs,
        spatial.feature_names,
    )
    example_row = typed.obs.iloc[0]
    max_context_spots = int(cfg.get("context_model", {}).get("max_context_spots", 128))
    seed = int(cfg.get("seed", 42))

    if mode in {"pooled", "deep_sets", "set_only", "typed_hierarchical_transformer"}:
        sample_mask = (
            (typed.obs["donor_id"].astype(str) == str(example_row["donor_id"]))
            & (typed.obs["stage"].astype(str) == str(example_row["stage"]))
        )
        sample_rows = np.flatnonzero(sample_mask.to_numpy())
        if sample_rows.size > max_context_spots > 0:
            rng = np.random.default_rng(int(seed))
            sample_rows = np.sort(rng.choice(sample_rows, size=int(max_context_spots), replace=False))
        example_tokens, example_coords = select_stage_donor_token_context(
            typed.tokens,
            typed.coords,
            typed.obs,
            donor_id=str(example_row["donor_id"]),
            stage=str(example_row["stage"]),
            max_spots=max_context_spots,
            seed=seed,
        )
        example_confidence = typed.token_confidence[sample_rows]
        if mode == "pooled":
            encoder = PooledContextEncoder(
                input_dim=typed.tokens.shape[1],
                hidden_dim=int(cfg.get("context_model", {}).get("hidden_dim", 128)),
            )
        elif mode == "deep_sets":
            encoder = DeepSetsContextEncoder(
                input_dim=typed.tokens.shape[1],
                hidden_dim=int(cfg.get("context_model", {}).get("hidden_dim", 128)),
                dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
            )
        elif mode == "set_only":
            encoder = TypedSetContextEncoder(
                input_dim=typed.tokens.shape[1],
                hidden_dim=int(cfg.get("context_model", {}).get("hidden_dim", 128)),
                num_heads=int(cfg.get("context_model", {}).get("num_heads", 4)),
                num_inducing_points=int(cfg.get("context_model", {}).get("num_inducing_points", 16)),
                dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
                num_token_types=len(typed.schema.typed_feature_names),
                use_spatial_rpe=bool(cfg.get("context_model", {}).get("use_spatial_rpe", True)),
                token_dropout_rate=float(cfg.get("context_model", {}).get("token_dropout_rate", 0.05)),
                use_confidence_gate=bool(cfg.get("context_model", {}).get("use_confidence_gate", True)),
            )
        else:
            active_edge = resolve_disease_edge(cfg.get("transition_model", {}).get("active_edge"))
            encoder = TypedHierarchicalTransformerEncoder(
                input_dim=typed.tokens.shape[1],
                hidden_dim=int(cfg.get("context_model", {}).get("hidden_dim", 128)),
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
        with torch.no_grad():
            if mode == "set_only":
                summary = encoder(
                    torch.tensor(example_tokens, dtype=torch.float32),
                    token_coords=torch.tensor(example_coords, dtype=torch.float32),
                    token_confidence=torch.tensor(example_confidence, dtype=torch.float32),
                )
            elif mode == "typed_hierarchical_transformer":
                summary = encoder(
                    torch.tensor(example_tokens, dtype=torch.float32),
                    token_type_ids=torch.tensor(typed.token_type_ids[sample_rows], dtype=torch.long),
                    token_coords=torch.tensor(example_coords, dtype=torch.float32),
                    token_confidence=torch.tensor(example_confidence, dtype=torch.float32),
                    dataset_ids=torch.tensor([dataset_name_to_id(str(cfg.get("data", {}).get("dataset", "luad_evo")))], dtype=torch.long),
                    edge_ids=torch.tensor([edge_id_map()[edge_label(active_edge)]], dtype=torch.long),
                    return_attention=True,
                )
            else:
                summary = encoder(torch.tensor(example_tokens, dtype=torch.float32))

        return {
            "ok": True,
            "pipeline": "context_model",
            "status": "complete",
            "context_model": {
                "mode": mode,
                "typed_token_summary": typed.summary(),
                "spatial_mapping_method": spatial.method,
                "example_context_norm": float(torch.norm(summary.pooled_context).item()),
                "example_context_dim": int(np.asarray(summary.pooled_context).shape[0]),
                "mean_token_confidence": typed.summary().get("mean_token_confidence", 0.0),
                "example_context_tokens": 0 if getattr(summary, "context_tokens", None) is None else int(summary.context_tokens.shape[-2]),
                "dataset_name": str(cfg.get("data", {}).get("dataset", "luad_evo")),
                "dataset_embedding_enabled": bool(mode == "typed_hierarchical_transformer"),
            },
            "typed_tokens": typed,
            "set_encoder": encoder,
        }

    if mode != "graph_of_sets":
        raise ValueError(f"Unsupported context_model.mode '{mode}'.")

    node_tokens, node_coords, node_obs = _sample_stage_donor_rows(
        typed,
        donor_id=str(example_row["donor_id"]),
        stage=str(example_row["stage"]),
        max_spots=max_context_spots,
        seed=seed,
    )
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
        hidden_dim=int(cfg.get("context_model", {}).get("hidden_dim", 128)),
        num_graph_layers=int(cfg.get("context_model", {}).get("graph_num_layers", 2)),
        num_heads=int(cfg.get("context_model", {}).get("graph_num_heads", 4)),
        dropout=float(cfg.get("transition_model", {}).get("stochastic_dynamics", {}).get("dropout", 0.1)),
    )
    with torch.no_grad():
        graph_summary = encoder(torch.tensor(node_tokens, dtype=torch.float32), graph)

    return {
        "ok": True,
        "pipeline": "context_model",
        "status": "complete",
            "context_model": {
                "mode": mode,
                "typed_token_summary": typed.summary(),
                "spatial_mapping_method": spatial.method,
                "graph_context_norm": float(torch.norm(graph_summary.pooled_context).item()),
                "graph_context_dim": int(np.asarray(graph_summary.pooled_context).shape[0]),
                "graph_num_nodes": int(graph_summary.num_nodes),
                "graph_num_edges": int(graph_summary.num_edges),
            },
        "typed_tokens": typed,
        "graph_encoder": encoder,
        "graph_summary": graph_summary,
        "graph": graph,
    }
