from __future__ import annotations

import numpy as np
import torch

from stagebridge.context_model.graph_builder import build_spatial_knn_graph
from stagebridge.context_model.graph_encoder import GraphOfSetsContextEncoder


def test_build_spatial_knn_graph_creates_edges() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    graph = build_spatial_knn_graph(
        coords=coords,
        patient_ids=["P1", "P1", "P1", "P1"],
        stage_indices=[0, 0, 0, 0],
        dataset_ids=["luad_evo"] * 4,
        k=2,
        include_cross_patient=False,
        include_cross_stage=False,
    )

    assert graph.num_nodes == 4
    assert graph.edge_index.shape[1] > 0
    assert graph.edge_type.shape[0] == graph.edge_index.shape[1]


def test_graph_of_sets_context_encoder_returns_pooled_context() -> None:
    coords = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    graph = build_spatial_knn_graph(
        coords=coords,
        patient_ids=["P1", "P1", "P1", "P1"],
        stage_indices=[0, 0, 0, 0],
        dataset_ids=["luad_evo"] * 4,
        k=2,
        include_cross_patient=False,
        include_cross_stage=False,
    )
    encoder = GraphOfSetsContextEncoder(
        input_dim=4,
        hidden_dim=16,
        num_graph_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    node_features = torch.rand(4, 4)

    summary = encoder(node_features, graph)

    assert summary.pooled_context.shape == (16,)
    assert summary.node_contexts.shape == (4, 16)
    assert summary.num_nodes == 4
    assert summary.num_edges == int(graph.edge_index.shape[1])
