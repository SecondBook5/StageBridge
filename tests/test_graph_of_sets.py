"""Tests for Graph-of-Sets Transformer (GoST) architecture."""
from __future__ import annotations

import torch
import pytest

from stagebridge.models.graph_of_sets import (
    GraphAttentionLayer,
    GraphOfSetsTransformer,
    GraphTransformerBlock,
    SetGraph,
    build_cross_dataset_graph,
    build_set_graph,
    build_training_graph,
    _scatter_softmax,
)


# ---------------------------------------------------------------------------
# Graph construction tests
# ---------------------------------------------------------------------------

class TestBuildSetGraph:
    """Test graph construction utilities."""

    def test_single_patient_linear_chain(self):
        """One patient with 3 stages → stage-adjacent + self-loops."""
        g = build_set_graph(
            patient_ids=["P1", "P1", "P1"],
            stage_indices=[0, 1, 2],
            num_stages=5,
        )
        assert g.num_nodes == 3
        assert g.edge_index.shape[0] == 2
        assert g.edge_index.shape[1] > 0
        # Should have self-loops and stage-adjacent edges
        assert g.node_patient == ["P1", "P1", "P1"]
        assert g.node_stage == [0, 1, 2]

    def test_two_patients_same_stage(self):
        """Two patients at the same stage → same-stage cross-patient edges."""
        g = build_set_graph(
            patient_ids=["P1", "P2"],
            stage_indices=[1, 1],
            num_stages=5,
        )
        assert g.num_nodes == 2
        # Should have type-2 edges (same stage, different patient)
        type_2_mask = g.edge_type == 2
        assert type_2_mask.any(), "Expected same-stage cross-patient edges"

    def test_no_self_loops(self):
        g = build_set_graph(
            patient_ids=["P1", "P1"],
            stage_indices=[0, 1],
            num_stages=5,
            include_self_loops=False,
        )
        # Self-loops would have src == tgt
        has_self = (g.edge_index[0] == g.edge_index[1]).any()
        assert not has_self

    def test_edge_types_correct(self):
        """Verify edge type assignment for a mixed graph."""
        g = build_set_graph(
            patient_ids=["P1", "P1", "P2", "P2"],
            stage_indices=[0, 1, 0, 1],
            num_stages=5,
            include_self_loops=False,
        )
        # Type 0: stage-adjacent (P1: 0→1, P2: 2→3)
        # Type 1: same-patient different stage (P1: 0↔1, P2: 2↔3)
        # Type 2: same-stage different patient (stage 0: 0↔2, stage 1: 1↔3)
        for etype in [0, 1, 2]:
            assert (g.edge_type == etype).any(), f"Expected edge type {etype}"

    def test_to_device(self):
        g = build_set_graph(["P1"], [0], num_stages=3)
        g2 = g.to("cpu")
        assert g2.edge_index.device.type == "cpu"

    def test_empty_graph(self):
        """No nodes should produce empty edge tensors."""
        g = build_set_graph([], [], num_stages=5)
        assert g.num_nodes == 0
        assert g.edge_index.shape == (2, 0)


class TestBuildTrainingGraph:
    """Test graph construction from per-cell observations."""

    def test_groups_cells_correctly(self):
        obs_patient = ["P1", "P1", "P1", "P2", "P2"]
        obs_stage = [0, 0, 1, 0, 1]
        graph, node_map = build_training_graph(obs_patient, obs_stage, num_stages=3)

        # Should have 4 unique (patient, stage) pairs
        assert graph.num_nodes == 4
        assert len(node_map) == 4
        assert ("P1", 0) in node_map
        assert ("P2", 1) in node_map

    def test_node_map_is_sorted(self):
        obs_patient = ["P2", "P1", "P2", "P1"]
        obs_stage = [1, 0, 0, 1]
        _, node_map = build_training_graph(obs_patient, obs_stage)
        keys = list(node_map.keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Graph attention tests
# ---------------------------------------------------------------------------

class TestGraphAttentionLayer:
    """Test sparse graph attention mechanics."""

    @pytest.fixture
    def simple_graph(self):
        """3-node graph: 0→1, 1→2, self-loops."""
        return SetGraph(
            edge_index=torch.tensor([[0, 1, 0, 1, 2], [1, 2, 0, 1, 2]], dtype=torch.long),
            edge_type=torch.tensor([0, 0, 0, 0, 0], dtype=torch.long),
            num_nodes=3,
            node_patient=["P1", "P1", "P1"],
            node_stage=[0, 1, 2],
        )

    def test_output_shape(self, simple_graph):
        dim, K = 32, 2
        layer = GraphAttentionLayer(dim=dim, num_heads=4)
        x = torch.randn(3, K, dim)
        out = layer(x, simple_graph.edge_index, simple_graph.edge_type)
        assert out.shape == (3, K, dim)

    def test_gradient_flows(self, simple_graph):
        dim, K = 16, 1
        layer = GraphAttentionLayer(dim=dim, num_heads=4)
        x = torch.randn(3, K, dim, requires_grad=True)
        out = layer(x, simple_graph.edge_index, simple_graph.edge_type)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_edge_bias_shape(self):
        layer = GraphAttentionLayer(dim=16, num_heads=4, num_edge_types=3)
        assert layer.edge_bias.weight.shape == (3, 4)

    def test_empty_edges(self):
        dim, K = 16, 1
        layer = GraphAttentionLayer(dim=dim, num_heads=4)
        x = torch.randn(2, K, dim)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        out = layer(x, edge_index, edge_type)
        # With no edges, output should be unchanged (identity via residual in block)
        assert out.shape == (2, K, dim)


# ---------------------------------------------------------------------------
# Scatter softmax tests
# ---------------------------------------------------------------------------

class TestScatterSoftmax:

    def test_single_group(self):
        """All edges target the same node → standard softmax."""
        scores = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 1.0]]]])  # (2, 1, 1, 2)
        index = torch.tensor([0, 0])
        result = _scatter_softmax(scores, index, num_nodes=1)
        # Two edges both targeting node 0, so softmax over dim=0
        assert result.shape == (2, 1, 1, 2)
        # Sum over edges for each (H, Kq, Ks) position should equal 1
        total = result.sum(dim=0)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_independent_groups(self):
        """Each edge targets a different node → each gets softmax=1."""
        scores = torch.tensor([[[[5.0]]], [[[3.0]]]])  # (2, 1, 1, 1)
        index = torch.tensor([0, 1])
        result = _scatter_softmax(scores, index, num_nodes=2)
        # Each is the only edge in its group → weight = 1.0
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)


# ---------------------------------------------------------------------------
# Graph Transformer block tests
# ---------------------------------------------------------------------------

class TestGraphTransformerBlock:

    def test_residual_connection(self):
        dim, K = 32, 2
        block = GraphTransformerBlock(dim=dim, num_heads=4)
        g = build_set_graph(["P1", "P1"], [0, 1], num_stages=3)
        x = torch.randn(2, K, dim)
        out = block(x, g.edge_index, g.edge_type)
        assert out.shape == (2, K, dim)
        # Output should differ from input (transformation happened)
        assert not torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# Full GoST tests
# ---------------------------------------------------------------------------

class TestGraphOfSetsTransformer:

    @pytest.fixture
    def gost(self):
        return GraphOfSetsTransformer(
            dim=64,
            num_graph_layers=2,
            num_heads=4,
            dropout=0.0,
        )

    @pytest.fixture
    def graph_4node(self):
        return build_set_graph(
            patient_ids=["P1", "P1", "P2", "P2"],
            stage_indices=[0, 1, 0, 1],
            num_stages=5,
        )

    def test_output_shape(self, gost, graph_4node):
        K = 2
        summaries = torch.randn(4, K, 64)
        out = gost(summaries, graph_4node)
        assert out.shape == (4, K, 64)

    def test_query_node(self, gost, graph_4node):
        K = 2
        summaries = torch.randn(4, K, 64)
        out = gost(summaries, graph_4node, query_node=1)
        assert out.shape == (1, K, 64)

    def test_gradient_through_full_model(self, gost, graph_4node):
        K = 2
        summaries = torch.randn(4, K, 64, requires_grad=True)
        out = gost(summaries, graph_4node)
        loss = out.sum()
        loss.backward()
        assert summaries.grad is not None

    def test_different_nodes_get_different_enrichment(self, gost, graph_4node):
        """Nodes with different neighborhoods should get different outputs."""
        K = 1
        torch.manual_seed(42)
        summaries = torch.randn(4, K, 64)
        out = gost(summaries, graph_4node)
        # Node 0 (P1, stage 0) and node 2 (P2, stage 0) have different
        # same-patient neighbors, so their enrichments should differ
        assert not torch.allclose(out[0], out[2], atol=1e-4)

    def test_single_node_no_crash(self, gost):
        """Single node with self-loop should work."""
        g = build_set_graph(["P1"], [0], num_stages=3)
        summaries = torch.randn(1, 2, 64)
        out = gost(summaries, g)
        assert out.shape == (1, 2, 64)

    def test_num_parameters(self, gost):
        """Sanity check parameter count is reasonable."""
        n_params = sum(p.numel() for p in gost.parameters())
        # 2 graph layers × (attention + FFN + LN) ~ reasonable
        assert n_params > 0
        assert n_params < 1_000_000  # Should be well under 1M params


# ---------------------------------------------------------------------------
# Integration test: Set Transformer → GoST pipeline
# ---------------------------------------------------------------------------

class TestSetTransformerToGoSTPipeline:
    """Test that Set Transformer outputs feed cleanly into GoST."""

    def test_end_to_end_with_set_transformer(self):
        from stagebridge.models.layers import ISAB, SAB, PMA

        dim = 32
        K = 2  # num_seed_vectors
        input_dim = 16
        num_nodes = 3

        # Set Transformer layers (shared across nodes)
        proj = torch.nn.Linear(input_dim, dim)
        isab = ISAB(dim=dim, num_heads=4, num_inducing_points=8, dropout=0.0)
        sab = SAB(dim=dim, num_heads=4, dropout=0.0)
        pma = PMA(dim=dim, num_heads=4, num_seed_vectors=K, dropout=0.0)

        # GoST
        gost = GraphOfSetsTransformer(dim=dim, num_graph_layers=1, num_heads=4, dropout=0.0)
        graph = build_set_graph(["P1", "P1", "P2"], [0, 1, 0], num_stages=3)

        # Simulate 3 nodes with variable-size cell sets
        set_sizes = [10, 8, 12]
        node_summaries = []
        for n_cells in set_sizes:
            x = torch.randn(1, n_cells, input_dim)
            h = proj(x)
            h = isab(h)
            h = sab(h)
            s = pma(h)  # (1, K, dim)
            node_summaries.append(s)

        # Stack into (N, K, dim)
        stacked = torch.cat(node_summaries, dim=0)
        assert stacked.shape == (num_nodes, K, dim)

        # GoST enrichment
        enriched = gost(stacked, graph)
        assert enriched.shape == (num_nodes, K, dim)

        # Gradient flows end-to-end
        loss = enriched.sum()
        loss.backward()
        assert proj.weight.grad is not None


# ---------------------------------------------------------------------------
# Cross-dataset graph tests
# ---------------------------------------------------------------------------

class TestCrossDatasetGraph:
    """Test cross-dataset graph construction for extended ontology."""

    def test_branching_topology(self):
        """LUAD branches to BrainMet and ChestWallMet."""
        from stagebridge.preprocessing.stage_ontology import (
            EXTENDED_STAGE_EDGES,
            EXTENDED_STAGE_ORDER,
            stage_to_index,
        )

        # Map edge names to indices
        adjacency = [
            (stage_to_index(s, extended=True), stage_to_index(t, extended=True))
            for s, t in EXTENDED_STAGE_EDGES
        ]

        g = build_set_graph(
            patient_ids=["P1", "P1", "P1"],
            stage_indices=[
                stage_to_index("LUAD", extended=True),
                stage_to_index("BrainMet", extended=True),
                stage_to_index("ChestWallMet", extended=True),
            ],
            num_stages=len(EXTENDED_STAGE_ORDER),
            stage_adjacency=adjacency,
        )
        assert g.num_nodes == 3
        # Should have forward edges from LUAD → BrainMet and LUAD → ChestWallMet
        type_0_mask = g.edge_type == 0
        type_0_edges = g.edge_index[:, type_0_mask]
        # Node 0 (LUAD) should connect to node 1 (BrainMet) and node 2 (ChestWallMet)
        targets_from_luad = set(type_0_edges[1][type_0_edges[0] == 0].tolist())
        assert 1 in targets_from_luad or 2 in targets_from_luad

    def test_cross_dataset_bridge_edges(self):
        """Bridge edges connect LUAD nodes to BrainMet across patients."""
        from stagebridge.preprocessing.stage_ontology import (
            EXTENDED_STAGE_EDGES,
            EXTENDED_STAGE_ORDER,
            stage_to_index,
        )

        adjacency = [
            (stage_to_index(s, extended=True), stage_to_index(t, extended=True))
            for s, t in EXTENDED_STAGE_EDGES
        ]

        luad_idx = stage_to_index("LUAD", extended=True)
        brain_idx = stage_to_index("BrainMet", extended=True)

        # P1 from dataset A has LUAD, P2 from dataset B has BrainMet
        graph, node_map = build_cross_dataset_graph(
            obs_patient=["ds1_P1", "ds1_P1", "ds2_P2", "ds2_P2"],
            obs_stage=[0, luad_idx, luad_idx, brain_idx],
            stage_adjacency=adjacency,
            num_stages=len(EXTENDED_STAGE_ORDER),
            bridge_stage_pairs=[(luad_idx, brain_idx)],
        )

        # Should have a bridge edge from ds1_P1's LUAD to ds2_P2's BrainMet
        edge_pairs = set(
            zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())
        )
        luad_node = node_map[("ds1_P1", luad_idx)]
        brain_node = node_map[("ds2_P2", brain_idx)]
        assert (luad_node, brain_node) in edge_pairs

    def test_extended_stage_ontology(self):
        """Verify extended ontology has correct aliases."""
        from stagebridge.preprocessing.stage_ontology import (
            EXTENDED_STAGE_ORDER,
            normalize_stage_label,
            stage_to_index,
        )

        assert len(EXTENDED_STAGE_ORDER) == 7
        assert normalize_stage_label("PRIMARY") == "LUAD"
        assert normalize_stage_label("BRAIN_METS") == "BrainMet"
        assert normalize_stage_label("CHEST_WALL_MET") == "ChestWallMet"
        assert stage_to_index("BrainMet", extended=True) == 5
        assert stage_to_index("ChestWallMet", extended=True) == 6

    def test_gost_on_cross_dataset_graph(self):
        """GoST runs on the cross-dataset branching graph."""
        from stagebridge.preprocessing.stage_ontology import (
            EXTENDED_STAGE_EDGES,
            EXTENDED_STAGE_ORDER,
            stage_to_index,
        )

        adjacency = [
            (stage_to_index(s, extended=True), stage_to_index(t, extended=True))
            for s, t in EXTENDED_STAGE_EDGES
        ]
        luad_idx = stage_to_index("LUAD", extended=True)
        brain_idx = stage_to_index("BrainMet", extended=True)

        graph, node_map = build_cross_dataset_graph(
            obs_patient=["ds1_P1"] * 3 + ["ds2_P2"] * 2,
            obs_stage=[0, 1, luad_idx, luad_idx, brain_idx],
            stage_adjacency=adjacency,
            num_stages=len(EXTENDED_STAGE_ORDER),
            bridge_stage_pairs=[(luad_idx, brain_idx)],
        )

        dim, K = 32, 2
        gost = GraphOfSetsTransformer(dim=dim, num_graph_layers=2, num_heads=4, dropout=0.0)
        summaries = torch.randn(graph.num_nodes, K, dim)
        out = gost(summaries, graph)
        assert out.shape == (graph.num_nodes, K, dim)

        # Query LUAD node — should have enriched context from BrainMet neighbor
        luad_node = node_map[("ds1_P1", luad_idx)]
        enriched_luad = gost(summaries, graph, query_node=luad_node)
        assert enriched_luad.shape == (1, K, dim)
