"""Graph-of-Sets Transformer (GoST) for cross-population context propagation.

Architecture overview:

    Level 1 — Intra-set (Set Transformer):
        Each graph node holds a set of cells from one (patient, stage) pair.
        ISAB → SAB → PMA compresses each set into K summary tokens.

    Level 2 — Inter-set (Graph Transformer):
        Summary tokens from each node attend to summaries from neighboring
        nodes in a sparse graph.  Edges encode:
          • stage adjacency (forward progression, e.g. AAH → AIS)
          • same-patient links (different stages from the same donor)
          • same-stage links (different patients at the same stage)

    Level 3 — Flow conditioning:
        Graph-enriched summary tokens condition the OT flow matching drift
        head, replacing (or augmenting) the single-set PMA context.

This module provides the Graph Transformer layers and graph construction
utilities.  The intra-set encoding re-uses the rebuilt context-model
set-encoder blocks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from stagebridge.context_model.set_encoder import FeedForwardBlock


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

@dataclass
class SetGraph:
    """Sparse graph over cell sets.

    Each node ``i`` corresponds to a (patient, stage) cell set.

    Attributes
    ----------
    edge_index : (2, E) long tensor
        Source→target edges.  Self-loops are allowed.
    edge_type : (E,) long tensor
        Edge type encoding:
            0 = stage-adjacent (forward progression)
            1 = same-patient cross-stage
            2 = same-stage cross-patient
    num_nodes : int
        Total number of nodes in the graph.
    node_patient : list[str]
        Patient ID per node.
    node_stage : list[int]
        Stage index per node.
    """

    edge_index: Tensor
    edge_type: Tensor
    num_nodes: int
    node_patient: list[str]
    node_stage: list[int]

    def to(self, device: str | torch.device) -> "SetGraph":
        return SetGraph(
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            num_nodes=self.num_nodes,
            node_patient=self.node_patient,
            node_stage=self.node_stage,
        )


NUM_EDGE_TYPES = 3  # stage-adjacent, same-patient, same-stage (population graph)
NUM_SPATIAL_EDGE_TYPES = 4  # spatial-kNN, same-patient, same-stage, cross-dataset


def build_set_graph(
    patient_ids: list[str],
    stage_indices: list[int],
    num_stages: int = 5,
    include_self_loops: bool = True,
    stage_adjacency: list[tuple[int, int]] | None = None,
) -> SetGraph:
    """Build a graph over (patient, stage) nodes.

    Parameters
    ----------
    patient_ids : list[str]
        Patient ID per node (length N).
    stage_indices : list[int]
        Stage index per node (length N).
    num_stages : int
        Total stages in the ontology.
    include_self_loops : bool
        Whether to include self-loop edges (type 0).
    stage_adjacency : list of (src_stage, tgt_stage) pairs, optional
        Explicit stage adjacency edges.  When None, assumes linear chain
        (stage_i → stage_i+1).  For branching topologies (e.g., LUAD branching
        to BrainMet and ChestWallMet), pass the explicit edge list.

    Returns
    -------
    SetGraph with edges for stage adjacency, same-patient, and same-stage.
    """
    n = len(patient_ids)
    assert len(stage_indices) == n

    src_list: list[int] = []
    tgt_list: list[int] = []
    type_list: list[int] = []

    # Build lookup maps
    patient_to_nodes: dict[str, list[int]] = {}
    stage_to_nodes: dict[int, list[int]] = {}
    for i in range(n):
        patient_to_nodes.setdefault(patient_ids[i], []).append(i)
        stage_to_nodes.setdefault(stage_indices[i], []).append(i)

    # Build forward adjacency map from stage_adjacency or linear default
    forward_neighbors: dict[int, set[int]] = {}
    if stage_adjacency is not None:
        for s_src, s_tgt in stage_adjacency:
            forward_neighbors.setdefault(s_src, set()).add(s_tgt)
    else:
        for s in range(num_stages - 1):
            forward_neighbors[s] = {s + 1}

    for i in range(n):
        pid_i = patient_ids[i]
        stage_i = stage_indices[i]

        if include_self_loops:
            src_list.append(i)
            tgt_list.append(i)
            type_list.append(0)

        # Type 0: stage-adjacent (forward, following topology edges)
        for next_stage in forward_neighbors.get(stage_i, set()):
            if next_stage in stage_to_nodes:
                for j in stage_to_nodes[next_stage]:
                    # Same patient for within-dataset edges, or any patient
                    # for cross-dataset bridge edges (different datasets
                    # have different patient IDs, so same-patient check
                    # naturally allows cross-dataset when appropriate)
                    if patient_ids[j] == pid_i:
                        src_list.append(i)
                        tgt_list.append(j)
                        type_list.append(0)

        # Type 1: same patient, different stage (bidirectional)
        for j in patient_to_nodes[pid_i]:
            if j != i:
                src_list.append(i)
                tgt_list.append(j)
                type_list.append(1)

        # Type 2: same stage, different patient (bidirectional)
        for j in stage_to_nodes[stage_i]:
            if j != i:
                src_list.append(i)
                tgt_list.append(j)
                type_list.append(2)

    device = torch.device("cpu")
    if not src_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_type = torch.zeros((0,), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long, device=device)
        edge_type = torch.tensor(type_list, dtype=torch.long, device=device)

    return SetGraph(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=n,
        node_patient=list(patient_ids),
        node_stage=list(stage_indices),
    )


# ---------------------------------------------------------------------------
# Graph Transformer layers
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """Multi-head attention over graph neighbors with typed edge bias.

    For each target node i, attends over the set of source nodes j where
    (j, i) is an edge.  Edge types contribute learned bias to attention
    logits.

    This implements sparse attention — nodes only attend to their neighbors,
    not all other nodes.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_edge_types: int = NUM_EDGE_TYPES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Learned per-edge-type, per-head scalar bias
        self.edge_bias = nn.Embedding(num_edge_types, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """Graph attention forward pass.

        Parameters
        ----------
        x : (N, K, D)
            Node features — K tokens per node, D dimensions.
        edge_index : (2, E)
            Source→target edges.
        edge_type : (E,)
            Edge type per edge.

        Returns
        -------
        (N, K, D) — updated node features.
        """
        N, K, D = x.shape
        H = self.num_heads
        d_h = self.head_dim

        # Project all tokens
        Q = self.q_proj(x).reshape(N, K, H, d_h)  # (N, K, H, d_h)
        K_proj = self.k_proj(x).reshape(N, K, H, d_h)
        V = self.v_proj(x).reshape(N, K, H, d_h)

        src_nodes = edge_index[0]  # (E,)
        tgt_nodes = edge_index[1]  # (E,)
        E = src_nodes.shape[0]

        if E == 0:
            return x

        # Gather source key/value tokens: (E, K, H, d_h)
        k_src = K_proj[src_nodes]
        v_src = V[src_nodes]

        # Gather target query tokens: (E, K, H, d_h)
        q_tgt = Q[tgt_nodes]

        # Attention scores: each target token attends to each source token
        # (E, K, H, d_h) x (E, K, H, d_h) → (E, K_q, K_s, H)
        # For efficiency, compute per-edge attention as outer product over tokens
        # q_tgt: (E, K, H, d_h), k_src: (E, K, H, d_h)
        # scores: (E, H, K_q, K_s) via einsum
        scores = torch.einsum("eqhd,ekhd->ehqk", q_tgt, k_src) / self.scale

        # Add edge-type bias: (E, H) → (E, H, 1, 1) broadcast
        e_bias = self.edge_bias(edge_type)  # (E, H)
        scores = scores + e_bias.unsqueeze(-1).unsqueeze(-1)

        # Softmax per target node over its neighbors.
        # We need to group edges by target node and softmax within groups.
        # Use scatter-based softmax for efficiency.
        attn_weights = _scatter_softmax(scores, tgt_nodes, N)  # (E, H, K, K)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of source values
        # attn: (E, H, K_q, K_s), v_src: (E, K_s, H, d_h)
        # rearrange v_src to (E, H, K_s, d_h)
        v_src_h = v_src.permute(0, 2, 1, 3)  # (E, H, K, d_h)
        # out: (E, H, K_q, d_h) via einsum
        msg = torch.einsum("ehqk,ehkd->ehqd", attn_weights, v_src_h)

        # Scatter-add messages to target nodes
        # msg: (E, H, K, d_h) → (N, H, K, d_h)
        out = torch.zeros(N, H, K, d_h, device=x.device, dtype=x.dtype)
        tgt_expanded = tgt_nodes.view(E, 1, 1, 1).expand(-1, H, K, d_h)
        out.scatter_add_(0, tgt_expanded, msg)

        # Reshape: (N, H, K, d_h) → (N, K, D)
        out = out.permute(0, 2, 1, 3).reshape(N, K, D)
        return self.out_proj(out)


def _scatter_softmax(
    scores: Tensor,
    index: Tensor,
    num_nodes: int,
) -> Tensor:
    """Softmax over scores grouped by index (target node).

    Parameters
    ----------
    scores : (E, H, K_q, K_s)
    index : (E,) — target node indices
    num_nodes : int

    Returns
    -------
    (E, H, K_q, K_s) — attention weights normalized per target node.
    """
    E, H, Kq, Ks = scores.shape

    # Compute per-group max for numerical stability
    idx = index.view(E, 1, 1, 1).expand_as(scores)
    max_vals = torch.full((num_nodes, H, Kq, Ks), float("-inf"), device=scores.device, dtype=scores.dtype)
    max_vals.scatter_reduce_(0, idx, scores, reduce="amax", include_self=False)
    max_per_edge = max_vals[index]  # (E, H, Kq, Ks)

    exp_scores = (scores - max_per_edge).exp()

    # Sum per group
    sum_vals = torch.zeros(num_nodes, H, Kq, Ks, device=scores.device, dtype=scores.dtype)
    sum_vals.scatter_add_(0, idx, exp_scores)
    sum_per_edge = sum_vals[index].clamp_min(1e-12)

    return exp_scores / sum_per_edge


class GraphTransformerBlock(nn.Module):
    """One block of Graph Transformer: graph attention + FFN + residual + LN."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_edge_types: int = NUM_EDGE_TYPES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = GraphAttentionLayer(
            dim=dim,
            num_heads=num_heads,
            num_edge_types=num_edge_types,
            dropout=dropout,
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ff = FeedForwardBlock(dim=dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (N, K, D) — K summary tokens per node
        edge_index : (2, E)
        edge_type : (E,)
        """
        h = self.attn(x, edge_index, edge_type)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Full Graph-of-Sets Transformer
# ---------------------------------------------------------------------------

class GraphOfSetsTransformer(nn.Module):
    """Joint Set Transformer + Graph Transformer for cross-population flow conditioning.

    Architecture:
        1. Each (patient, stage) node's cell set is encoded by the shared
           Set Transformer (ISAB → SAB → PMA) into K summary tokens.
        2. Summary tokens are passed through L Graph Transformer blocks
           that propagate context along the population graph.
        3. The graph-enriched summary for the target (patient, stage) node
           is returned as conditioning for the flow matching drift head.

    The Set Transformer encoder is NOT owned by this module — it receives
    pre-computed summary tokens.  This allows the existing StageBridgeModel
    to compute intra-set summaries and then optionally pass them through
    GoST for inter-set enrichment.
    """

    def __init__(
        self,
        dim: int,
        num_graph_layers: int = 2,
        num_heads: int = 4,
        num_edge_types: int = NUM_EDGE_TYPES,
        dropout: float = 0.1,
        use_edge_distance: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_graph_layers = num_graph_layers
        self.use_edge_distance = use_edge_distance

        # Optional edge distance encoding (for spatial graphs)
        if use_edge_distance:
            self.dist_encoder = nn.Sequential(
                nn.Linear(1, num_heads),
                nn.GELU(),
            )

        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_edge_types=num_edge_types,
                dropout=dropout,
            )
            for _ in range(num_graph_layers)
        ])

    def forward(
        self,
        node_summaries: Tensor,
        graph: SetGraph,
        query_node: int | None = None,
    ) -> Tensor:
        """Enrich node summaries via graph message passing.

        Parameters
        ----------
        node_summaries : (N, K, D)
            Per-node summary tokens from the Set Transformer PMA.
            N = number of nodes, K = num_seed_vectors, D = hidden_dim.
        graph : SetGraph
            Graph structure with edge_index and edge_type.
        query_node : int or None
            If specified, return only the enriched summary for this node.
            Shape: (1, K, D) or (K, D) if squeezed.

        Returns
        -------
        Tensor
            (N, K, D) enriched summaries if query_node is None,
            (1, K, D) if query_node is specified.
        """
        x = node_summaries
        for block in self.blocks:
            x = block(x, graph.edge_index, graph.edge_type)

        if query_node is not None:
            return x[query_node].unsqueeze(0)
        return x


# ---------------------------------------------------------------------------
# Convenience: batch graph construction from training data
# ---------------------------------------------------------------------------

def build_training_graph(
    obs_patient: list[str],
    obs_stage: list[int],
    num_stages: int = 5,
) -> tuple[SetGraph, dict[tuple[str, int], int]]:
    """Build a SetGraph from training observations.

    Groups cells by (patient, stage) and constructs the inter-set graph.

    Parameters
    ----------
    obs_patient : list[str]
        Patient ID per cell.
    obs_stage : list[int]
        Stage index per cell.
    num_stages : int

    Returns
    -------
    (graph, node_map)
        graph: SetGraph
        node_map: dict mapping (patient_id, stage_idx) → node index
    """
    # Find unique (patient, stage) pairs
    pairs: list[tuple[str, int]] = sorted(set(zip(obs_patient, obs_stage)))
    node_map = {pair: idx for idx, pair in enumerate(pairs)}

    patient_ids = [p for p, _ in pairs]
    stage_indices = [s for _, s in pairs]

    graph = build_set_graph(
        patient_ids=patient_ids,
        stage_indices=stage_indices,
        num_stages=num_stages,
    )
    return graph, node_map


def build_cross_dataset_graph(
    obs_patient: list[str],
    obs_stage: list[int],
    stage_adjacency: list[tuple[int, int]],
    num_stages: int = 7,
    bridge_stage_pairs: list[tuple[int, int]] | None = None,
) -> tuple[SetGraph, dict[tuple[str, int], int]]:
    """Build a SetGraph spanning multiple datasets with branching topology.

    This enables cross-dataset OT flow: cells from dataset A (early progression)
    and dataset B (metastasis) are connected via bridge edges where their
    stage ontologies overlap (e.g., LUAD in dataset A = PRIMARY in dataset B).

    Parameters
    ----------
    obs_patient : list[str]
        Patient ID per cell (from both datasets, prefixed to avoid collision).
    obs_stage : list[int]
        Stage index per cell (in the extended ontology).
    stage_adjacency : list[(int, int)]
        Directed stage edges, e.g. from EXTENDED_STAGE_EDGES mapped to indices.
    num_stages : int
        Total stages in extended ontology.
    bridge_stage_pairs : list[(int, int)] or None
        Additional cross-dataset bridge edges.  E.g., if LUAD (idx=4) from
        dataset A should connect to BrainMet (idx=5) nodes from dataset B
        across patients, specify [(4, 5)].  These create type-0 (adjacency)
        edges even between different patients, enabling cross-dataset flow.

    Returns
    -------
    (graph, node_map)
    """
    pairs: list[tuple[str, int]] = sorted(set(zip(obs_patient, obs_stage)))
    node_map = {pair: idx for idx, pair in enumerate(pairs)}

    patient_ids = [p for p, _ in pairs]
    stage_indices = [s for _, s in pairs]

    # Start with the standard graph using the explicit topology
    graph = build_set_graph(
        patient_ids=patient_ids,
        stage_indices=stage_indices,
        num_stages=num_stages,
        stage_adjacency=stage_adjacency,
    )

    # Add cross-dataset bridge edges: connect nodes from different patients
    # when they sit at bridging stages
    if bridge_stage_pairs:
        stage_to_nodes: dict[int, list[int]] = {}
        for i, s in enumerate(stage_indices):
            stage_to_nodes.setdefault(s, []).append(i)

        extra_src: list[int] = []
        extra_tgt: list[int] = []
        extra_type: list[int] = []

        for s_from, s_to in bridge_stage_pairs:
            from_nodes = stage_to_nodes.get(s_from, [])
            to_nodes = stage_to_nodes.get(s_to, [])
            for i in from_nodes:
                for j in to_nodes:
                    if patient_ids[i] != patient_ids[j]:
                        extra_src.append(i)
                        extra_tgt.append(j)
                        extra_type.append(0)  # stage-adjacent type

        if extra_src:
            bridge_edges = torch.tensor([extra_src, extra_tgt], dtype=torch.long)
            bridge_types = torch.tensor(extra_type, dtype=torch.long)
            graph = SetGraph(
                edge_index=torch.cat([graph.edge_index, bridge_edges], dim=1),
                edge_type=torch.cat([graph.edge_type, bridge_types]),
                num_nodes=graph.num_nodes,
                node_patient=graph.node_patient,
                node_stage=graph.node_stage,
            )

    return graph, node_map
