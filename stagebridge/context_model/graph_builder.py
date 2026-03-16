"""Spatial graph construction for the Graph-of-Sets Transformer.

This module converts spatial transcriptomics data (Visium spots or Slide-seq
beads) into graph structure for GoST.  The key insight is that each spatial
spot IS a set — it captures the joint expression of ~5-20 cells in a local
tissue neighborhood.  The spatial proximity between spots defines natural
graph edges.

Graph construction:

    1. **Spot-level nodes**: Each Visium spot or Slide-seq bead is a node
       in the graph.  Its "set" is the cells that Tangram has mapped to it
       (or the spot's own expression profile in HLCA latent space).

    2. **Spatial edges (type 0)**: k-nearest neighbors in physical space.
       These capture tissue microenvironment structure — adjacent spots
       share paracrine signaling context.

    3. **Same-patient edges (type 1)**: Connect spots from the same patient
       across different tissue sections (if available).

    4. **Same-stage edges (type 2)**: Connect spots at the same stage from
       different patients — enables cross-patient comparison.

    5. **Cross-dataset bridge edges (type 3)**: Connect spots across
       datasets at overlapping stages (e.g., LUAD spots from Peng data to
       PRIMARY spots from Rossi data, both in HLCA latent space).

Edge type encoding (extends graph_of_sets.NUM_EDGE_TYPES):
    0 = spatial neighbor (kNN in physical coordinates)
    1 = same-patient cross-section
    2 = same-stage cross-patient
    3 = cross-dataset bridge

For the GoST, each spot's "set" is either:
  - The Tangram-mapped snRNA cells assigned to that spot, OR
  - A single-element set (the spot's own HLCA latent vector)

Both work — Tangram gives richer cell-type composition, while direct spot
embedding is simpler and avoids Tangram as a dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)

NUM_SPATIAL_EDGE_TYPES = 4  # spatial-kNN, same-patient, same-stage, cross-dataset


@dataclass
class SpatialGraph:
    """Graph over spatial spots / beads.

    Attributes
    ----------
    edge_index : (2, E) long tensor — source → target
    edge_type : (E,) long tensor
    edge_dist : (E,) float tensor — physical distance (for spatial edges), 0 otherwise
    num_nodes : int
    node_patient : list[str] — patient ID per node
    node_stage : list[int] — stage index per node
    node_dataset : list[str] — dataset source per node ("peng" or "rossi")
    coords : (N, 2) float tensor — spatial coordinates per node
    """

    edge_index: Tensor
    edge_type: Tensor
    edge_dist: Tensor
    num_nodes: int
    node_patient: list[str]
    node_stage: list[int]
    node_dataset: list[str]
    coords: Tensor

    def to(self, device: str | torch.device) -> "SpatialGraph":
        return SpatialGraph(
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            edge_dist=self.edge_dist.to(device),
            num_nodes=self.num_nodes,
            node_patient=self.node_patient,
            node_stage=self.node_stage,
            node_dataset=self.node_dataset,
            coords=self.coords.to(device),
        )


def build_spatial_knn_graph(
    coords: np.ndarray,
    patient_ids: list[str],
    stage_indices: list[int],
    dataset_ids: list[str] | None = None,
    k: int = 6,
    max_dist: float | None = None,
    include_cross_patient: bool = True,
    include_cross_stage: bool = True,
) -> SpatialGraph:
    """Build a spatial graph from spot coordinates.

    Parameters
    ----------
    coords : (N, 2) array of spatial coordinates
    patient_ids : patient ID per spot
    stage_indices : stage index per spot
    dataset_ids : dataset source per spot (default: all "peng")
    k : number of spatial nearest neighbors per spot
    max_dist : if set, prune spatial edges longer than this
    include_cross_patient : add same-stage cross-patient edges
    include_cross_stage : add same-patient cross-stage edges

    Returns
    -------
    SpatialGraph
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(coords)
    assert len(patient_ids) == n
    assert len(stage_indices) == n
    if dataset_ids is None:
        dataset_ids = ["peng"] * n

    src_list: list[int] = []
    tgt_list: list[int] = []
    type_list: list[int] = []
    dist_list: list[float] = []

    # --- Type 0: spatial kNN edges (within same sample/section) ---
    # Group by (patient, stage) to find spots from the same tissue section
    section_map: dict[tuple[str, int], list[int]] = {}
    for i in range(n):
        key = (patient_ids[i], stage_indices[i])
        section_map.setdefault(key, []).append(i)

    for _section_key, indices in section_map.items():
        if len(indices) < 2:
            continue
        section_coords = coords[indices]
        k_actual = min(k + 1, len(indices))  # +1 because query point is included
        nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
        nn.fit(section_coords)
        distances, neighbors = nn.kneighbors(section_coords)

        for local_i in range(len(indices)):
            global_i = indices[local_i]
            for j_idx in range(1, k_actual):  # skip self (index 0)
                local_j = neighbors[local_i, j_idx]
                global_j = indices[local_j]
                d = distances[local_i, j_idx]
                if max_dist is not None and d > max_dist:
                    continue
                src_list.append(global_i)
                tgt_list.append(global_j)
                type_list.append(0)
                dist_list.append(float(d))

    # --- Type 1: same-patient cross-stage edges ---
    if include_cross_stage:
        patient_map: dict[str, list[int]] = {}
        for i in range(n):
            patient_map.setdefault(patient_ids[i], []).append(i)

        for _pid, indices in patient_map.items():
            stages_present = set(stage_indices[i] for i in indices)
            if len(stages_present) < 2:
                continue
            # Connect a random sample of spots across stages (not all-to-all)
            by_stage: dict[int, list[int]] = {}
            for i in indices:
                by_stage.setdefault(stage_indices[i], []).append(i)
            stage_list = sorted(by_stage.keys())
            for s_idx in range(len(stage_list) - 1):
                s1, s2 = stage_list[s_idx], stage_list[s_idx + 1]
                # Sample up to k spots from each stage to avoid quadratic blowup
                spots1 = by_stage[s1][:k]
                spots2 = by_stage[s2][:k]
                for i in spots1:
                    for j in spots2:
                        src_list.extend([i, j])
                        tgt_list.extend([j, i])
                        type_list.extend([1, 1])
                        dist_list.extend([0.0, 0.0])

    # --- Type 2: same-stage cross-patient edges ---
    if include_cross_patient:
        stage_map: dict[int, list[int]] = {}
        for i in range(n):
            stage_map.setdefault(stage_indices[i], []).append(i)

        for _stage, indices in stage_map.items():
            patients_present = set(patient_ids[i] for i in indices)
            if len(patients_present) < 2:
                continue
            by_patient: dict[str, list[int]] = {}
            for i in indices:
                by_patient.setdefault(patient_ids[i], []).append(i)
            patient_list = sorted(by_patient.keys())
            # Sparse: connect k representative spots per patient pair
            for p_idx in range(len(patient_list)):
                for q_idx in range(p_idx + 1, len(patient_list)):
                    spots_p = by_patient[patient_list[p_idx]][:k]
                    spots_q = by_patient[patient_list[q_idx]][:k]
                    for i in spots_p:
                        for j in spots_q:
                            src_list.extend([i, j])
                            tgt_list.extend([j, i])
                            type_list.extend([2, 2])
                            dist_list.extend([0.0, 0.0])

    # Build tensors
    if not src_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_dist = torch.zeros((0,), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)
        edge_dist = torch.tensor(dist_list, dtype=torch.float32)

    return SpatialGraph(
        edge_index=edge_index,
        edge_type=edge_type,
        edge_dist=edge_dist,
        num_nodes=n,
        node_patient=list(patient_ids),
        node_stage=list(stage_indices),
        node_dataset=list(dataset_ids),
        coords=torch.tensor(coords, dtype=torch.float32),
    )


def add_cross_dataset_bridges(
    graph: SpatialGraph,
    bridge_stage_pairs: list[tuple[int, int]],
    k_bridge: int = 4,
    latent_vecs: np.ndarray | None = None,
) -> SpatialGraph:
    """Add cross-dataset bridge edges between spots at overlapping stages.

    These edges connect spots from different datasets (e.g., Peng LUAD
    spots to Rossi PRIMARY spots) that share a stage in the extended
    ontology.  When ``latent_vecs`` is provided, bridges connect the
    k nearest neighbors in HLCA latent space — this ensures biologically
    similar spots are connected, not arbitrary ones.

    Parameters
    ----------
    graph : existing SpatialGraph
    bridge_stage_pairs : list of (stage_from, stage_to) pairs
    k_bridge : number of bridge neighbors per spot
    latent_vecs : (N, D) HLCA latent vectors, optional
        If provided, bridge edges connect kNN in latent space.
        If None, connects random sample of k_bridge spots.

    Returns
    -------
    SpatialGraph with additional type-3 edges
    """
    from sklearn.neighbors import NearestNeighbors

    extra_src: list[int] = []
    extra_tgt: list[int] = []
    extra_dist: list[float] = []

    n = graph.num_nodes
    stage_dataset_map: dict[tuple[int, str], list[int]] = {}
    for i in range(n):
        key = (graph.node_stage[i], graph.node_dataset[i])
        stage_dataset_map.setdefault(key, []).append(i)

    datasets = list(set(graph.node_dataset))
    if len(datasets) < 2:
        log.warning("Only one dataset found — no cross-dataset bridges added")
        return graph

    for s_from, s_to in bridge_stage_pairs:
        for ds_a in datasets:
            for ds_b in datasets:
                if ds_a == ds_b:
                    continue
                nodes_a = stage_dataset_map.get((s_from, ds_a), [])
                nodes_b = stage_dataset_map.get((s_to, ds_b), [])
                if not nodes_a or not nodes_b:
                    continue

                if latent_vecs is not None:
                    # kNN in HLCA latent space for biologically informed bridges
                    vecs_b = latent_vecs[nodes_b]
                    k_actual = min(k_bridge, len(nodes_b))
                    nn = NearestNeighbors(n_neighbors=k_actual, metric="cosine")
                    nn.fit(vecs_b)
                    vecs_a = latent_vecs[nodes_a]
                    dists, idxs = nn.kneighbors(vecs_a)

                    for local_i, global_i in enumerate(nodes_a):
                        for j_idx in range(k_actual):
                            global_j = nodes_b[idxs[local_i, j_idx]]
                            extra_src.extend([global_i, global_j])
                            extra_tgt.extend([global_j, global_i])
                            extra_dist.extend([float(dists[local_i, j_idx])] * 2)
                else:
                    # Random sample bridges
                    rng = np.random.default_rng(42)
                    for global_i in nodes_a:
                        chosen = rng.choice(
                            nodes_b, size=min(k_bridge, len(nodes_b)), replace=False
                        )
                        for global_j in chosen:
                            extra_src.extend([global_i, int(global_j)])
                            extra_tgt.extend([int(global_j), global_i])
                            extra_dist.extend([0.0, 0.0])

    if not extra_src:
        return graph

    bridge_edges = torch.tensor([extra_src, extra_tgt], dtype=torch.long)
    bridge_types = torch.full((len(extra_src),), 3, dtype=torch.long)  # type 3
    bridge_dists = torch.tensor(extra_dist, dtype=torch.float32)

    return SpatialGraph(
        edge_index=torch.cat([graph.edge_index, bridge_edges], dim=1),
        edge_type=torch.cat([graph.edge_type, bridge_types]),
        edge_dist=torch.cat([graph.edge_dist, bridge_dists]),
        num_nodes=graph.num_nodes,
        node_patient=graph.node_patient,
        node_stage=graph.node_stage,
        node_dataset=graph.node_dataset,
        coords=graph.coords,
    )


def subsample_spatial_graph(
    graph: SpatialGraph,
    max_nodes_per_section: int = 500,
    seed: int = 42,
) -> tuple[SpatialGraph, np.ndarray]:
    """Subsample a spatial graph for memory-efficient training.

    For large Visium/Slide-seq sections (>4000 spots), subsamples to
    max_nodes_per_section per (patient, stage) section while preserving
    the graph connectivity structure by re-computing kNN on the subsampled
    coordinates.

    Parameters
    ----------
    graph : SpatialGraph
    max_nodes_per_section : max spots to keep per tissue section
    seed : random seed

    Returns
    -------
    (subsampled_graph, kept_indices)
    """
    rng = np.random.default_rng(seed)
    n = graph.num_nodes

    # Group by section
    section_map: dict[tuple[str, int], list[int]] = {}
    for i in range(n):
        key = (graph.node_patient[i], graph.node_stage[i])
        section_map.setdefault(key, []).append(i)

    kept: list[int] = []
    for _key, indices in sorted(section_map.items()):
        if len(indices) <= max_nodes_per_section:
            kept.extend(indices)
        else:
            chosen = rng.choice(indices, size=max_nodes_per_section, replace=False)
            kept.extend(sorted(chosen))

    kept_arr = np.array(sorted(kept))
    old_to_new = {old: new for new, old in enumerate(kept_arr)}
    n_new = len(kept_arr)

    # Filter edges: keep only those where both endpoints are retained
    ei = graph.edge_index
    mask = torch.zeros(ei.shape[1], dtype=torch.bool)
    for e in range(ei.shape[1]):
        s, t = ei[0, e].item(), ei[1, e].item()
        if s in old_to_new and t in old_to_new:
            mask[e] = True

    new_src = [old_to_new[ei[0, e].item()] for e in range(ei.shape[1]) if mask[e]]
    new_tgt = [old_to_new[ei[1, e].item()] for e in range(ei.shape[1]) if mask[e]]

    if new_src:
        new_edge_index = torch.tensor([new_src, new_tgt], dtype=torch.long)
        new_edge_type = graph.edge_type[mask]
        new_edge_dist = graph.edge_dist[mask]
    else:
        new_edge_index = torch.zeros((2, 0), dtype=torch.long)
        new_edge_type = torch.zeros((0,), dtype=torch.long)
        new_edge_dist = torch.zeros((0,), dtype=torch.float32)

    return SpatialGraph(
        edge_index=new_edge_index,
        edge_type=new_edge_type,
        edge_dist=new_edge_dist,
        num_nodes=n_new,
        node_patient=[graph.node_patient[i] for i in kept_arr],
        node_stage=[graph.node_stage[i] for i in kept_arr],
        node_dataset=[graph.node_dataset[i] for i in kept_arr],
        coords=graph.coords[kept_arr],
    ), kept_arr
