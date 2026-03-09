"""Dataset and collation utilities for lesion-level EA-MIST training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from stagebridge.utils.types import LesionBag, LesionBagBatch, LocalNicheExample


class LesionBagDataset(Dataset[LesionBag]):
    """Torch dataset wrapper over lesion-level bags."""

    def __init__(self, bags: list[LesionBag]) -> None:
        if not bags:
            raise ValueError("LesionBagDataset requires at least one bag.")
        self.bags = list(bags)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int) -> LesionBag:
        return self.bags[int(index)]


@dataclass(slots=True)
class NeighborhoodPretrainExample:
    """Flat neighborhood example used for local SSL pretraining."""

    lesion_id: str
    donor_id: str
    stage: str
    receiver_state_id: int
    flat_features: np.ndarray
    receiver_embedding: np.ndarray
    ring_compositions: np.ndarray
    lr_pathway_summary: np.ndarray
    neighborhood_stats: np.ndarray


class NeighborhoodPretrainDataset(Dataset[NeighborhoodPretrainExample]):
    """Flatten lesion bags into local neighborhood examples for SSL pretraining."""

    def __init__(self, bags: Iterable[LesionBag]) -> None:
        examples: list[NeighborhoodPretrainExample] = []
        for bag in bags:
            for neighborhood in bag.neighborhoods:
                examples.append(
                    NeighborhoodPretrainExample(
                        lesion_id=bag.lesion_id,
                        donor_id=bag.donor_id,
                        stage=bag.stage,
                        receiver_state_id=int(neighborhood.receiver_state_id),
                        flat_features=np.asarray(neighborhood.flat_features, dtype=np.float32),
                        receiver_embedding=np.asarray(neighborhood.receiver_embedding, dtype=np.float32),
                        ring_compositions=np.asarray(neighborhood.ring_compositions, dtype=np.float32),
                        lr_pathway_summary=np.asarray(neighborhood.lr_pathway_summary, dtype=np.float32),
                        neighborhood_stats=np.asarray(neighborhood.neighborhood_stats, dtype=np.float32),
                    )
                )
        if not examples:
            raise ValueError("NeighborhoodPretrainDataset requires at least one neighborhood.")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> NeighborhoodPretrainExample:
        return self.examples[int(index)]


def _validate_bag_shapes(bags: list[LesionBag]) -> tuple[int, int, int, int]:
    """Validate structured neighborhood dimensions and return the canonical sizes."""
    first = bags[0].neighborhoods[0]
    receiver_dim = int(np.asarray(first.receiver_embedding, dtype=np.float32).shape[0])
    num_rings, num_sender_features = np.asarray(first.ring_compositions, dtype=np.float32).shape
    lr_dim = int(np.asarray(first.lr_pathway_summary, dtype=np.float32).shape[0])
    stats_dim = int(np.asarray(first.neighborhood_stats, dtype=np.float32).shape[0])
    for bag in bags:
        if not bag.neighborhoods:
            raise ValueError(f"Lesion bag {bag.lesion_id} is empty.")
        for neighborhood in bag.neighborhoods:
            if np.asarray(neighborhood.receiver_embedding, dtype=np.float32).shape[0] != receiver_dim:
                raise ValueError("All neighborhoods must share the same receiver embedding dimension.")
            if tuple(np.asarray(neighborhood.ring_compositions, dtype=np.float32).shape) != (num_rings, num_sender_features):
                raise ValueError("All neighborhoods must share the same ring composition shape.")
            if np.asarray(neighborhood.lr_pathway_summary, dtype=np.float32).shape[0] != lr_dim:
                raise ValueError("All neighborhoods must share the same LR/pathway summary dimension.")
            if np.asarray(neighborhood.neighborhood_stats, dtype=np.float32).shape[0] != stats_dim:
                raise ValueError("All neighborhoods must share the same neighborhood stats dimension.")
    return receiver_dim, num_rings, num_sender_features, lr_dim


def collate_lesion_bags(bags: list[LesionBag]) -> LesionBagBatch:
    """Pad lesion bags into one EA-MIST batch."""
    if not bags:
        raise ValueError("Cannot collate an empty list of lesion bags.")
    receiver_dim, num_rings, num_sender_features, lr_dim = _validate_bag_shapes(bags)
    stats_dim = int(np.asarray(bags[0].neighborhoods[0].neighborhood_stats, dtype=np.float32).shape[0])
    flat_dim = int(np.asarray(bags[0].neighborhoods[0].flat_features, dtype=np.float32).shape[0])
    max_neighborhoods = max(bag.num_neighborhoods for bag in bags)

    receiver_embeddings = torch.zeros((len(bags), max_neighborhoods, receiver_dim), dtype=torch.float32)
    receiver_state_ids = torch.zeros((len(bags), max_neighborhoods), dtype=torch.long)
    ring_compositions = torch.zeros((len(bags), max_neighborhoods, num_rings, num_sender_features), dtype=torch.float32)
    lr_pathway_summary = torch.zeros((len(bags), max_neighborhoods, lr_dim), dtype=torch.float32)
    neighborhood_stats = torch.zeros((len(bags), max_neighborhoods, stats_dim), dtype=torch.float32)
    flat_features = torch.zeros((len(bags), max_neighborhoods, flat_dim), dtype=torch.float32)
    center_coords = torch.zeros((len(bags), max_neighborhoods, 2), dtype=torch.float32)
    mask = torch.zeros((len(bags), max_neighborhoods), dtype=torch.bool)

    evo_dim = None
    if any(bag.evolution_features is not None for bag in bags):
        evo_dim = max(int(np.asarray(bag.evolution_features, dtype=np.float32).shape[0]) for bag in bags if bag.evolution_features is not None)
        evolution = torch.zeros((len(bags), evo_dim), dtype=torch.float32)
    else:
        evolution = None

    for bag_idx, bag in enumerate(bags):
        for niche_idx, neighborhood in enumerate(bag.neighborhoods):
            receiver_embeddings[bag_idx, niche_idx] = torch.as_tensor(neighborhood.receiver_embedding, dtype=torch.float32)
            receiver_state_ids[bag_idx, niche_idx] = int(neighborhood.receiver_state_id)
            ring_compositions[bag_idx, niche_idx] = torch.as_tensor(neighborhood.ring_compositions, dtype=torch.float32)
            lr_pathway_summary[bag_idx, niche_idx] = torch.as_tensor(neighborhood.lr_pathway_summary, dtype=torch.float32)
            neighborhood_stats[bag_idx, niche_idx] = torch.as_tensor(neighborhood.neighborhood_stats, dtype=torch.float32)
            flat_features[bag_idx, niche_idx] = torch.as_tensor(neighborhood.flat_features, dtype=torch.float32)
            center_coords[bag_idx, niche_idx] = torch.as_tensor(neighborhood.center_coord, dtype=torch.float32)
            mask[bag_idx, niche_idx] = True
        if evolution is not None and bag.evolution_features is not None:
            evo = np.asarray(bag.evolution_features, dtype=np.float32)
            evolution[bag_idx, : evo.shape[0]] = torch.as_tensor(evo, dtype=torch.float32)

    return LesionBagBatch(
        receiver_embeddings=receiver_embeddings,
        receiver_state_ids=receiver_state_ids,
        ring_compositions=ring_compositions,
        lr_pathway_summary=lr_pathway_summary,
        neighborhood_stats=neighborhood_stats,
        flat_features=flat_features,
        center_coords=center_coords,
        neighborhood_mask=mask,
        edge_ids=torch.as_tensor([bag.edge_id for bag in bags], dtype=torch.long),
        labels=torch.as_tensor([bag.label for bag in bags], dtype=torch.float32),
        label_weights=torch.as_tensor([bag.label_weight for bag in bags], dtype=torch.float32),
        sample_ids=[bag.sample_id for bag in bags],
        lesion_ids=[bag.lesion_id for bag in bags],
        donor_ids=[bag.donor_id for bag in bags],
        patient_ids=[bag.patient_id for bag in bags],
        stages=[bag.stage for bag in bags],
        label_sources=[bag.label_source for bag in bags],
        evolution_features=evolution,
    )


def collate_pretrain_neighborhoods(examples: list[NeighborhoodPretrainExample]) -> dict[str, torch.Tensor | list[str]]:
    """Collate local neighborhood examples for SSL pretraining."""
    if not examples:
        raise ValueError("Cannot collate an empty list of neighborhood examples.")
    receiver_dim = int(examples[0].receiver_embedding.shape[0])
    ring_shape = tuple(examples[0].ring_compositions.shape)
    lr_dim = int(examples[0].lr_pathway_summary.shape[0])
    stats_dim = int(examples[0].neighborhood_stats.shape[0])
    flat_dim = int(examples[0].flat_features.shape[0])

    receiver_embeddings = torch.zeros((len(examples), receiver_dim), dtype=torch.float32)
    ring_compositions = torch.zeros((len(examples), *ring_shape), dtype=torch.float32)
    lr_pathway_summary = torch.zeros((len(examples), lr_dim), dtype=torch.float32)
    neighborhood_stats = torch.zeros((len(examples), stats_dim), dtype=torch.float32)
    flat_features = torch.zeros((len(examples), flat_dim), dtype=torch.float32)
    receiver_state_ids = torch.zeros((len(examples),), dtype=torch.long)
    stage_labels: list[str] = []
    lesion_ids: list[str] = []
    donor_ids: list[str] = []

    for idx, example in enumerate(examples):
        receiver_embeddings[idx] = torch.as_tensor(example.receiver_embedding, dtype=torch.float32)
        ring_compositions[idx] = torch.as_tensor(example.ring_compositions, dtype=torch.float32)
        lr_pathway_summary[idx] = torch.as_tensor(example.lr_pathway_summary, dtype=torch.float32)
        neighborhood_stats[idx] = torch.as_tensor(example.neighborhood_stats, dtype=torch.float32)
        flat_features[idx] = torch.as_tensor(example.flat_features, dtype=torch.float32)
        receiver_state_ids[idx] = int(example.receiver_state_id)
        stage_labels.append(str(example.stage))
        lesion_ids.append(str(example.lesion_id))
        donor_ids.append(str(example.donor_id))

    return {
        "receiver_embeddings": receiver_embeddings,
        "ring_compositions": ring_compositions,
        "lr_pathway_summary": lr_pathway_summary,
        "neighborhood_stats": neighborhood_stats,
        "flat_features": flat_features,
        "receiver_state_ids": receiver_state_ids,
        "stage_labels": stage_labels,
        "lesion_ids": lesion_ids,
        "donor_ids": donor_ids,
    }
