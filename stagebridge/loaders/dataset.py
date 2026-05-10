"""Dataset for receiver-centered niche data.

Loads neighborhoods.parquet with individual cells per ring. The model uses
learned ISAB+PMA pooling to determine which cells in each ring matter most.

This is the correct v1 architecture - NOT pre-pooled means.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from stagebridge.contracts import (
    STAGE_TO_IDX,
    LATENT_DIM,
    MAX_CELLS_PER_RING,
)


# =============================================================================
# NicheBatch: The standard batch format for StageBridge
# =============================================================================


@dataclass(slots=True)
class NicheBatch:
    """Batch for learned ring pooling with individual cells per ring.

    This is the correct v1 format: raw cells per ring, which NicheTokenizer
    pools using learned ISAB+PMA attention.

    Attributes:
        receiver: [B, 40] receiver cell embeddings (HLCA+LuCA concat)
        ring_cells: List of 4 tensors, each [B, max_cells, 40] (fused embeddings)
        ring_masks: List of 4 tensors, each [B, max_cells] (True = valid)
        hlca: [B, 30] HLCA reference embedding
        luca: [B, 10] LuCA reference embedding
        pathway: [B, 40] pathway features (optional, zeros if None)
        stats: [B, 7] stats features (optional, zeros if None)
        stage_idx: [B] stage indices
        donor_ids: List of donor IDs
        cell_ids: List of cell IDs
        pathway_targets: [B, 14] PROGENy pathway activity targets (optional)
        proliferation_target: [B] Ki67/proliferation label (optional)
        evolution_features: [B, E] WES/genomic features (optional)
    """

    receiver: Tensor
    ring_cells: list[Tensor]
    ring_masks: list[Tensor]
    hlca: Tensor
    luca: Tensor
    pathway: Tensor | None
    stats: Tensor | None
    stage_idx: Tensor
    donor_ids: list[str]
    cell_ids: list[str]
    pathway_targets: Tensor | None = None
    proliferation_target: Tensor | None = None
    evolution_features: Tensor | None = None

    def to(self, device: str | torch.device) -> "NicheBatch":
        """Move tensors to device."""
        return NicheBatch(
            receiver=self.receiver.to(device),
            ring_cells=[rc.to(device) for rc in self.ring_cells],
            ring_masks=[rm.to(device) for rm in self.ring_masks],
            hlca=self.hlca.to(device),
            luca=self.luca.to(device),
            pathway=self.pathway.to(device) if self.pathway is not None else None,
            stats=self.stats.to(device) if self.stats is not None else None,
            stage_idx=self.stage_idx.to(device),
            donor_ids=self.donor_ids,
            cell_ids=self.cell_ids,
            pathway_targets=self.pathway_targets.to(device) if self.pathway_targets is not None else None,
            proliferation_target=self.proliferation_target.to(device) if self.proliferation_target is not None else None,
            evolution_features=self.evolution_features.to(device) if self.evolution_features is not None else None,
        )

    def __len__(self) -> int:
        return self.receiver.shape[0]


@dataclass(slots=True)
class AMICIBatch:
    """Batch for AMICI-style continuous attention with raw distances.

    No ring binning - neighbors and distances are flat lists sorted by distance.
    ReceiverCenteredNicheEncoder uses learned distance decay per attention head.

    Attributes:
        receiver: [B, 40] receiver cell embeddings (HLCA+LuCA concat)
        neighbors: [B, max_neighbors, 40] neighbor embeddings (sorted by distance)
        distances: [B, max_neighbors] distances in microns
        neighbor_mask: [B, max_neighbors] True = valid neighbor
        hlca: [B, 30] HLCA reference embedding
        luca: [B, 10] LuCA reference embedding
        pathway: [B, 40] pathway features (optional)
        stats: [B, 7] stats features (optional)
        stage_idx: [B] stage indices
        donor_ids: List of donor IDs
        cell_ids: List of cell IDs
        pathway_targets: [B, 14] PROGENy pathway activity targets (optional)
        proliferation_target: [B] proliferation label (optional)
        evolution_features: [B, E] WES/genomic features (optional)
    """

    receiver: Tensor
    neighbors: Tensor
    distances: Tensor
    neighbor_mask: Tensor
    hlca: Tensor
    luca: Tensor
    pathway: Tensor | None
    stats: Tensor | None
    stage_idx: Tensor
    donor_ids: list[str]
    cell_ids: list[str]
    pathway_targets: Tensor | None = None
    proliferation_target: Tensor | None = None
    evolution_features: Tensor | None = None

    def to(self, device: str | torch.device) -> "AMICIBatch":
        """Move tensors to device."""
        return AMICIBatch(
            receiver=self.receiver.to(device),
            neighbors=self.neighbors.to(device),
            distances=self.distances.to(device),
            neighbor_mask=self.neighbor_mask.to(device),
            hlca=self.hlca.to(device),
            luca=self.luca.to(device),
            pathway=self.pathway.to(device) if self.pathway is not None else None,
            stats=self.stats.to(device) if self.stats is not None else None,
            stage_idx=self.stage_idx.to(device),
            donor_ids=self.donor_ids,
            cell_ids=self.cell_ids,
            pathway_targets=self.pathway_targets.to(device) if self.pathway_targets is not None else None,
            proliferation_target=self.proliferation_target.to(device) if self.proliferation_target is not None else None,
            evolution_features=self.evolution_features.to(device) if self.evolution_features is not None else None,
        )

    def __len__(self) -> int:
        return self.receiver.shape[0]


# =============================================================================
# StageBridgeDataset: Loads neighborhoods.parquet with individual cells per ring
# =============================================================================


class StageBridgeDataset(Dataset):
    """Dataset for receiver-centered niche modeling.

    Supports two formats:
    - Legacy ring format: ring_N_cells columns (backward compatible)
    - AMICI format: neighbor_cells + neighbor_distances (continuous attention)

    Args:
        data_dir: Directory containing neighborhoods.parquet
        donors: List of donor IDs to include (for train/val/test splits)
        latent_dim: Embedding dimension
        max_cells_per_ring: Maximum cells per ring (legacy format)
        max_neighbors: Maximum neighbors (AMICI format)
        stages: Optional list of stages to include
        shuffle_rings: If True, randomly shuffle cells across rings (ablation, legacy only)
    """

    NUM_RINGS = 4

    def __init__(
        self,
        data_dir: str | Path,
        donors: Sequence[str] | None = None,
        latent_dim: int = LATENT_DIM,
        max_cells_per_ring: int = MAX_CELLS_PER_RING,
        max_neighbors: int = 100,
        stages: Sequence[str] | None = None,
        shuffle_rings: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.latent_dim = latent_dim
        self.max_cells_per_ring = max_cells_per_ring
        self.max_neighbors = max_neighbors
        self.shuffle_rings = shuffle_rings

        neighborhoods_path = self.data_dir / "neighborhoods.parquet"
        if not neighborhoods_path.exists():
            raise FileNotFoundError(f"neighborhoods.parquet not found: {neighborhoods_path}")

        # Use pyarrow directly to handle large list columns (avoids int32 overflow)
        import pyarrow.parquet as pq
        table = pq.read_table(neighborhoods_path)
        self.neighborhoods = table.to_pandas()

        # Detect format
        self.amici_format = "neighbor_cells" in self.neighborhoods.columns
        self.ring_format = "ring_1_cells" in self.neighborhoods.columns
        self.tokenized_format = "tokens" in self.neighborhoods.columns

        if not self.amici_format and not self.ring_format:
            if self.tokenized_format:
                raise ValueError(
                    "neighborhoods.parquet has tokenized format (pre-pooled z_pooled), "
                    "which is deprecated. Run data prep with use_continuous_attention=True"
                )
            raise ValueError(
                "neighborhoods.parquet must have either neighbor_cells (AMICI format) "
                "or ring_N_cells (legacy ring format)"
            )

        if donors is not None:
            self.neighborhoods = self.neighborhoods[
                self.neighborhoods["donor_id"].isin(set(donors))
            ]

        if stages is not None:
            self.neighborhoods = self.neighborhoods[
                self.neighborhoods["stage"].isin(set(stages))
            ]

        self.neighborhoods = self.neighborhoods.reset_index(drop=True)

        # Detect feature dimensions from data
        self._pathway_dim: int | None = None
        self._stats_dim: int | None = None
        self._evolution_dim: int | None = None
        self._detect_feature_dims()

    def _detect_feature_dims(self) -> None:
        """Detect feature dimensions from first valid row."""
        if len(self.neighborhoods) == 0:
            return

        # Sample first row with data
        row = self.neighborhoods.iloc[0]

        if "pathway_z" in row and row["pathway_z"] is not None:
            self._pathway_dim = len(row["pathway_z"])

        if "stats_z" in row and row["stats_z"] is not None:
            self._stats_dim = len(row["stats_z"])

        if "evolution_features" in row and row["evolution_features"] is not None:
            self._evolution_dim = len(row["evolution_features"])

    @property
    def pathway_dim(self) -> int | None:
        """Pathway feature dimension detected from data."""
        return self._pathway_dim

    @property
    def stats_dim(self) -> int | None:
        """Stats feature dimension detected from data."""
        return self._stats_dim

    @property
    def evolution_dim(self) -> int | None:
        """Evolution feature dimension detected from data."""
        return self._evolution_dim

    def __len__(self) -> int:
        return len(self.neighborhoods)

    def __getitem__(self, idx: int) -> dict:
        """Get a single niche sample."""
        row = self.neighborhoods.iloc[idx]

        cell_id = row["cell_id"]
        donor_id = row["donor_id"]
        stage = row["stage"]
        stage_idx = STAGE_TO_IDX.get(stage, 0)

        if self.amici_format:
            return self._getitem_amici_format(row, cell_id, donor_id, stage_idx)
        elif self.tokenized_format:
            return self._getitem_tokenized(row, cell_id, donor_id, stage_idx)
        else:
            return self._getitem_ring_format(row, cell_id, donor_id, stage_idx)

    def _getitem_amici_format(self, row, cell_id, donor_id, stage_idx) -> dict:
        """Handle AMICI format (neighbor_cells + neighbor_distances)."""
        receiver = np.array(row["receiver_z"], dtype=np.float32)
        hlca = np.array(row["hlca_z"], dtype=np.float32)
        luca = np.array(row["luca_z"], dtype=np.float32)

        # Neighbors and distances (already sorted by distance in prep)
        neighbor_list = row["neighbor_cells"]
        distance_list = row["neighbor_distances"]
        n_neighbors = len(neighbor_list)

        # Pad to max_neighbors
        neighbors = np.zeros((self.max_neighbors, self.latent_dim), dtype=np.float32)
        distances = np.zeros(self.max_neighbors, dtype=np.float32)
        neighbor_mask = np.zeros(self.max_neighbors, dtype=bool)

        n_use = min(n_neighbors, self.max_neighbors)
        for j in range(n_use):
            neighbors[j] = np.array(neighbor_list[j], dtype=np.float32)[:self.latent_dim]
            distances[j] = distance_list[j]
            neighbor_mask[j] = True

        # Optional features
        pathway = None
        if "pathway_z" in row and row["pathway_z"] is not None:
            pathway = np.array(row["pathway_z"], dtype=np.float32)

        stats = None
        if "stats_z" in row and row["stats_z"] is not None:
            stats = np.array(row["stats_z"], dtype=np.float32)

        pathway_targets = None
        if "pathway_targets" in row and row["pathway_targets"] is not None:
            pathway_targets = np.array(row["pathway_targets"], dtype=np.float32)

        proliferation_target = None
        if "proliferation_label" in row:
            proliferation_target = float(row["proliferation_label"])
        elif "Ki67" in row:
            proliferation_target = float(row["Ki67"])

        evolution_features = None
        if "evolution_features" in row and row["evolution_features"] is not None:
            evolution_features = np.array(row["evolution_features"], dtype=np.float32)

        return {
            "receiver": receiver,
            "neighbors": neighbors,
            "distances": distances,
            "neighbor_mask": neighbor_mask,
            "hlca": hlca,
            "luca": luca,
            "pathway": pathway,
            "stats": stats,
            "stage_idx": stage_idx,
            "donor_id": donor_id,
            "cell_id": cell_id,
            "pathway_targets": pathway_targets,
            "proliferation_target": proliferation_target,
            "evolution_features": evolution_features,
            "format": "amici",
        }

    def _getitem_tokenized(self, row, cell_id, donor_id, stage_idx) -> dict:
        """Handle tokenized format (tokens column with 9 token dicts)."""
        tokens = row["tokens"]

        # Token 0 = receiver, tokens 1-4 = rings, token 5 = hlca, token 6 = luca
        # token 7 = pathway, token 8 = stats
        receiver_token = tokens[0]
        receiver = np.array(receiver_token.get("z_fused", np.zeros(self.latent_dim)), dtype=np.float32)

        # HLCA from token 5
        hlca_token = tokens[5] if len(tokens) > 5 else {}
        hlca = np.array(hlca_token.get("z_hlca", np.zeros(30)), dtype=np.float32)

        # LuCA from token 6
        luca_token = tokens[6] if len(tokens) > 6 else {}
        luca = np.array(luca_token.get("z_luca", np.zeros(10)), dtype=np.float32)

        # Ring cells from tokens 1-4 (use z_pooled which is the aggregated embedding)
        ring_cells = []
        ring_masks = []
        for i in range(1, self.NUM_RINGS + 1):
            ring_token = tokens[i] if len(tokens) > i else {}

            # z_pooled is the aggregated ring embedding, treat as single "cell"
            z_pooled = ring_token.get("z_pooled")
            if z_pooled is not None and hasattr(z_pooled, "__len__") and len(z_pooled) > 0:
                padded = np.zeros((self.max_cells_per_ring, self.latent_dim), dtype=np.float32)
                mask = np.zeros(self.max_cells_per_ring, dtype=bool)
                padded[0] = np.array(z_pooled, dtype=np.float32)[:self.latent_dim]
                mask[0] = True
            else:
                padded = np.zeros((self.max_cells_per_ring, self.latent_dim), dtype=np.float32)
                mask = np.zeros(self.max_cells_per_ring, dtype=bool)

            ring_cells.append(padded)
            ring_masks.append(mask)

        # Pathway from token 7 (may be None even if key exists)
        pathway = None
        pathway_token = tokens[7] if len(tokens) > 7 else {}
        pathway_val = pathway_token.get("z_fused")
        if pathway_val is not None and hasattr(pathway_val, "__len__"):
            pathway = np.array(pathway_val, dtype=np.float32)

        # Stats from token 8 (may be None even if key exists)
        stats = None
        stats_token = tokens[8] if len(tokens) > 8 else {}
        stats_val = stats_token.get("z_fused")
        if stats_val is not None and hasattr(stats_val, "__len__"):
            stats = np.array(stats_val, dtype=np.float32)

        # Auxiliary targets (may be in receiver token or row)
        pathway_targets = None
        proliferation_target = None

        # Try to get proliferation from receiver token stats
        if "S_score" in receiver_token and receiver_token["S_score"] is not None:
            s_score = receiver_token.get("S_score", 0) or 0
            g2m_score = receiver_token.get("G2M_score", 0) or 0
            proliferation_target = float(s_score + g2m_score)

        evolution_features = None

        return {
            "receiver": receiver,
            "ring_cells": ring_cells,
            "ring_masks": ring_masks,
            "hlca": hlca,
            "luca": luca,
            "pathway": pathway,
            "stats": stats,
            "stage_idx": stage_idx,
            "donor_id": donor_id,
            "cell_id": cell_id,
            "pathway_targets": pathway_targets,
            "proliferation_target": proliferation_target,
            "evolution_features": evolution_features,
        }

    def _getitem_ring_format(self, row, cell_id, donor_id, stage_idx) -> dict:
        """Handle ring column format (receiver_z, hlca_z, luca_z, ring_N_cells)."""
        receiver = np.array(row["receiver_z"], dtype=np.float32)
        hlca = np.array(row["hlca_z"], dtype=np.float32)
        luca = np.array(row["luca_z"], dtype=np.float32)

        # Collect all cells from all rings first
        all_cells = []
        for i in range(1, self.NUM_RINGS + 1):
            cells_list = row[f"ring_{i}_cells"]
            if cells_list is not None:
                if isinstance(cells_list, np.ndarray) and cells_list.size > 0:
                    all_cells.extend(cells_list)
                elif cells_list:
                    all_cells.extend(cells_list)

        # Shuffle all cells if random_niche ablation
        if self.shuffle_rings and len(all_cells) > 0:
            np.random.shuffle(all_cells)

        # Redistribute cells to rings (either original or shuffled)
        ring_cells = []
        ring_masks = []
        cell_idx = 0

        for i in range(1, self.NUM_RINGS + 1):
            if self.shuffle_rings:
                # Distribute shuffled cells evenly across rings
                cells_per_ring = len(all_cells) // self.NUM_RINGS
                extra = 1 if i <= (len(all_cells) % self.NUM_RINGS) else 0
                n_cells = cells_per_ring + extra
                ring_cell_list = all_cells[cell_idx:cell_idx + n_cells]
                cell_idx += n_cells
            else:
                # Use original ring assignment
                cells_list = row[f"ring_{i}_cells"]
                if cells_list is None:
                    ring_cell_list = []
                elif isinstance(cells_list, np.ndarray):
                    ring_cell_list = list(cells_list) if cells_list.size > 0 else []
                else:
                    ring_cell_list = list(cells_list) if cells_list else []
                n_cells = len(ring_cell_list)

            padded = np.zeros((self.max_cells_per_ring, self.latent_dim), dtype=np.float32)
            mask = np.zeros(self.max_cells_per_ring, dtype=bool)

            if n_cells > 0:
                n_use = min(n_cells, self.max_cells_per_ring)
                for j in range(n_use):
                    padded[j] = np.array(ring_cell_list[j], dtype=np.float32)[:self.latent_dim]
                    mask[j] = True

            ring_cells.append(padded)
            ring_masks.append(mask)

        pathway = None
        if "pathway_z" in row and row["pathway_z"] is not None:
            pathway = np.array(row["pathway_z"], dtype=np.float32)

        stats = None
        if "stats_z" in row and row["stats_z"] is not None:
            stats = np.array(row["stats_z"], dtype=np.float32)

        pathway_targets = None
        if "pathway_targets" in row and row["pathway_targets"] is not None:
            pathway_targets = np.array(row["pathway_targets"], dtype=np.float32)

        proliferation_target = None
        if "proliferation_label" in row:
            proliferation_target = float(row["proliferation_label"])
        elif "Ki67" in row:
            proliferation_target = float(row["Ki67"])

        # Evolution/WES features
        evolution_features = None
        if "evolution_features" in row and row["evolution_features"] is not None:
            evolution_features = np.array(row["evolution_features"], dtype=np.float32)
        elif "wes_features" in row and row["wes_features"] is not None:
            evolution_features = np.array(row["wes_features"], dtype=np.float32)

        return {
            "receiver": receiver,
            "ring_cells": ring_cells,
            "ring_masks": ring_masks,
            "hlca": hlca,
            "luca": luca,
            "pathway": pathway,
            "stats": stats,
            "stage_idx": stage_idx,
            "donor_id": donor_id,
            "cell_id": cell_id,
            "pathway_targets": pathway_targets,
            "proliferation_target": proliferation_target,
            "evolution_features": evolution_features,
        }

    def get_stage_distribution(self) -> dict[str, int]:
        """Get distribution of stages in the dataset."""
        return self.neighborhoods["stage"].value_counts().to_dict()

    def get_donor_distribution(self) -> dict[str, int]:
        """Get distribution of donors in the dataset."""
        return self.neighborhoods["donor_id"].value_counts().to_dict()


def collate_niche_batch(samples: list[dict]) -> NicheBatch | AMICIBatch:
    """Collate function for DataLoader. Auto-detects format from samples."""
    # Check if AMICI format
    if samples[0].get("format") == "amici":
        return collate_amici_batch(samples)

    # Legacy ring format
    receiver = torch.from_numpy(np.stack([s["receiver"] for s in samples]))
    hlca = torch.from_numpy(np.stack([s["hlca"] for s in samples]))
    luca = torch.from_numpy(np.stack([s["luca"] for s in samples]))
    stage_idx = torch.tensor([s["stage_idx"] for s in samples], dtype=torch.long)

    ring_cells = []
    ring_masks = []
    num_rings = len(samples[0]["ring_cells"])
    for i in range(num_rings):
        rc = torch.from_numpy(np.stack([s["ring_cells"][i] for s in samples]))
        rm = torch.from_numpy(np.stack([s["ring_masks"][i] for s in samples]))
        ring_cells.append(rc)
        ring_masks.append(rm)

    pathway = None
    if samples[0].get("pathway") is not None:
        pathway = torch.from_numpy(np.stack([s["pathway"] for s in samples]))

    stats = None
    if samples[0].get("stats") is not None:
        stats = torch.from_numpy(np.stack([s["stats"] for s in samples]))

    # Auxiliary targets
    pathway_targets = None
    if samples[0].get("pathway_targets") is not None:
        pathway_targets = torch.from_numpy(np.stack([s["pathway_targets"] for s in samples]))

    proliferation_target = None
    if samples[0].get("proliferation_target") is not None:
        proliferation_target = torch.tensor(
            [s["proliferation_target"] for s in samples], dtype=torch.float32
        )

    evolution_features = None
    if samples[0].get("evolution_features") is not None:
        evolution_features = torch.from_numpy(np.stack([s["evolution_features"] for s in samples]))

    donor_ids = [s["donor_id"] for s in samples]
    cell_ids = [s["cell_id"] for s in samples]

    return NicheBatch(
        receiver=receiver,
        ring_cells=ring_cells,
        ring_masks=ring_masks,
        hlca=hlca,
        luca=luca,
        pathway=pathway,
        stats=stats,
        stage_idx=stage_idx,
        donor_ids=donor_ids,
        pathway_targets=pathway_targets,
        proliferation_target=proliferation_target,
        evolution_features=evolution_features,
        cell_ids=cell_ids,
    )


def collate_amici_batch(samples: list[dict]) -> AMICIBatch:
    """Collate function for AMICI-style continuous attention format."""
    receiver = torch.from_numpy(np.stack([s["receiver"] for s in samples]))
    neighbors = torch.from_numpy(np.stack([s["neighbors"] for s in samples]))
    distances = torch.from_numpy(np.stack([s["distances"] for s in samples]))
    neighbor_mask = torch.from_numpy(np.stack([s["neighbor_mask"] for s in samples]))
    hlca = torch.from_numpy(np.stack([s["hlca"] for s in samples]))
    luca = torch.from_numpy(np.stack([s["luca"] for s in samples]))
    stage_idx = torch.tensor([s["stage_idx"] for s in samples], dtype=torch.long)

    pathway = None
    if samples[0].get("pathway") is not None:
        pathway = torch.from_numpy(np.stack([s["pathway"] for s in samples]))

    stats = None
    if samples[0].get("stats") is not None:
        stats = torch.from_numpy(np.stack([s["stats"] for s in samples]))

    pathway_targets = None
    if samples[0].get("pathway_targets") is not None:
        pathway_targets = torch.from_numpy(np.stack([s["pathway_targets"] for s in samples]))

    proliferation_target = None
    if samples[0].get("proliferation_target") is not None:
        proliferation_target = torch.tensor(
            [s["proliferation_target"] for s in samples], dtype=torch.float32
        )

    evolution_features = None
    if samples[0].get("evolution_features") is not None:
        evolution_features = torch.from_numpy(np.stack([s["evolution_features"] for s in samples]))

    donor_ids = [s["donor_id"] for s in samples]
    cell_ids = [s["cell_id"] for s in samples]

    return AMICIBatch(
        receiver=receiver,
        neighbors=neighbors,
        distances=distances,
        neighbor_mask=neighbor_mask,
        hlca=hlca,
        luca=luca,
        pathway=pathway,
        stats=stats,
        stage_idx=stage_idx,
        donor_ids=donor_ids,
        cell_ids=cell_ids,
        pathway_targets=pathway_targets,
        proliferation_target=proliferation_target,
        evolution_features=evolution_features,
    )


def create_dataloaders(
    data_dir: str | Path,
    fold_idx: int,
    manifest_path: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    latent_dim: int = LATENT_DIM,
    max_cells_per_ring: int = MAX_CELLS_PER_RING,
    shuffle_rings: bool = False,
    use_ddp: bool = False,
) -> tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """Create train/val/test DataLoaders for a fold.

    Args:
        data_dir: Directory containing neighborhoods.parquet
        fold_idx: Fold index for cross-validation
        manifest_path: Path to split_manifest.json
        batch_size: Batch size
        num_workers: DataLoader workers
        latent_dim: Embedding dimension
        max_cells_per_ring: Max cells per ring for padding
        shuffle_rings: If True, randomly shuffle cells across rings (random_niche ablation)
        use_ddp: If True, use DistributedSampler for multi-GPU training

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    from stagebridge.loaders.splits import load_split_manifest

    data_dir = Path(data_dir)
    if manifest_path is None:
        manifest_path = data_dir / "split_manifest.json"

    manifest = load_split_manifest(manifest_path)
    fold = manifest.get_fold(fold_idx)

    train_dataset = StageBridgeDataset(
        data_dir,
        donors=fold.train_donors,
        latent_dim=latent_dim,
        max_cells_per_ring=max_cells_per_ring,
        shuffle_rings=shuffle_rings,
    )
    val_dataset = StageBridgeDataset(
        data_dir,
        donors=fold.val_donors,
        latent_dim=latent_dim,
        max_cells_per_ring=max_cells_per_ring,
        shuffle_rings=shuffle_rings,
    )
    test_dataset = StageBridgeDataset(
        data_dir,
        donors=fold.test_donors,
        latent_dim=latent_dim,
        max_cells_per_ring=max_cells_per_ring,
        shuffle_rings=shuffle_rings,
    )

    # DDP samplers for distributed training
    train_sampler = None
    val_sampler = None
    if use_ddp and len(train_dataset) > 0:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        if len(val_dataset) > 0:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, max(1, len(train_dataset))),
        shuffle=(train_sampler is None and len(train_dataset) > 0),
        sampler=train_sampler,
        collate_fn=collate_niche_batch,
        num_workers=num_workers,
        drop_last=len(train_dataset) >= batch_size,
        pin_memory=True,
    ) if len(train_dataset) > 0 else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, max(1, len(val_dataset))),
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_niche_batch,
        num_workers=num_workers,
        pin_memory=True,
    ) if len(val_dataset) > 0 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(batch_size, max(1, len(test_dataset))),
        shuffle=False,
        collate_fn=collate_niche_batch,
        num_workers=num_workers,
        pin_memory=True,
    ) if len(test_dataset) > 0 else None

    return train_loader, val_loader, test_loader
