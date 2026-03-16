"""
Data loaders for StageBridge V1.

Provides unified API for loading both synthetic and real datasets
following the canonical data model specification.

Key features:
- Load cells.parquet, neighborhoods.parquet, stage_edges.parquet
- Parse split_manifest.json for donor-held-out CV
- Support batching with per-stage-edge sampling
- Compatible with both synthetic and real LUAD data
- Memory-efficient: only load required folds into memory
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from dataclasses import dataclass


@dataclass
class StageBridgeBatch:
    """Container for a batch of transition data."""

    # Cell identifiers
    cell_ids: list[str]
    donor_ids: list[str]

    # Stage information
    source_stages: list[str]
    target_stages: list[str]
    edge_ids: list[str]

    # Latent embeddings
    z_source: torch.Tensor  # (batch_size, latent_dim)
    z_target: torch.Tensor  # (batch_size, latent_dim)

    # Niche context (9 tokens per cell)
    niche_tokens: torch.Tensor  # (batch_size, 9, token_dim)
    niche_mask: torch.Tensor  # (batch_size, 9) - boolean mask for valid tokens

    # Evolutionary features (optional)
    wes_features: torch.Tensor | None = None  # (batch_size, n_wes_features)
    has_wes: torch.Tensor | None = None  # (batch_size,) - boolean mask

    # Ground truth (for synthetic data)
    niche_influence: torch.Tensor | None = None  # (batch_size,)

    def to(self, device: torch.device):
        """Move all tensors to device."""
        return StageBridgeBatch(
            cell_ids=self.cell_ids,
            donor_ids=self.donor_ids,
            source_stages=self.source_stages,
            target_stages=self.target_stages,
            edge_ids=self.edge_ids,
            z_source=self.z_source.to(device),
            z_target=self.z_target.to(device),
            niche_tokens=self.niche_tokens.to(device),
            niche_mask=self.niche_mask.to(device),
            wes_features=self.wes_features.to(device) if self.wes_features is not None else None,
            has_wes=self.has_wes.to(device) if self.has_wes is not None else None,
            niche_influence=self.niche_influence.to(device)
            if self.niche_influence is not None
            else None,
        )


class StageBridgeDataset(Dataset):
    """
    Dataset for cell-state transitions with spatial niche context.

    Loads data from canonical format:
    - cells.parquet: cell-level features and latent embeddings
    - neighborhoods.parquet: 9-token niche structure per cell
    - stage_edges.parquet: valid transition edges
    - split_manifest.json: donor-held-out CV splits

    Args:
        data_dir: Path to processed data directory
        fold: Which CV fold to load (0-4 for 5-fold CV)
        split: 'train', 'val', or 'test'
        latent_dim: Dimensionality of latent embeddings
        load_wes: Whether to load WES features
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        fold: int = 0,
        split: str = "train",
        latent_dim: int = 2,
        load_wes: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.fold = fold
        self.split = split
        self.latent_dim = latent_dim
        self.load_wes = load_wes

        # Load data
        self.cells = pd.read_parquet(self.data_dir / "cells.parquet")
        self.neighborhoods = pd.read_parquet(self.data_dir / "neighborhoods.parquet")
        self.stage_edges = pd.read_parquet(self.data_dir / "stage_edges.parquet")

        # Load split manifest
        with open(self.data_dir / "split_manifest.json") as f:
            splits = json.load(f)

        # Filter to current fold and split
        fold_spec = splits["folds"][fold]
        donor_list = fold_spec[f"{split}_donors"]
        self.cells = self.cells[self.cells["donor_id"].isin(donor_list)].reset_index(drop=True)
        self.neighborhoods = self.neighborhoods[
            self.neighborhoods["donor_id"].isin(donor_list)
        ].reset_index(drop=True)

        # Build index: for each stage edge, find all cells at source stage
        self._build_edge_index()

        print(f"Loaded {split} split (fold {fold}):")
        print(f"  Cells: {len(self.cells)}")
        print(f"  Donors: {self.cells['donor_id'].nunique()}")
        print(f"  Valid transitions: {len(self.edge_to_cells)}")

    def _build_edge_index(self):
        """Build index mapping stage edges to source cells."""
        self.edge_to_cells = {}

        # OPTIMIZED: Use itertuples() instead of iterrows() (10× faster)
        for edge in self.stage_edges.itertuples():
            edge_id = edge.edge_id
            source_stage = edge.source_stage

            # Find all cells at source stage
            source_cells = self.cells[self.cells["stage"] == source_stage]
            cell_indices = source_cells.index.tolist()

            if len(cell_indices) > 0:
                self.edge_to_cells[edge_id] = cell_indices

        # Flatten into (edge_id, cell_idx) pairs for sampling
        self.samples = []
        for edge_id, cell_indices in self.edge_to_cells.items():
            for cell_idx in cell_indices:
                self.samples.append((edge_id, cell_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a single transition example."""
        edge_id, cell_idx = self.samples[idx]

        # Get source cell
        source_cell = self.cells.iloc[cell_idx]

        # Get target stage (for this edge)
        edge = self.stage_edges[self.stage_edges["edge_id"] == edge_id].iloc[0]
        target_stage = edge["target_stage"]

        # Sample a target cell from target stage (same donor for matched pairs)
        target_candidates = self.cells[
            (self.cells["stage"] == target_stage)
            & (self.cells["donor_id"] == source_cell["donor_id"])
        ]

        if len(target_candidates) == 0:
            # Fallback: sample from any donor if no matched donor
            target_candidates = self.cells[self.cells["stage"] == target_stage]

        if len(target_candidates) == 0:
            # No target available - return source as target (identity transition)
            # This handles edge cases with small splits
            target_cell = source_cell
        else:
            target_cell = target_candidates.sample(n=1, random_state=idx).iloc[0]

        # Get latent embeddings
        z_source = np.array([source_cell[f"z_fused_{i}"] for i in range(self.latent_dim)])
        z_target = np.array([target_cell[f"z_fused_{i}"] for i in range(self.latent_dim)])

        # Get niche context (9 tokens)
        niche = self.neighborhoods[self.neighborhoods["cell_id"] == source_cell["cell_id"]].iloc[0]

        niche_tokens, niche_mask = self._parse_niche_tokens(niche)

        # Get WES features (optional)
        wes_features = None
        has_wes = False
        if self.load_wes and "tmb" in source_cell:
            wes_features = np.array(
                [
                    source_cell["tmb"],
                    source_cell.get("smoking_signature", 0.0),
                    source_cell.get("uv_signature", 0.0),
                ]
            )
            has_wes = True

        # Ground truth niche influence (for synthetic data only)
        niche_influence = niche.get("niche_influence", None)

        return {
            "cell_id": source_cell["cell_id"],
            "donor_id": source_cell["donor_id"],
            "source_stage": source_cell["stage"],
            "target_stage": target_stage,
            "edge_id": edge_id,
            "z_source": torch.from_numpy(z_source).float(),
            "z_target": torch.from_numpy(z_target).float(),
            "niche_tokens": torch.from_numpy(niche_tokens).float(),
            "niche_mask": torch.from_numpy(niche_mask).bool(),
            "wes_features": torch.from_numpy(wes_features).float()
            if wes_features is not None
            else None,
            "has_wes": torch.tensor(has_wes).bool(),
            "niche_influence": torch.tensor(niche_influence).float()
            if niche_influence is not None
            else None,
        }

    def _parse_niche_tokens(self, niche: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse 9-token niche structure into tensor.

        Returns:
            niche_tokens: (9, token_dim) array
            niche_mask: (9,) boolean mask
        """
        tokens = niche["tokens"]

        # Token dimensionality: latent_dim + extra features
        token_dim = self.latent_dim + 4  # +4 for cell type embedding, stats, etc.

        niche_array = np.zeros((9, token_dim))
        mask = np.zeros(9, dtype=bool)

        for token in tokens:
            idx = token["token_idx"]
            mask[idx] = True

            if token["token_type"] == "receiver":
                # Receiver: use z_fused
                z = token["z_fused"]
                niche_array[idx, : self.latent_dim] = z[: self.latent_dim]

            elif token["token_type"].startswith("ring"):
                # Ring: use pooled embedding
                z = token["z_pooled"]
                niche_array[idx, : self.latent_dim] = z[: self.latent_dim]
                # Add diversity as extra feature
                niche_array[idx, self.latent_dim] = token.get("n_cells", 0) / 5.0

            elif token["token_type"] == "hlca":
                # HLCA reference
                z = token["z_hlca"]
                niche_array[idx, : self.latent_dim] = z[: self.latent_dim]

            elif token["token_type"] == "luca":
                # LuCA reference
                z = token["z_luca"]
                niche_array[idx, : self.latent_dim] = z[: self.latent_dim]

            elif token["token_type"] == "pathway":
                # Pathway activity
                niche_array[idx, 0] = token.get("emt_score", 0.0)
                niche_array[idx, 1] = token.get("caf_fraction", 0.0)
                niche_array[idx, 2] = token.get("immune_fraction", 0.0)

            elif token["token_type"] == "stats":
                # Summary stats
                niche_array[idx, 0] = token.get("n_neighbors", 0) / 20.0  # Normalize
                niche_array[idx, 1] = token.get("diversity", 0) / 8.0  # Max 8 cell types

        return niche_array, mask


def collate_fn(batch: list[dict]) -> StageBridgeBatch:
    """Collate function for DataLoader."""
    return StageBridgeBatch(
        cell_ids=[x["cell_id"] for x in batch],
        donor_ids=[x["donor_id"] for x in batch],
        source_stages=[x["source_stage"] for x in batch],
        target_stages=[x["target_stage"] for x in batch],
        edge_ids=[x["edge_id"] for x in batch],
        z_source=torch.stack([x["z_source"] for x in batch]),
        z_target=torch.stack([x["z_target"] for x in batch]),
        niche_tokens=torch.stack([x["niche_tokens"] for x in batch]),
        niche_mask=torch.stack([x["niche_mask"] for x in batch]),
        wes_features=torch.stack([x["wes_features"] for x in batch])
        if batch[0]["wes_features"] is not None
        else None,
        has_wes=torch.stack([x["has_wes"] for x in batch]),
        niche_influence=torch.stack([x["niche_influence"] for x in batch])
        if batch[0]["niche_influence"] is not None
        else None,
    )


def get_dataloader(
    data_dir: Union[str, Path],
    fold: int = 0,
    split: str = "train",
    batch_size: int = 32,
    latent_dim: int = 2,
    load_wes: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convenience function to create a DataLoader.

    Args:
        data_dir: Path to processed data
        fold: CV fold (0-4)
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        latent_dim: Latent embedding dimensionality
        load_wes: Load WES features
        num_workers: Number of data loading workers
        shuffle: Shuffle data

    Returns:
        DataLoader instance
    """
    dataset = StageBridgeDataset(
        data_dir=data_dir,
        fold=fold,
        split=split,
        latent_dim=latent_dim,
        load_wes=load_wes,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


class NegativeControlDataset(Dataset):
    """
    Generate negative control samples for evaluation.

    Negative controls:
    1. Wrong stage edges (impossible transitions)
    2. Shuffled neighborhoods (randomized niche)
    3. Mismatched donors (wrong genomic context)
    """

    def __init__(
        self,
        base_dataset: StageBridgeDataset,
        control_type: str = "wrong_edge",
        seed: int = 42,
    ):
        """
        Args:
            base_dataset: Base dataset to generate controls from
            control_type: 'wrong_edge', 'shuffled_niche', or 'mismatched_donor'
            seed: Random seed
        """
        self.base_dataset = base_dataset
        self.control_type = control_type
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get negative control sample."""
        # Get base sample
        sample = self.base_dataset[idx]

        if self.control_type == "wrong_edge":
            # Replace target with invalid stage
            valid_stages = self.base_dataset.cells["stage"].unique()
            invalid_stages = [
                s
                for s in valid_stages
                if s != sample["source_stage"] and s != sample["target_stage"]
            ]

            if len(invalid_stages) > 0:
                wrong_stage = self.rng.choice(invalid_stages)
                wrong_target = (
                    self.base_dataset.cells[self.base_dataset.cells["stage"] == wrong_stage]
                    .sample(n=1, random_state=idx)
                    .iloc[0]
                )

                z_target = np.array(
                    [wrong_target[f"z_fused_{i}"] for i in range(self.base_dataset.latent_dim)]
                )
                sample["z_target"] = torch.from_numpy(z_target).float()
                sample["target_stage"] = wrong_stage

        elif self.control_type == "shuffled_niche":
            # Shuffle niche token order (break spatial structure)
            tokens = sample["niche_tokens"]
            mask = sample["niche_mask"]

            # Keep receiver (token 0) fixed, shuffle others
            valid_tokens = tokens[1:][mask[1:]]
            shuffled = valid_tokens[torch.randperm(len(valid_tokens))]

            tokens_shuffled = tokens.clone()
            tokens_shuffled[1:][mask[1:]] = shuffled
            sample["niche_tokens"] = tokens_shuffled

        elif self.control_type == "mismatched_donor":
            # Replace with different donor's genomic features
            if sample["wes_features"] is not None:
                other_cells = self.base_dataset.cells[
                    self.base_dataset.cells["donor_id"] != sample["donor_id"]
                ]

                if len(other_cells) > 0:
                    wrong_cell = other_cells.sample(n=1, random_state=idx).iloc[0]
                    wes_wrong = np.array(
                        [
                            wrong_cell["tmb"],
                            wrong_cell.get("smoking_signature", 0.0),
                            wrong_cell.get("uv_signature", 0.0),
                        ]
                    )
                    sample["wes_features"] = torch.from_numpy(wes_wrong).float()

        return sample


def get_negative_control_loader(
    base_dataset: StageBridgeDataset,
    control_type: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader for negative controls."""
    control_dataset = NegativeControlDataset(
        base_dataset=base_dataset,
        control_type=control_type,
    )

    return DataLoader(
        control_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    # Test data loading on synthetic data
    from stagebridge.data.synthetic import generate_synthetic_dataset

    print("Generating synthetic dataset...")
    data_dir = generate_synthetic_dataset(n_cells=500, n_donors=5)

    print("\nTesting data loader...")
    loader = get_dataloader(
        data_dir=data_dir,
        fold=0,
        split="train",
        batch_size=16,
        latent_dim=2,
    )

    print(f"DataLoader created: {len(loader)} batches")

    # Test one batch
    batch = next(iter(loader))
    print("\nSample batch:")
    print(f"  z_source shape: {batch.z_source.shape}")
    print(f"  z_target shape: {batch.z_target.shape}")
    print(f"  niche_tokens shape: {batch.niche_tokens.shape}")
    print(f"  niche_mask shape: {batch.niche_mask.shape}")
    if batch.wes_features is not None:
        print(f"  wes_features shape: {batch.wes_features.shape}")

    print("\n Data loading works!")
