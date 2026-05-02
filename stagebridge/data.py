"""Data loading API for StageBridge.

Provides clean interfaces for loading and preparing data from AnnData objects,
similar to scanpy/scvi-tools patterns.

Example usage:
    import stagebridge as sb
    import scanpy as sc

    # Load spatial data
    adata = sc.read_h5ad("spatial.h5ad")

    # Compute neighborhoods (receiver-centered rings)
    sb.prepare_neighborhoods(
        adata,
        ring_radii=[50, 100, 150, 200],  # microns
        embedding_key="X_scvi",
    )

    # Create dataset for training
    dataset = sb.StageBridgeDataset.from_anndata(adata)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from stagebridge.contracts import (
    LATENT_DIM,
    HLCA_DIM,
    LUCA_DIM,
    STAGE_TO_IDX,
    MAX_CELLS_PER_RING,
)

if TYPE_CHECKING:
    import anndata as ad


def prepare_neighborhoods(
    adata: "ad.AnnData",
    ring_radii: list[float] = [50, 100, 150, 200],
    embedding_key: str = "X_scvi",
    hlca_key: str | None = "X_scANVI_hlca",
    luca_key: str | None = "X_scVI_luca",
    spatial_key: str = "spatial",
    max_cells_per_ring: int = MAX_CELLS_PER_RING,
    stage_column: str = "stage",
    donor_column: str = "donor_id",
    copy: bool = False,
) -> "ad.AnnData | None":
    """Compute receiver-centered spatial neighborhoods.

    For each cell, finds neighbors in concentric rings and prepares the
    9-token structure for StageBridge:
        Token 0: Receiver cell
        Tokens 1-4: Spatial rings (closest to farthest)
        Token 5: HLCA reference embedding
        Token 6: LuCA reference embedding
        Token 7: Pathway features (optional)
        Token 8: Stats features (optional)

    Args:
        adata: AnnData with spatial coordinates and embeddings
        ring_radii: Radii for concentric rings in spatial units (microns)
        embedding_key: Key in obsm for cell embeddings to use
        hlca_key: Key in obsm for HLCA reference embeddings (or None)
        luca_key: Key in obsm for LuCA reference embeddings (or None)
        spatial_key: Key in obsm for spatial coordinates
        max_cells_per_ring: Maximum cells to keep per ring
        stage_column: Column in obs for disease stage
        donor_column: Column in obs for donor/patient ID
        copy: If True, return a copy of adata

    Returns:
        AnnData with neighborhoods in uns["X_neighborhoods"] (or copy)

    Example:
        import stagebridge as sb

        # Basic usage
        sb.prepare_neighborhoods(adata)

        # Custom ring radii
        sb.prepare_neighborhoods(
            adata,
            ring_radii=[100, 200, 300, 400],  # larger rings
        )
    """
    import scipy.spatial

    if copy:
        adata = adata.copy()

    # Validate inputs
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinates '{spatial_key}' not found in obsm")
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embeddings '{embedding_key}' not found in obsm")

    coords = adata.obsm[spatial_key]
    if coords.shape[1] < 2:
        raise ValueError(f"Spatial coordinates must have at least 2 dimensions")

    embeddings = adata.obsm[embedding_key]
    n_cells = adata.n_obs

    # Build KD-tree for efficient neighbor queries
    tree = scipy.spatial.cKDTree(coords[:, :2])

    # Get HLCA/LuCA embeddings if available
    hlca_embeddings = None
    luca_embeddings = None
    if hlca_key and hlca_key in adata.obsm:
        hlca_embeddings = adata.obsm[hlca_key]
    if luca_key and luca_key in adata.obsm:
        luca_embeddings = adata.obsm[luca_key]

    # Prepare rings with boundaries
    ring_boundaries = [0] + list(ring_radii)
    n_rings = len(ring_radii)

    # Build neighborhoods
    neighborhoods = []

    for i in range(n_cells):
        cell_id = adata.obs.index[i]
        coord = coords[i, :2]
        receiver_emb = embeddings[i]

        # Get stage and donor
        stage = adata.obs[stage_column].iloc[i] if stage_column in adata.obs.columns else "Unknown"
        donor = adata.obs[donor_column].iloc[i] if donor_column in adata.obs.columns else "Unknown"

        # Find neighbors within max radius
        max_radius = ring_radii[-1]
        neighbor_idx = tree.query_ball_point(coord, max_radius)
        neighbor_idx = [j for j in neighbor_idx if j != i]  # Exclude self

        if not neighbor_idx:
            neighbor_coords = np.array([]).reshape(0, 2)
            neighbor_embeddings = np.array([]).reshape(0, embeddings.shape[1])
        else:
            neighbor_coords = coords[neighbor_idx, :2]
            neighbor_embeddings = embeddings[neighbor_idx]

        # Compute distances
        if len(neighbor_idx) > 0:
            distances = np.sqrt(np.sum((neighbor_coords - coord) ** 2, axis=1))
        else:
            distances = np.array([])

        # Assign to rings
        ring_cells = []
        for r in range(n_rings):
            inner = ring_boundaries[r]
            outer = ring_boundaries[r + 1]

            mask = (distances >= inner) & (distances < outer)
            ring_idx = np.where(mask)[0]

            if len(ring_idx) > max_cells_per_ring:
                # Keep closest cells
                ring_distances = distances[ring_idx]
                sort_idx = np.argsort(ring_distances)[:max_cells_per_ring]
                ring_idx = ring_idx[sort_idx]

            if len(ring_idx) > 0:
                ring_emb = neighbor_embeddings[ring_idx].tolist()
            else:
                ring_emb = []

            ring_cells.append(ring_emb)

        # Prepare HLCA/LuCA
        hlca_z = hlca_embeddings[i].tolist() if hlca_embeddings is not None else [0.0] * HLCA_DIM
        luca_z = luca_embeddings[i].tolist() if luca_embeddings is not None else [0.0] * LUCA_DIM

        # Fused embedding (concat HLCA + LuCA or just the embedding)
        if hlca_embeddings is not None and luca_embeddings is not None:
            fused = np.concatenate([hlca_embeddings[i][:HLCA_DIM], luca_embeddings[i][:LUCA_DIM]])
            receiver_z = fused.tolist()
        else:
            receiver_z = receiver_emb[:LATENT_DIM].tolist()

        neighborhood = {
            "cell_id": cell_id,
            "donor_id": donor,
            "stage": stage,
            "receiver_z": receiver_z,
            "ring_1_cells": ring_cells[0] if n_rings > 0 else [],
            "ring_2_cells": ring_cells[1] if n_rings > 1 else [],
            "ring_3_cells": ring_cells[2] if n_rings > 2 else [],
            "ring_4_cells": ring_cells[3] if n_rings > 3 else [],
            "hlca_z": hlca_z,
            "luca_z": luca_z,
        }
        neighborhoods.append(neighborhood)

    # Store in adata
    neighborhoods_df = pd.DataFrame(neighborhoods)
    adata.uns["X_neighborhoods"] = neighborhoods_df

    print(f"Prepared {len(neighborhoods_df)} neighborhoods")
    print(f"  Ring radii: {ring_radii}")
    print(f"  Max cells per ring: {max_cells_per_ring}")

    if copy:
        return adata
    return None


def prepare_neighborhoods_from_graph(
    adata: "ad.AnnData",
    n_neighbors: int = 15,
    embedding_key: str = "X_scvi",
    hlca_key: str | None = "X_scANVI_hlca",
    luca_key: str | None = "X_scVI_luca",
    connectivity_key: str = "connectivities",
    stage_column: str = "stage",
    donor_column: str = "donor_id",
    copy: bool = False,
) -> "ad.AnnData | None":
    """Compute neighborhoods from connectivity graph (for non-spatial data).

    Alternative to spatial rings when using k-NN graph-based neighbors.
    Assigns neighbors to 4 "pseudo-rings" based on graph distance.

    Args:
        adata: AnnData with computed neighbors
        n_neighbors: Number of neighbors per ring
        embedding_key: Key in obsm for embeddings
        hlca_key: Key in obsm for HLCA embeddings
        luca_key: Key in obsm for LuCA embeddings
        connectivity_key: Key in obsp for connectivity matrix
        stage_column: Column for stage labels
        donor_column: Column for donor IDs
        copy: If True, return copy

    Returns:
        AnnData with neighborhoods prepared
    """
    import scipy.sparse

    if copy:
        adata = adata.copy()

    if connectivity_key not in adata.obsp:
        raise ValueError(f"Connectivity matrix '{connectivity_key}' not found. Run sc.pp.neighbors() first.")

    conn = adata.obsp[connectivity_key]
    embeddings = adata.obsm[embedding_key]
    n_cells = adata.n_obs

    hlca_embeddings = adata.obsm.get(hlca_key) if hlca_key else None
    luca_embeddings = adata.obsm.get(luca_key) if luca_key else None

    neighborhoods = []

    for i in range(n_cells):
        cell_id = adata.obs.index[i]
        receiver_emb = embeddings[i]

        stage = adata.obs[stage_column].iloc[i] if stage_column in adata.obs.columns else "Unknown"
        donor = adata.obs[donor_column].iloc[i] if donor_column in adata.obs.columns else "Unknown"

        # Get neighbors from connectivity matrix
        if scipy.sparse.issparse(conn):
            row = conn.getrow(i).toarray().flatten()
        else:
            row = conn[i]

        neighbor_idx = np.where(row > 0)[0]
        neighbor_weights = row[neighbor_idx]

        # Sort by weight (higher = closer)
        sort_idx = np.argsort(-neighbor_weights)
        neighbor_idx = neighbor_idx[sort_idx]

        # Split into 4 pseudo-rings by quartiles
        n_per_ring = len(neighbor_idx) // 4
        ring_cells = []
        for r in range(4):
            start = r * n_per_ring
            end = (r + 1) * n_per_ring if r < 3 else len(neighbor_idx)
            ring_idx = neighbor_idx[start:end]

            if len(ring_idx) > MAX_CELLS_PER_RING:
                ring_idx = ring_idx[:MAX_CELLS_PER_RING]

            if len(ring_idx) > 0:
                ring_emb = embeddings[ring_idx].tolist()
            else:
                ring_emb = []

            ring_cells.append(ring_emb)

        # HLCA/LuCA
        hlca_z = hlca_embeddings[i].tolist() if hlca_embeddings is not None else [0.0] * HLCA_DIM
        luca_z = luca_embeddings[i].tolist() if luca_embeddings is not None else [0.0] * LUCA_DIM

        if hlca_embeddings is not None and luca_embeddings is not None:
            fused = np.concatenate([hlca_embeddings[i][:HLCA_DIM], luca_embeddings[i][:LUCA_DIM]])
            receiver_z = fused.tolist()
        else:
            receiver_z = receiver_emb[:LATENT_DIM].tolist()

        neighborhood = {
            "cell_id": cell_id,
            "donor_id": donor,
            "stage": stage,
            "receiver_z": receiver_z,
            "ring_1_cells": ring_cells[0],
            "ring_2_cells": ring_cells[1],
            "ring_3_cells": ring_cells[2],
            "ring_4_cells": ring_cells[3],
            "hlca_z": hlca_z,
            "luca_z": luca_z,
        }
        neighborhoods.append(neighborhood)

    neighborhoods_df = pd.DataFrame(neighborhoods)
    adata.uns["X_neighborhoods"] = neighborhoods_df

    print(f"Prepared {len(neighborhoods_df)} neighborhoods from graph")

    if copy:
        return adata
    return None


class StageBridgeDataset(Dataset):
    """PyTorch Dataset for StageBridge training.

    Can be created from:
    - AnnData with prepared neighborhoods
    - neighborhoods.parquet file
    - DataFrame with neighborhood data

    Example:
        # From AnnData
        dataset = StageBridgeDataset.from_anndata(adata)

        # From parquet file
        dataset = StageBridgeDataset.from_parquet("neighborhoods.parquet")

        # Create DataLoader
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
    """

    NUM_RINGS = 4

    def __init__(
        self,
        neighborhoods: pd.DataFrame,
        latent_dim: int = LATENT_DIM,
        max_cells_per_ring: int = MAX_CELLS_PER_RING,
        stages: Sequence[str] | None = None,
        donors: Sequence[str] | None = None,
    ):
        """Initialize dataset.

        Args:
            neighborhoods: DataFrame with neighborhood data
            latent_dim: Embedding dimension
            max_cells_per_ring: Maximum cells per ring for padding
            stages: Optional filter for specific stages
            donors: Optional filter for specific donors
        """
        self.latent_dim = latent_dim
        self.max_cells_per_ring = max_cells_per_ring

        # Apply filters
        data = neighborhoods.copy()
        if donors is not None:
            data = data[data["donor_id"].isin(set(donors))]
        if stages is not None:
            data = data[data["stage"].isin(set(stages))]

        self.neighborhoods = data.reset_index(drop=True)

        # Detect format
        self.ring_format = "ring_1_cells" in self.neighborhoods.columns
        self.tokenized_format = "tokens" in self.neighborhoods.columns

        if not self.ring_format and not self.tokenized_format:
            raise ValueError(
                "neighborhoods must have either ring columns (ring_1_cells, ...) "
                "or tokens column"
            )

    @classmethod
    def from_anndata(
        cls,
        adata: "ad.AnnData",
        latent_dim: int = LATENT_DIM,
        max_cells_per_ring: int = MAX_CELLS_PER_RING,
        stages: Sequence[str] | None = None,
        donors: Sequence[str] | None = None,
    ) -> "StageBridgeDataset":
        """Create dataset from AnnData.

        Args:
            adata: AnnData with prepared neighborhoods
            latent_dim: Embedding dimension
            max_cells_per_ring: Max cells per ring
            stages: Optional stage filter
            donors: Optional donor filter

        Returns:
            StageBridgeDataset

        Example:
            import stagebridge as sb

            sb.prepare_neighborhoods(adata)
            dataset = sb.StageBridgeDataset.from_anndata(adata)
        """
        if "X_neighborhoods" not in adata.uns:
            raise ValueError(
                "AnnData has no neighborhoods prepared. "
                "Run stagebridge.prepare_neighborhoods(adata) first."
            )

        return cls(
            neighborhoods=adata.uns["X_neighborhoods"],
            latent_dim=latent_dim,
            max_cells_per_ring=max_cells_per_ring,
            stages=stages,
            donors=donors,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        latent_dim: int = LATENT_DIM,
        max_cells_per_ring: int = MAX_CELLS_PER_RING,
        stages: Sequence[str] | None = None,
        donors: Sequence[str] | None = None,
    ) -> "StageBridgeDataset":
        """Create dataset from parquet file.

        Args:
            path: Path to neighborhoods.parquet
            latent_dim: Embedding dimension
            max_cells_per_ring: Max cells per ring
            stages: Optional stage filter
            donors: Optional donor filter

        Returns:
            StageBridgeDataset
        """
        neighborhoods = pd.read_parquet(path)
        return cls(
            neighborhoods=neighborhoods,
            latent_dim=latent_dim,
            max_cells_per_ring=max_cells_per_ring,
            stages=stages,
            donors=donors,
        )

    def __len__(self) -> int:
        return len(self.neighborhoods)

    def __getitem__(self, idx: int) -> dict:
        """Get a single neighborhood sample."""
        row = self.neighborhoods.iloc[idx]

        cell_id = row["cell_id"]
        donor_id = row["donor_id"]
        stage = row["stage"]
        stage_idx = STAGE_TO_IDX.get(stage, 0)

        if self.ring_format:
            return self._getitem_ring_format(row, cell_id, donor_id, stage_idx)
        else:
            return self._getitem_tokenized(row, cell_id, donor_id, stage_idx)

    def _getitem_ring_format(self, row, cell_id, donor_id, stage_idx) -> dict:
        """Parse ring column format."""
        receiver = np.array(row["receiver_z"], dtype=np.float32)
        hlca = np.array(row["hlca_z"], dtype=np.float32)
        luca = np.array(row["luca_z"], dtype=np.float32)

        ring_cells = []
        ring_masks = []

        for i in range(1, self.NUM_RINGS + 1):
            cells_list = row.get(f"ring_{i}_cells", [])
            if cells_list is None:
                cells_list = []
            elif isinstance(cells_list, np.ndarray) and cells_list.size == 0:
                cells_list = []

            n_cells = len(cells_list) if cells_list else 0

            padded = np.zeros((self.max_cells_per_ring, self.latent_dim), dtype=np.float32)
            mask = np.zeros(self.max_cells_per_ring, dtype=bool)

            if n_cells > 0:
                n_use = min(n_cells, self.max_cells_per_ring)
                for j in range(n_use):
                    padded[j] = np.array(cells_list[j], dtype=np.float32)[:self.latent_dim]
                    mask[j] = True

            ring_cells.append(padded)
            ring_masks.append(mask)

        pathway = None
        if "pathway_z" in row and row["pathway_z"] is not None:
            pathway = np.array(row["pathway_z"], dtype=np.float32)

        stats = None
        if "stats_z" in row and row["stats_z"] is not None:
            stats = np.array(row["stats_z"], dtype=np.float32)

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
        }

    def _getitem_tokenized(self, row, cell_id, donor_id, stage_idx) -> dict:
        """Parse tokenized format."""
        tokens = row["tokens"]

        receiver = np.array(tokens[0].get("z_fused", np.zeros(self.latent_dim)), dtype=np.float32)
        hlca = np.array(tokens[5].get("z_hlca", np.zeros(HLCA_DIM)) if len(tokens) > 5 else np.zeros(HLCA_DIM), dtype=np.float32)
        luca = np.array(tokens[6].get("z_luca", np.zeros(LUCA_DIM)) if len(tokens) > 6 else np.zeros(LUCA_DIM), dtype=np.float32)

        ring_cells = []
        ring_masks = []

        for i in range(1, self.NUM_RINGS + 1):
            ring_token = tokens[i] if len(tokens) > i else {}
            z_pooled = ring_token.get("z_pooled")

            padded = np.zeros((self.max_cells_per_ring, self.latent_dim), dtype=np.float32)
            mask = np.zeros(self.max_cells_per_ring, dtype=bool)

            if z_pooled is not None and hasattr(z_pooled, "__len__") and len(z_pooled) > 0:
                padded[0] = np.array(z_pooled, dtype=np.float32)[:self.latent_dim]
                mask[0] = True

            ring_cells.append(padded)
            ring_masks.append(mask)

        return {
            "receiver": receiver,
            "ring_cells": ring_cells,
            "ring_masks": ring_masks,
            "hlca": hlca,
            "luca": luca,
            "pathway": None,
            "stats": None,
            "stage_idx": stage_idx,
            "donor_id": donor_id,
            "cell_id": cell_id,
        }

    def get_stage_distribution(self) -> dict[str, int]:
        """Get distribution of stages in dataset."""
        return self.neighborhoods["stage"].value_counts().to_dict()

    def get_donor_distribution(self) -> dict[str, int]:
        """Get distribution of donors in dataset."""
        return self.neighborhoods["donor_id"].value_counts().to_dict()


def collate_fn(samples: list[dict]) -> dict:
    """Collate function for DataLoader.

    Example:
        loader = DataLoader(dataset, batch_size=64, collate_fn=sb.data.collate_fn)
    """
    from stagebridge.loaders.dataset import collate_niche_batch

    # Use existing collate function
    batch = collate_niche_batch(samples)
    return batch


def create_data_loaders(
    adata: "ad.AnnData",
    batch_size: int = 64,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    random_state: int = 42,
    num_workers: int = 0,
) -> tuple:
    """Create train/val/test DataLoaders from AnnData.

    Splits by donor to prevent data leakage.

    Args:
        adata: AnnData with prepared neighborhoods
        batch_size: Batch size
        train_frac: Fraction for training
        val_frac: Fraction for validation
        random_state: Random seed
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader, test_loader)

    Example:
        train_loader, val_loader, test_loader = sb.create_data_loaders(adata)
    """
    from torch.utils.data import DataLoader

    if "X_neighborhoods" not in adata.uns:
        raise ValueError("Run prepare_neighborhoods() first")

    neighborhoods = adata.uns["X_neighborhoods"]
    donors = neighborhoods["donor_id"].unique()

    # Shuffle and split donors
    rng = np.random.default_rng(random_state)
    rng.shuffle(donors)

    n_train = int(len(donors) * train_frac)
    n_val = int(len(donors) * val_frac)

    train_donors = set(donors[:n_train])
    val_donors = set(donors[n_train:n_train + n_val])
    test_donors = set(donors[n_train + n_val:])

    train_dataset = StageBridgeDataset(neighborhoods, donors=train_donors)
    val_dataset = StageBridgeDataset(neighborhoods, donors=val_donors)
    test_dataset = StageBridgeDataset(neighborhoods, donors=test_donors)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    ) if len(train_dataset) > 0 else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    ) if len(val_dataset) > 0 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    ) if len(test_dataset) > 0 else None

    print(f"Created data loaders:")
    print(f"  Train: {len(train_dataset)} samples from {len(train_donors)} donors")
    print(f"  Val: {len(val_dataset)} samples from {len(val_donors)} donors")
    print(f"  Test: {len(test_dataset)} samples from {len(test_donors)} donors")

    return train_loader, val_loader, test_loader
