"""
Synthetic spatial world generation for semi-synthetic benchmark.

Creates synthetic 2D spatial coordinates with controlled region structure
for testing distance-dependent interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class Region:
    """A spatial region with enrichment for certain cell groups."""

    region_id: str
    center: tuple[float, float]
    radius: float
    cell_group_weights: dict[str, float]
    stage_bias: str | None = None


@dataclass
class SyntheticWorld:
    """A complete synthetic spatial world."""

    world_id: str
    split: Literal["train", "val", "test"]
    width: float
    height: float
    seed: int
    regions: list[Region]
    cell_positions: pd.DataFrame  # Contains x, y, cell_group, region_id, etc.
    metadata: dict

    def get_cells_in_radius(
        self,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> pd.DataFrame:
        """Get cells within a radius of a point."""
        distances = np.sqrt(
            (self.cell_positions["x"] - center_x) ** 2
            + (self.cell_positions["y"] - center_y) ** 2
        )
        return self.cell_positions[distances <= radius]


class WorldGenerator:
    """Generate synthetic spatial worlds with controlled structure."""

    def __init__(
        self,
        width: float = 1000.0,
        height: float = 1000.0,
        n_regions: int = 4,
        region_overlap: float = 0.2,
        boundary_softness: float = 50.0,
    ):
        self.width = width
        self.height = height
        self.n_regions = n_regions
        self.region_overlap = region_overlap
        self.boundary_softness = boundary_softness

    def generate_world(
        self,
        world_id: str,
        split: Literal["train", "val", "test"],
        cell_pools: dict[str, pd.DataFrame],
        n_cells: int,
        seed: int,
        stages: list[str] | None = None,
    ) -> SyntheticWorld:
        """Generate a single synthetic world.

        Args:
            world_id: Unique identifier for this world
            split: Train/val/test split
            cell_pools: Dictionary mapping cell group name to pool of available cells
            n_cells: Total number of cells to place
            seed: Random seed for reproducibility
            stages: Optional list of stages for stage-biased regions

        Returns:
            SyntheticWorld with spatial coordinates and cell assignments
        """
        rng = np.random.default_rng(seed)

        # Generate regions
        regions = self._generate_regions(rng, stages)

        # Sample cells and assign positions
        cell_positions = self._sample_and_place_cells(
            rng, cell_pools, n_cells, regions
        )

        metadata = {
            "world_id": world_id,
            "split": split,
            "seed": seed,
            "n_cells": len(cell_positions),
            "n_regions": len(regions),
            "width": self.width,
            "height": self.height,
            "cell_group_counts": cell_positions["cell_group"].value_counts().to_dict(),
        }

        return SyntheticWorld(
            world_id=world_id,
            split=split,
            width=self.width,
            height=self.height,
            seed=seed,
            regions=regions,
            cell_positions=cell_positions,
            metadata=metadata,
        )

    def _generate_regions(
        self,
        rng: np.random.Generator,
        stages: list[str] | None = None,
    ) -> list[Region]:
        """Generate spatial regions with different characteristics."""
        regions = []

        # Divide space into a grid with some randomness
        n_cols = int(np.ceil(np.sqrt(self.n_regions)))
        n_rows = int(np.ceil(self.n_regions / n_cols))

        cell_width = self.width / n_cols
        cell_height = self.height / n_rows

        region_idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if region_idx >= self.n_regions:
                    break

                # Center with jitter
                center_x = (col + 0.5) * cell_width + rng.uniform(-cell_width * 0.2, cell_width * 0.2)
                center_y = (row + 0.5) * cell_height + rng.uniform(-cell_height * 0.2, cell_height * 0.2)

                # Radius with overlap allowance
                radius = min(cell_width, cell_height) * (0.5 + self.region_overlap * rng.uniform(0, 1))

                # Cell group weights - vary by region type
                region_type = region_idx % 4
                if region_type == 0:
                    # Epithelial-rich region
                    weights = {
                        "epithelial_receiver": 0.6,
                        "caf_sender": 0.15,
                        "immune_sender": 0.15,
                        "endothelial_background": 0.1,
                    }
                elif region_type == 1:
                    # Stroma-rich region (CAF enriched)
                    weights = {
                        "epithelial_receiver": 0.3,
                        "caf_sender": 0.4,
                        "immune_sender": 0.15,
                        "endothelial_background": 0.15,
                    }
                elif region_type == 2:
                    # Immune-rich region
                    weights = {
                        "epithelial_receiver": 0.25,
                        "caf_sender": 0.15,
                        "immune_sender": 0.45,
                        "endothelial_background": 0.15,
                    }
                else:
                    # Mixed region
                    weights = {
                        "epithelial_receiver": 0.35,
                        "caf_sender": 0.25,
                        "immune_sender": 0.25,
                        "endothelial_background": 0.15,
                    }

                # Stage bias
                stage_bias = None
                if stages and region_idx < len(stages):
                    stage_bias = stages[region_idx % len(stages)]

                regions.append(
                    Region(
                        region_id=f"region_{region_idx}",
                        center=(center_x, center_y),
                        radius=radius,
                        cell_group_weights=weights,
                        stage_bias=stage_bias,
                    )
                )
                region_idx += 1

        return regions

    def _sample_and_place_cells(
        self,
        rng: np.random.Generator,
        cell_pools: dict[str, pd.DataFrame],
        n_cells: int,
        regions: list[Region],
    ) -> pd.DataFrame:
        """Sample cells from pools and assign spatial coordinates."""
        records = []

        # Compute global cell group distribution from regions
        global_weights: dict[str, float] = {}
        for region in regions:
            for group, weight in region.cell_group_weights.items():
                global_weights[group] = global_weights.get(group, 0) + weight

        # Normalize
        total = sum(global_weights.values())
        for group in global_weights:
            global_weights[group] /= total

        # Sample cells
        for _ in range(n_cells):
            # Choose region (weighted by area)
            region_probs = np.array([r.radius ** 2 for r in regions])
            region_probs = region_probs / region_probs.sum()
            region = rng.choice(regions, p=region_probs)

            # Choose cell group based on region weights
            available_groups = [g for g in region.cell_group_weights if g in cell_pools]
            if not available_groups:
                continue

            weights = np.array([region.cell_group_weights.get(g, 0) for g in available_groups])
            weights = weights / weights.sum()
            cell_group = rng.choice(available_groups, p=weights)

            # Sample a cell from the pool
            pool = cell_pools[cell_group]
            if len(pool) == 0:
                continue

            cell_idx = rng.integers(0, len(pool))
            cell_data = pool.iloc[cell_idx].to_dict()

            # Generate position within region (Gaussian around center)
            angle = rng.uniform(0, 2 * np.pi)
            distance = rng.exponential(region.radius / 2)
            distance = min(distance, region.radius)

            x = region.center[0] + distance * np.cos(angle)
            y = region.center[1] + distance * np.sin(angle)

            # Clamp to world bounds
            x = np.clip(x, 0, self.width)
            y = np.clip(y, 0, self.height)

            record = {
                "synthetic_cell_id": f"cell_{len(records):06d}",
                "x": x,
                "y": y,
                "cell_group": cell_group,
                "region_id": region.region_id,
                "region_stage_bias": region.stage_bias,
                **cell_data,
            }
            records.append(record)

        return pd.DataFrame(records)

    def compute_neighborhood_matrix(
        self,
        cell_positions: pd.DataFrame,
        radius: float,
    ) -> np.ndarray:
        """Compute binary neighborhood matrix within a radius.

        Returns:
            Boolean matrix where [i, j] = True if cell j is within radius of cell i
        """
        coords = cell_positions[["x", "y"]].values
        n = len(coords)

        # Compute pairwise distances
        # Using broadcasting for efficiency
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=2))

        # Create neighborhood matrix
        neighborhood = distances <= radius
        np.fill_diagonal(neighborhood, False)  # Cell is not its own neighbor

        return neighborhood

    def compute_neighborhood_composition(
        self,
        cell_positions: pd.DataFrame,
        radius: float,
        group_column: str = "cell_group",
    ) -> pd.DataFrame:
        """Compute neighborhood composition for each cell.

        Returns DataFrame with columns for each cell group showing
        the count of that group within the radius.
        """
        neighborhood = self.compute_neighborhood_matrix(cell_positions, radius)
        groups = cell_positions[group_column].values
        unique_groups = np.unique(groups)

        composition = {}
        for group in unique_groups:
            group_mask = groups == group
            # Count neighbors of this group for each cell
            counts = neighborhood @ group_mask.astype(int)
            composition[f"n_{group}"] = counts

        return pd.DataFrame(composition, index=cell_positions.index)


def generate_multiple_worlds(
    cell_pools: dict[str, pd.DataFrame],
    n_worlds: int,
    cells_per_world: int,
    split: Literal["train", "val", "test"],
    base_seed: int,
    stages: list[str] | None = None,
    width: float = 1000.0,
    height: float = 1000.0,
) -> list[SyntheticWorld]:
    """Generate multiple independent synthetic worlds.

    Args:
        cell_pools: Dictionary of cell pools by group name
        n_worlds: Number of worlds to generate
        cells_per_world: Cells per world
        split: Train/val/test designation
        base_seed: Base random seed (each world gets base_seed + i)
        stages: Optional stages for stage-biased regions
        width: World width
        height: World height

    Returns:
        List of SyntheticWorld objects
    """
    generator = WorldGenerator(width=width, height=height)

    worlds = []
    for i in range(n_worlds):
        world = generator.generate_world(
            world_id=f"{split}_world_{i:03d}",
            split=split,
            cell_pools=cell_pools,
            n_cells=cells_per_world,
            seed=base_seed + i * 1000,  # Ensure distinct seeds
            stages=stages,
        )
        worlds.append(world)

    return worlds
