"""
Synthetic data generator for StageBridge V1 testing.

Generates controlled synthetic datasets with known transition trajectories,
spatial neighborhoods, and evolutionary features for validating the model
before deploying to real data.

Design goals:
- Test all model layers (A-D) without expensive data processing
- Known ground truth for evaluation metrics
- Configurable complexity for debugging
- Compatible with canonical data model (cells.parquet, neighborhoods.parquet)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path
import json


class SyntheticDataGenerator:
    """
    Generate synthetic cell-state transition data with spatial context.

    Key features:
    - 4-stage progression: Normal → Preneoplastic → Invasive → Advanced
    - Known transition trajectories in 2D latent space
    - 9-token niche structure (receiver + 4 rings + HLCA + LuCA + pathway + stats)
    - Optional WES features with evolutionary compatibility
    - Configurable difficulty (noise, overlap, niche influence)
    """

    def __init__(
        self,
        n_cells: int = 1000,
        n_donors: int = 5,
        latent_dim: int = 2,
        n_celltypes: int = 8,
        seed: int = 42,
    ):
        """
        Initialize synthetic data generator.

        Args:
            n_cells: Total number of cells to generate
            n_donors: Number of synthetic donors
            latent_dim: Dimensionality of latent space (2 for visualization)
            n_celltypes: Number of cell types in niche
            seed: Random seed for reproducibility
        """
        self.n_cells = n_cells
        self.n_donors = n_donors
        self.latent_dim = latent_dim
        self.n_celltypes = n_celltypes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Define stage progression graph
        self.stages = ["Normal", "Preneoplastic", "Invasive", "Advanced"]
        self.stage_edges = [
            ("Normal", "Preneoplastic"),
            ("Preneoplastic", "Invasive"),
            ("Invasive", "Advanced"),
        ]

        # Define stage centroids in 2D latent space (for visualization)
        self.stage_centroids = {
            "Normal": np.array([0.0, 0.0]),
            "Preneoplastic": np.array([1.0, 0.0]),
            "Invasive": np.array([1.5, 1.0]),
            "Advanced": np.array([2.5, 1.5]),
        }

    def generate(
        self,
        noise_level: float = 0.1,
        niche_influence: float = 0.5,
        overlap: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete synthetic dataset.

        Args:
            noise_level: Gaussian noise std for latent positions
            niche_influence: Strength of niche effect on transitions (0-1)
            overlap: Stage overlap in latent space (0-1)

        Returns:
            cells: Cell table (cells.parquet schema)
            neighborhoods: Neighborhood table (neighborhoods.parquet schema)
            stage_edges: Stage transition graph
        """
        # Generate cell-level data
        cells = self._generate_cells(noise_level, overlap)

        # Generate spatial neighborhoods with 9-token structure
        neighborhoods = self._generate_neighborhoods(cells, niche_influence)

        # Generate stage edges table
        stage_edges_df = self._generate_stage_edges()

        return cells, neighborhoods, stage_edges_df

    def _generate_cells(
        self,
        noise_level: float,
        overlap: float,
    ) -> pd.DataFrame:
        """Generate cell-level data with latent embeddings and metadata."""
        cells_per_stage = self.n_cells // len(self.stages)

        records = []
        cell_id = 0

        for stage_idx, stage in enumerate(self.stages):
            centroid = self.stage_centroids[stage]

            # Expand centroid to match latent_dim
            if self.latent_dim > 2:
                centroid_expanded = np.zeros(self.latent_dim)
                centroid_expanded[:2] = centroid
            else:
                centroid_expanded = centroid

            # Generate latent positions with controlled overlap
            stage_std = noise_level + overlap * 0.3
            z_positions = self.rng.normal(
                loc=centroid_expanded,
                scale=stage_std,
                size=(cells_per_stage, self.latent_dim)
            )

            # Assign donors with stage enrichment
            # Early stages → early donors, late stages → late donors (simulate progression)
            if stage_idx < len(self.stages) // 2:
                donor_pool = list(range(self.n_donors // 2 + 1))
            else:
                donor_pool = list(range(self.n_donors // 2, self.n_donors))

            donor_ids = self.rng.choice(donor_pool, size=cells_per_stage)

            # Generate WES features (TMB, signature exposures)
            tmb = self.rng.gamma(
                shape=2.0 + stage_idx,  # Higher TMB in advanced stages
                scale=1.0,
                size=cells_per_stage
            )

            smoking_sig = self.rng.beta(
                a=2.0 + stage_idx * 0.5,
                b=5.0 - stage_idx * 0.3,
                size=cells_per_stage
            )

            uv_sig = self.rng.beta(a=1.5, b=8.0, size=cells_per_stage)

            # Create records
            for i in range(cells_per_stage):
                records.append({
                    "cell_id": f"cell_{cell_id:06d}",
                    "donor_id": f"donor_{donor_ids[i]:02d}",
                    "stage": stage,
                    "stage_idx": stage_idx,
                    "z_fused": z_positions[i].tolist(),  # Dual-reference latent (placeholder)
                    "z_hlca": (z_positions[i] + self.rng.normal(0, 0.05, self.latent_dim)).tolist(),
                    "z_luca": (z_positions[i] + self.rng.normal(0, 0.05, self.latent_dim)).tolist(),
                    "cell_type": self._assign_celltype(stage_idx),
                    "tmb": tmb[i],
                    "smoking_signature": smoking_sig[i],
                    "uv_signature": uv_sig[i],
                    "x_spatial": self.rng.uniform(0, 1000),  # Dummy spatial coords
                    "y_spatial": self.rng.uniform(0, 1000),
                })
                cell_id += 1

        df = pd.DataFrame(records)

        # Add latent dimension columns
        for dim in range(self.latent_dim):
            df[f"z_fused_{dim}"] = df["z_fused"].apply(lambda x: x[dim])
            df[f"z_hlca_{dim}"] = df["z_hlca"].apply(lambda x: x[dim])
            df[f"z_luca_{dim}"] = df["z_luca"].apply(lambda x: x[dim])

        return df

    def _assign_celltype(self, stage_idx: int) -> str:
        """Assign cell type with stage-dependent distribution."""
        celltypes = [
            "AT2", "AT1", "Club", "Basal",
            "Fibroblast", "Macrophage", "T_cell", "Endothelial"
        ]

        # AT2 enriched in early stages, fibroblasts/immune in late stages
        if stage_idx < 2:
            probs = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02]
        else:
            probs = [0.2, 0.1, 0.05, 0.05, 0.25, 0.2, 0.1, 0.05]

        return self.rng.choice(celltypes, p=probs)

    def _generate_neighborhoods(
        self,
        cells: pd.DataFrame,
        niche_influence: float,
    ) -> pd.DataFrame:
        """
        Generate spatial neighborhoods with 9-token structure.

        9 tokens:
        0. Receiver cell
        1-4. Ring 1-4 (spatial neighbors)
        5. HLCA context
        6. LuCA context
        7. Pathway activity
        8. Summary stats
        """
        records = []

        for idx, cell in cells.iterrows():
            # Find spatial neighbors (k=4 rings × cells per ring)
            # For synthetic data, randomly sample with distance-based probability
            distances = np.sqrt(
                (cells["x_spatial"] - cell["x_spatial"])**2 +
                (cells["y_spatial"] - cell["y_spatial"])**2
            )

            # Sort by distance and take top K neighbors
            k_total = 20  # 5 cells per ring × 4 rings
            neighbor_indices = np.argsort(distances)[1:k_total+1]  # Exclude self

            # Build 9-token neighborhood
            tokens = []

            # Token 0: Receiver
            tokens.append({
                "token_idx": 0,
                "token_type": "receiver",
                "cell_id": cell["cell_id"],
                "cell_type": cell["cell_type"],
                "z_fused": cell["z_fused"],
            })

            # Tokens 1-4: Rings (5 cells per ring)
            cells_per_ring = 5
            for ring in range(4):
                start = ring * cells_per_ring
                end = (ring + 1) * cells_per_ring
                ring_cells = cells.iloc[neighbor_indices[start:end]]

                # Pool cells in ring (mean embedding)
                z_pooled = np.mean([z for z in ring_cells["z_fused"]], axis=0)
                celltype_counts = ring_cells["cell_type"].value_counts().to_dict()

                tokens.append({
                    "token_idx": ring + 1,
                    "token_type": f"ring_{ring+1}",
                    "z_pooled": z_pooled.tolist(),
                    "celltype_composition": celltype_counts,
                    "n_cells": len(ring_cells),
                })

            # Token 5: HLCA reference context
            tokens.append({
                "token_idx": 5,
                "token_type": "hlca",
                "z_hlca": cell["z_hlca"],
            })

            # Token 6: LuCA disease context
            tokens.append({
                "token_idx": 6,
                "token_type": "luca",
                "z_luca": cell["z_luca"],
            })

            # Token 7: Pathway activity (simulate niche influence)
            # CAF/immune-enriched niches increase transition probability
            neighbor_cells = cells.iloc[neighbor_indices]
            caf_frac = (neighbor_cells["cell_type"] == "Fibroblast").mean()
            immune_frac = (neighbor_cells["cell_type"].isin(["Macrophage", "T_cell"])).mean()

            pathway_score = niche_influence * (0.6 * caf_frac + 0.4 * immune_frac)

            tokens.append({
                "token_idx": 7,
                "token_type": "pathway",
                "emt_score": pathway_score,
                "caf_fraction": caf_frac,
                "immune_fraction": immune_frac,
            })

            # Token 8: Summary stats
            tokens.append({
                "token_idx": 8,
                "token_type": "stats",
                "n_neighbors": k_total,
                "mean_distance": distances[neighbor_indices].mean(),
                "diversity": len(neighbor_cells["cell_type"].unique()),
            })

            records.append({
                "cell_id": cell["cell_id"],
                "donor_id": cell["donor_id"],
                "stage": cell["stage"],
                "tokens": tokens,
                "niche_influence": pathway_score,  # Ground truth for evaluation
            })

        return pd.DataFrame(records)

    def _generate_stage_edges(self) -> pd.DataFrame:
        """Generate stage transition graph."""
        records = []

        for source, target in self.stage_edges:
            records.append({
                "edge_id": f"{source}_{target}",
                "source_stage": source,
                "target_stage": target,
                "source_idx": self.stages.index(source),
                "target_idx": self.stages.index(target),
                "is_forward": True,
                "pseudotime_delta": 1.0,
            })

        return pd.DataFrame(records)

    def save(
        self,
        cells: pd.DataFrame,
        neighborhoods: pd.DataFrame,
        stage_edges: pd.DataFrame,
        output_dir: Path,
    ):
        """Save synthetic data to disk in canonical format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main tables
        cells.to_parquet(output_dir / "cells.parquet", index=False)
        neighborhoods.to_parquet(output_dir / "neighborhoods.parquet", index=False)
        stage_edges.to_parquet(output_dir / "stage_edges.parquet", index=False)

        # Generate split manifest (donor-held-out CV)
        splits = self._generate_splits(cells)
        with open(output_dir / "split_manifest.json", "w") as f:
            json.dump(splits, f, indent=2)

        # Save metadata
        metadata = {
            "n_cells": len(cells),
            "n_donors": cells["donor_id"].nunique(),
            "n_stages": len(self.stages),
            "stages": self.stages,
            "latent_dim": self.latent_dim,
            "n_celltypes": self.n_celltypes,
            "seed": self.seed,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_splits(self, cells: pd.DataFrame) -> Dict:
        """Generate donor-held-out cross-validation splits."""
        donors = sorted(cells["donor_id"].unique())
        n_donors = len(donors)
        n_folds = min(5, n_donors)  # 5-fold CV or fewer if not enough donors

        splits = {"folds": []}

        for fold_idx in range(n_folds):
            # Round-robin assignment
            test_start = fold_idx * (n_donors // n_folds)
            test_end = (fold_idx + 1) * (n_donors // n_folds)

            if fold_idx == n_folds - 1:
                test_end = n_donors  # Last fold gets remainder

            test_donors = donors[test_start:test_end]
            remaining = [d for d in donors if d not in test_donors]

            # 80-20 split of remaining for train/val
            n_val = max(1, len(remaining) // 5)
            val_donors = remaining[:n_val]
            train_donors = remaining[n_val:]

            splits["folds"].append({
                "fold": fold_idx,
                "train_donors": train_donors,
                "val_donors": val_donors,
                "test_donors": list(test_donors),
            })

        return splits


def generate_synthetic_dataset(
    output_dir: str = "data/processed/synthetic",
    n_cells: int = 1000,
    n_donors: int = 5,
    latent_dim: int = 2,
    noise_level: float = 0.1,
    niche_influence: float = 0.5,
    overlap: float = 0.2,
    seed: int = 42,
) -> Path:
    """
    Convenience function to generate and save synthetic dataset.

    Args:
        output_dir: Where to save generated data
        n_cells: Total number of cells
        n_donors: Number of synthetic donors
        latent_dim: Latent space dimensionality
        noise_level: Gaussian noise std for latent positions
        niche_influence: Strength of niche effect (0-1)
        overlap: Stage overlap in latent space (0-1)
        seed: Random seed

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)

    print("Generating synthetic dataset...")
    print(f"  n_cells: {n_cells}")
    print(f"  n_donors: {n_donors}")
    print(f"  latent_dim: {latent_dim}")
    print(f"  noise_level: {noise_level}")
    print(f"  niche_influence: {niche_influence}")
    print(f"  seed: {seed}")

    generator = SyntheticDataGenerator(
        n_cells=n_cells,
        n_donors=n_donors,
        latent_dim=latent_dim,
        seed=seed,
    )

    cells, neighborhoods, stage_edges = generator.generate(
        noise_level=noise_level,
        niche_influence=niche_influence,
        overlap=overlap,
    )

    print("\nGenerated:")
    print(f"  Cells: {len(cells)}")
    print(f"  Neighborhoods: {len(neighborhoods)}")
    print(f"  Stage edges: {len(stage_edges)}")
    print(f"  Stages: {cells['stage'].value_counts().to_dict()}")

    generator.save(cells, neighborhoods, stage_edges, output_path)

    print(f"\nSaved to: {output_path}")
    print("  cells.parquet")
    print("  neighborhoods.parquet")
    print("  stage_edges.parquet")
    print("  split_manifest.json")
    print("  metadata.json")

    return output_path


if __name__ == "__main__":
    # Generate default synthetic dataset
    output_dir = generate_synthetic_dataset()
    print(f"\n Synthetic dataset ready at: {output_dir}")
