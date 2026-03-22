"""
Synthetic data generator for StageBridge validation.

This module provides synthetic data generation for testing and validation:
- SyntheticDataGenerator: Simple generator for quick tests (original API)
- SyntheticDataGeneratorV2: Enhanced generator with full ground truth recovery
- generate_synthetic_dataset: Quick generation function (simple mode)
- generate_synthetic_v2: Enhanced generation function (full ground truth)

Ground truth suites (V2):
- Suite A: Branching cell-state transitions with known drift/diffusion
- Suite B: Recoverable niche influence (sender -> receiver causality)
- Suite C: Clone compatibility (matched > shuffled)
- Suite D: Semisynthetic spatial structure
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# =============================================================================
# V2 Enhanced Generator (Full Ground Truth Recovery)
# =============================================================================


@dataclass
class SyntheticConfig:
    """Configuration for enhanced synthetic data generation."""

    # Core dimensions
    n_cells: int = 2000
    n_donors: int = 10
    n_stages: int = 4
    latent_dim: int = 32
    n_genes: int = 2000
    n_celltypes: int = 8

    # Spatial structure
    k_neighbors: int = 20
    n_rings: int = 4
    spatial_scale: float = 1000.0

    # Clone/evolution
    n_clones_per_donor: int = 3
    clone_divergence: float = 0.3

    # Dynamics
    drift_strength: float = 1.0
    diffusion_strength: float = 0.2

    # Niche influence (Suite B)
    niche_influence_strength: float = 0.5
    n_influential_celltypes: int = 3

    # Difficulty
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Adjust parameters based on difficulty."""
        if self.difficulty == "easy":
            self.diffusion_strength = 0.1
            self.niche_influence_strength = 0.8
            self.clone_divergence = 0.5
        elif self.difficulty == "hard":
            self.diffusion_strength = 0.4
            self.niche_influence_strength = 0.3
            self.clone_divergence = 0.15


class SyntheticDataGeneratorV2:
    """
    Enhanced synthetic data generator with recoverable ground truth.

    Generates data that tests all StageBridge components:
    - Layer A: Dual-reference geometry (distinct HLCA/LuCA manifolds)
    - Layer B: Niche influence (causal sender->receiver effects)
    - Layer C: Hierarchical structure (cells->niches->samples)
    - Layer E: Transition dynamics (known flow field)
    - Layer F: Evolutionary compatibility (clone structure)
    """

    def __init__(self, config: SyntheticConfig | None = None):
        self.config = config or SyntheticConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Stage definitions
        self.stages = ["Normal", "Preneoplastic", "Invasive", "Advanced"][
            : self.config.n_stages
        ]
        self.stage_edges = [
            (self.stages[i], self.stages[i + 1]) for i in range(len(self.stages) - 1)
        ]

        # Cell type definitions with biological roles
        self.celltypes = [
            "AT2",
            "AT1",
            "Club",
            "Basal",
            "Fibroblast",
            "Macrophage",
            "T_cell",
            "Endothelial",
        ][: self.config.n_celltypes]

        # Define which cell types have causal influence on transitions
        self.influential_celltypes = ["Fibroblast", "Macrophage", "T_cell"][
            : self.config.n_influential_celltypes
        ]

        # Ground truth storage
        self.ground_truth: dict = {}

        # Initialize components
        self._init_influence_coefficients()
        self._init_gene_expression_basis()
        self._init_reference_geometry()
        self._init_flow_field()

    def _init_influence_coefficients(self):
        """Initialize causal influence coefficients (Suite B ground truth)."""
        self.influence_vectors = {}
        for ct in self.influential_celltypes:
            v = self.rng.standard_normal(self.config.latent_dim)
            v = v / np.linalg.norm(v)
            self.influence_vectors[ct] = v * self.config.niche_influence_strength

        self.ground_truth = {
            "influence_vectors": {k: v.tolist() for k, v in self.influence_vectors.items()},
            "influential_celltypes": self.influential_celltypes,
        }

    def _init_gene_expression_basis(self):
        """Initialize PCA-like basis for gene expression reconstruction."""
        self.gene_loadings = self.rng.standard_normal(
            (self.config.latent_dim, self.config.n_genes)
        )
        self.gene_loadings = self.gene_loadings / np.linalg.norm(
            self.gene_loadings, axis=0, keepdims=True
        )

        self.celltype_signatures = {}
        for ct in self.celltypes:
            sig = self.rng.standard_normal(self.config.n_genes) * 0.5
            self.celltype_signatures[ct] = sig

    def _init_reference_geometry(self):
        """Initialize distinct HLCA and LuCA reference geometries."""
        self.hlca_center = np.zeros(self.config.latent_dim)
        self.hlca_basis = np.eye(self.config.latent_dim)

        rotation_angle = np.pi / 6
        self.luca_rotation = np.eye(self.config.latent_dim)
        self.luca_rotation[0, 0] = np.cos(rotation_angle)
        self.luca_rotation[0, 1] = -np.sin(rotation_angle)
        self.luca_rotation[1, 0] = np.sin(rotation_angle)
        self.luca_rotation[1, 1] = np.cos(rotation_angle)

        self.luca_shift = np.zeros(self.config.latent_dim)
        self.luca_shift[0] = 2.0

        self.ground_truth["luca_rotation"] = self.luca_rotation[:2, :2].tolist()
        self.ground_truth["luca_shift"] = self.luca_shift[:4].tolist()

    def _init_flow_field(self):
        """Initialize stage transition flow field (Suite A ground truth)."""
        self.stage_centroids = {}
        base_trajectory = np.linspace(0, 3, self.config.n_stages)

        for i, stage in enumerate(self.stages):
            centroid = np.zeros(self.config.latent_dim)
            centroid[0] = base_trajectory[i]
            if i >= 2:
                centroid[1] = 0.5 * (i - 1)
            self.stage_centroids[stage] = centroid

        self.ground_truth["stage_centroids"] = {
            k: v[:4].tolist() for k, v in self.stage_centroids.items()
        }
        self.ground_truth["drift_strength"] = self.config.drift_strength
        self.ground_truth["diffusion_strength"] = self.config.diffusion_strength

    def generate(self) -> dict[str, pd.DataFrame]:
        """Generate complete synthetic dataset with ground truth."""
        donors, clones = self._generate_donors_and_clones()
        cells = self._generate_cells(donors, clones)
        cells = self._assign_spatial_coordinates(cells)
        cells = self._apply_niche_influence(cells)
        cells = self._compute_reference_projections(cells)
        cells = self._reconstruct_expression(cells)
        neighborhoods = self._build_neighborhoods(cells)
        stage_edges = self._generate_stage_edges()
        transitions = self._generate_transition_pairs(cells)

        return {
            "cells": cells,
            "neighborhoods": neighborhoods,
            "stage_edges": stage_edges,
            "transitions": transitions,
            "ground_truth": pd.DataFrame([self.ground_truth]),
        }

    def _generate_donors_and_clones(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate donor and clone structure (Suite C)."""
        donor_records = []
        clone_records = []

        for d in range(self.config.n_donors):
            donor_id = f"donor_{d:03d}"
            base_tmb = self.rng.gamma(shape=2.0, scale=2.0)
            smoking_exposure = self.rng.beta(a=2.0, b=3.0)

            donor_records.append({
                "donor_id": donor_id,
                "base_tmb": base_tmb,
                "smoking_exposure": smoking_exposure,
                "stage_distribution": self._donor_stage_distribution(d),
            })

            for c in range(self.config.n_clones_per_donor):
                clone_id = f"{donor_id}_clone_{c:02d}"
                clone_tmb = base_tmb * (1 + self.rng.normal(0, 0.2))
                clone_signature = self.rng.standard_normal(self.config.latent_dim)
                clone_signature = (
                    clone_signature
                    / np.linalg.norm(clone_signature)
                    * self.config.clone_divergence
                )

                clone_records.append({
                    "clone_id": clone_id,
                    "donor_id": donor_id,
                    "clone_idx": c,
                    "tmb": max(0, clone_tmb),
                    "signature": clone_signature.tolist(),
                    "is_dominant": c == 0,
                })

        return pd.DataFrame(donor_records), pd.DataFrame(clone_records)

    def _donor_stage_distribution(self, donor_idx: int) -> dict[str, float]:
        """Assign stage distribution based on donor index."""
        progression = donor_idx / max(1, self.config.n_donors - 1)
        probs = np.zeros(len(self.stages))
        for i, stage in enumerate(self.stages):
            stage_center = i / max(1, len(self.stages) - 1)
            probs[i] = np.exp(-3 * (progression - stage_center) ** 2)
        probs = probs / probs.sum()
        return {stage: float(probs[i]) for i, stage in enumerate(self.stages)}

    def _generate_cells(self, donors: pd.DataFrame, clones: pd.DataFrame) -> pd.DataFrame:
        """Generate cell-level data with latent coordinates."""
        records = []
        cell_id = 0
        cells_per_donor = self.config.n_cells // self.config.n_donors

        for _, donor in donors.iterrows():
            donor_id = donor["donor_id"]
            stage_probs = donor["stage_distribution"]
            donor_clones = clones[clones["donor_id"] == donor_id]

            for _ in range(cells_per_donor):
                stage = self.rng.choice(self.stages, p=[stage_probs[s] for s in self.stages])
                stage_idx = self.stages.index(stage)
                clone = donor_clones.sample(n=1, random_state=int(self.rng.integers(1e9))).iloc[0]

                centroid = self.stage_centroids[stage]
                z_base = centroid + self.rng.standard_normal(self.config.latent_dim) * self.config.diffusion_strength
                clone_sig = np.array(clone["signature"])
                z_with_clone = z_base + clone_sig
                celltype = self._sample_celltype(stage_idx)

                records.append({
                    "cell_id": f"cell_{cell_id:06d}",
                    "donor_id": donor_id,
                    "clone_id": clone["clone_id"],
                    "stage": stage,
                    "stage_idx": stage_idx,
                    "cell_type": celltype,
                    "z_base": z_base.tolist(),
                    "z_with_clone": z_with_clone.tolist(),
                    "tmb": clone["tmb"],
                    "smoking_exposure": donor["smoking_exposure"],
                })
                cell_id += 1

        return pd.DataFrame(records)

    def _sample_celltype(self, stage_idx: int) -> str:
        """Sample cell type with stage-dependent distribution."""
        probs = np.ones(len(self.celltypes))
        at2_idx = self.celltypes.index("AT2") if "AT2" in self.celltypes else 0
        probs[at2_idx] = 3.0 if stage_idx < 2 else 1.0

        for ct in ["Fibroblast", "Macrophage", "T_cell"]:
            if ct in self.celltypes:
                idx = self.celltypes.index(ct)
                probs[idx] = 1.0 + stage_idx * 0.5

        probs = probs / probs.sum()
        return self.rng.choice(self.celltypes, p=probs)

    def _assign_spatial_coordinates(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Assign spatial coordinates with clustering by stage/celltype."""
        cells = cells.copy()
        n = len(cells)
        scale = self.config.spatial_scale

        x = self.rng.uniform(0, scale, n)
        y = self.rng.uniform(0, scale, n)

        for stage_idx, stage in enumerate(self.stages):
            mask = cells["stage"] == stage
            n_stage = mask.sum()
            if n_stage > 0:
                offset_x = (stage_idx % 2) * scale * 0.3
                offset_y = (stage_idx // 2) * scale * 0.3
                x[mask] = self.rng.normal(scale * 0.5 + offset_x, scale * 0.2, n_stage).clip(0, scale)
                y[mask] = self.rng.normal(scale * 0.5 + offset_y, scale * 0.2, n_stage).clip(0, scale)

        cells["x_spatial"] = x
        cells["y_spatial"] = y
        return cells

    def _apply_niche_influence(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Apply causal niche influence to cell states (Suite B)."""
        cells = cells.copy()
        coords = cells[["x_spatial", "y_spatial"]].values
        distances = cdist(coords, coords)

        z_influenced = []
        niche_influence_scores = []

        for i in range(len(cells)):
            z_base = np.array(cells.iloc[i]["z_with_clone"])
            dists = distances[i]
            neighbor_mask = (dists > 0) & (dists < self.config.spatial_scale * 0.1)
            neighbor_indices = np.where(neighbor_mask)[0]

            influence_vec = np.zeros(self.config.latent_dim)
            influence_score = 0.0

            if len(neighbor_indices) > 0:
                neighbor_celltypes = cells.iloc[neighbor_indices]["cell_type"].values
                for ct in self.influential_celltypes:
                    ct_count = (neighbor_celltypes == ct).sum()
                    if ct_count > 0:
                        weight = ct_count / len(neighbor_indices)
                        influence_vec += self.influence_vectors[ct] * weight
                        influence_score += weight

            z_final = z_base + influence_vec
            z_influenced.append(z_final.tolist())
            niche_influence_scores.append(influence_score)

        cells["z_final"] = z_influenced
        cells["niche_influence_score"] = niche_influence_scores
        return cells

    def _compute_reference_projections(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Compute HLCA and LuCA reference projections."""
        cells = cells.copy()
        z_hlca_list, z_luca_list, z_fused_list = [], [], []

        for _, cell in cells.iterrows():
            z = np.array(cell["z_final"])
            z_hlca = z - self.hlca_center
            z_luca = self.luca_rotation @ z - self.luca_shift

            stage_idx = cell["stage_idx"]
            hlca_weight = 1.0 - stage_idx / max(1, len(self.stages) - 1)
            luca_weight = stage_idx / max(1, len(self.stages) - 1)
            z_fused = hlca_weight * z_hlca + luca_weight * z_luca

            z_hlca_list.append(z_hlca.tolist())
            z_luca_list.append(z_luca.tolist())
            z_fused_list.append(z_fused.tolist())

        cells["z_hlca"] = z_hlca_list
        cells["z_luca"] = z_luca_list
        cells["z_fused"] = z_fused_list

        # Create individual dimension columns for all latent dimensions
        for dim in range(self.config.latent_dim):
            cells[f"z_fused_{dim}"] = cells["z_fused"].apply(lambda x, d=dim: x[d])
            cells[f"z_hlca_{dim}"] = cells["z_hlca"].apply(lambda x, d=dim: x[d])
            cells[f"z_luca_{dim}"] = cells["z_luca"].apply(lambda x, d=dim: x[d])

        return cells

    def _reconstruct_expression(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct gene expression from latent coordinates."""
        cells = cells.copy()
        expressions = []

        for _, cell in cells.iterrows():
            z = np.array(cell["z_fused"])
            expr = z @ self.gene_loadings
            ct_sig = self.celltype_signatures.get(cell["cell_type"], np.zeros(self.config.n_genes))
            expr = expr + ct_sig
            expr = expr + self.rng.normal(0, 0.1, self.config.n_genes)
            expr = np.maximum(expr, 0)
            expressions.append(expr.tolist())

        cells["expression"] = expressions
        return cells

    def _build_neighborhoods(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Build 9-token neighborhood structure."""
        coords = cells[["x_spatial", "y_spatial"]].values
        distances = cdist(coords, coords)
        records = []

        for i in range(len(cells)):
            cell = cells.iloc[i]
            dists = distances[i]
            neighbor_order = np.argsort(dists)[1:self.config.k_neighbors + 1]

            tokens = [{"token_idx": 0, "token_type": "receiver", "cell_id": cell["cell_id"], "cell_type": cell["cell_type"], "z_fused": cell["z_fused"]}]

            cells_per_ring = self.config.k_neighbors // self.config.n_rings
            for ring in range(self.config.n_rings):
                start = ring * cells_per_ring
                end = (ring + 1) * cells_per_ring
                ring_indices = neighbor_order[start:end]

                if len(ring_indices) > 0:
                    ring_cells = cells.iloc[ring_indices]
                    z_pooled = np.mean([np.array(z) for z in ring_cells["z_fused"]], axis=0)
                    ct_counts = ring_cells["cell_type"].value_counts().to_dict()
                    ring_influence = sum(ct_counts.get(ct, 0) / len(ring_indices) for ct in self.influential_celltypes)
                    tokens.append({"token_idx": ring + 1, "token_type": f"ring_{ring + 1}", "z_pooled": z_pooled.tolist(), "celltype_composition": ct_counts, "n_cells": len(ring_indices), "mean_distance": float(dists[ring_indices].mean()), "ring_influence": ring_influence})
                else:
                    tokens.append({"token_idx": ring + 1, "token_type": f"ring_{ring + 1}", "z_pooled": [0.0] * self.config.latent_dim, "celltype_composition": {}, "n_cells": 0, "mean_distance": 0.0, "ring_influence": 0.0})

            tokens.append({"token_idx": 5, "token_type": "hlca", "z_hlca": cell["z_hlca"], "hlca_confidence": 1.0 - cell["stage_idx"] / len(self.stages)})
            tokens.append({"token_idx": 6, "token_type": "luca", "z_luca": cell["z_luca"], "luca_confidence": cell["stage_idx"] / len(self.stages)})

            neighbor_cells = cells.iloc[neighbor_order]
            caf_frac = (neighbor_cells["cell_type"] == "Fibroblast").mean() if "Fibroblast" in self.celltypes else 0
            immune_frac = (neighbor_cells["cell_type"].isin(["Macrophage", "T_cell"])).mean()
            tokens.append({"token_idx": 7, "token_type": "pathway", "emt_score": float(caf_frac * 0.6 + immune_frac * 0.4), "caf_fraction": float(caf_frac), "immune_fraction": float(immune_frac)})
            tokens.append({"token_idx": 8, "token_type": "stats", "n_neighbors": len(neighbor_order), "mean_distance": float(dists[neighbor_order].mean()), "std_distance": float(dists[neighbor_order].std()), "celltype_diversity": len(neighbor_cells["cell_type"].unique()), "stage_homogeneity": float((neighbor_cells["stage"] == cell["stage"]).mean())})

            records.append({"cell_id": cell["cell_id"], "donor_id": cell["donor_id"], "stage": cell["stage"], "tokens": tokens, "niche_influence_score": cell["niche_influence_score"], "total_ring_influence": sum(t.get("ring_influence", 0) for t in tokens if "ring" in t.get("token_type", ""))})

        return pd.DataFrame(records)

    def _generate_stage_edges(self) -> pd.DataFrame:
        """Generate stage transition graph."""
        records = []
        for source, target in self.stage_edges:
            src_idx = self.stages.index(source)
            tgt_idx = self.stages.index(target)
            src_centroid = self.stage_centroids[source]
            tgt_centroid = self.stage_centroids[target]
            drift = (tgt_centroid - src_centroid) * self.config.drift_strength

            records.append({"edge_id": f"{source}_to_{target}", "source_stage": source, "target_stage": target, "source_idx": src_idx, "target_idx": tgt_idx, "drift_vector": drift[:4].tolist(), "expected_distance": float(np.linalg.norm(tgt_centroid - src_centroid))})

        return pd.DataFrame(records)

    def _generate_transition_pairs(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Generate ground-truth transition pairs for training."""
        records = []
        pair_id = 0

        for source_stage, target_stage in self.stage_edges:
            source_cells = cells[cells["stage"] == source_stage]
            src_centroid = self.stage_centroids[source_stage]
            tgt_centroid = self.stage_centroids[target_stage]

            for _, src_cell in source_cells.iterrows():
                z_src = np.array(src_cell["z_fused"])
                direction = tgt_centroid - src_centroid
                direction = direction / np.linalg.norm(direction)
                drift = direction * self.config.drift_strength
                noise = self.rng.standard_normal(self.config.latent_dim) * self.config.diffusion_strength
                z_tgt = z_src + drift + noise

                records.append({"pair_id": f"pair_{pair_id:06d}", "source_cell_id": src_cell["cell_id"], "source_stage": source_stage, "target_stage": target_stage, "donor_id": src_cell["donor_id"], "clone_id": src_cell["clone_id"], "z_source": z_src.tolist(), "z_target": z_tgt.tolist(), "drift_applied": drift[:4].tolist(), "niche_influence": src_cell["niche_influence_score"]})
                pair_id += 1

        return pd.DataFrame(records)

    def save(self, data: dict[str, pd.DataFrame], output_dir: str | Path) -> Path:
        """Save generated data to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data["cells"].to_parquet(output_dir / "cells.parquet", index=False)
        data["neighborhoods"].to_parquet(output_dir / "neighborhoods.parquet", index=False)
        data["stage_edges"].to_parquet(output_dir / "stage_edges.parquet", index=False)
        data["transitions"].to_parquet(output_dir / "transitions.parquet", index=False)

        gt = self.ground_truth.copy()
        gt["config"] = {"n_cells": self.config.n_cells, "n_donors": self.config.n_donors, "latent_dim": self.config.latent_dim, "difficulty": self.config.difficulty, "niche_influence_strength": self.config.niche_influence_strength, "drift_strength": self.config.drift_strength, "diffusion_strength": self.config.diffusion_strength, "seed": self.config.seed}
        with open(output_dir / "ground_truth.json", "w") as f:
            json.dump(gt, f, indent=2)

        splits = self._generate_splits(data["cells"])
        with open(output_dir / "split_manifest.json", "w") as f:
            json.dump(splits, f, indent=2)

        metadata = {"n_cells": len(data["cells"]), "n_donors": data["cells"]["donor_id"].nunique(), "n_stages": len(self.stages), "stages": self.stages, "celltypes": self.celltypes, "influential_celltypes": self.influential_celltypes, "latent_dim": self.config.latent_dim, "n_genes": self.config.n_genes, "difficulty": self.config.difficulty, "seed": self.config.seed, "version": "v2"}
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return output_dir

    def _generate_splits(self, cells: pd.DataFrame) -> dict:
        """Generate donor-held-out cross-validation splits."""
        donors = sorted(cells["donor_id"].unique())
        n_donors = len(donors)
        n_folds = min(5, n_donors)
        folds = []
        fold_size = n_donors // n_folds

        for fold_idx in range(n_folds):
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_donors
            test_donors = donors[test_start:test_end]
            remaining = [d for d in donors if d not in test_donors]
            n_val = max(1, len(remaining) // 5)
            val_donors = remaining[:n_val]
            train_donors = remaining[n_val:]
            folds.append({"fold": fold_idx, "train_donors": train_donors, "val_donors": val_donors, "test_donors": list(test_donors)})

        return {"n_folds": n_folds, "folds": folds}


# =============================================================================
# V1 Simple Generator (Backward Compatibility)
# =============================================================================


class SyntheticDataGenerator:
    """
    Simple synthetic data generator for quick testing.

    This is the original V1 generator with basic features.
    For full ground truth recovery, use SyntheticDataGeneratorV2.
    """

    def __init__(
        self,
        n_cells: int = 1000,
        n_donors: int = 5,
        latent_dim: int = 2,
        n_celltypes: int = 8,
        seed: int = 42,
    ):
        self.n_cells = n_cells
        self.n_donors = n_donors
        self.latent_dim = latent_dim
        self.n_celltypes = n_celltypes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.stages = ["Normal", "Preneoplastic", "Invasive", "Advanced"]
        self.stage_edges = [
            ("Normal", "Preneoplastic"),
            ("Preneoplastic", "Invasive"),
            ("Invasive", "Advanced"),
        ]

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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate synthetic dataset (simple mode)."""
        cells = self._generate_cells(noise_level, overlap)
        neighborhoods = self._generate_neighborhoods(cells, niche_influence)
        stage_edges_df = self._generate_stage_edges()
        return cells, neighborhoods, stage_edges_df

    def _generate_cells(self, noise_level: float, overlap: float) -> pd.DataFrame:
        """Generate cell-level data."""
        cells_per_stage = self.n_cells // len(self.stages)
        records = []
        cell_id = 0

        for stage_idx, stage in enumerate(self.stages):
            centroid = self.stage_centroids[stage]
            if self.latent_dim > 2:
                centroid_expanded = np.zeros(self.latent_dim)
                centroid_expanded[:2] = centroid
            else:
                centroid_expanded = centroid

            stage_std = noise_level + overlap * 0.3
            z_positions = self.rng.normal(loc=centroid_expanded, scale=stage_std, size=(cells_per_stage, self.latent_dim))

            if stage_idx < len(self.stages) // 2:
                donor_pool = list(range(self.n_donors // 2 + 1))
            else:
                donor_pool = list(range(self.n_donors // 2, self.n_donors))

            donor_ids = self.rng.choice(donor_pool, size=cells_per_stage)
            tmb = self.rng.gamma(shape=2.0 + stage_idx, scale=1.0, size=cells_per_stage)
            smoking_sig = self.rng.beta(a=2.0 + stage_idx * 0.5, b=5.0 - stage_idx * 0.3, size=cells_per_stage)
            uv_sig = self.rng.beta(a=1.5, b=8.0, size=cells_per_stage)

            for i in range(cells_per_stage):
                records.append({
                    "cell_id": f"cell_{cell_id:06d}",
                    "donor_id": f"donor_{donor_ids[i]:02d}",
                    "stage": stage,
                    "stage_idx": stage_idx,
                    "z_fused": z_positions[i].tolist(),
                    "z_hlca": (z_positions[i] + self.rng.normal(0, 0.05, self.latent_dim)).tolist(),
                    "z_luca": (z_positions[i] + self.rng.normal(0, 0.05, self.latent_dim)).tolist(),
                    "cell_type": self._assign_celltype(stage_idx),
                    "tmb": tmb[i],
                    "smoking_signature": smoking_sig[i],
                    "uv_signature": uv_sig[i],
                    "x_spatial": self.rng.uniform(0, 1000),
                    "y_spatial": self.rng.uniform(0, 1000),
                })
                cell_id += 1

        df = pd.DataFrame(records)
        for dim in range(self.latent_dim):
            df[f"z_fused_{dim}"] = df["z_fused"].apply(lambda x, d=dim: x[d])
            df[f"z_hlca_{dim}"] = df["z_hlca"].apply(lambda x, d=dim: x[d])
            df[f"z_luca_{dim}"] = df["z_luca"].apply(lambda x, d=dim: x[d])

        return df

    def _assign_celltype(self, stage_idx: int) -> str:
        """Assign cell type with stage-dependent distribution."""
        celltypes = ["AT2", "AT1", "Club", "Basal", "Fibroblast", "Macrophage", "T_cell", "Endothelial"]
        if stage_idx < 2:
            probs = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02]
        else:
            probs = [0.2, 0.1, 0.05, 0.05, 0.25, 0.2, 0.1, 0.05]
        return self.rng.choice(celltypes, p=probs)

    def _generate_neighborhoods(self, cells: pd.DataFrame, niche_influence: float) -> pd.DataFrame:
        """Generate 9-token neighborhoods."""
        records = []
        for idx, cell in cells.iterrows():
            distances = np.sqrt((cells["x_spatial"] - cell["x_spatial"]) ** 2 + (cells["y_spatial"] - cell["y_spatial"]) ** 2)
            k_total = 20
            neighbor_indices = np.argsort(distances)[1:k_total + 1]

            tokens = [{"token_idx": 0, "token_type": "receiver", "cell_id": cell["cell_id"], "cell_type": cell["cell_type"], "z_fused": cell["z_fused"]}]

            cells_per_ring = 5
            for ring in range(4):
                start = ring * cells_per_ring
                end = (ring + 1) * cells_per_ring
                ring_cells = cells.iloc[neighbor_indices[start:end]]
                z_pooled = np.mean([z for z in ring_cells["z_fused"]], axis=0)
                celltype_counts = ring_cells["cell_type"].value_counts().to_dict()
                tokens.append({"token_idx": ring + 1, "token_type": f"ring_{ring + 1}", "z_pooled": z_pooled.tolist(), "celltype_composition": celltype_counts, "n_cells": len(ring_cells)})

            tokens.append({"token_idx": 5, "token_type": "hlca", "z_hlca": cell["z_hlca"]})
            tokens.append({"token_idx": 6, "token_type": "luca", "z_luca": cell["z_luca"]})

            neighbor_cells = cells.iloc[neighbor_indices]
            caf_frac = (neighbor_cells["cell_type"] == "Fibroblast").mean()
            immune_frac = (neighbor_cells["cell_type"].isin(["Macrophage", "T_cell"])).mean()
            pathway_score = niche_influence * (0.6 * caf_frac + 0.4 * immune_frac)
            tokens.append({"token_idx": 7, "token_type": "pathway", "emt_score": pathway_score, "caf_fraction": caf_frac, "immune_fraction": immune_frac})
            tokens.append({"token_idx": 8, "token_type": "stats", "n_neighbors": k_total, "mean_distance": distances[neighbor_indices].mean(), "diversity": len(neighbor_cells["cell_type"].unique())})

            records.append({"cell_id": cell["cell_id"], "donor_id": cell["donor_id"], "stage": cell["stage"], "tokens": tokens, "niche_influence": pathway_score})

        return pd.DataFrame(records)

    def _generate_stage_edges(self) -> pd.DataFrame:
        """Generate stage transition graph."""
        records = []
        for source, target in self.stage_edges:
            records.append({"edge_id": f"{source}_{target}", "source_stage": source, "target_stage": target, "source_idx": self.stages.index(source), "target_idx": self.stages.index(target), "is_forward": True, "pseudotime_delta": 1.0})
        return pd.DataFrame(records)

    def save(self, cells: pd.DataFrame, neighborhoods: pd.DataFrame, stage_edges: pd.DataFrame, output_dir: Path):
        """Save synthetic data to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cells.to_parquet(output_dir / "cells.parquet", index=False)
        neighborhoods.to_parquet(output_dir / "neighborhoods.parquet", index=False)
        stage_edges.to_parquet(output_dir / "stage_edges.parquet", index=False)

        splits = self._generate_splits(cells)
        with open(output_dir / "split_manifest.json", "w") as f:
            json.dump(splits, f, indent=2)

        metadata = {"n_cells": len(cells), "n_donors": cells["donor_id"].nunique(), "n_stages": len(self.stages), "stages": self.stages, "latent_dim": self.latent_dim, "n_celltypes": self.n_celltypes, "seed": self.seed}
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_splits(self, cells: pd.DataFrame) -> dict:
        """Generate donor-held-out cross-validation splits."""
        donors = sorted(cells["donor_id"].unique())
        n_donors = len(donors)
        n_folds = min(5, n_donors)
        splits = {"folds": []}

        for fold_idx in range(n_folds):
            test_start = fold_idx * (n_donors // n_folds)
            test_end = (fold_idx + 1) * (n_donors // n_folds)
            if fold_idx == n_folds - 1:
                test_end = n_donors

            test_donors = donors[test_start:test_end]
            remaining = [d for d in donors if d not in test_donors]
            n_val = max(1, len(remaining) // 5)
            val_donors = remaining[:n_val]
            train_donors = remaining[n_val:]

            splits["folds"].append({"fold": fold_idx, "train_donors": train_donors, "val_donors": val_donors, "test_donors": list(test_donors)})

        return splits


# =============================================================================
# Convenience Functions
# =============================================================================


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
    """Generate and save simple synthetic dataset (backward compatible)."""
    output_path = Path(output_dir)

    print("Generating synthetic dataset...")
    print(f"  n_cells: {n_cells}")
    print(f"  n_donors: {n_donors}")
    print(f"  latent_dim: {latent_dim}")
    print(f"  seed: {seed}")

    generator = SyntheticDataGenerator(n_cells=n_cells, n_donors=n_donors, latent_dim=latent_dim, seed=seed)
    cells, neighborhoods, stage_edges = generator.generate(noise_level=noise_level, niche_influence=niche_influence, overlap=overlap)

    print(f"\nGenerated: {len(cells)} cells, {len(neighborhoods)} neighborhoods")
    generator.save(cells, neighborhoods, stage_edges, output_path)
    print(f"Saved to: {output_path}")

    return output_path


def generate_synthetic_v2(
    output_dir: str = "data/processed/synthetic_v2",
    n_cells: int = 2000,
    n_donors: int = 10,
    latent_dim: int = 32,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    seed: int = 42,
) -> Path:
    """Generate and save enhanced synthetic dataset with ground truth."""
    config = SyntheticConfig(n_cells=n_cells, n_donors=n_donors, latent_dim=latent_dim, difficulty=difficulty, seed=seed)

    print("=" * 60)
    print("SYNTHETIC DATA GENERATOR V2")
    print("=" * 60)
    print(f"  n_cells: {config.n_cells}")
    print(f"  n_donors: {config.n_donors}")
    print(f"  latent_dim: {config.latent_dim}")
    print(f"  difficulty: {config.difficulty}")
    print(f"  niche_influence_strength: {config.niche_influence_strength}")
    print(f"  seed: {config.seed}")

    generator = SyntheticDataGeneratorV2(config)
    data = generator.generate()

    print(f"\nGenerated: {len(data['cells'])} cells, {len(data['transitions'])} transitions")
    output_path = generator.save(data, output_dir)
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    # Generate datasets at all difficulty levels
    for diff in ["easy", "medium", "hard"]:
        generate_synthetic_v2(output_dir=f"data/processed/synthetic_v2_{diff}", difficulty=diff)
