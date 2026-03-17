"""
Unified benchmark generator for StageBridge evaluation.

Consolidates synthetic_v2 and semi-synthetic into a single system that:
- Supports fully synthetic, semi-synthetic, and hybrid modes
- Generates recoverable ground truth for all four suites (A, B, C, D)
- Produces world-based train/val/test splits
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from stagebridge.logging_utils import get_logger
from stagebridge.benchmarks.unified.config import (
    UnifiedBenchmarkConfig,
    SmokeTestConfig,
    CellGroupSpec,
)
from stagebridge.benchmarks.unified.ground_truth import (
    GroundTruth,
    build_ground_truth_from_config,
)

log = get_logger(__name__)


@dataclass
class CellPool:
    """Pool of cells available for sampling."""

    group_name: str
    cells: pd.DataFrame
    source: str
    role: str


@dataclass
class SyntheticWorld:
    """A single synthetic spatial world."""

    world_id: str
    split: str
    seed: int
    cell_positions: pd.DataFrame
    metadata: dict[str, Any]


@dataclass
class GenerationReport:
    """Report on benchmark generation."""

    config_name: str
    mode: str
    n_cells_generated: int = 0
    n_worlds: dict[str, int] = field(default_factory=dict)
    cell_pools: dict[str, int] = field(default_factory=dict)
    interaction_summary: dict[str, Any] = field(default_factory=dict)
    output_paths: list[str] = field(default_factory=list)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "mode": self.mode,
            "n_cells_generated": self.n_cells_generated,
            "n_worlds": self.n_worlds,
            "cell_pools": self.cell_pools,
            "interaction_summary": self.interaction_summary,
            "output_paths": self.output_paths,
            "success": self.success,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class UnifiedBenchmarkGenerator:
    """Generate unified benchmarks with full ground truth tracking.

    Supports three modes:
    - fully_synthetic: All data is synthetic, fast generation
    - semi_synthetic: Real expression profiles, synthetic spatial
    - hybrid: Real profiles with causal niche dynamics applied
    """

    def __init__(self, config: UnifiedBenchmarkConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Initialize ground truth
        self.ground_truth = build_ground_truth_from_config(config, self.rng)

        # Cell pools and worlds
        self.cell_pools: dict[str, CellPool] = {}
        self.worlds: dict[str, list[SyntheticWorld]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        # Gene expression basis (for synthetic mode)
        self._gene_loadings: np.ndarray | None = None
        self._celltype_signatures: dict[str, np.ndarray] = {}

        # Reference geometry
        self._init_reference_geometry()

    def _init_reference_geometry(self) -> None:
        """Initialize HLCA/LuCA reference geometry."""
        self.hlca_center = np.zeros(self.config.latent_dim)
        self.hlca_basis = np.eye(self.config.latent_dim)

        rotation_angle = self.config.dynamics.hlca_luca_rotation
        self.luca_rotation = np.eye(self.config.latent_dim)
        self.luca_rotation[0, 0] = np.cos(rotation_angle)
        self.luca_rotation[0, 1] = -np.sin(rotation_angle)
        self.luca_rotation[1, 0] = np.sin(rotation_angle)
        self.luca_rotation[1, 1] = np.cos(rotation_angle)

        self.luca_shift = np.zeros(self.config.latent_dim)
        self.luca_shift[0] = self.config.dynamics.hlca_luca_shift

    def generate(self, use_fallback: bool = True) -> GenerationReport:
        """Generate the complete benchmark.

        Args:
            use_fallback: Use synthetic fallback if real data unavailable

        Returns:
            GenerationReport with generation details
        """
        report = GenerationReport(
            config_name=self.config.benchmark_name,
            mode=self.config.mode,
        )

        try:
            # Step 1: Build cell pools
            log.info("Step 1: Building cell pools (mode=%s)...", self.config.mode)
            self._build_cell_pools(use_fallback)
            for name, pool in self.cell_pools.items():
                report.cell_pools[name] = len(pool.cells)

            # Step 2: Generate worlds
            log.info("Step 2: Generating synthetic worlds...")
            self._generate_all_worlds()
            for split, worlds in self.worlds.items():
                report.n_worlds[split] = len(worlds)

            # Step 3: Apply causal dynamics (if applicable)
            if self.config.applies_causal_dynamics:
                log.info("Step 3: Applying causal niche dynamics...")
                self._apply_causal_dynamics()

            # Step 4: Apply interaction rules
            log.info("Step 4: Applying interaction rules...")
            interaction_stats = self._apply_interaction_rules()
            report.interaction_summary = interaction_stats

            # Step 5: Compute reference projections
            log.info("Step 5: Computing reference projections...")
            self._compute_reference_projections()

            # Step 6: Generate expression (if synthetic)
            if self.config.is_synthetic:
                log.info("Step 6: Generating gene expression...")
                self._generate_expression()

            # Step 7: Build neighborhoods
            log.info("Step 7: Building neighborhoods...")
            self._build_neighborhoods()

            # Step 8: Export
            log.info("Step 8: Exporting benchmark...")
            output_paths = self._export_benchmark()
            report.output_paths = [str(p) for p in output_paths]

            # Count total cells
            for split, worlds in self.worlds.items():
                for world in worlds:
                    report.n_cells_generated += len(world.cell_positions)

            report.success = True
            log.info("Benchmark generation complete: %d cells", report.n_cells_generated)

        except Exception as e:
            log.error("Generation failed: %s", e)
            raise

        return report

    def _build_cell_pools(self, use_fallback: bool) -> None:
        """Build cell pools from real data or synthetic fallback."""
        for group_spec in self.config.cell_groups:
            if self.config.is_synthetic or (use_fallback and not self._try_load_real_data(group_spec)):
                pool = self._create_synthetic_pool(group_spec)
            else:
                pool = self._load_real_pool(group_spec)

            if pool is not None:
                self.cell_pools[group_spec.name] = pool
                log.info(
                    "Built cell pool '%s': %d cells from %s",
                    group_spec.name,
                    len(pool.cells),
                    pool.source,
                )

    def _try_load_real_data(self, spec: CellGroupSpec) -> bool:
        """Check if real data is available for this group."""
        # Check if any source path exists
        for source in spec.source_datasets:
            if source == "hlca" and self.config.hlca_path and self.config.hlca_path.exists():
                return True
            if source == "luca" and self.config.luca_path and self.config.luca_path.exists():
                return True
            if source == "progression" and self.config.progression_path and self.config.progression_path.exists():
                return True
        return False

    def _load_real_pool(self, spec: CellGroupSpec) -> CellPool | None:
        """Load cell pool from real data sources."""
        # This would use DataSourceLoader from semi_synthetic
        # For now, fall back to synthetic
        log.warning("Real data loading not implemented, using synthetic fallback")
        return self._create_synthetic_pool(spec)

    def _create_synthetic_pool(self, spec: CellGroupSpec) -> CellPool:
        """Create a fully synthetic cell pool."""
        n_cells = self.rng.integers(spec.min_cells, spec.max_cells + 1)

        # Generate base latent positions
        # Position depends on cell group role
        z_base = self.rng.standard_normal((n_cells, self.config.latent_dim))

        # Add role-specific bias
        if spec.latent_position_bias is not None:
            bias = np.array(spec.latent_position_bias)
            z_base = z_base + bias[:self.config.latent_dim]
        elif spec.role == "sender":
            # Senders slightly offset in latent space
            z_base[:, 1] += 0.5

        # Sample stages
        stages = self.rng.choice(self.config.stages, size=n_cells)

        # Build DataFrame
        cells = pd.DataFrame({
            "cell_id": [f"{spec.name}_{i:06d}" for i in range(n_cells)],
            "cell_group": spec.name,
            "role": spec.role,
            "cell_type": spec.base_expression_profile or spec.name.split("_")[0],
            "stage": stages,
        })

        # Store latent positions
        cells["z_base"] = [z.tolist() for z in z_base]

        return CellPool(
            group_name=spec.name,
            cells=cells,
            source="synthetic",
            role=spec.role,
        )

    def _generate_all_worlds(self) -> None:
        """Generate all synthetic worlds for train/val/test splits."""
        # Train worlds
        self.worlds["train"] = self._generate_worlds(
            n_worlds=self.config.n_worlds_train,
            split="train",
            base_seed=self.config.seed,
        )

        # Validation worlds (different seed range)
        self.worlds["val"] = self._generate_worlds(
            n_worlds=self.config.n_worlds_val,
            split="val",
            base_seed=self.config.seed + 100000,
        )

        # Test worlds (another seed range)
        self.worlds["test"] = self._generate_worlds(
            n_worlds=self.config.n_worlds_test,
            split="test",
            base_seed=self.config.seed + 200000,
        )

    def _generate_worlds(
        self,
        n_worlds: int,
        split: str,
        base_seed: int,
    ) -> list[SyntheticWorld]:
        """Generate multiple worlds for a split."""
        worlds = []

        for world_idx in range(n_worlds):
            world_seed = base_seed + world_idx * 1000
            world_rng = np.random.default_rng(world_seed)

            world_id = f"{split}_world_{world_idx:04d}"

            # Sample cells from pools
            all_cells = []
            cells_per_group = self.config.cells_per_world // len(self.cell_pools)

            for pool_name, pool in self.cell_pools.items():
                n_sample = min(cells_per_group, len(pool.cells))
                indices = world_rng.choice(len(pool.cells), size=n_sample, replace=True)
                sampled = pool.cells.iloc[indices].copy()
                sampled = sampled.reset_index(drop=True)

                # Generate unique cell IDs for this world
                sampled["synthetic_cell_id"] = [
                    f"{world_id}_{pool_name}_{i:06d}"
                    for i in range(len(sampled))
                ]

                all_cells.append(sampled)

            # Combine all cells
            cell_positions = pd.concat(all_cells, ignore_index=True)

            # Assign spatial coordinates
            n = len(cell_positions)
            width = self.config.world_width
            height = self.config.world_height

            # Add clustering by stage/role
            x = np.zeros(n)
            y = np.zeros(n)

            for i, (_, row) in enumerate(cell_positions.iterrows()):
                stage_idx = self.config.stages.index(row["stage"]) if row["stage"] in self.config.stages else 0
                role = row.get("role", "background")

                # Position based on stage and role
                if role == "receiver":
                    # Receivers clustered in center
                    center_x = width * 0.5
                    center_y = height * 0.5
                elif role == "sender":
                    # Senders distributed around receivers
                    center_x = width * 0.5 + world_rng.uniform(-0.2, 0.2) * width
                    center_y = height * 0.5 + world_rng.uniform(-0.2, 0.2) * height
                else:
                    # Background scattered
                    center_x = world_rng.uniform(0.2, 0.8) * width
                    center_y = world_rng.uniform(0.2, 0.8) * height

                # Add stage-based offset
                stage_offset_x = (stage_idx % 3) * width * 0.1
                stage_offset_y = (stage_idx // 3) * height * 0.1

                x[i] = np.clip(
                    world_rng.normal(center_x + stage_offset_x, width * 0.15),
                    0,
                    width,
                )
                y[i] = np.clip(
                    world_rng.normal(center_y + stage_offset_y, height * 0.15),
                    0,
                    height,
                )

            cell_positions["x"] = x
            cell_positions["y"] = y

            # Create world
            world = SyntheticWorld(
                world_id=world_id,
                split=split,
                seed=world_seed,
                cell_positions=cell_positions,
                metadata={
                    "width": width,
                    "height": height,
                    "n_cells": n,
                    "seed": world_seed,
                    "split": split,
                },
            )

            worlds.append(world)

        return worlds

    def _apply_causal_dynamics(self) -> None:
        """Apply causal niche influence to cell states (Suite B)."""
        for split, worlds in self.worlds.items():
            for world in worlds:
                self._apply_niche_influence_to_world(world)

    def _apply_niche_influence_to_world(self, world: SyntheticWorld) -> None:
        """Apply niche influence vectors to cells in a world."""
        cells = world.cell_positions
        coords = cells[["x", "y"]].values
        groups = cells["cell_group"].values

        # Compute pairwise distances
        distances = cdist(coords, coords)

        # Initialize influenced latent positions
        z_influenced_list = []
        influence_scores = []

        for idx in range(len(cells)):
            z_base = np.array(cells.iloc[idx]["z_base"])
            cell_stage = cells.iloc[idx].get("stage", "Normal")

            # Compute influence from senders
            influence_vec = np.zeros(self.config.latent_dim)
            total_influence = 0.0

            for rule in self.config.interaction_rules:
                if rule.niche_influence is None:
                    continue

                # Find senders within radius
                sender_mask = (groups == rule.sender_group)
                dists = distances[idx]
                within_radius = (dists <= rule.interaction_radius) & (dists > 0)
                nearby_senders = sender_mask & within_radius
                n_senders = nearby_senders.sum()

                if n_senders > 0:
                    # Get influence direction from ground truth
                    influence_name = rule.niche_influence.influence_name
                    if influence_name in self.ground_truth.influence_vectors:
                        direction = np.array(self.ground_truth.influence_vectors[influence_name])

                        # Weight by sender count and stage
                        effective_strength = rule.niche_influence.get_effective_strength(cell_stage)
                        weight = effective_strength * (1 - np.exp(-n_senders / 2))

                        influence_vec += direction * weight
                        total_influence += weight

            # Apply influence
            z_final = z_base + influence_vec
            z_influenced_list.append(z_final.tolist())
            influence_scores.append(total_influence)

        # Update cells
        cells["z_influenced"] = z_influenced_list
        cells["niche_influence_score"] = influence_scores
        world.cell_positions = cells

    def _apply_interaction_rules(self) -> dict[str, Any]:
        """Apply interaction rules to determine interacting state."""
        total_receivers = 0
        total_interacting = 0
        rule_counts: dict[str, int] = {}

        for split, worlds in self.worlds.items():
            for world in worlds:
                stats = self._apply_rules_to_world(world)
                total_receivers += stats["n_receivers"]
                total_interacting += stats["n_interacting"]
                for rule_id, count in stats.get("rule_counts", {}).items():
                    rule_counts[rule_id] = rule_counts.get(rule_id, 0) + count

        return {
            "total_receivers": total_receivers,
            "total_interacting": total_interacting,
            "interaction_rate": total_interacting / max(1, total_receivers),
            "rule_counts": rule_counts,
        }

    def _apply_rules_to_world(self, world: SyntheticWorld) -> dict[str, Any]:
        """Apply interaction rules to a single world."""
        cells = world.cell_positions
        coords = cells[["x", "y"]].values
        groups = cells["cell_group"].values

        world_rng = np.random.default_rng(world.seed + 1)

        # Initialize columns
        cells["is_interacting"] = False
        cells["triggered_rules"] = ""
        cells["interaction_strength"] = 0.0
        cells["n_senders"] = 0

        n_receivers = 0
        n_interacting = 0
        rule_counts: dict[str, int] = {}

        receiver_groups = {
            rule.receiver_group
            for rule in self.config.interaction_rules
        }

        for idx in range(len(cells)):
            cell_group = groups[idx]

            if cell_group not in receiver_groups:
                continue

            n_receivers += 1
            cell_stage = cells.iloc[idx].get("stage", "Normal")
            triggered = []
            max_strength = 0.0
            total_senders = 0

            for rule in self.config.interaction_rules:
                if rule.receiver_group != cell_group:
                    continue

                # Count senders within radius
                dists = np.sqrt(((coords - coords[idx]) ** 2).sum(axis=1))
                sender_mask = (groups == rule.sender_group) & (dists <= rule.interaction_radius) & (dists > 0)
                n_senders = sender_mask.sum()

                if n_senders > 0:
                    total_senders += n_senders

                    # Compute interaction probability
                    effect_strength = rule.get_stage_effect(cell_stage)
                    interaction_prob = effect_strength * (1 - np.exp(-n_senders / 2))

                    # Stochastic determination
                    if world_rng.random() < interaction_prob:
                        triggered.append(rule.rule_id)
                        rule_counts[rule.rule_id] = rule_counts.get(rule.rule_id, 0) + 1
                        if interaction_prob > max_strength:
                            max_strength = interaction_prob

            # Update cell
            is_interacting = len(triggered) > 0
            cells.at[cells.index[idx], "is_interacting"] = is_interacting
            cells.at[cells.index[idx], "triggered_rules"] = ",".join(triggered)
            cells.at[cells.index[idx], "interaction_strength"] = max_strength
            cells.at[cells.index[idx], "n_senders"] = total_senders

            if is_interacting:
                n_interacting += 1

        # Compute ground truth labels (deterministic)
        cells = self._compute_ground_truth_labels(cells, coords, groups)

        world.cell_positions = cells

        return {
            "n_receivers": n_receivers,
            "n_interacting": n_interacting,
            "rule_counts": rule_counts,
        }

    def _compute_ground_truth_labels(
        self,
        cells: pd.DataFrame,
        coords: np.ndarray,
        groups: np.ndarray,
    ) -> pd.DataFrame:
        """Compute deterministic ground truth labels."""
        for rule in self.config.interaction_rules:
            col_name = f"gt_{rule.rule_id}_strength"
            strengths = np.zeros(len(cells))

            for idx in range(len(cells)):
                if groups[idx] != rule.receiver_group:
                    continue

                dists = np.sqrt(((coords - coords[idx]) ** 2).sum(axis=1))
                sender_mask = (groups == rule.sender_group) & (dists <= rule.interaction_radius) & (dists > 0)
                n_senders = sender_mask.sum()

                if n_senders > 0:
                    cell_stage = cells.iloc[idx].get("stage", "Normal")
                    effect = rule.get_stage_effect(cell_stage)
                    strengths[idx] = effect * (1 - np.exp(-n_senders / 2))

            cells[col_name] = strengths

        # Aggregate
        gt_cols = [c for c in cells.columns if c.startswith("gt_") and c.endswith("_strength")]
        if gt_cols:
            cells["gt_max_strength"] = cells[gt_cols].max(axis=1)
            cells["gt_should_interact"] = cells["gt_max_strength"] > 0.3

        return cells

    def _compute_reference_projections(self) -> None:
        """Compute HLCA and LuCA reference projections."""
        for split, worlds in self.worlds.items():
            for world in worlds:
                cells = world.cell_positions

                z_hlca_list = []
                z_luca_list = []
                z_fused_list = []

                z_col = "z_influenced" if "z_influenced" in cells.columns else "z_base"

                for _, cell in cells.iterrows():
                    z = np.array(cell[z_col])

                    # HLCA projection
                    z_hlca = z - self.hlca_center

                    # LuCA projection
                    z_luca = self.luca_rotation @ z - self.luca_shift

                    # Fused (stage-weighted)
                    stage = cell.get("stage", "Normal")
                    stage_idx = self.config.stages.index(stage) if stage in self.config.stages else 0
                    hlca_weight = 1.0 - stage_idx / max(1, len(self.config.stages) - 1)
                    luca_weight = stage_idx / max(1, len(self.config.stages) - 1)

                    z_fused = hlca_weight * z_hlca + luca_weight * z_luca

                    z_hlca_list.append(z_hlca.tolist())
                    z_luca_list.append(z_luca.tolist())
                    z_fused_list.append(z_fused.tolist())

                cells["z_hlca"] = z_hlca_list
                cells["z_luca"] = z_luca_list
                cells["z_fused"] = z_fused_list

                world.cell_positions = cells

    def _generate_expression(self) -> None:
        """Generate synthetic gene expression from latent positions."""
        # Initialize gene loadings
        if self._gene_loadings is None:
            self._gene_loadings = self.rng.standard_normal(
                (self.config.latent_dim, self.config.n_genes)
            )
            self._gene_loadings = self._gene_loadings / np.linalg.norm(
                self._gene_loadings, axis=0, keepdims=True
            )

        # Cell type signatures
        cell_types = set()
        for pool in self.cell_pools.values():
            cell_types.update(pool.cells["cell_type"].unique())

        for ct in cell_types:
            if ct not in self._celltype_signatures:
                self._celltype_signatures[ct] = self.rng.standard_normal(self.config.n_genes) * 0.5

        # Generate expression for each world
        for split, worlds in self.worlds.items():
            for world in worlds:
                cells = world.cell_positions
                expressions = []

                for _, cell in cells.iterrows():
                    z = np.array(cell.get("z_fused", cell.get("z_base", [0] * self.config.latent_dim)))

                    # Base expression from latent
                    expr = z @ self._gene_loadings

                    # Add cell type signature
                    ct = cell.get("cell_type", "unknown")
                    if ct in self._celltype_signatures:
                        expr = expr + self._celltype_signatures[ct]

                    # Add noise
                    expr = expr + self.rng.normal(0, 0.1, self.config.n_genes)
                    expr = np.maximum(expr, 0)

                    expressions.append(expr.tolist())

                cells["expression"] = expressions
                world.cell_positions = cells

    def _build_neighborhoods(self) -> None:
        """Build 9-token neighborhood structure for each cell."""
        for split, worlds in self.worlds.items():
            for world in worlds:
                self._build_world_neighborhoods(world)

    def _build_world_neighborhoods(self, world: SyntheticWorld) -> None:
        """Build neighborhoods for a single world."""
        cells = world.cell_positions
        coords = cells[["x", "y"]].values
        distances = cdist(coords, coords)

        neighborhoods = []

        for i in range(len(cells)):
            cell = cells.iloc[i]

            # Find k nearest neighbors
            dists = distances[i]
            neighbor_order = np.argsort(dists)[1:self.config.k_neighbors + 1]

            # Build 9 tokens
            tokens = []

            # Token 0: Receiver
            tokens.append({
                "token_idx": 0,
                "token_type": "receiver",
                "cell_id": cell.get("synthetic_cell_id", cell.get("cell_id")),
            })

            # Tokens 1-4: Spatial rings
            cells_per_ring = self.config.k_neighbors // self.config.n_rings
            for ring in range(self.config.n_rings):
                start = ring * cells_per_ring
                end = (ring + 1) * cells_per_ring
                ring_indices = neighbor_order[start:end]

                if len(ring_indices) > 0:
                    ring_cells = cells.iloc[ring_indices]
                    ct_counts = ring_cells["cell_group"].value_counts().to_dict()

                    tokens.append({
                        "token_idx": ring + 1,
                        "token_type": f"ring_{ring + 1}",
                        "celltype_composition": ct_counts,
                        "n_cells": len(ring_indices),
                        "mean_distance": float(dists[ring_indices].mean()),
                    })
                else:
                    tokens.append({
                        "token_idx": ring + 1,
                        "token_type": f"ring_{ring + 1}",
                        "n_cells": 0,
                    })

            # Token 5: HLCA context
            tokens.append({
                "token_idx": 5,
                "token_type": "hlca",
            })

            # Token 6: LuCA context
            tokens.append({
                "token_idx": 6,
                "token_type": "luca",
            })

            # Token 7: Pathway context
            if len(neighbor_order) > 0:
                neighbor_cells = cells.iloc[neighbor_order]
                caf_frac = (neighbor_cells["cell_group"] == "caf_sender").mean()
                immune_frac = (neighbor_cells["cell_group"] == "immune_sender").mean()
            else:
                caf_frac = 0.0
                immune_frac = 0.0

            tokens.append({
                "token_idx": 7,
                "token_type": "pathway",
                "caf_fraction": float(caf_frac),
                "immune_fraction": float(immune_frac),
            })

            # Token 8: Statistics
            tokens.append({
                "token_idx": 8,
                "token_type": "stats",
                "n_neighbors": len(neighbor_order),
            })

            neighborhoods.append({
                "cell_id": cell.get("synthetic_cell_id", cell.get("cell_id")),
                "tokens": tokens,
            })

        world.neighborhoods = pd.DataFrame(neighborhoods)

    def _export_benchmark(self) -> list[Path]:
        """Export benchmark to disk."""
        output_dir = self.config.output_dir / self.config.benchmark_name
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_paths = []

        # Export manifest
        manifest = {
            "benchmark_name": self.config.benchmark_name,
            "mode": self.config.mode,
            "difficulty": self.config.difficulty,
            "stages": self.config.stages,
            "latent_dim": self.config.latent_dim,
            "n_hvg": self.config.n_hvg,
            "splits": {split: len(worlds) for split, worlds in self.worlds.items()},
        }

        manifest_path = output_dir / "benchmark_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        exported_paths.append(manifest_path)

        # Export ground truth
        gt_path = output_dir / "ground_truth.json"
        self.ground_truth.save(gt_path)
        exported_paths.append(gt_path)

        # Export each world
        for split, worlds in self.worlds.items():
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for world in worlds:
                world_dir = split_dir / world.world_id
                world_dir.mkdir(parents=True, exist_ok=True)

                # Export cells
                cells_path = world_dir / "cells.parquet"
                # Convert list columns to JSON strings for parquet compatibility
                cells_df = world.cell_positions.copy()
                for col in cells_df.columns:
                    if cells_df[col].apply(lambda x: isinstance(x, list)).any():
                        cells_df[col] = cells_df[col].apply(json.dumps)
                cells_df.to_parquet(cells_path, index=False)
                exported_paths.append(cells_path)

                # Export coordinates
                coords_path = world_dir / "coordinates.parquet"
                world.cell_positions[["synthetic_cell_id", "x", "y"]].to_parquet(
                    coords_path, index=False
                )
                exported_paths.append(coords_path)

                # Export ground truth labels
                gt_cols = ["synthetic_cell_id", "cell_group", "is_interacting", "interaction_strength", "niche_influence_score"]
                gt_cols.extend([c for c in world.cell_positions.columns if c.startswith("gt_")])
                gt_cols = [c for c in gt_cols if c in world.cell_positions.columns]

                gt_path = world_dir / "ground_truth.parquet"
                world.cell_positions[gt_cols].to_parquet(gt_path, index=False)
                exported_paths.append(gt_path)

                # Export neighborhoods if available
                if hasattr(world, "neighborhoods") and world.neighborhoods is not None:
                    nb_path = world_dir / "neighborhoods.parquet"
                    nb_df = world.neighborhoods.copy()
                    nb_df["tokens"] = nb_df["tokens"].apply(json.dumps)
                    nb_df.to_parquet(nb_path, index=False)
                    exported_paths.append(nb_path)

                # Export metadata
                meta_path = world_dir / "world_metadata.json"
                with open(meta_path, "w") as f:
                    json.dump(world.metadata, f, indent=2)
                exported_paths.append(meta_path)

        return exported_paths


def generate_benchmark(
    config: UnifiedBenchmarkConfig | None = None,
    mode: str | None = None,
    smoke: bool = False,
    use_fallback: bool = True,
) -> GenerationReport:
    """Convenience function to generate a benchmark.

    Args:
        config: Benchmark configuration (uses default if None)
        mode: Generation mode override ("fully_synthetic", "semi_synthetic", "hybrid")
        smoke: If True, use smoke test configuration
        use_fallback: If True, use fallback data when real data unavailable

    Returns:
        GenerationReport
    """
    if config is None:
        config = SmokeTestConfig() if smoke else UnifiedBenchmarkConfig()

    if mode is not None:
        config.mode = mode

    generator = UnifiedBenchmarkGenerator(config)
    return generator.generate(use_fallback=use_fallback)
