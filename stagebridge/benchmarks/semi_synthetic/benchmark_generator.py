"""
Main benchmark generator for semi-synthetic StageBridge evaluation.

Orchestrates data loading, cell selection, world generation, and
interaction rule application to create complete benchmark datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger
from stagebridge.benchmarks.semi_synthetic.configs import (
    BenchmarkConfig,
    CellGroupSpec,
    SmokeConfig,
)
from stagebridge.benchmarks.semi_synthetic.data_sources import (
    DataSourceLoader,
    DataSourceReport,
    create_fallback_data,
)
from stagebridge.benchmarks.semi_synthetic.feature_harmonization import (
    FeatureHarmonizer,
    HarmonizationReport,
)
from stagebridge.benchmarks.semi_synthetic.world_generator import (
    WorldGenerator,
    SyntheticWorld,
    generate_multiple_worlds,
)
from stagebridge.benchmarks.semi_synthetic.interaction_rules import (
    InteractionRuleEngine,
    InteractionApplicationReport,
    compute_ground_truth_labels,
)

log = get_logger(__name__)


@dataclass
class CellPool:
    """Pool of cells available for sampling in benchmark generation."""

    group_name: str
    cells: pd.DataFrame
    expression_source: str
    n_cells: int
    stages_available: list[str]


@dataclass
class BenchmarkGenerationReport:
    """Complete report on benchmark generation."""

    config_name: str
    data_source_report: DataSourceReport | None = None
    harmonization_report: HarmonizationReport | None = None
    cell_pools: dict[str, dict] = field(default_factory=dict)
    worlds_generated: dict[str, int] = field(default_factory=dict)
    interaction_reports: list[dict] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "data_source_report": self.data_source_report.to_dict() if self.data_source_report else None,
            "harmonization_report": self.harmonization_report.to_dict() if self.harmonization_report else None,
            "cell_pools": self.cell_pools,
            "worlds_generated": self.worlds_generated,
            "interaction_summary": {
                "n_worlds": len(self.interaction_reports),
                "avg_interaction_rate": np.mean([r.get("interaction_rate", 0) for r in self.interaction_reports]) if self.interaction_reports else 0,
            },
            "output_paths": self.output_paths,
            "warnings": self.warnings,
            "success": self.success,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class SemiSyntheticBenchmarkGenerator:
    """Generate semi-synthetic benchmarks for StageBridge evaluation."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_loader: DataSourceLoader | None = None
        self.harmonizer: FeatureHarmonizer | None = None
        self.cell_pools: dict[str, CellPool] = {}
        self.worlds: dict[str, list[SyntheticWorld]] = {"train": [], "val": [], "test": []}
        self._report: BenchmarkGenerationReport | None = None

    def generate(self, use_fallback_if_missing: bool = True) -> BenchmarkGenerationReport:
        """Generate the complete benchmark dataset.

        Args:
            use_fallback_if_missing: If True, use synthetic fallback data
                when real data sources are unavailable.

        Returns:
            BenchmarkGenerationReport with generation details
        """
        report = BenchmarkGenerationReport(config_name=self.config.benchmark_name)

        try:
            # Step 1: Load data sources
            log.info("Step 1: Loading data sources...")
            report.data_source_report = self._load_data_sources(use_fallback_if_missing)

            # Step 2: Harmonize features
            log.info("Step 2: Harmonizing features...")
            report.harmonization_report = self._harmonize_features()

            # Step 3: Build cell pools
            log.info("Step 3: Building cell pools...")
            self._build_cell_pools()
            for name, pool in self.cell_pools.items():
                report.cell_pools[name] = {
                    "n_cells": pool.n_cells,
                    "expression_source": pool.expression_source,
                    "stages": pool.stages_available,
                }

            # Step 4: Generate worlds
            log.info("Step 4: Generating synthetic worlds...")
            self._generate_all_worlds()
            for split, worlds in self.worlds.items():
                report.worlds_generated[split] = len(worlds)

            # Step 5: Apply interaction rules
            log.info("Step 5: Applying interaction rules...")
            interaction_reports = self._apply_interaction_rules()
            report.interaction_reports = [r.to_dict() for r in interaction_reports]

            # Step 6: Export benchmark
            log.info("Step 6: Exporting benchmark...")
            output_paths = self._export_benchmark()
            report.output_paths = [str(p) for p in output_paths]

            report.success = True
            log.info("Benchmark generation complete!")

        except Exception as e:
            report.warnings.append(f"Generation failed: {str(e)}")
            log.error("Benchmark generation failed: %s", e)
            raise

        self._report = report
        return report

    def _load_data_sources(self, use_fallback: bool) -> DataSourceReport:
        """Load available data sources."""
        self.data_loader = DataSourceLoader(
            hlca_path=self.config.hlca_path,
            luca_path=self.config.luca_path,
            progression_path=self.config.progression_path,
        )

        report = self.data_loader.load_all(backed=True)

        # Check if we have any real data
        if not report.sources_loaded and use_fallback:
            log.warning("No real data sources found, using fallback synthetic data")
            self._create_fallback_sources()
            report.warnings.append("Using fallback synthetic data (no real sources)")

        return report

    def _create_fallback_sources(self) -> None:
        """Create fallback synthetic data sources."""
        # Create synthetic data for each expected source
        n_cells = self.config.cells_per_world * 2  # Enough for sampling

        for source_name in ["hlca", "progression"]:
            adata = create_fallback_data(
                n_cells=n_cells,
                n_genes=self.config.n_hvg,
                stages=self.config.stages,
                seed=self.config.seed + hash(source_name) % 10000,
            )
            # Store as a fallback
            if self.data_loader is not None:
                if source_name == "hlca":
                    self.data_loader._hlca = self.data_loader._inspect_source(adata, "hlca")
                elif source_name == "progression":
                    self.data_loader._progression = self.data_loader._inspect_source(adata, "progression")

    def _harmonize_features(self) -> HarmonizationReport:
        """Harmonize gene features across sources."""
        if self.data_loader is None:
            raise RuntimeError("Data loader not initialized")

        self.harmonizer = FeatureHarmonizer(
            n_hvg=self.config.n_hvg,
            require_all_sources=False,
        )

        # Collect gene sets from available sources
        gene_sets: dict[str, set[str]] = {}
        reference_adata = None

        for source_name in ["hlca", "luca", "progression"]:
            source = self.data_loader.get_source(source_name)
            if source is not None:
                gene_sets[source_name] = set(source.adata.var_names)
                if reference_adata is None:
                    reference_adata = source.adata

        return self.harmonizer.harmonize(gene_sets, reference_adata)

    def _build_cell_pools(self) -> None:
        """Build pools of cells for each group specification."""
        if self.data_loader is None or self.harmonizer is None:
            raise RuntimeError("Data loader or harmonizer not initialized")

        for group_spec in self.config.cell_groups:
            pool = self._build_pool_for_group(group_spec)
            if pool is not None:
                self.cell_pools[group_spec.name] = pool
                log.info(
                    "Built cell pool '%s': %d cells from %s",
                    group_spec.name,
                    pool.n_cells,
                    pool.expression_source,
                )

    def _build_pool_for_group(self, spec: CellGroupSpec) -> CellPool | None:
        """Build a cell pool for a single group specification."""
        all_cells = []

        for source_name in spec.source_datasets:
            sampled = self.data_loader.sample_cells_by_keywords(
                source_name=source_name,
                keywords=spec.selection_keywords,
                n_cells=spec.max_cells,
                stage_filter=spec.stage_filter,
                seed=abs(self.config.seed + hash(source_name)) % (2**31),
            )

            if sampled is not None and len(sampled) > 0:
                all_cells.append(sampled)

        if not all_cells:
            log.warning("No cells found for group '%s'", spec.name)
            return None

        combined = pd.concat(all_cells, ignore_index=True)

        # Limit to max cells
        if len(combined) > spec.max_cells:
            rng = np.random.default_rng(self.config.seed)
            indices = rng.choice(len(combined), size=spec.max_cells, replace=False)
            combined = combined.iloc[indices]

        # Get available stages
        stage_col = combined.get("_source_stage_col", pd.Series(["stage"] * len(combined))).iloc[0]
        if stage_col and stage_col in combined.columns:
            stages = combined[stage_col].dropna().unique().tolist()
        else:
            stages = []

        return CellPool(
            group_name=spec.name,
            cells=combined,
            expression_source=",".join(spec.source_datasets),
            n_cells=len(combined),
            stages_available=stages,
        )

    def _generate_all_worlds(self) -> None:
        """Generate all synthetic worlds for train/val/test splits."""
        # Prepare cell pool DataFrames
        pool_dfs = {name: pool.cells for name, pool in self.cell_pools.items()}

        # Generate train worlds
        self.worlds["train"] = generate_multiple_worlds(
            cell_pools=pool_dfs,
            n_worlds=self.config.n_worlds_train,
            cells_per_world=self.config.cells_per_world,
            split="train",
            base_seed=self.config.seed,
            stages=self.config.stages,
            width=self.config.world_width,
            height=self.config.world_height,
        )

        # Generate val worlds (different seed range)
        self.worlds["val"] = generate_multiple_worlds(
            cell_pools=pool_dfs,
            n_worlds=self.config.n_worlds_val,
            cells_per_world=self.config.cells_per_world,
            split="val",
            base_seed=self.config.seed + 100000,
            stages=self.config.stages,
            width=self.config.world_width,
            height=self.config.world_height,
        )

        # Generate test worlds (another seed range)
        self.worlds["test"] = generate_multiple_worlds(
            cell_pools=pool_dfs,
            n_worlds=self.config.n_worlds_test,
            cells_per_world=self.config.cells_per_world,
            split="test",
            base_seed=self.config.seed + 200000,
            stages=self.config.stages,
            width=self.config.world_width,
            height=self.config.world_height,
        )

    def _apply_interaction_rules(self) -> list[InteractionApplicationReport]:
        """Apply interaction rules to all worlds."""
        reports = []

        for split, worlds in self.worlds.items():
            for world in worlds:
                engine = InteractionRuleEngine(
                    rules=self.config.interaction_rules,
                    seed=world.seed + 1,
                )

                # Apply rules
                updated_positions, report = engine.apply_to_world(
                    world.cell_positions,
                    cell_group_column="cell_group",
                    stage_column="stage" if "stage" in world.cell_positions.columns else None,
                )

                # Compute ground truth labels
                updated_positions = compute_ground_truth_labels(
                    updated_positions,
                    self.config.interaction_rules,
                )

                # Update world
                world.cell_positions = updated_positions
                reports.append(report)

                log.debug(
                    "World %s: %d/%d cells interacting (%.1f%%)",
                    world.world_id,
                    report.n_interacting,
                    report.n_cells_processed,
                    100 * report.n_interacting / max(1, report.n_cells_processed),
                )

        return reports

    def _extract_expression_for_world(
        self,
        world: SyntheticWorld,
        harmonized_genes: np.ndarray | None = None,
    ) -> anndata.AnnData:
        """Extract expression matrices for all cells in a world.

        Args:
            world: The synthetic world to extract expression for
            harmonized_genes: List of harmonized gene names to use

        Returns:
            AnnData object with expression data aligned to world cells
        """
        import anndata

        if self.data_loader is None:
            raise RuntimeError("Data loader not initialized")

        cells_df = world.cell_positions

        # Group cells by source dataset
        source_groups = cells_df.groupby("_source_name")

        expression_matrices = []
        obs_records = []

        for source_name, group in source_groups:
            source_indices = group["_source_idx"].values

            # Get expression matrix from source
            X = self.data_loader.get_expression_matrix(
                source_name=source_name,
                indices=source_indices,
                genes=harmonized_genes.tolist() if harmonized_genes is not None else None,
            )

            if X is None:
                log.warning(f"Could not extract expression from {source_name}")
                continue

            expression_matrices.append(X)

            # Build obs DataFrame aligned with expression rows
            obs_subset = group[[
                "synthetic_cell_id", "x", "y", "cell_group", "stage"
            ]].copy() if "stage" in group.columns else group[[
                "synthetic_cell_id", "x", "y", "cell_group"
            ]].copy()
            obs_subset["source"] = source_name
            obs_records.append(obs_subset)

        # Concatenate all expression matrices
        if not expression_matrices:
            raise ValueError("No expression data could be extracted")

        X_combined = np.vstack(expression_matrices)
        obs_combined = pd.concat(obs_records, ignore_index=True)

        # Create AnnData
        adata = anndata.AnnData(
            X=X_combined,
            obs=obs_combined,
        )

        # Add gene names
        if harmonized_genes is not None:
            adata.var_names = harmonized_genes
        else:
            adata.var_names = [f"gene_{i}" for i in range(X_combined.shape[1])]

        # Add world metadata
        adata.uns["world_id"] = world.world_id
        adata.uns["split"] = world.split
        adata.uns["world_seed"] = world.seed

        return adata

    def _export_benchmark(self) -> list[Path]:
        """Export benchmark to disk in canonical format."""
        output_dir = self.config.output_dir / self.config.benchmark_name
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_paths = []

        # Get harmonized gene list
        harmonized_genes = None
        if self.harmonizer is not None and len(self.harmonizer.harmonized_genes) > 0:
            harmonized_genes = np.array(self.harmonizer.harmonized_genes)
            log.info(f"Using {len(harmonized_genes)} harmonized genes for expression export")

        # Export manifest
        manifest = {
            "benchmark_name": self.config.benchmark_name,
            "benchmark_family": self.config.benchmark_family,
            "n_hvg": self.config.n_hvg,
            "latent_dim": self.config.latent_dim,
            "stages": self.config.stages,
            "harmonized_genes": harmonized_genes.tolist() if harmonized_genes is not None else None,
            "splits": {
                "train": len(self.worlds["train"]),
                "val": len(self.worlds["val"]),
                "test": len(self.worlds["test"]),
            },
            "interaction_rules": [
                {
                    "rule_id": r.rule_id,
                    "sender_group": r.sender_group,
                    "receiver_group": r.receiver_group,
                    "interaction_radius": r.interaction_radius,
                    "effect_strength": r.effect_strength,
                    "effect_name": r.effect_name,
                }
                for r in self.config.interaction_rules
            ],
        }

        manifest_path = output_dir / "benchmark_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        exported_paths.append(manifest_path)

        # Export each world
        for split, worlds in self.worlds.items():
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for world in worlds:
                world_dir = split_dir / world.world_id
                world_dir.mkdir(parents=True, exist_ok=True)

                # Export cell positions
                cells_path = world_dir / "cells.parquet"
                world.cell_positions.to_parquet(cells_path, index=False)
                exported_paths.append(cells_path)

                # Export coordinates separately for easy access
                coords_path = world_dir / "coordinates.parquet"
                world.cell_positions[["synthetic_cell_id", "x", "y"]].to_parquet(
                    coords_path, index=False
                )
                exported_paths.append(coords_path)

                # Export ground truth
                gt_cols = [
                    "synthetic_cell_id",
                    "cell_group",
                    "is_interacting",
                    "triggered_rules",
                    "dominant_interaction",
                    "interaction_strength",
                    "n_effective_senders",
                ]
                gt_cols = [c for c in gt_cols if c in world.cell_positions.columns]
                # Add ground truth columns
                gt_cols.extend([c for c in world.cell_positions.columns if c.startswith("gt_")])

                gt_path = world_dir / "ground_truth.parquet"
                world.cell_positions[gt_cols].to_parquet(gt_path, index=False)
                exported_paths.append(gt_path)

                # Export expression data
                if "_source_name" in world.cell_positions.columns and "_source_idx" in world.cell_positions.columns:
                    expr_path = world_dir / "expression.h5ad"
                    try:
                        expr_adata = self._extract_expression_for_world(world, harmonized_genes)
                        expr_adata.write_h5ad(expr_path)
                        exported_paths.append(expr_path)
                        log.info(f"Exported expression for {world.world_id}: {expr_adata.shape}")
                    except Exception as e:
                        log.warning(f"Failed to export expression for {world.world_id}: {e}")

                # Export world metadata
                meta_path = world_dir / "world_metadata.json"
                with open(meta_path, "w") as f:
                    json.dump(world.metadata, f, indent=2)
                exported_paths.append(meta_path)

        # Save generation report
        if self._report is not None:
            report_path = output_dir / "generation_report.json"
            self._report.save(report_path)
            exported_paths.append(report_path)

        return exported_paths


def generate_benchmark(
    config: BenchmarkConfig | None = None,
    smoke: bool = False,
    use_fallback: bool = True,
) -> BenchmarkGenerationReport:
    """Convenience function to generate a benchmark.

    Args:
        config: Benchmark configuration (uses default if None)
        smoke: If True, use smoke test configuration
        use_fallback: If True, use fallback data when real data unavailable

    Returns:
        BenchmarkGenerationReport
    """
    if config is None:
        config = SmokeConfig() if smoke else BenchmarkConfig()

    generator = SemiSyntheticBenchmarkGenerator(config)
    return generator.generate(use_fallback_if_missing=use_fallback)
