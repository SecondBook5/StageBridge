"""AMICI-style semi-synthetic data generator with ground-truth interaction rules.

This module generates semi-synthetic data following the approach from AMICI
(Hong biorxiv 2025):
- Real single-cell expression profiles (or realistic synthetic)
- Explicit sender-receiver interaction rules with distance thresholds
- "Interacting" subpopulations with distinct DE gene signatures
- Spatial layouts with gradients creating interacting vs non-interacting zones
- Full ground truth labels for benchmarking

The key insight from AMICI: we can validate cell-cell communication methods
by creating a semi-synthetic benchmark where:
1. Spatial proximity between sender and receiver is explicitly controlled
2. Downstream gene activation occurs ONLY when cells are within interaction range
3. Ground truth is known: which cells interact, what genes are activated

This allows rigorous evaluation of StageBridge's niche-aware attention:
does the model learn to weight nearby senders appropriately?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class InteractionRule:
    """Defines a sender-receiver interaction with distance threshold.

    When a receiver of type `receiver_type` is within `max_distance` of
    a sender of type `sender_type`, the `downstream_genes` are activated
    with specified fold changes.

    This models biological interactions like:
    - IL1B-IL1R1: Macrophage (sender) → Epithelial (receiver)
    - CXCL12-CXCR4: Fibroblast (sender) → T cell (receiver)
    """
    sender_type: str
    receiver_type: str
    max_distance: float  # microns
    downstream_genes: list[str]
    fold_changes: list[float]  # per gene
    interaction_name: str = ""

    def __post_init__(self):
        if not self.interaction_name:
            self.interaction_name = f"{self.sender_type}→{self.receiver_type}"
        if len(self.downstream_genes) != len(self.fold_changes):
            raise ValueError("downstream_genes and fold_changes must have same length")


@dataclass
class SemiSyntheticConfig:
    """Configuration for AMICI-style semi-synthetic data generation.

    Default values create a scenario with:
    - 3 cell types (like AMICI's PBMC setup)
    - 2 interaction rules at different length scales
    - Spatial gradient creating interacting and non-interacting zones
    """
    # Cell type composition
    cell_types: list[str] = field(default_factory=lambda: [
        "Epithelial", "Macrophage", "Fibroblast"
    ])
    cell_type_fractions: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])

    # Spatial layout
    n_cells: int = 5000
    tissue_size: float = 1000.0  # microns
    spatial_pattern: Literal["gradient", "clusters", "uniform"] = "gradient"

    # Gene expression
    n_genes: int = 500
    n_latent: int = 40  # latent dim for StageBridge

    # Ring radii for neighborhood computation
    ring_radii: list[float] = field(default_factory=lambda: [50, 100, 150, 200])
    max_cells_per_ring: int = 32

    # Disease stages for progression
    stages: list[str] = field(default_factory=lambda: [
        "Normal", "AAH", "AIS", "MIA", "LUAD"
    ])

    seed: int = 42


@dataclass
class SemiSyntheticGroundTruth:
    """Ground truth labels for semi-synthetic benchmark evaluation.

    This enables rigorous validation:
    - Which cells are "interacting" (within range of relevant sender)?
    - Which genes should be differentially expressed?
    - What is the expected attention pattern?
    """
    # Per-cell labels
    cell_types: np.ndarray  # [n_cells] string cell type labels
    is_interacting: np.ndarray  # [n_cells] bool - within interaction range
    nearest_sender_distance: np.ndarray  # [n_cells] float - distance to nearest relevant sender
    activated_genes: np.ndarray  # [n_cells, n_genes] bool - which genes are activated

    # Interaction metadata
    interaction_rules: list[InteractionRule]
    receiver_sender_pairs: pd.DataFrame  # receiver_idx, sender_idx, distance, rule_name

    # Spatial neighborhoods
    ring_compositions: list[pd.DataFrame]  # per-ring cell type counts

    # Expected attention targets (which ring should have highest attention)
    expected_attention_ring: np.ndarray  # [n_cells] int - ring idx with relevant sender


class AMICISemiSyntheticGenerator:
    """Generates semi-synthetic data following AMICI methodology.

    Key design choices from AMICI:
    1. Use real or realistic expression profiles as base
    2. Define explicit interaction rules with distance thresholds
    3. Create spatial layout with gradient pattern
    4. Activate downstream genes only for cells within interaction range
    5. Provide complete ground truth for evaluation

    Usage:
        generator = AMICISemiSyntheticGenerator(config)
        generator.add_interaction_rule(InteractionRule(...))
        data, ground_truth = generator.generate()
    """

    def __init__(self, config: SemiSyntheticConfig | None = None):
        self.config = config or SemiSyntheticConfig()
        self.interaction_rules: list[InteractionRule] = []
        self.rng = np.random.default_rng(self.config.seed)

    def add_interaction_rule(self, rule: InteractionRule) -> "AMICISemiSyntheticGenerator":
        """Add an interaction rule. Chainable."""
        self.interaction_rules.append(rule)
        return self

    def add_default_rules(self) -> "AMICISemiSyntheticGenerator":
        """Add default IL1B-IL1R1 and CXCL12-CXCR4 interactions.

        These mirror the biological interactions we expect StageBridge to capture:
        - IL1B-IL1R1: Short-range macrophage→epithelial signaling
        - CXCL12-CXCR4: Longer-range fibroblast→epithelial signaling
        """
        # Short-range interaction (50 microns)
        self.add_interaction_rule(InteractionRule(
            sender_type="Macrophage",
            receiver_type="Epithelial",
            max_distance=50.0,
            downstream_genes=["IL1R1", "NFkB_target1", "NFkB_target2", "CXCL8"],
            fold_changes=[2.0, 3.0, 2.5, 4.0],
            interaction_name="IL1B-IL1R1"
        ))

        # Longer-range interaction (100 microns)
        self.add_interaction_rule(InteractionRule(
            sender_type="Fibroblast",
            receiver_type="Epithelial",
            max_distance=100.0,
            downstream_genes=["CXCR4", "ERK_target1", "MMP9"],
            fold_changes=[2.5, 2.0, 3.0],
            interaction_name="CXCL12-CXCR4"
        ))

        return self

    def _generate_spatial_layout(self) -> np.ndarray:
        """Generate spatial coordinates with specified pattern."""
        n = self.config.n_cells
        size = self.config.tissue_size

        if self.config.spatial_pattern == "uniform":
            coords = self.rng.uniform(0, size, (n, 2))

        elif self.config.spatial_pattern == "gradient":
            # Gradient pattern: cell type composition varies along x-axis
            # This creates natural "interacting" and "non-interacting" zones
            coords = self.rng.uniform(0, size, (n, 2))

        elif self.config.spatial_pattern == "clusters":
            # Clustered pattern: cells of same type tend to group
            n_clusters = len(self.config.cell_types) * 3
            cluster_centers = self.rng.uniform(0.1 * size, 0.9 * size, (n_clusters, 2))
            cluster_stds = self.rng.uniform(20, 50, n_clusters)

            coords = []
            for _ in range(n):
                cluster_idx = self.rng.integers(n_clusters)
                coord = self.rng.normal(cluster_centers[cluster_idx], cluster_stds[cluster_idx])
                coord = np.clip(coord, 0, size)
                coords.append(coord)
            coords = np.array(coords)
        else:
            raise ValueError(f"Unknown spatial pattern: {self.config.spatial_pattern}")

        return coords

    def _assign_cell_types(self, coords: np.ndarray) -> np.ndarray:
        """Assign cell types based on spatial position and configured fractions."""
        n = len(coords)

        if self.config.spatial_pattern == "gradient":
            # Cell type probability varies with x-coordinate
            # Creates natural spatial segregation
            x_norm = coords[:, 0] / self.config.tissue_size

            # Modulate fractions based on x position
            cell_types = []
            for i in range(n):
                # Epithelial more common on left, Macrophage in middle, Fibroblast on right
                adjusted_fracs = np.array(self.config.cell_type_fractions).copy()
                if x_norm[i] < 0.33:
                    adjusted_fracs[0] *= 1.5  # More epithelial
                elif x_norm[i] < 0.66:
                    adjusted_fracs[1] *= 1.5  # More macrophage
                else:
                    adjusted_fracs[2] *= 1.5  # More fibroblast
                adjusted_fracs /= adjusted_fracs.sum()

                cell_types.append(self.rng.choice(self.config.cell_types, p=adjusted_fracs))
            return np.array(cell_types)
        else:
            # Simple random assignment
            return self.rng.choice(
                self.config.cell_types,
                size=n,
                p=self.config.cell_type_fractions
            )

    def _generate_base_expression(self, cell_types: np.ndarray) -> np.ndarray:
        """Generate baseline expression profiles per cell type."""
        n = len(cell_types)
        n_genes = self.config.n_genes

        # Cell type-specific expression means
        type_means = {}
        for ct in self.config.cell_types:
            type_means[ct] = self.rng.exponential(1.0, n_genes)

        # Generate expression with noise
        expression = np.zeros((n, n_genes))
        for i, ct in enumerate(cell_types):
            base = type_means[ct]
            # Log-normal noise typical of scRNA-seq
            noise = self.rng.lognormal(0, 0.5, n_genes)
            expression[i] = base * noise

        return expression

    def _apply_interaction_effects(
        self,
        coords: np.ndarray,
        cell_types: np.ndarray,
        expression: np.ndarray,
        gene_names: list[str]
    ) -> tuple[np.ndarray, SemiSyntheticGroundTruth]:
        """Apply interaction effects to expression and compute ground truth."""
        n = len(coords)
        n_genes = expression.shape[1]

        # Build spatial index
        tree = cKDTree(coords)

        # Track ground truth
        is_interacting = np.zeros(n, dtype=bool)
        nearest_sender_dist = np.full(n, np.inf)
        activated_genes = np.zeros((n, n_genes), dtype=bool)
        expected_attention_ring = np.full(n, -1, dtype=int)
        receiver_sender_pairs = []

        # Gene name to index mapping
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Process each interaction rule
        for rule in self.interaction_rules:
            # Find sender and receiver indices
            sender_mask = cell_types == rule.sender_type
            receiver_mask = cell_types == rule.receiver_type

            sender_idx = np.where(sender_mask)[0]
            receiver_idx = np.where(receiver_mask)[0]

            if len(sender_idx) == 0 or len(receiver_idx) == 0:
                continue

            # For each receiver, find nearby senders
            for recv_i in receiver_idx:
                recv_coord = coords[recv_i:recv_i+1]

                # Find senders within max_distance
                nearby_indices = tree.query_ball_point(recv_coord, rule.max_distance)[0]
                nearby_senders = [i for i in nearby_indices if i in sender_idx]

                if nearby_senders:
                    # This receiver is interacting!
                    is_interacting[recv_i] = True

                    # Find nearest sender
                    distances = np.linalg.norm(coords[nearby_senders] - recv_coord, axis=1)
                    nearest_idx = np.argmin(distances)
                    nearest_sender = nearby_senders[nearest_idx]
                    nearest_dist = distances[nearest_idx]

                    if nearest_dist < nearest_sender_dist[recv_i]:
                        nearest_sender_dist[recv_i] = nearest_dist

                        # Determine which ring the sender falls into
                        for ring_idx, (r_inner, r_outer) in enumerate(zip(
                            [0] + self.config.ring_radii[:-1],
                            self.config.ring_radii
                        )):
                            if r_inner <= nearest_dist < r_outer:
                                expected_attention_ring[recv_i] = ring_idx
                                break

                    # Activate downstream genes
                    for gene_name, fc in zip(rule.downstream_genes, rule.fold_changes):
                        if gene_name in gene_to_idx:
                            gene_idx = gene_to_idx[gene_name]
                            expression[recv_i, gene_idx] *= fc
                            activated_genes[recv_i, gene_idx] = True

                    # Record pairs for ground truth
                    for sender_i in nearby_senders:
                        dist = np.linalg.norm(coords[sender_i] - recv_coord)
                        receiver_sender_pairs.append({
                            "receiver_idx": recv_i,
                            "sender_idx": sender_i,
                            "distance": dist,
                            "rule_name": rule.interaction_name
                        })

        # Create receiver-sender pairs DataFrame
        pairs_df = pd.DataFrame(receiver_sender_pairs) if receiver_sender_pairs else pd.DataFrame(
            columns=["receiver_idx", "sender_idx", "distance", "rule_name"]
        )

        # Compute ring compositions (for each cell, what cell types are in each ring)
        ring_compositions = self._compute_ring_compositions(coords, cell_types)

        ground_truth = SemiSyntheticGroundTruth(
            cell_types=cell_types,
            is_interacting=is_interacting,
            nearest_sender_distance=nearest_sender_dist,
            activated_genes=activated_genes,
            interaction_rules=self.interaction_rules,
            receiver_sender_pairs=pairs_df,
            ring_compositions=ring_compositions,
            expected_attention_ring=expected_attention_ring
        )

        return expression, ground_truth

    def _compute_ring_compositions(
        self,
        coords: np.ndarray,
        cell_types: np.ndarray
    ) -> list[pd.DataFrame]:
        """Compute cell type composition per ring for each cell."""
        tree = cKDTree(coords)
        ring_comps = []

        radii = [0] + self.config.ring_radii

        for ring_idx in range(len(self.config.ring_radii)):
            r_inner = radii[ring_idx]
            r_outer = radii[ring_idx + 1]

            compositions = []
            for i in range(len(coords)):
                # Find cells in this ring
                inner_cells = set(tree.query_ball_point(coords[i], r_inner))
                outer_cells = set(tree.query_ball_point(coords[i], r_outer))
                ring_cells = list(outer_cells - inner_cells)

                # Count cell types
                ring_types = cell_types[ring_cells] if ring_cells else []
                type_counts = {ct: 0 for ct in self.config.cell_types}
                for ct in ring_types:
                    type_counts[ct] += 1
                type_counts["cell_idx"] = i
                compositions.append(type_counts)

            ring_comps.append(pd.DataFrame(compositions))

        return ring_comps

    def _compute_embeddings(self, expression: np.ndarray) -> np.ndarray:
        """Compute latent embeddings from expression (simple PCA-like)."""
        # Center and scale
        expr_centered = expression - expression.mean(axis=0)
        expr_scaled = expr_centered / (expression.std(axis=0) + 1e-6)

        # Random projection to latent dim (simulates learned embedding)
        n_genes = expression.shape[1]
        projection = self.rng.standard_normal((n_genes, self.config.n_latent))
        projection /= np.linalg.norm(projection, axis=0, keepdims=True)

        embeddings = expr_scaled @ projection
        return embeddings

    def _build_neighborhoods(
        self,
        coords: np.ndarray,
        cell_types: np.ndarray,
        embeddings: np.ndarray,
        stage_labels: np.ndarray,
    ) -> pd.DataFrame:
        """Build neighborhoods.parquet format compatible with StageBridgeDataset."""
        n = len(coords)
        tree = cKDTree(coords)
        radii = [0] + self.config.ring_radii
        max_cells = self.config.max_cells_per_ring

        records = []
        for i in range(n):
            record = {
                "cell_id": f"cell_{i:05d}",
                "donor_id": "semisynthetic_001",
                "stage": stage_labels[i],
                "x": coords[i, 0],
                "y": coords[i, 1],
                "cell_type": cell_types[i],
            }

            # Receiver embedding (HLCA concat LuCA format: first 30 are HLCA, last 10 are LuCA)
            # We'll use first 30 dims as "HLCA-like" and last 10 as "LuCA-like"
            record["receiver_hlca"] = embeddings[i, :30].tolist()
            record["receiver_luca"] = embeddings[i, 30:].tolist()
            record["hlca"] = embeddings[i, :30].tolist()  # Same for reference
            record["luca"] = embeddings[i, 30:].tolist()

            # Ring cells
            for ring_idx in range(len(self.config.ring_radii)):
                r_inner = radii[ring_idx]
                r_outer = radii[ring_idx + 1]

                inner_cells = set(tree.query_ball_point(coords[i], r_inner))
                outer_cells = set(tree.query_ball_point(coords[i], r_outer))
                ring_cell_indices = list(outer_cells - inner_cells - {i})

                # Limit to max_cells (take closest)
                if len(ring_cell_indices) > max_cells:
                    distances = np.linalg.norm(coords[ring_cell_indices] - coords[i], axis=1)
                    sorted_idx = np.argsort(distances)[:max_cells]
                    ring_cell_indices = [ring_cell_indices[j] for j in sorted_idx]

                # Store fused embeddings (HLCA+LuCA concat)
                ring_embeddings = embeddings[ring_cell_indices] if ring_cell_indices else np.zeros((0, self.config.n_latent))
                record[f"ring_{ring_idx+1}_cells"] = ring_embeddings.tolist()
                record[f"ring_{ring_idx+1}_n"] = len(ring_cell_indices)

            records.append(record)

        return pd.DataFrame(records)

    def generate(self) -> tuple[pd.DataFrame, SemiSyntheticGroundTruth]:
        """Generate semi-synthetic data with ground truth.

        Returns:
            neighborhoods: DataFrame in StageBridgeDataset format
            ground_truth: Full ground truth for evaluation
        """
        if not self.interaction_rules:
            self.add_default_rules()

        print(f"Generating semi-synthetic data with {len(self.interaction_rules)} interaction rules...")

        # 1. Generate spatial layout
        coords = self._generate_spatial_layout()
        print(f"  Created spatial layout: {len(coords)} cells")

        # 2. Assign cell types
        cell_types = self._assign_cell_types(coords)
        for ct in self.config.cell_types:
            n_ct = (cell_types == ct).sum()
            print(f"    {ct}: {n_ct} ({100*n_ct/len(coords):.1f}%)")

        # 3. Generate base expression
        gene_names = [f"gene_{i:03d}" for i in range(self.config.n_genes)]
        # Add known interaction genes
        for rule in self.interaction_rules:
            for gene in rule.downstream_genes:
                if gene not in gene_names:
                    gene_names[len(gene_names) % self.config.n_genes] = gene
        expression = self._generate_base_expression(cell_types)
        print(f"  Generated expression: {expression.shape}")

        # 4. Apply interaction effects
        expression, ground_truth = self._apply_interaction_effects(
            coords, cell_types, expression, gene_names
        )
        n_interacting = ground_truth.is_interacting.sum()
        print(f"  Interacting cells: {n_interacting} ({100*n_interacting/len(coords):.1f}%)")

        # 5. Compute embeddings
        embeddings = self._compute_embeddings(expression)
        print(f"  Computed embeddings: {embeddings.shape}")

        # 6. Assign stage labels (based on spatial gradient for interesting pattern)
        x_norm = coords[:, 0] / self.config.tissue_size
        stage_probs = np.zeros((len(coords), len(self.config.stages)))
        for i, stage in enumerate(self.config.stages):
            # Each stage most common in different x region
            center = i / (len(self.config.stages) - 1)
            stage_probs[:, i] = np.exp(-((x_norm - center) ** 2) / 0.1)
        stage_probs /= stage_probs.sum(axis=1, keepdims=True)
        stage_labels = np.array([
            self.rng.choice(self.config.stages, p=stage_probs[i])
            for i in range(len(coords))
        ])

        # 7. Build neighborhoods DataFrame
        neighborhoods = self._build_neighborhoods(coords, cell_types, embeddings, stage_labels)
        print(f"  Built neighborhoods: {len(neighborhoods)} rows")

        return neighborhoods, ground_truth


def create_demo_semisynthetic(
    output_dir: str | None = None,
    n_cells: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, SemiSyntheticGroundTruth]:
    """Create demo semi-synthetic dataset for notebook/presentation.

    This creates a small but complete semi-synthetic dataset with:
    - 3 cell types (Epithelial, Macrophage, Fibroblast)
    - 2 interaction rules (IL1B and CXCL12 pathways)
    - Gradient spatial pattern
    - Full ground truth for evaluation

    Args:
        output_dir: If provided, save neighborhoods.parquet here
        n_cells: Number of cells (default 2000 for quick demo)
        seed: Random seed

    Returns:
        neighborhoods: DataFrame ready for StageBridgeDataset
        ground_truth: Full ground truth for evaluation
    """
    from pathlib import Path

    config = SemiSyntheticConfig(
        n_cells=n_cells,
        seed=seed,
        spatial_pattern="gradient",
    )

    generator = AMICISemiSyntheticGenerator(config)
    generator.add_default_rules()

    neighborhoods, ground_truth = generator.generate()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save neighborhoods
        neighborhoods.to_parquet(output_path / "neighborhoods.parquet")
        print(f"\nSaved to: {output_path / 'neighborhoods.parquet'}")

        # Save ground truth summary
        gt_summary = {
            "n_cells": len(neighborhoods),
            "n_interacting": int(ground_truth.is_interacting.sum()),
            "cell_types": list(np.unique(ground_truth.cell_types)),
            "interaction_rules": [
                {
                    "name": r.interaction_name,
                    "sender": r.sender_type,
                    "receiver": r.receiver_type,
                    "max_distance": r.max_distance,
                    "downstream_genes": r.downstream_genes,
                }
                for r in ground_truth.interaction_rules
            ]
        }
        import json
        with open(output_path / "ground_truth_summary.json", "w") as f:
            json.dump(gt_summary, f, indent=2)
        print(f"Saved to: {output_path / 'ground_truth_summary.json'}")

        # Save receiver-sender pairs
        ground_truth.receiver_sender_pairs.to_csv(
            output_path / "receiver_sender_pairs.csv", index=False
        )
        print(f"Saved to: {output_path / 'receiver_sender_pairs.csv'}")

    return neighborhoods, ground_truth


if __name__ == "__main__":
    # Demo: generate semi-synthetic data
    import argparse

    parser = argparse.ArgumentParser(description="Generate AMICI-style semi-synthetic data")
    parser.add_argument("--output", "-o", type=str, default="data/semisynthetic_demo",
                       help="Output directory")
    parser.add_argument("--n-cells", "-n", type=int, default=2000,
                       help="Number of cells")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()

    neighborhoods, ground_truth = create_demo_semisynthetic(
        output_dir=args.output,
        n_cells=args.n_cells,
        seed=args.seed,
    )

    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    print(f"Total cells: {len(neighborhoods)}")
    print(f"Interacting cells: {ground_truth.is_interacting.sum()}")
    print(f"Cell types: {list(np.unique(ground_truth.cell_types))}")
    print(f"Interaction pairs: {len(ground_truth.receiver_sender_pairs)}")
