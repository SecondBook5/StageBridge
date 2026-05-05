#!/usr/bin/env python
"""Create semi-synthetic benchmark from real expression profiles.

AMICI-style approach:
1. Sample REAL expression profiles from your h5ad
2. Place them in CONTROLLED spatial layouts
3. Apply interaction rules with KNOWN ground truth
4. Output compressed dataset for local testing

This gives you:
- Real biological variation in expression
- Controlled spatial structure for ground truth
- Small enough to run locally

Usage:
    # From snRNA data (uses UMAP for initial layout, then controlled placement)
    python scripts/create_semisynthetic_benchmark.py \
        --h5ad /path/to/snrna.h5ad \
        --output data/semisynthetic_benchmark \
        --n-cells 5000

    # Quick local test set
    python scripts/create_semisynthetic_benchmark.py \
        --h5ad /path/to/snrna.h5ad \
        --output data/local_test \
        --n-cells 1000 \
        --stages Normal,AAH,AIS
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class InteractionRule:
    """Sender-receiver interaction with distance threshold."""
    sender_type: str
    receiver_type: str
    max_distance: float
    ligand_gene: str
    receptor_gene: str
    interaction_name: str = ""

    def __post_init__(self):
        if not self.interaction_name:
            self.interaction_name = f"{self.ligand_gene}-{self.receptor_gene}"


# Key biological interactions for LUAD
DEFAULT_RULES = [
    InteractionRule("Macrophage", "Epithelial", 50.0, "IL1B", "IL1R1"),
    InteractionRule("Macrophage", "Epithelial", 50.0, "IL1A", "IL1R1"),
    InteractionRule("Macrophage", "Epithelial", 60.0, "TNF", "TNFRSF1A"),
    InteractionRule("Fibroblast", "Epithelial", 100.0, "CXCL12", "CXCR4"),
    InteractionRule("Fibroblast", "Epithelial", 75.0, "TGFB1", "TGFBR1"),
    InteractionRule("Fibroblast", "Epithelial", 100.0, "HGF", "MET"),
    InteractionRule("Macrophage", "Epithelial", 60.0, "SPP1", "CD44"),
    InteractionRule("Macrophage", "Epithelial", 70.0, "IL6", "IL6R"),
    InteractionRule("Epithelial", "T cell", 40.0, "CD274", "PDCD1", "PDL1-PD1"),
]


def load_real_profiles(
    h5ad_path: str,
    n_cells: int,
    cell_type_key: str = "cell_type",
    stages: list[str] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """Load and sample real expression profiles from h5ad.

    Returns:
        expression: [n_cells, n_genes] expression matrix
        cell_types: [n_cells] cell type labels
        stage_labels: [n_cells] stage labels
        gene_names: list of gene names
        metadata: dict with cell_ids, original indices, etc.
    """
    import scanpy as sc

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Filter by stages if specified
    if stages:
        if "stage" in adata.obs.columns:
            mask = adata.obs["stage"].isin(stages)
            adata = adata[mask].copy()
            print(f"  Filtered to stages {stages}: {adata.n_obs:,} cells")

    # Get cell types
    if cell_type_key not in adata.obs.columns:
        # Try alternatives
        for alt in ["luca_cell_type", "cell_type_fine", "celltype"]:
            if alt in adata.obs.columns:
                cell_type_key = alt
                break
        else:
            raise ValueError(f"Cell type column not found. Available: {list(adata.obs.columns)}")

    print(f"  Using cell type column: {cell_type_key}")

    # Stratified sampling to preserve cell type proportions
    rng = np.random.default_rng(seed)

    cell_type_counts = adata.obs[cell_type_key].value_counts()
    total = cell_type_counts.sum()

    sampled_indices = []
    for ct, count in cell_type_counts.items():
        # Sample proportionally
        n_sample = max(1, int(n_cells * count / total))
        ct_indices = np.where(adata.obs[cell_type_key] == ct)[0]
        if len(ct_indices) > n_sample:
            sampled = rng.choice(ct_indices, n_sample, replace=False)
        else:
            sampled = ct_indices
        sampled_indices.extend(sampled)

    # Shuffle and trim to exact n_cells
    rng.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:n_cells]

    # Extract data
    adata_sampled = adata[sampled_indices].copy()

    if hasattr(adata_sampled.X, "toarray"):
        expression = adata_sampled.X.toarray()
    else:
        expression = np.array(adata_sampled.X)

    cell_types = adata_sampled.obs[cell_type_key].values

    # Stage labels
    if "stage" in adata_sampled.obs.columns:
        stage_labels = adata_sampled.obs["stage"].values
    else:
        stage_labels = np.array(["Unknown"] * len(sampled_indices))

    gene_names = list(adata_sampled.var_names)

    metadata = {
        "cell_ids": list(adata_sampled.obs_names),
        "original_indices": sampled_indices,
        "source_file": str(h5ad_path),
    }

    print(f"  Sampled {len(sampled_indices)} cells")
    print(f"  Cell types: {len(np.unique(cell_types))}")
    print(f"  Stages: {list(np.unique(stage_labels))}")

    return expression, cell_types, stage_labels, gene_names, metadata


def create_spatial_layout(
    cell_types: np.ndarray,
    pattern: str = "gradient",
    tissue_size: float = 1000.0,
    seed: int = 42,
) -> np.ndarray:
    """Create controlled spatial layout.

    Patterns:
    - "gradient": Cell types segregated along x-axis (creates interacting zones)
    - "mixed": Random mixing (high interaction potential)
    - "clustered": Cell type clusters (some interaction at boundaries)
    """
    rng = np.random.default_rng(seed)
    n = len(cell_types)

    if pattern == "gradient":
        # Sort by cell type, then add noise
        # This creates zones where different cell types meet
        unique_types = np.unique(cell_types)
        type_to_x = {ct: i / (len(unique_types) - 1) for i, ct in enumerate(unique_types)}

        coords = np.zeros((n, 2))
        for i, ct in enumerate(cell_types):
            base_x = type_to_x.get(ct, 0.5) * tissue_size
            # Add substantial noise to create mixing at boundaries
            coords[i, 0] = base_x + rng.normal(0, tissue_size * 0.15)
            coords[i, 1] = rng.uniform(0, tissue_size)

        # Clip to tissue bounds
        coords = np.clip(coords, 0, tissue_size)

    elif pattern == "mixed":
        # Uniform random - maximum interaction potential
        coords = rng.uniform(0, tissue_size, (n, 2))

    elif pattern == "clustered":
        # Create clusters per cell type
        unique_types = np.unique(cell_types)
        n_clusters_per_type = 3

        # Generate cluster centers
        cluster_centers = {}
        for ct in unique_types:
            centers = rng.uniform(0.1 * tissue_size, 0.9 * tissue_size,
                                  (n_clusters_per_type, 2))
            cluster_centers[ct] = centers

        coords = np.zeros((n, 2))
        for i, ct in enumerate(cell_types):
            centers = cluster_centers[ct]
            # Pick random cluster
            cluster_idx = rng.integers(len(centers))
            center = centers[cluster_idx]
            # Add Gaussian noise
            coords[i] = rng.normal(center, tissue_size * 0.08)

        coords = np.clip(coords, 0, tissue_size)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return coords


def compute_embeddings(
    expression: np.ndarray,
    n_latent: int = 40,
    seed: int = 42,
) -> np.ndarray:
    """Compute latent embeddings from expression.

    Uses simple PCA-like projection for speed.
    For real use, you'd use scVI or the actual HLCA/LuCA mappers.
    """
    rng = np.random.default_rng(seed)

    # Log-normalize if not already
    expr_sum = expression.sum(axis=1, keepdims=True)
    expr_norm = np.log1p(expression / (expr_sum + 1) * 10000)

    # Center
    expr_centered = expr_norm - expr_norm.mean(axis=0)

    # Random projection (fast approximation to PCA)
    n_genes = expression.shape[1]
    projection = rng.standard_normal((n_genes, n_latent))
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)

    embeddings = expr_centered @ projection

    # Normalize
    embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-6)

    return embeddings


def label_interactions(
    coords: np.ndarray,
    cell_types: np.ndarray,
    expression: np.ndarray,
    gene_names: list[str],
    rules: list[InteractionRule],
    ring_radii: list[float] = [50, 100, 150, 200],
    ligand_threshold: float = 0.5,
    receptor_threshold: float = 0.5,
) -> dict:
    """Apply interaction rules and compute ground truth labels."""
    n = len(coords)
    tree = cKDTree(coords)

    # Gene name to index
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Initialize labels
    is_interacting = np.zeros(n, dtype=bool)
    interaction_type = np.array(["none"] * n, dtype=object)
    nearest_sender_dist = np.full(n, np.inf)
    nearest_sender_id = np.full(n, -1, dtype=int)
    expected_ring = np.full(n, -1, dtype=int)
    pairs = []

    for rule in rules:
        # Find senders by cell type + ligand expression
        sender_type_mask = np.array([
            rule.sender_type.lower() in str(ct).lower()
            for ct in cell_types
        ])

        if rule.ligand_gene in gene_to_idx:
            ligand_idx = gene_to_idx[rule.ligand_gene]
            sender_expr_mask = expression[:, ligand_idx] > ligand_threshold
            sender_mask = sender_type_mask & sender_expr_mask
        else:
            sender_mask = sender_type_mask

        # Find receivers by cell type + receptor expression
        receiver_type_mask = np.array([
            rule.receiver_type.lower() in str(ct).lower()
            for ct in cell_types
        ])

        if rule.receptor_gene in gene_to_idx:
            receptor_idx = gene_to_idx[rule.receptor_gene]
            receiver_expr_mask = expression[:, receptor_idx] > receptor_threshold
            receiver_mask = receiver_type_mask & receiver_expr_mask
        else:
            receiver_mask = receiver_type_mask

        sender_idx = np.where(sender_mask)[0]
        receiver_idx = np.where(receiver_mask)[0]

        if len(sender_idx) == 0 or len(receiver_idx) == 0:
            continue

        # Build sender tree for fast queries
        sender_coords = coords[sender_idx]
        sender_tree = cKDTree(sender_coords)

        # Query all receivers at once
        receiver_coords = coords[receiver_idx]
        nearby_lists = sender_tree.query_ball_point(receiver_coords, rule.max_distance)

        for i, recv_i in enumerate(receiver_idx):
            nearby_local = nearby_lists[i]
            if not nearby_local:
                continue

            nearby_senders = sender_idx[nearby_local]
            distances = np.linalg.norm(coords[nearby_senders] - coords[recv_i], axis=1)
            nearest_local = np.argmin(distances)
            nearest_sender = nearby_senders[nearest_local]
            dist = distances[nearest_local]

            if dist < nearest_sender_dist[recv_i]:
                is_interacting[recv_i] = True
                interaction_type[recv_i] = rule.interaction_name
                nearest_sender_dist[recv_i] = dist
                nearest_sender_id[recv_i] = nearest_sender

                # Determine ring
                radii = [0] + ring_radii
                for ring_idx in range(len(ring_radii)):
                    if radii[ring_idx] <= dist < radii[ring_idx + 1]:
                        expected_ring[recv_i] = ring_idx
                        break

            # Record all pairs
            for j, sender_i in enumerate(nearby_senders):
                pairs.append({
                    "receiver_idx": recv_i,
                    "sender_idx": sender_i,
                    "distance": distances[j],
                    "rule_name": rule.interaction_name,
                })

    return {
        "is_interacting": is_interacting,
        "interaction_type": interaction_type,
        "nearest_sender_distance": nearest_sender_dist,
        "nearest_sender_id": nearest_sender_id,
        "expected_attention_ring": expected_ring,
        "pairs": pd.DataFrame(pairs) if pairs else pd.DataFrame(
            columns=["receiver_idx", "sender_idx", "distance", "rule_name"]
        ),
    }


def build_neighborhoods(
    coords: np.ndarray,
    cell_types: np.ndarray,
    stage_labels: np.ndarray,
    embeddings: np.ndarray,
    ring_radii: list[float] = [50, 100, 150, 200],
    max_cells_per_ring: int = 32,
) -> pd.DataFrame:
    """Build neighborhoods.parquet compatible with StageBridgeDataset."""
    n = len(coords)
    tree = cKDTree(coords)
    radii = [0] + ring_radii

    records = []
    for i in range(n):
        # Split embedding into HLCA (30) and LuCA (10) portions
        hlca_z = embeddings[i, :30].tolist() if embeddings.shape[1] >= 30 else embeddings[i].tolist()
        luca_z = embeddings[i, 30:40].tolist() if embeddings.shape[1] >= 40 else [0.0] * 10
        receiver_z = (hlca_z + luca_z)[:40]  # Fused

        record = {
            "cell_id": f"cell_{i:06d}",
            "donor_id": "semisynthetic",
            "stage": stage_labels[i],
            "x": coords[i, 0],
            "y": coords[i, 1],
            "cell_type": cell_types[i],
            "receiver_z": receiver_z,
            "hlca_z": hlca_z,
            "luca_z": luca_z,
        }

        # Ring cells
        for ring_idx in range(len(ring_radii)):
            r_inner = radii[ring_idx]
            r_outer = radii[ring_idx + 1]

            inner_cells = set(tree.query_ball_point(coords[i], r_inner))
            outer_cells = set(tree.query_ball_point(coords[i], r_outer))
            ring_cell_indices = list(outer_cells - inner_cells - {i})

            # Limit to max_cells (take closest)
            if len(ring_cell_indices) > max_cells_per_ring:
                distances = np.linalg.norm(
                    coords[ring_cell_indices] - coords[i], axis=1
                )
                sorted_idx = np.argsort(distances)[:max_cells_per_ring]
                ring_cell_indices = [ring_cell_indices[j] for j in sorted_idx]

            # Store fused embeddings for ring cells
            if ring_cell_indices:
                ring_embeddings = []
                for j in ring_cell_indices:
                    hlca_j = embeddings[j, :30].tolist() if embeddings.shape[1] >= 30 else embeddings[j].tolist()
                    luca_j = embeddings[j, 30:40].tolist() if embeddings.shape[1] >= 40 else [0.0] * 10
                    ring_embeddings.append((hlca_j + luca_j)[:40])
            else:
                ring_embeddings = []

            record[f"ring_{ring_idx + 1}_cells"] = ring_embeddings

        records.append(record)

    return pd.DataFrame(records)


def create_semisynthetic_benchmark(
    h5ad_path: str,
    output_dir: str,
    n_cells: int = 5000,
    cell_type_key: str = "cell_type",
    stages: list[str] | None = None,
    spatial_pattern: str = "gradient",
    seed: int = 42,
):
    """Create semi-synthetic benchmark dataset.

    Args:
        h5ad_path: Path to real h5ad data
        output_dir: Output directory
        n_cells: Number of cells to sample
        cell_type_key: Column for cell types
        stages: Filter to these stages (None = all)
        spatial_pattern: "gradient", "mixed", or "clustered"
        seed: Random seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating Semi-Synthetic Benchmark")
    print("=" * 60)

    # 1. Load real expression profiles
    expression, cell_types, stage_labels, gene_names, metadata = load_real_profiles(
        h5ad_path, n_cells, cell_type_key, stages, seed
    )

    # 2. Create controlled spatial layout
    print(f"\nCreating spatial layout ({spatial_pattern})...")
    coords = create_spatial_layout(cell_types, spatial_pattern, seed=seed)
    print(f"  Tissue size: {coords.max(axis=0) - coords.min(axis=0)}")

    # 3. Compute embeddings
    print("\nComputing embeddings...")
    embeddings = compute_embeddings(expression, n_latent=40, seed=seed)
    print(f"  Shape: {embeddings.shape}")

    # 4. Apply interaction rules
    print("\nApplying interaction rules...")
    ground_truth = label_interactions(
        coords, cell_types, expression, gene_names, DEFAULT_RULES
    )
    n_interacting = ground_truth["is_interacting"].sum()
    print(f"  Interacting cells: {n_interacting} ({100 * n_interacting / n_cells:.1f}%)")
    print(f"  Interaction pairs: {len(ground_truth['pairs'])}")

    # 5. Build neighborhoods
    print("\nBuilding neighborhoods...")
    neighborhoods = build_neighborhoods(
        coords, cell_types, stage_labels, embeddings
    )
    print(f"  Shape: {neighborhoods.shape}")

    # 6. Save outputs
    print("\nSaving outputs...")

    # Main data file
    neighborhoods.to_parquet(output_path / "neighborhoods.parquet")
    print(f"  neighborhoods.parquet: {len(neighborhoods)} rows")

    # Ground truth labels - use actual length from ground_truth arrays
    actual_n = len(ground_truth["is_interacting"])
    gt_labels = pd.DataFrame({
        "cell_id": [f"cell_{i:06d}" for i in range(actual_n)],
        "is_interacting": ground_truth["is_interacting"],
        "interaction_type": ground_truth["interaction_type"],
        "nearest_sender_distance": ground_truth["nearest_sender_distance"],
        "nearest_sender_id": ground_truth["nearest_sender_id"],
        "expected_attention_ring": ground_truth["expected_attention_ring"],
    })
    gt_labels.to_parquet(output_path / "ground_truth_labels.parquet")
    print(f"  ground_truth_labels.parquet")

    # Interaction pairs
    ground_truth["pairs"].to_parquet(output_path / "receiver_sender_pairs.parquet")
    print(f"  receiver_sender_pairs.parquet: {len(ground_truth['pairs'])} pairs")

    # Coordinates (for visualization)
    coords_df = pd.DataFrame({
        "cell_id": [f"cell_{i:06d}" for i in range(actual_n)],
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cell_type": cell_types,
        "stage": stage_labels,
        "is_interacting": ground_truth["is_interacting"],
    })
    coords_df.to_parquet(output_path / "coordinates.parquet")
    print(f"  coordinates.parquet")

    # Summary
    summary = {
        "n_cells": actual_n,
        "n_interacting": int(n_interacting),
        "pct_interacting": float(100 * n_interacting / actual_n),
        "cell_types": list(np.unique(cell_types)),
        "stages": list(np.unique(stage_labels)),
        "spatial_pattern": spatial_pattern,
        "source_file": str(h5ad_path),
        "interaction_rules": [
            {
                "name": r.interaction_name,
                "sender": r.sender_type,
                "receiver": r.receiver_type,
                "max_distance": r.max_distance,
                "ligand": r.ligand_gene,
                "receptor": r.receptor_gene,
            }
            for r in DEFAULT_RULES
        ],
        "n_pairs": len(ground_truth["pairs"]),
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary.json")

    # Create split manifest for training
    from stagebridge.loaders.splits import create_split_manifest
    try:
        # Use simple random split since we have single donor
        manifest = {
            "folds": [
                {
                    "train_donors": ["semisynthetic"],
                    "val_donors": ["semisynthetic"],
                    "test_donors": ["semisynthetic"],
                }
            ],
            "all_donors": ["semisynthetic"],
        }
        with open(output_path / "split_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  split_manifest.json")
    except Exception as e:
        print(f"  Warning: Could not create split manifest: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Cells: {actual_n}")
    print(f"Interacting: {n_interacting} ({100 * n_interacting / actual_n:.1f}%)")
    print(f"Cell types: {len(np.unique(cell_types))}")
    print(f"Stages: {list(np.unique(stage_labels))}")
    print(f"Pairs: {len(ground_truth['pairs'])}")
    print("=" * 60)

    print("\nTo train on this data:")
    print(f"  python scripts/train.py --data-dir {output_path} --epochs 10")

    return neighborhoods, ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Create semi-synthetic benchmark from real expression"
    )
    parser.add_argument(
        "--h5ad", "-i", type=str, required=True,
        help="Input h5ad file with real expression"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--n-cells", "-n", type=int, default=5000,
        help="Number of cells to sample (default: 5000)"
    )
    parser.add_argument(
        "--cell-type-key", type=str, default="cell_type",
        help="Column for cell types (default: cell_type)"
    )
    parser.add_argument(
        "--stages", type=str, default=None,
        help="Comma-separated stages to include (default: all)"
    )
    parser.add_argument(
        "--pattern", type=str, default="gradient",
        choices=["gradient", "mixed", "clustered"],
        help="Spatial pattern (default: gradient)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    stages = args.stages.split(",") if args.stages else None

    create_semisynthetic_benchmark(
        h5ad_path=args.h5ad,
        output_dir=args.output,
        n_cells=args.n_cells,
        cell_type_key=args.cell_type_key,
        stages=stages,
        spatial_pattern=args.pattern,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
