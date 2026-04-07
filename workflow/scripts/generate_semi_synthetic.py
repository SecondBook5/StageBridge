#!/usr/bin/env python3
"""Generate expression-aware semi-synthetic benchmark with ground truth.

Snakemake script - uses snakemake.output for paths.

This uses the proper semi-synthetic protocol:
1. Load real scRNA-seq expression profiles (or synthetic fallback)
2. Subcluster each cell type into "interacting" vs "non-interacting" pools
3. Generate synthetic spatial layout
4. Assign cells from interacting pool if sender within radius
5. Ground truth = DE genes between subclusters + interaction labels
"""

from pathlib import Path
import torch
import json
import numpy as np

from stagebridge.benchmarks.unified import (
    ExpressionSemisyntheticConfig,
    ExpressionSemisyntheticGenerator,
    InteractionSpec,
    create_expression_config,
)

# Get output paths from Snakemake
output_dir = Path(snakemake.output.benchmark).parent
output_dir.mkdir(parents=True, exist_ok=True)

# Try to get expression source from input manifest if available
expression_source = None
try:
    with open(snakemake.input.manifest) as f:
        manifest = json.load(f)
    # Check for snrna_path in manifest (may not exist)
    snrna_path_str = manifest.get("snrna_path", "")
    if snrna_path_str:
        expression_source = Path(snrna_path_str)
        if not expression_source.exists() or not expression_source.suffix == ".h5ad":
            expression_source = None
except Exception:
    expression_source = None

print("=" * 60)
print("Generating Expression-Aware Semi-Synthetic Benchmark")
print("=" * 60)

# Configure the benchmark
config = ExpressionSemisyntheticConfig(
    # Data sources
    expression_source=expression_source,

    # Subclustering parameters
    n_subclusters=2,
    min_cells_per_subcluster=50,
    leiden_resolution=0.5,

    # Spatial parameters
    n_worlds=10,
    cells_per_world=1000,

    # Gene selection
    n_hvg=2000,

    # Biologically relevant interactions (lung cancer focused)
    # Key biological hypothesis: IL1B-IL1R1 signaling is stronger in early stages
    interactions=[
        InteractionSpec(
            sender_celltype="Macrophage",
            receiver_celltype="AT2",
            interaction_radius=50.0,
            interaction_name="IL1B_IL1R1",
            # Interaction probability by stage
            stage_weights={"Normal": 0.3, "AAH": 1.2, "AIS": 1.0, "MIA": 0.8, "LUAD": 0.6},
            # DE effect size by stage (stronger in early progression)
            stage_effect_sizes={"Normal": 0.5, "AAH": 1.5, "AIS": 1.3, "MIA": 1.0, "LUAD": 0.7},
            # Associated pathways that should be activated
            associated_pathways=["NFkB", "TNFa", "JAK-STAT"],
        ),
        InteractionSpec(
            sender_celltype="Fibroblast",
            receiver_celltype="AT2",
            interaction_radius=30.0,
            interaction_name="CAF_EMT",
            # CAF effect increases with progression
            stage_weights={"Normal": 0.2, "AAH": 0.6, "AIS": 0.8, "MIA": 1.0, "LUAD": 1.2},
            stage_effect_sizes={"Normal": 0.3, "AAH": 0.7, "AIS": 1.0, "MIA": 1.3, "LUAD": 1.5},
            associated_pathways=["TGFb", "WNT", "Hypoxia"],
        ),
        InteractionSpec(
            sender_celltype="T_cell",
            receiver_celltype="AT2",
            interaction_radius=40.0,
            interaction_name="Immune_surveillance",
            # Immune surveillance decreases with progression
            stage_weights={"Normal": 1.0, "AAH": 0.9, "AIS": 0.7, "MIA": 0.5, "LUAD": 0.3},
            stage_effect_sizes={"Normal": 1.2, "AAH": 1.0, "AIS": 0.8, "MIA": 0.6, "LUAD": 0.4},
            associated_pathways=["JAK-STAT", "NFkB"],
        ),
    ],

    # Include pathway scoring
    include_pathways=True,

    # Progression stages
    stages=["Normal", "AAH", "AIS", "MIA", "LUAD"],

    # Output
    output_dir=output_dir.parent,
    benchmark_name=output_dir.name,
    seed=42,
)

# Generate benchmark
generator = ExpressionSemisyntheticGenerator(config)
report = generator.generate(use_fallback=True)

# Also create consolidated tensor file for training compatibility
print("Creating consolidated tensor file...")
all_expression = []
all_positions = []
all_labels = []
all_celltypes = []
all_stages = []
all_pathways = {}  # pathway_name -> list of arrays

benchmark_dir = output_dir
for i in range(config.n_worlds):
    world_path = benchmark_dir / f"world_{i:04d}.parquet"
    expr_path = benchmark_dir / f"world_{i:04d}_expression.npy"

    if world_path.exists() and expr_path.exists():
        import pandas as pd
        world_meta = pd.read_parquet(world_path)
        expr = np.load(expr_path)

        all_expression.append(expr)
        all_positions.append(world_meta[["x", "y"]].values)
        all_celltypes.append(world_meta["celltype"].values)
        all_stages.append(world_meta["stage"].values)

        # Get interaction labels (columns start with "is_interacting_")
        interaction_cols = [c for c in world_meta.columns if c.startswith("is_interacting_")]
        if interaction_cols:
            # Use first interaction or combine with OR
            labels = world_meta[interaction_cols[0]].values
            for col in interaction_cols[1:]:
                labels = labels | world_meta[col].values
            all_labels.append(labels)
        else:
            all_labels.append(np.zeros(len(world_meta), dtype=bool))

        # Collect pathway scores
        pathway_cols = [c for c in world_meta.columns if c.startswith("pathway_")]
        for col in pathway_cols:
            pathway_name = col.replace("pathway_", "")
            if pathway_name not in all_pathways:
                all_pathways[pathway_name] = []
            all_pathways[pathway_name].append(world_meta[col].values)

if all_expression:
    # Stack and convert to tensors
    celltypes_concat = np.concatenate(all_celltypes)
    stages_concat = np.concatenate(all_stages)

    # Encode cell types and stages as integers
    unique_celltypes = np.unique(celltypes_concat)
    unique_stages = np.unique(stages_concat)
    celltype_to_idx = {ct: i for i, ct in enumerate(unique_celltypes)}
    stage_to_idx = {s: i for i, s in enumerate(unique_stages)}

    tensors = {
        "expression": torch.from_numpy(np.concatenate(all_expression)).float(),
        "positions": torch.from_numpy(np.concatenate(all_positions)).float(),
        "is_interacting": torch.from_numpy(np.concatenate(all_labels).astype(np.int64)).long(),
        "celltype_idx": torch.tensor([celltype_to_idx[ct] for ct in celltypes_concat]).long(),
        "stage_idx": torch.tensor([stage_to_idx[s] for s in stages_concat]).long(),
        "gene_names": np.load(benchmark_dir / "gene_names.npy"),
        "celltype_names": unique_celltypes,
        "stage_names": unique_stages,
    }

    # Add pathway scores as ground truth
    if all_pathways:
        pathway_names = sorted(all_pathways.keys())
        pathway_tensor = np.stack([
            np.concatenate(all_pathways[p]) for p in pathway_names
        ], axis=1)
        tensors["pathway_scores"] = torch.from_numpy(pathway_tensor).float()
        tensors["pathway_names"] = np.array(pathway_names)
        print(f"  - Pathway scores shape: {tensors['pathway_scores'].shape}")
        print(f"  - Pathways: {pathway_names}")

    torch.save(tensors, output_dir / "semi_synthetic.pt")
    print(f"Saved consolidated tensors to {output_dir / 'semi_synthetic.pt'}")
    print(f"  - Expression shape: {tensors['expression'].shape}")
    print(f"  - Positions shape: {tensors['positions'].shape}")
    print(f"  - Cell types: {list(unique_celltypes)}")
    print(f"  - Stages: {list(unique_stages)}")

# Save ground truth summary
ground_truth = report.to_dict()
with open(output_dir / "ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2, default=str)
print(f"Saved ground truth to {output_dir / 'ground_truth.json'}")

print(f"\nBenchmark generated: {report.n_cells_total} cells across {report.n_worlds} worlds")
print(f"DE gene sets: {len(report.de_gene_sets)}")
print("=" * 60)
