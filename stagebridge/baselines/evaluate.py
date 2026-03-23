"""Unified baseline evaluation on semi-synthetic benchmark.

Tests the core novelty claim:
"Cross-sectional progression becomes more identifiable when cell representations
are conditioned on receiver-centered local niche context."

Baseline ladder:
1. PoolingMLP - No structure (bag-of-cells)
2. DeepSets - Permutation invariance only
3. SetTransformer - Flat attention without spatial structure
4. GraphSAGE - Spatial graph structure
5. StageBridge - Receiver-centered niche + dual references (full model)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anndata
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class BenchmarkWorld:
    """Loaded semi-synthetic world for evaluation."""

    world_id: str
    split: Literal["train", "val", "test"]
    expression: anndata.AnnData  # [n_cells, n_genes]
    coordinates: pd.DataFrame  # columns: synthetic_cell_id, x, y
    ground_truth: pd.DataFrame  # columns: stage, cell_group, is_interacting, etc.
    metadata: dict


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single baseline."""

    baseline_name: str
    split: str

    # Stage prediction
    stage_accuracy: float
    stage_balanced_accuracy: float
    stage_f1_macro: float

    # Transition quality (if applicable)
    transition_l2: float | None = None
    transition_cosine: float | None = None

    # Niche awareness (if applicable)
    niche_consistency: float | None = None
    interaction_detection_auc: float | None = None


def load_benchmark_world(world_dir: Path) -> BenchmarkWorld:
    """Load a single world from exported benchmark."""
    import json

    # Load expression
    expr_path = world_dir / "expression.h5ad"
    if not expr_path.exists():
        raise FileNotFoundError(f"Expression data not found: {expr_path}")
    expression = anndata.read_h5ad(expr_path)

    # Load coordinates
    coords_path = world_dir / "coordinates.parquet"
    coordinates = pd.read_parquet(coords_path)

    # Load ground truth
    gt_path = world_dir / "ground_truth.parquet"
    ground_truth = pd.read_parquet(gt_path)

    # Load metadata
    meta_path = world_dir / "world_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    # Infer world_id and split from path
    world_id = world_dir.name
    split = world_dir.parent.name

    return BenchmarkWorld(
        world_id=world_id,
        split=split,
        expression=expression,
        coordinates=coordinates,
        ground_truth=ground_truth,
        metadata=metadata,
    )


def load_benchmark_split(
    benchmark_dir: Path, split: Literal["train", "val", "test"]
) -> list[BenchmarkWorld]:
    """Load all worlds from a split."""
    split_dir = benchmark_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    worlds = []
    for world_dir in sorted(split_dir.glob("world_*")):
        try:
            world = load_benchmark_world(world_dir)
            worlds.append(world)
        except Exception as e:
            log.warning(f"Failed to load {world_dir.name}: {e}")

    log.info(f"Loaded {len(worlds)} worlds from {split} split")
    return worlds


class BenchmarkDataset(Dataset):
    """PyTorch dataset wrapper for benchmark worlds."""

    def __init__(self, worlds: list[BenchmarkWorld]):
        self.worlds = worlds

    def __len__(self) -> int:
        return len(self.worlds)

    def __getitem__(self, idx: int) -> dict:
        world = self.worlds[idx]

        # Expression matrix
        X = torch.from_numpy(world.expression.X).float()

        # Coordinates
        coords = torch.from_numpy(world.coordinates[["x", "y"]].values).float()

        # Stage labels (if available)
        if "stage" in world.ground_truth.columns:
            # Map stage names to indices
            stage_map = {"Normal": 0, "AAH": 1, "AIS": 2, "MIA": 3, "LUAD": 4}
            stages = world.ground_truth["stage"].map(stage_map).values
            stage_labels = torch.from_numpy(stages).long()
        else:
            stage_labels = torch.zeros(X.shape[0], dtype=torch.long)

        # Interaction labels (if available)
        if "is_interacting" in world.ground_truth.columns:
            interactions = torch.from_numpy(world.ground_truth["is_interacting"].values).float()
        else:
            interactions = torch.zeros(X.shape[0])

        return {
            "expression": X,
            "coordinates": coords,
            "stage_labels": stage_labels,
            "interactions": interactions,
            "world_id": world.world_id,
        }


def evaluate_baseline(
    model: torch.nn.Module,
    dataloader: DataLoader,
    split: str,
    device: torch.device,
) -> EvaluationMetrics:
    """Evaluate a baseline model on a dataloader.

    Args:
        model: Baseline model (must have forward method returning stage_logits)
        dataloader: DataLoader of BenchmarkDataset
        split: "train", "val", or "test"
        device: Device to run on

    Returns:
        EvaluationMetrics with all computed metrics
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating on {split}"):
            x = batch["expression"].to(device)
            coords = batch["coordinates"].to(device)
            labels = batch["stage_labels"].to(device)

            # Forward pass (model-specific interface)
            if hasattr(model, "forward"):
                output = model(x.unsqueeze(0), coords.unsqueeze(0))  # Add batch dim
                if isinstance(output, dict):
                    logits = output["stage_logits"]
                else:
                    logits = output.stage_logits
            else:
                raise AttributeError(f"Model {type(model)} has no forward method")

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return EvaluationMetrics(
        baseline_name=model.__class__.__name__,
        split=split,
        stage_accuracy=accuracy,
        stage_balanced_accuracy=balanced_acc,
        stage_f1_macro=f1_macro,
    )


def run_baseline_comparison(
    benchmark_dir: Path,
    output_dir: Path,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Run all baselines on benchmark and return comparison table.

    Args:
        benchmark_dir: Path to exported benchmark directory
        output_dir: Where to save results
        device: Device to run on (auto-detect if None)

    Returns:
        DataFrame with metrics for all baselines
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    log.info(f"Loading benchmark from {benchmark_dir}")
    train_worlds = load_benchmark_split(benchmark_dir, "train")
    val_worlds = load_benchmark_split(benchmark_dir, "val")
    test_worlds = load_benchmark_split(benchmark_dir, "test")

    # Create dataloaders
    DataLoader(BenchmarkDataset(train_worlds), batch_size=1, shuffle=False)
    val_loader = DataLoader(BenchmarkDataset(val_worlds), batch_size=1, shuffle=False)
    test_loader = DataLoader(BenchmarkDataset(test_worlds), batch_size=1, shuffle=False)

    # Get input dimensions from first world
    sample_world = train_worlds[0]
    input_dim = sample_world.expression.shape[1]
    log.info(f"Input dimension: {input_dim} genes")

    # Initialize baselines
    from stagebridge.context_model.baselines_lesion import (
        PooledLesionBaseline,
        DeepSetsLesionBaseline,
        LesionSetTransformerBaseline,
    )
    from stagebridge.baselines.graph_sage import GraphSAGEBaseline

    baselines = {
        "PoolingMLP": PooledLesionBaseline(input_dim=input_dim, hidden_dim=128),
        "DeepSets": DeepSetsLesionBaseline(input_dim=input_dim, hidden_dim=128),
        "SetTransformer": LesionSetTransformerBaseline(input_dim=input_dim, hidden_dim=128),
        "GraphSAGE": GraphSAGEBaseline(input_dim=input_dim, hidden_dim=128),
    }

    # Evaluate each baseline
    results = []
    for name, model in baselines.items():
        log.info(f"\n{'=' * 60}")
        log.info(f"Evaluating {name}")
        log.info(f"{'=' * 60}")

        # Evaluate on validation (for model selection)
        val_metrics = evaluate_baseline(model, val_loader, "val", device)
        results.append(val_metrics)

        # Evaluate on test (for final results)
        test_metrics = evaluate_baseline(model, test_loader, "test", device)
        results.append(test_metrics)

        log.info(f"{name} Val Accuracy: {val_metrics.stage_accuracy:.3f}")
        log.info(f"{name} Test Accuracy: {test_metrics.stage_accuracy:.3f}")

    # Create results DataFrame
    results_df = pd.DataFrame(
        [
            {
                "baseline": r.baseline_name,
                "split": r.split,
                "accuracy": r.stage_accuracy,
                "balanced_accuracy": r.stage_balanced_accuracy,
                "f1_macro": r.stage_f1_macro,
            }
            for r in results
        ]
    )

    # Save results
    results_path = output_dir / "baseline_comparison.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"\nResults saved to: {results_path}")

    return results_df


def main():
    """CLI entry point for baseline evaluation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate baselines on benchmark")
    parser.add_argument("--data_dir", type=Path, required=True, help="Canonical data directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--baseline", type=str, required=True,
                        choices=["pooling_mlp", "deep_sets", "set_transformer", "graph_sage"],
                        help="Baseline to evaluate")
    parser.add_argument("--validation_fold", type=int, default=0, help="Validation fold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info(f"Baseline: {args.baseline}")
    log.info(f"Fold: {args.validation_fold}, Seed: {args.seed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark from canonical directory
    benchmark_dir = args.data_dir / "benchmark"
    if not benchmark_dir.exists():
        log.error(f"Benchmark directory not found: {benchmark_dir}")
        raise FileNotFoundError(f"Run semi_synthetic_benchmark rule first: {benchmark_dir}")

    # Run comparison (simplified for single baseline)
    results_df = run_baseline_comparison(
        benchmark_dir=benchmark_dir,
        output_dir=args.output_dir,
        device=device,
    )

    # Filter to requested baseline
    baseline_map = {
        "pooling_mlp": "PoolingMLP",
        "deep_sets": "DeepSets",
        "set_transformer": "SetTransformer",
        "graph_sage": "GraphSAGE",
    }
    baseline_name = baseline_map[args.baseline]
    baseline_results = results_df[results_df["baseline"] == baseline_name]

    # Save results JSON
    test_row = baseline_results[baseline_results["split"] == "test"].iloc[0]
    results = {
        "baseline": args.baseline,
        "fold": args.validation_fold,
        "seed": args.seed,
        "accuracy": float(test_row["accuracy"]),
        "balanced_accuracy": float(test_row["balanced_accuracy"]),
        "f1_macro": float(test_row["f1_macro"]),
    }

    results_path = args.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save metrics CSV
    baseline_results.to_csv(args.output_dir / "metrics.csv", index=False)

    log.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
