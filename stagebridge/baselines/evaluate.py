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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single baseline."""

    baseline_name: str
    split: str
    accuracy: float
    balanced_accuracy: float
    f1_macro: float


def load_benchmark_tensors(benchmark_dir: Path) -> dict:
    """Load benchmark from semi_synthetic.pt file.

    Args:
        benchmark_dir: Path to benchmark directory containing semi_synthetic.pt

    Returns:
        Dictionary with tensors: expression, positions, stage_idx, is_interacting, etc.
    """
    benchmark_path = benchmark_dir / "semi_synthetic.pt"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    log.info(f"Loading benchmark from {benchmark_path}")
    tensors = torch.load(benchmark_path, map_location="cpu", weights_only=False)

    log.info(f"  Expression: {tensors['expression'].shape}")
    log.info(f"  Stages: {tensors.get('stage_names', 'N/A')}")

    return tensors


def create_splits(tensors: dict, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """Split benchmark tensors into train/val/test.

    Args:
        tensors: Dictionary from load_benchmark_tensors
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data, test_data) TensorDatasets
    """
    expression = tensors["expression"]
    positions = tensors["positions"]
    stage_idx = tensors["stage_idx"]

    n_samples = len(expression)
    indices = np.arange(n_samples)

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=stage_idx.numpy()
    )

    # Second split: train vs val
    val_size = val_ratio / (1 - test_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, random_state=seed,
        stratify=stage_idx[train_val_idx].numpy()
    )

    log.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    def make_dataset(idx):
        return TensorDataset(
            expression[idx],
            positions[idx],
            stage_idx[idx],
        )

    return make_dataset(train_idx), make_dataset(val_idx), make_dataset(test_idx)


class SimpleBaseline(nn.Module):
    """Simple MLP baseline for stage classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class PoolingMLPBaseline(nn.Module):
    """Mean pooling + MLP baseline."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        # x: [batch, features]
        h = self.encoder(x)
        return self.classifier(h)


class DeepSetsBaseline(nn.Module):
    """DeepSets-style baseline with permutation invariance."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 5):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        h = self.phi(x)
        return self.rho(h)


class SetTransformerBaseline(nn.Module):
    """Simplified Set Transformer baseline."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 5, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        # x: [batch, features]
        h = self.embed(x).unsqueeze(1)  # [batch, 1, hidden]
        h, _ = self.attn(h, h, h)
        h = h.squeeze(1)  # [batch, hidden]
        return self.classifier(h)


class GraphSAGEBaseline(nn.Module):
    """Simplified GraphSAGE-style baseline (no actual graph, just spatial-aware)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 5):
        super().__init__()
        # Include position info
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),  # +2 for x,y coords
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, coords=None):
        if coords is not None:
            x = torch.cat([x, coords], dim=-1)
        h = self.encoder(x)
        return self.classifier(h)


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    use_coords: bool = False,
) -> nn.Module:
    """Train a baseline model.

    Args:
        model: Baseline model
        train_loader: Training data
        val_loader: Validation data
        device: Device
        epochs: Number of epochs
        lr: Learning rate
        use_coords: Whether model uses coordinates

    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            expr, coords, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            if use_coords:
                logits = model(expr, coords)
            else:
                logits = model(expr)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                expr, coords, labels = [b.to(device) for b in batch]
                if use_coords:
                    logits = model(expr, coords)
                else:
                    logits = model(expr)
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model


def evaluate_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_coords: bool = False,
) -> tuple[float, float, float]:
    """Evaluate baseline on a dataloader.

    Returns:
        Tuple of (accuracy, balanced_accuracy, f1_macro)
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            expr, coords, labels = [b.to(device) for b in batch]
            if use_coords:
                logits = model(expr, coords)
            else:
                logits = model(expr)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        accuracy_score(all_labels, all_preds),
        balanced_accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="macro", zero_division=0),
    )


def main():
    """CLI entry point for baseline evaluation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate baselines on semi-synthetic benchmark")
    parser.add_argument("--data_dir", type=Path, required=True, help="Canonical data directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--baseline", type=str, required=True,
                        choices=["pooling_mlp", "deep_sets", "set_transformer", "graph_sage"],
                        help="Baseline to evaluate")
    parser.add_argument("--validation_fold", type=int, default=0, help="Validation fold (for seed variation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info(f"Baseline: {args.baseline}")
    log.info(f"Fold: {args.validation_fold}, Seed: {args.seed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    benchmark_dir = args.data_dir / "benchmark"
    if not benchmark_dir.exists():
        log.error(f"Benchmark directory not found: {benchmark_dir}")
        raise FileNotFoundError(f"Run semi_synthetic_benchmark rule first: {benchmark_dir}")

    tensors = load_benchmark_tensors(benchmark_dir)

    # Create splits (use seed + fold for variation)
    effective_seed = args.seed + args.validation_fold * 1000
    train_data, val_data, test_data = create_splits(tensors, seed=effective_seed)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # Get dimensions
    input_dim = tensors["expression"].shape[1]
    n_classes = len(tensors.get("stage_names", [])) or int(tensors["stage_idx"].max() + 1)
    log.info(f"Input dim: {input_dim}, Classes: {n_classes}")

    # Create baseline model
    use_coords = args.baseline == "graph_sage"

    if args.baseline == "pooling_mlp":
        model = PoolingMLPBaseline(input_dim, n_classes=n_classes)
    elif args.baseline == "deep_sets":
        model = DeepSetsBaseline(input_dim, n_classes=n_classes)
    elif args.baseline == "set_transformer":
        model = SetTransformerBaseline(input_dim, n_classes=n_classes)
    elif args.baseline == "graph_sage":
        model = GraphSAGEBaseline(input_dim, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    log.info("Training...")
    model = train_baseline(
        model, train_loader, val_loader, device,
        epochs=args.epochs, use_coords=use_coords
    )

    # Evaluate
    log.info("Evaluating...")
    val_acc, val_bal_acc, val_f1 = evaluate_baseline(model, val_loader, device, use_coords)
    test_acc, test_bal_acc, test_f1 = evaluate_baseline(model, test_loader, device, use_coords)

    log.info(f"Validation: acc={val_acc:.4f}, bal_acc={val_bal_acc:.4f}, f1={val_f1:.4f}")
    log.info(f"Test: acc={test_acc:.4f}, bal_acc={test_bal_acc:.4f}, f1={test_f1:.4f}")

    # Save results
    results = {
        "baseline": args.baseline,
        "fold": args.validation_fold,
        "seed": args.seed,
        "accuracy": float(test_acc),
        "balanced_accuracy": float(test_bal_acc),
        "f1_macro": float(test_f1),
        "val_accuracy": float(val_acc),
        "val_balanced_accuracy": float(val_bal_acc),
        "val_f1_macro": float(val_f1),
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save metrics CSV
    metrics_df = pd.DataFrame([
        {"split": "val", "accuracy": val_acc, "balanced_accuracy": val_bal_acc, "f1_macro": val_f1},
        {"split": "test", "accuracy": test_acc, "balanced_accuracy": test_bal_acc, "f1_macro": test_f1},
    ])
    metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)

    # Save model checkpoint
    torch.save(model.state_dict(), args.output_dir / "model.pt")

    log.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
