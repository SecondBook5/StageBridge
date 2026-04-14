#!/usr/bin/env python3
"""Evaluate set-based baselines on neighborhood classification.

This properly tests H2: "Cross-sectional progression becomes more
identifiable when conditioned on receiver-centered local niche context."

Each sample is a NEIGHBORHOOD (9 tokens), not a single cell.
Task: Classify the receiver's stage based on its niche context.

Usage:
    python -m stagebridge.baselines.evaluate_set_baselines \
        --data_dir /path/to/canonical \
        --output_dir /path/to/results \
        --baseline pooling_mlp \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from stagebridge.baselines.set_baselines import create_baseline, BASELINE_REGISTRY
from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


class NeighborhoodDataset(Dataset):
    """Dataset of neighborhoods for set-based classification.

    Each sample is a 9-token neighborhood with the receiver's stage as label.
    """

    def __init__(
        self,
        neighborhoods_df: pd.DataFrame,
        stage_to_idx: dict[str, int] | None = None,
    ):
        """
        Args:
            neighborhoods_df: DataFrame with columns [cell_id, donor_id, stage, tokens]
            stage_to_idx: Mapping from stage name to index
        """
        self.neighborhoods = neighborhoods_df.reset_index(drop=True)

        # Create stage mapping if not provided
        if stage_to_idx is None:
            stages = sorted(self.neighborhoods["stage"].unique())
            self.stage_to_idx = {s: i for i, s in enumerate(stages)}
        else:
            self.stage_to_idx = stage_to_idx

        self.idx_to_stage = {i: s for s, i in self.stage_to_idx.items()}

    def __len__(self) -> int:
        return len(self.neighborhoods)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens: [K, D] token embeddings
            stage_idx: scalar stage index
            mask: [K] valid token mask
        """
        row = self.neighborhoods.iloc[idx]
        tokens_list = row["tokens"]

        # Extract embeddings from tokens
        embeddings = []
        mask = []
        for tok in tokens_list:
            z = tok.get("z_fused")
            if z is not None and np.array(z).sum() != 0:
                embeddings.append(np.array(z, dtype=np.float32))
                mask.append(True)
            else:
                # Placeholder for missing embedding
                embeddings.append(np.zeros(40, dtype=np.float32))
                mask.append(False)

        tokens = torch.tensor(np.stack(embeddings), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        stage_idx = torch.tensor(self.stage_to_idx[row["stage"]], dtype=torch.long)

        return tokens, stage_idx, mask


def collate_neighborhoods(batch):
    """Collate function for neighborhood batches."""
    tokens, stages, masks = zip(*batch)
    return (
        torch.stack(tokens),  # [B, K, D]
        torch.stack(stages),  # [B]
        torch.stack(masks),   # [B, K]
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for tokens, stages, masks in dataloader:
        tokens = tokens.to(device)
        stages = stages.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        output = model(tokens, mask=masks)
        loss = criterion(output.logits, stages)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on dataloader."""
    model.eval()
    all_preds = []
    all_labels = []

    for tokens, stages, masks in dataloader:
        tokens = tokens.to(device)
        masks = masks.to(device)

        output = model(tokens, mask=masks)
        preds = output.logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(stages.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def train_and_evaluate(
    baseline_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage_to_idx: dict[str, int],
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
) -> dict:
    """Train a baseline and evaluate on val/test sets."""
    # Create datasets
    train_dataset = NeighborhoodDataset(train_df, stage_to_idx)
    val_dataset = NeighborhoodDataset(val_df, stage_to_idx)
    test_dataset = NeighborhoodDataset(test_df, stage_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_neighborhoods,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_neighborhoods,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_neighborhoods,
        num_workers=0,
    )

    # Create model
    model = create_baseline(
        baseline_name,
        input_dim=40,
        hidden_dim=128,
        num_classes=len(stage_to_idx),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Training loop with early stopping
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["balanced_accuracy"])

        if val_metrics["balanced_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["balanced_accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            log.info(
                f"  Epoch {epoch+1}: loss={train_loss:.4f}, "
                f"val_acc={val_metrics['accuracy']:.4f}, "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
            )

        if patience_counter >= patience:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    return {
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_f1_macro": val_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "best_epoch": epochs - patience_counter,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate set-based baselines")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing neighborhoods.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=list(BASELINE_REGISTRY.keys()),
        help="Baseline to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Baseline: {args.baseline}")
    log.info(f"Fold: {args.fold}/{args.n_folds}, Seed: {args.seed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load neighborhoods
    neighborhoods_path = args.data_dir / "neighborhoods.parquet"
    if not neighborhoods_path.exists():
        # Try results directory
        neighborhoods_path = Path("results/neighborhoods.parquet")

    if not neighborhoods_path.exists():
        raise FileNotFoundError(f"neighborhoods.parquet not found in {args.data_dir}")

    log.info(f"Loading neighborhoods from {neighborhoods_path}")
    neighborhoods = pd.read_parquet(neighborhoods_path)
    log.info(f"  Loaded {len(neighborhoods)} neighborhoods")

    # Check for valid embeddings
    sample_tokens = neighborhoods.iloc[0]["tokens"]
    sample_z = sample_tokens[0].get("z_fused")
    if sample_z is None or np.array(sample_z).sum() == 0:
        log.warning("WARNING: z_fused embeddings appear to be missing or all zeros!")
        log.warning("You may need to regenerate neighborhoods.parquet with the fixed pipeline.")

    # Create stage mapping
    stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    stage_to_idx = {s: i for i, s in enumerate(stages)}

    # Filter to known stages
    neighborhoods = neighborhoods[neighborhoods["stage"].isin(stages)].reset_index(drop=True)
    log.info(f"  Filtered to {len(neighborhoods)} neighborhoods with known stages")

    # Donor-held-out cross-validation
    donors = neighborhoods["donor_id"].unique()
    log.info(f"  {len(donors)} unique donors")

    # Split donors into folds
    kfold = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    # Get donor-level stage (majority stage per donor)
    donor_stages = neighborhoods.groupby("donor_id")["stage"].agg(
        lambda x: x.value_counts().index[0]
    )

    fold_idx = 0
    for train_donors_idx, test_donors_idx in kfold.split(donors, [donor_stages[d] for d in donors]):
        if fold_idx == args.fold:
            train_donors = donors[train_donors_idx]
            test_donors = donors[test_donors_idx]
            break
        fold_idx += 1

    # Further split train into train/val (by donor)
    val_size = max(1, len(train_donors) // 5)
    val_donors = train_donors[:val_size]
    train_donors = train_donors[val_size:]

    log.info(f"  Train donors: {len(train_donors)}, Val donors: {len(val_donors)}, Test donors: {len(test_donors)}")

    train_df = neighborhoods[neighborhoods["donor_id"].isin(train_donors)]
    val_df = neighborhoods[neighborhoods["donor_id"].isin(val_donors)]
    test_df = neighborhoods[neighborhoods["donor_id"].isin(test_donors)]

    log.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Train and evaluate
    log.info(f"Training {args.baseline}...")
    results = train_and_evaluate(
        baseline_name=args.baseline,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        stage_to_idx=stage_to_idx,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Add metadata
    results["baseline"] = args.baseline
    results["fold"] = args.fold
    results["seed"] = args.seed
    results["n_train"] = len(train_df)
    results["n_val"] = len(val_df)
    results["n_test"] = len(test_df)

    # Save results
    output_file = args.output_dir / f"{args.baseline}_fold{args.fold}_seed{args.seed}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved to {output_file}")
    log.info(f"  Test accuracy: {results['test_accuracy']:.4f}")
    log.info(f"  Test balanced accuracy: {results['test_balanced_accuracy']:.4f}")
    log.info(f"  Test F1 macro: {results['test_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
