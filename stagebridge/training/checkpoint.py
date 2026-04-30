"""Checkpoint management for training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from stagebridge.training.distributed import is_main_process


class CheckpointManager:
    """Manages model checkpoints with versioning and pruning."""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        keep_top_k: int = 3,
        metric_name: str = "val_loss",
        mode: str = "min",
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            keep_top_k: Number of best checkpoints to keep
            metric_name: Metric to use for ranking
            mode: "min" or "max" for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric_name = metric_name
        self.mode = mode
        self.history: list[dict] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        config: dict,
        is_best: bool = False,
        auxiliary_heads: dict[str, nn.Module] | None = None,
    ) -> Path | None:
        """Save checkpoint.

        Args:
            model: Model to save (handles DDP wrapper)
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Metrics dictionary
            config: Training configuration
            is_best: Whether this is the best checkpoint
            auxiliary_heads: Optional dict of auxiliary head modules

        Returns:
            Path to saved checkpoint, or None if not main process
        """
        if not is_main_process():
            return None

        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }

        if auxiliary_heads:
            for name, head in auxiliary_heads.items():
                head_state = head.module.state_dict() if hasattr(head, "module") else head.state_dict()
                checkpoint[f"{name}_state_dict"] = head_state

        filepath = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, filepath)

        metric_value = metrics.get(self.metric_name, float("inf"))
        self.history.append({
            "path": str(filepath),
            "epoch": epoch,
            "metric_value": metric_value,
        })

        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)

        self._prune()
        return filepath

    def _prune(self):
        """Remove old checkpoints, keeping only top-k."""
        if len(self.history) <= self.keep_top_k:
            return

        sorted_history = sorted(
            self.history,
            key=lambda x: x["metric_value"],
            reverse=(self.mode == "max"),
        )

        keep_paths = {h["path"] for h in sorted_history[:self.keep_top_k]}
        keep_paths.add(self.history[-1]["path"])

        for h in self.history:
            if h["path"] not in keep_paths:
                path = Path(h["path"])
                if path.exists():
                    path.unlink()

        self.history = [h for h in self.history if h["path"] in keep_paths]

    def save_final(
        self,
        model: nn.Module,
        config: dict,
        metrics: dict,
        auxiliary_heads: dict[str, nn.Module] | None = None,
    ) -> Path | None:
        """Save final checkpoint for inference."""
        if not is_main_process():
            return None

        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        checkpoint = {
            "model_state_dict": state_dict,
            "config": config,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "is_final": True,
        }

        if auxiliary_heads:
            for name, head in auxiliary_heads.items():
                head_state = head.module.state_dict() if hasattr(head, "module") else head.state_dict()
                checkpoint[f"{name}_state_dict"] = head_state

        final_path = self.checkpoint_dir / "final_checkpoint.pt"
        torch.save(checkpoint, final_path)

        weights_dir = self.checkpoint_dir.parent / "weights"
        weights_dir.mkdir(exist_ok=True)
        torch.save(checkpoint, weights_dir / "final_model.pt")

        return final_path

    @staticmethod
    def load(checkpoint_path: Path | str, device: torch.device) -> dict:
        """Load checkpoint."""
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
