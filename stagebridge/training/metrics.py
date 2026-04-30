"""Metrics logging and tracking."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from stagebridge.training.distributed import is_main_process


class MetricsLogger:
    """Log and track training metrics.

    Handles:
    - Per-epoch metric accumulation
    - Best metric tracking
    - JSON serialization
    """

    def __init__(self, log_dir: Path | str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[float]] = defaultdict(list)
        self.best_metrics: dict[str, float] = {}
        self.current_epoch_metrics: dict[str, list[float]] = defaultdict(list)

    def log(self, name: str, value: float):
        """Log a metric value for the current epoch."""
        self.current_epoch_metrics[name].append(value)

    def end_epoch(self, epoch: int) -> dict[str, float]:
        """Finalize epoch metrics and return averages."""
        epoch_averages = {}

        for name, values in self.current_epoch_metrics.items():
            if values:
                avg = sum(values) / len(values)
                self.history[name].append(avg)
                epoch_averages[name] = avg

        self.current_epoch_metrics = defaultdict(list)
        return epoch_averages

    def update_best(self, metric_name: str, value: float, mode: str = "min") -> bool:
        """Update best metric if improved.

        Returns:
            True if this is a new best
        """
        current_best = self.best_metrics.get(metric_name)

        if current_best is None:
            self.best_metrics[metric_name] = value
            return True

        if mode == "min" and value < current_best:
            self.best_metrics[metric_name] = value
            return True
        elif mode == "max" and value > current_best:
            self.best_metrics[metric_name] = value
            return True

        return False

    def save(self, filename: str = "metrics.json"):
        """Save metrics history to JSON."""
        if not is_main_process():
            return

        filepath = self.log_dir / filename
        data = {
            "history": dict(self.history),
            "best": self.best_metrics,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filename: str = "metrics.json"):
        """Load metrics history from JSON."""
        filepath = self.log_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            self.history = defaultdict(list, data.get("history", {}))
            self.best_metrics = data.get("best", {})

    def get_summary(self) -> dict[str, Any]:
        """Get summary of training metrics."""
        return {
            "n_epochs": len(next(iter(self.history.values()), [])),
            "best_metrics": self.best_metrics,
            "final_metrics": {k: v[-1] if v else None for k, v in self.history.items()},
        }
