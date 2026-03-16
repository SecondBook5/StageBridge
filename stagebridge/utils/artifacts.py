"""Run artifact management: save/load config, metrics, and model summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunArtifacts:
    """Structured record of a pipeline run's outputs."""

    run_id: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    status: str = "PASS"
    timestamp_utc: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_utc is None:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()

    def save(self, artifacts_dir: str | Path) -> Path:
        """Write artifacts to disk. Returns the artifacts directory."""
        d = Path(artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)

        with (d / "config.yaml").open("w") as f:
            import yaml

            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        with (d / "metrics.json").open("w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        with (d / "pipeline_artifacts.json").open("w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

        return d

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> "RunArtifacts":
        """Load artifacts from disk."""
        d = Path(artifacts_dir)
        with (d / "pipeline_artifacts.json").open() as f:
            data = json.load(f)
        return cls(**data)
