"""
Configuration classes for semi-synthetic benchmark generation.

Defines interaction rules, cell group specifications, and benchmark parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class CellGroupSpec:
    """Specification for a cell group in the benchmark."""

    name: str
    role: Literal["receiver", "sender", "background"]
    source_datasets: list[str]  # Which datasets to sample from: hlca, luca, progression
    selection_keywords: list[str]  # Keywords for cell type selection
    min_cells: int = 100
    max_cells: int = 5000
    stage_filter: list[str] | None = None  # Optional stage filtering


@dataclass
class InteractionRule:
    """Defines a sender->receiver interaction rule with explicit radius."""

    rule_id: str
    sender_group: str
    receiver_group: str
    interaction_radius: float  # In spatial units
    effect_strength: float  # 0-1, probability of state shift
    effect_name: str  # Name of the effect (e.g., "EMT_induction")
    stage_modulation: dict[str, float] | None = None  # Stage-dependent modulation

    def get_stage_effect(self, stage: str) -> float:
        """Get the effective strength for a given stage."""
        if self.stage_modulation is None:
            return self.effect_strength
        return self.effect_strength * self.stage_modulation.get(stage, 1.0)


@dataclass
class RegionSpec:
    """Specification for a spatial region in the synthetic world."""

    region_id: str
    center_x: float
    center_y: float
    radius: float
    enriched_groups: dict[str, float]  # Group name -> enrichment factor
    stage_bias: str | None = None  # If set, this region is biased toward this stage


@dataclass
class WorldConfig:
    """Configuration for a single synthetic world."""

    world_id: str
    width: float = 1000.0
    height: float = 1000.0
    n_cells: int = 2000
    regions: list[RegionSpec] = field(default_factory=list)
    seed: int = 42


@dataclass
class BenchmarkConfig:
    """Full configuration for semi-synthetic benchmark generation."""

    # Benchmark identity
    benchmark_name: str = "niche_interaction_v1"
    benchmark_family: Literal["local_niche", "stage_progression", "dual_reference"] = "local_niche"

    # Data sources
    hlca_path: Path | None = None
    luca_path: Path | None = None
    progression_path: Path | None = None

    # Feature harmonization
    n_hvg: int = 2000
    latent_dim: int = 40
    use_shared_genes_only: bool = True

    # Cell groups
    cell_groups: list[CellGroupSpec] = field(default_factory=list)

    # Interaction rules
    interaction_rules: list[InteractionRule] = field(default_factory=list)

    # World generation
    n_worlds_train: int = 10
    n_worlds_val: int = 3
    n_worlds_test: int = 5
    cells_per_world: int = 2000
    world_width: float = 1000.0
    world_height: float = 1000.0

    # Stage structure (for progression benchmark)
    stages: list[str] = field(default_factory=lambda: ["Normal", "AAH", "AIS", "MIA", "LUAD"])

    # Output
    output_dir: Path = field(default_factory=lambda: Path("data/processed/semi_synthetic"))

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Set up default cell groups and interaction rules if not provided."""
        if not self.cell_groups:
            self.cell_groups = self._default_cell_groups()
        if not self.interaction_rules:
            self.interaction_rules = self._default_interaction_rules()

    def _default_cell_groups(self) -> list[CellGroupSpec]:
        """Default cell group specification aligned with StageBridge doctrine."""
        return [
            # Receivers: epithelial cells that can transition
            CellGroupSpec(
                name="epithelial_receiver",
                role="receiver",
                source_datasets=["hlca", "progression"],
                selection_keywords=["AT2", "alveolar", "epithelial", "EPCAM"],
                min_cells=500,
                max_cells=3000,
            ),
            # Senders: CAF-like fibroblasts
            CellGroupSpec(
                name="caf_sender",
                role="sender",
                source_datasets=["luca", "progression"],
                selection_keywords=["fibroblast", "CAF", "stromal", "mesenchymal"],
                min_cells=200,
                max_cells=1000,
            ),
            # Senders: immune cells
            CellGroupSpec(
                name="immune_sender",
                role="sender",
                source_datasets=["hlca", "luca", "progression"],
                selection_keywords=["macrophage", "T cell", "immune", "myeloid"],
                min_cells=200,
                max_cells=1000,
            ),
            # Background: endothelial and other
            CellGroupSpec(
                name="endothelial_background",
                role="background",
                source_datasets=["hlca", "luca"],
                selection_keywords=["endothelial", "vascular", "capillary"],
                min_cells=100,
                max_cells=500,
            ),
        ]

    def _default_interaction_rules(self) -> list[InteractionRule]:
        """Default interaction rules for niche modeling."""
        return [
            # CAF-induced EMT-like transition
            InteractionRule(
                rule_id="caf_emt",
                sender_group="caf_sender",
                receiver_group="epithelial_receiver",
                interaction_radius=100.0,
                effect_strength=0.6,
                effect_name="CAF_induced_EMT",
                stage_modulation={
                    "Normal": 0.3,
                    "AAH": 0.5,
                    "AIS": 0.8,
                    "MIA": 1.0,
                    "LUAD": 1.2,
                },
            ),
            # Immune-modulated transition (with different radius)
            InteractionRule(
                rule_id="immune_modulation",
                sender_group="immune_sender",
                receiver_group="epithelial_receiver",
                interaction_radius=75.0,
                effect_strength=0.4,
                effect_name="immune_modulation",
                stage_modulation={
                    "Normal": 0.2,
                    "AAH": 0.4,
                    "AIS": 0.6,
                    "MIA": 0.8,
                    "LUAD": 1.0,
                },
            ),
            # Distant CAF effect (tests radius sensitivity)
            InteractionRule(
                rule_id="caf_distant",
                sender_group="caf_sender",
                receiver_group="epithelial_receiver",
                interaction_radius=200.0,
                effect_strength=0.2,
                effect_name="CAF_distant_effect",
            ),
        ]


@dataclass
class SmokeConfig(BenchmarkConfig):
    """Minimal configuration for smoke testing."""

    benchmark_name: str = "smoke_test"
    n_hvg: int = 500
    n_worlds_train: int = 2
    n_worlds_val: int = 1
    n_worlds_test: int = 1
    cells_per_world: int = 200

    def __post_init__(self):
        super().__post_init__()
        # Reduce cell counts for smoke testing
        for group in self.cell_groups:
            group.min_cells = min(group.min_cells, 50)
            group.max_cells = min(group.max_cells, 200)


def load_config_from_yaml(path: Path) -> BenchmarkConfig:
    """Load benchmark configuration from YAML file."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse cell groups
    cell_groups = []
    for g in data.get("cell_groups", []):
        cell_groups.append(CellGroupSpec(**g))

    # Parse interaction rules
    interaction_rules = []
    for r in data.get("interaction_rules", []):
        interaction_rules.append(InteractionRule(**r))

    return BenchmarkConfig(
        benchmark_name=data.get("benchmark_name", "benchmark"),
        benchmark_family=data.get("benchmark_family", "local_niche"),
        hlca_path=Path(data["hlca_path"]) if data.get("hlca_path") else None,
        luca_path=Path(data["luca_path"]) if data.get("luca_path") else None,
        progression_path=Path(data["progression_path"]) if data.get("progression_path") else None,
        n_hvg=data.get("n_hvg", 2000),
        latent_dim=data.get("latent_dim", 40),
        cell_groups=cell_groups,
        interaction_rules=interaction_rules,
        n_worlds_train=data.get("n_worlds_train", 10),
        n_worlds_val=data.get("n_worlds_val", 3),
        n_worlds_test=data.get("n_worlds_test", 5),
        cells_per_world=data.get("cells_per_world", 2000),
        output_dir=Path(data.get("output_dir", "data/processed/semi_synthetic")),
        seed=data.get("seed", 42),
    )
