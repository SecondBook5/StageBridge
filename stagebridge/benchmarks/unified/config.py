"""
Unified configuration for StageBridge benchmarks.

Combines synthetic_v2 config (dynamics, ground truth) with semi-synthetic
config (interaction rules, cell groups, real data sources).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class NicheInfluenceSpec:
    """Specification for causal niche influence (Suite B ground truth).

    Defines how sender cells influence receiver cell state in latent space.
    This is the core ground truth to be recovered by models.
    """

    sender_group: str
    influence_name: str
    latent_direction: list[float] | None = None  # Set during generation
    strength: float = 0.5
    radius: float = 100.0
    stage_modulation: dict[str, float] | None = None

    def get_effective_strength(self, stage: str) -> float:
        """Get strength modulated by stage."""
        if self.stage_modulation is None:
            return self.strength
        return self.strength * self.stage_modulation.get(stage, 1.0)


@dataclass
class InteractionRule:
    """Explicit sender->receiver interaction rule.

    Compatible with semi-synthetic benchmark but now linked to
    NicheInfluenceSpec for causal dynamics.
    """

    rule_id: str
    sender_group: str
    receiver_group: str
    interaction_radius: float
    effect_strength: float
    effect_name: str
    stage_modulation: dict[str, float] | None = None
    niche_influence: NicheInfluenceSpec | None = None  # Link to causal dynamics

    def get_stage_effect(self, stage: str) -> float:
        """Get effective strength for a given stage."""
        if self.stage_modulation is None:
            return self.effect_strength
        return self.effect_strength * self.stage_modulation.get(stage, 1.0)


@dataclass
class CellGroupSpec:
    """Specification for a cell group in the benchmark."""

    name: str
    role: Literal["receiver", "sender", "background"]
    source_datasets: list[str] = field(default_factory=lambda: ["synthetic"])
    selection_keywords: list[str] = field(default_factory=list)
    min_cells: int = 100
    max_cells: int = 5000
    stage_filter: list[str] | None = None

    # Synthetic-specific parameters
    base_expression_profile: str | None = None  # Cell type for expression basis
    latent_position_bias: list[float] | None = None  # Bias in latent space


@dataclass
class DynamicsConfig:
    """Configuration for ODE-based transition dynamics (Suite A ground truth)."""

    drift_strength: float = 1.0
    diffusion_strength: float = 0.2
    flow_field_type: Literal["linear", "radial", "branching"] = "linear"

    # Clone/evolutionary structure (Suite C)
    n_clones_per_donor: int = 3
    clone_divergence: float = 0.3

    # Reference geometry
    hlca_luca_rotation: float = 0.523  # ~30 degrees
    hlca_luca_shift: float = 2.0


@dataclass
class UnifiedBenchmarkConfig:
    """Complete configuration for unified StageBridge benchmark.

    Supports three generation modes:
    - fully_synthetic: Pure synthetic data with full ground truth recovery
    - semi_synthetic: Real expression profiles with synthetic spatial structure
    - hybrid: Real profiles with causal niche dynamics applied to latent space
    """

    # Benchmark identity
    benchmark_name: str = "unified_v1"
    benchmark_family: Literal["local_niche", "stage_progression", "dual_reference"] = "local_niche"

    # Generation mode
    mode: Literal["fully_synthetic", "semi_synthetic", "hybrid"] = "hybrid"
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    # Data sources (for semi-synthetic/hybrid modes)
    hlca_path: Path | None = None
    luca_path: Path | None = None
    progression_path: Path | None = None

    # Core dimensions
    n_cells: int = 2000
    n_donors: int = 10
    n_stages: int = 5
    latent_dim: int = 32
    n_genes: int = 2000
    n_celltypes: int = 8

    # Feature harmonization
    n_hvg: int = 2000
    use_shared_genes_only: bool = True

    # Stage structure
    stages: list[str] = field(default_factory=lambda: ["Normal", "AAH", "AIS", "MIA", "LUAD"])

    # Spatial structure
    k_neighbors: int = 20
    n_rings: int = 4
    spatial_scale: float = 1000.0
    world_width: float = 1000.0
    world_height: float = 1000.0

    # World generation
    n_worlds_train: int = 10
    n_worlds_val: int = 3
    n_worlds_test: int = 5
    cells_per_world: int = 2000

    # Cell groups
    cell_groups: list[CellGroupSpec] = field(default_factory=list)

    # Interaction rules (links to niche influence)
    interaction_rules: list[InteractionRule] = field(default_factory=list)

    # Dynamics
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)

    # Output
    output_dir: Path = field(default_factory=lambda: Path("data/processed/unified_benchmark"))

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Set up defaults and adjust for difficulty."""
        if not self.cell_groups:
            self.cell_groups = self._default_cell_groups()
        if not self.interaction_rules:
            self.interaction_rules = self._default_interaction_rules()

        self._adjust_for_difficulty()

    def _adjust_for_difficulty(self) -> None:
        """Adjust parameters based on difficulty level."""
        if self.difficulty == "easy":
            self.dynamics.diffusion_strength = 0.1
            self.dynamics.clone_divergence = 0.5
            for rule in self.interaction_rules:
                rule.effect_strength = min(1.0, rule.effect_strength * 1.5)
        elif self.difficulty == "hard":
            self.dynamics.diffusion_strength = 0.4
            self.dynamics.clone_divergence = 0.15
            for rule in self.interaction_rules:
                rule.effect_strength = rule.effect_strength * 0.6

    def _default_cell_groups(self) -> list[CellGroupSpec]:
        """Default cell groups aligned with StageBridge doctrine."""
        return [
            # Receivers: epithelial cells that can transition
            CellGroupSpec(
                name="epithelial_receiver",
                role="receiver",
                source_datasets=["hlca", "progression", "synthetic"],
                selection_keywords=["AT2", "alveolar", "epithelial", "EPCAM"],
                min_cells=500,
                max_cells=3000,
                base_expression_profile="AT2",
            ),
            # Senders: CAF-like fibroblasts
            CellGroupSpec(
                name="caf_sender",
                role="sender",
                source_datasets=["luca", "progression", "synthetic"],
                selection_keywords=["fibroblast", "CAF", "stromal", "mesenchymal"],
                min_cells=200,
                max_cells=1000,
                base_expression_profile="Fibroblast",
            ),
            # Senders: immune cells
            CellGroupSpec(
                name="immune_sender",
                role="sender",
                source_datasets=["hlca", "luca", "progression", "synthetic"],
                selection_keywords=["macrophage", "T cell", "immune", "myeloid"],
                min_cells=200,
                max_cells=1000,
                base_expression_profile="Macrophage",
            ),
            # Background: endothelial and other
            CellGroupSpec(
                name="endothelial_background",
                role="background",
                source_datasets=["hlca", "luca", "synthetic"],
                selection_keywords=["endothelial", "vascular", "capillary"],
                min_cells=100,
                max_cells=500,
                base_expression_profile="Endothelial",
            ),
        ]

    def _default_interaction_rules(self) -> list[InteractionRule]:
        """Default interaction rules with linked niche influence."""
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
                niche_influence=NicheInfluenceSpec(
                    sender_group="caf_sender",
                    influence_name="EMT_direction",
                    strength=0.6,
                    radius=100.0,
                    stage_modulation={
                        "Normal": 0.3,
                        "AAH": 0.5,
                        "AIS": 0.8,
                        "MIA": 1.0,
                        "LUAD": 1.2,
                    },
                ),
            ),
            # Immune-modulated transition
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
                niche_influence=NicheInfluenceSpec(
                    sender_group="immune_sender",
                    influence_name="immune_direction",
                    strength=0.4,
                    radius=75.0,
                    stage_modulation={
                        "Normal": 0.2,
                        "AAH": 0.4,
                        "AIS": 0.6,
                        "MIA": 0.8,
                        "LUAD": 1.0,
                    },
                ),
            ),
            # Distant CAF effect (tests radius sensitivity)
            InteractionRule(
                rule_id="caf_distant",
                sender_group="caf_sender",
                receiver_group="epithelial_receiver",
                interaction_radius=200.0,
                effect_strength=0.2,
                effect_name="CAF_distant_effect",
                niche_influence=NicheInfluenceSpec(
                    sender_group="caf_sender",
                    influence_name="distant_effect",
                    strength=0.2,
                    radius=200.0,
                ),
            ),
        ]

    @property
    def is_synthetic(self) -> bool:
        """Check if using fully synthetic mode."""
        return self.mode == "fully_synthetic"

    @property
    def uses_real_data(self) -> bool:
        """Check if using real data (semi_synthetic or hybrid)."""
        return self.mode in ("semi_synthetic", "hybrid")

    @property
    def applies_causal_dynamics(self) -> bool:
        """Check if causal niche dynamics should be applied."""
        return self.mode in ("fully_synthetic", "hybrid")


@dataclass
class SmokeTestConfig(UnifiedBenchmarkConfig):
    """Minimal configuration for smoke testing."""

    benchmark_name: str = "smoke_test"
    mode: Literal["fully_synthetic", "semi_synthetic", "hybrid"] = "fully_synthetic"
    n_hvg: int = 500
    latent_dim: int = 16
    n_cells: int = 400
    n_donors: int = 4
    n_worlds_train: int = 2
    n_worlds_val: int = 1
    n_worlds_test: int = 1
    cells_per_world: int = 200
    world_width: float = 500.0
    world_height: float = 500.0

    def __post_init__(self):
        super().__post_init__()
        # Reduce cell counts for smoke testing
        for group in self.cell_groups:
            group.min_cells = min(group.min_cells, 50)
            group.max_cells = min(group.max_cells, 200)


@dataclass
class FullBenchmarkConfig(UnifiedBenchmarkConfig):
    """Full-scale configuration for comprehensive evaluation."""

    benchmark_name: str = "unified_full_v1"
    mode: Literal["fully_synthetic", "semi_synthetic", "hybrid"] = "hybrid"
    n_hvg: int = 3000
    latent_dim: int = 64
    n_cells: int = 10000
    n_donors: int = 20
    n_worlds_train: int = 20
    n_worlds_val: int = 5
    n_worlds_test: int = 10
    cells_per_world: int = 3000


def load_config_from_yaml(path: Path) -> UnifiedBenchmarkConfig:
    """Load benchmark configuration from YAML file."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse cell groups
    cell_groups = []
    for g in data.get("cell_groups", []):
        cell_groups.append(CellGroupSpec(**g))

    # Parse interaction rules with niche influence
    interaction_rules = []
    for r in data.get("interaction_rules", []):
        niche_data = r.pop("niche_influence", None)
        niche_influence = NicheInfluenceSpec(**niche_data) if niche_data else None
        interaction_rules.append(InteractionRule(**r, niche_influence=niche_influence))

    # Parse dynamics
    dynamics_data = data.get("dynamics", {})
    dynamics = DynamicsConfig(**dynamics_data) if dynamics_data else DynamicsConfig()

    return UnifiedBenchmarkConfig(
        benchmark_name=data.get("benchmark_name", "benchmark"),
        benchmark_family=data.get("benchmark_family", "local_niche"),
        mode=data.get("mode", "hybrid"),
        difficulty=data.get("difficulty", "medium"),
        hlca_path=Path(data["hlca_path"]) if data.get("hlca_path") else None,
        luca_path=Path(data["luca_path"]) if data.get("luca_path") else None,
        progression_path=Path(data["progression_path"]) if data.get("progression_path") else None,
        n_hvg=data.get("n_hvg", 2000),
        latent_dim=data.get("latent_dim", 32),
        stages=data.get("stages", ["Normal", "AAH", "AIS", "MIA", "LUAD"]),
        cell_groups=cell_groups,
        interaction_rules=interaction_rules,
        dynamics=dynamics,
        n_worlds_train=data.get("n_worlds_train", 10),
        n_worlds_val=data.get("n_worlds_val", 3),
        n_worlds_test=data.get("n_worlds_test", 5),
        cells_per_world=data.get("cells_per_world", 2000),
        output_dir=Path(data.get("output_dir", "data/processed/unified_benchmark")),
        seed=data.get("seed", 42),
    )
