"""
Ablation Registry - defines all ablation experiments for StageBridge V1.

Each ablation is a controlled experiment that removes or modifies one component
to quantify its contribution to overall model performance.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class AblationTier(Enum):
    """Ablation priority tiers."""
    TIER_1 = 1  # Required for V1 publication
    TIER_2 = 2  # V2 ablations
    TIER_3 = 3  # If time allows


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    description: str
    tier: AblationTier
    config_deltas: dict[str, Any] = field(default_factory=dict)
    baseline_name: str = "full_model"
    expected_degradation: str = "unknown"  # e.g., "significant", "moderate", "minimal"
    hypothesis: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "tier": self.tier.value,
            "config_deltas": self.config_deltas,
            "baseline_name": self.baseline_name,
            "expected_degradation": self.expected_degradation,
            "hypothesis": self.hypothesis,
        }


class AblationRegistry:
    """Registry of all defined ablations."""

    _ablations: dict[str, AblationConfig] = {}

    @classmethod
    def register(cls, config: AblationConfig) -> None:
        """Register an ablation configuration."""
        cls._ablations[config.name] = config

    @classmethod
    def get(cls, name: str) -> AblationConfig:
        """Get ablation by name."""
        if name not in cls._ablations:
            raise KeyError(f"Ablation '{name}' not found. Available: {list(cls._ablations.keys())}")
        return cls._ablations[name]

    @classmethod
    def get_tier(cls, tier: AblationTier) -> list[AblationConfig]:
        """Get all ablations in a tier."""
        return [a for a in cls._ablations.values() if a.tier == tier]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered ablation names."""
        return list(cls._ablations.keys())


# =============================================================================
# TIER 1 ABLATIONS (V1 Required)
# =============================================================================

# 1. Deterministic vs Stochastic
AblationRegistry.register(AblationConfig(
    name="deterministic_transition",
    description="Replace flow matching with deterministic transition head",
    tier=AblationTier.TIER_1,
    config_deltas={"transition_model.stochastic": False},
    expected_degradation="significant",
    hypothesis="Stochastic dynamics capture uncertainty in cross-sectional data better than deterministic mapping",
))

# 2. Niche Ablations
AblationRegistry.register(AblationConfig(
    name="no_niche",
    description="Remove niche context entirely - receiver cell only",
    tier=AblationTier.TIER_1,
    config_deltas={"context_model.use_niche": False},
    expected_degradation="significant",
    hypothesis="Local niche context is essential for accurate transition modeling",
))

AblationRegistry.register(AblationConfig(
    name="pooled_niche",
    description="Replace influence tensor with simple pooled neighborhood",
    tier=AblationTier.TIER_1,
    config_deltas={"context_model.niche_mode": "pooled"},
    expected_degradation="moderate",
    hypothesis="Influence tensor captures sender-receiver relationships better than pooling",
))

# 3. Genomics Ablations
AblationRegistry.register(AblationConfig(
    name="no_genomics",
    description="Remove genomics/WES features entirely",
    tier=AblationTier.TIER_1,
    config_deltas={"use_genomics": False},
    expected_degradation="moderate",
    hypothesis="Genomics provide evolutionary compatibility constraints",
))

AblationRegistry.register(AblationConfig(
    name="genomics_as_feature",
    description="Use genomics as concatenated feature, not constraint",
    tier=AblationTier.TIER_1,
    config_deltas={"genomics_mode": "feature"},
    expected_degradation="minimal",
    hypothesis="Genomics as explicit constraint is better than as side feature",
))

# 4. Hierarchy Ablation
AblationRegistry.register(AblationConfig(
    name="flat_pooling",
    description="Replace hierarchical set transformer with flat pooling",
    tier=AblationTier.TIER_1,
    config_deltas={"context_model.hierarchical": False},
    expected_degradation="moderate",
    hypothesis="Hierarchical structure preserves multi-scale information",
))

# 5. Reference Ablations
AblationRegistry.register(AblationConfig(
    name="hlca_only",
    description="Use only HLCA reference (healthy anchor)",
    tier=AblationTier.TIER_1,
    config_deltas={"reference.use_luca": False},
    expected_degradation="moderate",
    hypothesis="Dual reference captures both healthy and disease geometry",
))

AblationRegistry.register(AblationConfig(
    name="luca_only",
    description="Use only LuCA reference (disease anchor)",
    tier=AblationTier.TIER_1,
    config_deltas={"reference.use_hlca": False},
    expected_degradation="moderate",
    hypothesis="Dual reference captures both healthy and disease geometry",
))

# 6. Spatial Backend Ablation
AblationRegistry.register(AblationConfig(
    name="tangram_backend",
    description="Use Tangram spatial backend",
    tier=AblationTier.TIER_1,
    config_deltas={"spatial_backend": "tangram"},
    expected_degradation="minimal",
    hypothesis="Results should be robust across spatial backends",
))

AblationRegistry.register(AblationConfig(
    name="destvi_backend",
    description="Use DestVI spatial backend",
    tier=AblationTier.TIER_1,
    config_deltas={"spatial_backend": "destvi"},
    expected_degradation="minimal",
    hypothesis="Results should be robust across spatial backends",
))

# 7. Fusion Strategy Ablations
AblationRegistry.register(AblationConfig(
    name="learned_fusion",
    description="Use learned weighted fusion instead of concatenation",
    tier=AblationTier.TIER_1,
    config_deltas={
        "fusion.method": "learned",
        "fusion.learned_init_hlca_weight": 0.5,
    },
    expected_degradation="minimal",
    hypothesis="Learned fusion may improve over fixed concatenation by adapting weights",
))

AblationRegistry.register(AblationConfig(
    name="weighted_fusion",
    description="Use confidence-weighted fusion instead of concatenation",
    tier=AblationTier.TIER_1,
    config_deltas={"fusion.method": "weighted"},
    expected_degradation="minimal",
    hypothesis="Confidence-weighted fusion uses mapping quality to weight references",
))

# 8. SSL Loss Weight Ablations
AblationRegistry.register(AblationConfig(
    name="equal_loss_weights",
    description="Use equal weights for all SSL losses (0.2 each)",
    tier=AblationTier.TIER_1,
    config_deltas={
        "ssl_loss_weights.masked_token": 0.20,
        "ssl_loss_weights.ranking": 0.20,
        "ssl_loss_weights.provider_consistency": 0.20,
        "ssl_loss_weights.coordinate_corruption": 0.20,
        "ssl_loss_weights.group_relation": 0.20,
    },
    expected_degradation="moderate",
    hypothesis="Receiver reconstruction (masked_token) should dominate for niche-aware learning",
))

AblationRegistry.register(AblationConfig(
    name="no_auxiliary_losses",
    description="Use only masked token loss (no auxiliary objectives)",
    tier=AblationTier.TIER_1,
    config_deltas={
        "ssl_loss_weights.masked_token": 1.0,
        "ssl_loss_weights.ranking": 0.0,
        "ssl_loss_weights.provider_consistency": 0.0,
        "ssl_loss_weights.coordinate_corruption": 0.0,
        "ssl_loss_weights.group_relation": 0.0,
    },
    expected_degradation="moderate",
    hypothesis="Auxiliary losses provide complementary supervision signals",
))

# 9. Confidence Calibration Ablations
AblationRegistry.register(AblationConfig(
    name="no_calibration",
    description="Disable temperature scaling for confidence scores",
    tier=AblationTier.TIER_1,
    config_deltas={"calibration.method": "none"},
    expected_degradation="minimal",
    hypothesis="Calibration improves reliability of confidence-based decisions",
))

AblationRegistry.register(AblationConfig(
    name="temperature_calibration",
    description="Use temperature scaling for confidence calibration",
    tier=AblationTier.TIER_1,
    config_deltas={"calibration.method": "temperature"},
    expected_degradation="minimal",
    hypothesis="Temperature scaling reduces expected calibration error",
))
