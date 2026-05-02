"""Cancer type configuration system for StageBridge.

This module provides configurable cancer type definitions to support
generalization beyond LUAD to other cancer types (e.g., PDAC, colorectal).

The configuration system allows:
- Stage progressions per cancer type
- Cell type markers and signatures
- Reference atlas options
- Known biological mechanisms for validation

Usage:
    from stagebridge.config import get_cancer_config, CancerConfig

    # Get LUAD config (default)
    config = get_cancer_config("luad")

    # Get PDAC config
    config = get_cancer_config("pdac")

    # Use with contracts
    from stagebridge.contracts import get_stage_system
    stages, s2i, i2s = get_stage_system("3", cancer_type="luad")
"""

from stagebridge.config.cancer_types import (
    CancerConfig,
    ReferenceAtlasConfig,
    StageConfig,
    CellTypeMarkers,
    BiologicalMechanism,
    get_cancer_config,
    get_available_cancer_types,
    register_cancer_config,
    LUAD_CONFIG,
    PDAC_CONFIG,
)

__all__ = [
    "CancerConfig",
    "ReferenceAtlasConfig",
    "StageConfig",
    "CellTypeMarkers",
    "BiologicalMechanism",
    "get_cancer_config",
    "get_available_cancer_types",
    "register_cancer_config",
    "LUAD_CONFIG",
    "PDAC_CONFIG",
]
