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

    # Cancer type-aware functions (preferred for new code)
    from stagebridge.config import (
        set_default_cancer_type,
        get_stage_system_for_cancer,
        get_valid_stages,
        get_stage_colors,
        validate_contract_for_cancer,
    )

    set_default_cancer_type("pdac")
    stages, s2i, i2s = get_stage_system_for_cancer("3")
"""

from stagebridge.config.cancer_types import (
    CancerConfig,
    ReferenceAtlasConfig,
    StageConfig,
    CellTypeMarkers,
    BiologicalMechanism,
    MechanismType,
    get_cancer_config,
    get_available_cancer_types,
    register_cancer_config,
    load_cancer_config_from_yaml,
    LUAD_CONFIG,
    PDAC_CONFIG,
)

from stagebridge.config.cancer_support import (
    set_default_cancer_type,
    get_default_cancer_type,
    get_stage_system_for_cancer,
    convert_stage_for_cancer,
    get_valid_stages,
    get_stage_colors,
    stage_to_idx_for_cancer,
    idx_to_stage_for_cancer,
    get_reference_dims,
    get_fused_dim_for_cancer,
    get_token_structure,
    get_token_type_ids,
    validate_contract_for_cancer,
    get_known_mechanisms,
    get_cell_markers,
)

__all__ = [
    # Configuration classes
    "CancerConfig",
    "ReferenceAtlasConfig",
    "StageConfig",
    "CellTypeMarkers",
    "BiologicalMechanism",
    "MechanismType",
    # Configuration registry
    "get_cancer_config",
    "get_available_cancer_types",
    "register_cancer_config",
    "load_cancer_config_from_yaml",
    # Pre-defined configs
    "LUAD_CONFIG",
    "PDAC_CONFIG",
    # Cancer type-aware functions
    "set_default_cancer_type",
    "get_default_cancer_type",
    "get_stage_system_for_cancer",
    "convert_stage_for_cancer",
    "get_valid_stages",
    "get_stage_colors",
    "stage_to_idx_for_cancer",
    "idx_to_stage_for_cancer",
    "get_reference_dims",
    "get_fused_dim_for_cancer",
    "get_token_structure",
    "get_token_type_ids",
    "validate_contract_for_cancer",
    "get_known_mechanisms",
    "get_cell_markers",
]
