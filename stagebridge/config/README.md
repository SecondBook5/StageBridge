# Cancer Type Configuration System

This module provides configurable cancer type definitions to support StageBridge generalization beyond LUAD (lung adenocarcinoma) to other cancer types like PDAC (pancreatic ductal adenocarcinoma).

## Quick Start

### Using PDAC instead of LUAD

```python
from stagebridge.config import (
    set_default_cancer_type,
    get_stage_system_for_cancer,
    get_valid_stages,
    validate_contract_for_cancer,
)

# Option 1: Set default for session
set_default_cancer_type("pdac")
stages, s2i, i2s = get_stage_system_for_cancer("3")  # Uses PDAC

# Option 2: Pass explicitly
stages, s2i, i2s = get_stage_system_for_cancer("3", cancer_type="pdac")

# Validate data for PDAC
validate_contract_for_cancer("/path/to/data", cancer_type="pdac")
```

### Environment Variable

```bash
export STAGEBRIDGE_CANCER_TYPE=pdac
```

## Supported Cancer Types

### LUAD (Lung Adenocarcinoma) - Default

- **Stages**: Normal -> AAH -> AIS -> MIA -> LUAD
- **References**: HLCA (30d) + LuCA (10d)
- **Key mechanisms**: IL1B-IL1R1 axis, KAC progenitors, CAF dynamics

### PDAC (Pancreatic Ductal Adenocarcinoma)

- **Stages**: Normal -> PanIN1 -> PanIN2 -> PanIN3 -> PDAC
- **References**: Currently none (can be added when available)
- **Key mechanisms**: KRAS activation, ADM, desmoplastic stroma

## Adding a New Cancer Type

### Option 1: Python Configuration

```python
from stagebridge.config import (
    CancerConfig,
    StageConfig,
    ReferenceAtlasConfig,
    BiologicalMechanism,
    MechanismType,
    register_cancer_config,
)

my_stages = StageConfig(
    stages_full=("Normal", "Stage1", "Stage2", "Cancer"),
    stages_3_mapping={
        "Normal": "Normal",
        "Stage1": "Preinvasive",
        "Stage2": "Preinvasive",
        "Cancer": "Invasive",
    },
    stage_colors={
        "Normal": "#228B22",
        "Stage1": "#90EE90",
        "Stage2": "#FFD700",
        "Cancer": "#8B0000",
    },
)

my_config = CancerConfig(
    name="my_cancer",
    description="My cancer type progression",
    stages=my_stages,
    reference_mode="none",  # or "single" or "dual"
)

register_cancer_config(my_config)
```

### Option 2: YAML Configuration

Create a YAML file (see `workflow/configs/pdac.yaml` for template):

```python
from stagebridge.config import load_cancer_config_from_yaml, register_cancer_config

config = load_cancer_config_from_yaml("/path/to/my_cancer.yaml")
register_cancer_config(config)
```

## Key Functions

| Function | Description |
|----------|-------------|
| `get_cancer_config(type)` | Get full configuration for a cancer type |
| `get_stage_system_for_cancer(system, type)` | Get stage names and mappings |
| `get_valid_stages(type)` | Get all valid stage names |
| `get_stage_colors(type)` | Get visualization colors |
| `get_token_structure(type)` | Get token count and names |
| `validate_contract_for_cancer(path, type)` | Validate data contract |
| `get_known_mechanisms(type)` | Get biological mechanisms for validation |
| `get_cell_markers(type)` | Get cell type marker genes |

## Backward Compatibility

All existing code using `stagebridge.contracts` continues to work unchanged:

```python
# This still works - defaults to LUAD
from stagebridge.contracts import get_stage_system, STAGES_5, HLCA_DIM
```

The new functions in `stagebridge.config` provide cancer type-aware alternatives without modifying the original contracts.

## Reference Atlas Configuration

For cancer types with reference atlases:

```python
from stagebridge.config import get_reference_dims, get_fused_dim_for_cancer

# Get reference dimensions
dims = get_reference_dims("luad")  # {"hlca": 30, "luca": 10}

# Get fused embedding dimension
fused = get_fused_dim_for_cancer("concat", "luad")  # 40
```

For cancer types without references (like PDAC currently):

```python
dims = get_reference_dims("pdac")  # {}
fused = get_fused_dim_for_cancer("concat", "pdac")  # 0
```

## Token Structure

Token structure adapts to reference configuration:

- **Dual reference (LUAD)**: 9 tokens (receiver, 4 rings, ref1, ref2, pathway, stats)
- **Single reference**: 8 tokens
- **No reference (PDAC)**: 7 tokens

```python
from stagebridge.config import get_token_structure

n_tokens, token_names = get_token_structure("luad")
# (9, ('receiver', 'ring1', 'ring2', 'ring3', 'ring4', 'hlca', 'luca', 'pathway', 'stats'))

n_tokens, token_names = get_token_structure("pdac")
# (7, ('receiver', 'ring1', 'ring2', 'ring3', 'ring4', 'pathway', 'stats'))
```
