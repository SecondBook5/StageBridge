"""CCRT contracts — the low-level law of the project.

Owns canonical field names, forbidden-field enforcement, standardized table
schemas, tensor-shape contracts, and composed validation helpers. This package
is system-agnostic and imports nothing outside the standard library.
"""

from __future__ import annotations

from .errors import (
    CCRTContractError,
    CCRTForbiddenFieldError,
    CCRTGrammarError,
    CCRTLeakageError,
    CCRTShapeError,
    CCRTSplitError,
    CCRTTableSchemaError,
    CCRTValidationError,
)
from .naming import (
    BIOLOGICAL_SYSTEM_ID,
    COUNTERFACTUAL_PERTURBATION_ID,
    FORBIDDEN_MECHANISM_FIELDS,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    RECEIVER_BEHAVIOR_ID,
    RECEIVER_STATE_ID,
    REGULATORY_MEDIATOR_ID,
    SENDER_CONTEXT_TYPE_ID,
    SIGNAL_PROGRAM_ID,
    TABLE_RECEIVERS,
    TABLE_REGULATORY_FEATURES,
    TABLE_SAMPLES,
    TABLE_SEMANTIC_FEATURES,
    TABLE_SENDER_CONTEXT,
    TABLE_TRANSITION_EDGES,
    TRANSITION_EDGE_ID,
    assert_no_forbidden_fields,
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
    is_forbidden_mechanism_field,
    is_forbidden_model_input_field,
    normalize_field_name,
)
from .tables import (
    RECEIVERS_SCHEMA,
    REGULATORY_FEATURES_SCHEMA,
    SAMPLES_SCHEMA,
    SEMANTIC_FEATURES_SCHEMA,
    SENDER_CONTEXT_SCHEMA,
    TABLE_SCHEMAS,
    TRANSITION_EDGES_SCHEMA,
    TableSchema,
    get_table_schema,
    validate_table_fields,
)
from .tensors import (
    B,
    D_R,
    D_REG,
    D_S,
    D_Z,
    K,
    require_rank,
    require_same_prefix,
    shape_of,
)
from .validation import (
    validate_field_names,
    validate_no_extra_fields,
    validate_required_fields,
)

__all__ = [
    # errors
    "CCRTContractError",
    "CCRTValidationError",
    "CCRTForbiddenFieldError",
    "CCRTLeakageError",
    "CCRTShapeError",
    "CCRTGrammarError",
    "CCRTTableSchemaError",
    "CCRTSplitError",
    # forbidden field sets
    "FORBIDDEN_MECHANISM_FIELDS",
    "FORBIDDEN_MODEL_INPUT_FIELDS",
    # grammar field names
    "BIOLOGICAL_SYSTEM_ID",
    "RECEIVER_STATE_ID",
    "TRANSITION_EDGE_ID",
    "SENDER_CONTEXT_TYPE_ID",
    "SIGNAL_PROGRAM_ID",
    "RECEIVER_BEHAVIOR_ID",
    "REGULATORY_MEDIATOR_ID",
    "COUNTERFACTUAL_PERTURBATION_ID",
    # table names
    "TABLE_RECEIVERS",
    "TABLE_SENDER_CONTEXT",
    "TABLE_SEMANTIC_FEATURES",
    "TABLE_REGULATORY_FEATURES",
    "TABLE_TRANSITION_EDGES",
    "TABLE_SAMPLES",
    # naming helpers
    "normalize_field_name",
    "is_forbidden_mechanism_field",
    "is_forbidden_model_input_field",
    "assert_no_forbidden_mechanism_fields",
    "assert_no_model_input_leakage_fields",
    "assert_no_forbidden_fields",
    # table schemas
    "TableSchema",
    "RECEIVERS_SCHEMA",
    "SENDER_CONTEXT_SCHEMA",
    "SEMANTIC_FEATURES_SCHEMA",
    "REGULATORY_FEATURES_SCHEMA",
    "TRANSITION_EDGES_SCHEMA",
    "SAMPLES_SCHEMA",
    "TABLE_SCHEMAS",
    "get_table_schema",
    "validate_table_fields",
    # tensor helpers
    "B",
    "K",
    "D_R",
    "D_S",
    "D_Z",
    "D_REG",
    "shape_of",
    "require_rank",
    "require_same_prefix",
    # validation helpers
    "validate_field_names",
    "validate_required_fields",
    "validate_no_extra_fields",
]
