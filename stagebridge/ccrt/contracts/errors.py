"""CCRT contract exception hierarchy.

These exceptions are the vocabulary of CCRT contract failures. Every guardrail
in the ``contracts/``, ``grammar/``, and ``data/`` packages raises one of these
rather than a bare ``ValueError`` / ``KeyError``, so callers (and tests) can
distinguish *what kind* of contract was violated.

The hierarchy is intentionally shallow and boring:

    CCRTContractError
      └── CCRTValidationError
            ├── CCRTForbiddenFieldError    (forbidden mechanism field/identifier)
            ├── CCRTLeakageError           (forbidden model-input leakage field)
            ├── CCRTShapeError             (tensor rank / prefix mismatch)
            ├── CCRTGrammarError           (BiologicalSystemSpec / entity violation)
            ├── CCRTTableSchemaError       (standardized table schema violation)
            └── CCRTSplitError             (split-manifest granularity / overlap)
"""

from __future__ import annotations

__all__ = [
    "CCRTContractError",
    "CCRTValidationError",
    "CCRTForbiddenFieldError",
    "CCRTLeakageError",
    "CCRTShapeError",
    "CCRTGrammarError",
    "CCRTTableSchemaError",
    "CCRTSplitError",
]


class CCRTContractError(Exception):
    """Base class for every CCRT contract violation."""


class CCRTValidationError(CCRTContractError):
    """A validation check failed (base for all specific validation errors)."""


class CCRTForbiddenFieldError(CCRTValidationError):
    """A forbidden *mechanism* field/identifier was used.

    Raised for the binned / world-token spatial-mechanism identifiers
    enumerated in :data:`stagebridge.ccrt.contracts.naming.FORBIDDEN_MECHANISM_FIELDS`
    (matched case- and whitespace-insensitively). CCRT expresses local influence
    only through continuous distance modulation, so these are forbidden anywhere.
    """


class CCRTLeakageError(CCRTValidationError):
    """A forbidden model-input *leakage* field was used.

    Raised for ``target_stage_expression``, ``future_expression``,
    ``outcome_label``, ``patient_response``, ``test_split_label`` appearing where
    a model input would see them.
    """


class CCRTShapeError(CCRTValidationError):
    """A tensor rank or prefix-dimension contract was violated."""


class CCRTGrammarError(CCRTValidationError):
    """A grammar entity or ``BiologicalSystemSpec`` invariant was violated."""


class CCRTTableSchemaError(CCRTValidationError):
    """A standardized table schema (required/extra/forbidden fields) was violated."""


class CCRTSplitError(CCRTValidationError):
    """A split-manifest granularity, grouping-key, or disjointness rule was violated."""
