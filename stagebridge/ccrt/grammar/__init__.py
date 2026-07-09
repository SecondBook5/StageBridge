"""CCRT grammar — the biological meaning layer.

Owns the entity dataclasses and ``BiologicalSystemSpec``. This is what makes
CCRT unified: the grammar classes are identical across systems, while their IDs
carry system-specific vocabulary. Imports only ``contracts`` and the standard
library — never adapters, operators, or any downstream layer.
"""

from __future__ import annotations

from .entities import (
    CounterfactualPerturbation,
    ReceiverBehavior,
    ReceiverState,
    RegulatoryMediator,
    SenderContextType,
    SignalProgram,
    TransitionEdge,
)
from .spec import (
    DEATH_REMOVAL,
    GROWTH_MASS,
    INFECTION_STATE_TRANSITION,
    MOTILITY_INVASION,
    REPAIR_RECOVERY,
    SEMANTIC_DRIFT,
    BiologicalSystemSpec,
)
from .validation import (
    require_ids_exist,
    require_unique_ids,
    validate_biological_system_spec,
)

__all__ = [
    # entities
    "ReceiverState",
    "TransitionEdge",
    "SenderContextType",
    "SignalProgram",
    "ReceiverBehavior",
    "RegulatoryMediator",
    "CounterfactualPerturbation",
    # spec
    "BiologicalSystemSpec",
    # standard behavior constants
    "SEMANTIC_DRIFT",
    "GROWTH_MASS",
    "DEATH_REMOVAL",
    "MOTILITY_INVASION",
    "INFECTION_STATE_TRANSITION",
    "REPAIR_RECOVERY",
    # validation helpers
    "require_unique_ids",
    "require_ids_exist",
    "validate_biological_system_spec",
]
