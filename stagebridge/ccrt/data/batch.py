"""``CCRTBatch`` — the model-ready batch container and its validation contract.

This milestone implements *validation only*: shape/rank checks (via
``contracts.tensors``) and forbidden-field guards (via ``contracts.naming``).
There are no tensor operations, no torch, no attention. Feature tensors may be
anything array-like with a ``.shape`` or a rectangular nested list/tuple, so
tests can use plain Python lists.

Field values carried in the ``*_id`` slots are grammar-typed vocabulary and are
system-specific; the field *names* and *shapes* here are fixed and
system-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts.errors import CCRTValidationError
from ..contracts.naming import (
    assert_no_forbidden_mechanism_fields,
    assert_no_model_input_leakage_fields,
)
from ..contracts.tensors import require_rank, require_same_prefix

__all__ = ["CCRTBatch"]


def _is_provided_conditioning(value: Any) -> bool:
    """True if a conditioning id (system/edge) was actually supplied."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence):
        return len(value) > 0
    return True


@dataclass
class CCRTBatch:
    """A validated, model-ready CCRT batch.

    Shapes (symbolic; see ``TENSOR_CONTRACT.md``):

        receiver_features    [B, D_R]
        sender_features      [B, K, D_S]
        sender_mask          [B, K]
        distance_to_receiver [B, K]
        uncertainty          [B, K]     (optional)
        semantic_features    [B, D_Z]   (optional)
        regulatory_features  [B, D_REG] (optional)
    """

    receiver_features: Any
    sender_features: Any
    sender_mask: Any
    distance_to_receiver: Any

    biological_system_id: str | Sequence[str]
    transition_edge_id: str | Sequence[str]

    receiver_state_id: str | Sequence[str] | None = None
    uncertainty: Any | None = None
    semantic_features: Any | None = None
    regulatory_features: Any | None = None

    model_inputs: Mapping[str, Any] = field(default_factory=dict)
    targets: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # -- sizes -------------------------------------------------------------

    def batch_size(self) -> int:
        """Return B (number of receivers), from ``receiver_features``."""
        shape = require_rank("receiver_features", self.receiver_features, 2)
        return shape[0]

    def max_sender_context(self) -> int:
        """Return K (padded sender-context slots), from ``sender_features``."""
        shape = require_rank("sender_features", self.sender_features, 3)
        return shape[1]

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Validate shapes, prefix consistency, and forbidden-field hygiene."""
        # -- conditioning presence --
        if not _is_provided_conditioning(self.biological_system_id):
            raise CCRTValidationError(
                "CCRTBatch.biological_system_id must be provided (non-empty)"
            )
        if not _is_provided_conditioning(self.transition_edge_id):
            raise CCRTValidationError(
                "CCRTBatch.transition_edge_id must be provided (non-empty)"
            )

        # -- required tensor ranks --
        recv_shape = require_rank("receiver_features", self.receiver_features, 2)
        sender_shape = require_rank("sender_features", self.sender_features, 3)
        mask_shape = require_rank("sender_mask", self.sender_mask, 2)
        dist_shape = require_rank(
            "distance_to_receiver", self.distance_to_receiver, 2
        )

        # -- prefix consistency: receiver B vs sender B --
        require_same_prefix(
            "receiver_features", recv_shape, "sender_features", sender_shape, 1
        )
        # sender_mask [B, K] must match sender_features [B, K, *]
        require_same_prefix(
            "sender_mask", mask_shape, "sender_features", sender_shape, 2
        )
        # distance_to_receiver [B, K] must match sender_features [B, K, *]
        require_same_prefix(
            "distance_to_receiver", dist_shape, "sender_features", sender_shape, 2
        )

        # -- optional tensors --
        if self.uncertainty is not None:
            unc_shape = require_rank("uncertainty", self.uncertainty, 2)
            require_same_prefix(
                "uncertainty", unc_shape, "sender_features", sender_shape, 2
            )
        if self.semantic_features is not None:
            sem_shape = require_rank("semantic_features", self.semantic_features, 2)
            require_same_prefix(
                "semantic_features", sem_shape, "receiver_features", recv_shape, 1
            )
        if self.regulatory_features is not None:
            reg_shape = require_rank(
                "regulatory_features", self.regulatory_features, 2
            )
            require_same_prefix(
                "regulatory_features", reg_shape, "receiver_features", recv_shape, 1
            )

        # -- forbidden-field hygiene --
        # model_inputs: no mechanism fields, no leakage fields.
        model_input_keys = list(self.model_inputs.keys())
        assert_no_forbidden_mechanism_fields(model_input_keys)
        assert_no_model_input_leakage_fields(model_input_keys)

        # metadata: no mechanism fields, no leakage fields.
        metadata_keys = list(self.metadata.keys())
        assert_no_forbidden_mechanism_fields(metadata_keys)
        assert_no_model_input_leakage_fields(metadata_keys)

        # targets: training targets are allowed, but the *exact* forbidden
        # leakage names must not be smuggled in even here. Mechanism fields are
        # never legitimate anywhere.
        target_keys = list(self.targets.keys())
        assert_no_forbidden_mechanism_fields(target_keys)
        assert_no_model_input_leakage_fields(target_keys)

    def __post_init__(self) -> None:
        # Normalize the mapping fields into plain dicts for stable iteration.
        # Validation is deliberately NOT done here — it is explicit via
        # ``validate()`` so a batch can be constructed and inspected before the
        # contract is enforced.
        self.model_inputs = dict(self.model_inputs)
        self.targets = dict(self.targets)
        self.metadata = dict(self.metadata)
