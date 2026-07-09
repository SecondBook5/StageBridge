"""Shape conventions and tensor-shape validation for CCRT.

No dependency on torch / numpy. Shapes are inferred either from a ``.shape``
attribute (torch tensors, numpy arrays, anything array-like) or from nested
Python lists/tuples assumed to be rectangular. Ragged nested sequences are a
contract violation and fail with ``CCRTShapeError``.

This module does shape checking only — no model architecture, no attention, no
tensor math.

Canonical symbolic dimensions (values are fixed only during implementation):

    B      batch size / receiver count
    K      sender-context element count per receiver (padded)
    D_R    receiver feature dimension
    D_S    sender feature dimension
    D_Z    semantic feature dimension
    D_REG  regulatory feature dimension
"""

from __future__ import annotations

from typing import Any

from .errors import CCRTShapeError

__all__ = [
    "B",
    "K",
    "D_R",
    "D_S",
    "D_Z",
    "D_REG",
    "shape_of",
    "require_rank",
    "require_same_prefix",
]

# Symbolic dimension labels (documentation constants; not numeric sizes).
B = "B"
K = "K"
D_R = "D_R"
D_S = "D_S"
D_Z = "D_Z"
D_REG = "D_REG"


def _infer_nested_shape(value: Any) -> tuple[int, ...]:
    """Infer a rectangular shape from nested lists/tuples.

    A scalar (non-sequence) has shape ``()``. A rectangular nested sequence has
    shape ``(len, *inner_shape)`` where every element shares ``inner_shape``.
    Ragged sequences raise ``CCRTShapeError``.
    """
    # Treat str/bytes as scalars, not sequences of characters.
    if not isinstance(value, (list, tuple)):
        return ()

    length = len(value)
    if length == 0:
        # An empty list is a length-0 axis with no inferable inner shape.
        return (0,)

    child_shapes = [_infer_nested_shape(item) for item in value]
    first = child_shapes[0]
    for idx, cs in enumerate(child_shapes):
        if cs != first:
            raise CCRTShapeError(
                "ragged nested sequence: element 0 has shape "
                f"{first} but element {idx} has shape {cs}; "
                "CCRT requires rectangular tensors"
            )
    return (length, *first)


def shape_of(value: Any) -> tuple[int, ...]:
    """Return the shape of ``value`` as a tuple of ints.

    * If ``value`` has a ``.shape`` attribute (torch/numpy), return
      ``tuple(int(x) for x in value.shape)``.
    * If ``value`` is a nested list/tuple, infer a rectangular shape.
    * Otherwise (a scalar / unsupported) raise ``CCRTShapeError``.

    Ragged nested lists fail rather than silently flattening.
    """
    shape_attr = getattr(value, "shape", None)
    if shape_attr is not None:
        try:
            return tuple(int(x) for x in shape_attr)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise CCRTShapeError(
                f"could not interpret .shape={shape_attr!r} as integer dims"
            ) from exc

    if isinstance(value, (list, tuple)):
        return _infer_nested_shape(value)

    raise CCRTShapeError(
        f"cannot infer shape of value of type {type(value).__name__}; "
        "expected an array-like with .shape or a rectangular nested list/tuple"
    )


def require_rank(name: str, value: Any, rank: int) -> tuple[int, ...]:
    """Assert ``value`` has exactly ``rank`` dimensions; return its shape."""
    shape = shape_of(value)
    if len(shape) != rank:
        raise CCRTShapeError(
            f"'{name}' must have rank {rank} but has rank {len(shape)} "
            f"(shape={shape})"
        )
    return shape


def require_same_prefix(
    name_a: str,
    shape_a: tuple[int, ...],
    name_b: str,
    shape_b: tuple[int, ...],
    prefix_rank: int,
) -> None:
    """Assert the first ``prefix_rank`` dims of two shapes match.

    Used to tie, e.g., the ``[B, K]`` prefix of ``sender_mask`` to the
    ``[B, K, *]`` prefix of ``sender_features``.
    """
    if len(shape_a) < prefix_rank:
        raise CCRTShapeError(
            f"'{name_a}' has rank {len(shape_a)} < required prefix rank "
            f"{prefix_rank} (shape={shape_a})"
        )
    if len(shape_b) < prefix_rank:
        raise CCRTShapeError(
            f"'{name_b}' has rank {len(shape_b)} < required prefix rank "
            f"{prefix_rank} (shape={shape_b})"
        )
    prefix_a = shape_a[:prefix_rank]
    prefix_b = shape_b[:prefix_rank]
    if prefix_a != prefix_b:
        raise CCRTShapeError(
            f"prefix mismatch: '{name_a}' prefix {prefix_a} != "
            f"'{name_b}' prefix {prefix_b} (first {prefix_rank} dim(s))"
        )
