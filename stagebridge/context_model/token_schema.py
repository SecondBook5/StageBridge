"""Typed token schema used by the context model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TokenType:
    """One typed biological token category."""

    name: str
    prefix: str


@dataclass(slots=True, frozen=True)
class TypedTokenSchema:
    """Grouping of raw spatial mapping features into typed niche token families."""

    raw_feature_names: tuple[str, ...]
    typed_feature_names: tuple[str, ...]
    feature_to_group: dict[str, str]

    def group_for(self, feature_name: str) -> str:
        return self.feature_to_group.get(str(feature_name), "vascular_program")


DEFAULT_TYPED_FEATURE_NAMES: tuple[str, ...] = (
    "epithelial",
    "stromal",
    "immune",
    "vascular_program",
)


def default_typed_token_schema(raw_feature_names: list[str] | tuple[str, ...]) -> TypedTokenSchema:
    """Return the active LUAD typed-token schema for Tangram cell-state outputs."""
    feature_to_group: dict[str, str] = {}
    for feature_name in [str(name) for name in raw_feature_names]:
        lower = feature_name.lower()
        if any(key in lower for key in ("at2", "basal", "ciliated", "secretory", "epithelial")):
            feature_to_group[feature_name] = "epithelial"
        elif any(key in lower for key in ("fibroblast", "stromal")):
            feature_to_group[feature_name] = "stromal"
        elif any(key in lower for key in ("macrophage", "mast", "t cell", "immune")):
            feature_to_group[feature_name] = "immune"
        else:
            feature_to_group[feature_name] = "vascular_program"
    return TypedTokenSchema(
        raw_feature_names=tuple(str(name) for name in raw_feature_names),
        typed_feature_names=DEFAULT_TYPED_FEATURE_NAMES,
        feature_to_group=feature_to_group,
    )
