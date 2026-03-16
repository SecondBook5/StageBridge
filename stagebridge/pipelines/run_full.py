"""Full pipeline orchestration entrypoint."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from .run_context_model import run_context_model
from .run_evaluation import run_evaluation
from .run_reference import run_reference
from .run_spatial_mapping import run_spatial_mapping
from .run_transition_model import run_transition_model


def run_full(cfg: DictConfig) -> dict[str, Any]:
    reference = run_reference(cfg)
    spatial_mapping = run_spatial_mapping(cfg, reference_output=reference)
    context_model = run_context_model(cfg, spatial_output=spatial_mapping)
    transition_model = run_transition_model(
        cfg,
        reference_output=reference,
        spatial_output=spatial_mapping,
        context_output=context_model,
    )
    evaluation = run_evaluation(
        cfg, transition_output=transition_model, context_output=context_model
    )
    return {
        "ok": True,
        "pipeline": "full",
        "steps": {
            "reference": reference,
            "spatial_mapping": spatial_mapping,
            "context_model": context_model,
            "transition_model": transition_model,
            "evaluation": evaluation,
        },
    }
