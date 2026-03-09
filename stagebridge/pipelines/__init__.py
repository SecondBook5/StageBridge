"""Canonical pipeline namespace for the StageBridge rebuild."""

from .evaluate_lesion import run_evaluate_lesion
from .pretrain_local import run_pretrain_local
from .run_context_model import run_context_model
from .run_eamist_reporting import run_eamist_reporting
from .run_evaluation import run_evaluation
from .run_full import run_full
from .run_label_repair import (
    run_label_cna,
    run_label_clonal,
    run_label_manifest,
    run_label_phylogeny,
    run_label_refinement,
    run_label_repair,
    run_label_support,
)
from .run_reference import run_reference
from .run_spatial_mapping import run_spatial_mapping
from .run_transition_model import run_transition_model
from .train_lesion import run_train_lesion

__all__ = [
    "run_evaluate_lesion",
    "run_context_model",
    "run_eamist_reporting",
    "run_evaluation",
    "run_full",
    "run_label_cna",
    "run_label_clonal",
    "run_label_manifest",
    "run_label_phylogeny",
    "run_label_refinement",
    "run_label_repair",
    "run_label_support",
    "run_pretrain_local",
    "run_reference",
    "run_spatial_mapping",
    "run_train_lesion",
    "run_transition_model",
]
