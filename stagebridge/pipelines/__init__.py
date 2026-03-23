"""
Pipeline entrypoints for StageBridge.

This module provides the canonical pipeline execution order for StageBridge:

1. run_data_prep.py          - QC/merge snRNA and spatial data
2. download_references.py    - Fetch HLCA/LuCA reference atlases
3. run_reference.py          - Model-based dual-reference mapping
4. run_spatial_benchmark.py  - Spatial backend comparison (Tangram/DestVI/TACCO)
5. complete_data_prep.py     - Prepare canonical training format
6. run_v1_complete.py        - Full training pipeline

Individual pipelines can be imported and run directly:
    >>> from stagebridge.pipelines import run_data_prep
    >>> run_data_prep.main(data_root="/path/to/data")

Or use the unified CLI:
    $ python -m stagebridge step data_prep --data-root /path/to/data

See stagebridge/pipelines/README.md for detailed documentation.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, str] = {
    "run_communication_benchmark": ".run_communication_benchmark",
    "run_context_model": ".run_context_model",
    "run_data_prep": ".run_data_prep",
    "run_eamist_reporting": ".run_eamist_reporting",
    "run_evaluate_lesion": ".evaluate_lesion",
    "run_evaluation": ".run_evaluation",
    "run_full": ".run_full",
    "run_label_cna": ".run_label_repair",
    "run_label_clonal": ".run_label_repair",
    "run_label_manifest": ".run_label_repair",
    "run_label_phylogeny": ".run_label_repair",
    "run_label_refinement": ".run_label_repair",
    "run_label_repair": ".run_label_repair",
    "run_label_support": ".run_label_repair",
    "run_pretrain_local": ".pretrain_local",
    "run_reference": ".run_reference",
    "run_spatial_mapping": ".run_spatial_mapping",
    "run_transition_model": ".run_transition_model",
    "run_train_lesion": ".train_lesion",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "run_communication_benchmark",
    "run_context_model",
    "run_data_prep",
    "run_eamist_reporting",
    "run_evaluate_lesion",
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
