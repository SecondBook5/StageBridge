"""LUAD evolution data exports for lesion-level EA-MIST preprocessing."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, str] = {
    "LesionBagDataset": ".bag_dataset",
    "NeighborhoodPretrainDataset": ".bag_dataset",
    "collate_lesion_bags": ".bag_dataset",
    "collate_pretrain_neighborhoods": ".bag_dataset",
    "build_expression_templates": ".feature_builder",
    "summarize_neighborhood_build": ".feature_builder",
    "LuadEvoDataset": ".metadata",
    "resolve_luad_evo_paths": ".metadata",
    "NeighborhoodBuildResult": ".neighborhood_builder",
    "build_lesion_bags": ".neighborhood_builder",
    "build_lesion_bags_from_config": ".neighborhood_builder",
    "build_lesion_label_table": ".neighborhood_builder",
    "infer_edge_label": ".neighborhood_builder",
    "load_curated_lesion_labels": ".neighborhood_builder",
    "load_luad_evo_snrna_latent": ".snrna",
    "LesionFold": ".splits",
    "assert_no_split_leakage": ".splits",
    "build_lesion_folds": ".splits",
    "load_luad_evo_spatial_mapping": ".visium",
    "WES_FEATURE_COLS": ".wes",
    "load_luad_evo_wes_features": ".wes",
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
    "LesionBagDataset",
    "LesionFold",
    "LuadEvoDataset",
    "NeighborhoodBuildResult",
    "NeighborhoodPretrainDataset",
    "WES_FEATURE_COLS",
    "assert_no_split_leakage",
    "build_expression_templates",
    "build_lesion_bags",
    "build_lesion_bags_from_config",
    "build_lesion_folds",
    "build_lesion_label_table",
    "collate_lesion_bags",
    "collate_pretrain_neighborhoods",
    "infer_edge_label",
    "load_curated_lesion_labels",
    "load_luad_evo_snrna_latent",
    "load_luad_evo_spatial_mapping",
    "load_luad_evo_wes_features",
    "resolve_luad_evo_paths",
    "summarize_neighborhood_build",
]
