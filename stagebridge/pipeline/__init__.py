"""
stagebridge.pipeline
High-level pipeline orchestration for the StageBridge benchmark.

Public surface
--------------
steps  : stub step functions — one per logical pipeline stage
run    : run_smoke() — lightweight integration smoke-test
"""
from .run import run_smoke
from .steps import (
    audit_data,
    build_niche_tokens,
    build_snrna_anndata,
    build_spatial_anndata,
    eval_benchmark,
    make_poster_assets,
    map_hlca,
    run_tangram,
    train_benchmark,
)

__all__ = [
    "audit_data",
    "build_niche_tokens",
    "build_snrna_anndata",
    "build_spatial_anndata",
    "eval_benchmark",
    "make_poster_assets",
    "map_hlca",
    "run_smoke",
    "run_tangram",
    "train_benchmark",
]
