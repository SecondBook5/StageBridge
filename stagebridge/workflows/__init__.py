"""Package-first workflow APIs used by notebook and CLI entrypoints."""

from .eval import run_evaluation
from .pipeline import build_snrna, build_spatial, map_hlca, run_tangram
from .train import run_training

__all__ = [
    "build_snrna",
    "build_spatial",
    "map_hlca",
    "run_tangram",
    "run_training",
    "run_evaluation",
]
