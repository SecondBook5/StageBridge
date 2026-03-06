"""StageBridge: transformer-first stage transition modeling for lung progression."""
__version__ = "0.1.0"

from .notebook_api import compose_config, run_pipeline, run_step

__all__ = ["__version__", "compose_config", "run_step", "run_pipeline"]
