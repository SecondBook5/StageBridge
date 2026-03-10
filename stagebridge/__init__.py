"""StageBridge: transformer-first stage transition modeling for lung progression."""
__version__ = "0.1.0"


def compose_config(*args, **kwargs):
	from .notebook_api import compose_config as _compose_config

	return _compose_config(*args, **kwargs)


def run_step(*args, **kwargs):
	from .notebook_api import run_step as _run_step

	return _run_step(*args, **kwargs)


def run_pipeline(*args, **kwargs):
	from .notebook_api import run_pipeline as _run_pipeline

	return _run_pipeline(*args, **kwargs)

__all__ = ["__version__", "compose_config", "run_step", "run_pipeline"]
