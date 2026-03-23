"""
StageBridge: Receiver-centered niche modeling for lung cancer progression.

StageBridge models cell state transitions during LUAD progression using a
receiver-centered attention architecture that captures how local niche context
influences cell behavior. The framework integrates:

- Dual-reference mapping (HLCA + LuCA) for positioning cells in healthy vs. cancer space
- Receiver-centered niche encoding with distance-aware attention
- SSL pretraining via masked receiver reconstruction
- Flow matching transition model for progression dynamics
- Biological interpretation layer linking attention to L-R signaling

Key biological insight: IL1B+ macrophage niches drive early LUAD progression
(Peng et al. 2020 Cancer Cell). StageBridge provides tools to quantify and
interpret these niche interactions at single-cell resolution.

Example:
    >>> from stagebridge.pipelines import run_v1_complete
    >>> run_v1_complete.main(data_dir="path/to/data", output_dir="results/")

For pipeline orchestration, use the notebook API:
    >>> import stagebridge
    >>> cfg = stagebridge.compose_config(data_root="path/to/data")
    >>> stagebridge.run_pipeline(cfg)
"""

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
