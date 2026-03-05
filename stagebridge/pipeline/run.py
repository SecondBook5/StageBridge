"""
stagebridge/pipeline/run.py
Lightweight pipeline runner.

run_smoke(cfg)
    Calls every step in dependency order.  All steps currently raise
    ``NotImplementedError`` — run_smoke catches these and reports them as
    "STUB" rather than failures, so the function can be used as a
    pipeline connectivity check without real data.

    A step that raises anything *other* than NotImplementedError is
    treated as a genuine failure and re-raised.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from omegaconf import DictConfig

log = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    """Summary of a pipeline run (smoke or real)."""

    steps_ok: list[str] = field(default_factory=list)
    steps_stub: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.steps_failed) == 0


def run_smoke(cfg: "DictConfig") -> PipelineReport:
    """
    Execute all pipeline steps in order.

    Stub steps (``NotImplementedError``) are recorded but do not halt
    execution.  Any other exception halts the pipeline and is recorded
    as a failure.

    Parameters
    ----------
    cfg:
        Hydra config (or any object with a ``data`` attribute that provides
        path resolution).  A ``None`` value uses an empty stand-in so the
        function can be called without a real config in tests.

    Returns
    -------
    PipelineReport
        ``report.ok`` is True iff no step raised a non-stub exception.
    """
    report = PipelineReport()

    _STEPS = [
        ("audit_data",          audit_data),
        ("build_snrna_anndata", build_snrna_anndata),
        ("build_spatial_anndata", build_spatial_anndata),
        ("map_hlca",            map_hlca),
        ("run_tangram",         run_tangram),
        ("build_niche_tokens",  build_niche_tokens),
        ("train_benchmark",     train_benchmark),
        ("eval_benchmark",      eval_benchmark),
        ("make_poster_assets",  make_poster_assets),
    ]

    for name, fn in _STEPS:
        log.info("[pipeline] running step: %s", name)
        try:
            fn(cfg)  # type: ignore[arg-type]
            report.steps_ok.append(name)
            log.info("[pipeline] %s — OK", name)
        except NotImplementedError:
            report.steps_stub.append(name)
            log.info("[pipeline] %s — STUB (not yet implemented)", name)
        except Exception as exc:  # noqa: BLE001
            report.steps_failed.append(name)
            report.errors[name] = str(exc)
            log.error("[pipeline] %s — FAILED: %s", name, exc)
            raise

    return report
