"""
StageBridge Ablation Runner Module.

Provides infrastructure for controlled ablation experiments:
- Registry of ablation definitions
- Config delta specifications
- Execution orchestration
- Metric computation and comparison
- Visualization generation
- Summary table generation

Tier 1 Ablations (V1 Required):
1. Deterministic vs stochastic transition (flow matching)
2. No niche vs pooled niche vs influence tensor niche
3. No genomics vs genomics as feature vs genomics as constraint
4. Flat set pooling vs hierarchical pooling
5. HLCA only vs LuCA only vs dual reference
6. Canonical spatial backend vs alternatives
"""

from .registry import AblationRegistry, AblationConfig
from .runner import run_ablation, run_ablation_suite
from .metrics import compute_ablation_metrics, compute_degradation
from .summary import (
    generate_ablation_summary,
    generate_ablation_table,
    generate_ablation_visualizations,
    generate_full_ablation_report,
)

__all__ = [
    "AblationRegistry",
    "AblationConfig",
    "run_ablation",
    "run_ablation_suite",
    "compute_ablation_metrics",
    "compute_degradation",
    "generate_ablation_summary",
    "generate_ablation_table",
    "generate_ablation_visualizations",
    "generate_full_ablation_report",
]
