"""
Backend selection with justification for canonical backend decision.

Provides logic to select the best backend based on comparison results
and generate detailed justification reports.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
from datetime import datetime

import pandas as pd

from .comparison import ComparisonResult


@dataclass
class BackendSelection:
    """
    Canonical backend selection result.

    Contains the selected backend, justification, and alternatives.
    """

    # Selected canonical backend
    canonical_backend: str

    # Overall selection score (0-1)
    selection_score: float

    # Justification text
    justification: str

    # Detailed scores by category
    category_scores: dict[str, float] = field(default_factory=dict)

    # Alternative backends (ranked)
    alternatives: list[str] = field(default_factory=list)

    # Alternative scores
    alternative_scores: dict[str, float] = field(default_factory=dict)

    # Selection metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "canonical_backend": self.canonical_backend,
            "selection_score": self.selection_score,
            "justification": self.justification,
            "category_scores": self.category_scores,
            "alternatives": self.alternatives,
            "alternative_scores": self.alternative_scores,
            "metadata": self.metadata,
        }


def select_canonical_backend(
    comparison_result: ComparisonResult,
    weights: dict[str, float] | None = None,
    min_score_threshold: float = 0.3,
) -> BackendSelection:
    """
    Select the canonical backend from comparison results.

    Selection criteria:
    1. Must have completed successfully
    2. Weighted score across upstream, downstream, spatial, and runtime
    3. Prefer backends with good downstream utility (StageBridge-focused)

    Args:
        comparison_result: ComparisonResult from backend comparison
        weights: Optional custom weights for selection criteria
        min_score_threshold: Minimum score to be considered

    Returns:
        BackendSelection with canonical backend and justification
    """
    if weights is None:
        # Default weights emphasize downstream utility
        weights = {
            "upstream": 0.25,
            "downstream": 0.40,
            "spatial": 0.20,
            "robustness": 0.10,
            "runtime": 0.05,
        }

    df = comparison_result.comparison_table
    if df is None or len(df) == 0:
        raise ValueError("No comparison results available")

    # Filter to successful backends
    successful = df[df["success"]].copy()
    if len(successful) == 0:
        raise ValueError("No backends completed successfully")

    # Compute category scores for each backend
    backend_scores = {}
    category_scores_by_backend = {}

    for idx, row in successful.iterrows():
        backend = row["backend"]
        scores = {}

        # Upstream score
        upstream_cols = [c for c in df.columns if c.startswith("upstream_")]
        if upstream_cols:
            upstream_vals = [row[c] for c in upstream_cols if pd.notna(row.get(c))]
            scores["upstream"] = _normalize_scores(upstream_vals)
        else:
            scores["upstream"] = 0.5

        # Downstream score
        downstream_cols = [c for c in df.columns if c.startswith("downstream_")]
        if downstream_cols:
            downstream_vals = [row[c] for c in downstream_cols if pd.notna(row.get(c))]
            scores["downstream"] = _normalize_scores(downstream_vals)
        else:
            scores["downstream"] = 0.5

        # Spatial score
        spatial_cols = [c for c in df.columns if c.startswith("spatial_")]
        if spatial_cols:
            spatial_vals = [row[c] for c in spatial_cols if pd.notna(row.get(c))]
            scores["spatial"] = _normalize_scores(spatial_vals)
        else:
            scores["spatial"] = 0.5

        # Robustness score
        robustness_cols = [c for c in df.columns if c.startswith("robustness_")]
        if robustness_cols:
            robustness_vals = [row[c] for c in robustness_cols if pd.notna(row.get(c))]
            scores["robustness"] = _normalize_scores(robustness_vals)
        else:
            scores["robustness"] = 0.5

        # Runtime score (normalized, lower is better)
        max_runtime = successful["runtime_seconds"].max()
        if max_runtime > 0:
            scores["runtime"] = 1 - (row["runtime_seconds"] / max_runtime)
        else:
            scores["runtime"] = 1.0

        category_scores_by_backend[backend] = scores

        # Compute weighted overall score
        overall = sum(weights.get(cat, 0) * score for cat, score in scores.items())
        backend_scores[backend] = overall

    # Rank backends
    ranked = sorted(backend_scores.items(), key=lambda x: x[1], reverse=True)

    # Select canonical
    canonical_backend = ranked[0][0]
    canonical_score = ranked[0][1]

    # Get alternatives
    alternatives = [name for name, _ in ranked[1:]]
    alternative_scores = {name: score for name, score in ranked[1:]}

    # Generate justification
    justification = _generate_justification(
        canonical_backend=canonical_backend,
        canonical_score=canonical_score,
        category_scores=category_scores_by_backend[canonical_backend],
        alternatives=alternatives,
        alternative_scores=alternative_scores,
        weights=weights,
    )

    return BackendSelection(
        canonical_backend=canonical_backend,
        selection_score=canonical_score,
        justification=justification,
        category_scores=category_scores_by_backend[canonical_backend],
        alternatives=alternatives,
        alternative_scores=alternative_scores,
        metadata={
            "selection_weights": weights,
            "min_score_threshold": min_score_threshold,
            "n_successful_backends": len(successful),
            "selection_timestamp": datetime.now().isoformat(),
        },
    )


def _normalize_scores(values: list[float]) -> float:
    """Normalize a list of metric values to [0, 1] and average."""
    if not values:
        return 0.5

    # Filter out NaN
    valid = [v for v in values if pd.notna(v)]
    if not valid:
        return 0.5

    # Most metrics are already in [0, 1], just average
    return sum(valid) / len(valid)


def _generate_justification(
    canonical_backend: str,
    canonical_score: float,
    category_scores: dict[str, float],
    alternatives: list[str],
    alternative_scores: dict[str, float],
    weights: dict[str, float],
) -> str:
    """Generate detailed justification text for backend selection."""
    lines = [
        f"# Canonical Backend Selection: {canonical_backend.upper()}",
        "",
        f"**Overall Score:** {canonical_score:.3f}",
        "",
        "## Selection Criteria",
        "",
    ]

    # Category breakdown
    lines.append("| Category | Weight | Score |")
    lines.append("|----------|--------|-------|")
    for cat in ["downstream", "upstream", "spatial", "robustness", "runtime"]:
        weight = weights.get(cat, 0)
        score = category_scores.get(cat, 0.5)
        lines.append(f"| {cat.title()} | {weight:.0%} | {score:.3f} |")

    lines.append("")

    # Key strengths
    lines.append("## Key Strengths")
    lines.append("")

    top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    for cat, score in top_categories:
        if score > 0.6:
            lines.append(f"- **{cat.title()}**: Strong performance ({score:.3f})")

    lines.append("")

    # Alternatives
    if alternatives:
        lines.append("## Alternatives")
        lines.append("")
        lines.append("| Backend | Score | Gap |")
        lines.append("|---------|-------|-----|")
        for alt in alternatives[:3]:
            alt_score = alternative_scores[alt]
            gap = canonical_score - alt_score
            lines.append(f"| {alt} | {alt_score:.3f} | -{gap:.3f} |")
        lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")

    if canonical_score > 0.7:
        lines.append(
            f"{canonical_backend.upper()} is the clear choice with strong performance "
            f"across all criteria. Use as the canonical backend for StageBridge v1."
        )
    elif canonical_score > 0.5:
        lines.append(
            f"{canonical_backend.upper()} is recommended as the canonical backend. "
            f"Performance is adequate across criteria. Consider validating with "
            f"alternative ({alternatives[0] if alternatives else 'none'}) for robustness."
        )
    else:
        lines.append(
            f"{canonical_backend.upper()} is selected but with moderate confidence. "
            f"All backends showed limited performance. Consider investigating "
            f"data quality or parameter tuning."
        )

    return "\n".join(lines)


def generate_selection_report(
    comparison_result: ComparisonResult,
    selection: BackendSelection,
    output_path: Path | None = None,
) -> str:
    """
    Generate comprehensive selection report in Markdown format.

    Args:
        comparison_result: Full comparison results
        selection: Backend selection decision
        output_path: Optional path to save report

    Returns:
        Markdown report string
    """
    lines = [
        "# Spatial Backend Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # Executive summary
    lines.extend(
        [
            "## Executive Summary",
            "",
            f"**Canonical Backend:** {selection.canonical_backend.upper()}",
            f"**Selection Score:** {selection.selection_score:.3f}",
            "",
            selection.justification,
            "",
            "---",
            "",
        ]
    )

    # Comparison table
    lines.extend(
        [
            "## Backend Comparison",
            "",
        ]
    )

    if comparison_result.comparison_table is not None:
        df = comparison_result.comparison_table

        # Summary table
        summary_cols = ["backend", "success", "runtime_seconds"]
        score_cols = [c for c in df.columns if "_score" in c or "overall" in c]
        display_cols = summary_cols + score_cols[:5]
        display_cols = [c for c in display_cols if c in df.columns]

        if display_cols:
            try:
                lines.append(df[display_cols].to_markdown(index=False))
            except ImportError:
                # tabulate not installed, use simple format
                lines.append("| " + " | ".join(display_cols) + " |")
                lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
                for _, row in df[display_cols].iterrows():
                    vals = [str(row[c])[:20] for c in display_cols]
                    lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    # Rankings
    lines.extend(
        [
            "## Rankings by Criteria",
            "",
        ]
    )

    for criterion, ranking in comparison_result.rankings.items():
        lines.append(f"**{criterion.title()}:** {' > '.join(ranking)}")

    lines.append("")

    # Failed backends
    failed = comparison_result.get_failed_backends()
    if failed:
        lines.extend(
            [
                "## Failed Backends",
                "",
            ]
        )
        for name in failed:
            result = comparison_result.results.get(name)
            if result and result.error:
                lines.append(f"- **{name}:** {result.error[:200]}")
        lines.append("")

    # Recommendations
    lines.extend(
        [
            "---",
            "",
            "## Next Steps",
            "",
            f"1. Use **{selection.canonical_backend}** as the canonical backend for StageBridge",
            f"2. Preserve **{selection.alternatives[0] if selection.alternatives else 'N/A'}** as alternative for robustness checks",
            "3. Monitor downstream transition quality with canonical backend",
            "4. Re-run benchmark if data or requirements change significantly",
            "",
        ]
    )

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report


def save_canonical_decision(
    selection: BackendSelection,
    output_dir: Path,
) -> Path:
    """
    Save canonical backend decision as JSON artifact.

    Creates:
    - canonical_backend.json: Machine-readable selection
    - backend_selection_report.md: Human-readable report

    Args:
        selection: BackendSelection result
        output_dir: Output directory

    Returns:
        Path to canonical_backend.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "canonical_backend.json"
    with open(json_path, "w") as f:
        json.dump(selection.to_dict(), f, indent=2)

    # Save justification as separate markdown
    md_path = output_dir / "backend_selection_report.md"
    with open(md_path, "w") as f:
        f.write(selection.justification)

    return json_path


def load_canonical_decision(output_dir: Path) -> BackendSelection:
    """Load canonical backend decision from JSON artifact."""
    json_path = Path(output_dir) / "canonical_backend.json"

    with open(json_path) as f:
        data = json.load(f)

    return BackendSelection(
        canonical_backend=data["canonical_backend"],
        selection_score=data["selection_score"],
        justification=data["justification"],
        category_scores=data.get("category_scores", {}),
        alternatives=data.get("alternatives", []),
        alternative_scores=data.get("alternative_scores", {}),
        metadata=data.get("metadata", {}),
    )
