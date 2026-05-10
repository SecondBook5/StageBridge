"""Comprehensive results aggregation for StageBridge.

Produces a one-stop-shop summary of all pipeline outputs:
- Model performance metrics (full, ablations, baselines)
- Biological validation results
- Interpretation analysis results
- Data statistics

Outputs:
- results_summary.json: Machine-readable complete summary
- results_summary.csv: Flat table of key metrics for quick comparison
- results_summary.md: Human-readable markdown report

Usage:
    python -m stagebridge.evaluation.results_summary \
        --results-dir /path/to/results \
        --output-dir /path/to/summary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MetricSummary:
    """Summary statistics for a metric across runs."""
    mean: float
    std: float
    min: float
    max: float
    n: int
    values: list[float] = field(default_factory=list)

    @classmethod
    def from_values(cls, values: list[float]) -> "MetricSummary":
        if not values:
            return cls(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, n=0, values=[])
        return cls(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            n=len(values),
            values=values,
        )


@dataclass
class ModelResult:
    """Results for a single model configuration."""
    name: str
    category: str  # "full", "ablation", "baseline"
    val_loss: MetricSummary | None = None
    stage_accuracy: MetricSummary | None = None
    stage_f1: MetricSummary | None = None
    pathway_r2: MetricSummary | None = None
    proliferation_r2: MetricSummary | None = None
    delta_vs_full: float | None = None
    extra_metrics: dict = field(default_factory=dict)


@dataclass
class ResultsSummary:
    """Complete pipeline results summary."""
    generated_at: str
    results_dir: str

    # Model results
    full_model: ModelResult | None = None
    ablations: dict[str, ModelResult] = field(default_factory=dict)
    baselines: dict[str, ModelResult] = field(default_factory=dict)

    # HPO results
    hpo_best_params: dict = field(default_factory=dict)
    hpo_best_value: float | None = None
    hpo_n_trials: int = 0

    # Data statistics
    n_cells: int = 0
    n_samples: int = 0
    n_donors: int = 0
    stage_distribution: dict[str, int] = field(default_factory=dict)

    # Interpretation results
    interpretation: dict = field(default_factory=dict)

    # Biological validation
    biological: dict = field(default_factory=dict)


def load_training_metrics(json_path: Path) -> dict[str, float]:
    """Extract metrics from training_summary.json."""
    with open(json_path) as f:
        data = json.load(f)

    metrics = {}

    # Handle different nesting structures
    if "transition" in data:
        trans = data["transition"]
        if "best_val_loss" in trans:
            metrics["val_loss"] = trans["best_val_loss"]
        if "best_metrics" in trans:
            bm = trans["best_metrics"]
            metrics["stage_accuracy"] = bm.get("stage_accuracy")
            metrics["stage_f1"] = bm.get("stage_f1_macro")
            metrics["pathway_r2"] = bm.get("pathway_r2")
            metrics["proliferation_r2"] = bm.get("proliferation_r2")

    if "best_val_loss" in data:
        metrics["val_loss"] = data["best_val_loss"]

    if "metrics" in data:
        m = data["metrics"]
        metrics["val_loss"] = m.get("val_loss", m.get("best_val_loss"))
        metrics["stage_accuracy"] = m.get("stage_accuracy")
        metrics["stage_f1"] = m.get("stage_f1_macro", m.get("stage_f1"))

    return {k: v for k, v in metrics.items() if v is not None}


def load_ablation_metrics(json_path: Path) -> dict[str, float]:
    """Extract metrics from ablation_*.json."""
    with open(json_path) as f:
        data = json.load(f)

    metrics = {}

    if "metrics" in data:
        m = data["metrics"]
        if "transition" in m:
            metrics["val_loss"] = m["transition"].get("best_val_loss")
        else:
            metrics["val_loss"] = m.get("val_loss", m.get("best_val_loss"))
        metrics["stage_accuracy"] = m.get("stage_accuracy")
        metrics["stage_f1"] = m.get("stage_f1_macro", m.get("stage_f1"))

    if "delta_vs_full" in data:
        metrics["delta_vs_full"] = data["delta_vs_full"]

    return {k: v for k, v in metrics.items() if v is not None}


def aggregate_model_results(
    results_dir: Path,
    pattern: str,
    category: str,
    name_extractor: callable,
) -> dict[str, ModelResult]:
    """Aggregate results across folds/seeds for a model category."""
    results_by_name: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for json_path in results_dir.glob(pattern):
        name = name_extractor(json_path)

        if "ablation" in pattern:
            metrics = load_ablation_metrics(json_path)
        else:
            metrics = load_training_metrics(json_path)

        for metric_name, value in metrics.items():
            if value is not None and not np.isnan(value):
                results_by_name[name][metric_name].append(value)

    model_results = {}
    for name, metrics_dict in results_by_name.items():
        result = ModelResult(name=name, category=category)

        if "val_loss" in metrics_dict:
            result.val_loss = MetricSummary.from_values(metrics_dict["val_loss"])
        if "stage_accuracy" in metrics_dict:
            result.stage_accuracy = MetricSummary.from_values(metrics_dict["stage_accuracy"])
        if "stage_f1" in metrics_dict:
            result.stage_f1 = MetricSummary.from_values(metrics_dict["stage_f1"])
        if "pathway_r2" in metrics_dict:
            result.pathway_r2 = MetricSummary.from_values(metrics_dict["pathway_r2"])
        if "proliferation_r2" in metrics_dict:
            result.proliferation_r2 = MetricSummary.from_values(metrics_dict["proliferation_r2"])

        model_results[name] = result

    return model_results


def load_hpo_results(results_dir: Path) -> tuple[dict, float | None, int]:
    """Load HPO best params and stats."""
    hpo_path = results_dir / "hpo" / "best_params.json"
    if not hpo_path.exists():
        return {}, None, 0

    with open(hpo_path) as f:
        data = json.load(f)

    best_params = data.get("params", data)
    best_value = data.get("best_value")
    n_trials = data.get("n_trials", 0)

    return best_params, best_value, n_trials


def load_data_stats(data_dir: Path) -> dict:
    """Load data statistics from cells/neighborhoods parquet."""
    stats = {}

    cells_path = data_dir / "cells.parquet"
    if cells_path.exists():
        try:
            cells = pd.read_parquet(cells_path, columns=["stage", "sample_id", "donor_id"])
            stats["n_cells"] = len(cells)
            stats["n_samples"] = cells["sample_id"].nunique() if "sample_id" in cells else 0
            stats["n_donors"] = cells["donor_id"].nunique() if "donor_id" in cells else 0
            stats["stage_distribution"] = cells["stage"].value_counts().to_dict()
        except Exception:
            pass

    neighborhoods_path = data_dir / "neighborhoods.parquet"
    if neighborhoods_path.exists() and "n_cells" not in stats:
        try:
            nhoods = pd.read_parquet(neighborhoods_path, columns=["stage"], engine="fastparquet")
            stats["n_cells"] = len(nhoods)
            stats["stage_distribution"] = nhoods["stage"].value_counts().to_dict()
        except Exception:
            pass

    return stats


def load_interpretation_results(results_dir: Path) -> dict:
    """Load interpretation analysis results."""
    interp = {}

    # Ablation importance
    ablation_path = results_dir / "interpretation" / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            interp["token_ablation"] = json.load(f)

    # Attention summary
    attention_path = results_dir / "interpretation" / "attention_summary.json"
    if attention_path.exists():
        with open(attention_path) as f:
            interp["attention"] = json.load(f)

    # Network summary
    network_path = results_dir / "interpretation" / "network_summary.json"
    if network_path.exists():
        with open(network_path) as f:
            interp["network"] = json.load(f)

    # Manifold summary
    manifold_path = results_dir / "interpretation" / "manifold_summary.json"
    if manifold_path.exists():
        with open(manifold_path) as f:
            interp["manifold"] = json.load(f)

    return interp


def compute_deltas(summary: ResultsSummary) -> None:
    """Compute delta vs full model for ablations and baselines."""
    if summary.full_model is None or summary.full_model.val_loss is None:
        return

    full_loss = summary.full_model.val_loss.mean

    for result in summary.ablations.values():
        if result.val_loss is not None:
            result.delta_vs_full = (result.val_loss.mean - full_loss) / full_loss * 100

    for result in summary.baselines.values():
        if result.val_loss is not None:
            result.delta_vs_full = (result.val_loss.mean - full_loss) / full_loss * 100


def generate_summary(results_dir: Path, data_dir: Path | None = None) -> ResultsSummary:
    """Generate complete results summary."""
    summary = ResultsSummary(
        generated_at=datetime.now().isoformat(),
        results_dir=str(results_dir),
    )

    # Full model results
    full_results = aggregate_model_results(
        results_dir,
        "full/fold_*/seed_*/training_summary.json",
        "full",
        lambda p: "full",
    )
    if "full" in full_results:
        summary.full_model = full_results["full"]

    # Ablation results
    summary.ablations = aggregate_model_results(
        results_dir,
        "ablations/*/fold_*/seed_*/ablation_*.json",
        "ablation",
        lambda p: p.parent.parent.parent.name,
    )

    # Baseline results
    summary.baselines = aggregate_model_results(
        results_dir,
        "baselines/*/fold_*/seed_*/baseline_*.json",
        "baseline",
        lambda p: p.parent.parent.parent.name,
    )
    # Also check alternative pattern
    alt_baselines = aggregate_model_results(
        results_dir,
        "baselines/*/fold_*/seed_*/training_summary.json",
        "baseline",
        lambda p: p.parent.parent.parent.name,
    )
    summary.baselines.update(alt_baselines)

    # Compute deltas
    compute_deltas(summary)

    # HPO results
    summary.hpo_best_params, summary.hpo_best_value, summary.hpo_n_trials = load_hpo_results(results_dir)

    # Data stats
    if data_dir and data_dir.exists():
        stats = load_data_stats(data_dir)
        summary.n_cells = stats.get("n_cells", 0)
        summary.n_samples = stats.get("n_samples", 0)
        summary.n_donors = stats.get("n_donors", 0)
        summary.stage_distribution = stats.get("stage_distribution", {})

    # Interpretation results
    summary.interpretation = load_interpretation_results(results_dir)

    return summary


def summary_to_dict(summary: ResultsSummary) -> dict:
    """Convert summary to JSON-serializable dict."""
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)

    return convert(summary)


def summary_to_dataframe(summary: ResultsSummary) -> pd.DataFrame:
    """Convert summary to flat DataFrame for easy comparison."""
    rows = []

    # Full model
    if summary.full_model:
        row = {
            "model": "StageBridge (full)",
            "category": "full",
            "val_loss_mean": summary.full_model.val_loss.mean if summary.full_model.val_loss else None,
            "val_loss_std": summary.full_model.val_loss.std if summary.full_model.val_loss else None,
            "stage_acc_mean": summary.full_model.stage_accuracy.mean if summary.full_model.stage_accuracy else None,
            "stage_f1_mean": summary.full_model.stage_f1.mean if summary.full_model.stage_f1 else None,
            "delta_vs_full": 0.0,
            "n_runs": summary.full_model.val_loss.n if summary.full_model.val_loss else 0,
        }
        rows.append(row)

    # Ablations
    for name, result in sorted(summary.ablations.items()):
        row = {
            "model": name,
            "category": "ablation",
            "val_loss_mean": result.val_loss.mean if result.val_loss else None,
            "val_loss_std": result.val_loss.std if result.val_loss else None,
            "stage_acc_mean": result.stage_accuracy.mean if result.stage_accuracy else None,
            "stage_f1_mean": result.stage_f1.mean if result.stage_f1 else None,
            "delta_vs_full": result.delta_vs_full,
            "n_runs": result.val_loss.n if result.val_loss else 0,
        }
        rows.append(row)

    # Baselines
    for name, result in sorted(summary.baselines.items()):
        row = {
            "model": name,
            "category": "baseline",
            "val_loss_mean": result.val_loss.mean if result.val_loss else None,
            "val_loss_std": result.val_loss.std if result.val_loss else None,
            "stage_acc_mean": result.stage_accuracy.mean if result.stage_accuracy else None,
            "stage_f1_mean": result.stage_f1.mean if result.stage_f1 else None,
            "delta_vs_full": result.delta_vs_full,
            "n_runs": result.val_loss.n if result.val_loss else 0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summary_to_markdown(summary: ResultsSummary) -> str:
    """Generate human-readable markdown report."""
    lines = [
        "# StageBridge Results Summary",
        f"\nGenerated: {summary.generated_at}",
        f"Results directory: `{summary.results_dir}`",
        "",
    ]

    # Data overview
    if summary.n_cells > 0:
        lines.extend([
            "## Data Overview",
            "",
            f"- **Cells**: {summary.n_cells:,}",
            f"- **Samples**: {summary.n_samples}",
            f"- **Donors**: {summary.n_donors}",
            "",
            "**Stage Distribution**:",
            "",
        ])
        for stage, count in sorted(summary.stage_distribution.items()):
            lines.append(f"- {stage}: {count:,}")
        lines.append("")

    # HPO
    if summary.hpo_best_params:
        lines.extend([
            "## HPO Results",
            "",
            f"- **Best value**: {summary.hpo_best_value:.4f}" if summary.hpo_best_value else "",
            f"- **Trials**: {summary.hpo_n_trials}",
            "",
            "**Best parameters**:",
            "```json",
            json.dumps(summary.hpo_best_params, indent=2),
            "```",
            "",
        ])

    # Full model
    if summary.full_model and summary.full_model.val_loss:
        fm = summary.full_model
        lines.extend([
            "## Full Model Performance",
            "",
            f"- **Val Loss**: {fm.val_loss.mean:.4f} +/- {fm.val_loss.std:.4f} (n={fm.val_loss.n})",
        ])
        if fm.stage_accuracy:
            lines.append(f"- **Stage Accuracy**: {fm.stage_accuracy.mean:.3f} +/- {fm.stage_accuracy.std:.3f}")
        if fm.stage_f1:
            lines.append(f"- **Stage F1**: {fm.stage_f1.mean:.3f} +/- {fm.stage_f1.std:.3f}")
        lines.append("")

    # Ablations
    if summary.ablations:
        lines.extend([
            "## Ablation Study",
            "",
            "| Ablation | Val Loss | Delta | n |",
            "|----------|----------|-------|---|",
        ])
        sorted_abl = sorted(
            summary.ablations.items(),
            key=lambda x: x[1].delta_vs_full if x[1].delta_vs_full else 0,
            reverse=True,
        )
        for name, result in sorted_abl:
            if result.val_loss:
                delta_str = f"{result.delta_vs_full:+.1f}%" if result.delta_vs_full else "N/A"
                lines.append(f"| {name} | {result.val_loss.mean:.4f} | {delta_str} | {result.val_loss.n} |")
        lines.append("")

    # Baselines
    if summary.baselines:
        lines.extend([
            "## Baseline Comparison",
            "",
            "| Baseline | Val Loss | Delta | n |",
            "|----------|----------|-------|---|",
        ])
        sorted_base = sorted(
            summary.baselines.items(),
            key=lambda x: x[1].val_loss.mean if x[1].val_loss else float("inf"),
        )
        for name, result in sorted_base:
            if result.val_loss:
                delta_str = f"{result.delta_vs_full:+.1f}%" if result.delta_vs_full else "N/A"
                lines.append(f"| {name} | {result.val_loss.mean:.4f} | {delta_str} | {result.val_loss.n} |")
        lines.append("")

    # Interpretation highlights
    if summary.interpretation:
        lines.extend([
            "## Interpretation Highlights",
            "",
        ])
        if "token_ablation" in summary.interpretation:
            lines.append("**Token Importance** (by ablation delta):")
            ta = summary.interpretation["token_ablation"]
            if isinstance(ta, dict):
                sorted_tokens = sorted(ta.items(), key=lambda x: x[1].get("delta_loss", 0), reverse=True)
                for token, data in sorted_tokens[:5]:
                    if isinstance(data, dict) and "delta_loss" in data:
                        lines.append(f"- {token}: {data['delta_loss']:.4f}")
            lines.append("")

        if "attention" in summary.interpretation:
            attn = summary.interpretation["attention"]
            if isinstance(attn, dict):
                lines.append("**Attention Summary**:")
                for key, val in list(attn.items())[:5]:
                    lines.append(f"- {key}: {val}")
                lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive results summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs:
  results_summary.json  Complete machine-readable summary
  results_summary.csv   Flat table of model metrics
  results_summary.md    Human-readable markdown report
        """,
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=Path,
        required=True,
        help="Results directory containing full/ablations/baselines subdirs",
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=None,
        help="Data directory for loading cell statistics",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: results_dir)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("StageBridge Results Summary Generator")
    print("=" * 60)
    print(f"Results: {args.results_dir}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Generate summary
    print("\nAggregating results...")
    summary = generate_summary(args.results_dir, args.data_dir)

    # Save JSON
    json_path = output_dir / "results_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_to_dict(summary), f, indent=2)
    print(f"  Saved: {json_path}")

    # Save CSV
    csv_path = output_dir / "results_summary.csv"
    df = summary_to_dataframe(summary)
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save Markdown
    md_path = output_dir / "results_summary.md"
    with open(md_path, "w") as f:
        f.write(summary_to_markdown(summary))
    print(f"  Saved: {md_path}")

    # Print summary to console
    print("\n" + summary_to_markdown(summary))

    return 0


if __name__ == "__main__":
    exit(main())
