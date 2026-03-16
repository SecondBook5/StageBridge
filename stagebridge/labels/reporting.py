"""Reporting and figure generation for the label-repair workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig, OmegaConf


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    """Read a dotted config value from OmegaConf or dict payloads.

    Args:
        cfg: Config tree.
        dotted: Dotted key path.
        default: Fallback when the key is missing.
    """
    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, dotted)
        return default if value is None else value
    current: Any = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def _ensure_dir(path: str | Path) -> Path:
    """Create a directory when missing and return the resolved path.

    Args:
        path: Directory path to create.
    """
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _save_placeholder(path: Path, *, title: str, message: str) -> None:
    """Save a placeholder figure when a backend produced no usable outputs.

    Args:
        path: Output image path.
        title: Panel title.
        message: Body text.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_label_repair_reports(
    *,
    cleaned_manifest: pd.DataFrame,
    cna_summary: pd.DataFrame,
    clonal_summary: pd.DataFrame,
    phylogeny_summary: pd.DataFrame,
    pathology_summary: pd.DataFrame,
    refined_labels: pd.DataFrame,
    edge_support: pd.DataFrame,
    donor_support: pd.DataFrame,
    split_report: dict[str, Any],
    output_root: Path,
    cfg: DictConfig | dict[str, Any],
) -> dict[str, str]:
    """Write tables, figures, and recommendation report for label repair.

    Args:
        cleaned_manifest: Cleaned cohort manifest.
        cna_summary: Normalized CNA summary table.
        clonal_summary: Normalized clonal summary table.
        phylogeny_summary: Normalized phylogeny summary table.
        pathology_summary: Optional pathology summary table.
        refined_labels: Refined lesion target table.
        edge_support: Edge-level support diagnostics.
        donor_support: Donor-level support diagnostics.
        split_report: JSON-serializable viability report.
        output_root: Report root directory.
        cfg: Active config tree.
    """
    reports_root = _ensure_dir(output_root)
    tables_root = _ensure_dir(reports_root / "tables")
    figures_root = _ensure_dir(reports_root / "figures")
    artifacts_root = _ensure_dir(reports_root / "artifacts")

    cleaned_manifest.to_csv(tables_root / "cleaned_cohort_manifest.csv", index=False)
    cna_summary.to_csv(tables_root / "lesion_cna_summary.csv", index=False)
    clonal_summary.to_csv(tables_root / "lesion_clone_summary.csv", index=False)
    phylogeny_summary.to_csv(tables_root / "lesion_phylogeny_summary.csv", index=False)
    pathology_summary.to_csv(tables_root / "lesion_pathology_summary.csv", index=False)
    refined_labels.to_csv(tables_root / "lesion_refined_labels.csv", index=False)
    refined_labels.loc[
        :,
        [
            "lesion_id",
            "patient_id",
            "donor_id",
            "stage",
            "edge_label",
            "progression_risk_score",
            "confidence_tier",
        ],
    ].to_csv(
        tables_root / "lesion_progression_risk_scores.csv",
        index=False,
    )
    edge_support.to_csv(tables_root / "edge_label_support_summary.csv", index=False)
    donor_support.to_csv(tables_root / "donor_support_summary.csv", index=False)
    (artifacts_root / "split_viability_report.json").write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )

    dataset_table = cleaned_manifest.groupby(
        ["stage", "edge_label"], dropna=False, as_index=False
    ).agg(
        n_lesions=("lesion_id", "nunique"),
        n_donors=("donor_id", "nunique"),
        n_wes_supported=("has_wes", lambda values: int(pd.Series(values).sum())),
    )
    dataset_table.to_csv(
        tables_root / "table1_cohort_composition_and_wes_availability.csv", index=False
    )
    cna_summary.to_csv(tables_root / "table2_cna_summary_by_lesion_and_stage.csv", index=False)
    phylo_table = clonal_summary.merge(
        phylogeny_summary[
            ["lesion_id", "clone_sharing_score", "descendant_sharing_score", "tree_available"]
        ],
        on="lesion_id",
        how="left",
    )
    phylo_table.to_csv(tables_root / "table3_clonal_phylogeny_summary.csv", index=False)
    edge_support[
        [
            "edge_label",
            "positive_lesions",
            "negative_lesions",
            "uncertain_lesions",
            "excluded_lesions",
        ]
    ].to_csv(
        tables_root / "table4_refined_label_decision_counts.csv",
        index=False,
    )
    donor_support.to_csv(tables_root / "table5_donor_held_out_viability.csv", index=False)
    edge_support[["edge_label", "recommended_target", "reason"]].to_csv(
        tables_root / "table6_recommended_target_type.csv",
        index=False,
    )

    # Figure 1: before/after support.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    before = cleaned_manifest.groupby("edge_label", as_index=False)["original_label"].value_counts(
        dropna=False
    )
    if not before.empty:
        before["label_name"] = (
            before["original_label"].map({1.0: "positive", 0.0: "negative"}).fillna("missing")
        )
        pivot = before.pivot(index="edge_label", columns="label_name", values="count").fillna(0.0)
        pivot.plot(kind="bar", stacked=True, ax=axes[0], title="Before refinement")
    else:
        axes[0].axis("off")
    after = edge_support.set_index("edge_label")[
        ["positive_lesions", "negative_lesions", "uncertain_lesions", "excluded_lesions"]
    ]
    if not after.empty:
        after.plot(kind="bar", stacked=True, ax=axes[1], title="After refinement")
    else:
        axes[1].axis("off")
    fig.suptitle("Figure 1. Cohort label support before and after refinement")
    fig.tight_layout()
    fig.savefig(
        figures_root / "figure1_cohort_label_support_before_after.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: evolutionary evidence summaries.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    evidence_sources = [
        ("tmb", cleaned_manifest),
        (
            "driver_proxy",
            cleaned_manifest.assign(
                driver_proxy=cleaned_manifest["availability_trace"]
                .astype(str)
                .str.contains("wes")
                .astype(int)
            ),
        ),
        ("cna_burden", cna_summary),
        ("descendant_sharing_score", phylogeny_summary),
    ]
    for ax, (column, frame) in zip(axes.flat, evidence_sources):
        if (
            column not in frame.columns
            or pd.to_numeric(frame[column], errors="coerce").dropna().empty
        ):
            ax.axis("off")
            ax.set_title(column)
            ax.text(0.5, 0.5, "No parsed evidence available", ha="center", va="center")
            continue
        ax.hist(pd.to_numeric(frame[column], errors="coerce").dropna(), bins=10, color="#4c78a8")
        ax.set_title(column)
    fig.suptitle("Figure 2. Evolutionary evidence summaries")
    fig.tight_layout()
    fig.savefig(
        figures_root / "figure2_evolutionary_evidence_summaries.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # Figure 3: patient-level phylogeny summary panels.
    if phylogeny_summary["tree_available"].fillna(False).astype(bool).any():
        fig, ax = plt.subplots(figsize=(12, 6))
        heat = phylogeny_summary.pivot_table(
            index="patient_id",
            columns="stage",
            values="descendant_sharing_score",
            aggfunc="mean",
        )
        image = ax.imshow(heat.fillna(0.0).to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(list(heat.columns), rotation=45, ha="right")
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(list(heat.index))
        ax.set_title("Figure 3. Patient-level phylogeny summary panels")
        fig.colorbar(image, ax=ax, label="Descendant sharing score")
        fig.tight_layout()
        fig.savefig(
            figures_root / "figure3_patient_phylogeny_summary_panels.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        _save_placeholder(
            figures_root / "figure3_patient_phylogeny_summary_panels.png",
            title="Figure 3. Patient-level phylogeny summary panels",
            message="No parsed PhylogicNDT, Pairtree, or Treeomics summaries were available for the active run.",
        )

    # Figure 4: refined target diagnostics.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    refined_labels["progression_risk_score"].plot(
        kind="hist", bins=12, ax=axes[0], title="Risk score distribution"
    )
    refined_labels["refined_binary_label"].value_counts(dropna=False).plot(
        kind="bar", ax=axes[1], title="Refined label composition"
    )
    ctab = pd.crosstab(refined_labels["edge_label"], refined_labels["refined_binary_label"])
    if not ctab.empty:
        ctab.plot(kind="bar", stacked=True, ax=axes[2], title="Edge-by-label composition")
    else:
        axes[2].axis("off")
    fig.suptitle("Figure 4. Refined target diagnostics")
    fig.tight_layout()
    fig.savefig(
        figures_root / "figure4_refined_target_diagnostics.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # Figure 5: split viability diagnostics.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    edge_support.set_index("edge_label")[["positive_donors", "negative_donors"]].plot(
        kind="bar", ax=axes[0], title="Donor support by class"
    )
    viability_plot = edge_support.set_index("edge_label")[
        ["binary_viable", "continuous_viable"]
    ].astype(int)
    viability_plot.plot(kind="bar", ax=axes[1], title="Target viability")
    fig.suptitle("Figure 5. Split viability diagnostics")
    fig.tight_layout()
    fig.savefig(
        figures_root / "figure5_split_viability_diagnostics.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    recommendation_lines = [
        "# Target Recommendation Report",
        "",
        "This report summarizes whether each edge should be treated as a binary classification task, a continuous risk target, a descriptive-only analysis, or excluded from formal supervised benchmarking.",
        "",
    ]
    for row in edge_support.itertuples(index=False):
        recommendation_lines.extend(
            [
                f"## {row.edge_label}",
                f"- Recommended target: `{row.recommended_target}`",
                f"- Binary viable: `{bool(row.binary_viable)}`",
                f"- Continuous viable: `{bool(row.continuous_viable)}`",
                f"- Positive donors: `{int(row.positive_donors)}`",
                f"- Negative donors: `{int(row.negative_donors)}`",
                f"- Reason: {row.reason}",
                "",
            ]
        )
    readme_lines = [
        "# Label Repair Workflow",
        "",
        "## Required inputs",
        "- Active LUAD lesion metadata via the existing StageBridge data layer",
        "- Existing lesion WES proxy features from `wes_features.parquet`",
        "- Optional parse-only CNA, clonal, phylogeny, or pathology summaries",
        "",
        "## Parse-only mode",
        "- Set `labels.parse_only=true`",
        "- Provide backend summary paths under `labels.inputs.*`",
        "",
        "## External-tool mode",
        "- Set `labels.parse_only=false`",
        "- Provide executable names and command templates under `labels.external_tools.*`",
        "- The wrappers will fail loudly if a requested executable is unavailable",
        "",
        "## Refined labels",
        "- `positive`, `negative`, `uncertain`, and `exclude` are derived by a transparent rule-based engine",
        "- `progression_risk_score` is continuous and auditable from its component contributions",
        "",
        "## Viability report",
        "- Use `binary_classification` only when donor support exists for both classes",
        "- Use `continuous_risk` when score diversity and donor coverage exist but binary support fails",
        "- Use `descriptive_only` or `exclude` when support remains too weak",
        "",
    ]
    recommendation_path = reports_root / "target_recommendation_report.md"
    recommendation_path.write_text("\n".join(recommendation_lines), encoding="utf-8")
    (reports_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    developer_note_lines = [
        "# Developer Note",
        "",
        "## Current assumptions",
        "- `sample_id` is used as the lesion identifier.",
        "- Existing `wes_features.parquet` is treated as a lesion-level WES proxy layer.",
        "- CNA, clonal, phylogeny, and pathology backends default to parse-only and may produce empty normalized summaries when no external results are configured.",
        "",
        "## External tools expected but not bundled",
        "- FACETS",
        "- CNVkit",
        "- Sequenza",
        "- PyClone-VI",
        "- PhylogicNDT",
        "- Pairtree",
        "- Treeomics",
        "- QuPath",
        "- QuST",
        "",
        "## Current data limitations",
        "- `AAH->AIS` still lacks enough negative donor support for donor-held-out binary benchmarking.",
        "- The current run used no parsed external CNA/clonal/phylogeny/pathology outputs, so refinement relied on curated labels, heuristic provenance, later-stage support, and existing WES proxy features.",
        "- AAH label repair currently supports a conservative continuous-risk recommendation rather than a repaired binary benchmark.",
        "",
    ]
    (reports_root / "DEVELOPER_NOTE.md").write_text(
        "\n".join(developer_note_lines), encoding="utf-8"
    )
    return {
        "reports_root": str(reports_root),
        "tables_root": str(tables_root),
        "figures_root": str(figures_root),
        "artifacts_root": str(artifacts_root),
        "recommendation_report": str(recommendation_path),
    }
