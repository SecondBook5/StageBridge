"""Genomic interpretation report generation.

Generates publication-ready reports with appropriate limitation language
and clear scientific framing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LIMITATION_TEXT = """
## Limitations and Caveats

### Mutation Truth Set
WES-derived variants were treated as the mutation truth set. Spatial transcriptomics
and snRNA-seq were used only to assess expressed variant evidence or clone-associated
transcriptional programs. **Absence of alternate reads in RNA-based data should not
be interpreted as absence of the underlying DNA alteration.**

### Germline Interpretation
Germline variants were annotated using ACMG/AMP-aligned classification based on
ClinVar significance. These are **automated interpretations, not clinical-grade
ACMG adjudications**. All pathogenic/likely pathogenic calls require review by
a clinical geneticist before clinical use.

### Somatic Actionability
Somatic variant actionability was assessed using OncoKB or similar knowledgebases.
Variants in known driver genes without variant-level evidence were labeled as
"gene_level_cancer_relevance" rather than "clinically actionable."

### Clonality Estimation
When tumor purity and copy number data were unavailable, clonality was estimated
using naive VAF thresholds. This is **approximate only** and should not be used
for definitive clonality calls without proper correction for purity and CNV.

### Clinical Validation
This analysis represents **translational interpretation, not clinical validation**.
Clinical validation requires:
- Longitudinal lesion tracking
- Independent validation cohorts
- Endpoint-linked patient outcomes (survival, recurrence, treatment response)

### RNA-Based Variant Evidence
RNA expression of mutations is sparse. Low/absent reads do not exclude DNA-level
mutations. Expression evidence is supportive, not definitive.
"""


def write_markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert DataFrame to Markdown table.

    Args:
        df: DataFrame to convert
        max_rows: Maximum rows to include

    Returns:
        Markdown table string
    """
    if len(df) == 0:
        return "*No data*\n"

    df_display = df.head(max_rows)

    lines = []

    headers = " | ".join(str(c) for c in df_display.columns)
    lines.append(f"| {headers} |")

    separator = " | ".join("---" for _ in df_display.columns)
    lines.append(f"| {separator} |")

    for _, row in df_display.iterrows():
        values = " | ".join(str(v)[:50] for v in row.values)
        lines.append(f"| {values} |")

    if len(df) > max_rows:
        lines.append(f"\n*Showing {max_rows} of {len(df)} rows*")

    return "\n".join(lines)


def format_enrichment_results(df: pd.DataFrame) -> str:
    """Format enrichment results for report.

    Args:
        df: Enrichment results DataFrame

    Returns:
        Formatted Markdown string
    """
    if len(df) == 0:
        return "*No enrichment tests performed*\n"

    lines = []

    significant = df[df["q_value"] < 0.05].sort_values("q_value")
    if len(significant) > 0:
        lines.append("### Significant Enrichments (FDR < 0.05)\n")
        lines.append(write_markdown_table(
            significant[["feature_name", "effect_size", "p_value", "q_value", "n_high", "n_low"]]
        ))
    else:
        lines.append("### No Significant Enrichments Found (FDR < 0.05)\n")

    lines.append("\n### All Tested Features\n")
    lines.append(write_markdown_table(
        df[["feature_name", "effect_size", "p_value", "q_value"]].sort_values("p_value")
    ))

    return "\n".join(lines)


def generate_genomic_interpretation_report(
    output_dir: str | Path,
    variant_master_df: pd.DataFrame | None = None,
    germline_df: pd.DataFrame | None = None,
    actionability_df: pd.DataFrame | None = None,
    clonality_df: pd.DataFrame | None = None,
    spatial_evidence_df: pd.DataFrame | None = None,
    enrichment_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> Path:
    """Generate comprehensive genomic interpretation report.

    Args:
        output_dir: Output directory
        variant_master_df: Variant master table
        germline_df: Germline annotations
        actionability_df: Somatic actionability annotations
        clonality_df: Clonality estimates
        spatial_evidence_df: Spatial variant evidence
        enrichment_df: Transition enrichment results
        config: Analysis configuration

    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "genomic_interpretation_report.md"

    sections = []

    sections.append("# StageBridge Genomic Interpretation Report\n")
    sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    sections.append("## Overview\n")
    sections.append(
        "This report provides **translational genomic interpretation** by integrating "
        "WES-derived variants with StageBridge transition field analysis. The goal is to "
        "test whether high-transition niches (putative progression zones) are enriched "
        "for clinically interpretable genomic features.\n"
    )

    sections.append("## Data Summary\n")

    if variant_master_df is not None:
        n_variants = len(variant_master_df)
        n_samples = variant_master_df["sample_id"].nunique() if "sample_id" in variant_master_df.columns else "unknown"
        n_genes = variant_master_df["gene"].nunique() if "gene" in variant_master_df.columns else "unknown"
        sections.append(f"- **Total variants**: {n_variants}\n")
        sections.append(f"- **Samples**: {n_samples}\n")
        sections.append(f"- **Genes affected**: {n_genes}\n")
    else:
        sections.append("- *No variant data provided*\n")

    sections.append("\n## Germline ACMG-Aligned Interpretation\n")

    if germline_df is not None and len(germline_df) > 0:
        class_counts = germline_df["acmg_aligned_classification"].value_counts()
        sections.append("### Classification Summary\n")
        for cls, count in class_counts.items():
            sections.append(f"- **{cls}**: {count}\n")

        path_lp = germline_df[germline_df["acmg_aligned_classification"].isin([
            "pathogenic", "likely_pathogenic"
        ])]
        if len(path_lp) > 0:
            sections.append("\n### Pathogenic/Likely Pathogenic Variants\n")
            sections.append(write_markdown_table(
                path_lp[["variant_id", "gene", "acmg_aligned_classification", "clinvar_significance"]]
            ))

        sections.append("\n> **Note**: These are ACMG/AMP-aligned interpretations, not clinical-grade ")
        sections.append("ACMG adjudications. Requires clinical genetics review.\n")
    else:
        sections.append("*No germline variants analyzed*\n")

    sections.append("\n## Somatic Actionability (OncoKB-Informed)\n")

    if actionability_df is not None and len(actionability_df) > 0:
        level_counts = actionability_df["actionability_level"].value_counts()
        sections.append("### Actionability Level Summary\n")
        for level, count in level_counts.items():
            sections.append(f"- **{level}**: {count}\n")

        actionable = actionability_df[actionability_df["actionability_level"].isin([
            "level_1", "level_2", "level_3A", "level_3B", "level_4"
        ])]
        if len(actionable) > 0:
            sections.append("\n### Actionable Variants\n")
            sections.append(write_markdown_table(
                actionable[["variant_id", "gene", "actionability_level", "therapeutic_implication"]]
            ))

        sections.append("\n> **Note**: Variants in driver genes without variant-level evidence ")
        sections.append("are labeled 'gene_level_cancer_relevance', not 'clinically actionable'.\n")
    else:
        sections.append("*No somatic variants analyzed for actionability*\n")

    sections.append("\n## Clonality Analysis\n")

    if clonality_df is not None and len(clonality_df) > 0:
        label_counts = clonality_df["clonality_label"].value_counts()
        sections.append("### Clonality Distribution\n")
        for label, count in label_counts.items():
            sections.append(f"- **{label}**: {count}\n")

        method_counts = clonality_df["method"].value_counts()
        sections.append("\n### Estimation Methods Used\n")
        for method, count in method_counts.items():
            sections.append(f"- {method}: {count} variants\n")

        if "naive_vaf" in method_counts.index:
            sections.append("\n> **Caution**: Naive VAF-based clonality is approximate. ")
            sections.append("Proper correction requires purity and copy number data.\n")
    else:
        sections.append("*No clonality analysis performed*\n")

    sections.append("\n## Spatial Variant Evidence\n")

    if spatial_evidence_df is not None and len(spatial_evidence_df) > 0:
        evidence_counts = spatial_evidence_df["evidence_label"].value_counts()
        sections.append("### Evidence Label Distribution\n")
        for label, count in evidence_counts.items():
            sections.append(f"- **{label}**: {count}\n")

        alt_supported = spatial_evidence_df[spatial_evidence_df["evidence_label"] == "alt_supported"]
        if len(alt_supported) > 0:
            sections.append(f"\n### Variants with Expressed Alt Allele Support\n")
            sections.append(f"Found {len(alt_supported)} cell-variant pairs with alt-supporting reads.\n")

        sections.append("\n> **Critical**: Absence of alternate reads does NOT mean the mutation ")
        sections.append("is absent. RNA coverage is sparse and expression-dependent.\n")
    else:
        sections.append("*No spatial variant evidence analyzed*\n")

    sections.append("\n## Transition Zone Genomic Enrichment\n")

    if enrichment_df is not None and len(enrichment_df) > 0:
        sections.append(format_enrichment_results(enrichment_df))
    else:
        sections.append("*No enrichment analysis performed*\n")

    sections.append(LIMITATION_TEXT)

    sections.append("\n## Recommended Next Steps\n")
    sections.append(
        "1. **Clinical genetics review** of pathogenic/likely pathogenic germline variants\n"
        "2. **Validation** of actionable somatic variants in independent cohort\n"
        "3. **Orthogonal confirmation** of expressed variant evidence (e.g., targeted DNA sequencing)\n"
        "4. **Integration with clinical outcomes** to assess prognostic/predictive value\n"
        "5. **Functional validation** of transition-enriched genomic features\n"
    )

    report_content = "\n".join(sections)

    with open(report_path, "w") as f:
        f.write(report_content)

    logger.info(f"Wrote genomic interpretation report to {report_path}")

    return report_path


def save_summary_json(
    output_dir: str | Path,
    variant_master_df: pd.DataFrame | None = None,
    germline_df: pd.DataFrame | None = None,
    actionability_df: pd.DataFrame | None = None,
    clonality_df: pd.DataFrame | None = None,
    enrichment_df: pd.DataFrame | None = None,
) -> Path:
    """Save analysis summary as JSON.

    Args:
        output_dir: Output directory
        Various DataFrames with results

    Returns:
        Path to JSON file
    """
    output_dir = Path(output_dir)
    json_path = output_dir / "genomic_summary.json"

    summary = {
        "generated": datetime.now().isoformat(),
        "analysis_type": "translational_genomic_interpretation",
        "variants": {},
        "germline": {},
        "somatic": {},
        "clonality": {},
        "enrichment": {},
    }

    if variant_master_df is not None:
        summary["variants"] = {
            "n_total": len(variant_master_df),
            "n_samples": variant_master_df["sample_id"].nunique() if "sample_id" in variant_master_df.columns else 0,
            "n_genes": variant_master_df["gene"].nunique() if "gene" in variant_master_df.columns else 0,
        }

    if germline_df is not None:
        summary["germline"] = {
            "n_total": len(germline_df),
            "by_classification": germline_df["acmg_aligned_classification"].value_counts().to_dict(),
        }

    if actionability_df is not None:
        summary["somatic"] = {
            "n_total": len(actionability_df),
            "by_actionability": actionability_df["actionability_level"].value_counts().to_dict(),
        }

    if clonality_df is not None:
        summary["clonality"] = {
            "n_total": len(clonality_df),
            "by_label": clonality_df["clonality_label"].value_counts().to_dict(),
        }

    if enrichment_df is not None:
        sig = enrichment_df[enrichment_df["q_value"] < 0.05]
        summary["enrichment"] = {
            "n_tests": len(enrichment_df),
            "n_significant_fdr05": len(sig),
            "top_features": sig.nsmallest(5, "q_value")["feature_name"].tolist() if len(sig) > 0 else [],
        }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return json_path


def create_publication_ready_summary(
    enrichment_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Create publication-ready summary table.

    Args:
        enrichment_df: Enrichment results
        output_path: Output path

    Returns:
        Path to output file
    """
    output_path = Path(output_path)

    if len(enrichment_df) == 0:
        pd.DataFrame().to_csv(output_path, sep="\t", index=False)
        return output_path

    pub_df = enrichment_df.copy()

    pub_df["effect_size"] = pub_df["effect_size"].round(3)
    pub_df["p_value"] = pub_df["p_value"].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    pub_df["q_value"] = pub_df["q_value"].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")

    pub_df = pub_df.rename(columns={
        "feature_name": "Feature",
        "effect_size": "Effect Size (Cohen's d / log2 OR)",
        "p_value": "P-value",
        "q_value": "FDR",
        "n_high": "N (High Transition)",
        "n_low": "N (Low Transition)",
    })

    pub_df.to_csv(output_path, sep="\t", index=False)

    return output_path
