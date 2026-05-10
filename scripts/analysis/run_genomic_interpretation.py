#!/usr/bin/env python
"""Run genomic interpretation pipeline for StageBridge.

This script provides post-hoc translational interpretation by integrating
WES-derived variants with StageBridge transition field analysis.

IMPORTANT SCIENTIFIC FRAMING:
- WES defines the high-confidence variant truth set
- Spatial/snRNA data localizes expressed variant evidence only
- This is a POST-HOC interpretation layer, not model supervision

Usage:
    python scripts/run_genomic_interpretation.py \\
        --config configs/genomic_interpretation.yaml \\
        --output-dir results/genomics

    # Or with direct arguments:
    python scripts/run_genomic_interpretation.py \\
        --somatic-vcf data/variants/somatic.vcf \\
        --transition-scores results/transition_scores.parquet \\
        --output-dir results/genomics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(
    output_dir: Path,
    somatic_vcf: Path | None = None,
    germline_vcf: Path | None = None,
    annotated_variant_table: Path | None = None,
    clinvar_table: Path | None = None,
    oncokb_table: Path | None = None,
    civic_table: Path | None = None,
    purity_table: Path | None = None,
    cna_table: Path | None = None,
    spatial_metadata: Path | None = None,
    transition_scores: Path | None = None,
    spatial_variant_counts: Path | None = None,
    snrna_variant_counts: Path | None = None,
    adata_path: Path | None = None,
    config: dict | None = None,
    overwrite: bool = False,
) -> dict:
    """Run full genomic interpretation pipeline.

    Args:
        output_dir: Output directory
        Various input paths
        config: Configuration dictionary
        overwrite: Whether to overwrite existing outputs

    Returns:
        Dictionary with output paths
    """
    import pandas as pd

    from stagebridge.genomics.variant_annotation import (
        build_variant_master_table,
        annotate_all_variants,
    )
    from stagebridge.genomics.clonality import estimate_clonality_for_variants
    from stagebridge.genomics.spatial_variant_evidence import (
        load_vartrix_output,
        merge_variant_counts_with_spatial_metadata,
        annotate_spatial_variant_evidence,
    )
    from stagebridge.genomics.transition_enrichment import (
        load_transition_scores,
        generate_transition_genomic_enrichment_table,
    )
    from stagebridge.genomics.reports import (
        generate_genomic_interpretation_report,
        save_summary_json,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    warnings = []

    logger.info("=" * 60)
    logger.info("StageBridge Genomic Interpretation Pipeline")
    logger.info("=" * 60)

    logger.info("\nStep 1: Building variant master table...")

    if not any([somatic_vcf, germline_vcf, annotated_variant_table]):
        logger.error("No variant input provided. Need at least one of: "
                    "--somatic-vcf, --germline-vcf, or --annotated-variant-table")
        return {"error": "No variant input"}

    variant_df = build_variant_master_table(
        somatic_vcf=somatic_vcf,
        germline_vcf=germline_vcf,
        annotated_table=annotated_variant_table,
    )

    if len(variant_df) == 0:
        logger.error("No variants loaded")
        return {"error": "No variants"}

    variant_path = output_dir / "variant_master_table.parquet"
    variant_df.to_parquet(variant_path, index=False)
    outputs["variant_master_table"] = variant_path
    logger.info(f"  Loaded {len(variant_df)} variants")

    logger.info("\nStep 2: Annotating variants...")

    clinvar_df = None
    if clinvar_table and Path(clinvar_table).exists():
        clinvar_df = pd.read_csv(clinvar_table, sep="\t")
        logger.info(f"  Loaded ClinVar annotations: {len(clinvar_df)} records")

    oncokb_df = None
    if oncokb_table and Path(oncokb_table).exists():
        oncokb_df = pd.read_csv(oncokb_table, sep="\t")
        logger.info(f"  Loaded OncoKB annotations: {len(oncokb_df)} records")

    germline_df, somatic_df = annotate_all_variants(
        variant_df,
        clinvar_df=clinvar_df,
        oncokb_df=oncokb_df,
    )

    if germline_df is not None and len(germline_df) > 0:
        germline_path = output_dir / "germline_acmg_aligned_annotations.parquet"
        germline_df.to_parquet(germline_path, index=False)
        outputs["germline_annotations"] = germline_path
        logger.info(f"  Annotated {len(germline_df)} germline variants")
    else:
        germline_df = None
        warnings.append("No germline variants to annotate")

    if somatic_df is not None and len(somatic_df) > 0:
        somatic_path = output_dir / "somatic_actionability_annotations.parquet"
        somatic_df.to_parquet(somatic_path, index=False)
        outputs["somatic_annotations"] = somatic_path
        logger.info(f"  Annotated {len(somatic_df)} somatic variants")
    else:
        somatic_df = None
        warnings.append("No somatic variants to annotate")

    logger.info("\nStep 3: Estimating clonality...")

    purity_df = None
    if purity_table and Path(purity_table).exists():
        purity_df = pd.read_csv(purity_table, sep="\t")
        logger.info(f"  Loaded purity estimates: {len(purity_df)} samples")

    cna_df = None
    if cna_table and Path(cna_table).exists():
        cna_df = pd.read_csv(cna_table, sep="\t")
        logger.info(f"  Loaded CNA segments: {len(cna_df)} segments")

    clonality_df = estimate_clonality_for_variants(
        variant_df,
        purity_df=purity_df,
        cna_df=cna_df,
    )

    clonality_path = output_dir / "clonality_estimates.parquet"
    clonality_df.to_parquet(clonality_path, index=False)
    outputs["clonality_estimates"] = clonality_path
    logger.info(f"  Estimated clonality for {len(clonality_df)} variants")

    logger.info("\nStep 4: Loading spatial variant evidence...")

    spatial_evidence_df = None
    if spatial_variant_counts and Path(spatial_variant_counts).exists():
        try:
            counts_path = Path(spatial_variant_counts)
            if counts_path.is_dir():
                spatial_evidence_df = load_vartrix_output(counts_path)
            else:
                spatial_evidence_df = pd.read_csv(counts_path, sep="\t")

            if spatial_metadata and Path(spatial_metadata).exists():
                meta_df = pd.read_csv(spatial_metadata, sep="\t")
                spatial_evidence_df = merge_variant_counts_with_spatial_metadata(
                    spatial_evidence_df, meta_df
                )

            spatial_evidence_df = annotate_spatial_variant_evidence(spatial_evidence_df)

            evidence_path = output_dir / "spatial_variant_evidence.parquet"
            spatial_evidence_df.to_parquet(evidence_path, index=False)
            outputs["spatial_variant_evidence"] = evidence_path
            logger.info(f"  Loaded {len(spatial_evidence_df)} spatial variant observations")

        except Exception as e:
            logger.warning(f"  Could not load spatial variant evidence: {e}")
            warnings.append(f"Spatial variant evidence failed: {e}")
    else:
        warnings.append("No spatial variant counts provided - skipping spatial evidence")
        logger.info("  Skipped (no spatial variant counts provided)")

    logger.info("\nStep 5: Running transition zone enrichment analysis...")

    enrichment_df = None
    if transition_scores and Path(transition_scores).exists():
        try:
            transition_df = load_transition_scores(transition_scores)
            logger.info(f"  Loaded transition scores for {len(transition_df)} cells/spots")

            high_q = config.get("transition_enrichment", {}).get("high_transition_quantile", 0.90) if config else 0.90
            low_q = config.get("transition_enrichment", {}).get("low_transition_quantile", 0.50) if config else 0.50

            enrichment_df = generate_transition_genomic_enrichment_table(
                transition_df=transition_df,
                variant_evidence_df=spatial_evidence_df,
                actionability_df=somatic_df,
                clonality_df=clonality_df,
                high_quantile=high_q,
                low_quantile=low_q,
            )

            if len(enrichment_df) > 0:
                enrichment_path = output_dir / "transition_genomic_enrichment.parquet"
                enrichment_df.to_parquet(enrichment_path, index=False)
                outputs["transition_enrichment"] = enrichment_path

                summary_path = output_dir / "transition_genomic_summary.tsv"
                enrichment_df.to_csv(summary_path, sep="\t", index=False)
                outputs["transition_summary"] = summary_path

                n_sig = (enrichment_df["q_value"] < 0.05).sum()
                logger.info(f"  Tested {len(enrichment_df)} features, {n_sig} significant (FDR < 0.05)")

        except Exception as e:
            logger.warning(f"  Enrichment analysis failed: {e}")
            warnings.append(f"Enrichment analysis failed: {e}")
    else:
        warnings.append("No transition scores provided - skipping enrichment analysis")
        logger.info("  Skipped (no transition scores provided)")

    logger.info("\nStep 6: Generating reports...")

    report_path = generate_genomic_interpretation_report(
        output_dir=output_dir,
        variant_master_df=variant_df,
        germline_df=germline_df,
        actionability_df=somatic_df,
        clonality_df=clonality_df,
        spatial_evidence_df=spatial_evidence_df,
        enrichment_df=enrichment_df,
        config=config,
    )
    outputs["report"] = report_path

    json_path = save_summary_json(
        output_dir=output_dir,
        variant_master_df=variant_df,
        germline_df=germline_df,
        actionability_df=somatic_df,
        clonality_df=clonality_df,
        enrichment_df=enrichment_df,
    )
    outputs["summary_json"] = json_path

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)

    logger.info("\nOutputs:")
    for name, path in outputs.items():
        logger.info(f"  {name}: {path}")

    if warnings:
        logger.info("\nWarnings:")
        for w in warnings:
            logger.info(f"  - {w}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="StageBridge Genomic Interpretation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--somatic-vcf",
        type=Path,
        help="Path to somatic VCF file",
    )
    parser.add_argument(
        "--germline-vcf",
        type=Path,
        help="Path to germline VCF file",
    )
    parser.add_argument(
        "--annotated-variant-table",
        type=Path,
        help="Path to pre-annotated variant table (TSV/CSV/Parquet)",
    )
    parser.add_argument(
        "--clinvar-table",
        type=Path,
        help="Path to ClinVar annotations table",
    )
    parser.add_argument(
        "--oncokb-table",
        type=Path,
        help="Path to OncoKB annotations table",
    )
    parser.add_argument(
        "--civic-table",
        type=Path,
        help="Path to CIViC annotations table",
    )
    parser.add_argument(
        "--purity-table",
        type=Path,
        help="Path to tumor purity estimates",
    )
    parser.add_argument(
        "--cna-table",
        type=Path,
        help="Path to copy number segments",
    )
    parser.add_argument(
        "--spatial-metadata",
        type=Path,
        help="Path to spatial metadata (barcodes, coordinates)",
    )
    parser.add_argument(
        "--transition-scores",
        type=Path,
        help="Path to StageBridge transition scores",
    )
    parser.add_argument(
        "--spatial-variant-counts",
        type=Path,
        help="Path to spatial variant counts (VarTrix output or TSV)",
    )
    parser.add_argument(
        "--snrna-variant-counts",
        type=Path,
        help="Path to snRNA variant counts",
    )
    parser.add_argument(
        "--adata",
        type=Path,
        help="Path to AnnData h5ad file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )

    args = parser.parse_args()

    config = {}
    if args.config and args.config.exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

        inputs = config.get("inputs", {})
        if not args.somatic_vcf and inputs.get("somatic_vcf"):
            args.somatic_vcf = Path(inputs["somatic_vcf"])
        if not args.germline_vcf and inputs.get("germline_vcf"):
            args.germline_vcf = Path(inputs["germline_vcf"])
        if not args.annotated_variant_table and inputs.get("annotated_variant_table"):
            args.annotated_variant_table = Path(inputs["annotated_variant_table"])
        if not args.transition_scores and inputs.get("transition_scores"):
            args.transition_scores = Path(inputs["transition_scores"])

    outputs = run_pipeline(
        output_dir=args.output_dir,
        somatic_vcf=args.somatic_vcf,
        germline_vcf=args.germline_vcf,
        annotated_variant_table=args.annotated_variant_table,
        clinvar_table=args.clinvar_table,
        oncokb_table=args.oncokb_table,
        civic_table=args.civic_table,
        purity_table=args.purity_table,
        cna_table=args.cna_table,
        spatial_metadata=args.spatial_metadata,
        transition_scores=args.transition_scores,
        spatial_variant_counts=args.spatial_variant_counts,
        snrna_variant_counts=args.snrna_variant_counts,
        adata_path=args.adata,
        config=config,
        overwrite=args.overwrite,
    )

    if "error" in outputs:
        sys.exit(1)


if __name__ == "__main__":
    main()
