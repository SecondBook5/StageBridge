#!/usr/bin/env python
"""Run biological validation pipeline for StageBridge.

This pipeline:
1. Loads a trained model and extracts attention patterns
2. Validates recovery of known biological mechanisms
3. Computes attention-weighted L-R interaction scores
4. Generates novel hypotheses from unexplained patterns
5. Produces publication-ready validation report

Usage:
    python -m stagebridge.pipelines.run_biological_validation \\
        --checkpoint results/training/best_model.pt \\
        --data-dir /data/processed/luad_evo/canonical \\
        --output-dir results/biological_validation

    # For testing with synthetic data:
    python -m stagebridge.pipelines.run_biological_validation \\
        --synthetic --output-dir results/test_validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stagebridge.biology.lr_scoring import (
    compute_attention_weighted_lr_scores,
    aggregate_lr_scores_by_stage,
    identify_stage_specific_interactions,
)
from stagebridge.biology.known_biology import (
    validate_known_mechanisms,
    compute_mechanism_recovery_score,
)
from stagebridge.biology.novel_discovery import (
    generate_novel_hypotheses,
    rank_hypotheses_by_confidence,
)
from stagebridge.biology.validation_report import (
    generate_validation_report,
    export_for_publication,
)


def generate_synthetic_data(
    n_cells: int = 1000,
    n_genes: int = 500,
    n_stages: int = 3,
) -> dict:
    """Generate synthetic data for testing the pipeline."""
    np.random.seed(42)

    stages = np.random.choice(
        ["Normal", "Preinvasive", "Invasive"],
        size=n_cells,
        p=[0.3, 0.4, 0.3],
    )

    cell_types = np.random.choice(
        ["Epithelial", "Macrophage", "CAF", "T_cell", "Endothelial"],
        size=n_cells,
        p=[0.4, 0.2, 0.15, 0.15, 0.1],
    )

    known_genes = [
        "IL1B", "IL1R1", "IL6", "IL6R", "CXCL12", "CXCR4",
        "EGF", "EGFR", "TGFB1", "TGFBR1", "ACTA2", "COL1A1",
        "VIM", "CDH1", "CDH2", "KRT5", "KRT17", "SOX9",
    ]
    other_genes = [f"GENE_{i}" for i in range(n_genes - len(known_genes))]
    gene_names = known_genes + other_genes

    expression = np.random.exponential(0.5, size=(n_cells, len(gene_names)))

    for i, gene in enumerate(gene_names[:len(known_genes)]):
        if gene in ["IL1B", "IL1R1", "IL6", "CXCL12"]:
            preinvasive_mask = stages == "Preinvasive"
            expression[preinvasive_mask, i] *= 3.0
        elif gene in ["ACTA2", "COL1A1", "VIM", "CDH2"]:
            invasive_mask = stages == "Invasive"
            expression[invasive_mask, i] *= 2.5
        elif gene == "CDH1":
            invasive_mask = stages == "Invasive"
            expression[invasive_mask, i] *= 0.3

    attention = np.random.beta(2, 5, size=(n_cells, 8))

    for i in range(n_cells):
        if stages[i] == "Preinvasive":
            attention[i] *= 1.5
        if cell_types[i] == "Macrophage":
            attention[i] *= 1.3

    attention = attention / attention.sum(axis=1, keepdims=True)

    return {
        "expression": expression,
        "gene_names": gene_names,
        "attention": attention,
        "stages": stages,
        "cell_types": cell_types,
        "sample_ids": [f"sample_{i % 10}" for i in range(n_cells)],
    }


def load_model_outputs(
    checkpoint_path: Path,
    data_dir: Path,
) -> dict:
    """Load trained model and extract attention/expression data."""
    raise NotImplementedError(
        "Real data loading not yet implemented. Use --synthetic for testing."
    )


def run_validation_pipeline(
    expression: np.ndarray,
    gene_names: list[str],
    attention: np.ndarray,
    stages: np.ndarray,
    cell_types: np.ndarray,
    sample_ids: list[str],
    output_dir: Path,
    model_name: str = "StageBridge",
) -> dict:
    """Run full biological validation pipeline.

    Args:
        expression: [N_cells, N_genes] expression matrix
        gene_names: Gene names
        attention: [N_cells, K_neighbors] attention weights
        stages: Stage labels per cell
        cell_types: Cell type labels per cell
        sample_ids: Sample IDs per cell
        output_dir: Output directory
        model_name: Model name for report

    Returns:
        Dict with validation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Validating known mechanisms...")

    unique_stages = np.unique(stages)
    attention_by_stage = {}
    expression_by_stage = {}

    for stage in unique_stages:
        mask = stages == stage
        attention_by_stage[stage] = attention[mask]
        expression_by_stage[stage] = expression[mask]

    known_results = validate_known_mechanisms(
        attention_by_stage=attention_by_stage,
        expression_by_stage=expression_by_stage,
        gene_names=gene_names,
    )

    recovery = compute_mechanism_recovery_score(known_results)

    print(f"  Known mechanism recovery: {recovery['overall']:.2f}")
    print(f"  Confirmed: {recovery['n_confirmed']}/{recovery['n_total']}")

    print("\nStep 2: Computing L-R interaction scores...")

    lr_scores = compute_attention_weighted_lr_scores(
        attention_weights=attention,
        cell_types=cell_types.tolist(),
        expression_matrix=expression,
        gene_names=gene_names,
    )

    print(f"  Scored {len(lr_scores)} L-R pairs")
    if lr_scores:
        top3 = lr_scores[:3]
        for s in top3:
            print(f"    {s.pair.name}: weighted={s.weighted_score:.3f}")

    print("\nStep 3: Aggregating by stage...")

    scores_by_sample = {}
    for sample_id in set(sample_ids):
        mask = np.array(sample_ids) == sample_id
        sample_lr = compute_attention_weighted_lr_scores(
            attention_weights=attention[mask],
            cell_types=np.array(cell_types)[mask].tolist(),
            expression_matrix=expression[mask],
            gene_names=gene_names,
        )
        scores_by_sample[sample_id] = sample_lr

    sample_stages = {
        sid: stages[np.array(sample_ids) == sid][0]
        for sid in set(sample_ids)
    }

    aggregated = aggregate_lr_scores_by_stage(scores_by_sample, sample_stages)
    stage_specific = identify_stage_specific_interactions(aggregated)

    print(f"  Found {len(stage_specific)} stage-specific interactions")

    print("\nStep 4: Generating novel hypotheses...")

    discovery = generate_novel_hypotheses(
        attention_weights=attention,
        expression_matrix=expression,
        gene_names=gene_names,
        cell_types=cell_types.tolist(),
        stages=stages.tolist(),
    )

    ranked = rank_hypotheses_by_confidence(discovery.hypotheses)
    print(f"  Generated {len(ranked)} hypotheses")
    print(f"  Recovered {discovery.known_recovered} known genes in top attention")

    print("\nStep 5: Generating validation report...")

    report = generate_validation_report(
        model_name=model_name,
        known_validation=known_results,
        lr_scores=lr_scores,
        stage_specific_lr=stage_specific,
        discovery_result=discovery,
    )

    outputs = export_for_publication(report, output_dir)

    print(f"\n{'='*60}")
    print("BIOLOGICAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Biological Validity Score: {report.biological_validity_score:.2f}")
    print(f"Discovery Potential Score: {report.discovery_potential_score:.2f}")
    print(f"Publication Readiness: {report.publication_readiness}")
    print(f"\nKnown Mechanisms:")
    print(f"  - Confirmed: {report.n_confirmed}")
    print(f"  - Partial: {report.n_partial}")
    print(f"  - Not Detected: {report.n_not_detected}")
    print(f"\nNovel Discoveries:")
    print(f"  - High/Medium Confidence: {report.n_novel_high_confidence}")
    print(f"  - Speculative: {report.n_novel_speculative}")
    print(f"\nOutputs written to: {output_dir}")
    for name, path in outputs.items():
        print(f"  - {name}: {path.name}")

    return {
        "report": report,
        "known_results": known_results,
        "lr_scores": lr_scores,
        "stage_specific": stage_specific,
        "discovery": discovery,
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description="Run biological validation pipeline")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for validation results",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="StageBridge",
        help="Model name for report",
    )

    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic data for testing...")
        data = generate_synthetic_data()
    else:
        if not args.checkpoint or not args.data_dir:
            parser.error("--checkpoint and --data-dir required unless --synthetic")
        data = load_model_outputs(args.checkpoint, args.data_dir)

    run_validation_pipeline(
        expression=data["expression"],
        gene_names=data["gene_names"],
        attention=data["attention"],
        stages=data["stages"],
        cell_types=data["cell_types"],
        sample_ids=data["sample_ids"],
        output_dir=args.output_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
