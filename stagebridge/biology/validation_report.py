"""Publication-ready biological validation reports.

Generates comprehensive reports that clearly separate:
1. CONFIRMED: Known biology recovered (builds trust)
2. SUPPORTED: Novel findings with strong evidence
3. HYPOTHESES: Speculative findings requiring validation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stagebridge.biology.known_biology import (
    ValidationResult,
    ValidationStatus,
    compute_mechanism_recovery_score,
)
from stagebridge.biology.novel_discovery import (
    NovelHypothesis,
    DiscoveryResult,
    HypothesisConfidence,
    rank_hypotheses_by_confidence,
    filter_spurious_associations,
)
from stagebridge.biology.lr_scoring import LRScoreResult


@dataclass
class BiologyValidationReport:
    """Comprehensive biological validation report.

    This report is designed to be publication-ready with clear
    separation of confirmed, supported, and speculative findings.
    """
    model_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Known mechanism recovery
    known_validation: list[ValidationResult] = field(default_factory=list)
    recovery_score: dict[str, Any] = field(default_factory=dict)

    # L-R scoring results
    lr_scores: list[LRScoreResult] = field(default_factory=list)
    stage_specific_lr: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Novel discoveries
    discovery_result: DiscoveryResult | None = None

    # Summary statistics
    n_confirmed: int = 0
    n_partial: int = 0
    n_not_detected: int = 0
    n_novel_high_confidence: int = 0
    n_novel_speculative: int = 0

    # Overall assessment
    biological_validity_score: float = 0.0
    discovery_potential_score: float = 0.0
    publication_readiness: str = "not_ready"

    def compute_summary(self) -> None:
        """Compute summary statistics."""
        self.n_confirmed = sum(
            1 for v in self.known_validation
            if v.status == ValidationStatus.CONFIRMED
        )
        self.n_partial = sum(
            1 for v in self.known_validation
            if v.status == ValidationStatus.PARTIAL
        )
        self.n_not_detected = sum(
            1 for v in self.known_validation
            if v.status == ValidationStatus.NOT_DETECTED
        )

        if self.discovery_result:
            high_conf = filter_spurious_associations(
                self.discovery_result.hypotheses,
                min_confidence=HypothesisConfidence.MEDIUM,
            )
            self.n_novel_high_confidence = len(high_conf)
            self.n_novel_speculative = (
                len(self.discovery_result.hypotheses) - self.n_novel_high_confidence
            )

        if self.known_validation:
            self.recovery_score = compute_mechanism_recovery_score(self.known_validation)
            self.biological_validity_score = self.recovery_score.get("overall", 0.0)

        if self.discovery_result and self.discovery_result.hypotheses:
            high_evidence = [
                h for h in self.discovery_result.hypotheses
                if h.confidence in (HypothesisConfidence.HIGH, HypothesisConfidence.MEDIUM)
            ]
            self.discovery_potential_score = len(high_evidence) / 10.0

        self._assess_publication_readiness()

    def _assess_publication_readiness(self) -> None:
        """Assess if results are publication-ready."""
        priority_1_confirmed = sum(
            1 for v in self.known_validation
            if v.mechanism.priority == 1 and v.status == ValidationStatus.CONFIRMED
        )
        priority_1_total = sum(
            1 for v in self.known_validation
            if v.mechanism.priority == 1
        )

        if priority_1_total > 0 and priority_1_confirmed == priority_1_total:
            if self.biological_validity_score > 0.7:
                self.publication_readiness = "ready"
            elif self.biological_validity_score > 0.5:
                self.publication_readiness = "needs_minor_revision"
            else:
                self.publication_readiness = "needs_major_revision"
        elif priority_1_confirmed > 0:
            self.publication_readiness = "needs_major_revision"
        else:
            self.publication_readiness = "not_ready"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "summary": {
                "biological_validity_score": self.biological_validity_score,
                "discovery_potential_score": self.discovery_potential_score,
                "publication_readiness": self.publication_readiness,
                "n_confirmed": self.n_confirmed,
                "n_partial": self.n_partial,
                "n_not_detected": self.n_not_detected,
                "n_novel_high_confidence": self.n_novel_high_confidence,
                "n_novel_speculative": self.n_novel_speculative,
            },
            "recovery_score": self.recovery_score,
            "known_validation": [
                {
                    "mechanism": v.mechanism.name,
                    "status": v.status.value,
                    "score": v.score,
                    "expected_stage": v.mechanism.expected_stage,
                    "observed_stage": v.observed_stage,
                    "explanation": v.explanation,
                }
                for v in self.known_validation
            ],
            "novel_hypotheses": (
                [h.to_dict() for h in rank_hypotheses_by_confidence(
                    self.discovery_result.hypotheses
                )]
                if self.discovery_result else []
            ),
            "DISCLAIMER": (
                "Novel hypotheses are model-generated and require experimental validation. "
                "Only confirmed mechanisms have literature support."
            ),
        }


def generate_validation_report(
    model_name: str,
    known_validation: list[ValidationResult],
    lr_scores: list[LRScoreResult] | None = None,
    stage_specific_lr: pd.DataFrame | None = None,
    discovery_result: DiscoveryResult | None = None,
) -> BiologyValidationReport:
    """Generate comprehensive biological validation report.

    Args:
        model_name: Name/version of the model
        known_validation: Results from validate_known_mechanisms
        lr_scores: Optional L-R scoring results
        stage_specific_lr: Optional stage-specific L-R table
        discovery_result: Optional novel discovery results

    Returns:
        BiologyValidationReport with all summaries computed
    """
    report = BiologyValidationReport(
        model_name=model_name,
        known_validation=known_validation,
        lr_scores=lr_scores or [],
        stage_specific_lr=stage_specific_lr if stage_specific_lr is not None else pd.DataFrame(),
        discovery_result=discovery_result,
    )
    report.compute_summary()
    return report


def export_for_publication(
    report: BiologyValidationReport,
    output_dir: str | Path,
    include_speculative: bool = False,
) -> dict[str, Path]:
    """Export report in publication-ready formats.

    Args:
        report: Validation report
        output_dir: Output directory
        include_speculative: Include speculative hypotheses

    Returns:
        Dict mapping output type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    outputs["summary"] = summary_path

    known_records = [
        {
            "mechanism": v.mechanism.name,
            "type": v.mechanism.mechanism_type.value,
            "priority": v.mechanism.priority,
            "status": v.status.value,
            "score": v.score,
            "expected_stage": v.mechanism.expected_stage,
            "observed_stage": v.observed_stage,
            "literature": v.mechanism.literature_source,
        }
        for v in report.known_validation
    ]
    known_df = pd.DataFrame(known_records)
    known_path = output_dir / "known_mechanism_recovery.csv"
    known_df.to_csv(known_path, index=False)
    outputs["known_mechanisms"] = known_path

    if report.discovery_result and report.discovery_result.hypotheses:
        hypotheses = report.discovery_result.hypotheses
        if not include_speculative:
            hypotheses = filter_spurious_associations(
                hypotheses,
                min_confidence=HypothesisConfidence.LOW,
            )

        hyp_records = [h.to_dict() for h in rank_hypotheses_by_confidence(hypotheses)]
        hyp_df = pd.DataFrame(hyp_records)
        hyp_path = output_dir / "novel_hypotheses.csv"
        hyp_df.to_csv(hyp_path, index=False)
        outputs["novel_hypotheses"] = hyp_path

    if report.lr_scores:
        lr_records = [
            {
                "lr_pair": s.pair.name,
                "ligand": s.pair.ligand,
                "receptor": s.pair.receptor,
                "family": s.pair.family,
                "raw_score": s.raw_score,
                "attention_weight": s.attention_weight,
                "weighted_score": s.weighted_score,
                "confidence": s.confidence,
            }
            for s in sorted(report.lr_scores, key=lambda x: x.weighted_score, reverse=True)
        ]
        lr_df = pd.DataFrame(lr_records)
        lr_path = output_dir / "lr_interaction_scores.csv"
        lr_df.to_csv(lr_path, index=False)
        outputs["lr_scores"] = lr_path

    if not report.stage_specific_lr.empty:
        stage_lr_path = output_dir / "stage_specific_lr.csv"
        report.stage_specific_lr.to_csv(stage_lr_path, index=False)
        outputs["stage_specific_lr"] = stage_lr_path

    readme_content = f"""# Biological Validation Report

Model: {report.model_name}
Generated: {report.timestamp}

## Summary

- Biological Validity Score: {report.biological_validity_score:.2f}
- Discovery Potential Score: {report.discovery_potential_score:.2f}
- Publication Readiness: {report.publication_readiness}

## Known Mechanism Recovery

- Confirmed: {report.n_confirmed}
- Partial: {report.n_partial}
- Not Detected: {report.n_not_detected}

## Novel Discoveries

- High/Medium Confidence: {report.n_novel_high_confidence}
- Speculative: {report.n_novel_speculative}

## IMPORTANT DISCLAIMER

Novel hypotheses are MODEL-GENERATED and require experimental validation.
Only "confirmed" mechanisms have been validated against published literature.

Do NOT report novel hypotheses as findings without explicit labeling as hypotheses.
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    outputs["readme"] = readme_path

    return outputs
