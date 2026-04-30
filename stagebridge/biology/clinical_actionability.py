"""Clinical actionability assessment using ACMG and OncoKB frameworks.

Integrates mutation data with clinical annotation databases to identify:
1. ACMG-classified germline variants (pathogenic, likely pathogenic, VUS)
2. OncoKB-annotated somatic mutations (levels 1-4, resistance)
3. Druggable targets with FDA-approved or investigational therapies

This enables translation from biological discovery to clinical intervention.

References:
- ACMG/AMP 2015: Richards et al., Genet Med 17:405-424
- OncoKB: Chakravarty et al., JCO Precis Oncol 2017:1-16
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

import pandas as pd


class ACMGClassification(StrEnum):
    """ACMG/AMP variant classification (Richards et al. 2015)."""
    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    VUS = "uncertain_significance"
    LIKELY_BENIGN = "likely_benign"
    BENIGN = "benign"


class OncoKBLevel(StrEnum):
    """OncoKB evidence levels for therapeutic actionability."""
    LEVEL_1 = "1"       # FDA-approved, same tumor type
    LEVEL_2 = "2"       # Standard care, same tumor type
    LEVEL_3A = "3A"     # Compelling clinical evidence
    LEVEL_3B = "3B"     # Standard care, different tumor type
    LEVEL_4 = "4"       # Compelling biological evidence
    R1 = "R1"           # Resistance, standard care
    R2 = "R2"           # Resistance, investigational


class MutationType(StrEnum):
    """Mutation type classification."""
    SOMATIC = "somatic"
    GERMLINE = "germline"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClinicalVariant:
    """A clinically annotated variant."""
    gene: str
    variant: str
    mutation_type: MutationType
    acmg_class: ACMGClassification | None = None
    oncokb_level: OncoKBLevel | None = None
    drugs: tuple[str, ...] = ()
    cancer_types: tuple[str, ...] = ()
    pmids: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_actionable(self) -> bool:
        """Check if variant has therapeutic implications."""
        if self.oncokb_level in (OncoKBLevel.LEVEL_1, OncoKBLevel.LEVEL_2,
                                  OncoKBLevel.LEVEL_3A, OncoKBLevel.LEVEL_3B):
            return True
        if self.acmg_class in (ACMGClassification.PATHOGENIC,
                                ACMGClassification.LIKELY_PATHOGENIC):
            return True
        return False

    @property
    def actionability_tier(self) -> int:
        """Return actionability tier (1=highest, 4=lowest, 0=not actionable)."""
        if self.oncokb_level == OncoKBLevel.LEVEL_1:
            return 1
        elif self.oncokb_level == OncoKBLevel.LEVEL_2:
            return 2
        elif self.oncokb_level in (OncoKBLevel.LEVEL_3A, OncoKBLevel.LEVEL_3B):
            return 3
        elif self.oncokb_level == OncoKBLevel.LEVEL_4:
            return 4
        elif self.acmg_class == ACMGClassification.PATHOGENIC:
            return 2
        elif self.acmg_class == ACMGClassification.LIKELY_PATHOGENIC:
            return 3
        return 0


@dataclass
class ActionabilityReport:
    """Clinical actionability report for a sample."""
    sample_id: str
    variants: list[ClinicalVariant] = field(default_factory=list)
    tier1_count: int = 0
    tier2_count: int = 0
    tier3_count: int = 0
    tier4_count: int = 0
    resistance_count: int = 0
    recommended_therapies: list[str] = field(default_factory=list)
    clinical_trials_suggested: bool = False

    def compute_summary(self) -> None:
        """Compute summary statistics."""
        self.tier1_count = sum(1 for v in self.variants if v.actionability_tier == 1)
        self.tier2_count = sum(1 for v in self.variants if v.actionability_tier == 2)
        self.tier3_count = sum(1 for v in self.variants if v.actionability_tier == 3)
        self.tier4_count = sum(1 for v in self.variants if v.actionability_tier == 4)
        self.resistance_count = sum(
            1 for v in self.variants
            if v.oncokb_level in (OncoKBLevel.R1, OncoKBLevel.R2)
        )

        therapies = set()
        for v in self.variants:
            if v.is_actionable:
                therapies.update(v.drugs)
        self.recommended_therapies = sorted(therapies)

        self.clinical_trials_suggested = self.tier3_count > 0 or self.tier4_count > 0


LUNG_CANCER_ONCOKB_DATABASE: dict[str, dict[str, ClinicalVariant]] = {
    "EGFR": {
        "L858R": ClinicalVariant(
            gene="EGFR",
            variant="L858R",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Osimertinib", "Erlotinib", "Gefitinib", "Afatinib"),
            cancer_types=("NSCLC",),
            pmids=("29151359", "28841389"),
            notes="First-line EGFR TKI indicated",
        ),
        "exon19del": ClinicalVariant(
            gene="EGFR",
            variant="exon19del",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Osimertinib", "Erlotinib", "Gefitinib", "Afatinib"),
            cancer_types=("NSCLC",),
            pmids=("29151359",),
            notes="First-line EGFR TKI indicated",
        ),
        "T790M": ClinicalVariant(
            gene="EGFR",
            variant="T790M",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Osimertinib",),
            cancer_types=("NSCLC",),
            pmids=("28841389",),
            notes="Resistance to 1st/2nd gen TKI, osimertinib indicated",
        ),
        "C797S": ClinicalVariant(
            gene="EGFR",
            variant="C797S",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.R1,
            drugs=(),
            cancer_types=("NSCLC",),
            notes="Resistance to osimertinib",
        ),
    },
    "KRAS": {
        "G12C": ClinicalVariant(
            gene="KRAS",
            variant="G12C",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Sotorasib", "Adagrasib"),
            cancer_types=("NSCLC",),
            pmids=("34096690", "35462752"),
            notes="KRAS G12C inhibitor indicated for previously treated NSCLC",
        ),
        "G12D": ClinicalVariant(
            gene="KRAS",
            variant="G12D",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_4,
            drugs=(),
            cancer_types=("NSCLC", "PDAC"),
            notes="G12D inhibitors in development",
        ),
        "G12V": ClinicalVariant(
            gene="KRAS",
            variant="G12V",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_4,
            drugs=(),
            cancer_types=("NSCLC", "CRC"),
            notes="No approved targeted therapy",
        ),
    },
    "ALK": {
        "fusion": ClinicalVariant(
            gene="ALK",
            variant="fusion",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Alectinib", "Brigatinib", "Lorlatinib", "Crizotinib"),
            cancer_types=("NSCLC",),
            pmids=("28586544", "29596029"),
            notes="First-line ALK TKI indicated",
        ),
    },
    "ROS1": {
        "fusion": ClinicalVariant(
            gene="ROS1",
            variant="fusion",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Entrectinib", "Crizotinib"),
            cancer_types=("NSCLC",),
            pmids=("28586544",),
            notes="ROS1 TKI indicated",
        ),
    },
    "BRAF": {
        "V600E": ClinicalVariant(
            gene="BRAF",
            variant="V600E",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Dabrafenib + Trametinib",),
            cancer_types=("NSCLC", "Melanoma"),
            pmids=("28586544",),
            notes="BRAF/MEK inhibitor combination indicated",
        ),
    },
    "MET": {
        "exon14skip": ClinicalVariant(
            gene="MET",
            variant="exon14skip",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Capmatinib", "Tepotinib"),
            cancer_types=("NSCLC",),
            pmids=("32402160",),
            notes="MET exon 14 skipping, MET TKI indicated",
        ),
        "amplification": ClinicalVariant(
            gene="MET",
            variant="amplification",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_2,
            drugs=("Capmatinib", "Tepotinib"),
            cancer_types=("NSCLC",),
            notes="High-level amplification",
        ),
    },
    "RET": {
        "fusion": ClinicalVariant(
            gene="RET",
            variant="fusion",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Selpercatinib", "Pralsetinib"),
            cancer_types=("NSCLC", "Thyroid"),
            pmids=("32846060",),
            notes="RET inhibitor indicated",
        ),
    },
    "NTRK1": {
        "fusion": ClinicalVariant(
            gene="NTRK1",
            variant="fusion",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Larotrectinib", "Entrectinib"),
            cancer_types=("All solid tumors",),
            pmids=("29466156",),
            notes="Tumor-agnostic indication",
        ),
    },
    "TP53": {
        "R175H": ClinicalVariant(
            gene="TP53",
            variant="R175H",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_4,
            drugs=(),
            cancer_types=("Multiple",),
            notes="Loss of function, no approved targeted therapy",
        ),
        "R248W": ClinicalVariant(
            gene="TP53",
            variant="R248W",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_4,
            drugs=(),
            cancer_types=("Multiple",),
            notes="Loss of function, no approved targeted therapy",
        ),
    },
    "STK11": {
        "loss": ClinicalVariant(
            gene="STK11",
            variant="loss",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_4,
            drugs=(),
            cancer_types=("NSCLC",),
            notes="Associated with poor immunotherapy response",
        ),
    },
    "KEAP1": {
        "loss": ClinicalVariant(
            gene="KEAP1",
            variant="loss",
            mutation_type=MutationType.SOMATIC,
            oncokb_level=OncoKBLevel.LEVEL_4,
            drugs=(),
            cancer_types=("NSCLC",),
            notes="Associated with poor immunotherapy response, co-occurs with KRAS/STK11",
        ),
    },
}

GERMLINE_ACMG_DATABASE: dict[str, dict[str, ClinicalVariant]] = {
    "BRCA1": {
        "pathogenic": ClinicalVariant(
            gene="BRCA1",
            variant="pathogenic_variant",
            mutation_type=MutationType.GERMLINE,
            acmg_class=ACMGClassification.PATHOGENIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Olaparib", "Rucaparib", "Niraparib"),
            cancer_types=("Breast", "Ovarian", "Pancreatic", "Prostate"),
            notes="PARP inhibitor indicated for HRD tumors",
        ),
    },
    "BRCA2": {
        "pathogenic": ClinicalVariant(
            gene="BRCA2",
            variant="pathogenic_variant",
            mutation_type=MutationType.GERMLINE,
            acmg_class=ACMGClassification.PATHOGENIC,
            oncokb_level=OncoKBLevel.LEVEL_1,
            drugs=("Olaparib", "Rucaparib", "Niraparib"),
            cancer_types=("Breast", "Ovarian", "Pancreatic", "Prostate"),
            notes="PARP inhibitor indicated for HRD tumors",
        ),
    },
    "ATM": {
        "pathogenic": ClinicalVariant(
            gene="ATM",
            variant="pathogenic_variant",
            mutation_type=MutationType.GERMLINE,
            acmg_class=ACMGClassification.PATHOGENIC,
            oncokb_level=OncoKBLevel.LEVEL_3A,
            drugs=("Olaparib",),
            cancer_types=("Prostate",),
            notes="PARP inhibitor may benefit, clinical trials",
        ),
    },
    "PALB2": {
        "pathogenic": ClinicalVariant(
            gene="PALB2",
            variant="pathogenic_variant",
            mutation_type=MutationType.GERMLINE,
            acmg_class=ACMGClassification.PATHOGENIC,
            oncokb_level=OncoKBLevel.LEVEL_3A,
            drugs=("Olaparib",),
            cancer_types=("Breast", "Pancreatic"),
            notes="PARP inhibitor may benefit",
        ),
    },
    "CHEK2": {
        "pathogenic": ClinicalVariant(
            gene="CHEK2",
            variant="pathogenic_variant",
            mutation_type=MutationType.GERMLINE,
            acmg_class=ACMGClassification.PATHOGENIC,
            drugs=(),
            cancer_types=("Breast",),
            notes="Moderate penetrance, increased surveillance recommended",
        ),
    },
    "TP53": {
        "germline_pathogenic": ClinicalVariant(
            gene="TP53",
            variant="germline_pathogenic",
            mutation_type=MutationType.GERMLINE,
            acmg_class=ACMGClassification.PATHOGENIC,
            drugs=(),
            cancer_types=("Li-Fraumeni syndrome",),
            notes="High penetrance, intensive surveillance protocol",
        ),
    },
}


def annotate_variants(
    mutation_data: pd.DataFrame,
    gene_col: str = "gene",
    variant_col: str = "variant",
    mutation_type_col: str | None = None,
) -> list[ClinicalVariant]:
    """Annotate variants with ACMG/OncoKB classifications.

    Args:
        mutation_data: DataFrame with mutation calls
        gene_col: Column name for gene
        variant_col: Column name for variant
        mutation_type_col: Optional column for somatic/germline

    Returns:
        List of annotated ClinicalVariant objects
    """
    annotated = []

    for _, row in mutation_data.iterrows():
        gene = row[gene_col].upper()
        variant = row[variant_col]

        mut_type = MutationType.UNKNOWN
        if mutation_type_col and mutation_type_col in row:
            mt = row[mutation_type_col].lower()
            if "somatic" in mt:
                mut_type = MutationType.SOMATIC
            elif "germline" in mt:
                mut_type = MutationType.GERMLINE

        annotation = None

        if gene in LUNG_CANCER_ONCOKB_DATABASE:
            gene_db = LUNG_CANCER_ONCOKB_DATABASE[gene]
            if variant in gene_db:
                annotation = gene_db[variant]
            else:
                for var_key, var_annotation in gene_db.items():
                    if var_key.lower() in variant.lower():
                        annotation = var_annotation
                        break

        if annotation is None and gene in GERMLINE_ACMG_DATABASE:
            gene_db = GERMLINE_ACMG_DATABASE[gene]
            if "pathogenic" in gene_db:
                annotation = gene_db["pathogenic"]

        if annotation is not None:
            annotated.append(annotation)
        else:
            annotated.append(ClinicalVariant(
                gene=gene,
                variant=variant,
                mutation_type=mut_type,
            ))

    return annotated


def generate_actionability_report(
    sample_id: str,
    variants: list[ClinicalVariant],
) -> ActionabilityReport:
    """Generate clinical actionability report for a sample.

    Args:
        sample_id: Sample identifier
        variants: List of annotated variants

    Returns:
        ActionabilityReport with recommendations
    """
    report = ActionabilityReport(sample_id=sample_id, variants=variants)
    report.compute_summary()
    return report


def prioritize_therapies(
    report: ActionabilityReport,
    cancer_type: str = "NSCLC",
) -> list[dict]:
    """Prioritize therapies based on actionability tiers.

    Args:
        report: ActionabilityReport
        cancer_type: Cancer type for filtering

    Returns:
        List of therapy recommendations with priority
    """
    recommendations = []

    for variant in sorted(report.variants, key=lambda v: v.actionability_tier):
        if not variant.is_actionable:
            continue

        if variant.cancer_types and cancer_type not in variant.cancer_types:
            if "All solid tumors" not in variant.cancer_types:
                continue

        for drug in variant.drugs:
            recommendations.append({
                "drug": drug,
                "gene": variant.gene,
                "variant": variant.variant,
                "tier": variant.actionability_tier,
                "oncokb_level": variant.oncokb_level.value if variant.oncokb_level else None,
                "acmg_class": variant.acmg_class.value if variant.acmg_class else None,
                "evidence": variant.notes,
            })

    return recommendations


def check_resistance_mutations(
    variants: list[ClinicalVariant],
    current_therapy: str | None = None,
) -> list[ClinicalVariant]:
    """Check for resistance mutations.

    Args:
        variants: List of variants
        current_therapy: Currently prescribed therapy

    Returns:
        List of resistance variants
    """
    resistance = [
        v for v in variants
        if v.oncokb_level in (OncoKBLevel.R1, OncoKBLevel.R2)
    ]

    if current_therapy:
        therapy_lower = current_therapy.lower()
        for v in resistance:
            if any(therapy_lower in d.lower() for d in v.drugs):
                v = ClinicalVariant(
                    gene=v.gene,
                    variant=v.variant,
                    mutation_type=v.mutation_type,
                    oncokb_level=v.oncokb_level,
                    drugs=v.drugs,
                    notes=f"RESISTANCE TO CURRENT THERAPY: {current_therapy}",
                )

    return resistance


def export_actionability_table(
    reports: list[ActionabilityReport],
    output_path: str,
) -> pd.DataFrame:
    """Export actionability reports to table format.

    Args:
        reports: List of ActionabilityReport
        output_path: Output CSV path

    Returns:
        DataFrame with all variants and annotations
    """
    records = []
    for report in reports:
        for v in report.variants:
            records.append({
                "sample_id": report.sample_id,
                "gene": v.gene,
                "variant": v.variant,
                "mutation_type": v.mutation_type.value,
                "oncokb_level": v.oncokb_level.value if v.oncokb_level else "",
                "acmg_class": v.acmg_class.value if v.acmg_class else "",
                "actionability_tier": v.actionability_tier,
                "is_actionable": v.is_actionable,
                "drugs": "; ".join(v.drugs),
                "cancer_types": "; ".join(v.cancer_types),
                "notes": v.notes,
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df
