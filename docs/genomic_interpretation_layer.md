# Genomic Interpretation Layer

## Overview

The genomic interpretation layer provides **translational interpretation** of StageBridge transition field analysis by integrating WES-derived germline and somatic variants with spatial transcriptomics evidence.

This is a **post-hoc analysis layer** that tests whether high-transition niches (putative progression zones identified by StageBridge) are enriched for clinically interpretable genomic features.

## Why This Module Exists

StageBridge learns transition fields that identify cells/spots likely to be in active progression between disease stages. The biological question is: **Are these transition zones genomically distinct?**

Specifically:
- Do high-transition regions have more driver mutations?
- Are actionable variants enriched in progression zones?
- Is clonal vs subclonal architecture different in transition zones?
- Can we find expressed variant evidence in spatial data?

## Scientific Framing

### WES is the Mutation Truth Set

**Spatial transcriptomics and snRNA-seq CANNOT reliably call germline or somatic mutations de novo.** These modalities have:
- Sparse RNA coverage
- Expression-dependent detection
- No matched normal comparison
- Limited ability to distinguish germline from somatic

Therefore, **all mutation calls come from WES**. Spatial/snRNA data is used only to:
1. Localize expressed variant evidence
2. Identify clone-associated transcriptional programs
3. Map CNV-like states (via inferCNV/CopyKAT)

### ACMG/AMP-Aligned Germline Interpretation

Germline variants are interpreted using **ACMG/AMP-aligned** classification:

| Category | Description |
|----------|-------------|
| Pathogenic | Strong evidence of disease causation |
| Likely Pathogenic | Moderate evidence |
| VUS | Uncertain significance |
| Likely Benign | Moderate evidence against pathogenicity |
| Benign | Strong evidence against pathogenicity |

**IMPORTANT**: These are automated interpretations based on ClinVar significance, not clinical-grade ACMG adjudications. All pathogenic/likely pathogenic calls require review by a clinical geneticist.

### OncoKB Somatic Actionability

Somatic variants are interpreted using **OncoKB** or similar cancer-specific knowledgebases:

| Level | Description |
|-------|-------------|
| Level 1 | FDA-approved, standard of care |
| Level 2 | Standard of care, compelling evidence |
| Level 3A | Clinical evidence, investigational |
| Level 3B | Clinical evidence, another tumor type |
| Level 4 | Biological evidence |
| R1/R2 | Resistance mutations |

**IMPORTANT**: We do NOT use ACMG as the primary somatic actionability framework. ACMG is for germline pathogenicity; OncoKB is for cancer actionability.

### Clonality Estimation

Clonality is estimated from WES variant allele fractions:

**With purity/CNV data:**
```
CCF = VAF * (purity * local_CN + (1-purity) * 2) / (purity * multiplicity)
```

**Without purity/CNV data (naive VAF):**
- VAF >= 0.30 → clonal_like
- 0.10 <= VAF < 0.30 → intermediate
- 0.02 <= VAF < 0.10 → subclonal_like
- VAF < 0.02 → low_confidence

**IMPORTANT**: Naive VAF clonality is approximate and explicitly labeled as such.

### Transition Enrichment Testing

The core analysis tests whether high-transition niches differ from low-transition regions:

1. Define groups by transition score quantile
2. Test enrichment for genomic features
3. Apply FDR correction

Statistical tests:
- Continuous features: Mann-Whitney U
- Binary features: Fisher exact test
- Categorical features: Chi-square

## Module Structure

```
stagebridge/genomics/
├── __init__.py
├── schemas.py              # Data schemas (VariantRecord, etc.)
├── vcf_io.py               # VCF reading and normalization
├── acmg.py                 # Germline ACMG-aligned annotation
├── somatic_actionability.py # Somatic OncoKB annotation
├── clonality.py            # Clonality estimation
├── spatial_variant_evidence.py # RNA-based variant evidence
├── cnv_proxy.py            # Expression-derived CNV proxy
├── transition_enrichment.py # Enrichment testing
├── reports.py              # Report generation
└── variant_annotation.py   # Combined annotation pipeline
```

## Usage

### Command Line

```bash
python scripts/run_genomic_interpretation.py \
    --somatic-vcf data/variants/somatic.vcf \
    --germline-vcf data/variants/germline.vcf \
    --transition-scores results/transition_scores.parquet \
    --oncokb-table data/annotations/oncokb.tsv \
    --clinvar-table data/annotations/clinvar.tsv \
    --output-dir results/genomics
```

### Configuration

See `configs/genomic_interpretation.yaml` for all options.

### Python API

```python
from stagebridge.genomics.variant_annotation import build_variant_master_table, annotate_all_variants
from stagebridge.genomics.clonality import estimate_clonality_for_variants
from stagebridge.genomics.transition_enrichment import generate_transition_genomic_enrichment_table

# Build variant table
variant_df = build_variant_master_table(
    somatic_vcf="path/to/somatic.vcf",
    germline_vcf="path/to/germline.vcf",
)

# Annotate
germline_df, somatic_df = annotate_all_variants(
    variant_df,
    clinvar_df=clinvar_df,
    oncokb_df=oncokb_df,
)

# Test enrichment
enrichment_df = generate_transition_genomic_enrichment_table(
    transition_df=transition_scores,
    actionability_df=somatic_df,
    clonality_df=clonality_df,
)
```

## Outputs

| File | Description |
|------|-------------|
| variant_master_table.parquet | All variants with normalized IDs |
| germline_acmg_aligned_annotations.parquet | Germline pathogenicity |
| somatic_actionability_annotations.parquet | Somatic actionability |
| clonality_estimates.parquet | CCF/VAF-based clonality |
| spatial_variant_evidence.parquet | RNA-based variant evidence |
| transition_genomic_enrichment.parquet | Enrichment test results |
| genomic_interpretation_report.md | Human-readable report |

## Key Limitations

See [Clinical Translation Limitations](clinical_translation_limitations.md) for detailed caveats.

1. **No clinical outcomes** - This is translational interpretation, not clinical validation
2. **WES truth set** - Mutations are NOT called from spatial/snRNA data
3. **RNA evidence is sparse** - Absence of reads ≠ absence of mutation
4. **Naive VAF clonality** - Approximate without purity/CNV correction
5. **Automated ACMG** - Requires clinical genetics review
