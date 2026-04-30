# Clinical Translation Limitations

This document describes the limitations of the StageBridge genomic interpretation layer for clinical translation. **This module provides translational interpretation, not clinical validation.**

## Core Scientific Limitations

### 1. No Clinical Outcomes Data

StageBridge genomic interpretation does **not** have access to:
- Patient survival data
- Recurrence events
- Treatment response
- Longitudinal lesion tracking
- Disease-free survival

**Implication**: We cannot claim that genomic features enriched in transition zones predict patient outcomes. This requires endpoint-linked validation in prospective cohorts.

### 2. WES as Mutation Truth Set

All germline and somatic mutation calls come from **whole-exome sequencing (WES)**. Spatial transcriptomics and snRNA-seq are used **only** to:
- Localize expressed variant evidence
- Identify clone-associated transcriptional programs
- Provide supportive (not definitive) evidence

**What we CANNOT do:**
- Call new germline mutations from spatial data
- Call new somatic mutations from snRNA-seq
- Definitively exclude mutations based on RNA evidence

### 3. RNA-Based Variant Evidence is Sparse

Expression-based mutation detection has fundamental limitations:
- Not all genes are expressed in all cells
- Expression levels vary dramatically
- Allele-specific expression affects detection
- Sequencing depth is limited per cell
- Many mutations are in non-expressed regions

**Critical caveat**: Absence of alternate reads in spatial/snRNA data **should NOT be interpreted as absence of the underlying DNA mutation**.

### 4. Clonality Estimation Uncertainty

When tumor purity and copy number data are unavailable, clonality is estimated using **naive VAF thresholds**. This is:
- Approximate only
- Not corrected for tumor purity
- Not corrected for local copy number changes
- Subject to sampling bias

**Proper clonality calls require**:
- Matched normal WES
- Tumor purity estimation
- Copy number profiling
- Ploidy estimation

### 5. ACMG-Aligned vs Clinical ACMG

The germline pathogenicity classifications in this module are **ACMG/AMP-aligned**, meaning:
- Based on ClinVar significance when available
- Automated interpretation
- NOT clinical-grade ACMG adjudication
- NOT suitable for clinical decision-making without review

**Clinical use requires**:
- Review by clinical geneticist
- Family history assessment
- Functional studies for VUS
- Multi-disciplinary tumor board review

### 6. OncoKB Annotations are Informational

OncoKB actionability levels indicate:
- Evidence for therapeutic relevance
- NOT guaranteed treatment response
- NOT personalized treatment recommendations

**Clinical treatment decisions require**:
- Oncologist evaluation
- Full genomic context
- Patient-specific factors
- Drug availability and eligibility

## What StageBridge CAN and CANNOT Claim

### CAN Claim
- "Transition zones show enrichment for variants in driver gene X"
- "High-transition regions have higher clonal VAF on average"
- "Actionable variants are more frequent in progression-associated niches"
- "Expressed variant evidence is detectable in spatial data for WES-confirmed mutations"

### CANNOT Claim
- "Transition zones predict patient survival"
- "Spatial transcriptomics detected a new mutation"
- "This patient should receive treatment X"
- "This variant is definitively pathogenic"
- "Absence of RNA reads means the mutation is absent"

## Path to Clinical Validation

To move from translational interpretation to clinical validation:

1. **Longitudinal cohort** - Track lesions from pre-cancer through progression
2. **Outcome data** - Link transition scores to survival, recurrence, treatment response
3. **Independent validation** - Replicate findings in external cohorts
4. **Prospective testing** - Test predictions before outcomes are known
5. **Regulatory review** - Formal validation for clinical use

## Language Guidelines

### Use These Terms
- Translational interpretation layer
- Genomic actionability annotation
- ACMG/AMP-aligned germline pathogenicity
- OncoKB-informed somatic actionability
- Expressed variant evidence
- Clone-state proxy
- Post-hoc validation
- Transition-field enrichment

### Avoid These Terms
- Clinical-grade mutation calling
- Spatial transcriptomics mutation discovery
- Definitive ACMG classification
- Clinically validated risk score
- Proves clinical utility
- Predicts patient outcome (unless actual endpoint data are provided)
- Diagnostic test result

## Summary

The StageBridge genomic interpretation layer is a **research tool** for understanding the genomic landscape of progression-associated niches. It provides:
- Systematic annotation of WES variants
- Integration with spatial transcriptomics
- Statistical testing of enrichment

It does **NOT** provide:
- Clinical mutation calling
- Validated prognostic scores
- Treatment recommendations
- Diagnostic results

**All findings require validation before clinical application.**
