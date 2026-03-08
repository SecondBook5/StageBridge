# ADR 001 Lung Only V1

Decision: the flagship v1 path targets the within-lung LUAD initiation ladder only.

## Context

Two cohorts available: Peng LUAD evolution (Normal through LUAD, snRNA + spatial + WES) and Rossi brain metastasis (LUAD through brain/chest wall mets, snRNA + Slide-seq + lpWGS + TCR). Modeling the full continuum would mean bridging two independent cohorts with no shared patients, mixing lung and brain tissue contexts, and claiming continuous progression across a dissemination gap.

## Rationale

- All five within-lung stages share one cohort with shared donors
- Donor-held-out validation and cross-modal alignment are possible within one cohort
- The biological question (niche-gated initiation) is tightly scoped and testable
- Brain mets data is retained for future extension

## Consequences

- v1 is scientifically tighter and more defensible
- Cross-cohort bridging is a clearly scoped v2 task
- Paper narrative focuses on niche-gated initiation, not metastatic progression
