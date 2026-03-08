# 002 Data Contract

Active v1 data contracts are organized under:

- `stagebridge/data/common/`
- `stagebridge/data/luad_evo/`
- `stagebridge/data/brainmets/`

## LUAD Evolution Dataset (active v1)

| Accession | Modality | Role |
|-----------|----------|------|
| GSE308103 | snRNA-seq | Cell-level transcriptomes across five stages |
| GSE307534 | 10x Visium | Spatial tissue architecture for niche definition |
| GSE307529 | WES | Evolutionary state, transition regularization |

Same patient cohort, shared donor identifiers.

## Brain Metastasis Dataset (reserved, not v1)

GSE223499 (snRNA), GSE223501 (Slide-seq), GSE223500 (TCR), GSE223502 (lpWGS). Not required for v1.

## Required `.obs` Columns

| Column | Type | Description |
|--------|------|-------------|
| `stage` | str | Canonical: `Normal`, `AAH`, `AIS`, `MIA`, `LUAD` |
| `donor_id` | str | Unique, consistent across modalities |
| `sample_id` | str | Unique sample identifier |

## Stage Naming

Canonical names only. Alias mapping at ingestion, not downstream.

## Donor Identity

Consistent across modalities. Required for donor-held-out CV, cross-modal alignment, WES lookup.

## WES Schema

Donor-level parquet: `donor_id`, mutation burden, driver mutations (KRAS, EGFR, STK11), CNV summary.

## Spatial Schema

Spot coordinates in `.obsm['spatial']`, `sample_id` and `stage` in `.obs`. After mapping, cell-type composition scores added.

## v1 Minimum

1. snRNA-seq AnnData with all five stages
2. Spatial AnnData with matching samples
3. HLCA reference atlas
4. WES (required for full mode, optional for RNA-only and set-only)
