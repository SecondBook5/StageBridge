# EA-MIST Label Provenance and Method Narrative

## Purpose
This note makes the active EA-MIST supervision geometry explicit and auditable.

The previous receiver-level communication benchmark failed because it used a **lesion-level proxy label** as if it were a **local neighborhood label**. EA-MIST fixes that mismatch by:

- treating each lesion as the supervised unit
- treating local neighborhoods as instances inside the lesion bag
- using a lesion-level Set Transformer as the central model
- injecting evolutionary context at the lesion level only

## What Changed
The old local communication-relay benchmark asked one receiver-centered niche to predict a lesion-level progression proxy. That created three avoidable problems:

1. The supervision lived at the wrong scale.
2. The model was rewarded for memorizing lesion-level signal with local tokens.
3. Rich communication structure was forced onto a tiny weak-label task.

EA-MIST changes the geometry:

- `sample_id` is the default `lesion_id`
- a lesion is represented as a bag of compact local niche embeddings
- a prototype bottleneck compresses recurrent local motifs
- lesion labels supervise only the lesion bag
- WES/evolution features are fused only after lesion summarization

## Curated Label Provenance
The curated label manifest is stored in [curated_progression_labels.csv](/home/ajbook/projects/StageBridge/stagebridge/data/luad_evo/curated_progression_labels.csv).

The resolved provenance table for the active spatial cohort is stored in [table0_curated_label_provenance.csv](/home/ajbook/projects/StageBridge/reports/tables/eamist/table0_curated_label_provenance.csv).

Curated labels come from Peng lesion evolution patterns:

- `pattern1a` and `pattern1b` are treated as `progression-competent = 1`
- `pattern2` is treated as `progression-competent = 0`

The `label_source` column records the figure path used during curation:

- `peng_fig_2c_pattern1a`
- `peng_fig_2c_pattern2`
- `peng_fig_5e_pattern1b`
- `peng_fig_s3_pattern1a`
- `peng_fig_s3_pattern1b`
- `peng_fig_s3_pattern2`

## ID Resolution Rule
The paper-derived curated IDs and the active spatial cohort use different GSM prefixes:

- curated manifest: `GSM923...`
- active spatial cohort: `GSM922...`

EA-MIST resolves this with canonical lesion-key normalization in [neighborhood_builder.py](/home/ajbook/projects/StageBridge/stagebridge/data/luad_evo/neighborhood_builder.py):

- strip the GSM prefix
- preserve donor and stage identity
- normalize replicate suffixes like `AAH1` -> `AAH-1` and `AIS1` -> `AIS-1`

Result:

- curated rows: `21`
- canonically resolved to spatial lesions: `21`
- unresolved curated rows: `0`

All curated rows currently resolve by exact canonical key; donor-stage fallback is present in code as a defensive backup but was not needed for the active table.

## Final Lesion Label Table Used by EA-MIST
The lesion-level training table is built by:

1. resolving curated labels onto spatial lesion IDs
2. adding heuristic-expanded `AAH->AIS` labels only where no curated label exists

Current active lesion table:

- total lesion bags: `24`
- `AAH->AIS`: `11` lesions
- `AIS->MIA`: `13` lesions

Class balance:

- `AAH->AIS`: `10` positive, `1` negative
- `AIS->MIA`: `9` positive, `4` negative

Label confidence weights:

- curated lesion labels: `1.0`
- heuristic-expanded `AAH->AIS` labels: `0.5`

The heuristic-expanded `AAH->AIS` lesions currently are:

- `GSM9226168_P1_AAH`
- `GSM9226170_P2_AAH`
- `GSM9226214_P22_AAH`

These are included to keep `AAH->AIS` active without pretending they have the same evidentiary strength as curated Peng pattern assignments.

## Why AIS->MIA Is the Main Acceptance Edge
EA-MIST keeps both edges active, but `AIS->MIA` is the primary acceptance edge because:

- it has the strongest existing transformer signal in the repo
- it has the cleaner curated label support
- it is biologically closer to the invasion bottleneck
- `AAH->AIS` remains heavily imbalanced even after expansion

So the honest success criterion is:

- a clear lesion-level gain on `AIS->MIA`
- interpretable stable prototype motifs
- mixed or limited results on `AAH->AIS` are acceptable

## Method Narrative for the Paper and Notebook
The paper/story should frame EA-MIST as a **supervision-geometry correction**:

> We replaced receiver-level weak-label prediction with lesion-level multiple-instance learning. Local spatial niches are encoded as compact instances, compressed through a prototype bottleneck, and aggregated by a lesion-level Set Transformer conditioned on lesion-level evolutionary context.

That is the central methodological claim, not that the model reconstructs raw communication grammar from sparse lesion labels.

## Current Run Alignment
The active full workflow uses this exact label resolution path and lesion table construction:

- live run: `eamist_full_20260309_fixed`
- workflow log: [workflow.log](/home/ajbook/projects/StageBridge/outputs/scratch/eamist_full_20260309_fixed/workflow.log)

This note is therefore aligned to the live run and should be treated as part of the active benchmark record.
