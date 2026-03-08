# Run Provenance

## Runs used in figures and tables

All results come from the results registry at `results/registry/results_registry.csv`.

### Core Mode Comparison (Figure 2, Table 1)

| Run | Registry Row | Timestamp | Git Hash | Edge | Mode | Metric |
|-----|-------------|-----------|----------|------|------|--------|
| mission3_rna_only_smoke_aah_to_ais | 18 | 2026-03-08T01:52 | 20e0d11 | AAH->AIS | rna_only | 17.252 |
| mission3_set_only_aah_to_ais | 19 | 2026-03-08T01:55 | 20e0d11 | AAH->AIS | set_only | 17.817 |
| mission3_pooled_aah_to_ais | 20 | 2026-03-08T02:01 | 20e0d11 | AAH->AIS | pooled | 18.097 |
| mission3_graph_of_sets_aah_to_ais | 21 | 2026-03-08T02:03 | 20e0d11 | AAH->AIS | graph_of_sets | 18.683 |
| mission3_set_only_ais_to_mia | 23 | 2026-03-08T02:20 | 20e0d11 | AIS->MIA | set_only | 15.758 |
| mission3_pooled_ais_to_mia | 24 | 2026-03-08T02:20 | 20e0d11 | AIS->MIA | pooled | 15.909 |
| mission3_graph_of_sets_ais_to_mia | 25 | 2026-03-08T02:20 | 20e0d11 | AIS->MIA | graph_of_sets | 16.002 |
| mission3_rna_only_ais_to_mia | 27 | 2026-03-08T02:26 | 20e0d11 | AIS->MIA | rna_only | 16.297 |

### Context Ablation (Figure 3)

Derived from the same 8 runs listed above.

### Extension Status (Figure 4)

Derived from gate results:
- WES gate (rows 31-32): AAH->AIS WES=27.33, AIS->MIA WES=15.70
- Diffusion gate (rows 33-34): AAH->AIS diffusion=26.04, AIS->MIA diffusion=16.21

## Source artifacts

All runs executed on commit `20e0d11` (main branch), using the active StageBridge v1 codebase. Data from `$STAGEBRIDGE_DATA_ROOT` (GSE308103 snRNA-seq, GSE307534 Visium).

## Reuse policy

All results were reused as-is from the registry. No results were regenerated for this course package.

## New runs

**No new runs were executed** for the course package figures and tables. All artifacts derive from previously recorded runs in the results registry.
