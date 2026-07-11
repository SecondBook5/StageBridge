# LUAD Ontology Mapping

Source-backed mapping from verified LUAD labels to canonical CCRT grammar ids.
Biological system id: **`luad_premalignant_progression`**. Every mapping cites
source evidence; unknown labels fail in strict mode (never silently bucketed).

> Accuracy caveat: canonical ids are structural mappings of the *observed* source
> labels in this local copy (see `SOURCE_AUDIT.md`); they do not assert that the
> underlying data are biologically final.

## Receiver states (epithelial progression, verified in GSM filenames)

| Source label | Canonical CCRT id | Role | Included | Evidence |
|---|---|---|---|---|
| `Normal` | `normal` | receiver_state | yes | GSM filenames `*_Normal`, `stage` col |
| `AAH` | `aah` | receiver_state | yes | GSM `*_AAH` |
| `AIS` | `ais` | receiver_state | yes | GSM `*_AIS` |
| `MIA` | `mia` | receiver_state | yes | GSM `*_MIA` |
| `LUAD` | `invasive_luad` | receiver_state | yes | GSM `*_LUAD` |

## Transition edges (adjacent, cross-sectional)

Ordering by documented LUAD premalignant progression
(normal → AAH → AIS → MIA → invasive). Adjacent edges only.

| Canonical edge id | source | target |
|---|---|---|
| `normal__to__aah` | `normal` | `aah` |
| `aah__to__ais` | `aah` | `ais` |
| `ais__to__mia` | `ais` | `mia` |
| `mia__to__invasive_luad` | `mia` | `invasive_luad` |

Not every pairwise transition is created. Cross-sectional; no longitudinal claim.

## Sender-context types (HLCA niche tokens / Tangram cell types, verified)

| Source label | Canonical CCRT id | Role | Included | Evidence |
|---|---|---|---|---|
| `AT2` | `at2` | sender_context_type | yes | tangram scores + `tok_smooth_AT2` |
| `Basal` | `basal` | sender_context_type | yes | tangram / tokens |
| `Capillary` | `capillary` | sender_context_type | yes | tangram / tokens |
| `Ciliated` | `ciliated` | sender_context_type | yes | tangram / tokens |
| `Fibroblast lineage` | `fibroblast` | sender_context_type | yes | tangram / tokens |
| `Macrophages` | `macrophage` | sender_context_type | yes | tangram / tokens |
| `Mast cells` | `mast_cell` | sender_context_type | yes | tangram / tokens |
| `Secretory` | `secretory` | sender_context_type | yes | tangram / tokens |
| `T cell lineage` | `t_cell` | sender_context_type | yes | tangram / tokens |

No IL1B narrative is hard-coded; the ontology is the verified HLCA/Tangram cell
vocabulary. Unknown labels fail in strict mode.

## Signal programs

No discrete per-observation `signal_program` matrix is exported. A single generic
registered program `luad_context_signal` is declared so the grammar validates.

## Regulatory mediators

**Available** (unlike PanIN): the `lesion_evo_features` WES/evolutionary block is
the regulatory space (`r`). Represented via a single registered regulatory
mediator `luad_evolutionary_state`. Progression-adjacent evo columns
(`evo_progression_risk_score`, `evo_evidence_of_progression_link`) are treated
strictly as source-provided regulatory features and never as CCRT targets.

## Semantic representation

Semantic state (`z_sem`) = HLCA niche features (`niche_hlca_features.parquet`).
Registered in `features.py` / `ADAPTER_CONTRACT.md`. Not a sender-context entry.
