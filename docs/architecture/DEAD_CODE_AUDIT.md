# Dead/Duplicate Code Audit

Generated: 2026-04-13

## Summary

The codebase has architecture sprawl from multiple iterations. This document flags code that appears unused or duplicated.

## Actually Used (V1 Pipeline)

### Main Model
- `stagebridge/pipelines/run_v1_complete.py:StageBridgeV1Complete` - **THE MODEL**
- `stagebridge/context_model/receiver_niche_encoder.py:ReceiverCenteredNicheEncoder` - Used by V1Complete

### Baselines (Snakemake)
- `stagebridge/baselines/evaluate.py:PoolingMLPBaseline` - baseline ladder
- `stagebridge/baselines/evaluate.py:DeepSetsBaseline` - baseline ladder
- `stagebridge/baselines/evaluate.py:SetTransformerBaseline` - baseline ladder
- `stagebridge/baselines/evaluate.py:GraphSAGEBaseline` - baseline ladder

---

## DEAD CODE - Unused Models

### EA-MIST (Evolution-Aware Multiple-Instance Set Transformer)
**File:** `stagebridge/context_model/lesion_set_transformer.py`
**Classes:**
- `EAMISTModel` - Complex hierarchical model, NOT USED in training pipeline
- `LesionSetTransformerBackbone` - Part of EA-MIST
- `NicheTransitionScoreHead` - Part of EA-MIST

**Why dead:** `run_v1_complete.py` instantiates `StageBridgeV1Complete`, not `EAMISTModel`.

**Decision needed:** Delete or keep for future experiments?

---

### Lesion-Level Baselines
**File:** `stagebridge/context_model/baselines_lesion.py`
**Classes:**
- `PooledLesionBaseline`
- `DeepSetsLesionBaseline`
- `LesionSetTransformerBaseline`

**Why dead:** Snakemake baselines use `baselines/evaluate.py` classes, not these.

**Decision needed:** Were these for a different evaluation? Delete?

---

### Hierarchical Transformer
**File:** `stagebridge/context_model/hierarchical_transformer.py`
**Classes:**
- `TypedHierarchicalTransformerEncoder`
- `_GroupSetEncoder`

**Why dead:** Not imported anywhere in training pipeline.

---

### Graph of Sets
**File:** `stagebridge/context_model/graph_of_sets.py`
**Classes:**
- Various graph-based encoders

**Why dead:** Not used in V1 pipeline.

---

### Communication Models
**File:** `stagebridge/context_model/communication_relay.py`
**Classes:**
- `StageBridgeCommunicationModel`
- `PooledNeighborhoodModel`
- `DeepSetsCommunicationModel`
- `CommunicationEncoder`
- `RelayEncoder`

**Why dead:** Appear to be from communication/L-R modeling iteration, not used in V1.

---

### Local Niche Encoders (non-receiver-centered)
**File:** `stagebridge/context_model/local_niche_encoder.py`
**Classes:**
- `LocalNicheTransformerEncoder`
- `LocalNicheMLPEncoder`

**Why dead:** EA-MIST used these, but EA-MIST is dead.

---

### Set Encoders
**File:** `stagebridge/context_model/set_encoder.py`
**Classes:**
- `DeepSetsContextEncoder`
- `TypedSetContextEncoder`
- `DeepSetsTransformerHybridEncoder`
- `PooledContextEncoder`

**Why dead:** Old architecture iterations.

---

## DUPLICATE CODE - Same Class Multiple Files

### GraphSAGEBaseline (3 copies!)
1. `stagebridge/baselines/graph_sage.py:54` 
2. `stagebridge/baselines/evaluate.py:193`
3. `stagebridge/pipelines/run_v1_complete.py:753`

**Action:** Consolidate to one location, import elsewhere.

### DeepSetsBaseline (2 copies)
1. `stagebridge/baselines/evaluate.py:151`
2. `stagebridge/pipelines/run_v1_complete.py:680`

**Action:** Consolidate.

### SetTransformerBaseline (2 copies)
1. `stagebridge/baselines/evaluate.py:172`
2. `stagebridge/pipelines/run_v1_complete.py:696`

**Action:** Consolidate.

---

## Transition Model Baselines (Unclear Status)

**File:** `stagebridge/transition_model/baselines.py`
**Classes:**
- `DeepSetsEncoder`
- `DeepSetsFlowModel`
- `NoContextFlowModel`
- `LinearTransitionBaseline`

**Status:** May be used for ablations? Check if imported anywhere.

---

## Recommended Actions

1. **Immediate:** Delete or move to `archive/` folder:
   - `lesion_set_transformer.py` (EA-MIST)
   - `baselines_lesion.py`
   - `hierarchical_transformer.py`
   - `graph_of_sets.py`
   - `communication_relay.py`
   - `local_niche_encoder.py`
   - `set_encoder.py` (keep SAB/ISAB/PMA if used elsewhere)

2. **Consolidate duplicates:**
   - Move baseline classes to `stagebridge/baselines/models.py`
   - Import from there in evaluate.py and run_v1_complete.py

3. **Verify transition baselines:**
   - Check if `transition_model/baselines.py` is used in ablations
   - If not, delete

---

## Architecture Decision Needed

**Current V1 model (`StageBridgeV1Complete`) is simpler than EA-MIST:**

| Feature | EA-MIST | V1Complete |
|---------|---------|------------|
| Spatial rings | Yes (4 rings) | No |
| Prototype bottleneck | Yes (16 prototypes) | No |
| Evolution branch | Yes (genomic gating) | Just WES projection |
| Hierarchical pooling | Yes (cell→lesion) | No |
| Set transformer backbone | ISAB with inducing points | No |
| Local encoder | Full LocalNicheTransformer | ReceiverCenteredNicheEncoder |

**Question:** Is V1Complete sufficient for the hypothesis testing, or should we use EA-MIST's richer architecture?
