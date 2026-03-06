# TopoFlow: Nature Methods Roadmap + HCA Poster Plan

Today: March 4, 2026. HCA Poster: March 31. Nature Methods target: 6–12 months.

---

## What NetFlow Actually Is (Important Correction)

NetFlow is **patient-level**, not cell-level:
- Observations = **patients** (bulk RNA, CNV, multi-omics)
- Features = genes / molecular markers
- Core algorithm: DPT (Diffusion Pseudotime, adapted from Haghverdi 2016) → branch detection → **POSE** topology graph
- Output: NetworkX graph where nodes = patient cohort subgroups, edges = progression directions
- Also: Wasserstein distances between patient groups, Kaplan-Meier clinical outcome integration, Dash interactive visualization

StageBridge is **cell-level**:
- Observations = **cells** (scRNA-seq, spatial)
- Core algorithm: OT flow matching with Set Transformer niche conditioning
- Output: Flow field v_φ(x, t | c_niche) that moves cells between stage distributions

**These operate at different biological scales** — and that is the Nature Methods story.

---

## The Nature Methods Gap

| Method | Scale | What it does | What's missing |
|--------|-------|-------------|----------------|
| NetFlow | Patient | Discovers disease topology (which patient groups branch where) | No cell-level dynamics; no spatial context |
| CellRank / Palantir | Cell | RNA-velocity trajectories | No OT transport; no niche conditioning; assumes continuous |
| Moscot / CellOT | Cell | OT between predefined stage distributions | Assumes linear progression; no topology discovery |
| StageBridge | Cell | OT flow matching between labeled stages | Stage graph pre-specified; no patient-level validation |
| **TopoFlow** | **Multi-scale** | **Patient topology → cell-level flow, with spatial niche conditioning** | **This is the gap** |

---

## The Core Innovation: Multi-Scale Consistency

TopoFlow imposes **consistency across scales**:

1. **Patient level (NetFlow POSE)**: Discover which patients are "transitional" between stages, where branches occur in patient space, and which molecular features drive branching.

2. **Cell level (StageBridge OT-CFM)**: Learn flow fields for cells in transitional patients. Patients near branch points in POSE should have higher per-cell transition entropy (high sigma in SB-CFM).

3. **Cross-scale validation loop**:
   - NetFlow's pairwise Wasserstein distances between patient groups ↔ StageBridge's Sinkhorn distances between cell-stage distributions — **these should agree**.
   - High sigma at SB-CFM branch = high uncertainty = patient is topologically "between" POSE nodes.
   - This cross-scale coherence is testable and falsifiable — exactly what Nature Methods wants.

4. **Clinical grounding**: NetFlow's Kaplan-Meier integration validates that high niche-dependence score from StageBridge correlates with survival outcome. This closes the loop from cell biology to clinical relevance.

---

## Why This Is Nature Methods, Not Just a Preprint

Nature Methods publishes methods that are:
1. **Broadly applicable** — not one disease. TopoFlow works wherever NetFlow works: MM, GBM, BC, LUAD.
2. **Technically novel** — multi-scale OT consistency between patient topology and cell flow is new.
3. **Validated biologically** — Kaplan-Meier outcome validation in ≥3 cancer types.
4. **Open source, reproducible** — unified package with tutorials.
5. **Better than existing alternatives** — measured on standard benchmarks.

The key claim: *"TopoFlow discovers patient-level progression topology and learns spatially-conditioned cell-level flow along that topology, with provable cross-scale Wasserstein consistency."*

---

## Division of Contribution

| Component | Lead |
|-----------|------|
| POSE patient topology (NetFlow) | Rena |
| Wasserstein distance framework (NetFlow) | Rena |
| Kaplan-Meier + clinical stats (NetFlow) | Rena |
| Interactive visualization — Dash/Cytoscape (NetFlow probe) | Rena |
| MM / GBM / BC dataset pipelines | Rena |
| OT flow matching between cell stages (StageBridge) | You |
| Set Transformer niche conditioning (StageBridge) | You |
| SB-CFM stochastic bridge integration (StageBridge) | You |
| HLCA latent space + spatial tokenization | You |
| LUAD biological validation | You |
| Cross-scale Wasserstein consistency math | Joint |
| TopoFlow unified package | Joint |

---

## Track 1: HCA Poster (March 31 — 27 days)

**Scope is locked.** StageBridge alone. No NetFlow integration yet.

**Core claim**: *"We model LUAD stage progression in the HLCA latent space using niche-conditioned OT flow matching, and show early transitions are disproportionately driven by the spatial microenvironment."*

**3 figures**:
1. HLCA UMAP + predicted flow trajectories (visual centrepiece)
2. Benchmark: StageBridge vs DeepSets vs NoContext vs Linear vs SB-CFM (Sinkhorn, MMD, AUC)
3. Context sensitivity heatmap: niche-dependence gradient across stage transitions (the biological finding)

**Implement before March 31 (only these)**:
1. SB-CFM: losses.py + stagebridge.py + baselines.py + configs (~2 hrs)
2. Data migration from psg-orbit (~30 min)

**Schedule**:
- Mar 4–10: SB-CFM + training run (overnight)
- Mar 11–17: Analysis + figure generation
- Mar 18–24: Poster design + abstract
- Mar 25–31: Buffer

---

## Track 2: TopoFlow Paper (6–12 months)

### Phase 1: StageBridge baseline (April–May, course project)
Complete StageBridge with BrainMet extension and cross-dataset evaluation. This becomes StageBridge's contribution chapter in the paper.

### Phase 2: POSE integration (May–July)
- Feed StageBridge's Sinkhorn distances into NetFlow's patient distance matrix
- Run POSE on LUAD patient cohort using StageBridge-derived cell transition distances as patient-level signal
- Validate: do POSE branch points correspond to histologically ambiguous samples (AAH/AIS boundary)?
- Implement: `topoflow.bridge.PatientCellBridge` — maps patient POSE position to cell sigma in SB-CFM

### Phase 3: Cross-scale consistency formalism (June–August)
- Prove (or empirically show): pairwise Wasserstein between patient groups ≈ Sinkhorn between their cell stage distributions
- This is the central mathematical contribution
- Implement: `topoflow.validate.cross_scale_coherence()`

### Phase 4: Cross-cancer validation (August–October)
Port to NetFlow's existing datasets:
- MM (multiple myeloma): NetFlow already has full pipeline
- GBM (glioblastoma): NetFlow already has full pipeline
- BC (breast cancer): NetFlow already has full pipeline
For each: run StageBridge-equivalent cell-level flow matching on scRNA if available, or adapt to bulk with pseudo-single-cell strategy

### Phase 5: Unified package + paper (October–December)
```python
import topoflow as tf
keeper = tf.Keeper(adata_patients, modalities=["X_bulk", "X_cnv"])
topology = tf.POSE(keeper, root="Normal", n_branches=3)
model = tf.TopoFlow(topology, flow="sb_cfm", context_encoder="set_transformer")
model.fit(adata_cells, patient_col="patient_id", donor_col="patient_id")
trajectories = model.integrate(adata_test, source_state="AAH")
coherence = tf.validate.cross_scale_coherence(model, keeper)
```

---

## SB-CFM Implementation (This Week)

Files:
1. `stagebridge/utils/types.py` — add `sigma: float = 0.0`, `use_stochastic_bridge: bool = False`
2. `stagebridge/training/losses.py` — Brownian bridge: `x_t = (1-t)x_i + t·y_j + sqrt(t*(1-t))·σ·Z`
3. `stagebridge/models/stagebridge.py` — `integrate_euler_maruyama(x0, c_s, stage_pair_id, num_steps, sigma)`
4. `stagebridge/models/baselines.py` — same EM integration for DeepSets + NoContext
5. `configs/training/default.yaml` — `sigma: 0.1`, `use_stochastic_bridge: false`
6. `configs/experiment/full_benchmark.yaml` — add `stagebridge_sb` variant
7. `tests/test_flow_matching_toy.py` — SB path + sigma=0 recovery test

Verification: `pytest tests/ -q` passes; sigma=0.0 matches `integrate_euler()`.

---

## Full Dataset Stack

| Dataset | Scale | Use |
|---------|-------|-----|
| GSE308103 + GSE307534 | Cell (LUAD) | StageBridge primary training |
| GSE223503 | Cell (NSCLC brain mets) | BrainMet extension |
| GSE131907 (Kim 2020) | Cell | Cross-dataset validation |
| NetFlow MM cohort | Patient | TopoFlow cross-cancer validation |
| NetFlow GBM cohort | Patient | TopoFlow cross-cancer validation |
| NetFlow BC cohort | Patient | TopoFlow cross-cancer validation |
