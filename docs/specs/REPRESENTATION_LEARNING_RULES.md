# StageBridge Representation Learning Rules

This document defines the core representation learning principles that guide StageBridge development.

## Core Identity

StageBridge is a **cell-level representation learning framework**. Every design decision must reinforce this identity.

## Rule 1: Cells Are the Scientific Unit

- The model learns on **cells and cell neighborhoods**, not patients or lesions
- Bags/lesions/stage samples are **computational containers**, not the scientific center
- Any model variant whose central objective is lesion/patient classification is a **baseline**, not the flagship

## Rule 2: Representation Learning First

Before any downstream task, StageBridge learns:
1. **Cell state representations** in dual-reference geometry (HLCA + LuCA)
2. **Niche-aware representations** that capture local context influence
3. **Progression-aware representations** that encode stage-relevant information

Downstream tasks (transition modeling, classification) operate on these learned representations.

## Rule 3: Receiver-Centered Niche Modeling

The local niche encoder must:
- Be **receiver-centered**: model how neighbors influence a focal receiver cell
- Use **masked receiver reconstruction** as the primary self-supervised objective
- Maintain **distance-aware attention** (spatial proximity matters)
- Produce **interpretable influence tensors** (who influences whom)
- Support **neighbor ablation** for interpretation

## Rule 4: Dual-Reference Geometry

Every cell has coordinates in two reference spaces:
- **HLCA (healthy)**: proximity to normal lung cell atlas
- **LuCA (disease)**: proximity to cancer cell atlas
- **Fused**: learned combination capturing both anchors

This dual geometry is non-negotiable for V1.

## Rule 5: Progression, Not Classification

The downstream objective is **transition/progression modeling**:
- Model how cell states evolve across disease stages
- Use flow matching (V1) or neural SDE (V2) for dynamics
- Classification (if any) is auxiliary, not central

## Anti-Patterns to Avoid

1. **Lesion-centric design**: Making lesion embeddings the primary output
2. **Classification-first**: Optimizing for patient/stage classification as main loss
3. **Vague context**: Using "neighborhood context" without receiver-centered specifics
4. **Feature concatenation only**: Adding genomics as features without constraint logic
5. **Scope creep**: Adding V2/V3 features before V1 core is complete

## V1 Scope Boundaries

### V1 Core (Required)
- Euclidean dual-reference geometry
- Receiver-centered niche encoder (Layer B)
- Hierarchical set transformer (Layer C)
- Flow matching transition model (Layer E)
- Evolutionary compatibility (Layer F)
- Donor-held-out evaluation

### V2 (Deferred)
- Non-Euclidean geometry (hyperbolic/spherical)
- Neural SDE backend
- Phase portrait decoder
- Cohort transport layer

### V3 (Future)
- Destination-conditioned transitions
- Hypergraph extensions
- Cross-dataset transfer

## Validation Checklist

Before merging any code, verify:
- [ ] Does it preserve cells as the learning unit?
- [ ] Does it support representation learning objectives?
- [ ] Is niche modeling receiver-centered?
- [ ] Does it use dual references appropriately?
- [ ] Is it within V1 scope?

## References

- AGENTS.md Section 3: Scientific thesis
- docs/specs/V1_DOCTRINE.md: Full doctrine and V1 scope
- docs/architecture/typed_niche_context_model.md: Niche encoder details
