# V2 Ideas and Open Questions

Ideas deferred from V1 and open methodological questions for future work.

---

## Open Methodological Questions

These are places where StageBridge is still scientifically underdetermined.

### Critical Gaps

1. **Dual-reference fusion is pragmatic, not principled** - Concatenation works but lacks theoretical grounding. Future: gated encoder, transport-based fusion, or confidence-aware shared-plus-specific latent.

2. **Spot-based niche context is latent, not observed** - In Visium-like data, the niche is reconstructed via deconvolution, not directly measured. Future: jointly infer neighborhoods with uncertainty.

3. **Cross-sectional snapshots and transition claims** - Single-cell measurements are destructive. What makes a transition model from stage snapshots identifiable? Future: flow matching, Schrödinger bridges for constrained coupling.

4. **Niche effects are associative, not causal** - Attention weights show association, not intervention effects. Future: neighborhood perturbation models, masked receiver generative modeling.

5. **Latent structure vs. biological mechanism** - A predictive embedding isn't automatically mechanistic. Future: program-aware auxiliary objectives, multi-view latents separating cell-state and regulatory information.

### Key Papers
- AMICI: Receiver-centered attention for cell-cell interaction
- OSDR: Tissue dynamics from spatial snapshot
- DestVI: Spot deconvolution with cell-state variation
- Flow matching / Schrödinger bridges: Constrained coupling

---

## Deferred Features

### Geometry Extensions
- **Hyperbolic embeddings** for cell type hierarchy
- **Spherical geometry** for cell cycle / periodic states
- **Product manifolds** combining Euclidean + non-Euclidean

### Model Extensions
- **Destination conditioning** - condition on target stage
- **Phase portrait analysis** - fixed points, basins of attraction
- **Neural SDE backends** beyond flow matching
- **Multi-scale temporal dynamics** - fast/slow processes

### Data Extensions
- **Multi-dataset training** - combine LUAD-Evo with other cohorts
- **Cross-tissue transfer** - generalize beyond lung
- **Temporal data integration** - if longitudinal data available

### Infrastructure Extensions
- **Real-time inference API**
- **Interactive visualization**
- **Bayesian uncertainty quantification**

### Research Directions
- Birth-death dynamics (OSDR-style population dynamics)
- Communication inference module
- Causal intervention modeling

---

Last updated: 2026-03-28
