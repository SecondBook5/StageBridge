# Open Methodological Questions

These are not arbitrary extensions. They are the places where the current StageBridge idea is still scientifically underdetermined and may eventually need a new method rather than cleaner engineering.

---

## The Two Most Critical Gaps

> **1. Dual-reference fusion is still pragmatic rather than principled.**

> **2. Receiver-centered niche context in spot-based data is a latent object, not an observed one.**

These are the two places where a future StageBridge methods paper could become genuinely new rather than just more elaborate.

---

## 1. How should healthy and disease references be fused in a principled way?

StageBridge uses healthy and disease-aware references because they solve different problems. A healthy atlas anchors normal lung structure, while a disease-aware atlas captures malignant or progression-associated states. The unresolved issue is not whether to use both, but how to fuse them in a principled way when they differ in density, cohort composition, and coverage. Current fusion by concatenation is operational, but not yet methodologically satisfying. A future method may need a confidence-aware shared-plus-specific latent, a gated encoder, or a transport-based comparative fusion.

## 2. How should spot-based niche context be inferred when the neighborhood is not directly observed?

If the spatial data are spot-based, the local neighborhood is not directly observed. Methods such as DestVI exist precisely because many spatial transcriptomics technologies have spot sizes larger than a single cell, and current methods infer cell-type proportions and cell-state variation inside each spot rather than directly observing exact local cell neighborhoods.

So if StageBridge depends on a receiver-centered niche object, then in Visium-like data that object is partly reconstructed, not measured. That creates a real methodological gap: how should uncertainty from spot-to-cell reconstruction propagate into a niche-aware transition model? A future method may need to jointly infer receiver-centered local neighborhoods with uncertainty, rather than consuming a single deconvolution output as ground truth.

## 3. How can cross-sectional snapshots justify transition claims?

Single-cell measurements are destructive, so for most biological systems you do not track the same cell through time. Instead, you observe disjoint snapshots and try to infer population dynamics from them. That is exactly why transport-style and dynamics-from-snapshot methods exist. OSDR is a strong example: it explicitly frames the problem as inferring tissue dynamics from a single spatial biopsy and even shows prediction of responder versus non-responder behavior from early-treatment biopsies.

Your project is built on the same pressure point. The real open question is not just whether a classifier works, but what makes a transition model from stage snapshots identifiable enough to be believed. A stronger future method may require latent transport, flow matching, or bridge-like modeling so that progression is framed as a constrained coupling problem rather than a purely discriminative one.

## 4. How can niche effects move beyond associative attention?

A niche encoder can learn which neighbors are associated with a receiver state, but attention weights alone are not intervention effects. That means another real open question is how to estimate what would happen to a receiver cell if its neighborhood were changed.

This matters because snapshot-based cell-cell interaction methods are still largely associative. OSDR is relevant here because it shows that neighborhood composition can be used to infer tissue-level dynamics from a spatial snapshot, but it also depends on strong assumptions about how neighborhood composition drives cell population change.

A future StageBridge method might need a neighborhood perturbation model, masked receiver generative modeling, or another conditional model that estimates receiver-state change under perturbations of sender composition.

## 5. What biological structure should the latent preserve besides cell identity?

A predictive cell embedding is not automatically a mechanistic one. An open question is whether the latent should preserve only cell identity and niche context, or whether it should also explicitly preserve gene programs, regulons, or pathway-level structure.

This matters because a latent that is merely good for prediction may wash out the biology you actually want to explain. A future method might therefore need a program-aware auxiliary objective or a multi-view latent that separates cell-state and regulatory-program information.

---

## Relevant Papers

- **AMICI** (Hong et al.): Attention-based cell-cell interaction, but still associative
- **OSDR** (Somer et al.): Tissue dynamics from snapshot via neighborhood → division rate
- **DestVI**: Spot deconvolution with cell-state variation
- **Flow matching / Schrödinger bridges**: Constrained coupling for trajectory inference
