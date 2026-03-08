# ADR 003 Graph Must Earn It

Decision: graph-of-sets is an ablation candidate until it materially beats the set-only context model.

## Context

The context model has two modes: (1) set-only, where each (patient, stage) biological set is encoded independently by the Set Transformer; (2) graph-of-sets, where a Graph Transformer adds inter-set communication via typed edges.

Graph-of-sets is architecturally richer. It can capture cross-patient patterns and stage-adjacent dependencies. But it also adds parameters, compute cost, and interpretive complexity.

## Decision

Set-only is the default spatial baseline. Graph-of-sets is available but treated as an ablation candidate. It becomes the default only if ablation experiments demonstrate:

1. Measurable improvement on held-out transition metrics (not just training loss)
2. Interpretable edge-type biases (the model should reveal which relationships matter)
3. Improvement not fully explained by additional parameters (compare against a parameter-matched set-only variant)

## Rationale

- Set-only already captures spatial niche information via typed tokens from Tangram
- Adding graph structure is a hypothesis (inter-set communication helps), not an assumption
- Premature adoption of the more complex model risks masking problems with simpler failure modes
- The ablation comparison is itself a scientific result worth reporting

## Consequences

- Both modes must be maintained and testable
- Ablation experiments must use identical data splits and evaluation protocols
- Results must report both modes, not just the winner
