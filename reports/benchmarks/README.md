# Benchmark Results

Reproducible benchmark tables for StageBridge. All results use donor-held-out cross-validation.

## Transition model

Niche-conditioned cell-state transition prediction (Schrödinger bridge, latent space MSE).

- [`transition_model/core_mode_comparison.csv`](transition_model/core_mode_comparison.csv) — All context encoder modes across disease edges
- [`transition_model/winning_modes_by_edge.csv`](transition_model/winning_modes_by_edge.csv) — Best model per edge

## Communication relay

Clonal-proxy classification benchmark across communication model architectures.

- [`communication_relay/ais_model_family_summary.csv`](communication_relay/ais_model_family_summary.csv) — Model comparison on AIS edge
- [`communication_relay/ais_context_shuffle_summary.csv`](communication_relay/ais_context_shuffle_summary.csv) — Context shuffle ablation
- [`communication_relay/label_balance_summary.csv`](communication_relay/label_balance_summary.csv) — Label distribution

## Cross-task summary

- [`story/transition_vs_communication_story.csv`](story/transition_vs_communication_story.csv) — Comparison across both task types
