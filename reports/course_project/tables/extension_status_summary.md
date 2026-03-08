# Extension Status Summary

| Component | Tested? | Status | Evidence | In Main Story? | Future Role |
|-----------|---------|--------|----------|---------------|-------------|
| Graph-of-Sets Transformer | Yes | Mixed | Does not consistently outperform set_only; underperforms on both edges | Extension only | Potential improvement with graph construction tuning |
| WES Regularization | Yes | Mixed | WES gate shows slight improvement on AAH->AIS but not AIS->MIA | Extension only | Optional genomic constraint on transport |
| State-Dependent Diffusion | Yes | Mixed | Diffusion gate shows no consistent improvement over drift-only | Extension only | Required for full stochastic SDE interpretation |

## Decision

All three extensions are retained as optional modules in the codebase and discussed in the paper as planned future work. None are promoted to the main course story, which centers on the Set Transformer context encoder as the primary transformer contribution.
