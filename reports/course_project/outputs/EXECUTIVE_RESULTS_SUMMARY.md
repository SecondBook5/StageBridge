# Executive Results Summary

## What was compared

Four context-encoding modes were compared on two LUAD progression edges (AAH->AIS, AIS->MIA) using the StageBridge framework:

1. **RNA-only**: No local context. Cells represented solely by their gene expression latent.
2. **Pooled**: Local niche context via mean/std/max pooling of typed spatial tokens. No transformer.
3. **Set Transformer (set_only)**: Local niche context via ISAB + SAB + PMA attention over typed tokens.
4. **Graph-of-Sets (graph_of_sets)**: Set Transformer + Graph Transformer for tissue-level propagation.

All comparisons used donor-holdout splits, Tangram spatial mapping, no WES, and no state-dependent diffusion.

## Strongest current result

**AIS->MIA**: Set Transformer achieves the best Sinkhorn divergence (15.758) among all modes, outperforming pooled (15.909, -1.0%), graph_of_sets (16.002, -1.5%), and RNA-only (16.297, -3.3%).

**AAH->AIS**: RNA-only achieves the best score (17.252). Set Transformer (17.817) outperforms pooled (18.097) and graph_of_sets (18.683), but does not beat the RNA-only baseline on this specific edge.

## What the transformer contribution currently supports

1. **Set Transformer consistently outperforms pooled context** on both edges. This confirms that attention-based set encoding provides a genuine improvement over simple statistical pooling of niche information.

2. **Set Transformer outperforms RNA-only on the more clinically relevant transition** (AIS->MIA, the invasive transition), suggesting that local niche context matters most for transitions involving microenvironment remodeling.

3. **Graph-of-Sets does not earn flagship status**. It underperforms set_only on both edges, indicating that tissue-level graph context does not yet add value beyond local niche encoding in this setting.

## What remains preliminary

- WES regularization and state-dependent diffusion were tested via scientific gates but showed mixed evidence. Both are retained as optional extensions.
- Only two stage edges have been systematically compared. The full 5-stage chain (Normal->AAH->AIS->MIA->LUAD) has not been run as a complete matched comparison.
- Current metrics are from moderate-scale runs (24-cell subsets per stage in smoke mode, full donor counts in production runs).
- Brain metastasis extension datasets are integrated in the codebase but not included in the course comparison.
