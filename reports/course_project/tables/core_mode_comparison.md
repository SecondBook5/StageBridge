# Core Mode Comparison

Results from matched context-mode comparisons on the LUAD evolution dataset (donor-holdout split, no WES, no diffusion).

Primary metric: Sinkhorn divergence (lower = better).

## AAH -> AIS

| Mode | Transformer? | Local Context? | Graph Context? | Sinkhorn | Interpretation |
|------|-------------|---------------|---------------|----------|----------------|
| rna_only | No | No | No | 17.252 | Baseline |
| pooled | No | Yes | No | 18.097 | Non-transformer context baseline |
| set_only | Yes | Yes | No | **17.817** | Main transformer mode |
| graph_of_sets | Yes | Yes | Yes | 18.683 | Optional extension |

## AIS -> MIA

| Mode | Transformer? | Local Context? | Graph Context? | Sinkhorn | Interpretation |
|------|-------------|---------------|---------------|----------|----------------|
| rna_only | No | No | No | 16.297 | Baseline |
| pooled | No | Yes | No | 15.909 | Non-transformer context baseline |
| set_only | Yes | Yes | No | **15.758** | Best active transformer mode |
| graph_of_sets | Yes | Yes | Yes | 16.002 | Optional extension |

## Summary

- **AIS->MIA**: Set Transformer context achieves best Sinkhorn divergence (15.758), outperforming pooled (15.909), graph_of_sets (16.002), and rna_only (16.297).
- **AAH->AIS**: RNA-only baseline achieves best score (17.252). Set Transformer (17.817) outperforms pooled (18.097) and graph_of_sets (18.683), but does not beat the simpler RNA-only baseline on this edge.
- **Overall**: Set Transformer is the most consistent transformer-based mode. Graph-of-sets does not earn flagship status.
