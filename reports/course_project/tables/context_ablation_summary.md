# Context Ablation Summary

## Per-Edge Results

| Edge | Best Mode | Set > Pooled? | Set > RNA-only? | GoST Earned? | Evidence |
|------|-----------|--------------|----------------|-------------|----------|
| AAH->AIS | rna_only | Yes (17.82 < 18.10) | No (17.82 > 17.25) | No | Weak pass |
| AIS->MIA | set_only | Yes (15.76 < 15.91) | Yes (15.76 < 16.30) | No | Pass |

## Interpretation

1. **Set Transformer consistently outperforms pooled context** on both edges, confirming that attention-based set encoding provides a genuine improvement over simple pooling.
2. **Set Transformer outperforms RNA-only on AIS->MIA** (the more clinically relevant invasive transition), but not on AAH->AIS. This suggests local niche context matters more for transitions involving significant microenvironment remodeling.
3. **Graph-of-sets does not earn flagship status** on either edge. It consistently underperforms set_only and often underperforms pooled context. This is an honest result: tissue-level graph context does not yet add value beyond local niche context in this setting.
