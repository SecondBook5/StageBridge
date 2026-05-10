# Context Encoding

Receiver-centered niche encoding for StageBridge.

## Key Modules

### `encoder.py` - ReceiverCenteredNicheEncoder

AMICI-style attention where the receiver cell is the query and neighbors are keys/values. Information flows TO the receiver.

**AMICI Features (Hong et al., bioRxiv 2025):**
- Distance coefficient enforced positive via Softplus, then subtracted
- Guarantees attention monotonically decreases with distance
- Empty neighbor token allows attention to "escape" when no neighbor is informative
- L1 penalty on value vectors for sparse influence patterns

```python
# Attention score for neighbor j:
score_j = phenotype_score(q, k_j) - softplus(b) * distance_j

# With empty token:
scores = [score_1, ..., score_K, empty_token_score]
attention = softmax(scores)
```

### `layers.py` - Set Transformer Components

- **SAB**: Self-Attention Block
- **ISAB**: Induced Set Attention Block (with inducing points)
- **PMA**: Pooling by Multihead Attention

### `tokenizer.py` - NicheTokenizer (Legacy)

Converts ring-binned neighbors to 9-token sequence. Deprecated in favor of AMICI continuous attention.

### `aggregation.py` - HierarchicalAggregator

Aggregates multiple niche embeddings to sample-level representation via ISAB + PMA.

## Architecture

```
Receiver + K Neighbors + Distances
           │
           ▼
┌─────────────────────────────────┐
│  ReceiverCenteredNicheEncoder   │
│  - Receiver as query            │
│  - Neighbors as keys/values     │
│  - Distance-weighted attention  │
│  - Empty token option           │
└─────────────────────────────────┘
           │
           ▼
    [B, hidden_dim] context
```

## Reference

Our receiver-centered attention mechanism is adapted from AMICI. If you use this module, please cite:

```bibtex
@article{Hong2025.09.22.677860,
  title = {AMICI: Attention Mechanism Interpretation of Cell-cell Interactions},
  author = {Hong, Justin and Desai, Khushi and Nguyen, Tu Duyen and Nazaret, Achille and Levy, Nathan and Ergen, Can and Plitas, George and Azizi, Elham},
  doi = {10.1101/2025.09.22.677860},
  journal = {bioRxiv},
  publisher = {Cold Spring Harbor Laboratory},
  year = {2025},
}
```

**Original implementation**: https://github.com/azizilab/amici

**License notice**: AMICI is licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives). Justin Hong, Khushi Desai, and Elham Azizi are inventors on provisional patent application U.S. Serial No. 63/884,704 (filed September 19, 2025) directed to the subject matter of AMICI.
