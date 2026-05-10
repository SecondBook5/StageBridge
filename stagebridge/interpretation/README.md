# Interpretation

Interpretability tools for StageBridge, adapted from AMICI for stage progression analysis.

## Key Modules

### `ablation.py` - Token Ablation Analysis

Measure importance of each input component by ablating it and measuring prediction change.

```python
from stagebridge.interpretation import compute_token_ablation

# Ablate each token type and measure reconstruction loss change
results = compute_token_ablation(model, batch)
# results["ring_1"] = importance of ring 1 neighbors
# results["hlca"] = importance of healthy reference
```

### `attention.py` - Attention Pattern Extraction

Extract and visualize attention weights from AMICI encoder.

```python
from stagebridge.interpretation import extract_attention_patterns

patterns = extract_attention_patterns(model, batch)
# patterns.neighbor_attention: [B, K] attention to each neighbor
# patterns.empty_attention: [B] attention to empty token
```

### `networks.py` - Cell-Cell Interaction Networks

Build interaction networks from attention weights.

```python
from stagebridge.interpretation import build_interaction_network

network = build_interaction_network(
    attention_weights,
    cell_types,
    threshold=0.1,
)
```

### `dynamics.py` - Trajectory Analysis

Analyze transition trajectories and fate probabilities.

### `plotting.py` - AMICI-Style Visualizations

Publication-quality plots adapted from AMICI:
- `plot_interaction_network` - Network graph
- `plot_interaction_heatmap` - Cell type x cell type
- `plot_ring_attention_decay` - Distance vs attention
- `plot_ablation_importance` - Token importance bars

## Reference

Interpretation methods adapted from AMICI. Please cite:

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

**License notice**: AMICI is licensed under CC BY-NC-ND 4.0. Patent pending (U.S. Serial No. 63/884,704).
