# Local Niche Encoder Specification

This document specifies the design principles for the local neighborhood/niche encoder in StageBridge.

## Design Philosophy

The niche encoder models **how a cell's local neighborhood influences its state and trajectory**. It is receiver-centered: we ask "what does this cell receive from its neighbors?" not "what is the aggregate neighborhood state?"

## Required Properties

### 1. Receiver-Centered Architecture

```
Neighbors ──────┐
                │
    ┌───────────▼───────────┐
    │   Attention/Aggregation│
    │   (receiver as query)  │
    └───────────┬───────────┘
                │
                ▼
          Receiver Update
```

The focal cell (receiver) is the query. Neighbors are keys/values. Information flows TO the receiver.

**Implementation:**
```python
# Correct: receiver-centered
query = receiver_embedding  # [B, D]
keys = neighbor_embeddings  # [B, K, D]
values = neighbor_embeddings
context = attention(query, keys, values)  # What receiver gets from neighbors

# Wrong: symmetric/bag-level
pooled = mean(all_cell_embeddings)  # Loses receiver-centering
```

### 2. Distance-Aware Attention

Spatial distance must explicitly modulate attention weights.

**Options (choose one or combine):**

a) **Additive distance bias:**
```python
attn_logits = Q @ K.T + distance_bias(distances)
```

b) **Multiplicative distance decay:**
```python
attn_weights = softmax(Q @ K.T) * exp(-distances / sigma)
```

c) **Distance as feature:**
```python
K_with_dist = concat(K, distance_embedding(distances))
```

**NOT acceptable:**
- Ignoring distance entirely
- Learning distance implicitly through position encodings only

### 3. Sparsity/Entropy Regularization

Attention should be sparse (few informative neighbors) not diffuse (everything equally weighted).

**Regularization options:**

a) **Entropy penalty:**
```python
loss += lambda * entropy(attention_weights)
```

b) **Top-k hard attention:**
```python
attention_weights = top_k_softmax(logits, k=5)
```

c) **Sparsemax:**
```python
attention_weights = sparsemax(logits)  # Projects to simplex with sparsity
```

### 4. Interpretability via Neighbor Ablation

The encoder must support:
- Masking individual neighbors to measure influence
- Identifying which neighbors most affect the receiver
- Generating neighbor importance scores

**Interface:**
```python
def forward(self, receiver, neighbors, neighbor_mask=None):
    # neighbor_mask: [B, K] boolean, False = ablated
    ...
    return context, attention_weights
```

### 5. Self-Supervised Learning Signal

**Primary task: Masked Receiver Reconstruction**

Given a receiver's neighborhood, predict the receiver's state (or a masked portion of it).

```python
# During training
receiver_masked = mask_features(receiver)
context = niche_encoder(receiver_masked, neighbors)
receiver_reconstructed = decoder(context)
loss = reconstruction_loss(receiver_reconstructed, receiver)
```

This forces the encoder to extract receiver-relevant information from neighbors.

**NOT acceptable:**
- Only predicting pooled neighborhood statistics
- Predicting neighbor states (this is communication inference, not receiver-centering)

### 6. Cell-Type Conditioning (Optional)

Cell type labels can be used as auxiliary context, but:
- They are **optional helper features**, not ground truth
- The model should work without them (graceful degradation)
- They should not override learned representations

```python
# Acceptable: type as soft bias
type_embedding = cell_type_encoder(cell_types)
context = niche_encoder(receiver, neighbors, type_hint=type_embedding)

# NOT acceptable: type as hard constraint
context = niche_encoder(receiver, neighbors, cell_type=labels)  # Rigid
```

## Architecture Template

```python
class ReceiverCenteredNicheEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        max_neighbors: int = 20,
        distance_encoding: str = "rbf",  # or "mlp", "sinusoidal"
        sparsity_type: str = "entropy",  # or "topk", "sparsemax"
        sparsity_weight: float = 0.01,
    ):
        ...

    def forward(
        self,
        receiver: Tensor,           # [B, D]
        neighbors: Tensor,          # [B, K, D]
        distances: Tensor,          # [B, K]
        neighbor_mask: Tensor,      # [B, K] bool
        cell_type_hint: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            context: [B, D] - what receiver gets from neighborhood
            attention_weights: [B, K] - interpretable neighbor importance
        """
        ...
```

## Anti-Patterns

### Wrong: Bag-Level Pooling
```python
# This treats all cells equally, no receiver-centering
def forward(self, all_cells):
    return mean(all_cells, dim=1)
```

### Wrong: Symmetric Message Passing
```python
# This is communication inference, not receiver-centered
for layer in self.layers:
    all_cells = layer(all_cells, adjacency)  # All cells update equally
```

### Wrong: Vague "Context"
```python
# No explicit receiver, no distance, no sparsity
def get_context(self, neighbors):
    return self.mlp(mean(neighbors))
```

## Validation Checklist

Before accepting any niche encoder implementation:

- [ ] Is there a designated receiver cell?
- [ ] Does the receiver serve as the attention query?
- [ ] Is spatial distance explicitly used?
- [ ] Is attention regularized for sparsity?
- [ ] Can individual neighbors be ablated?
- [ ] Is there a masked receiver reconstruction loss?
- [ ] Does it work without cell type labels?

## Document Maintenance

This specification is maintained by the `research-director` agent.
