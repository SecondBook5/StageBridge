#!/usr/bin/env python3
"""
Add educational cells teaching transformer fundamentals to StageBridge notebook.

This script inserts comprehensive transformer education cells after cell 39 (5fceee58),
teaching core concepts using StageBridge as the concrete example.
"""

import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

def create_educational_cells():
    """Create educational cells teaching transformer concepts."""

    cells = []

    # ========================================================================
    # SECTION HEADER
    # ========================================================================
    cells.append(new_markdown_cell("""---

## TRANSFORMER ARCHITECTURE DEEP DIVE

**Educational Section**: Understanding Transformers Through StageBridge

This section teaches transformer fundamentals using our model as a concrete example. After this section, you should understand:

1. **Self-Attention Mechanism** - How attention computes relationships
2. **Scaled Dot-Product Attention** - Why scaling prevents saturation
3. **Multi-Head Attention** - How parallel heads capture different patterns
4. **Positional/Type Embeddings** - How transformers encode position/type
5. **Layer Normalization & Residuals** - How gradients flow through deep networks
6. **Feed-Forward Networks** - The MLP after each attention layer
7. **Set Transformer Components** - ISAB/PMA for efficient variable-size sets
8. **Complete Encoder Architecture** - How all pieces fit together

**Target audience**: Deep learning students and practitioners

**Prerequisites**: Basic linear algebra, PyTorch familiarity

---"""))

    # ========================================================================
    # CELL 1: Self-Attention Mechanism
    # ========================================================================
    cells.append(new_markdown_cell("""### 1. Self-Attention Mechanism

**Core Idea**: Attention allows each token to look at all other tokens and decide which ones are relevant.

**Mathematical Formulation**:

Given input sequence $X \\in \\mathbb{R}^{n \\times d}$ (n tokens, d dimensions):

1. **Project to Query, Key, Value**:
   - $Q = XW_Q$ where $W_Q \\in \\mathbb{R}^{d \\times d_k}$
   - $K = XW_K$ where $W_K \\in \\mathbb{R}^{d \\times d_k}$
   - $V = XW_V$ where $W_V \\in \\mathbb{R}^{d \\times d_v}$

2. **Compute attention scores**:
   $$\\text{scores} = \\frac{QK^T}{\\sqrt{d_k}}$$

3. **Apply softmax** (normalize to probabilities):
   $$\\text{attention\\_weights} = \\text{softmax}(\\text{scores})$$

4. **Weighted combination of values**:
   $$\\text{output} = \\text{attention\\_weights} \\cdot V$$

**Intuition**:
- Query: "What am I looking for?"
- Key: "What do I represent?"
- Value: "What information do I carry?"
- Attention weight = similarity between query and key"""))

    cells.append(new_code_cell("""# Extract self-attention from StageBridge SAB module
import torch
import torch.nn.functional as F
from stagebridge.context_model.set_encoder import SAB
import matplotlib.pyplot as plt
import seaborn as sns

# Create a simple SAB (Self-Attention Block)
dim = 64
num_heads = 4
sab = SAB(dim=dim, num_heads=num_heads, dropout=0.0)

# Create synthetic token sequence (9 tokens for StageBridge: receiver + 4 rings + HLCA + LuCA + pathway + stats)
batch_size = 1
num_tokens = 9
x = torch.randn(batch_size, num_tokens, dim)

# Get attention weights
with torch.no_grad():
    output, attn_weights = sab(x, return_attention=True)

print("=" * 70)
print("SELF-ATTENTION MECHANISM")
print("=" * 70)
print(f"\\nInput shape: {x.shape}")
print(f"  - Batch size: {batch_size}")
print(f"  - Number of tokens: {num_tokens} (receiver, 4 rings, HLCA, LuCA, pathway, stats)")
print(f"  - Token dimension: {dim}")

# Extract Q, K, V weight matrices from the first head
mha = sab.mha
d_k = dim // num_heads  # dimension per head

print(f"\\nMulti-Head Attention Parameters:")
print(f"  - Number of heads: {num_heads}")
print(f"  - Dimension per head (d_k): {d_k}")
print(f"  - Q, K, V weight matrices shape: {mha.in_proj_weight.shape}")
print(f"    (Combined weights: [Q; K; V] stacked)")

# Attention weights shape
print(f"\\nAttention weights shape: {attn_weights.shape}")
print(f"  - [batch_size, num_heads, num_tokens_query, num_tokens_key]")
print(f"  - [{attn_weights.shape[0]}, {attn_weights.shape[1]}, {attn_weights.shape[2]}, {attn_weights.shape[3]}]")

print(f"\\nAttention weight properties:")
print(f"  - Each row sums to 1.0 (softmax normalization): {attn_weights[0, 0, 0, :].sum():.4f}")
print(f"  - All values are non-negative: min = {attn_weights.min():.4f}, max = {attn_weights.max():.4f}")

# Visualize attention pattern for first head
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Head 0 attention pattern
sns.heatmap(attn_weights[0, 0].numpy(), annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Recv', 'R1', 'R2', 'R3', 'R4', 'HLCA', 'LuCA', 'Path', 'Stats'],
            yticklabels=['Recv', 'R1', 'R2', 'R3', 'R4', 'HLCA', 'LuCA', 'Path', 'Stats'],
            ax=axes[0], cbar_kws={'label': 'Attention Weight'})
axes[0].set_title(f'Self-Attention Pattern (Head 0)\\nEach row shows where that token attends', fontsize=12)
axes[0].set_xlabel('Key (attended to)', fontsize=10)
axes[0].set_ylabel('Query (attending from)', fontsize=10)

# Head 1 attention pattern
sns.heatmap(attn_weights[0, 1].numpy(), annot=True, fmt='.2f', cmap='Oranges',
            xticklabels=['Recv', 'R1', 'R2', 'R3', 'R4', 'HLCA', 'LuCA', 'Path', 'Stats'],
            yticklabels=['Recv', 'R1', 'R2', 'R3', 'R4', 'HLCA', 'LuCA', 'Path', 'Stats'],
            ax=axes[1], cbar_kws={'label': 'Attention Weight'})
axes[1].set_title(f'Self-Attention Pattern (Head 1)\\nDifferent heads learn different patterns', fontsize=12)
axes[1].set_xlabel('Key (attended to)', fontsize=10)
axes[1].set_ylabel('Query (attending from)', fontsize=10)

plt.tight_layout()
save_figure(fig, 'transformer_self_attention_mechanism')
plt.show()

print("\\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("Each token (row) attends to all tokens (columns) with learned weights.")
print("Different heads capture different relationships (e.g., local vs global).")
print("=" * 70)"""))

    # ========================================================================
    # CELL 2: Scaled Dot-Product Attention
    # ========================================================================
    cells.append(new_markdown_cell("""### 2. Scaled Dot-Product Attention

**Why do we divide by $\\sqrt{d_k}$?**

The dot product $QK^T$ grows with dimension $d_k$. Without scaling:
- High-dimensional dot products → large magnitudes
- Large magnitudes → softmax saturation
- Saturation → vanishing gradients

**Formula**:
$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$

**Effect of scaling**:
- Keeps dot products in a reasonable range
- Prevents softmax from producing near-zero gradients
- Acts like temperature control in softmax"""))

    cells.append(new_code_cell("""# Demonstrate the effect of scaling on attention distribution
import numpy as np

# Simulate Q and K for different dimensions
dimensions = [16, 64, 256, 1024]
num_tokens = 9

print("=" * 70)
print("SCALED DOT-PRODUCT ATTENTION")
print("=" * 70)
print("\\nEffect of dimension on dot product magnitude:\\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, d_k in enumerate(dimensions):
    # Random Q and K
    Q = torch.randn(1, num_tokens, d_k)
    K = torch.randn(1, num_tokens, d_k)

    # Unscaled attention scores
    scores_unscaled = torch.bmm(Q, K.transpose(1, 2))  # (1, 9, 9)

    # Scaled attention scores
    scores_scaled = scores_unscaled / np.sqrt(d_k)

    # Apply softmax
    attn_unscaled = F.softmax(scores_unscaled, dim=-1)
    attn_scaled = F.softmax(scores_scaled, dim=-1)

    # Statistics
    print(f"d_k = {d_k}:")
    print(f"  Unscaled score range: [{scores_unscaled.min():.2f}, {scores_unscaled.max():.2f}]")
    print(f"  Scaled score range:   [{scores_scaled.min():.2f}, {scores_scaled.max():.2f}]")
    print(f"  Unscaled attention entropy: {-(attn_unscaled * torch.log(attn_unscaled + 1e-9)).sum(-1).mean():.3f}")
    print(f"  Scaled attention entropy:   {-(attn_scaled * torch.log(attn_scaled + 1e-9)).sum(-1).mean():.3f}")
    print()

    # Visualize
    ax = axes[idx]
    positions = np.arange(num_tokens)
    width = 0.35

    ax.bar(positions - width/2, attn_unscaled[0, 0].numpy(), width, label='Unscaled', alpha=0.7)
    ax.bar(positions + width/2, attn_scaled[0, 0].numpy(), width, label='Scaled', alpha=0.7)

    ax.set_xlabel('Token Index')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'd_k = {d_k}\\nScaling prevents saturation at high dimensions')
    ax.legend()
    ax.set_xticks(positions)
    ax.set_xticklabels(['R', 'R1', 'R2', 'R3', 'R4', 'H', 'L', 'P', 'S'])
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_figure(fig, 'transformer_scaled_attention')
plt.show()

print("=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("Without scaling, high-dimensional dot products saturate softmax,")
print("producing near-uniform or near-one-hot distributions (low entropy).")
print("Scaling by sqrt(d_k) keeps attention well-behaved across dimensions.")
print("=" * 70)"""))

    # ========================================================================
    # CELL 3: Multi-Head Attention
    # ========================================================================
    cells.append(new_markdown_cell("""### 3. Multi-Head Attention

**Why multiple heads?**

Different heads can specialize in different types of relationships:
- **Head 1**: Local interactions (adjacent tokens)
- **Head 2**: Global context (receiver ↔ reference atlases)
- **Head 3**: Hierarchical structure (rings at different scales)
- **Head 4**: Cross-modality relationships (spatial ↔ genomic)

**Architecture**:

Instead of one attention operation with dimension $d_{\\text{model}}$, we split into $h$ heads:

1. **Split**: $d_k = d_v = d_{\\text{model}} / h$
2. **Parallel attention**: Each head computes attention independently
3. **Concatenate**: Combine all head outputs
4. **Project**: Linear layer to $d_{\\text{model}}$

$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h) W^O$$

where $\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$"""))

    cells.append(new_code_cell("""# Analyze what different attention heads learn
from stagebridge.context_model.set_encoder import SAB

# Create SAB with multiple heads
dim = 128
num_heads = 8
sab = SAB(dim=dim, num_heads=num_heads, dropout=0.0)

# Create 9-token sequence with structure
batch_size = 1
num_tokens = 9

# Structured input: receiver has different pattern than rings, references, etc.
x = torch.zeros(batch_size, num_tokens, dim)
x[0, 0, :] = 1.0  # Receiver (distinct)
x[0, 1:5, :] = torch.randn(4, dim) * 0.5  # Rings (similar to each other)
x[0, 5, :] = 2.0  # HLCA (distinct)
x[0, 6, :] = 2.0  # LuCA (distinct, similar to HLCA)
x[0, 7, :] = torch.randn(dim) * 0.3  # Pathway
x[0, 8, :] = torch.randn(dim) * 0.3  # Stats

# Get attention weights
with torch.no_grad():
    output, attn_weights = sab(x, return_attention=True)

print("=" * 70)
print("MULTI-HEAD ATTENTION")
print("=" * 70)
print(f"\\nArchitecture:")
print(f"  - Model dimension: {dim}")
print(f"  - Number of heads: {num_heads}")
print(f"  - Dimension per head: {dim // num_heads}")
print(f"\\nAttention weights shape: {attn_weights.shape}")
print(f"  - [batch, heads, query_tokens, key_tokens]")

# Analyze head specialization
token_labels = ['Recv', 'R1', 'R2', 'R3', 'R4', 'HLCA', 'LuCA', 'Path', 'Stats']

print("\\n" + "-" * 70)
print("HEAD SPECIALIZATION ANALYSIS")
print("-" * 70)

for head_idx in range(num_heads):
    attn_head = attn_weights[0, head_idx].numpy()

    # Compute statistics
    receiver_to_rings = attn_head[0, 1:5].mean()  # Receiver attending to rings
    receiver_to_refs = attn_head[0, 5:7].mean()  # Receiver attending to references
    rings_to_rings = attn_head[1:5, 1:5].mean()  # Rings attending to each other
    refs_to_receiver = attn_head[5:7, 0].mean()  # References attending to receiver

    print(f"\\nHead {head_idx}:")
    print(f"  Receiver → Rings:      {receiver_to_rings:.3f}")
    print(f"  Receiver → References: {receiver_to_refs:.3f}")
    print(f"  Rings ↔ Rings:        {rings_to_rings:.3f}")
    print(f"  References → Receiver: {refs_to_receiver:.3f}")

# Visualize all heads
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for head_idx in range(num_heads):
    ax = axes[head_idx]
    sns.heatmap(attn_weights[0, head_idx].numpy(), annot=False, fmt='.2f',
                cmap='viridis', vmin=0, vmax=0.3,
                xticklabels=token_labels, yticklabels=token_labels,
                ax=ax, cbar_kws={'label': 'Weight'})
    ax.set_title(f'Head {head_idx}', fontsize=11)
    ax.set_xlabel('Key', fontsize=9)
    ax.set_ylabel('Query', fontsize=9)
    ax.tick_params(labelsize=8)

plt.suptitle('Multi-Head Attention: Each Head Learns Different Patterns', fontsize=14, y=1.00)
plt.tight_layout()
save_figure(fig, 'transformer_multihead_attention')
plt.show()

print("\\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("Different heads specialize in different relationships:")
print("  - Some heads focus on local structure (ring-to-ring)")
print("  - Some heads focus on global context (receiver-to-references)")
print("  - Some heads capture hierarchical relationships")
print("Parallelism allows the model to capture multiple relationship types simultaneously.")
print("=" * 70)"""))

    # ========================================================================
    # CELL 4: Positional and Type Embeddings
    # ========================================================================
    cells.append(new_markdown_cell("""### 4. Positional and Type Embeddings

**Problem**: Self-attention is permutation-invariant!

Without positional information, the model can't distinguish:
- Token 1 vs Token 2 (position matters)
- Receiver vs Ring vs Reference (type matters)

**Solutions**:

1. **Standard Transformers**: Sinusoidal positional encoding
   $$PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d})$$
   $$PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d})$$

2. **StageBridge**: Learned type embeddings
   - Type 0: Receiver token
   - Type 1: Spatial ring tokens (with ring ID embedding)
   - Type 2: HLCA reference token
   - Type 3: LuCA reference token
   - Type 4: Pathway summary token
   - Type 5: Neighborhood statistics token

**Why type embeddings?** Our tokens have semantic meaning (not just sequential positions)."""))

    cells.append(new_code_cell("""# Demonstrate type embeddings in StageBridge
from stagebridge.context_model.local_niche_encoder import LocalNicheTokenizer

# Create tokenizer
tokenizer = LocalNicheTokenizer(
    receiver_dim=64,
    sender_feature_dim=32,
    hlca_dim=30,
    luca_dim=10,
    lr_summary_dim=20,
    stats_dim=10,
    model_dim=128,
    num_receiver_states=32,
    num_rings=4,
    dropout=0.0
)

print("=" * 70)
print("POSITIONAL AND TYPE EMBEDDINGS")
print("=" * 70)

# Check embedding parameters
print("\\nType Embedding Parameters:")
print(f"  - Number of token types: 7")
print(f"  - Embedding dimension: {tokenizer.token_type_embedding.weight.shape[1]}")
print(f"  - Token types:")
print(f"      0: Receiver")
print(f"      1: Spatial ring")
print(f"      2: HLCA reference")
print(f"      3: LuCA reference")
print(f"      4: Pathway summary")
print(f"      5: Neighborhood statistics")
print(f"      6: Atlas contrast (optional)")

print("\\nRing ID Embedding Parameters:")
print(f"  - Number of rings: {tokenizer.ring_embedding.weight.shape[0]}")
print(f"  - Embedding dimension: {tokenizer.ring_embedding.weight.shape[1]}")
print(f"  - Purpose: Distinguish ring 1 (innermost) from ring 4 (outermost)")

# Extract type embeddings
with torch.no_grad():
    type_embeddings = tokenizer.token_type_embedding.weight.numpy()
    ring_embeddings = tokenizer.ring_embedding.weight.numpy()

# Visualize type embeddings (first 32 dimensions for visualization)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Type embeddings
ax = axes[0]
im = ax.imshow(type_embeddings[:6, :32], aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
ax.set_yticks(range(6))
ax.set_yticklabels(['Receiver', 'Ring', 'HLCA', 'LuCA', 'Pathway', 'Stats'])
ax.set_xlabel('Embedding Dimension (first 32 shown)')
ax.set_ylabel('Token Type')
ax.set_title('Type Embeddings (Learned)\\nEach type gets a distinct embedding vector')
plt.colorbar(im, ax=ax, label='Weight Value')

# Ring ID embeddings
ax = axes[1]
im = ax.imshow(ring_embeddings[:, :32], aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
ax.set_yticks(range(4))
ax.set_yticklabels(['Ring 1 (inner)', 'Ring 2', 'Ring 3', 'Ring 4 (outer)'])
ax.set_xlabel('Embedding Dimension (first 32 shown)')
ax.set_ylabel('Ring ID')
ax.set_title('Ring ID Embeddings (Learned)\\nCaptures hierarchical spatial structure')
plt.colorbar(im, ax=ax, label='Weight Value')

plt.tight_layout()
save_figure(fig, 'transformer_type_embeddings')
plt.show()

# Compute pairwise distances between type embeddings
from scipy.spatial.distance import pdist, squareform

dist_matrix = squareform(pdist(type_embeddings[:6], metric='cosine'))

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(dist_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=['Recv', 'Ring', 'HLCA', 'LuCA', 'Path', 'Stats'],
            yticklabels=['Recv', 'Ring', 'HLCA', 'LuCA', 'Path', 'Stats'],
            ax=ax, cbar_kws={'label': 'Cosine Distance'})
ax.set_title('Type Embedding Similarity\\nSimilar types have smaller distances')
plt.tight_layout()
save_figure(fig, 'transformer_type_similarity')
plt.show()

print("\\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("Type embeddings are ADDED to token features, injecting semantic type information.")
print("The model learns which types should be similar (e.g., HLCA and LuCA both references).")
print("Ring embeddings capture hierarchical spatial scale (inner vs outer neighborhoods).")
print("=" * 70)"""))

    # ========================================================================
    # CELL 5: Layer Normalization and Residual Connections
    # ========================================================================
    cells.append(new_markdown_cell("""### 5. Layer Normalization and Residual Connections

**Two critical architectural choices for deep networks:**

#### **Residual Connections** (Skip Connections)

$$\\text{output} = \\text{input} + \\text{sublayer}(\\text{input})$$

**Benefits**:
- Gradient flow: Gradients can bypass sublayers
- Identity mapping: Model can learn to keep input unchanged
- Training stability: Prevents degradation in deep networks

#### **Layer Normalization**

Normalize across feature dimension (not batch):

$$\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta$$

where $\\mu$ and $\\sigma$ are computed per token.

**Pre-Norm vs Post-Norm**:
- **Pre-Norm** (used in StageBridge): Normalize before sublayer → more stable training
- **Post-Norm** (original Transformer): Normalize after sublayer → slightly better final performance

**Standard Transformer Block**:
```
x = LayerNorm(x)
x = x + MultiHeadAttention(x)
x = LayerNorm(x)
x = x + FeedForward(x)
```"""))

    cells.append(new_code_cell("""# Demonstrate layer normalization and residual connections
from stagebridge.context_model.set_encoder import SAB

# Create SAB (has built-in LayerNorm and residual connections)
dim = 128
sab = SAB(dim=dim, num_heads=4, dropout=0.0)

# Input
batch_size = 2
num_tokens = 9
x = torch.randn(batch_size, num_tokens, dim)

print("=" * 70)
print("LAYER NORMALIZATION AND RESIDUAL CONNECTIONS")
print("=" * 70)

# Track intermediate values
with torch.no_grad():
    # Step 1: Attention (without residual)
    attn_out, _ = sab.mha(x, x, x, need_weights=True)

    # Step 2: Add residual
    after_residual = x + attn_out

    # Step 3: Layer norm
    after_ln1 = sab.ln1(after_residual)

    # Step 4: Feed-forward
    ff_out = sab.ff(after_ln1)

    # Step 5: Add residual
    after_residual2 = after_ln1 + ff_out

    # Step 6: Layer norm
    final_out = sab.ln2(after_residual2)

print("\\nIntermediate Statistics (first token, first sample):")
print(f"  Input:                   mean={x[0,0].mean():.4f}, std={x[0,0].std():.4f}")
print(f"  After Attention:         mean={attn_out[0,0].mean():.4f}, std={attn_out[0,0].std():.4f}")
print(f"  After Residual (x+attn): mean={after_residual[0,0].mean():.4f}, std={after_residual[0,0].std():.4f}")
print(f"  After LayerNorm:         mean={after_ln1[0,0].mean():.4f}, std={after_ln1[0,0].std():.4f}")
print(f"  After FeedForward:       mean={ff_out[0,0].mean():.4f}, std={ff_out[0,0].std():.4f}")
print(f"  After Residual (x+ff):   mean={after_residual2[0,0].mean():.4f}, std={after_residual2[0,0].std():.4f}")
print(f"  After LayerNorm:         mean={final_out[0,0].mean():.4f}, std={final_out[0,0].std():.4f}")

# Visualize the effect of LayerNorm
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Plot distributions at different stages
stages = [
    (x[0, 0].numpy(), "Input"),
    (attn_out[0, 0].numpy(), "After Attention"),
    (after_residual[0, 0].numpy(), "After Residual 1"),
    (after_ln1[0, 0].numpy(), "After LayerNorm 1"),
    (ff_out[0, 0].numpy(), "After FeedForward"),
    (final_out[0, 0].numpy(), "After LayerNorm 2")
]

for idx, (data, title) in enumerate(stages):
    ax = axes[idx // 3, idx % 3]
    ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Effect of LayerNorm: Normalizes distribution to mean≈0, std≈1', fontsize=14)
plt.tight_layout()
save_figure(fig, 'transformer_layernorm_effect')
plt.show()

# Demonstrate gradient flow with residual connections
print("\\n" + "-" * 70)
print("GRADIENT FLOW ANALYSIS")
print("-" * 70)
print("\\nWithout residual connections:")
print("  gradient must flow through all layers → can vanish/explode")
print("\\nWith residual connections:")
print("  gradient has direct path: ∂L/∂x = ∂L/∂output + ∂L/∂sublayer")
print("  The '+1' gradient ensures signal always flows backward")

print("\\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("LayerNorm: Stabilizes activations (mean≈0, std≈1) → easier optimization")
print("Residuals: Enable gradient flow through deep networks → trainable depth")
print("Pre-Norm: More stable training, especially for deep transformers")
print("=" * 70)"""))

    # ========================================================================
    # CELL 6: Feed-Forward Networks
    # ========================================================================
    cells.append(new_markdown_cell("""### 6. Feed-Forward Networks (FFN)

**Applied after each attention layer** (position-wise):

$$\\text{FFN}(x) = \\text{GELU}(xW_1 + b_1)W_2 + b_2$$

**Architecture**:
- **Expand**: $d_{\\text{model}} \\rightarrow d_{\\text{ff}}$ (typically $d_{\\text{ff}} = 4 \\times d_{\\text{model}}$)
- **Activation**: GELU (Gaussian Error Linear Unit)
- **Contract**: $d_{\\text{ff}} \\rightarrow d_{\\text{model}}$

**Purpose**:
- Attention is **information routing** (where to look)
- FFN is **information processing** (what to do with it)
- Adds nonlinear transformations and computational capacity

**GELU vs ReLU**:
- ReLU: $\\text{ReLU}(x) = \\max(0, x)$ (hard cutoff)
- GELU: $\\text{GELU}(x) \\approx x \\cdot \\Phi(x)$ (smooth, probabilistic)
- GELU is smoother → better gradients"""))

    cells.append(new_code_cell("""# Examine Feed-Forward Network in StageBridge
from stagebridge.context_model.set_encoder import FeedForwardBlock

# Create FFN
dim = 128
hidden_dim = 512  # 4x expansion
ffn = FeedForwardBlock(dim=dim, hidden_dim=hidden_dim, dropout=0.0)

print("=" * 70)
print("FEED-FORWARD NETWORKS (FFN)")
print("=" * 70)

print("\\nArchitecture:")
print(f"  Input dimension:  {dim}")
print(f"  Hidden dimension: {hidden_dim} (4x expansion)")
print(f"  Output dimension: {dim}")
print(f"\\nLayers:")
for idx, layer in enumerate(ffn.net):
    print(f"  {idx}. {layer}")

# Compare GELU vs ReLU
x_range = torch.linspace(-3, 3, 200)
gelu_output = torch.nn.functional.gelu(x_range)
relu_output = torch.nn.functional.relu(x_range)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Activation functions
ax = axes[0]
ax.plot(x_range.numpy(), gelu_output.numpy(), label='GELU', linewidth=2)
ax.plot(x_range.numpy(), relu_output.numpy(), label='ReLU', linewidth=2, linestyle='--')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Input')
ax.set_ylabel('Output')
ax.set_title('Activation Functions\\nGELU is smooth, ReLU has hard cutoff')
ax.legend()
ax.grid(alpha=0.3)

# FFN processing
batch_size = 1
num_tokens = 9
x = torch.randn(batch_size, num_tokens, dim)

with torch.no_grad():
    # Step through FFN
    x1 = ffn.net[0](x)  # Linear expansion
    x2 = ffn.net[1](x1)  # GELU
    x3 = ffn.net[3](x2)  # Linear contraction

    print(f"\\nIntermediate shapes:")
    print(f"  Input:            {x.shape}")
    print(f"  After expansion:  {x1.shape}")
    print(f"  After GELU:       {x2.shape}")
    print(f"  After contraction: {x3.shape}")

# Visualize token transformation
ax = axes[1]
token_norms_before = x[0].norm(dim=1).numpy()
token_norms_after = x3[0].norm(dim=1).numpy()
token_labels = ['Recv', 'R1', 'R2', 'R3', 'R4', 'HLCA', 'LuCA', 'Path', 'Stats']
x_pos = np.arange(len(token_labels))
width = 0.35
ax.bar(x_pos - width/2, token_norms_before, width, label='Before FFN', alpha=0.7)
ax.bar(x_pos + width/2, token_norms_after, width, label='After FFN', alpha=0.7)
ax.set_xlabel('Token')
ax.set_ylabel('L2 Norm')
ax.set_title('Token Transformation Through FFN\\nNonlinear processing changes token representations')
ax.set_xticks(x_pos)
ax.set_xticklabels(token_labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Hidden layer activations
ax = axes[2]
with torch.no_grad():
    hidden_activations = ffn.net[1](ffn.net[0](x[0, 0:1, :]))  # First token
ax.hist(hidden_activations.numpy().flatten(), bins=50, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Activation Value')
ax.set_ylabel('Frequency')
ax.set_title(f'Hidden Layer Activations (Token 0)\\nExpanded to {hidden_dim} dimensions')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_figure(fig, 'transformer_feedforward_network')
plt.show()

print("\\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("Attention: Routes information (decides what to attend to)")
print("FFN: Processes information (nonlinear transformations)")
print("Expansion (4x): Increases model capacity for complex computations")
print("GELU: Smooth activation → better gradient flow than ReLU")
print("=" * 70)"""))

    # ========================================================================
    # CELL 7: Set Transformer Components (ISAB/PMA)
    # ========================================================================
    cells.append(new_markdown_cell("""### 7. Set Transformer Components: ISAB and PMA

**Problem**: Standard attention is $O(n^2)$ in sequence length.

For variable-size sets (e.g., different numbers of cells per ring), we need:
- **Efficiency**: Handle varying set sizes without quadratic cost
- **Permutation invariance**: Order shouldn't matter
- **Fixed output**: Pool to fixed-size representation

**Solutions**:

#### **ISAB (Induced Set Attention Block)**

Uses $m$ learnable inducing points to reduce complexity from $O(n^2)$ to $O(nm)$:

1. **Inducing points attend to input**: $H = \\text{Attention}(I, X, X)$ where $I \\in \\mathbb{R}^{m \\times d}$
2. **Input attends to inducing points**: $Y = \\text{Attention}(X, H, H)$

Complexity: $O(nm)$ instead of $O(n^2)$

#### **PMA (Pooling by Multihead Attention)**

Uses $k$ learnable seed vectors to pool variable-size set to fixed output:

$$\\text{PMA}(X) = \\text{Attention}(S, X, X)$$

where $S \\in \\mathbb{R}^{k \\times d}$ are learnable seeds.

**In StageBridge**: Each spatial ring uses ISAB → SAB → PMA to aggregate variable numbers of cells into fixed-size ring token."""))

    cells.append(new_code_cell("""# Demonstrate ISAB and PMA for efficient set processing
from stagebridge.context_model.set_encoder import ISAB, PMA

dim = 128
num_heads = 4
num_inducing_points = 16
num_seed_vectors = 1

# Create modules
isab = ISAB(dim=dim, num_heads=num_heads, num_inducing_points=num_inducing_points, dropout=0.0)
pma = PMA(dim=dim, num_heads=num_heads, num_seed_vectors=num_seed_vectors, dropout=0.0)

print("=" * 70)
print("SET TRANSFORMER: ISAB AND PMA")
print("=" * 70)

# Variable-size inputs (simulating different ring sizes)
batch_size = 3
set_sizes = [50, 100, 200]  # Different numbers of cells per ring

print("\\nISAB (Induced Set Attention Block):")
print(f"  - Inducing points: {num_inducing_points}")
print(f"  - Purpose: Reduce O(n²) to O(nm) complexity")

print("\\nPMA (Pooling by Multihead Attention):")
print(f"  - Seed vectors: {num_seed_vectors}")
print(f"  - Purpose: Pool variable-size set to fixed output")

# Process variable-size inputs
results = []
for set_size in set_sizes:
    x = torch.randn(1, set_size, dim)

    # ISAB: n tokens → n tokens (via m inducing points)
    with torch.no_grad():
        isab_out = isab(x)
        pma_out = pma(isab_out)

    results.append((set_size, isab_out.shape, pma_out.shape))
    print(f"\\nInput size: {set_size}")
    print(f"  After ISAB: {isab_out.shape} (same size, but O(nm) complexity)")
    print(f"  After PMA:  {pma_out.shape} (pooled to fixed size)")

# Visualize inducing points
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Inducing points
ax = axes[0]
with torch.no_grad():
    inducing_points = isab.inducing_points[0].numpy()
im = ax.imshow(inducing_points.T[:32, :], aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
ax.set_xlabel('Inducing Point Index')
ax.set_ylabel('Dimension (first 32 shown)')
ax.set_title(f'ISAB: {num_inducing_points} Learnable Inducing Points\\nCompress information from variable-size input')
plt.colorbar(im, ax=ax, label='Weight')

# PMA seed vectors
ax = axes[1]
with torch.no_grad():
    seed_vectors = pma.seed_vectors[0].numpy()
im = ax.imshow(seed_vectors.T[:32, :], aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
ax.set_xlabel('Seed Vector Index')
ax.set_ylabel('Dimension (first 32 shown)')
ax.set_title(f'PMA: {num_seed_vectors} Learnable Seed Vector(s)\\nPool to fixed-size output')
plt.colorbar(im, ax=ax, label='Weight')

# Complexity comparison
ax = axes[2]
n_values = np.arange(10, 500, 10)
m = num_inducing_points
standard_complexity = n_values ** 2
isab_complexity = n_values * m

ax.plot(n_values, standard_complexity, label='Standard Attention O(n²)', linewidth=2)
ax.plot(n_values, isab_complexity, label=f'ISAB O(nm) with m={m}', linewidth=2, linestyle='--')
ax.set_xlabel('Set Size (n)')
ax.set_ylabel('Computational Cost (arbitrary units)')
ax.set_title('Complexity: ISAB vs Standard Attention\\nISAB scales linearly, not quadratically')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(10, 500)

plt.tight_layout()
save_figure(fig, 'transformer_isab_pma')
plt.show()

# Demonstrate permutation invariance
print("\\n" + "-" * 70)
print("PERMUTATION INVARIANCE TEST")
print("-" * 70)

set_size = 50
x1 = torch.randn(1, set_size, dim)
x2 = x1[:, torch.randperm(set_size), :]  # Shuffled

with torch.no_grad():
    out1 = pma(isab(x1))
    out2 = pma(isab(x2))

difference = (out1 - out2).abs().max().item()
print(f"\\nMax difference after permutation: {difference:.6f}")
print(f"  (Should be very small, confirming permutation invariance)")

print("\\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("ISAB: Efficient attention via inducing points (O(nm) instead of O(n²))")
print("PMA: Pools variable-size sets to fixed output (learnable aggregation)")
print("Both are permutation-invariant: order doesn't matter")
print("Perfect for spatial rings with varying numbers of cells!")
print("=" * 70)"""))

    # ========================================================================
    # CELL 8: Complete Encoder Architecture
    # ========================================================================
    cells.append(new_markdown_cell("""### 8. Complete Transformer Encoder Architecture

**StageBridge 9-Token Architecture**:

```
Token Sequence (9 tokens):
┌─────────────┬──────────────────┬──────────┬──────────┬─────────┬────────┐
│  Receiver   │  Ring 1-4        │  HLCA    │  LuCA    │ Pathway │ Stats  │
│  (masked)   │  (spatial niche) │  (ref)   │  (ref)   │ (LR)    │ (nbhd) │
└─────────────┴──────────────────┴──────────┴──────────┴─────────┴────────┘
       ↓              ↓              ↓          ↓          ↓          ↓
    Type 0         Type 1          Type 2     Type 3     Type 4     Type 5
       ↓              ↓              ↓          ↓          ↓          ↓
    [Add Type Embeddings + Ring ID Embeddings]
       ↓              ↓              ↓          ↓          ↓          ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │                    ISAB (Induced Attention)                        │
    │              O(nm) complexity via inducing points                  │
    └────────────────────────────────────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │                     SAB (Self-Attention)                           │
    │              All tokens attend to all tokens                       │
    └────────────────────────────────────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │                   PMA (Pooling)                                    │
    │              Pool to single context vector                         │
    └────────────────────────────────────────────────────────────────────┘
                                  ↓
                          Context Embedding
```

**Each block contains**:
1. Multi-head attention (with residual)
2. Layer normalization
3. Feed-forward network (with residual)
4. Layer normalization

**SSL Pretraining Task**: Mask receiver token, predict from context (tokens 1-8)."""))

    cells.append(new_code_cell("""# Demonstrate complete encoder on 9-token sequence
from stagebridge.context_model.local_niche_encoder import LocalNicheTransformerEncoder

# Create complete encoder
encoder = LocalNicheTransformerEncoder(
    receiver_dim=64,
    sender_feature_dim=32,
    hlca_dim=30,
    luca_dim=10,
    lr_summary_dim=20,
    stats_dim=10,
    model_dim=128,
    num_heads=4,
    num_layers=2,
    num_rings=4,
    dropout=0.0
)

print("=" * 70)
print("COMPLETE TRANSFORMER ENCODER ARCHITECTURE")
print("=" * 70)

# Create synthetic input (batch of 4 cells)
batch_size = 4
receiver_embeddings = torch.randn(batch_size, 64)
receiver_state_ids = torch.randint(0, 32, (batch_size,))
ring_compositions = torch.randn(batch_size, 4, 32)  # 4 rings
hlca_features = torch.randn(batch_size, 30)
luca_features = torch.randn(batch_size, 10)
lr_pathway_summary = torch.randn(batch_size, 20)
neighborhood_stats = torch.randn(batch_size, 10)

print("\\nInput Components:")
print(f"  - Receiver embeddings:    {receiver_embeddings.shape}")
print(f"  - Receiver state IDs:     {receiver_state_ids.shape}")
print(f"  - Ring compositions:      {ring_compositions.shape} (4 spatial rings)")
print(f"  - HLCA features:          {hlca_features.shape}")
print(f"  - LuCA features:          {luca_features.shape}")
print(f"  - Pathway summary:        {lr_pathway_summary.shape}")
print(f"  - Neighborhood stats:     {neighborhood_stats.shape}")

# Forward pass
with torch.no_grad():
    output = encoder(
        receiver_embeddings=receiver_embeddings,
        receiver_state_ids=receiver_state_ids,
        ring_compositions=ring_compositions,
        hlca_features=hlca_features,
        luca_features=luca_features,
        lr_pathway_summary=lr_pathway_summary,
        neighborhood_stats=neighborhood_stats,
        return_attention=True
    )

print(f"\\nOutput:")
print(f"  - Context embedding:      {output.neighborhood_embedding.shape}")
print(f"  - Token embeddings:       {output.token_embeddings.shape} (9 tokens)")
print(f"  - Attention weights:      {output.attention_weights.shape if output.attention_weights is not None else 'None'}")

# Visualize architecture
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

# Token sequence
ax1 = fig.add_subplot(gs[0, :])
token_labels = ['Receiver\\n(masked)', 'Ring 1\\n(inner)', 'Ring 2', 'Ring 3', 'Ring 4\\n(outer)',
                'HLCA\\n(ref)', 'LuCA\\n(ref)', 'Pathway\\n(LR)', 'Stats\\n(nbhd)']
colors = ['#FF6B6B', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#95E1D3', '#95E1D3', '#F9CA24', '#F9CA24']
bars = ax1.bar(range(9), [1]*9, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylim(0, 1.2)
ax1.set_xlim(-0.5, 8.5)
ax1.set_xticks(range(9))
ax1.set_xticklabels(token_labels, fontsize=10)
ax1.set_yticks([])
ax1.set_title('9-Token Sequence (StageBridge Architecture)', fontsize=14, fontweight='bold')
ax1.text(0, 1.1, 'Type 0', ha='center', fontsize=8)
for i in range(1, 5):
    ax1.text(i, 1.1, 'Type 1', ha='center', fontsize=8)
ax1.text(5, 1.1, 'Type 2', ha='center', fontsize=8)
ax1.text(6, 1.1, 'Type 3', ha='center', fontsize=8)
ax1.text(7, 1.1, 'Type 4', ha='center', fontsize=8)
ax1.text(8, 1.1, 'Type 5', ha='center', fontsize=8)

# Token embeddings (after encoding)
ax2 = fig.add_subplot(gs[1, 0])
with torch.no_grad():
    token_emb = output.token_embeddings[0].numpy()  # First sample
im = ax2.imshow(token_emb.T[:32, :], aspect='auto', cmap='viridis')
ax2.set_xlabel('Token Index')
ax2.set_ylabel('Embedding Dimension (first 32 shown)')
ax2.set_title('Token Embeddings After Encoding', fontsize=12)
ax2.set_xticks(range(9))
ax2.set_xticklabels(['R', 'R1', 'R2', 'R3', 'R4', 'H', 'L', 'P', 'S'])
plt.colorbar(im, ax=ax2, label='Value')

# Attention pattern (last layer)
ax3 = fig.add_subplot(gs[1, 1])
if output.attention_weights is not None:
    attn = output.attention_weights[0].mean(0).numpy()  # Average over heads
    sns.heatmap(attn, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['R', 'R1', 'R2', 'R3', 'R4', 'H', 'L', 'P', 'S'],
                yticklabels=['R', 'R1', 'R2', 'R3', 'R4', 'H', 'L', 'P', 'S'],
                ax=ax3, cbar_kws={'label': 'Attention'})
    ax3.set_title('Self-Attention Pattern (Final Layer)', fontsize=12)
    ax3.set_xlabel('Key')
    ax3.set_ylabel('Query')

# Context embedding
ax4 = fig.add_subplot(gs[2, :])
context_emb = output.neighborhood_embedding[0].numpy()
ax4.bar(range(len(context_emb)), context_emb, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Dimension')
ax4.set_ylabel('Value')
ax4.set_title('Final Context Embedding (Pooled from 9 tokens)', fontsize=12)
ax4.axhline(0, color='black', linewidth=0.5)
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Complete Transformer Encoder: Input → Tokens → Attention → Context',
             fontsize=15, fontweight='bold', y=0.995)
save_figure(fig, 'transformer_complete_architecture')
plt.show()

print("\\n" + "-" * 70)
print("ARCHITECTURAL FLOW")
print("-" * 70)
print("1. Tokenization:    7 input components → 9 tokens (with type embeddings)")
print("2. ISAB:            Efficient attention via inducing points")
print("3. SAB (x2):        Self-attention layers (all tokens interact)")
print("4. PMA:             Pool 9 tokens → 1 context vector")
print("5. LayerNorm:       Final normalization")
print("\\nSSL Task: Mask receiver (token 0), predict from context (tokens 1-8)")

print("\\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("This is a UNIFIED attention space:")
print("  - Spatial rings (tokens 1-4) attend to each other")
print("  - References (tokens 5-6) provide anchors")
print("  - Pathway/stats (tokens 7-8) add biological context")
print("  - Receiver (token 0) integrates all information")
print("\\nNOT a dual-branch architecture — all tokens in single self-attention!")
print("=" * 70)"""))

    # ========================================================================
    # CELL 9: Summary and Key Takeaways
    # ========================================================================
    cells.append(new_markdown_cell("""---

## Summary: Transformer Fundamentals

### Core Mechanisms

1. **Self-Attention**: Learns relationships between all pairs of tokens
   - Query: "What am I looking for?"
   - Key: "What do I represent?"
   - Value: "What information do I carry?"

2. **Scaled Dot-Product**: Division by $\\sqrt{d_k}$ prevents saturation

3. **Multi-Head Attention**: Parallel heads learn different relationship types

4. **Type Embeddings**: Inject semantic information (receiver vs ring vs reference)

5. **LayerNorm + Residuals**: Enable deep networks with stable gradients

6. **Feed-Forward Networks**: Process information after attention routes it

7. **ISAB/PMA**: Efficient set processing for variable-size inputs

### StageBridge Architecture

**9-Token Unified Attention**:
- Token 0: Receiver (masked during SSL)
- Tokens 1-4: Spatial rings (hierarchical neighborhoods)
- Tokens 5-6: HLCA/LuCA references (healthy/disease anchors)
- Tokens 7-8: Pathway/stats (biological/spatial context)

**Key Insight**: NOT a dual-branch architecture. All tokens participate in unified self-attention, allowing the model to learn rich relationships between spatial, reference, and biological features.

### Further Reading

- **Original Transformer**: Vaswani et al. "Attention is All You Need" (2017)
- **Set Transformer**: Lee et al. "Set Transformer" (2019)
- **Layer Norm**: Ba et al. "Layer Normalization" (2016)
- **GELU**: Hendrycks & Gimpel "Gaussian Error Linear Units" (2016)

---"""))

    return cells


def insert_cells_after(notebook_path, target_cell_id, new_cells):
    """Insert new cells after the specified cell ID."""

    # Read notebook
    nb = nbformat.read(notebook_path, as_version=4)

    # Find insertion point
    insertion_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.get('id') == target_cell_id:
            insertion_idx = i + 1
            break

    if insertion_idx is None:
        raise ValueError(f"Cell ID {target_cell_id} not found in notebook")

    # Insert new cells
    for i, cell in enumerate(new_cells):
        nb.cells.insert(insertion_idx + i, cell)

    # Write back
    nbformat.write(nb, notebook_path)

    print(f"✓ Inserted {len(new_cells)} educational cells after cell {target_cell_id}")
    print(f"✓ New total: {len(nb.cells)} cells")


def main():
    notebook_path = '/home/booka/projects/StageBridge/StageBridge_V1.ipynb'
    target_cell_id = '5fceee58'  # Cell 39 (after existing content)

    print("=" * 70)
    print("ADDING TRANSFORMER EDUCATIONAL CELLS")
    print("=" * 70)
    print(f"Notebook: {notebook_path}")
    print(f"Insertion point: After cell {target_cell_id}")
    print()

    # Create cells
    new_cells = create_educational_cells()
    print(f"Created {len(new_cells)} new cells")

    # Insert
    insert_cells_after(notebook_path, target_cell_id, new_cells)

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print("Educational cells added to notebook.")
    print("These cells teach transformer fundamentals using StageBridge as the example.")
    print()
    print("Sections added:")
    print("  1. Self-Attention Mechanism")
    print("  2. Scaled Dot-Product Attention")
    print("  3. Multi-Head Attention")
    print("  4. Positional/Type Embeddings")
    print("  5. Layer Normalization & Residual Connections")
    print("  6. Feed-Forward Networks")
    print("  7. Set Transformer Components (ISAB/PMA)")
    print("  8. Complete Encoder Architecture")
    print("  9. Summary and Key Takeaways")
    print()
    print("Next steps:")
    print("  1. Review the notebook")
    print("  2. Run cells to generate visualizations")
    print("  3. Adjust explanations as needed for your course")
    print("=" * 70)


if __name__ == '__main__':
    main()
