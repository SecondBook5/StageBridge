# Transformer Educational Cells: Documentation

**Purpose**: Teach transformer fundamentals using StageBridge as a concrete example for deep learning courses.

**Location**: `StageBridge_V1.ipynb`, cells 40-57 (18 cells total)

**Target Audience**: Deep learning students and practitioners learning transformer architectures

---

## Overview

These cells provide a comprehensive educational walkthrough of transformer architecture components, using StageBridge's actual implementation as working examples. Students learn by examining real model weights, attention patterns, and intermediate computations.

### Key Educational Principles

1. **Learn by doing**: Extract actual weights and compute step-by-step
2. **Visualize everything**: Attention patterns, embeddings, distributions
3. **Real architecture**: Use StageBridge components, not toy examples
4. **Progressive complexity**: Start simple (attention), build to complete encoder

---

## Cell Structure

### Section Header (Cell 40)
- Introduction to the educational section
- Learning objectives (8 key concepts)
- Prerequisites and target audience

### 1. Self-Attention Mechanism (Cells 41-42)

**Mathematical Formulation**:
- Query, Key, Value projections: $Q = XW_Q$, $K = XW_K$, $V = XW_V$
- Attention scores: $\text{scores} = \frac{QK^T}{\sqrt{d_k}}$
- Softmax normalization: $\text{attention\_weights} = \text{softmax}(\text{scores})$
- Weighted combination: $\text{output} = \text{attention\_weights} \cdot V$

**Code Demonstrations**:
- Extract SAB (Self-Attention Block) from StageBridge
- Create 9-token sequence (receiver + 4 rings + HLCA + LuCA + pathway + stats)
- Compute and visualize attention weights
- Show that rows sum to 1.0 (softmax property)
- Compare attention patterns across multiple heads

**Visualizations**:
- Attention heatmaps (2 heads side-by-side)
- Token labels match StageBridge architecture
- Color-coded by attention strength

**Key Insight**: Each token (row) attends to all tokens (columns) with learned weights. Different heads capture different relationships.

---

### 2. Scaled Dot-Product Attention (Cells 43-44)

**Why Scale by $\sqrt{d_k}$?**
- High-dimensional dot products → large magnitudes
- Large magnitudes → softmax saturation
- Saturation → vanishing gradients
- Scaling keeps values in reasonable range

**Code Demonstrations**:
- Simulate Q, K for dimensions [16, 64, 256, 1024]
- Compare unscaled vs scaled attention distributions
- Measure entropy (saturation indicator)
- Show how saturation worsens at high dimensions

**Visualizations**:
- Bar charts: unscaled vs scaled attention for each dimension
- Demonstrates saturation effect at high dimensions
- Entropy measurements show distribution quality

**Key Insight**: Without scaling, high-dimensional dot products saturate softmax. Scaling by $\sqrt{d_k}$ keeps attention well-behaved across dimensions.

---

### 3. Multi-Head Attention (Cells 45-46)

**Why Multiple Heads?**
- Different heads specialize in different relationship types:
  - Local interactions (adjacent tokens)
  - Global context (receiver ↔ references)
  - Hierarchical structure (ring scales)
  - Cross-modality (spatial ↔ genomic)

**Architecture**:
- Split $d_{\text{model}}$ into $h$ heads: $d_k = d_{\text{model}} / h$
- Each head computes attention independently
- Concatenate and project: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{heads}) W^O$

**Code Demonstrations**:
- Create SAB with 8 heads
- Structured input (receiver, rings, references have distinct patterns)
- Analyze head specialization:
  - Receiver → Rings attention
  - Receiver → References attention
  - Rings ↔ Rings attention
  - References → Receiver attention
- Visualize all 8 head patterns

**Visualizations**:
- 2×4 grid of attention heatmaps (one per head)
- Each head shows different specialization
- Quantitative analysis of attention patterns

**Key Insight**: Different heads specialize in different relationships. Parallelism captures multiple relationship types simultaneously.

---

### 4. Positional and Type Embeddings (Cells 47-48)

**Problem**: Self-attention is permutation-invariant!
- Can't distinguish token position or semantic type

**Solutions**:
1. **Standard Transformers**: Sinusoidal positional encoding
   - $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$
   - $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$

2. **StageBridge**: Learned type embeddings
   - Type 0: Receiver
   - Type 1: Spatial ring (+ ring ID embedding)
   - Type 2: HLCA reference
   - Type 3: LuCA reference
   - Type 4: Pathway summary
   - Type 5: Neighborhood statistics
   - Type 6: Atlas contrast (optional)

**Code Demonstrations**:
- Extract LocalNicheTokenizer
- Visualize type embedding matrix (7 types × model_dim)
- Visualize ring ID embeddings (4 rings × model_dim)
- Compute pairwise cosine distances between type embeddings
- Show that similar types have smaller distances

**Visualizations**:
- Type embedding heatmap (first 32 dimensions)
- Ring ID embedding heatmap (hierarchical spatial scales)
- Type similarity matrix (cosine distance)

**Key Insight**: Type embeddings are ADDED to token features, injecting semantic information. The model learns which types should be similar (e.g., HLCA and LuCA both references).

---

### 5. Layer Normalization and Residual Connections (Cells 49-50)

**Two Critical Components**:

1. **Residual Connections** (Skip Connections):
   - $\text{output} = \text{input} + \text{sublayer}(\text{input})$
   - Enables gradient flow through deep networks
   - Model can learn identity mapping

2. **Layer Normalization**:
   - Normalize across feature dimension: $\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$
   - Stabilizes activations (mean ≈ 0, std ≈ 1)

**Pre-Norm vs Post-Norm**:
- Pre-Norm (StageBridge): Normalize before sublayer → more stable
- Post-Norm (original): Normalize after sublayer → slightly better final performance

**Code Demonstrations**:
- Track intermediate values through SAB:
  1. Input
  2. After attention
  3. After residual (x + attn)
  4. After LayerNorm
  5. After feed-forward
  6. After residual (x + ff)
  7. After LayerNorm
- Visualize distributions at each stage
- Show gradient flow analysis

**Visualizations**:
- 2×3 grid of histograms (6 stages)
- Shows how LayerNorm standardizes distributions
- Demonstrates mean ≈ 0, std ≈ 1 after normalization

**Key Insight**: LayerNorm stabilizes activations → easier optimization. Residuals enable gradient flow → trainable depth.

---

### 6. Feed-Forward Networks (Cells 51-52)

**Architecture**:
- Expand: $d_{\text{model}} \rightarrow d_{\text{ff}}$ (typically $4 \times d_{\text{model}}$)
- Activation: GELU (Gaussian Error Linear Unit)
- Contract: $d_{\text{ff}} \rightarrow d_{\text{model}}$

**Formula**: $\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$

**Purpose**:
- Attention = information routing (where to look)
- FFN = information processing (what to do with it)
- Adds nonlinear transformations and capacity

**GELU vs ReLU**:
- ReLU: $\text{ReLU}(x) = \max(0, x)$ (hard cutoff)
- GELU: $\text{GELU}(x) \approx x \cdot \Phi(x)$ (smooth, probabilistic)
- GELU has smoother gradients

**Code Demonstrations**:
- Extract FeedForwardBlock from StageBridge
- Compare GELU vs ReLU activation functions
- Track token transformation through FFN
- Visualize hidden layer activations (512 dimensions)
- Show token norm changes (before vs after)

**Visualizations**:
- GELU vs ReLU comparison plot
- Token norm comparison (bar chart)
- Hidden activation distribution (histogram)

**Key Insight**: Attention routes information (decides what to attend to). FFN processes information (nonlinear transformations). 4× expansion increases model capacity.

---

### 7. Set Transformer Components: ISAB and PMA (Cells 53-54)

**Problem**: Standard attention is $O(n^2)$ in sequence length.

**Solutions**:

1. **ISAB (Induced Set Attention Block)**:
   - Uses $m$ learnable inducing points
   - Reduces complexity from $O(n^2)$ to $O(nm)$
   - Two-step attention:
     1. Inducing points attend to input: $H = \text{Attention}(I, X, X)$
     2. Input attends to inducing points: $Y = \text{Attention}(X, H, H)$

2. **PMA (Pooling by Multihead Attention)**:
   - Uses $k$ learnable seed vectors
   - Pools variable-size set to fixed output: $\text{PMA}(X) = \text{Attention}(S, X, X)$
   - Enables fixed-size output from variable-size input

**In StageBridge**: Each spatial ring uses ISAB → SAB → PMA to aggregate variable numbers of cells.

**Code Demonstrations**:
- Create ISAB and PMA modules
- Process variable-size inputs (50, 100, 200 cells)
- Show that output is always fixed-size
- Visualize inducing points and seed vectors
- Plot complexity comparison: $O(n^2)$ vs $O(nm)$
- Test permutation invariance

**Visualizations**:
- Inducing points visualization (16 learnable vectors)
- PMA seed vectors (1 learnable vector)
- Complexity curves: quadratic vs linear
- Demonstrates efficiency at scale

**Key Insight**: ISAB provides efficient attention via inducing points. PMA pools variable-size sets to fixed output. Both are permutation-invariant. Perfect for spatial rings with varying cell counts.

---

### 8. Complete Transformer Encoder Architecture (Cells 55-56)

**StageBridge 9-Token Architecture**:

```
Token Sequence:
┌─────────────┬──────────────────┬──────────┬──────────┬─────────┬────────┐
│  Receiver   │  Ring 1-4        │  HLCA    │  LuCA    │ Pathway │ Stats  │
│  (masked)   │  (spatial niche) │  (ref)   │  (ref)   │ (LR)    │ (nbhd) │
└─────────────┴──────────────────┴──────────┴──────────┴─────────┴────────┘
       ↓              ↓              ↓          ↓          ↓          ↓
    [Add Type Embeddings + Ring ID Embeddings]
       ↓
    [ISAB - Induced Attention (O(nm) complexity)]
       ↓
    [SAB - Self-Attention (all tokens interact)]
       ↓
    [PMA - Pool to single context vector]
       ↓
    Context Embedding
```

**Each block contains**:
1. Multi-head attention (with residual)
2. Layer normalization
3. Feed-forward network (with residual)
4. Layer normalization

**SSL Pretraining Task**: Mask receiver token, predict from context.

**Code Demonstrations**:
- Create complete LocalNicheTransformerEncoder
- Prepare all 7 input components
- Forward pass with attention tracking
- Visualize complete architecture flow

**Visualizations**:
- 9-token sequence diagram (color-coded by type)
- Token embeddings heatmap (after encoding)
- Self-attention pattern (averaged over heads)
- Final context embedding (bar chart)
- Complete 4-panel architectural overview

**Key Insight**: This is a UNIFIED attention space. All tokens participate in single self-attention. NOT a dual-branch architecture.

---

### 9. Summary and Key Takeaways (Cell 57)

**Core Mechanisms Recap**:
1. Self-attention learns relationships
2. Scaled dot-product prevents saturation
3. Multi-head captures multiple patterns
4. Type embeddings inject semantic info
5. LayerNorm + residuals enable depth
6. FFN processes after attention routes
7. ISAB/PMA handle variable-size efficiently

**StageBridge Architecture Summary**:
- 9-token unified attention
- Receiver (masked) + 4 rings + 2 references + 2 summaries
- NOT dual-branch: all tokens in shared attention space

**Further Reading**:
- Original Transformer (Vaswani et al., 2017)
- Set Transformer (Lee et al., 2019)
- Layer Normalization (Ba et al., 2016)
- GELU (Hendrycks & Gimpel, 2016)

---

## Pedagogical Design

### Learning Progression

1. **Foundation** (Cells 41-44): Core attention mechanism
2. **Parallelism** (Cells 45-46): Multi-head attention
3. **Structure** (Cells 47-48): Embeddings for position/type
4. **Stability** (Cells 49-50): Normalization and residuals
5. **Capacity** (Cells 51-52): Feed-forward networks
6. **Efficiency** (Cells 53-54): Set Transformer components
7. **Integration** (Cells 55-56): Complete architecture
8. **Synthesis** (Cell 57): Summary and takeaways

### Educational Features

- **Real Implementation**: Uses actual StageBridge code, not toy examples
- **Step-by-Step**: Every mathematical operation shown in code
- **Visual Learning**: Heatmaps, bar charts, distributions for every concept
- **Quantitative Analysis**: Print statistics, shapes, parameter counts
- **Conceptual Insight**: "Key Insight" boxes summarize main points
- **Progressive Build**: Each concept builds on previous ones
- **Working Example**: 9-token architecture ties everything together

### Assessment Opportunities

After completing these cells, students should be able to:

1. Explain how self-attention computes relationships
2. Derive why scaling by $\sqrt{d_k}$ is necessary
3. Describe what different attention heads learn
4. Implement type/positional embeddings
5. Explain the purpose of LayerNorm and residuals
6. Distinguish attention (routing) from FFN (processing)
7. Compare ISAB complexity to standard attention
8. Diagram a complete transformer encoder
9. Apply transformer concepts to new problem domains

---

## Usage in Courses

### Deep Learning Fundamentals Course

**Week 8: Attention Mechanisms**
- Cells 41-42: Self-attention mechanics
- Cells 43-44: Scaling and normalization
- Assignment: Implement scaled dot-product attention

**Week 9: Transformer Architecture**
- Cells 45-46: Multi-head attention
- Cells 47-50: Structural components
- Cells 51-52: Feed-forward networks
- Assignment: Build complete transformer block

**Week 10: Advanced Architectures**
- Cells 53-54: Set Transformers
- Cells 55-56: Complete encoder
- Project: Apply to student's domain

### Computational Biology Course

**Module: Deep Learning for Single-Cell**
- Focus on biological interpretation:
  - Why 9 tokens? (Biological features)
  - What do heads learn? (Spatial vs genomic)
  - How are references used? (Normal vs disease)
- Assignment: Design token sequence for new biological problem

### Advanced ML Course

**Topic: Efficient Transformers**
- Deep dive into ISAB/PMA (cells 53-54)
- Complexity analysis and trade-offs
- Compare to other efficient attention methods (Linformer, Performer, etc.)
- Assignment: Implement and benchmark efficient attention variant

---

## Customization

### For Different Audiences

**Undergraduate Level**:
- Focus on cells 41-46 (core attention)
- Skip mathematical derivations
- Emphasize visualizations
- Simplified assignments

**Graduate Level**:
- Cover all cells 41-57
- Add mathematical proofs
- Compare to alternative architectures
- Research-oriented projects

**Industry Practitioners**:
- Focus on cells 53-56 (practical components)
- Emphasize efficiency and scalability
- Production deployment considerations
- Case studies and best practices

### For Different Domains

**Computer Vision**:
- Compare to Vision Transformer (ViT)
- Patch embeddings vs token embeddings
- Spatial attention patterns

**Natural Language Processing**:
- Compare to BERT/GPT architectures
- Positional encodings vs type embeddings
- Causal vs bidirectional attention

**Computational Biology**:
- Single-cell representation learning
- Multi-modal integration (spatial + genomic)
- Biological interpretability

---

## Implementation Notes

### Dependencies

All cells use standard StageBridge imports:
```python
from stagebridge.context_model.set_encoder import SAB, ISAB, PMA, FeedForwardBlock
from stagebridge.context_model.local_niche_encoder import LocalNicheTokenizer, LocalNicheTransformerEncoder
```

### Data Requirements

- **Synthetic data**: All demonstrations use `torch.randn()` for reproducibility
- **No external data**: Cells run standalone (perfect for teaching)
- **Configurable sizes**: Easy to adjust dimensions for exploration

### Execution Time

- Each cell: 1-5 seconds
- Total section: ~30 seconds
- Suitable for live demos and workshops

### Visualization

Uses existing StageBridge plotting utilities:
- `save_figure()` for publication-quality output
- Seaborn/Matplotlib for heatmaps and charts
- Consistent styling with research notebook

---

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure StageBridge is installed and in Python path
2. **Shape mismatches**: Check batch dimensions (some modules expect 3D, others 2D)
3. **Visualization**: Ensure `save_figure()` function is defined in earlier cells
4. **CUDA errors**: All demos use CPU (no GPU required)

### Modifications

To adjust for different model dimensions:
```python
# Current: dim=128, num_heads=4
# For larger models:
dim = 512
num_heads = 8
num_inducing_points = 32
```

To change token sequence:
```python
# Current: 9 tokens
# For custom architecture:
num_tokens = 12  # Your sequence length
token_labels = ['Custom', 'Token', 'Labels', ...]
```

---

## Future Enhancements

Potential additions for future versions:

1. **Masked Self-Attention** (causal attention for autoregressive models)
2. **Cross-Attention** (encoder-decoder interaction)
3. **Sparse Attention Patterns** (block-sparse, strided, etc.)
4. **Attention Variants** (Linformer, Performer, Flash Attention)
5. **Training Dynamics** (learning curves, gradient flow)
6. **Interpretability** (attention rollout, gradient-based attribution)
7. **Comparison to CNNs/RNNs** (when to use transformers)
8. **Production Deployment** (quantization, pruning, distillation)

---

## References

### Original Papers

1. Vaswani et al. "Attention is All You Need" (NeurIPS 2017)
2. Lee et al. "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks" (ICML 2019)
3. Ba et al. "Layer Normalization" (arXiv 2016)
4. Hendrycks & Gimpel "Gaussian Error Linear Units (GELUs)" (arXiv 2016)
5. He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)

### Educational Resources

- **Stanford CS224N**: Natural Language Processing with Deep Learning
- **MIT 6.S191**: Introduction to Deep Learning
- **DeepLearning.AI**: Attention Models in NLP
- **Transformers from Scratch**: Blog series by Peter Bloem

### StageBridge Papers

- AMICI: Receiver-centered attention for cell-cell interaction
- OSDR: Tissue dynamics from spatial snapshots
- LuCA: Lung Cancer reference atlas

---

## Acknowledgments

These educational cells were designed to complement StageBridge's research goals with pedagogical value. They demonstrate that real research architectures can serve as excellent teaching examples when properly explained.

**Design principles**:
- Use real implementations, not simplified toys
- Provide both mathematical formulation and code
- Visualize everything for intuitive understanding
- Build progressively from simple to complex
- Connect to biological application throughout

This makes transformer concepts accessible to students while showing how they're applied in cutting-edge computational biology research.

---

**Last Updated**: 2025-03-21
**Notebook Version**: StageBridge_V1.ipynb (cells 40-57)
**Contact**: See StageBridge documentation for questions
