# StageBridge Architecture Cells - Documentation

**Date**: 2026-03-21
**Agent**: Notebook Assembly Agent
**Notebook**: `StageBridge_V1.ipynb`

## Summary

Added 16 educational cells to showcase **StageBridge's specific architectural innovations** for a deep learning course. These cells focus on what makes StageBridge unique, not generic transformer concepts.

## New Cells Added (Cells 66-81)

### Section Header
Overview of StageBridge's architectural innovations for domain-specific niche modeling.

### Cell 1: Receiver-Centered Prediction (AMICI-Inspired)
**Visual**: Comparison of receiver-centered vs sender-centered vs pairwise approaches
**Key Points**:
- Core innovation: Mask token 0 (receiver), predict from niche context (tokens 1-8)
- AMICI showed receiver-centered is more biologically interpretable
- Ideal for cross-sectional data (no temporal dynamics)
- Hypothesis: "Cell state is predictable from local niche context"

### Cell 2: The 9-Token Sequence Design
**Visual**: Detailed diagram of all 9 tokens with dimensions and roles
**Key Points**:
- Token 0: Receiver (masked during training)
- Tokens 1-4: Spatial rings at 25, 50, 100, 200 μm
- Token 5: HLCA reference (30D scANVI latent)
- Token 6: LuCA reference (10D scANVI latent)
- Token 7: Pathway activity (ligand-receptor)
- Token 8: Neighborhood stats
- All tokens in unified self-attention (NOT dual-branch)

### Cell 3: Spatial Ring Hierarchy
**Visual**: Concentric rings + Set Transformer pipeline
**Key Points**:
- 4 rings at biologically meaningful scales
- Ring 1 (0-25μm): Direct contact
- Ring 2 (25-50μm): Paracrine signaling
- Ring 3 (50-100μm): Tissue architecture
- Ring 4 (100-200μm): Lesion boundaries
- Set Transformer (ISAB→SAB→PMA) handles variable cell counts (5-50+)

### Cell 4: Dual Reference as Tokens (NOT Branches)
**Visual**: Side-by-side comparison of dual-branch vs token-based
**Key Points**:
- Traditional: Separate HLCA/LuCA encoders → late fusion
- StageBridge: HLCA and LuCA are tokens in the sequence
- Enables cross-attention between spatial context and references
- More interpretable: can analyze attention weights
- Biological question: "How does this cell compare to both healthy and cancer states?"

### Cell 5: Type Embeddings (Semantic, Not Positional)
**Visual**: Comparison of positional vs type embeddings
**Key Points**:
- 7 token types (not position-specific like standard transformers)
- Type 0: Receiver, Type 1: Spatial (rings), Type 2: HLCA, Type 3: LuCA, etc.
- Spatial tokens share type but differ by ring ID embedding
- Respects biological structure (rings are unordered sets)
- Formula: `token_embedding = content + type_embedding + (ring_embedding if spatial)`

### Cell 6: SSL Pretraining Objectives
**Visual**: Pie chart of loss weights + objective diagrams
**Key Points**:
- Masked receiver reconstruction: **70%** (PRIMARY)
- Ranking: 10% (positive/negative discrimination)
- Provider consistency: 10% (cross-view invariance)
- Coordinate corruption: 5% (spatial awareness)
- Group relation: 5% (biological grouping)
- 70% weight reflects core novelty: receiver prediction from niche

### Cell 7: Attention Flow Analysis
**Visual**: 9x9 attention matrix heatmap + receiver attention breakdown
**Key Points**:
- Receiver attends most to nearby rings (hierarchical distance decay)
- Rings attend to each other (hierarchical spatial reasoning)
- References attend to spatial context (grounded in observed data)
- All tokens participate in unified attention (rich cross-modal reasoning)
- Can analyze which tokens the model relies on

### Cell 8: Summary of Architectural Innovations
**Content**: Comprehensive summary of all 7 innovations
**Key Points**:
- Every design choice motivated by biological requirements and data constraints
- Cross-sectional snapshots → receiver-centered prediction
- Variable cell counts → Set Transformer
- Spatial hierarchy → 4 rings at meaningful scales
- Multi-modal context → tokenized references
- Interpretability → unified attention with analyzable weights
- **This is domain-specific architecture, not generic transformers**

## Design Philosophy

All cells follow this pattern:
1. **Motivation**: Why this design choice?
2. **Comparison**: What are the alternatives?
3. **Visualization**: Diagram or code demonstration
4. **Interpretation**: What does this mean biologically?
5. **Summary**: Key takeaways

## Target Audience

Deep learning course students learning about:
- Domain-specific transformer architectures
- Biological niche modeling
- Cross-sectional inference from spatial data
- Self-supervised learning for biological applications

## Technical Details

### Notebook Statistics
- **Original cells**: 69
- **New cells added**: 16
- **Total cells**: 85
- **Insertion point**: Cell 66 (before "FINAL SUMMARY")

### Script Used
`add_stagebridge_architecture_cells.py` - Can be reused as template for future cell additions.

### Key Architectural Files Referenced
- `stagebridge/context_model/local_niche_encoder.py`: 9-token tokenizer
- `stagebridge/context_model/set_encoder.py`: ISAB, SAB, PMA
- `stagebridge/transition_model/relational_pretraining.py`: SSL objectives
- `stagebridge/models/dual_reference.py`: Reference integration

## What Makes This Different from Generic Transformer Tutorials

These cells focus on **StageBridge-SPECIFIC innovations**:

**NOT covered** (generic concepts):
- What is self-attention?
- How does layer normalization work?
- What are residual connections?

**COVERED** (StageBridge-specific):
- Why receiver-centered instead of sender-centered?
- Why 9 tokens with this specific composition?
- Why 4 rings at 25/50/100/200 μm?
- Why references as tokens instead of dual-branch?
- Why type embeddings instead of positional?
- Why 70% weight on receiver reconstruction?
- What attention patterns does the model learn?

## Validation

All cells include:
- Working visualizations (matplotlib/seaborn)
- Executable code examples
- Biological interpretations
- Clear explanations of design choices

The cells are designed to:
- Run in a clean environment
- Use standard libraries (numpy, matplotlib, seaborn)
- Produce publication-quality figures
- Be pedagogically clear

## Next Steps

The notebook now has:
1. Generic transformer fundamentals (cells 40-57)
2. Transformer internals (cells 58-65)
3. **StageBridge-specific innovations (cells 66-81)** ← NEW
4. Final summary (cells 82+)

Students can progress from basic transformers → StageBridge's unique adaptations for biological niche modeling.

## Files Created

1. `add_stagebridge_architecture_cells.py` - Script to add the cells
2. `STAGEBRIDGE_ARCHITECTURE_CELLS_ADDED.md` - This documentation
3. Updated `StageBridge_V1.ipynb` - Notebook with new cells
4. Updated `.claude/agent-memory/notebook-assembly/MEMORY.md` - Agent memory

## Absolute File Paths

- Notebook: `/home/booka/projects/StageBridge/StageBridge_V1.ipynb`
- Script: `/home/booka/projects/StageBridge/add_stagebridge_architecture_cells.py`
- Documentation: `/home/booka/projects/StageBridge/STAGEBRIDGE_ARCHITECTURE_CELLS_ADDED.md`
