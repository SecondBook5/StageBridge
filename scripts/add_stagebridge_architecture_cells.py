"""
Add StageBridge-specific architectural innovation cells to the notebook.

This script adds educational cells that showcase STAGEBRIDGE'S UNIQUE design choices,
not generic transformer concepts. Focus on:
1. Receiver-centered prediction (AMICI-inspired)
2. The specific 9-token sequence design
3. Spatial ring hierarchy (4 rings at specific radii)
4. Dual reference as tokens (NOT branches)
5. Type embeddings (our positional encoding)
6. Set Transformer for ring aggregation
7. SSL pretraining objectives (70% receiver reconstruction)
8. Attention flow analysis
"""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

# Load the notebook
notebook_path = "/home/booka/projects/StageBridge/StageBridge_V1.ipynb"
nb = nbformat.read(notebook_path, as_version=4)

# Find the insertion point (after the last transformer cell, before final summary)
# Look for the "FINAL SUMMARY" markdown cell
insertion_idx = None
for i, cell in enumerate(nb.cells):
    if cell['cell_type'] == 'markdown' and 'FINAL SUMMARY' in cell['source']:
        insertion_idx = i
        break

if insertion_idx is None:
    # Fall back to end of notebook
    insertion_idx = len(nb.cells)

print(f"Inserting StageBridge architecture cells at index {insertion_idx}")

# Create the new cells
new_cells = []

# ==============================================================================
# SECTION HEADER
# ==============================================================================
new_cells.append(new_markdown_cell("""---

## STAGEBRIDGE ARCHITECTURAL INNOVATIONS

**What Makes StageBridge Unique?**

This section demonstrates StageBridge's specific design choices for biological niche modeling:

1. **Receiver-Centered Prediction** (AMICI-inspired) - Why mask the receiver?
2. **The 9-Token Sequence** - What does each token represent?
3. **Spatial Ring Hierarchy** - Why 4 rings at these specific radii?
4. **Dual Reference as Tokens** - HLCA/LuCA are tokens, NOT separate branches
5. **Type Embeddings** - Our positional encoding strategy
6. **Set Transformer for Rings** - Variable-size aggregation
7. **SSL Pretraining** - 70% weight on receiver reconstruction
8. **Attention Flow** - What does the model attend to?

**Target audience**: Deep learning course students learning about domain-specific transformer architectures.
"""))

# ==============================================================================
# CELL 1: RECEIVER-CENTERED PREDICTION
# ==============================================================================
new_cells.append(new_markdown_cell("""### 1. Receiver-Centered Prediction (AMICI-Inspired)

**The Core Innovation**: Predict the receiver cell state from its local niche context.

**Why receiver-centered?**
- **AMICI**: Showed receiver-centered attention is more biologically interpretable than sender-centered
- **Cross-sectional inference**: We only have snapshots, not time-series
- **Key hypothesis**: A cell's progression state is predictable from its neighborhood

**The SSL task**: Mask token 0 (receiver), predict it from tokens 1-8 (niche context).

This is fundamentally different from:
- **Sender-centered**: Predict how a cell affects others (requires temporal data)
- **Pairwise**: Predict cell-cell interactions (doesn't capture collective niche effects)
"""))

new_cells.append(new_code_cell("""# ============================================================================
# Demonstrate receiver-centered masking
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Receiver-centered (our approach)
ax = axes[0]
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Draw receiver (masked)
receiver = Circle((0, 0), 0.15, color='red', alpha=0.3, linestyle='--', linewidth=2, fill=False)
ax.add_patch(receiver)
ax.text(0, 0, '?', fontsize=20, ha='center', va='center', color='red', weight='bold')

# Draw niche cells (context)
niche_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
for i, angle in enumerate(niche_angles):
    x, y = 0.6 * np.cos(angle), 0.6 * np.sin(angle)
    cell = Circle((x, y), 0.12, color='green', alpha=0.7)
    ax.add_patch(cell)

# Draw arrow from niche to receiver
ax.annotate('', xy=(0, 0), xytext=(0.6, 0),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax.text(0.3, 0.15, 'Predict', fontsize=11, color='blue', weight='bold')

ax.set_title('A. Receiver-Centered (StageBridge)\\nMask receiver, predict from niche',
             fontsize=13, weight='bold')

# Panel B: Sender-centered (alternative)
ax = axes[1]
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Draw sender
sender = Circle((0, 0), 0.15, color='purple', alpha=0.8)
ax.add_patch(sender)
ax.text(0, 0, 'S', fontsize=14, ha='center', va='center', color='white', weight='bold')

# Draw receiver cells (masked)
for i, angle in enumerate(niche_angles[:4]):
    x, y = 0.6 * np.cos(angle), 0.6 * np.sin(angle)
    cell = Circle((x, y), 0.12, color='red', alpha=0.3, linestyle='--', linewidth=2, fill=False)
    ax.add_patch(cell)
    ax.text(x, y, '?', fontsize=12, ha='center', va='center', color='red', weight='bold')

# Draw arrow from sender to receivers
ax.annotate('', xy=(0.6, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax.set_title('B. Sender-Centered (Alternative)\\nPredict sender effect on receivers',
             fontsize=13, weight='bold')

# Panel C: Pairwise (alternative)
ax = axes[2]
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Draw cell pairs
cell1 = Circle((-0.3, 0), 0.12, color='blue', alpha=0.8)
cell2 = Circle((0.3, 0), 0.12, color='orange', alpha=0.8)
ax.add_patch(cell1)
ax.add_patch(cell2)

# Bidirectional arrow
ax.annotate('', xy=(0.3, 0), xytext=(-0.3, 0),
            arrowprops=dict(arrowstyle='<->', lw=2, color='gray'))
ax.text(0, 0.25, 'Interaction?', fontsize=11, ha='center', weight='bold')

ax.set_title('C. Pairwise (Alternative)\\nPredict cell-cell interactions',
             fontsize=13, weight='bold')

plt.tight_layout()
plt.savefig('receiver_centered_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nKey Insight:")
print("=============")
print("Receiver-centered prediction is ideal for cross-sectional data because:")
print("  1. We can observe the niche (tokens 1-8) directly")
print("  2. The receiver's state is the unknown we want to predict")
print("  3. Captures collective niche effects, not just pairwise interactions")
print("  4. Biologically interpretable: 'What should this cell become given its neighbors?'")
"""))

# ==============================================================================
# CELL 2: THE 9-TOKEN SEQUENCE
# ==============================================================================
new_cells.append(new_markdown_cell("""### 2. The 9-Token Sequence Design

**Why these specific 9 tokens?**

Each token serves a specific biological purpose:

| Token | Type | Purpose | Dimension |
|-------|------|---------|-----------|
| 0 | Receiver | The cell we're predicting | Varies (embedding dim) |
| 1-4 | Spatial Rings | Hierarchical neighborhood (25μm, 50μm, 100μm, 200μm) | Cell-type composition |
| 5 | HLCA | Healthy lung reference anchor | 30D (from scANVI model) |
| 6 | LuCA | Cancer reference anchor | 10D (from scANVI model) |
| 7 | Pathway | Ligand-receptor activity summary | Pathway activity scores |
| 8 | Stats | Neighborhood statistics | Density, diversity, etc. |

**Key design principle**: All 9 tokens participate in **unified self-attention**.

This is NOT a dual-branch model. HLCA and LuCA are tokens in the sequence, allowing cross-attention between spatial context and references.
"""))

new_cells.append(new_code_cell("""# ============================================================================
# Visualize the 9-token sequence with actual dimensions
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Token definitions with actual StageBridge dimensions
tokens = [
    {"name": "Token 0\\nReceiver", "type": "receiver", "color": "#FF6B6B",
     "desc": "Cell to predict", "dim": "D (model dim)", "example": "128D"},
    {"name": "Token 1\\nRing 1", "type": "spatial", "color": "#4ECDC4",
     "desc": "0-25μm radius", "dim": "4 cell types", "example": "Epithelial, Stromal, Immune, Vasc"},
    {"name": "Token 2\\nRing 2", "type": "spatial", "color": "#4ECDC4",
     "desc": "25-50μm radius", "dim": "4 cell types", "example": "Composition vector"},
    {"name": "Token 3\\nRing 3", "type": "spatial", "color": "#4ECDC4",
     "desc": "50-100μm radius", "dim": "4 cell types", "example": "Composition vector"},
    {"name": "Token 4\\nRing 4", "type": "spatial", "color": "#4ECDC4",
     "desc": "100-200μm radius", "dim": "4 cell types", "example": "Composition vector"},
    {"name": "Token 5\\nHLCA", "type": "reference", "color": "#95E1D3",
     "desc": "Healthy anchor", "dim": "30D", "example": "scANVI latent (HLCA)"},
    {"name": "Token 6\\nLuCA", "type": "reference", "color": "#F38181",
     "desc": "Cancer anchor", "dim": "10D", "example": "scANVI latent (LuCA)"},
    {"name": "Token 7\\nPathway", "type": "pathway", "color": "#AA96DA",
     "desc": "LR activity", "dim": "Varies", "example": "IL1B-IL1R1, VEGF, etc."},
    {"name": "Token 8\\nStats", "type": "stats", "color": "#FCBAD3",
     "desc": "Neighborhood", "dim": "Varies", "example": "Density, diversity, spatial"},
]

# Draw tokens as boxes
y_pos = 7
for i, token in enumerate(tokens):
    x_pos = i * 1.0 + 0.5

    # Token box
    box = FancyBboxPatch(
        (x_pos - 0.4, y_pos - 0.3), 0.8, 0.6,
        boxstyle="round,pad=0.05",
        edgecolor='black', facecolor=token['color'],
        linewidth=2, alpha=0.8
    )
    ax.add_patch(box)

    # Token name
    ax.text(x_pos, y_pos, token['name'],
            ha='center', va='center', fontsize=9, weight='bold')

    # Description below
    ax.text(x_pos, y_pos - 1.0, token['desc'],
            ha='center', va='top', fontsize=7, style='italic')

    # Dimension info
    ax.text(x_pos, y_pos - 1.5, f"Dim: {token['dim']}",
            ha='center', va='top', fontsize=7, weight='bold', color='darkblue')

    # Example
    ax.text(x_pos, y_pos - 2.0, token['example'],
            ha='center', va='top', fontsize=6, color='gray')

# Add unified attention indicator
ax.text(5, 9, 'All 9 tokens → Unified Self-Attention',
        ha='center', fontsize=14, weight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Add arrows showing attention flow
arrow = FancyArrowPatch((1, 6.5), (8, 6.5),
                       arrowstyle='<->', mutation_scale=20,
                       linewidth=2, color='blue', alpha=0.5)
ax.add_patch(arrow)
ax.text(4.5, 6.2, 'Cross-attention between all tokens',
        ha='center', fontsize=9, color='blue', style='italic')

# Add type embedding legend
legend_elements = [
    mpatches.Patch(color='#FF6B6B', label='Type 0: Receiver (masked during training)'),
    mpatches.Patch(color='#4ECDC4', label='Type 1: Spatial (4 rings)'),
    mpatches.Patch(color='#95E1D3', label='Type 2: HLCA reference'),
    mpatches.Patch(color='#F38181', label='Type 3: LuCA reference'),
    mpatches.Patch(color='#AA96DA', label='Type 4: Pathway'),
    mpatches.Patch(color='#FCBAD3', label='Type 5: Stats'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
          title='Token Type Embeddings', framealpha=0.9)

ax.set_title('StageBridge 9-Token Sequence Architecture', fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('9_token_sequence.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nKey Design Choices:")
print("===================")
print("1. Token 0 is MASKED during SSL pretraining → forces model to use niche context")
print("2. Tokens 1-4 capture spatial hierarchy at biologically meaningful scales")
print("3. Tokens 5-6 are reference anchors (NOT separate branches!)")
print("4. Tokens 7-8 provide additional biological context (pathway + spatial stats)")
print("5. All tokens use type embeddings to signal their role")
print("\\nWhy 9 tokens? This is the minimal set that captures:")
print("  - Receiver state (1 token)")
print("  - Hierarchical spatial context (4 tokens)")
print("  - Dual reference geometry (2 tokens)")
print("  - Biological context (2 tokens)")
"""))

# ==============================================================================
# CELL 3: SPATIAL RING HIERARCHY
# ==============================================================================
new_cells.append(new_markdown_cell("""### 3. Spatial Ring Hierarchy

**Why 4 concentric rings at 25μm, 50μm, 100μm, 200μm?**

These radii are chosen based on **biological interaction scales** in lung tissue:

- **Ring 1 (0-25μm)**: Direct cell-cell contact, immediate microenvironment
- **Ring 2 (25-50μm)**: Local paracrine signaling range
- **Ring 3 (50-100μm)**: Intermediate niche, captures local tissue architecture
- **Ring 4 (100-200μm)**: Broader tissue context, captures lesion boundaries

**Key challenge**: Each ring contains a **variable number of cells** (5-50+).

**Solution**: Use Set Transformer (ISAB → SAB → PMA) to aggregate each ring into a fixed-size token.
"""))

new_cells.append(new_code_cell("""# ============================================================================
# Visualize spatial ring hierarchy
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Concentric rings with cells
ax = axes[0]
ax.set_xlim(-250, 250)
ax.set_ylim(-250, 250)
ax.set_aspect('equal')
ax.axis('off')

# Define ring radii (in micrometers)
radii = [25, 50, 100, 200]
colors = ['#E8F4F8', '#B8E0E8', '#88CCD8', '#58B8C8']
ring_names = ['Ring 1', 'Ring 2', 'Ring 3', 'Ring 4']

# Draw concentric rings
for i, (r, color, name) in enumerate(zip(radii, colors, ring_names)):
    circle = Circle((0, 0), r, color=color, alpha=0.5, linewidth=2, edgecolor='black')
    ax.add_patch(circle)

    # Add ring label
    angle = 45 + i * 15
    x_label = r * 0.7 * np.cos(np.radians(angle))
    y_label = r * 0.7 * np.sin(np.radians(angle))
    ax.text(x_label, y_label, name, fontsize=11, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Draw receiver cell at center
receiver = Circle((0, 0), 8, color='red', alpha=0.8, zorder=10)
ax.add_patch(receiver)
ax.text(0, 0, 'R', fontsize=12, ha='center', va='center',
        color='white', weight='bold', zorder=11)

# Draw example cells in rings
np.random.seed(42)
cell_types = ['epithelial', 'immune', 'stromal', 'vascular']
cell_colors = {'epithelial': '#FF6B6B', 'immune': '#4ECDC4',
               'stromal': '#95E1D3', 'vascular': '#AA96DA'}

for ring_idx, r_outer in enumerate(radii):
    r_inner = radii[ring_idx - 1] if ring_idx > 0 else 0
    r_mean = (r_inner + r_outer) / 2

    # Number of cells in ring (increases with area)
    n_cells = int(5 + ring_idx * 3)

    for _ in range(n_cells):
        # Random position in ring
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(r_inner + 5, r_outer - 5)
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Random cell type
        cell_type = np.random.choice(cell_types)

        # Draw cell
        cell = Circle((x, y), 4, color=cell_colors[cell_type],
                     alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_patch(cell)

ax.set_title('A. Spatial Ring Hierarchy\\n(Concentric neighborhoods)',
             fontsize=14, weight='bold')

# Add scale bar
ax.plot([150, 200], [-220, -220], 'k-', linewidth=3)
ax.text(175, -235, '50 μm', ha='center', fontsize=10, weight='bold')

# Panel B: Ring aggregation pipeline
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Show pipeline for one ring
y_start = 8

# Variable-size cell set
ax.text(1, y_start, 'Variable-size\\ncell set', ha='center', fontsize=10, weight='bold')
for i in range(5):
    circle = Circle((1, y_start - 1 - i*0.3), 0.1, color='lightblue', alpha=0.7)
    ax.add_patch(circle)
ax.text(1, y_start - 2, '5-50+ cells', ha='center', fontsize=8, style='italic')

# Arrow
ax.annotate('', xy=(2.5, y_start - 1), xytext=(1.5, y_start - 1),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# ISAB
box1 = mpatches.FancyBboxPatch((2.5, y_start - 1.5), 1, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(box1)
ax.text(3, y_start - 1, 'ISAB', ha='center', va='center', fontsize=11, weight='bold')
ax.text(3, y_start - 1.8, 'O(NM) complexity', ha='center', fontsize=7, style='italic')

# Arrow
ax.annotate('', xy=(4, y_start - 1), xytext=(3.5, y_start - 1),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# SAB
box2 = mpatches.FancyBboxPatch((4, y_start - 1.5), 1, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(box2)
ax.text(4.5, y_start - 1, 'SAB', ha='center', va='center', fontsize=11, weight='bold')
ax.text(4.5, y_start - 1.8, 'Self-attention', ha='center', fontsize=7, style='italic')

# Arrow
ax.annotate('', xy=(5.5, y_start - 1), xytext=(5, y_start - 1),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# PMA
box3 = mpatches.FancyBboxPatch((5.5, y_start - 1.5), 1, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor='purple', facecolor='plum', linewidth=2)
ax.add_patch(box3)
ax.text(6, y_start - 1, 'PMA', ha='center', va='center', fontsize=11, weight='bold')
ax.text(6, y_start - 1.8, 'Pool to fixed size', ha='center', fontsize=7, style='italic')

# Arrow
ax.annotate('', xy=(7.5, y_start - 1), xytext=(6.5, y_start - 1),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Fixed-size ring token
ax.text(8, y_start, 'Fixed-size\\nring token', ha='center', fontsize=10, weight='bold')
circle = Circle((8, y_start - 1), 0.2, color='gold', alpha=0.8, linewidth=2, edgecolor='black')
ax.add_patch(circle)
ax.text(8, y_start - 1.6, '1 token\\n(D dims)', ha='center', fontsize=8, style='italic')

# Add explanation boxes
ax.text(5, 4, 'This pipeline is applied to each of the 4 rings independently',
        ha='center', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.text(5, 2.5, 'Result: 4 fixed-size ring tokens, regardless of cell count in each ring',
        ha='center', fontsize=10, style='italic')

ax.set_title('B. Set Transformer Aggregation\\n(Variable → Fixed size)',
             fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('spatial_ring_hierarchy.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nSpatial Ring Design:")
print("===================")
print("Ring 1 (0-25μm):    Direct contact, immediate microenvironment")
print("Ring 2 (25-50μm):   Local paracrine signaling")
print("Ring 3 (50-100μm):  Intermediate niche, tissue architecture")
print("Ring 4 (100-200μm): Broader context, lesion boundaries")
print("\\nWhy Set Transformer?")
print("  - Each ring has VARIABLE number of cells (5-50+)")
print("  - Need PERMUTATION-INVARIANT aggregation (order doesn't matter)")
print("  - ISAB reduces complexity from O(N²) to O(NM)")
print("  - PMA pools to FIXED-SIZE output (required for transformer input)")
"""))

# ==============================================================================
# CELL 4: DUAL REFERENCE AS TOKENS
# ==============================================================================
new_cells.append(new_markdown_cell("""### 4. Dual Reference as Tokens (NOT Branches!)

**Critical design choice**: HLCA and LuCA are **tokens in the sequence**, not separate encoder branches.

**Why this matters**:

Traditional dual-branch approach:
```
x → HLCA_encoder → z_hlca \\
                             → concatenate → fused
x → LuCA_encoder → z_luca /
```

StageBridge token-based approach:
```
[Receiver, Ring1-4, HLCA, LuCA, Pathway, Stats] → Unified Self-Attention
```

**Advantages**:
1. **Cross-attention**: Spatial tokens can attend to reference tokens (and vice versa)
2. **Interpretability**: Can see how much the model relies on healthy vs cancer reference
3. **Flexibility**: References participate in full context reasoning, not isolated encoding
4. **Biological meaning**: "How does this cell compare to both healthy and cancer states?"
"""))

new_cells.append(new_code_cell("""# ============================================================================
# Compare dual-branch vs token-based reference integration
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel A: Dual-branch (traditional)
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Input
input_box = FancyBboxPatch((4, 8.5), 2, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgray', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 8.9, 'Cell Expression (x)', ha='center', fontsize=11, weight='bold')

# HLCA branch
ax.annotate('', xy=(2, 7), xytext=(4.5, 8.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
hlca_box = FancyBboxPatch((1, 6), 2, 2, boxstyle="round,pad=0.1",
                          edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(hlca_box)
ax.text(2, 7, 'HLCA\\nEncoder', ha='center', va='center', fontsize=11, weight='bold')

# LuCA branch
ax.annotate('', xy=(7, 7), xytext=(5.5, 8.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
luca_box = FancyBboxPatch((6, 6), 2, 2, boxstyle="round,pad=0.1",
                          edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(luca_box)
ax.text(7, 7, 'LuCA\\nEncoder', ha='center', va='center', fontsize=11, weight='bold')

# Concatenation
ax.annotate('', xy=(4.5, 4), xytext=(2, 6),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))
ax.annotate('', xy=(5.5, 4), xytext=(7, 6),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
concat_box = FancyBboxPatch((4, 3), 2, 2, boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='plum', linewidth=2)
ax.add_patch(concat_box)
ax.text(5, 4, 'Concatenate\\n[z_hlca, z_luca]', ha='center', va='center',
        fontsize=10, weight='bold')

# Fused output
ax.annotate('', xy=(5, 1.5), xytext=(5, 3),
            arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
fused_box = FancyBboxPatch((4, 0.5), 2, 1, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='lavender', linewidth=2)
ax.add_patch(fused_box)
ax.text(5, 1, 'Fused\\nEmbedding', ha='center', va='center', fontsize=10, weight='bold')

# Add limitations
ax.text(5, -0.5, 'Limitation: No cross-attention\\nbetween references',
        ha='center', fontsize=9, style='italic', color='darkred',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.set_title('A. Dual-Branch Approach (Traditional)', fontsize=13, weight='bold')

# Panel B: Token-based (StageBridge)
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Token sequence
token_y = 8
token_names = ['R', 'Ring1', 'Ring2', 'Ring3', 'Ring4', 'HLCA', 'LuCA', 'Path', 'Stats']
token_colors = ['#FF6B6B', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#4ECDC4',
                '#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']

for i, (name, color) in enumerate(zip(token_names, token_colors)):
    x_pos = 0.5 + i * 1.0
    box = FancyBboxPatch((x_pos - 0.35, token_y - 0.25), 0.7, 0.5,
                        boxstyle="round,pad=0.05",
                        edgecolor='black', facecolor=color, linewidth=1.5, alpha=0.7)
    ax.add_patch(box)
    ax.text(x_pos, token_y, name, ha='center', va='center',
            fontsize=7, weight='bold')

# Unified self-attention
ax.text(5, 6.5, 'Unified Self-Attention', ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Attention matrix visualization
attention_y = 4.5
ax.text(5, attention_y + 1, 'Cross-Attention Between All Tokens',
        ha='center', fontsize=10, weight='bold')

# Draw simplified attention heatmap
n_tokens = 9
cell_size = 0.5
for i in range(n_tokens):
    for j in range(n_tokens):
        x = 1 + j * cell_size
        y = attention_y - i * cell_size

        # Simulate attention pattern
        if i == 0:  # Receiver attends to all
            alpha = 0.8
        elif 1 <= i <= 4 and 1 <= j <= 4:  # Rings attend to rings
            alpha = 0.6
        elif i == 5 and j == 6:  # HLCA ↔ LuCA
            alpha = 0.7
        elif i == 6 and j == 5:
            alpha = 0.7
        else:
            alpha = 0.3

        rect = mpatches.Rectangle((x, y), cell_size, cell_size,
                                 facecolor='blue', alpha=alpha, edgecolor='gray', linewidth=0.5)
        ax.add_patch(rect)

# Labels
ax.text(0.5, attention_y, 'From', ha='right', va='center', fontsize=8, weight='bold', rotation=90)
ax.text(3.5, attention_y + 1.5, 'To', ha='center', va='bottom', fontsize=8, weight='bold')

# Output
ax.annotate('', xy=(5, 0.5), xytext=(5, attention_y - 5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
output_box = FancyBboxPatch((4, 0), 2, 0.5, boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='gold', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 0.25, 'Context Embedding', ha='center', va='center',
        fontsize=10, weight='bold')

# Add advantages
ax.text(5, -0.8, 'Advantage: Full cross-attention\\nbetween spatial & references',
        ha='center', fontsize=9, style='italic', color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax.set_title('B. Token-Based Approach (StageBridge)', fontsize=13, weight='bold')

plt.tight_layout()
plt.savefig('dual_reference_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nKey Differences:")
print("================")
print("\\nDual-Branch (Traditional):")
print("  ✗ Separate encoders for HLCA and LuCA")
print("  ✗ References processed independently")
print("  ✗ Late fusion via concatenation")
print("  ✗ No cross-attention between references and spatial context")
print("\\nToken-Based (StageBridge):")
print("  ✓ HLCA and LuCA are tokens in the sequence")
print("  ✓ Unified self-attention across all tokens")
print("  ✓ Spatial tokens can attend to reference tokens")
print("  ✓ References can attend to each other and spatial context")
print("  ✓ More interpretable: can analyze attention weights")
print("\\nBiological Interpretation:")
print("  'How does this cell's spatial context relate to both healthy and cancer states?'")
print("  The model learns to weight references based on local niche composition.")
"""))

# ==============================================================================
# CELL 5: TYPE EMBEDDINGS
# ==============================================================================
new_cells.append(new_markdown_cell("""### 5. Type Embeddings (Our Positional Encoding)

**Problem**: Self-attention is permutation-invariant. Without positional information, the model can't distinguish token roles.

**Standard solution**: Positional embeddings (e.g., sinusoidal, learned per-position)

**StageBridge solution**: **Type embeddings** (learned per-token-type, NOT per-position)

**Why type instead of position?**

1. **Rings are unordered sets**: Within each ring, cell order doesn't matter (permutation invariance is desired)
2. **Token roles are semantic**: "This is a reference" vs "this is spatial" matters more than "this is position 5"
3. **Hierarchical structure**: Ring ID embeddings provide ordering when needed

**7 token types** (in current implementation):
- Type 0: Receiver
- Type 1: Spatial (shared across rings 1-4, ring ID adds specificity)
- Type 2: HLCA reference
- Type 3: LuCA reference
- Type 4: Pathway/LR
- Type 5: Neighborhood stats
- Type 6: Atlas contrast (optional)
"""))

new_cells.append(new_code_cell("""# ============================================================================
# Visualize type embedding strategy
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Standard positional encoding
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off']

# Tokens
tokens = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
y_pos = 7
for i, token in enumerate(tokens):
    x = 1 + i * 1.0
    box = FancyBboxPatch((x - 0.3, y_pos - 0.3), 0.6, 0.6,
                        boxstyle="round,pad=0.05",
                        edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y_pos, token, ha='center', va='center', fontsize=10, weight='bold')

# Positional embeddings
y_pos = 5
ax.text(0.5, y_pos, 'Position:', ha='right', va='center', fontsize=9, weight='bold')
for i in range(9):
    x = 1 + i * 1.0
    box = FancyBboxPatch((x - 0.3, y_pos - 0.3), 0.6, 0.6,
                        boxstyle="round,pad=0.05",
                        edgecolor='blue', facecolor='lightcyan', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y_pos, f'P{i}', ha='center', va='center', fontsize=9)

# Arrows
for i in range(9):
    x = 1 + i * 1.0
    ax.annotate('', xy=(x, y_pos + 0.3), xytext=(x, y_pos - 0.3 + 2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))

# Add explanation
ax.text(5, 3, 'Problem: Position 5 and position 6 have\\ndifferent embeddings even if they have\\nsimilar semantic roles',
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.set_title('A. Standard Positional Encoding\\n(Position-specific)', fontsize=13, weight='bold')

# Panel B: Type embedding (StageBridge)
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Tokens with type colors
token_data = [
    ('R', 'Type 0', '#FF6B6B'),
    ('Ring1', 'Type 1', '#4ECDC4'),
    ('Ring2', 'Type 1', '#4ECDC4'),
    ('Ring3', 'Type 1', '#4ECDC4'),
    ('Ring4', 'Type 1', '#4ECDC4'),
    ('HLCA', 'Type 2', '#95E1D3'),
    ('LuCA', 'Type 3', '#F38181'),
    ('Path', 'Type 4', '#AA96DA'),
    ('Stats', 'Type 5', '#FCBAD3'),
]

y_pos = 7
for i, (name, ttype, color) in enumerate(token_data):
    x = 1 + i * 1.0
    box = FancyBboxPatch((x - 0.3, y_pos - 0.3), 0.6, 0.6,
                        boxstyle="round,pad=0.05",
                        edgecolor='black', facecolor=color, linewidth=2, alpha=0.7)
    ax.add_patch(box)
    ax.text(x, y_pos, name, ha='center', va='center', fontsize=8, weight='bold')

# Type embeddings (grouped by semantic role)
y_pos = 5
ax.text(0.3, y_pos, 'Type:', ha='right', va='center', fontsize=9, weight='bold')

type_positions = {
    'Type 0': [0],
    'Type 1': [1, 2, 3, 4],
    'Type 2': [5],
    'Type 3': [6],
    'Type 4': [7],
    'Type 5': [8],
}

for ttype, positions in type_positions.items():
    for pos in positions:
        x = 1 + pos * 1.0
        type_num = int(ttype.split()[1])
        color = token_data[pos][2]
        box = FancyBboxPatch((x - 0.3, y_pos - 0.3), 0.6, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor='purple', facecolor=color, linewidth=2, alpha=0.5)
        ax.add_patch(box)
        ax.text(x, y_pos, f'T{type_num}', ha='center', va='center', fontsize=9, weight='bold')

# Ring ID embeddings (for spatial tokens only)
y_pos = 3.5
ax.text(0.1, y_pos, 'Ring ID:', ha='right', va='center', fontsize=8, weight='bold')
for ring_idx in range(4):
    x = 1 + (ring_idx + 1) * 1.0
    box = FancyBboxPatch((x - 0.25, y_pos - 0.25), 0.5, 0.5,
                        boxstyle="round,pad=0.05",
                        edgecolor='darkgreen', facecolor='lightgreen', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y_pos, f'R{ring_idx}', ha='center', va='center', fontsize=8)

# Arrows
for i, (_, _, _) in enumerate(token_data):
    x = 1 + i * 1.0
    ax.annotate('', xy=(x, y_pos + 0.3 + 1.5), xytext=(x, y_pos - 0.3 + 1.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))

# Ring ID arrows
for ring_idx in range(4):
    x = 1 + (ring_idx + 1) * 1.0
    ax.annotate('', xy=(x, y_pos + 0.25 + 1.5), xytext=(x, y_pos + 0.25),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='darkgreen'))

# Add explanation
ax.text(5, 1.5, 'Advantage: Ring tokens (1-4) share Type 1\\nbut differ by Ring ID embedding.\\nSemantic grouping + hierarchical specificity!',
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax.set_title('B. Type + Ring ID Embeddings (StageBridge)\\n(Semantic role-specific)',
             fontsize=13, weight='bold')

plt.tight_layout()
plt.savefig('type_embeddings.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nType Embedding Strategy:")
print("========================")
print("\\nToken Type Assignments:")
print("  Type 0: Receiver (masked during training)")
print("  Type 1: Spatial (rings 1-4 share this type)")
print("  Type 2: HLCA reference")
print("  Type 3: LuCA reference")
print("  Type 4: Pathway/LR summary")
print("  Type 5: Neighborhood statistics")
print("\\nRing ID Embeddings (additional):")
print("  Ring 0: 0-25μm")
print("  Ring 1: 25-50μm")
print("  Ring 2: 50-100μm")
print("  Ring 3: 100-200μm")
print("\\nWhy This Design?")
print("  1. Rings 1-4 share semantic meaning (spatial context)")
print("  2. Ring ID provides hierarchical distance information")
print("  3. Reference tokens get unique types (semantically distinct)")
print("  4. Model learns to group tokens by biological role")
print("\\nFormally:")
print("  token_embedding = content_embedding + type_embedding + (ring_embedding if spatial)")
"""))

# ==============================================================================
# CELL 6: SSL PRETRAINING OBJECTIVES
# ==============================================================================
new_cells.append(new_markdown_cell("""### 6. SSL Pretraining Objectives

**Weight distribution reflects the core novelty**:

| Objective | Weight | Purpose |
|-----------|--------|---------|
| Masked receiver reconstruction | 70% | **PRIMARY**: Predict receiver from niche context |
| Ranking (positive/negative) | 10% | Auxiliary: Control discrimination |
| Provider consistency | 10% | Auxiliary: Cross-view consistency |
| Coordinate corruption | 5% | Auxiliary: Spatial awareness |
| Group relation | 5% | Auxiliary: Biological grouping |

**Why 70% on masked receiver?**

This is the CORE REPRESENTATION-LEARNING SIGNAL:
- Forces the model to encode niche context effectively
- Directly tests the hypothesis: "receiver state is predictable from local niche"
- Aligns with the biological question: "What should this cell become given its neighbors?"

The auxiliary objectives (30%) provide additional supervision but are NOT the main learning signal.
"""))

new_cells.append(new_code_cell("""# ============================================================================
# Visualize SSL pretraining objectives
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, FancyBboxPatch

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Loss weight distribution (pie chart)
ax1 = fig.add_subplot(gs[0, 0])

weights = [70, 10, 10, 5, 5]
labels = ['Masked Token\\n(70%)', 'Ranking\\n(10%)', 'Provider\\nConsistency\\n(10%)',
          'Coordinate\\nCorruption\\n(5%)', 'Group\\nRelation\\n(5%)']
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA']
explode = (0.1, 0, 0, 0, 0)  # Emphasize primary objective

wedges, texts, autotexts = ax1.pie(weights, labels=labels, colors=colors, autopct='%1.0f%%',
                                     startangle=90, explode=explode, textprops={'fontsize': 11, 'weight': 'bold'})

# Emphasize primary objective
autotexts[0].set_color('white')
autotexts[0].set_fontsize(14)
autotexts[0].set_weight('extra bold')

ax1.set_title('A. SSL Loss Weight Distribution\\n(70% on receiver reconstruction)',
              fontsize=13, weight='bold')

# Panel B: Masked receiver reconstruction
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Input sequence with masked receiver
y_pos = 8
tokens_masked = ['?', 'R1', 'R2', 'R3', 'R4', 'H', 'L', 'P', 'S']
token_colors = ['white', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#4ECDC4',
                '#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']

for i, (name, color) in enumerate(zip(tokens_masked, token_colors)):
    x = 1 + i * 1.0
    if i == 0:  # Masked receiver
        box = FancyBboxPatch((x - 0.3, y_pos - 0.3), 0.6, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor='red', facecolor='white', linewidth=3, linestyle='--')
        ax2.add_patch(box)
        ax2.text(x, y_pos, '?', ha='center', va='center', fontsize=16,
                weight='bold', color='red')
    else:
        box = FancyBboxPatch((x - 0.3, y_pos - 0.3), 0.6, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color, linewidth=2, alpha=0.7)
        ax2.add_patch(box)
        ax2.text(x, y_pos, name, ha='center', va='center', fontsize=9, weight='bold')

# Transformer encoder
y_pos = 5.5
encoder_box = FancyBboxPatch((1, y_pos - 0.5), 8, 1,
                            boxstyle="round,pad=0.1",
                            edgecolor='blue', facecolor='lightblue', linewidth=2)
ax2.add_patch(encoder_box)
ax2.text(5, y_pos, 'Transformer Encoder\\n(Self-Attention over tokens 1-8)',
        ha='center', va='center', fontsize=10, weight='bold')

# Decoder
y_pos = 3
ax2.annotate('', xy=(2, y_pos + 0.5), xytext=(2, y_pos + 1.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
decoder_box = FancyBboxPatch((1, y_pos - 0.5), 2, 1,
                            boxstyle="round,pad=0.1",
                            edgecolor='purple', facecolor='plum', linewidth=2)
ax2.add_patch(decoder_box)
ax2.text(2, y_pos, 'Decoder\\n(MLP)', ha='center', va='center', fontsize=10, weight='bold')

# Predicted receiver
y_pos = 1
ax2.annotate('', xy=(2, y_pos + 0.5), xytext=(2, y_pos + 1),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
pred_box = FancyBboxPatch((1.3, y_pos - 0.3), 1.4, 0.6,
                         boxstyle="round,pad=0.05",
                         edgecolor='red', facecolor='#FF6B6B', linewidth=2, alpha=0.7)
ax2.add_patch(pred_box)
ax2.text(2, y_pos, 'Predicted R', ha='center', va='center', fontsize=10, weight='bold')

# Target receiver
target_box = FancyBboxPatch((6.3, y_pos - 0.3), 1.4, 0.6,
                           boxstyle="round,pad=0.05",
                           edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax2.add_patch(target_box)
ax2.text(7, y_pos, 'Target R', ha='center', va='center', fontsize=10, weight='bold')

# Loss
ax2.annotate('', xy=(4.5, y_pos), xytext=(3.5, y_pos),
            arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
ax2.text(4, y_pos - 0.5, 'MSE Loss', ha='center', fontsize=10, weight='bold', color='red')

ax2.set_title('B. Masked Receiver Reconstruction (70% weight)', fontsize=13, weight='bold')

# Panel C: Auxiliary objectives
ax3 = fig.add_subplot(gs[1, :])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')

# Ranking objective
x_start = 0.5
y_center = 7
ax3.text(x_start + 1.25, y_center + 1.5, 'Ranking (10%)', ha='center',
        fontsize=11, weight='bold')
pos_box = FancyBboxPatch((x_start, y_center - 0.3), 1, 0.6,
                        boxstyle="round,pad=0.05",
                        edgecolor='green', facecolor='lightgreen', linewidth=2)
ax3.add_patch(pos_box)
ax3.text(x_start + 0.5, y_center, 'Positive\\nContext', ha='center', va='center',
        fontsize=8, weight='bold')

neg_box = FancyBboxPatch((x_start + 1.5, y_center - 0.3), 1, 0.6,
                        boxstyle="round,pad=0.05",
                        edgecolor='red', facecolor='lightcoral', linewidth=2)
ax3.add_patch(neg_box)
ax3.text(x_start + 2, y_center, 'Negative\\nControl', ha='center', va='center',
        fontsize=8, weight='bold')

ax3.text(x_start + 1.25, y_center - 1, 'score(pos) > score(neg) + margin',
        ha='center', fontsize=8, style='italic')

# Provider consistency
x_start = 3.5
ax3.text(x_start + 1, y_center + 1.5, 'Provider Consistency (10%)', ha='center',
        fontsize=11, weight='bold')
view1_box = FancyBboxPatch((x_start, y_center - 0.3), 0.8, 0.6,
                          boxstyle="round,pad=0.05",
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
ax3.add_patch(view1_box)
ax3.text(x_start + 0.4, y_center, 'View 1', ha='center', va='center',
        fontsize=8, weight='bold')

view2_box = FancyBboxPatch((x_start + 1.2, y_center - 0.3), 0.8, 0.6,
                          boxstyle="round,pad=0.05",
                          edgecolor='blue', facecolor='lightcyan', linewidth=2)
ax3.add_patch(view2_box)
ax3.text(x_start + 1.6, y_center, 'View 2', ha='center', va='center',
        fontsize=8, weight='bold')

ax3.annotate('', xy=(x_start + 1.6, y_center), xytext=(x_start + 0.4, y_center),
            arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
ax3.text(x_start + 1, y_center - 1, 'cosine similarity', ha='center',
        fontsize=8, style='italic')

# Coordinate corruption
x_start = 6
ax3.text(x_start + 1, y_center + 1.5, 'Coordinate Corruption (5%)', ha='center',
        fontsize=11, weight='bold')
real_box = FancyBboxPatch((x_start, y_center - 0.3), 0.8, 0.6,
                         boxstyle="round,pad=0.05",
                         edgecolor='green', facecolor='lightgreen', linewidth=2)
ax3.add_patch(real_box)
ax3.text(x_start + 0.4, y_center, 'Real\\nCoords', ha='center', va='center',
        fontsize=7, weight='bold')

corrupt_box = FancyBboxPatch((x_start + 1.2, y_center - 0.3), 0.8, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor='red', facecolor='lightcoral', linewidth=2)
ax3.add_patch(corrupt_box)
ax3.text(x_start + 1.6, y_center, 'Corrupt\\nCoords', ha='center', va='center',
        fontsize=7, weight='bold')

ax3.text(x_start + 1, y_center - 1, 'binary classifier', ha='center',
        fontsize=8, style='italic')

# Group relation
x_start = 8.5
ax3.text(x_start + 0.75, y_center + 1.5, 'Group Relation (5%)', ha='center',
        fontsize=11, weight='bold')
same_box = FancyBboxPatch((x_start, y_center - 0.3), 0.6, 0.6,
                         boxstyle="round,pad=0.05",
                         edgecolor='green', facecolor='lightgreen', linewidth=2)
ax3.add_patch(same_box)
ax3.text(x_start + 0.3, y_center, 'Same\\nCtx', ha='center', va='center',
        fontsize=7, weight='bold')

diff_box = FancyBboxPatch((x_start + 0.9, y_center - 0.3), 0.6, 0.6,
                         boxstyle="round,pad=0.05",
                         edgecolor='red', facecolor='lightcoral', linewidth=2)
ax3.add_patch(diff_box)
ax3.text(x_start + 1.2, y_center, 'Diff\\nCtx', ha='center', va='center',
        fontsize=7, weight='bold')

ax3.text(x_start + 0.75, y_center - 1, 'group coherence', ha='center',
        fontsize=8, style='italic')

# Add summary text
ax3.text(5, 2, 'Auxiliary objectives (30% total) provide additional supervision:',
        ha='center', fontsize=11, weight='bold')
ax3.text(5, 1, '• Ranking: Discriminate real niche from negative controls\\n'
               '• Provider consistency: Cross-view invariance\\n'
               '• Coordinate corruption: Spatial structure awareness\\n'
               '• Group relation: Biological group coherence',
        ha='center', fontsize=9, style='italic')

ax3.set_title('C. Auxiliary SSL Objectives (30% total weight)', fontsize=13, weight='bold')

plt.tight_layout()
plt.savefig('ssl_pretraining_objectives.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nSSL Pretraining Strategy:")
print("=========================")
print("\\nPrimary Objective (70%):")
print("  - Masked receiver reconstruction")
print("  - Mask token 0, predict from tokens 1-8")
print("  - Forces model to learn niche-aware representations")
print("  - Directly tests biological hypothesis")
print("\\nAuxiliary Objectives (30%):")
print("  - Ranking (10%): Positive vs negative control discrimination")
print("  - Provider consistency (10%): Cross-view invariance")
print("  - Coordinate corruption (5%): Spatial structure awareness")
print("  - Group relation (5%): Biological group coherence")
print("\\nWhy 70% on receiver reconstruction?")
print("  This is the CORE learning signal that makes StageBridge unique.")
print("  It encodes the hypothesis: 'cell state is predictable from niche context'")
print("\\nTotal Loss:")
print("  L_total = 0.70 * L_masked + 0.10 * L_ranking + 0.10 * L_consistency")
print("            + 0.05 * L_coord + 0.05 * L_group")
"""))

# ==============================================================================
# CELL 7: ATTENTION FLOW ANALYSIS
# ==============================================================================
new_cells.append(new_markdown_cell("""### 7. Attention Flow Analysis

**What does the model attend to?**

With 9 tokens in unified self-attention, we can analyze:
1. Which tokens does the **receiver** attend to? (niche dependency)
2. Which tokens do **spatial rings** attend to? (hierarchical structure)
3. How do **reference tokens** interact? (HLCA ↔ LuCA)
4. What do **pathway/stats tokens** attend to? (context integration)

**Attention matrix**: 9x9 matrix showing attention weights between all token pairs.

**Key patterns to look for**:
- Receiver should attend strongly to nearby rings (1-2)
- Rings should attend to each other (hierarchical spatial attention)
- References should attend to spatial context (not just themselves)
- Pathway/stats should integrate information from multiple sources
"""))

new_cells.append(new_code_cell("""# ============================================================================
# Analyze attention flow in the 9-token architecture
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate attention matrix (in practice, extract from trained model)
np.random.seed(42)

# Create realistic attention pattern
n_tokens = 9
token_names = ['Receiver', 'Ring1', 'Ring2', 'Ring3', 'Ring4',
               'HLCA', 'LuCA', 'Pathway', 'Stats']

# Initialize with small random values
attn = np.random.uniform(0.01, 0.05, (n_tokens, n_tokens))

# Receiver attends strongly to nearby rings and references
attn[0, 1:5] = np.array([0.25, 0.20, 0.10, 0.05])  # Spatial rings
attn[0, 5:7] = np.array([0.15, 0.12])  # References
attn[0, 7:9] = np.array([0.08, 0.05])  # Pathway, stats

# Rings attend to each other (hierarchical)
for i in range(1, 5):
    attn[i, 1:5] = np.random.uniform(0.15, 0.25, 4)
    attn[i, i] = 0.3  # Self-attention
    attn[i, 5:7] = np.random.uniform(0.05, 0.10, 2)  # References

# References attend to spatial context
attn[5, 1:5] = np.random.uniform(0.15, 0.25, 4)  # HLCA → rings
attn[5, 6] = 0.12  # HLCA → LuCA
attn[6, 1:5] = np.random.uniform(0.15, 0.25, 4)  # LuCA → rings
attn[6, 5] = 0.12  # LuCA → HLCA

# Pathway and stats integrate broadly
attn[7, :] = np.random.uniform(0.08, 0.15, n_tokens)
attn[8, :] = np.random.uniform(0.08, 0.15, n_tokens)

# Normalize rows to sum to 1
attn = attn / attn.sum(axis=1, keepdims=True)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Attention heatmap
ax = axes[0]
im = ax.imshow(attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.3)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Attention Weight', fontsize=11, weight='bold')

# Set ticks and labels
ax.set_xticks(np.arange(n_tokens))
ax.set_yticks(np.arange(n_tokens))
ax.set_xticklabels(token_names, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(token_names, fontsize=10)

# Add grid
ax.set_xticks(np.arange(n_tokens) - 0.5, minor=True)
ax.set_yticks(np.arange(n_tokens) - 0.5, minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Add value annotations
for i in range(n_tokens):
    for j in range(n_tokens):
        text = ax.text(j, i, f'{attn[i, j]:.2f}',
                      ha='center', va='center', color='black' if attn[i, j] < 0.15 else 'white',
                      fontsize=8)

ax.set_title('A. Attention Matrix (9x9)\\nRow i attends to Column j',
             fontsize=13, weight='bold')
ax.set_xlabel('Attending TO', fontsize=11, weight='bold')
ax.set_ylabel('Attending FROM', fontsize=11, weight='bold')

# Panel B: Receiver attention breakdown
ax = axes[1]

# Receiver's attention distribution
receiver_attn = attn[0, :]
token_types = ['Receiver', 'Ring1', 'Ring2', 'Ring3', 'Ring4', 'HLCA', 'LuCA', 'Path', 'Stats']
colors = ['#FF6B6B', '#4ECDC4', '#4ECDC4', '#4ECDC4', '#4ECDC4',
          '#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']

bars = ax.barh(token_types, receiver_attn, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Attention Weight', fontsize=11, weight='bold')
ax.set_title('B. Receiver Attention Distribution\\nWhat does the receiver attend to?',
             fontsize=13, weight='bold')
ax.set_xlim(0, 0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, receiver_attn)):
    ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, weight='bold')

# Add interpretation boxes
ax.text(0.15, -1.5,
        'Key Observations:\\n'
        '• Receiver attends most to nearby rings (Ring1 > Ring2 > Ring3)\\n'
        '• Moderate attention to references (HLCA, LuCA)\\n'
        '• Lower attention to pathway and stats',
        fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('attention_flow_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Additional analysis: Attention statistics
print("\\nAttention Flow Statistics:")
print("==========================")
print("\\nReceiver (Token 0) attention distribution:")
for i, (name, val) in enumerate(zip(token_names, receiver_attn)):
    print(f"  {name:12s}: {val:.3f} ({val*100:.1f}%)")

print("\\nSpatial Ring (Token 1-4) average attention:")
ring_attn = attn[1:5, :].mean(axis=0)
for i, (name, val) in enumerate(zip(token_names, ring_attn)):
    print(f"  {name:12s}: {val:.3f}")

print("\\nReference (HLCA, LuCA) cross-attention:")
print(f"  HLCA → LuCA:  {attn[5, 6]:.3f}")
print(f"  LuCA → HLCA:  {attn[6, 5]:.3f}")
print(f"  (Symmetric? {abs(attn[5, 6] - attn[6, 5]) < 0.05})")

print("\\nKey Interpretations:")
print("  1. Receiver attends most to nearby rings → local niche is most important")
print("  2. Rings attend to each other → hierarchical spatial reasoning")
print("  3. References attend to spatial context → grounding in observed data")
print("  4. All tokens participate in unified attention → rich cross-modal reasoning")
"""))

# ==============================================================================
# SUMMARY CELL
# ==============================================================================
new_cells.append(new_markdown_cell("""---

## Summary: StageBridge Architectural Innovations

### What Makes StageBridge Unique?

1. **Receiver-Centered Prediction** (AMICI-inspired)
   - Mask the receiver, predict from niche context
   - Ideal for cross-sectional biological snapshots
   - Captures collective niche effects, not just pairwise

2. **9-Token Sequence Design**
   - Minimal set capturing all biological context
   - Hierarchical spatial (4 rings) + dual references + biological features
   - All tokens in unified self-attention (NOT dual-branch)

3. **Spatial Ring Hierarchy**
   - 4 concentric rings at biologically meaningful scales (25, 50, 100, 200 μm)
   - Set Transformer (ISAB → SAB → PMA) for variable-size aggregation
   - Permutation-invariant within rings, hierarchical across rings

4. **Dual Reference as Tokens**
   - HLCA and LuCA are tokens in the sequence, not separate branches
   - Enables cross-attention between spatial context and references
   - More interpretable: can analyze attention weights

5. **Type Embeddings**
   - Semantic role-based, not position-based
   - 7 token types + ring ID embeddings for spatial hierarchy
   - Respects biological structure (rings are unordered sets)

6. **SSL Pretraining**
   - 70% weight on masked receiver reconstruction (PRIMARY)
   - 30% on auxiliary objectives (ranking, consistency, etc.)
   - Directly encodes biological hypothesis

7. **Attention Flow**
   - Receiver attends to nearby rings and references
   - Rings exhibit hierarchical spatial attention
   - References attend to spatial context (grounded in data)

### Design Philosophy

Every architectural choice is motivated by **biological requirements** and **data constraints**:
- Cross-sectional snapshots → receiver-centered prediction
- Variable cell counts → Set Transformer
- Spatial hierarchy → 4 rings at meaningful scales
- Multi-modal context → tokenized references
- Interpretability → unified attention with analyzable weights

**This is domain-specific architecture, not generic transformers.**
"""))

# Insert all new cells at the identified position
for i, cell in enumerate(new_cells):
    nb.cells.insert(insertion_idx + i, cell)

# Save the modified notebook
nbformat.write(nb, notebook_path)

print(f"\nSuccessfully added {len(new_cells)} StageBridge architecture cells!")
print(f"Cells inserted at index {insertion_idx}")
print(f"New total cell count: {len(nb.cells)}")
print("\nNew cells showcase:")
print("  1. Receiver-centered prediction (AMICI-inspired)")
print("  2. The 9-token sequence design")
print("  3. Spatial ring hierarchy (4 rings, Set Transformer)")
print("  4. Dual reference as tokens (NOT branches)")
print("  5. Type embeddings (semantic role-based)")
print("  6. SSL pretraining objectives (70% receiver reconstruction)")
print("  7. Attention flow analysis")
print("  8. Summary of architectural innovations")
