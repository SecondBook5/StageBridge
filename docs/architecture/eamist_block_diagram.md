# EA-MIST Architecture Block Diagram (Layers B+C)

## Overview

EA-MIST (Evolution-Aware Multiple-Instance Set Transformer) provides **Layers B and C** of the StageBridge architecture. These layers encode local niches and aggregate them into context vectors that condition the transition model (Layer D).

```
                         STAGEBRIDGE ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer A: Dual-Reference Latent (HLCA + LuCA)                               │
│  Layer B: Local Niche Encoder (9-token transformer)    ← EA-MIST           │
│  Layer C: Hierarchical Aggregation (Set Transformer)   ← EA-MIST           │
│  Layer D: Stochastic Transition Model (Flow Matching)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Input Data

```
                                      INPUT DATA
+-------------------------------------------------------------------------------------------------+
|  Spatial Transcriptomics          snRNA-seq               WES                                   |
|  (10x Visium spots)               (cell states)           (mutations, CNA)                      |
+--------------+------------------------+---------------------------+-----------------------------+
               |                        |                           |
               v                        v                           |
+---------------------------------------------------+               |
|           SPATIAL NICHE EXTRACTION                |               |
|  - Receiver cell + 4 neighborhood rings           |               |
|  - HLCA/LuCA atlas alignment (Layer A)            |               |
|  - LR pathway activity                            |               |
|  - Neighborhood statistics                        |               |
+---------------------------+-----------------------+               |
                            v                                       |
```

## Layer B: Local Niche Encoder (per niche)

```
===================================================================================================
                         LAYER B: LOCAL NICHE ENCODER (per niche)
===================================================================================================
                                                                    |
+-------------------------------------------------------------+     |
|                   LOCAL NICHE TOKENIZER                     |     |
|  +---------+ +---------+ +---------+ +---------+            |     |
|  |Receiver | | Ring 1  | | Ring 2  | | Ring 3  |            |     |
|  |  Token  | |  Token  | |  Token  | |  Token  |            |     |
|  |         | |         | |         | |         |            |     |
|  | expr +  | |cell-type| |cell-type| |cell-type|            |     |
|  | state   | | compos. | | compos. | | compos. |            |     |
|  | embed   | | @ r1    | | @ r2    | | @ r3    |            |     |
|  +----+----+ +----+----+ +----+----+ +----+----+            |     |
|       |           |           |           |                 |     |
|  +----+----+ +----+----+ +----+----+ +----+----+ +-------+  |     |
|  | Ring 4  | |  HLCA   | |  LuCA   | |Pathway  | | Stats |  |     |
|  |  Token  | |  Token  | |  Token  | |  Token  | | Token |  |     |
|  |         | |         | |         | |         | |       |  |     |
|  |cell-type| | healthy | | tumor   | |  L-R    | |density|  |     |
|  | compos. | |  atlas  | |  atlas  | |activity | |entropy|  |     |
|  | @ r4    | | sim.    | | sim.    | | summary | |       |  |     |
|  +----+----+ +----+----+ +----+----+ +----+----+ +---+---+  |     |
|       |           |           |           |          |      |     |
|       +-----------+-----------+-----------+----------+      |     |
|                               v                             |     |
|               +-------------------------------+             |     |
|               |  9 Tokens x model_dim (128)   |             |     |
|               |  + Token Type Embeddings      |             |     |
|               |  + Ring Position Embeddings   |             |     |
|               +---------------+---------------+             |     |
+-------------------------------+-----------------------------+     |
                                v                                   |
+-------------------------------------------------------------------+
|              LOCAL NICHE TRANSFORMER                              |
|  +---------------------------------------------------------+      |
|  |  SAB (Self-Attention Block) x 2 layers                  |      |
|  |  +-------------+    +-------------+                     |      |
|  |  | MultiHead   |    | MultiHead   |                     |      |
|  |  | Attention   |--->| Attention   |                     |      |
|  |  | + LayerNorm |    | + LayerNorm |                     |      |
|  |  | + FFN       |    | + FFN       |                     |      |
|  |  +-------------+    +-------------+                     |      |
|  +-------------------------+-------------------------------+      |
|                            v                                      |
|  +---------------------------------------------------------+      |
|  |  PMA (Pooling by Multihead Attention)                   |      |
|  |  - 1 seed vector queries all 9 tokens                   |      |
|  |  - Produces single niche embedding                      |      |
|  +-------------------------+-------------------------------+      |
|                            v                                      |
|  +---------------------------------------------------------+      |
|  |  Niche Embedding (B x N, 128)                           |      |
|  +-------------------------+-------------------------------+      |
+----------------------------+--------------------------------------+
                             v
```

## Layer C: Hierarchical Aggregation (per sample)

```
===================================================================================================
                    LAYER C: HIERARCHICAL AGGREGATION (per sample)
===================================================================================================

+-------------------------------------------------------------------+
|              PROTOTYPE BOTTLENECK (optional)                      |
|  +---------------------------------------------------------+      |
|  |  - K learnable prototypes (default K=16)                |      |
|  |  - Soft assignment: niche -> prototype similarities     |      |
|  |  - Encourages interpretable niche clustering            |      |
|  +-------------------------+-------------------------------+      |
+----------------------------+--------------------------------------+
                             v
+-------------------------------------------------------------------+
|           SET TRANSFORMER BACKBONE                                |
|  +---------------------------------------------------------+      |
|  |  ISAB (Induced Set Attention Block) x num_layers        |      |
|  |  +------------------------------------------------------+      |
|  |  |  - M inducing points (default M=16)                  |      |
|  |  |  - O(N x M) complexity instead of O(N^2)             |      |
|  |  |  - Permutation-invariant over niches                 |      |
|  |  +------------------------------------------------------+      |
|  +-------------------------+-------------------------------+      |
|                            v                                      |
|  +---------------------------------------------------------+      |
|  |  PMA (Pooling by Multihead Attention)                   |      |
|  |  - Aggregates all niche embeddings                      |      |
|  |  - Produces context vector for Layer D                  |      |
|  +-------------------------+-------------------------------+      |
+----------------------------+--------------------------------------+
                             |
                             v
+----------------------------+--------------------------------------+
|                    EVOLUTION BRANCH (optional)             <------|-- WES Features
|  +----------------------------------------------------------------------+
|  |  Gated or FiLM conditioning on evolutionary features                 |
|  +----------------------------------------------------------------------+
+----------------------------+--------------------------------------+
                             v
                    CONTEXT VECTOR (B, 128)
                             |
                             v
                    ┌────────────────────┐
                    │     LAYER D        │
                    │  Flow Matching     │
                    │  (Transition Model)│
                    └────────────────────┘
```

## Auxiliary Output Heads (for training signal)

```
+-------------------------------------------------------------------------------------------+
|                        AUXILIARY HEADS (not primary objective)                            |
|                                                                                           |
|  +---------------------+  +---------------------+  +-------------------------------+      |
|  |   STAGE HEAD        |  |  DISPLACEMENT HEAD  |  |      EDGE HEAD                |      |
|  |  5-way softmax      |  |  scalar [0,1]       |  |  pairwise logits              |      |
|  +---------------------+  +---------------------+  +-------------------------------+      |
|                                                                                           |
|  These provide auxiliary training signal. Primary evaluation is on Layer D transitions.  |
+-------------------------------------------------------------------------------------------+
```

## Token Details

| Token | Source | Description |
|-------|--------|-------------|
| Receiver | Cell identity | Target cell expression + learned state embedding |
| Ring 1-4 | Spatial neighborhood | Cell-type composition at increasing radii |
| HLCA | Reference atlas | Similarity to healthy lung cell types (Layer A) |
| LuCA | Tumor atlas | Similarity to tumor-aware cell states (Layer A) |
| Pathway | Gene programs | Ligand-receptor and pathway activity summary |
| Stats | Neighborhood | Local density, entropy, and composition statistics |

## Layer B+C Variants (for ablation)

| Variant | Layer B | Layer C | Use |
|---------|---------|---------|-----|
| `eamist` | Full 9-token encoder | Set transformer + prototypes | Primary |
| `eamist_no_prototypes` | Full encoder | Set transformer only | Ablation |
| `deep_sets` | Full encoder | DeepSets φ→ρ | Baseline |
| `pooled` | Full encoder | Mean pooling | Baseline |

## Data Flow Summary

```
Spatial + snRNA + WES
         |
         v
  +--------------+
  |   Layer A    |  -> HLCA/LuCA embeddings
  +--------------+
         |
         v
  +--------------+
  |   Layer B    |  -> (B, N, 128) per-niche embeddings
  +--------------+
         |
         v
  +--------------+
  |   Layer C    |  -> (B, 128) context vector
  +--------------+
         |
         v
  +--------------+
  |   Layer D    |  -> Cell-state transitions (trajectories)
  +--------------+
```
