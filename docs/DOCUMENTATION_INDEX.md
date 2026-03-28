# StageBridge Documentation Index

## Quick Start

| If you want to... | Read this |
|-------------------|-----------|
| Understand the project | [README.md](../README.md) |
| Write the paper | [paper/paper_outline.md](paper/paper_outline.md) |
| Understand the methods | [methods/v1_methods_overview.md](methods/v1_methods_overview.md) |
| Implement something | [system_architecture.md](system_architecture.md) |
| Run on HPC | [hpc/README.md](hpc/README.md) |

---

## Documentation Structure

```
docs/
├── specs/                    # Core doctrine
│   ├── V1_DOCTRINE.md        # Principles + scope
│   └── REPRESENTATION_LEARNING_RULES.md
│
├── methods/                  # Technical specification
│   ├── v1_methods_overview.md      # PRIMARY - all layers
│   ├── data_model_specification.md # Data schemas
│   └── evaluation_protocol.md      # Metrics + statistics
│
├── paper/                    # Publication
│   ├── paper_outline.md            # Paper structure
│   ├── figure_table_specifications.md
│   └── evidence_matrix.md          # Claims → evidence
│
├── architecture/             # Layer-by-layer design
│   ├── reference_latent_mapping.md    # Layer A
│   ├── typed_niche_context_model.md   # Layer B
│   ├── eamist_block_diagram.md        # Layer C
│   ├── stochastic_transition_model.md # Layer D
│   ├── spatial_mapping_layer.md       # Spatial backends
│   └── rescue_ablation_design.md      # Ablations
│
├── biology/                  # Biological context
│   ├── biological_context.md         # LUAD + niche hypothesis
│   └── wes_regularization_rationale.md
│
├── hpc/                      # HPC guide
│   └── README.md
│
├── system_architecture.md    # Infrastructure details
├── implementation_roadmap.md # Status tracking
└── V2_IDEAS.md               # Future work
```

---

## Key Documents

### For Paper Writing
1. **paper/paper_outline.md** - Complete structure, what to write
2. **paper/evidence_matrix.md** - Every claim mapped to evidence
3. **paper/figure_table_specifications.md** - All 8 figures detailed
4. **methods/evaluation_protocol.md** - Metrics for Results section

### For Implementation
1. **system_architecture.md** - Full technical infrastructure
2. **methods/data_model_specification.md** - Data schemas and APIs
3. **implementation_roadmap.md** - What's done, what's needed

### For Understanding the Science
1. **specs/V1_DOCTRINE.md** - Non-negotiable principles
2. **biology/biological_context.md** - LUAD progression, niche hypothesis
3. **methods/v1_methods_overview.md** - Complete technical spec

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Data pipeline | Complete |
| Reference mapping (HLCA/LuCA) | Complete |
| Spatial backends | Complete |
| Niche encoder (Layer B) | Complete |
| Set transformer (Layer C) | Complete |
| Flow matching (Layer D) | Complete |
| Evolutionary compatibility | Complete |
| Evaluation harness | Complete |
| Documentation | Complete |

---

## V1 Key Claims

1. Dual-reference geometry improves transition structure
2. Niche context significantly improves quality (d=1.2)
3. Stochastic flow enables calibrated uncertainty
4. Genomic constraints reduce implausible transitions
5. Results robust across spatial backends

See [evidence_matrix.md](paper/evidence_matrix.md) for supporting evidence.
