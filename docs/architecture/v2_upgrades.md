# StageBridge V2 Upgrades

These upgrades enhance the transformer architecture for better interpretability and performance.

## Upgrade 1: CrossAttentionDrift Head

**Status:** IMPLEMENTED and wired into V1Complete (2026-04-26)

**Current flow:**
```
9 tokens -> ReceiverCenteredNicheEncoder -> pooled context [B, D] -> MLP drift -> velocity
```

**Upgraded flow:**
```
9 tokens -> ReceiverCenteredNicheEncoder -> context tokens [B, K, D] -> CrossAttentionDrift -> velocity
```

**Implementation:**
1. Modify `ReceiverCenteredNicheEncoder` to return `context_tokens` (the intermediate `h[:, 1:, :]` from transformer)
2. Add `drift_head: Literal["mlp", "cross_attention"] = "mlp"` to V1Complete config
3. Wire CrossAttentionDrift when `drift_head="cross_attention"`

**Why it matters:**
- x_t (interpolated state) queries niche tokens to determine velocity
- Learns which niche factors matter for each transition
- Interpretable attention weights show transition-permissive contexts

## Upgrade 2: Set Transformer Context Refiner

**Status:** IMPLEMENTED and wired into V1Complete (2026-04-26)

**Current flow:**
```
context tokens -> (nothing) -> drift head
```

**Upgraded flow:**
```
context tokens -> SAB -> SAB -> refined tokens -> CrossAttentionDrift
```

**Implementation:**
1. Add `context_refiner: Literal["none", "set_transformer"] = "none"` to V1Complete config
2. Create simple `SetTransformerRefiner` module (1-2 SAB blocks)
3. Insert between encoder output and drift head

**Why it matters:**
- Models token-token interactions (e.g., immune suppression + ECM remodeling)
- Progression-permissive niches are combinatorial, not independent features
- Small sets (9 tokens) don't need ISAB, SAB is sufficient

## Upgrade 3: Context Token Interface

**Status:** IMPLEMENTED (2026-04-26)

`ReceiverNicheOutput` now includes:
```python
context: Tensor           # [B, D] pooled
context_tokens: Tensor    # [B, K, D] individual token representations (NEW)
attention_weights: Tensor
receiver_reconstruction: Tensor | None
```

Both `ReceiverCenteredNicheEncoder` and `SelfAttentionNicheEncoder` now return context_tokens.

## Upgrade 4: Interpretability Outputs

**Status:** IMPLEMENTED (2026-04-26)

`transition_forward` now returns:
```python
{
    "loss_transition": Tensor,
    "drift_pred": Tensor,
    "drift_true": Tensor,
    "z_t": Tensor,
    "ot_cost": float,
    "num_pairs": int,
    "per_pair_loss": Tensor,
    "per_stage_loss": dict,
    "src_idx": Tensor,
    "context_gate_mean": float,           # NEW: from CrossAttentionDrift
    "context_attention_entropy": float,   # NEW: from CrossAttentionDrift
}
```

## Upgrade 5: Clean Ablation Flags

```python
@dataclass
class V1CompleteConfig:
    # Existing
    latent_dim: int = 40
    niche_hidden_dim: int = 128
    ...
    
    # NEW ablation flags
    use_niche_context: bool = True      # False = receiver-only baseline
    use_receiver_centering: bool = True # False = mean-pool baseline  
    use_ot_pairing: bool = True         # False = random pairing
    use_wes: bool = True                # False = no genomic features
    drift_head: str = "mlp"             # "mlp" or "cross_attention"
    context_refiner: str = "none"       # "none" or "set_transformer"
```

## Publication Ablation Matrix

| Variant | drift_head | context_refiner | Tests |
|---------|------------|-----------------|-------|
| V1Complete (main) | mlp | none | Baseline |
| + CrossAttn | cross_attention | none | Does attention help? |
| + SetRefiner | cross_attention | set_transformer | Do token interactions help? |
| - Niche | mlp | none, use_niche_context=False | Does niche matter? |
| - OT | mlp | none, use_ot_pairing=False | Does OT pairing matter? |

## Implementation Summary (2026-04-26)

All upgrades implemented and tested:

1. **Context token interface** - `ReceiverNicheOutput.context_tokens` added
2. **CrossAttentionDrift wiring** - `drift_head="cross_attention"` config flag
3. **SetTransformerRefiner** - `context_refiner="set_transformer"` config flag  
4. **Ablation flags** - Available in `StageBridgeV1Complete.__init__`

## Usage

```python
# Full upgraded model
model = StageBridgeV1Complete(
    latent_dim=40,
    drift_head="cross_attention",      # Use cross-attention drift
    context_refiner="set_transformer", # Refine token interactions
)

# Get context tokens for drift head
ctx, tokens, attn = model.encode_niche_with_tokens(niche_tokens)

# Training forward
result = model.transition_forward(
    z_source=z_src,
    z_target=z_tgt, 
    context=ctx,
    context_tokens=tokens,  # Required for cross_attention drift
    stage_indices=stages,
)

# Inference trajectory
traj = model.sample_trajectory(
    z_source=z_src,
    context=ctx,
    context_tokens=tokens,
    stage_indices=stages,
)
```

## Decision Point

Re-run HPO with `drift_head="cross_attention"` and `context_refiner="set_transformer"` to see if upgrades improve performance. If not, can always fall back to MLP drift.

---

## Future: Adversarial Cell Cycle Regularization

**Status:** DEFERRED to v2/revision

**Motivation:**
Cell cycle (S_score, G2M_score) is a major confounder in progression modeling. If the model learns "high proliferation → invasive-like transition", the transition field may be biologically shallow - learning "cycling cells look aggressive" rather than true niche-conditioned progression.

**Proposed approach:**
Adversarial training to discourage cell-cycle predictability from context representation:

```python
class GradientReversalLayer(nn.Module):
    """Reverses gradients during backward pass."""
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class CellCycleDiscriminator(nn.Module):
    """Try to predict S_score/G2M_score from context."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # S_score, G2M_score
        )
    
    def forward(self, x):
        return self.head(x)
```

**Loss formulation:**
```
L_total = L_transition + λ_ssl * L_ssl + λ_cc * L_cell_cycle_adv

where L_cell_cycle_adv = MSE(discriminator(GRL(context)), [S_score, G2M_score])
```

The discriminator minimizes prediction error, but encoder receives reversed gradients - learning to make cell cycle harder to predict.

**Scientific claim this enables:**
> "Transition fields remain predictive of progression even when forced to be uninformative about cell cycle state"

This proves niche-conditioned progression signal is orthogonal to proliferation.

**Config flags (when implemented):**
```yaml
use_adversarial_cell_cycle: false  # Default off
adversarial_cell_cycle_weight: 0.01
gradient_reversal_lambda: 0.1
cell_cycle_target_keys: ["S_score", "G2M_score"]
```

**Where to apply:**
Apply gradient reversal to `pooled_context`, NOT raw latent z. Cell cycle can be real part of cell state - we want to test whether the niche context representation overly encodes it.

**Implementation plan:**
1. `stagebridge/training/gradient_reversal.py` - GRL layer
2. `stagebridge/training/adversarial.py` - CellCycleDiscriminator  
3. Config flags in TrainingConfig
4. Diagnostic: plot cell-cycle predictability vs transition accuracy

**When to use:**
- Post-hoc analysis shows transition signal correlates with S/G2M scores
- Reviewers ask "did you control for cell cycle?"
- Want stronger biological claim about progression-specific signal
