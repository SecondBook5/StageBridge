# Transition Modeling

OT-CFM (Optimal Transport Conditional Flow Matching) for stage transitions.

## Key Modules

### `drift.py` - CrossAttentionDrift

Predicts velocity field v(x_t, t) conditioned on niche context.

**Gated architecture**: Blends context-aware and latent-only predictions.

```python
# Context-aware path (cross-attention over niche tokens)
q = project([x_t; time_emb])
kv = [context_tokens; stage_token]
context_drift = cross_attention(q, kv)

# Latent-only path (no niche information)
latent_drift = mlp([x_t; time_emb; stage_emb])

# Learned gate decides when niche matters
gate = sigmoid(gate_network([q; context; stage_emb]))
v_t = gate * context_drift + (1 - gate) * latent_drift
```

After training, gate values reveal which transitions rely on niche context.

### `losses.py` - OT-CFM Loss

Flow matching objective with Sinkhorn OT coupling.

```python
# Sample OT-coupled pairs
coupling = sinkhorn(x_source, x_target, epsilon=0.05)
(i, j) = sample_from_coupling(coupling)

# Interpolate
t = uniform(0, 1)
x_t = (1-t) * x_source[i] + t * x_target[j]

# Target velocity is straight line
u_t = x_target[j] - x_source[i]

# Loss: predicted vs target velocity
loss = mse(model.forward_vector_field(x_t, t, context), u_t)
```

### `schrodinger_bridge.py` - Stochastic Dynamics (Optional)

Adds diffusion for stochastic transitions. Uses same drift head but adds learned score network.

## Training

Two-stage training in `training/trainer.py`:

1. **SSL (50 epochs)**: Masked receiver reconstruction
2. **Transition (100 epochs)**: OT-CFM flow matching

## Integration Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Euler | Simple x += dt * v | Fast inference |
| RK4 | 4th-order Runge-Kutta | Accurate trajectories |
| Euler-Maruyama | Euler + noise | Stochastic sampling |
