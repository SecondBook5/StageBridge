# Training

Two-stage training for StageBridge: SSL pretraining followed by OT-CFM transition learning.

## Two-Stage Training

### Stage 1: SSL Pretraining (50 epochs)

**Objective**: Learn niche-aware representations via masked receiver reconstruction.

The encoder learns to reconstruct the receiver cell's embedding from its neighborhood context. This forces the model to learn meaningful niche representations before seeing transition labels.

```python
# Mask receiver, encode from neighbors
niche_output = model.encode_niche(masked_receiver, neighbors, ...)
loss = mse(niche_output.receiver_reconstruction, true_receiver)
```

### Stage 2: Transition Learning (100 epochs)

**Objective**: Learn stage transition dynamics via OT-CFM flow matching.

```python
# Encode niche context
context = model.encode_niche(receiver, neighbors, ...).context

# OT-CFM: predict velocity at interpolated point
t = uniform(0, 1)
x_t = (1-t) * x_source + t * x_target
v_pred = model.forward_vector_field(x_t, t, context, stage_pair)
loss = mse(v_pred, x_target - x_source)
```

## Usage

### Command Line

```bash
python -m stagebridge.training.train \
    --data-dir /path/to/data \
    --output-dir outputs/run1 \
    --fold-idx 0 \
    --ssl-epochs 50 \
    --transition-epochs 100 \
    --learning-rate 1e-4 \
    --batch-size 64 \
    --data-parallel  # Use all GPUs
```

### With HPO Parameters

```bash
python -m stagebridge.training.train \
    --data-dir /path/to/data \
    --output-dir outputs/run1 \
    --hpo-params /path/to/best_params.json
```

### Via Snakemake (Recommended)

```bash
snakemake --profile workflow/slurm --jobs 20
```

## Configuration

### TrainerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| ssl_epochs | 50 | SSL pretraining epochs |
| transition_epochs | 100 | Transition training epochs |
| learning_rate | 1e-4 | Initial learning rate |
| transition_lr_factor | 0.1 | LR multiplier for transition phase |
| freeze_encoder | False | Freeze encoder during transition (ablation) |
| early_stopping_patience | 15 | Epochs without improvement before stopping |

### StageBridgeConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 256 | Internal representation dimension |
| num_heads | 8 | Attention heads |
| use_amici_attention | True | Use continuous distance attention |
| use_gw_fusion | True | Use learned GW for atlas fusion |
| gw_fusion_type | "learned_gw" | Fusion method |

## Checkpoints

Saved to `{output_dir}/checkpoints/`:
- `ssl_pretrained.pt` - After SSL phase
- `best_checkpoint.pt` - Best validation loss
- `final_checkpoint.pt` - End of training

## Multi-GPU Training

**DataParallel** (recommended for HPO):
```bash
python -m stagebridge.training.train --data-parallel
```

**DDP** (for large-scale training):
```bash
torchrun --nproc_per_node=4 -m stagebridge.training.train --ddp
```
