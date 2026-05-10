#!/usr/bin/env python3
"""DDP worker for HPO - launched by torchrun."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | rank%(rank)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gw-checkpoint", type=str, default=None)
    args = parser.parse_args()

    # DDP setup - torchrun sets these env vars
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Custom logging with rank
    log = logging.getLogger(__name__)
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s | %(levelname)-8s | rank{rank} | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Load params
    with open(args.params_file) as f:
        params = json.load(f)

    if rank == 0:
        log.info(f"Starting trial {params['trial_id']} on {world_size} GPUs")

    from stagebridge.loaders import create_dataloaders
    from stagebridge.models.stagebridge import StageBridge, StageBridgeConfig

    # Load data with DDP
    train_loader, val_loader, _ = create_dataloaders(
        args.data_dir, fold_idx=0, batch_size=args.batch_size, num_workers=4, use_ddp=True,
    )

    if train_loader is None:
        if rank == 0:
            with open(args.result_file, "w") as f:
                json.dump({"val_loss": float("inf")}, f)
        dist.destroy_process_group()
        return

    # Get data info
    sample_batch = next(iter(train_loader))
    evolution_dim = sample_batch.evolution_features.shape[-1] if sample_batch.evolution_features is not None else 0
    is_amici = hasattr(sample_batch, "neighbors")

    # Build config
    gw_fusion_type = params["gw_fusion_type"]
    use_gw_fusion = gw_fusion_type != "concat"
    gw_checkpoint_dir = args.gw_checkpoint if gw_fusion_type == "precompute_gw" else None

    config = StageBridgeConfig(
        hidden_dim=params["hidden_dim"],
        num_heads=params["num_heads"],
        dropout=params["dropout"],
        use_gw_fusion=use_gw_fusion,
        gw_fusion_type=gw_fusion_type,
        gw_checkpoint_dir=gw_checkpoint_dir,
        gw_output_dim=params.get("gw_output_dim", 40),
        use_amici_attention=is_amici,
        amici_num_heads=params.get("amici_num_heads", 4),
        amici_distance_scale=params.get("amici_distance_scale", 100.0),
        use_learned_ring_pooling=True,
        use_context_refiner=True,
        use_cross_attn_drift=True,
        use_pathway_head=True,
        use_proliferation_head=True,
        use_evolution_branch=evolution_dim > 0,
        evolution_dim=evolution_dim,
    )

    model = StageBridge(config).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # SB module if needed
    dynamics_type = params["dynamics_type"]
    sb_module = None
    if dynamics_type == "schrodinger_bridge":
        from stagebridge.transition.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
        sb_config = SchrodingerBridgeConfig(
            input_dim=config.input_dim,
            context_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_stages=config.num_stages,
            sigma=params.get("sb_sigma", 0.1),
            use_external_drift=True,
        )
        sb_module = SchrodingerBridge(sb_config).to(device)
        sb_module = DDP(sb_module, device_ids=[rank], output_device=rank)

        def external_drift_fn(x_t, t, context, stage_pair_id):
            return model.module.forward_vector_field(
                x_t=x_t, t=t, context=context, stage_pair_id=stage_pair_id, context_tokens=None,
            )
        sb_module.module.set_external_drift(external_drift_fn)

        all_params = list(model.parameters()) + list(sb_module.parameters())
        optimizer = AdamW(all_params, lr=params["lr"], weight_decay=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)

    # Training
    ssl_weight = params["ssl_weight"]
    pathway_weight = params["pathway_weight"]
    proliferation_weight = params["proliferation_weight"]

    model.train()
    if sb_module:
        sb_module.train()

    for epoch in range(args.n_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if hasattr(batch, "neighbors"):
                niche_output = model.module.encode_niche_amici(
                    receiver=batch.receiver,
                    neighbors=batch.neighbors,
                    distances=batch.distances,
                    neighbor_mask=batch.neighbor_mask,
                    hlca=batch.hlca,
                    luca=batch.luca,
                    pathway=batch.pathway,
                    stats=batch.stats,
                    evolution_features=batch.evolution_features,
                    return_reconstruction=True,
                )
            else:
                niche_output = model.module.encode_niche(
                    receiver=batch.receiver,
                    ring_cells=batch.ring_cells,
                    ring_masks=batch.ring_masks,
                    hlca=batch.hlca,
                    luca=batch.luca,
                    pathway=batch.pathway,
                    stats=batch.stats,
                    evolution_features=batch.evolution_features,
                    return_reconstruction=True,
                )

            ssl_loss = F.mse_loss(niche_output.receiver_reconstruction, batch.receiver)

            x0 = batch.receiver
            x1 = batch.receiver + 0.1 * torch.randn_like(batch.receiver)
            stage_pair_id = torch.zeros(x0.shape[0], dtype=torch.long, device=device)

            t = torch.rand(x0.shape[0], device=device)
            x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
            u_t = x1 - x0

            v_t = model.module.forward_vector_field(
                x_t=x_t, t=t, context=niche_output.context,
                stage_pair_id=stage_pair_id, context_tokens=niche_output.context_tokens,
            )
            drift_loss = F.mse_loss(v_t, u_t)

            if dynamics_type == "schrodinger_bridge" and sb_module is not None:
                from stagebridge.transition.schrodinger_bridge import schrodinger_bridge_loss
                sb_loss, _ = schrodinger_bridge_loss(
                    x_src=x0, x_tgt=x1, sb_module=sb_module.module,
                    context=niche_output.context, stage_pair_id=stage_pair_id, num_time_samples=4,
                )
                transition_loss = drift_loss + sb_loss
            else:
                transition_loss = drift_loss

            loss_pathway = torch.tensor(0.0, device=device)
            if model.module.pathway_head is not None and batch.pathway_targets is not None:
                pathway_logits = model.module.pathway_head(niche_output.context)
                loss_pathway = F.mse_loss(pathway_logits, batch.pathway_targets)

            loss_proliferation = torch.tensor(0.0, device=device)
            if model.module.proliferation_head is not None and batch.proliferation_target is not None:
                prolif_logit = model.module.proliferation_head(niche_output.context)
                loss_proliferation = F.binary_cross_entropy_with_logits(
                    prolif_logit.squeeze(-1), batch.proliferation_target
                )

            loss = (
                ssl_weight * ssl_loss
                + (1 - ssl_weight) * transition_loss
                + pathway_weight * loss_pathway
                + proliferation_weight * loss_proliferation
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if rank == 0:
            log.info(f"Epoch {epoch+1}/{args.n_epochs}, loss: {epoch_loss / n_batches:.6f}")

    # Validation on rank 0
    val_loss = epoch_loss / max(n_batches, 1)  # fallback
    if rank == 0:
        model.eval()
        val_total = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in (val_loader or []):
                batch = batch.to(device)
                if hasattr(batch, "neighbors"):
                    niche_output = model.module.encode_niche_amici(
                        receiver=batch.receiver, neighbors=batch.neighbors,
                        distances=batch.distances, neighbor_mask=batch.neighbor_mask,
                        hlca=batch.hlca, luca=batch.luca, pathway=batch.pathway,
                        stats=batch.stats, evolution_features=batch.evolution_features,
                        return_reconstruction=True,
                    )
                else:
                    niche_output = model.module.encode_niche(
                        receiver=batch.receiver, ring_cells=batch.ring_cells,
                        ring_masks=batch.ring_masks, hlca=batch.hlca, luca=batch.luca,
                        pathway=batch.pathway, stats=batch.stats,
                        evolution_features=batch.evolution_features,
                        return_reconstruction=True,
                    )
                ssl_loss = F.mse_loss(niche_output.receiver_reconstruction, batch.receiver)
                val_total += ssl_loss.item()
                n_val += 1

        val_loss = val_total / max(n_val, 1) if n_val > 0 else val_loss

        with open(args.result_file, "w") as f:
            json.dump({"val_loss": val_loss}, f)

        log.info(f"Trial complete. Val loss: {val_loss:.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
