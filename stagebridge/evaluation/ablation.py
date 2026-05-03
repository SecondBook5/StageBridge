"""CLI entrypoint for running ablation experiments.

Usage:
    python -m stagebridge.evaluation.ablation \
        --ablation no_niche \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --fold-idx 0

    # With GW fusion figures:
    python -m stagebridge.evaluation.ablation \
        --ablation gw_barycentric \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --figures
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from stagebridge.loaders import create_dataloaders
from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.training import StageBridgeTrainer, TrainerConfig

ABLATION_CONFIGS = {
    # Full model: GW fusion enabled (the complete StageBridge architecture)
    "full": {"use_gw_fusion": True, "gw_mode": "barycentric"},
    # Niche ablation: receiver only, no spatial context
    "no_niche": {"use_niche_context": False},
    "no_distance": {"refiner_use_spatial_rpe": False},
    "no_gate": {"use_cross_attn_drift": False},  # Falls back to MLP drift
    "random_niche": {},  # Handled at data level, not config - TODO: implement
    # Reference ablations: use only one atlas reference
    "hlca_only": {"use_gw_fusion": False, "use_luca_reference": False},
    "luca_only": {"use_gw_fusion": False, "use_hlca_reference": False},
    "no_token_types": {},  # Would need model change - TODO: implement
    "frozen_encoder": {},  # Special handling: loads pretrained encoder, freezes it
    "no_ring_pooling": {"use_learned_ring_pooling": False},
    "no_context_refiner": {"use_context_refiner": False},
    # GW fusion ablations
    "no_gw_fusion": {"use_gw_fusion": False},  # Fall back to concat
    "gw_project_hlca": {"use_gw_fusion": True, "gw_mode": "project_to_hlca"},
    "gw_project_luca": {"use_gw_fusion": True, "gw_mode": "project_to_luca"},
    "gw_barycentric": {"use_gw_fusion": True, "gw_mode": "barycentric"},
    # Prototype bottleneck ablations (interpretable archetypes)
    "with_prototypes": {
        "use_niche_prototypes": True,        # Neighborhood archetypes (IL1B-high, fibrotic, etc.)
        "num_niche_prototypes": 16,
        "hierarchical_use_prototypes": True,  # Patient archetypes (progressor vs indolent)
        "hierarchical_num_prototypes": 8,
    },
}


def run_frozen_encoder_ablation(
    data_dir: Path,
    output_dir: Path,
    pretrained_checkpoint: Path,
    fold_idx: int,
    seed: int,
    transition_epochs: int,
    train_loader,
    val_loader,
    device: torch.device,
) -> dict:
    """Run frozen encoder ablation: load pretrained encoder, freeze it, train only transition head.

    This tests whether the SSL-pretrained encoder learns good representations that transfer
    to the transition task without fine-tuning. A strong frozen encoder result validates
    the SSL pretraining objective.

    Freezes: niche_tokenizer, context_refiner, hierarchical_aggregator, stats_conditioner
    Trains: drift_head, time_embedding, stage_embedding, sample_heads, pathway_head, proliferation_head
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"Loading pretrained model from {pretrained_checkpoint}")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)

    # Extract config, inferring architecture settings from state_dict
    config = StageBridgeConfig.from_checkpoint(checkpoint)
    model = StageBridge(config).to(device)

    # Load pretrained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded pretrained weights")

    # Freeze encoder components (niche encoding + context refinement)
    # Keep trainable: drift_head, time/stage embeddings, prediction heads
    encoder_modules = [
        "niche_tokenizer",
        "context_refiner",
        "hierarchical_aggregator",
        "stats_conditioner",
        "evolution_branch",  # Also freeze evolution branch if present
    ]

    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        # Freeze if parameter belongs to encoder modules
        should_freeze = any(name.startswith(enc_name) for enc_name in encoder_modules)
        if should_freeze:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()

    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Use the standard trainer but skip SSL phase (ssl_epochs=0)
    # This ensures we use the exact same training loop as train_full
    trainer_config = TrainerConfig(
        output_dir=output_dir,
        run_name="",  # Save directly to output_dir
        ssl_epochs=0,  # Skip SSL - encoder is already trained and frozen
        transition_epochs=transition_epochs,
    )
    trainer = StageBridgeTrainer(model, trainer_config, device=device)
    metrics = trainer.train(train_loader, val_loader)

    # Save checkpoint
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        checkpoint_dir / "best_checkpoint.pt",
    )

    # Save results
    result = {
        "ablation": "frozen_encoder",
        "ablation_config": {"frozen_params": frozen_params, "trainable_params": trainable_params},
        "fold_idx": fold_idx,
        "seed": seed,
        "n_parameters": frozen_params + trainable_params,
        "n_trainable_parameters": trainable_params,
        "metrics": metrics,
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "completed_at": datetime.now().isoformat(),
    }

    result_path = output_dir / "ablation_frozen_encoder.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {result_path}")
    return result, model, train_loader


def run_ablation(
    ablation: str,
    data_dir: Path,
    output_dir: Path,
    fold_idx: int = 0,
    seed: int = 42,
    ssl_epochs: int = 50,
    transition_epochs: int = 100,
    pretrained_checkpoint: Path | None = None,
    hpo_params_path: Path | None = None,
    resume_from: Path | None = None,
) -> tuple[dict, StageBridge, any]:
    """Run a single ablation experiment.

    Args:
        ablation: Name of ablation to run
        data_dir: Path to data directory
        output_dir: Path to output directory
        fold_idx: Cross-validation fold index
        seed: Random seed
        ssl_epochs: Number of SSL pretraining epochs
        transition_epochs: Number of transition training epochs
        pretrained_checkpoint: Path to pretrained checkpoint (required for frozen_encoder)
        hpo_params_path: Path to HPO best_params.json for controlled comparison
        resume_from: Path to checkpoint to resume training from
    """
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ablation '{ablation}' on {device}, fold {fold_idx}")

    # Load HPO params if provided (for fair comparison)
    hpo_params = {}
    if hpo_params_path is not None and hpo_params_path.exists():
        with open(hpo_params_path) as f:
            hpo_params = json.load(f)
        print(f"Loaded HPO params: {hpo_params}")

    if ablation not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {ablation}. Available: {list(ABLATION_CONFIGS)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, fold_idx=fold_idx, batch_size=64
    )

    # Detect evolution_dim from data
    sample_batch = next(iter(train_loader))
    evolution_dim = sample_batch.evolution_features.shape[-1] if sample_batch.evolution_features is not None else 0
    print(f"Detected evolution_dim={evolution_dim} from data")

    # Handle frozen_encoder ablation specially
    if ablation == "frozen_encoder":
        if pretrained_checkpoint is None:
            raise ValueError(
                "frozen_encoder ablation requires --pretrained-checkpoint pointing to "
                "the full model checkpoint from train_full"
            )
        return run_frozen_encoder_ablation(
            data_dir=data_dir,
            output_dir=output_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            fold_idx=fold_idx,
            seed=seed,
            transition_epochs=transition_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

    # Build model config: HPO params first, then ablation overrides
    # This ensures ablation-specific settings (like use_gw_fusion) take precedence
    model_kwargs = {}

    # Apply HPO params (architecture)
    if "hidden_dim" in hpo_params:
        model_kwargs["hidden_dim"] = hpo_params["hidden_dim"]
    if "num_heads" in hpo_params:
        model_kwargs["num_heads"] = hpo_params["num_heads"]
    if "dropout" in hpo_params:
        model_kwargs["dropout"] = hpo_params["dropout"]
    # Note: use_gw_fusion from HPO is ignored - ablation controls this

    # Apply ablation config (overrides HPO where specified)
    ablation_kwargs = ABLATION_CONFIGS[ablation].copy()
    model_kwargs.update(ablation_kwargs)

    # Override evolution_dim with detected value if evolution branch is used
    if model_kwargs.get("use_evolution_branch", True) and evolution_dim > 0:
        model_kwargs["evolution_dim"] = evolution_dim
        model_kwargs["use_evolution_branch"] = True
    elif evolution_dim == 0:
        model_kwargs["use_evolution_branch"] = False

    config = StageBridgeConfig(**model_kwargs)
    model = StageBridge(config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Model config: {model_kwargs}")
    if hpo_params:
        print(f"HPO params applied: hidden_dim={config.hidden_dim}, num_heads={config.num_heads}, dropout={config.dropout}")

    # Build trainer config with HPO params
    trainer_kwargs = {
        "output_dir": output_dir,
        "run_name": "",  # Save directly to output_dir
        "ssl_epochs": ssl_epochs,
        "transition_epochs": transition_epochs,
    }
    if "lr" in hpo_params:
        trainer_kwargs["learning_rate"] = hpo_params["lr"]
    # Note: ssl_weight from HPO is not a TrainerConfig field - HPO used it internally
    # TrainerConfig has ssl_reconstruction_weight, ssl_entropy_weight instead
    if "pathway_weight" in hpo_params:
        trainer_kwargs["pathway_weight"] = hpo_params["pathway_weight"]
    if "proliferation_weight" in hpo_params:
        trainer_kwargs["proliferation_weight"] = hpo_params["proliferation_weight"]

    trainer_config = TrainerConfig(**trainer_kwargs)
    trainer = StageBridgeTrainer(model, trainer_config, device=device)
    metrics = trainer.train(train_loader, val_loader, resume_from=resume_from)

    # Save checkpoint (matches CheckpointManager naming)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        checkpoint_dir / "best_checkpoint.pt",
    )

    # Save results
    result = {
        "ablation": ablation,
        "ablation_config": ablation_kwargs,
        "model_config": model_kwargs,
        "hpo_params": hpo_params if hpo_params else None,
        "fold_idx": fold_idx,
        "seed": seed,
        "n_parameters": n_params,
        "metrics": metrics,
        "completed_at": datetime.now().isoformat(),
    }

    result_path = output_dir / f"ablation_{ablation}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {result_path}")
    return result, model, train_loader


def generate_gw_figures(
    model: StageBridge,
    data_loader,
    output_dir: Path,
    device: torch.device,
    n_batches: int = 100,
) -> None:
    """Generate publication-quality Gromov-Wasserstein fusion visualizations.

    Creates Nature Methods-quality figures with seaborn styling showing:
    1. Reference atlas alignment via optimal transport
    2. Geometry-preserving fusion in joint embedding space
    3. Quantitative analysis of structure preservation
    4. Biological interpretation of learned weights
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import ConnectionPatch, FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.lines import Line2D
    import seaborn as sns
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr, pearsonr
    import pandas as pd

    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    if not hasattr(model, "gw_fusion") or model.gw_fusion is None:
        print("Model does not have GW fusion enabled, skipping GW figures")
        return

    model.eval()
    gw_fusion = model.gw_fusion

    # =========================================================================
    # NATURE METHODS STYLE CONFIGURATION
    # =========================================================================
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # TrueType fonts for editing
        "ps.fonttype": 42,
    })

    # Publication color palette (colorblind-friendly)
    COLORS = {
        "hlca": "#0077B6",      # Deep blue - healthy reference
        "luca": "#E63946",      # Crimson - tumor reference
        "fused": "#2A9D8F",     # Teal - fused space
        "transport": "#264653", # Dark slate - transport
        "neutral": "#6C757D",   # Gray
        "highlight": "#F4A261", # Warm accent
    }

    # Custom colormaps
    cmap_hlca = LinearSegmentedColormap.from_list("hlca", ["#FFFFFF", COLORS["hlca"]], N=256)
    cmap_luca = LinearSegmentedColormap.from_list("luca", ["#FFFFFF", COLORS["luca"]], N=256)
    cmap_density = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.2, light=0.95, as_cmap=True)

    # =========================================================================
    # DATA COLLECTION
    # =========================================================================
    all_hlca, all_luca, all_fused, all_gw_costs = [], [], [], []
    all_stages = []

    print(f"Collecting GW fusion data from {n_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= n_batches:
                break

            hlca = batch.hlca.to(device)
            luca = batch.luca.to(device)
            fused, coupling, gw_cost = gw_fusion(hlca, luca, return_coupling=True)

            all_hlca.append(hlca.cpu().numpy())
            all_luca.append(luca.cpu().numpy())
            all_fused.append(fused.squeeze(1).cpu().numpy())
            all_gw_costs.append(gw_cost.cpu().numpy())

            if hasattr(batch, "stage_pair_id"):
                all_stages.append(batch.stage_pair_id.cpu().numpy())

    hlca_all = np.concatenate(all_hlca, axis=0)
    luca_all = np.concatenate(all_luca, axis=0)
    fused_all = np.concatenate(all_fused, axis=0)
    gw_costs = np.concatenate(all_gw_costs, axis=0)
    stages = np.concatenate(all_stages, axis=0) if all_stages else None

    n_cells = len(fused_all)
    print(f"  Collected {n_cells:,} cells")

    # Subsample for visualization
    n_vis = min(8000, n_cells)
    np.random.seed(42)
    vis_idx = np.random.choice(n_cells, n_vis, replace=False)

    # =========================================================================
    # COMPUTE EMBEDDINGS
    # =========================================================================
    print("  Computing joint UMAP embedding...")
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=30, min_dist=0.25, metric="cosine",
            random_state=42, n_jobs=-1, low_memory=True
        )
        combined = np.vstack([hlca_all[vis_idx], luca_all[vis_idx], fused_all[vis_idx]])
        embedding = reducer.fit_transform(combined)
        emb_hlca = embedding[:n_vis]
        emb_luca = embedding[n_vis:2*n_vis]
        emb_fused = embedding[2*n_vis:]
        embed_name = "UMAP"
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        combined = np.vstack([hlca_all[vis_idx], luca_all[vis_idx], fused_all[vis_idx]])
        embedding = reducer.fit_transform(combined)
        emb_hlca = embedding[:n_vis]
        emb_luca = embedding[n_vis:2*n_vis]
        emb_fused = embedding[2*n_vis:]
        embed_name = "PCA"

    # Geometry preservation metrics
    n_geom = min(1000, n_vis)
    geom_idx = np.random.choice(n_vis, n_geom, replace=False)

    dist_hlca = pdist(hlca_all[vis_idx][geom_idx])
    dist_luca = pdist(luca_all[vis_idx][geom_idx])
    dist_fused = pdist(fused_all[vis_idx][geom_idx])

    rho_hlca, _ = spearmanr(dist_hlca, dist_fused)
    rho_luca, _ = spearmanr(dist_luca, dist_fused)
    r_hlca, _ = pearsonr(dist_hlca, dist_fused)
    r_luca, _ = pearsonr(dist_luca, dist_fused)

    # =========================================================================
    # FIGURE 1: MAIN MULTI-PANEL FIGURE
    # =========================================================================
    print("  Generating main figure...")

    fig = plt.figure(figsize=(7.2, 6.5))  # Nature Methods single column width
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.2, 1],
        width_ratios=[1, 1, 1],
        hspace=0.35, wspace=0.35
    )

    # Panel A: Separate reference spaces
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(emb_hlca[:, 0], emb_hlca[:, 1], s=1, alpha=0.4,
                c=COLORS["hlca"], label="HLCA", rasterized=True)
    ax_a.scatter(emb_luca[:, 0], emb_luca[:, 1], s=1, alpha=0.4,
                c=COLORS["luca"], label="LuCA", rasterized=True)

    ax_a.set_xlabel(f"{embed_name} 1")
    ax_a.set_ylabel(f"{embed_name} 2")
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    # Custom legend with larger markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["hlca"],
               markersize=6, label='HLCA (healthy)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["luca"],
               markersize=6, label='LuCA (tumor)')
    ]
    ax_a.legend(handles=legend_elements, loc='upper right', frameon=False,
               handletextpad=0.3, borderpad=0.2)

    ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax_a.set_title('Reference atlas embeddings', fontsize=9, pad=8)

    # Panel B: GW-fused space
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(emb_fused[:, 0], emb_fused[:, 1], s=1, alpha=0.5,
                c=COLORS["fused"], rasterized=True)
    ax_b.set_xlabel(f"{embed_name} 1")
    ax_b.set_ylabel(f"{embed_name} 2")
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax_b.set_title('GW-fused representation', fontsize=9, pad=8)

    # Panel C: Transport arrows
    ax_c = fig.add_subplot(gs[0, 2])

    # Background density
    ax_c.scatter(emb_hlca[:, 0], emb_hlca[:, 1], s=0.5, alpha=0.15,
                c=COLORS["hlca"], rasterized=True)
    ax_c.scatter(emb_fused[:, 0], emb_fused[:, 1], s=0.5, alpha=0.15,
                c=COLORS["fused"], rasterized=True)

    # Transport arrows (subset)
    n_arrows = 150
    arrow_idx = np.random.choice(n_vis, n_arrows, replace=False)
    for i in arrow_idx:
        dx = emb_fused[i, 0] - emb_hlca[i, 0]
        dy = emb_fused[i, 1] - emb_hlca[i, 1]
        ax_c.arrow(emb_hlca[i, 0], emb_hlca[i, 1], dx*0.85, dy*0.85,
                  head_width=0.3, head_length=0.15,
                  fc=COLORS["transport"], ec='none', alpha=0.25, linewidth=0.3)

    ax_c.set_xlabel(f"{embed_name} 1")
    ax_c.set_ylabel(f"{embed_name} 2")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax_c.set_title('Optimal transport', fontsize=9, pad=8)

    # Panel D: Geometry preservation (density plot)
    ax_d = fig.add_subplot(gs[1, 0])

    # Subsample for density plot
    n_plot = min(10000, len(dist_hlca))
    plot_idx = np.random.choice(len(dist_hlca), n_plot, replace=False)

    # Hexbin for density
    hb = ax_d.hexbin(dist_hlca[plot_idx], dist_fused[plot_idx],
                     gridsize=40, cmap=cmap_density, mincnt=1,
                     linewidths=0.1)

    # Identity line
    lims = [min(dist_hlca.min(), dist_fused.min()),
            max(dist_hlca.max(), dist_fused.max())]
    ax_d.plot(lims, lims, '--', color=COLORS["neutral"], lw=1, alpha=0.7)

    ax_d.set_xlabel('HLCA pairwise distance')
    ax_d.set_ylabel('Fused pairwise distance')
    ax_d.text(-0.15, 1.05, 'd', transform=ax_d.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax_d.set_title(f'Geometry preservation\n($\\rho$ = {rho_hlca:.3f})', fontsize=9, pad=8)

    # Panel E: GW cost distribution by stage
    ax_e = fig.add_subplot(gs[1, 1])

    if stages is not None and len(np.unique(stages)) > 1:
        stage_names = {0: "Normal\nPreinvasive", 1: "Preinvasive\nInvasive", 2: "Normal\nInvasive"}
        df_costs = pd.DataFrame({
            "GW Cost": gw_costs,
            "Transition": [stage_names.get(s, f"Stage {s}") for s in stages]
        })

        palette = [COLORS["hlca"], COLORS["fused"], COLORS["luca"]]
        sns.violinplot(data=df_costs, x="Transition", y="GW Cost",
                      palette=palette[:len(np.unique(stages))],
                      ax=ax_e, inner="box", linewidth=0.8, saturation=0.9)

        ax_e.set_xlabel('')
        ax_e.set_ylabel('GW alignment cost')
        ax_e.tick_params(axis='x', rotation=0)
    else:
        sns.histplot(gw_costs, bins=50, color=COLORS["fused"], alpha=0.7,
                    edgecolor='white', linewidth=0.5, ax=ax_e)
        ax_e.axvline(gw_costs.mean(), color=COLORS["transport"], linestyle='--',
                    lw=1.5, label=f'Mean: {gw_costs.mean():.4f}')
        ax_e.set_xlabel('GW alignment cost')
        ax_e.set_ylabel('Count')
        ax_e.legend(frameon=False)

    ax_e.text(-0.15, 1.05, 'e', transform=ax_e.transAxes,
             fontsize=12, fontweight='bold', va='top')
    ax_e.set_title('Alignment cost distribution', fontsize=9, pad=8)

    # Panel F: Learned barycentric weight
    ax_f = fig.add_subplot(gs[1, 2])

    if hasattr(gw_fusion, "alpha") and isinstance(gw_fusion.alpha, torch.nn.Parameter):
        alpha_val = torch.sigmoid(gw_fusion.alpha).item()
        hlca_weight = 1 - alpha_val
        luca_weight = alpha_val

        # Stacked horizontal bar
        ax_f.barh([0], [hlca_weight], height=0.5, color=COLORS["hlca"],
                 label=f'HLCA: {hlca_weight:.1%}', edgecolor='white', linewidth=1)
        ax_f.barh([0], [luca_weight], left=[hlca_weight], height=0.5,
                 color=COLORS["luca"], label=f'LuCA: {luca_weight:.1%}',
                 edgecolor='white', linewidth=1)

        # Center line
        ax_f.axvline(0.5, color=COLORS["neutral"], linestyle=':', lw=1, alpha=0.7)

        # Annotations
        ax_f.annotate(f'{hlca_weight:.0%}', xy=(hlca_weight/2, 0),
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     color='white' if hlca_weight > 0.3 else COLORS["transport"])
        ax_f.annotate(f'{luca_weight:.0%}', xy=(hlca_weight + luca_weight/2, 0),
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     color='white' if luca_weight > 0.3 else COLORS["transport"])

        ax_f.set_xlim(0, 1)
        ax_f.set_ylim(-0.8, 0.8)
        ax_f.set_xlabel('Barycentric weight')
        ax_f.set_yticks([])
        ax_f.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                   ncol=2, frameon=False, fontsize=7)

        # Interpretation
        if alpha_val > 0.55:
            interp = "Tumor-weighted"
        elif alpha_val < 0.45:
            interp = "Healthy-weighted"
        else:
            interp = "Balanced"
        ax_f.set_title(f'Learned fusion weight\n({interp})', fontsize=9, pad=8)
    else:
        ax_f.text(0.5, 0.5, 'Fixed weight\n(not learned)',
                 ha='center', va='center', fontsize=9)
        ax_f.axis('off')
        ax_f.set_title('Fusion weight', fontsize=9, pad=8)

    ax_f.text(-0.15, 1.05, 'f', transform=ax_f.transAxes,
             fontsize=12, fontweight='bold', va='top')

    # Save main figure
    plt.savefig(fig_dir / "gw_fusion_main.png", dpi=300, facecolor='white')
    plt.savefig(fig_dir / "gw_fusion_main.pdf", facecolor='white')
    plt.close()
    print(f"  Saved: {fig_dir / 'gw_fusion_main.pdf'}")

    # =========================================================================
    # FIGURE 2: EXTENDED GEOMETRY ANALYSIS
    # =========================================================================
    print("  Generating geometry analysis figure...")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # HLCA preservation
    axes[0].hexbin(dist_hlca[plot_idx], dist_fused[plot_idx],
                   gridsize=40, cmap=cmap_hlca, mincnt=1, linewidths=0)
    axes[0].plot([0, dist_hlca.max()], [0, dist_hlca.max()], '--',
                color=COLORS["neutral"], lw=1)
    axes[0].set_xlabel('HLCA distance')
    axes[0].set_ylabel('Fused distance')
    axes[0].set_title(f'HLCA structure\n($\\rho$ = {rho_hlca:.3f}, r = {r_hlca:.3f})', fontsize=9)
    axes[0].text(-0.15, 1.1, 'a', transform=axes[0].transAxes,
                fontsize=12, fontweight='bold', va='top')

    # LuCA preservation
    axes[1].hexbin(dist_luca[plot_idx], dist_fused[plot_idx],
                   gridsize=40, cmap=cmap_luca, mincnt=1, linewidths=0)
    axes[1].plot([0, dist_luca.max()], [0, dist_luca.max()], '--',
                color=COLORS["neutral"], lw=1)
    axes[1].set_xlabel('LuCA distance')
    axes[1].set_ylabel('Fused distance')
    axes[1].set_title(f'LuCA structure\n($\\rho$ = {rho_luca:.3f}, r = {r_luca:.3f})', fontsize=9)
    axes[1].text(-0.15, 1.1, 'b', transform=axes[1].transAxes,
                fontsize=12, fontweight='bold', va='top')

    # Summary bar chart
    metrics = pd.DataFrame({
        'Reference': ['HLCA', 'HLCA', 'LuCA', 'LuCA'],
        'Metric': ['Spearman $\\rho$', 'Pearson r', 'Spearman $\\rho$', 'Pearson r'],
        'Value': [rho_hlca, r_hlca, rho_luca, r_luca]
    })

    sns.barplot(data=metrics, x='Reference', y='Value', hue='Metric',
               palette=[COLORS["transport"], COLORS["highlight"]], ax=axes[2],
               edgecolor='white', linewidth=1)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('Correlation')
    axes[2].set_xlabel('')
    axes[2].legend(title='', frameon=False, loc='upper right')
    axes[2].axhline(0.8, color=COLORS["neutral"], linestyle=':', lw=1, alpha=0.5)
    axes[2].text(-0.15, 1.1, 'c', transform=axes[2].transAxes,
                fontsize=12, fontweight='bold', va='top')
    axes[2].set_title('Geometry preservation\nsummary', fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "gw_geometry_analysis.png", dpi=300, facecolor='white')
    plt.savefig(fig_dir / "gw_geometry_analysis.pdf", facecolor='white')
    plt.close()
    print(f"  Saved: {fig_dir / 'gw_geometry_analysis.pdf'}")

    # =========================================================================
    # FIGURE 3: SUPPLEMENTARY - DETAILED COST ANALYSIS
    # =========================================================================
    print("  Generating supplementary figures...")

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

    # Cost vs geometry preservation scatter
    if stages is not None:
        stage_names_short = {0: "N-P", 1: "P-I", 2: "N-I"}
        colors_stage = [COLORS["hlca"], COLORS["fused"], COLORS["luca"]]

        for s in sorted(np.unique(stages)):
            mask = stages == s
            axes[0].scatter(gw_costs[mask], np.random.randn(mask.sum()) * 0.02,
                           s=3, alpha=0.3, c=colors_stage[s % len(colors_stage)],
                           label=stage_names_short.get(s, f"S{s}"), rasterized=True)

        axes[0].set_xlabel('GW alignment cost')
        axes[0].set_ylabel('Jittered (for visibility)')
        axes[0].legend(frameon=False, markerscale=2, title='Transition')
        axes[0].set_title('Cost by transition type', fontsize=9)
    else:
        axes[0].hist(gw_costs, bins=50, color=COLORS["fused"], alpha=0.7, edgecolor='white')
        axes[0].set_xlabel('GW alignment cost')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Cost distribution', fontsize=9)

    # QQ plot for cost distribution
    from scipy import stats
    stats.probplot(gw_costs, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor(COLORS["fused"])
    axes[1].get_lines()[0].set_markeredgecolor('white')
    axes[1].get_lines()[0].set_markersize(3)
    axes[1].get_lines()[1].set_color(COLORS["transport"])
    axes[1].set_title('Cost normality (Q-Q)', fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "gw_supplementary.png", dpi=300, facecolor='white')
    plt.savefig(fig_dir / "gw_supplementary.pdf", facecolor='white')
    plt.close()
    print(f"  Saved: {fig_dir / 'gw_supplementary.pdf'}")

    # =========================================================================
    # SUMMARY JSON
    # =========================================================================
    summary = {
        "n_cells": int(n_cells),
        "n_cells_visualized": int(n_vis),
        "gw_cost": {
            "mean": float(gw_costs.mean()),
            "std": float(gw_costs.std()),
            "median": float(np.median(gw_costs)),
            "min": float(gw_costs.min()),
            "max": float(gw_costs.max()),
        },
        "geometry_preservation": {
            "hlca_spearman_rho": float(rho_hlca),
            "hlca_pearson_r": float(r_hlca),
            "luca_spearman_rho": float(rho_luca),
            "luca_pearson_r": float(r_luca),
        },
        "embedding_method": embed_name,
    }

    if hasattr(gw_fusion, "alpha") and isinstance(gw_fusion.alpha, torch.nn.Parameter):
        alpha_val = torch.sigmoid(gw_fusion.alpha).item()
        summary["barycentric_weight"] = {
            "alpha_raw": float(gw_fusion.alpha.item()),
            "alpha_sigmoid": float(alpha_val),
            "hlca_weight": float(1 - alpha_val),
            "luca_weight": float(alpha_val),
            "interpretation": "tumor_weighted" if alpha_val > 0.55 else "healthy_weighted" if alpha_val < 0.45 else "balanced",
        }

    with open(fig_dir / "gw_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("GW FUSION ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Cells analyzed: {n_cells:,}")
    print(f"  GW cost: {gw_costs.mean():.4f} +/- {gw_costs.std():.4f}")
    print(f"  HLCA geometry: rho={rho_hlca:.3f}, r={r_hlca:.3f}")
    print(f"  LuCA geometry: rho={rho_luca:.3f}, r={r_luca:.3f}")
    if "barycentric_weight" in summary:
        bw = summary["barycentric_weight"]
        print(f"  Barycentric: {bw['hlca_weight']:.1%} HLCA / {bw['luca_weight']:.1%} LuCA ({bw['interpretation']})")
    print(f"\n  Figures saved to: {fig_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiment")
    parser.add_argument("--ablation", required=True, help="Ablation name")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssl-epochs", type=int, default=50)
    parser.add_argument("--transition-epochs", type=int, default=100)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Path to pretrained checkpoint (required for frozen_encoder ablation)",
    )
    parser.add_argument(
        "--hpo-params",
        type=Path,
        default=None,
        help="Path to HPO best_params.json for controlled comparison (uses same lr, hidden_dim, etc.)",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Generate GW fusion figures after training (only for gw_* ablations)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from (e.g., output_dir/checkpoints/checkpoint_epoch_30.pt)",
    )
    args = parser.parse_args()

    result, model, train_loader = run_ablation(
        ablation=args.ablation,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold_idx=args.fold_idx,
        seed=args.seed,
        ssl_epochs=args.ssl_epochs,
        transition_epochs=args.transition_epochs,
        pretrained_checkpoint=args.pretrained_checkpoint,
        hpo_params_path=args.hpo_params,
        resume_from=args.resume,
    )

    # Generate GW figures if requested and applicable
    if args.figures and args.ablation.startswith("gw_"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\nGenerating GW fusion figures...")
        generate_gw_figures(
            model=model,
            data_loader=train_loader,
            output_dir=args.output_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
