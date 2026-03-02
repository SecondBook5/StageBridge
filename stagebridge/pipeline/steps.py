"""
stagebridge/pipeline/steps.py
Stub step functions for the StageBridge benchmark pipeline.

Each function documents its contract (inputs, outputs, side-effects) and
raises ``NotImplementedError`` until a real implementation replaces it.
Stubs are intentionally thin — the heavy work lives in ``stagebridge.io``,
``stagebridge.preprocessing``, ``stagebridge.training``, etc.

Execution order (run_smoke follows this order):
    1.  audit_data
    2.  build_snrna_anndata
    3.  build_spatial_anndata
    4.  map_hlca
    5.  run_tangram
    6.  build_niche_tokens
    7.  train_benchmark
    8.  eval_benchmark
    9.  make_poster_assets
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# 1. Data audit
# ---------------------------------------------------------------------------

def audit_data(cfg: "DictConfig") -> dict[str, Any]:
    """
    Validate that all required input files exist and pass data-contract checks.

    Inputs  : cfg.data.{snrna_h5ad, spatial_h5ad, hlca_h5ad}
    Outputs : DatasetAuditReport written to outputs/tables/data_audit.json
    Returns : audit result dict (mirrors DatasetAuditReport)

    Raises
    ------
    FileNotFoundError
        If any required h5ad is missing.
    ValueError
        If an h5ad is missing required obs columns (stage, patient_id, sample_id).
    """
    # TODO: delegate to scripts/audit_data.py logic (stagebridge.utils.types.DatasetAuditReport)
    raise NotImplementedError("audit_data step not yet wired into pipeline")


# ---------------------------------------------------------------------------
# 2. Build snRNA AnnData objects
# ---------------------------------------------------------------------------

def build_snrna_anndata(cfg: "DictConfig") -> None:
    """
    Convert all GEO snRNA-seq raw files into per-sample h5ad objects and
    write a merged h5ad.

    Inputs  : cfg.data.snrna_raw_dir (*.raw_counts.mtx.txt.gz files)
    Outputs :
        - $DATA_ROOT/interim/anndata/snrna/<SAMPLE_ID>.h5ad  (per-sample)
        - $DATA_ROOT/processed/anndata/snrna_merged.h5ad     (merged)
        - $DATA_ROOT/interim/anndata/snrna/manifest.csv

    Notes
    -----
    The *.raw_counts.mtx.txt.gz format is NOT standard Matrix Market — see
    stagebridge/io/geo_snrna.py for the custom streaming parser.
    """
    # TODO: call stagebridge.io.geo_snrna.build_snrna_manifest + convert_sample + merge
    raise NotImplementedError("build_snrna_anndata step not yet wired into pipeline")


# ---------------------------------------------------------------------------
# 3. Build spatial AnnData objects
# ---------------------------------------------------------------------------

def build_spatial_anndata(cfg: "DictConfig") -> None:
    """
    Expand GEO tarballs and convert each Visium sample into an h5ad.

    Inputs  : cfg.data.spatial_raw_dir (*.tar.gz files or pre-extracted dirs)
    Outputs :
        - $DATA_ROOT/interim/anndata/spatial/<SAMPLE_ID>.h5ad  (per-sample)
        - $DATA_ROOT/processed/anndata/spatial_merged.h5ad     (merged)
        - $DATA_ROOT/interim/anndata/spatial/manifest.csv

    Notes
    -----
    Uses squidpy.read.visium() as the primary loader with a manual MTX
    fallback if squidpy is absent.
    """
    # TODO: call stagebridge.io.geo_spatial.build_spatial_manifest + convert + merge
    raise NotImplementedError("build_spatial_anndata step not yet wired into pipeline")


# ---------------------------------------------------------------------------
# 4. HLCA latent alignment
# ---------------------------------------------------------------------------

def map_hlca(cfg: "DictConfig") -> None:
    """
    Project snRNA-seq cells into the HLCA latent space.

    Reads   : $DATA_ROOT/processed/anndata/snrna_merged.h5ad
              $DATA_ROOT/data/reference/hlca/hlca_full_v1.h5ad
    Writes  : adata.obsm["X_hlca"] into snrna_merged.h5ad (in-place)

    Falls back to PCA-based projection (cfg.data.fallback_latent_key)
    if the HLCA h5ad is missing.
    """
    # TODO: call stagebridge.preprocessing.latent.build_latent(adata, method="hlca")
    raise NotImplementedError("map_hlca step not yet wired into pipeline")


# ---------------------------------------------------------------------------
# 5. Tangram spatial deconvolution
# ---------------------------------------------------------------------------

def run_tangram(cfg: "DictConfig") -> None:
    """
    Map snRNA-seq cell types onto Visium spots using Tangram.

    Reads   : snrna_merged.h5ad (with cell-type annotations)
              spatial_merged.h5ad
    Writes  : spatial_merged.h5ad with tangram deconvolution scores in obsm

    References: Biancalani et al. 2021 (Nature Methods)
    """
    # TODO: call stagebridge.io.tangram utilities
    raise NotImplementedError("run_tangram step not yet wired into pipeline")


# ---------------------------------------------------------------------------
# 6. Niche token construction
# ---------------------------------------------------------------------------

def build_niche_tokens(cfg: "DictConfig") -> None:
    """
    Build the per-transition 'source niche' token sets used by the
    Set Transformer context encoder.

    For each adjacent stage pair (src → tgt) and each donor:
    - Sample cfg.model.config.num_inducing_points context cells from src.
    - Store as a tokenised tensor dataset for the DataLoader.

    Writes  : $DATA_ROOT/processed/niche_tokens/<PAIR>.pt
    """
    # TODO: implement niche sampling logic
    raise NotImplementedError("build_niche_tokens step not yet wired into pipeline")


# ---------------------------------------------------------------------------
# 7. Train benchmark
# ---------------------------------------------------------------------------

def train_benchmark(cfg: "DictConfig") -> None:
    """
    Run the full donor-held-out cross-validation benchmark.

    Trains:
    - StageBridge (full model)
    - Ablations:  no_ot, no_context, no_stage_embedding, no_context_consistency
    - Baselines:  DeepSetsFlowModel, NoContextFlowModel, LinearTransitionBaseline

    Writes per-fold checkpoints to outputs/checkpoints/ and training
    histories to outputs/history_<run>_fold<k>.json.
    """
    # TODO: delegate to stagebridge.training.trainer.StageBridgeTrainer
    raise NotImplementedError("train_benchmark step not yet wired into pipeline")


# ---------------------------------------------------------------------------
# 8. Evaluate
# ---------------------------------------------------------------------------

def eval_benchmark(cfg: "DictConfig") -> dict[str, Any]:
    """
    Load all fold checkpoints and compute held-out metrics.

    Metrics (per transition pair, averaged across folds):
    - Sinkhorn distance  (transport quality)
    - MMD-RBF            (distribution shift)
    - Classifier 2-sample AUC
    - Composition JSD

    Writes : outputs/tables/metrics_<run>.json
             outputs/tables/run_manifest_<run>.json
    Returns: metrics dict
    """
    # TODO: delegate to stagebridge.training.eval
    raise NotImplementedError("eval_benchmark step not yet wired into pipeline")


# ---------------------------------------------------------------------------
# 9. Poster assets
# ---------------------------------------------------------------------------

def make_poster_assets(cfg: "DictConfig") -> None:
    """
    Generate all poster panels from a completed benchmark run.

    Inputs  : outputs/tables/metrics_<run>.json
    Outputs :
        - Panel A: method schematic (outputs/figures/poster_panel_a_*.pdf+png)
        - Panel B: benchmark bar chart
        - Panel C: context sensitivity (peaks at AAH/AIS — IL1B-IL1R1 niche)
        - Panel D: gene-context correlation heatmap

    See stagebridge/viz/poster.py for implementation.
    """
    # TODO: delegate to stagebridge.viz.poster.make_full_poster
    raise NotImplementedError("make_poster_assets step not yet wired into pipeline")
