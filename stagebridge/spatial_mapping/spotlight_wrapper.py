"""
SPOTlight backend wrapper.

SPOTlight uses non-negative least squares (NNLS) regression with
topic modeling (NMF) to deconvolve spatial transcriptomics data.

Reference: Elosua-Bayes et al. 2021, Nucleic Acids Research
https://github.com/MarcElosworthy/SPOTlight
"""

from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
import anndata as ad
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from .backend_base import (
    SpatialBackend,
    BackendMappingResult,
    compute_cell_type_entropy,
    compute_sparsity,
)


# R script for SPOTlight
SPOTLIGHT_SCRIPT = '''
#!/usr/bin/env Rscript
# SPOTlight spatial deconvolution script
# Called from Python wrapper

suppressPackageStartupMessages({
    library(SPOTlight)
    library(Matrix)
    library(SingleCellExperiment)
    library(SpatialExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- args[1]
output_dir <- args[2]

cat("SPOTlight: Loading data from", input_dir, "\\n")

# Load reference counts (Matrix Market format)
ref_counts <- readMM(file.path(input_dir, "ref_counts.mtx"))
ref_counts <- as(ref_counts, "dgCMatrix")

# Load metadata
ref_barcodes <- read.csv(file.path(input_dir, "ref_barcodes.csv"), stringsAsFactors=FALSE)$x
ref_genes <- read.csv(file.path(input_dir, "ref_genes.csv"), stringsAsFactors=FALSE)$x
ref_celltypes <- read.csv(file.path(input_dir, "ref_celltypes.csv"), stringsAsFactors=FALSE)$cell_type

rownames(ref_counts) <- ref_genes
colnames(ref_counts) <- ref_barcodes

# Load spatial counts
spatial_counts <- readMM(file.path(input_dir, "spatial_counts.mtx"))
spatial_counts <- as(spatial_counts, "dgCMatrix")
spatial_barcodes <- read.csv(file.path(input_dir, "spatial_barcodes.csv"), stringsAsFactors=FALSE)$x
spatial_genes <- read.csv(file.path(input_dir, "spatial_genes.csv"), stringsAsFactors=FALSE)$x

rownames(spatial_counts) <- spatial_genes
colnames(spatial_counts) <- spatial_barcodes

# Load spatial coordinates
coords <- read.csv(file.path(input_dir, "spatial_coords.csv"), row.names=1)

cat("SPOTlight: Reference:", ncol(ref_counts), "cells,", nrow(ref_counts), "genes\\n")
cat("SPOTlight: Spatial:", ncol(spatial_counts), "spots,", nrow(spatial_counts), "genes\\n")
cat("SPOTlight: Cell types:", length(unique(ref_celltypes)), "\\n")

# Create SingleCellExperiment for reference
sce <- SingleCellExperiment(
    assays = list(counts = ref_counts),
    colData = data.frame(
        cell_type = ref_celltypes,
        row.names = ref_barcodes
    )
)

# Create SpatialExperiment for spatial data
# SPOTlight expects genes x spots
spe <- SpatialExperiment(
    assays = list(counts = spatial_counts),
    colData = data.frame(row.names = spatial_barcodes),
    spatialCoords = as.matrix(coords)
)

cat("SPOTlight: Finding marker genes...\\n")

# Get marker genes per cell type using Seurat-style approach
# SPOTlight includes a function for this
mgs <- findMarkers(sce, groups = sce$cell_type)

# Convert to format SPOTlight expects
mgs_df <- lapply(names(mgs), function(ct) {
    m <- mgs[[ct]]
    # Get top markers (positive log fold change)
    top_genes <- rownames(m)[order(m$Top)[1:min(100, nrow(m))]]
    data.frame(
        gene = top_genes,
        cluster = ct,
        stringsAsFactors = FALSE
    )
})
mgs_df <- do.call(rbind, mgs_df)

cat("SPOTlight: Running deconvolution...\\n")

# Run SPOTlight
res <- SPOTlight(
    x = sce,
    y = spe,
    groups = sce$cell_type,
    mgs = mgs_df,
    weight_id = "cluster",
    group_id = "cluster",
    gene_id = "gene"
)

cat("SPOTlight: Extracting results...\\n")

# Extract cell type proportions
# SPOTlight returns a matrix with spots x cell types
proportions <- res$mat
proportions_df <- as.data.frame(proportions)
rownames(proportions_df) <- spatial_barcodes

# Normalize rows to sum to 1
row_sums <- rowSums(proportions_df)
proportions_df <- proportions_df / row_sums

# Save results
cat("SPOTlight: Saving results to", output_dir, "\\n")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(proportions_df, file.path(output_dir, "proportions.csv"), row.names = TRUE)

# Save NMF topic matrix if available
if (!is.null(res$NMF)) {
    saveRDS(res$NMF, file.path(output_dir, "nmf_model.rds"))
}

cat("SPOTlight: Done!\\n")
'''


class SPOTlightBackend(SpatialBackend):
    """
    SPOTlight spatial mapping wrapper.

    Uses subprocess to call R with SPOTlight package.

    Configuration options:
    - min_cells_per_type: Minimum cells required per cell type
    - n_hvg: Number of highly variable genes to use
    """

    def __init__(
        self,
        min_cells_per_type: int = 5,
        n_hvg: int = 3000,
        r_executable: str = "Rscript",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_cells_per_type = min_cells_per_type
        self.n_hvg = n_hvg
        self.r_executable = r_executable

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run SPOTlight spatial deconvolution."""
        print("SPOTlight: Starting map()...")
        print(f"  snRNA shape: {snrna.shape}, spatial shape: {spatial.shape}")

        # Validate and preprocess
        self.validate_inputs(snrna, spatial)
        snrna, spatial = self.preprocess(snrna, spatial)
        print(f"  After preprocess: snRNA {snrna.shape}, spatial {spatial.shape}")

        # Filter rare cell types
        if self.min_cells_per_type > 0:
            cell_type_counts = snrna.obs["cell_type"].value_counts()
            rare_types = cell_type_counts[cell_type_counts < self.min_cells_per_type].index.tolist()
            if rare_types:
                print(f"  Filtering {len(rare_types)} rare cell types with < {self.min_cells_per_type} cells")
                mask = ~snrna.obs["cell_type"].isin(rare_types)
                snrna = snrna[mask].copy()
                snrna.obs["cell_type"] = snrna.obs["cell_type"].cat.remove_unused_categories()

        # Create temporary directory for R data exchange
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            r_output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            r_output_dir.mkdir()

            # Save data for R
            self._save_for_r(snrna, spatial, input_dir)

            # Write R script
            script_path = Path(tmpdir) / "run_spotlight.R"
            with open(script_path, "w") as f:
                f.write(SPOTLIGHT_SCRIPT)

            # Run R
            print("SPOTlight: Calling R...")
            cmd = [self.r_executable, str(script_path), str(input_dir), str(r_output_dir)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"SPOTlight R stderr: {result.stderr}")
                raise RuntimeError(f"SPOTlight R script failed: {result.stderr}")

            print(result.stdout)

            # Load results
            proportions = pd.read_csv(r_output_dir / "proportions.csv", index_col=0)

        # Ensure proportions index matches spatial
        proportions.index = spatial.obs_names

        # Compute confidence (use entropy - lower entropy = higher confidence)
        entropy = compute_cell_type_entropy(proportions)
        confidence = 1 - entropy
        confidence = confidence.clip(0, 1)

        # Compute metrics
        upstream_metrics = {
            "n_spots": len(spatial),
            "n_celltypes": len(proportions.columns),
            "mean_entropy": float(entropy.mean()),
            "sparsity": float(compute_sparsity(proportions)),
            "coverage": float((confidence > 0.5).mean()),
            "mean_confidence": float(confidence.mean()),
        }

        # Create result
        mapping_result = BackendMappingResult(
            cell_type_proportions=proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata={
                "backend": "spotlight",
                "n_cell_types": len(proportions.columns),
                "n_spots": len(spatial),
            },
        )

        # Save if output_dir provided
        if output_dir is not None:
            mapping_result.save(output_dir)
            print(f"SPOTlight: Results saved to {output_dir}")

        return mapping_result

    def _save_for_r(self, snrna: ad.AnnData, spatial: ad.AnnData, output_dir: Path):
        """Save data in format readable by R."""
        # Reference counts (Matrix Market) - genes x cells for R
        # Use counts layer if available, cast to int for R
        if "counts" in snrna.layers:
            X_ref = snrna.layers["counts"].T
        else:
            X_ref = snrna.X.T
        X = X_ref if hasattr(X_ref, 'toarray') else csr_matrix(X_ref)
        X = X.astype(int)
        mmwrite(output_dir / "ref_counts.mtx", X)

        # Reference metadata
        pd.DataFrame({"x": snrna.obs_names}).to_csv(output_dir / "ref_barcodes.csv", index=False)
        pd.DataFrame({"x": snrna.var_names}).to_csv(output_dir / "ref_genes.csv", index=False)
        pd.DataFrame({"cell_type": snrna.obs["cell_type"].values}).to_csv(
            output_dir / "ref_celltypes.csv", index=False
        )

        # Spatial counts (Matrix Market) - genes x spots for R
        if "counts" in spatial.layers:
            X_sp_raw = spatial.layers["counts"].T
        else:
            X_sp_raw = spatial.X.T
        X_sp = X_sp_raw if hasattr(X_sp_raw, 'toarray') else csr_matrix(X_sp_raw)
        X_sp = X_sp.astype(int)
        mmwrite(output_dir / "spatial_counts.mtx", X_sp)

        # Spatial metadata
        pd.DataFrame({"x": spatial.obs_names}).to_csv(output_dir / "spatial_barcodes.csv", index=False)
        pd.DataFrame({"x": spatial.var_names}).to_csv(output_dir / "spatial_genes.csv", index=False)

        # Spatial coordinates
        coords = pd.DataFrame(
            spatial.obsm["spatial"],
            index=spatial.obs_names,
            columns=["x", "y"],
        )
        coords.to_csv(output_dir / "spatial_coords.csv")

    def compute_upstream_metrics(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: BackendMappingResult | None,
    ) -> dict[str, float]:
        """Compute upstream quality metrics."""
        if result is None or result.cell_type_proportions is None:
            return {}

        props = result.cell_type_proportions
        return {
            "mean_entropy": float(compute_cell_type_entropy(props).mean()),
            "sparsity": float(compute_sparsity(props)),
            "coverage": float((result.confidence > 0.5).mean()) if result.confidence is not None else 0.0,
            "mean_confidence": float(result.confidence.mean()) if result.confidence is not None else 0.0,
        }

    def estimate_confidence(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        result: BackendMappingResult | None,
    ) -> pd.Series:
        """Return confidence scores."""
        if result is not None and result.confidence is not None:
            return result.confidence
        return pd.Series(np.zeros(spatial.n_obs), index=spatial.obs_names)
