"""
RCTD (Robust Cell Type Decomposition) backend wrapper.

RCTD is an R package (spacexr) that uses a Bayesian approach to estimate
cell type proportions with robust handling of rare cell types.

Reference: Cable et al. 2021, Nature Biotechnology
https://github.com/dmcable/spacexr
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


# R script for RCTD
RCTD_SCRIPT = '''
#!/usr/bin/env Rscript
# RCTD spatial deconvolution script
# Called from Python wrapper

suppressPackageStartupMessages({
    library(spacexr)
    library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- args[1]
output_dir <- args[2]
mode <- ifelse(length(args) >= 3, args[3], "doublet")

cat("RCTD: Loading data from", input_dir, "\\n")

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

cat("RCTD: Reference:", ncol(ref_counts), "cells,", nrow(ref_counts), "genes\\n")
cat("RCTD: Spatial:", ncol(spatial_counts), "spots,", nrow(spatial_counts), "genes\\n")
cat("RCTD: Cell types:", length(unique(ref_celltypes)), "\\n")

# Create Reference object
cell_types <- factor(ref_celltypes)
names(cell_types) <- ref_barcodes
nUMI <- colSums(ref_counts)
names(nUMI) <- ref_barcodes

reference <- Reference(ref_counts, cell_types, nUMI)

# Create SpatialRNA object
nUMI_spatial <- colSums(spatial_counts)
puck <- SpatialRNA(coords, spatial_counts, nUMI_spatial)

cat("RCTD: Running RCTD in", mode, "mode...\\n")

# Run RCTD
myRCTD <- create.RCTD(puck, reference, max_cores = 4)
myRCTD <- run.RCTD(myRCTD, doublet_mode = mode)

cat("RCTD: Extracting results...\\n")

# Extract results based on mode
if (mode == "full") {
    # Full mode: get all cell type weights
    weights <- myRCTD@results$weights
    weights_df <- as.data.frame(as.matrix(weights))
} else {
    # Doublet mode: normalize weights
    weights <- normalize_weights(myRCTD@results$weights)
    weights_df <- as.data.frame(as.matrix(weights))
}

# Normalize rows to sum to 1
row_sums <- rowSums(weights_df)
weights_df <- weights_df / row_sums

# Save results
cat("RCTD: Saving results to", output_dir, "\\n")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(weights_df, file.path(output_dir, "proportions.csv"), row.names = TRUE)

# Save confidence scores (convergence)
if (!is.null(myRCTD@results$results_df)) {
    results_df <- myRCTD@results$results_df
    write.csv(results_df, file.path(output_dir, "rctd_results.csv"), row.names = TRUE)
}

cat("RCTD: Done!\\n")
'''


class RCTDBackend(SpatialBackend):
    """
    RCTD (Robust Cell Type Decomposition) spatial mapping wrapper.

    Uses subprocess to call R with spacexr package.

    Configuration options:
    - mode: "doublet" (default, 1-2 cell types per spot) or "full" (all cell types)
    - min_cells_per_type: Minimum cells required per cell type
    """

    def __init__(
        self,
        mode: str = "doublet",
        min_cells_per_type: int = 5,
        r_executable: str = "Rscript",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.min_cells_per_type = min_cells_per_type
        self.r_executable = r_executable

    def map(
        self,
        snrna: ad.AnnData,
        spatial: ad.AnnData,
        output_dir: Path | None = None,
    ) -> BackendMappingResult:
        """Run RCTD spatial deconvolution."""
        print(f"RCTD: Starting map() in {self.mode} mode...")
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
            script_path = Path(tmpdir) / "run_rctd.R"
            with open(script_path, "w") as f:
                f.write(RCTD_SCRIPT)

            # Run R
            print(f"RCTD: Calling R...")
            cmd = [self.r_executable, str(script_path), str(input_dir), str(r_output_dir), self.mode]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"RCTD R stderr: {result.stderr}")
                raise RuntimeError(f"RCTD R script failed: {result.stderr}")

            print(result.stdout)

            # Load results
            proportions = pd.read_csv(r_output_dir / "proportions.csv", index_col=0)

        # Ensure proportions index matches spatial
        proportions.index = spatial.obs_names

        # Compute confidence (use entropy - lower entropy = higher confidence)
        entropy = compute_cell_type_entropy(proportions)
        confidence = 1 - entropy  # Invert: low entropy = high confidence
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
        result = BackendMappingResult(
            cell_type_proportions=proportions,
            confidence=confidence,
            upstream_metrics=upstream_metrics,
            metadata={
                "backend": "rctd",
                "mode": self.mode,
                "n_cell_types": len(proportions.columns),
                "n_spots": len(spatial),
            },
        )

        # Save if output_dir provided
        if output_dir is not None:
            result.save(output_dir)
            print(f"RCTD: Results saved to {output_dir}")

        return result

    def _save_for_r(self, snrna: ad.AnnData, spatial: ad.AnnData, output_dir: Path):
        """Save data in format readable by R."""
        # Reference counts (Matrix Market)
        X = snrna.X.T if hasattr(snrna.X, 'toarray') else csr_matrix(snrna.X.T)
        mmwrite(output_dir / "ref_counts.mtx", X)

        # Reference metadata
        pd.DataFrame({"x": snrna.obs_names}).to_csv(output_dir / "ref_barcodes.csv", index=False)
        pd.DataFrame({"x": snrna.var_names}).to_csv(output_dir / "ref_genes.csv", index=False)
        pd.DataFrame({"cell_type": snrna.obs["cell_type"].values}).to_csv(
            output_dir / "ref_celltypes.csv", index=False
        )

        # Spatial counts (Matrix Market)
        X_sp = spatial.X.T if hasattr(spatial.X, 'toarray') else csr_matrix(spatial.X.T)
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
