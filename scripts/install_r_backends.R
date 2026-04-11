#!/usr/bin/env Rscript
# Install R packages required for spatial deconvolution backends
#
# Usage:
#   Rscript scripts/install_r_backends.R
#
# Or in R console:
#   source("scripts/install_r_backends.R")

cat("=== Installing R packages for spatial deconvolution ===\n\n")

# Function to check and install package
install_if_missing <- function(pkg, source = "CRAN") {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(paste0("Installing ", pkg, " from ", source, "...\n"))
        if (source == "CRAN") {
            install.packages(pkg, repos = "https://cloud.r-project.org")
        } else if (source == "Bioconductor") {
            if (!requireNamespace("BiocManager", quietly = TRUE)) {
                install.packages("BiocManager", repos = "https://cloud.r-project.org")
            }
            BiocManager::install(pkg, ask = FALSE, update = FALSE)
        }
    } else {
        cat(paste0(pkg, " already installed\n"))
    }
}

# Function to install from GitHub
install_github_if_missing <- function(pkg, repo) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(paste0("Installing ", pkg, " from GitHub (", repo, ")...\n"))
        if (!requireNamespace("devtools", quietly = TRUE)) {
            install.packages("devtools", repos = "https://cloud.r-project.org")
        }
        devtools::install_github(repo, upgrade = "never")
    } else {
        cat(paste0(pkg, " already installed\n"))
    }
}

# ============================================================================
# Core dependencies
# ============================================================================
cat("\n--- Core dependencies ---\n")
install_if_missing("Matrix", "CRAN")
install_if_missing("data.table", "CRAN")
install_if_missing("ggplot2", "CRAN")
install_if_missing("devtools", "CRAN")

# ============================================================================
# Bioconductor packages (required for SPOTlight)
# ============================================================================
cat("\n--- Bioconductor packages ---\n")
install_if_missing("BiocManager", "CRAN")
install_if_missing("SingleCellExperiment", "Bioconductor")
install_if_missing("SpatialExperiment", "Bioconductor")
install_if_missing("scater", "Bioconductor")
install_if_missing("scran", "Bioconductor")

# ============================================================================
# RCTD (spacexr)
# ============================================================================
cat("\n--- RCTD (spacexr) ---\n")
install_github_if_missing("spacexr", "dmcable/spacexr")

# ============================================================================
# CARD
# ============================================================================
cat("\n--- CARD ---\n")
install_github_if_missing("CARD", "YingMa0107/CARD")

# ============================================================================
# SPOTlight
# ============================================================================
cat("\n--- SPOTlight ---\n")
install_if_missing("SPOTlight", "Bioconductor")

# ============================================================================
# Verification
# ============================================================================
cat("\n=== Verification ===\n")

packages <- c("spacexr", "CARD", "SPOTlight", "Matrix",
              "SingleCellExperiment", "SpatialExperiment")

all_ok <- TRUE
for (pkg in packages) {
    if (requireNamespace(pkg, quietly = TRUE)) {
        version <- packageVersion(pkg)
        cat(paste0("  [OK] ", pkg, " (", version, ")\n"))
    } else {
        cat(paste0("  [FAIL] ", pkg, " - NOT INSTALLED\n"))
        all_ok <- FALSE
    }
}

if (all_ok) {
    cat("\n=== All R packages installed successfully! ===\n")
} else {
    cat("\n=== Some packages failed to install. Check errors above. ===\n")
    quit(status = 1)
}
