"""I/O utilities for GEO snRNA-seq, spatial datasets, and HLCA references."""

from .hlca import (
    HLCA_FULL_FILENAME,
    HLCA_FULL_URL,
    HLCAReference,
    download_hlca_reference,
    load_hlca_reference,
    map_to_hlca_latent,
)
from .tangram import (
    TangramMappingResult,
    run_tangram_hlca_projection,
)
from .interim_build import (
    InterimBuildResult,
    build_snrna_interim_anndata,
    build_spatial_interim_anndata,
)
from .niche_tokens import (
    NicheTokenBank,
    NicheTokenBankBuildResult,
    NicheTokenBuildResult,
    build_niche_tokens_dataframe,
    build_zarr_token_bank,
    write_niche_token_outputs,
    write_tokens_to_h5ad,
)

__all__ = [
    "HLCA_FULL_FILENAME",
    "HLCA_FULL_URL",
    "HLCAReference",
    "InterimBuildResult",
    "NicheTokenBank",
    "NicheTokenBankBuildResult",
    "NicheTokenBuildResult",
    "TangramMappingResult",
    "build_niche_tokens_dataframe",
    "build_snrna_interim_anndata",
    "build_spatial_interim_anndata",
    "build_zarr_token_bank",
    "download_hlca_reference",
    "load_hlca_reference",
    "map_to_hlca_latent",
    "run_tangram_hlca_projection",
    "write_niche_token_outputs",
    "write_tokens_to_h5ad",
]
