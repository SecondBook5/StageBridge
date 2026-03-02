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

__all__ = [
    "HLCA_FULL_FILENAME",
    "HLCA_FULL_URL",
    "HLCAReference",
    "InterimBuildResult",
    "TangramMappingResult",
    "build_snrna_interim_anndata",
    "build_spatial_interim_anndata",
    "download_hlca_reference",
    "load_hlca_reference",
    "map_to_hlca_latent",
    "run_tangram_hlca_projection",
]
