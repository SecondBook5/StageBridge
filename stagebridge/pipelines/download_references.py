#!/usr/bin/env python3
"""
Download HLCA and LuCA Reference Atlases

Required for dual-reference latent mapping in StageBridge V1.

Usage:
    python stagebridge/pipelines/download_references.py \
        --output_dir data/references \
        --download_hlca \
        --download_luca
"""

import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm


def download_file_with_progress(url: str, output_path: Path):
    """Download file with progress bar."""

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_hlca(output_dir: Path) -> Path:
    """
    Download Human Lung Cell Atlas (HLCA) scANVI model from scvi-tools hub.

    Uses the official scANVI model for query-to-reference mapping.
    Repository: scvi-tools/human-lung-cell-atlas-scanvi

    Returns path to the cached model directory.
    """
    print("\n" + "=" * 60)
    print("Downloading HLCA scANVI Model (Human Lung Cell Atlas)")
    print("=" * 60)

    output_dir = Path(output_dir) / "hlca"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if model already cached
    cache_dir = output_dir / "hub_cache"
    model_marker = cache_dir / "scvi-tools" / "human-lung-cell-atlas-scanvi"

    if model_marker.exists():
        print(f" HLCA scANVI model already cached: {model_marker}")
        return model_marker

    print("Downloading from: scvi-tools/human-lung-cell-atlas-scanvi (Hugging Face Hub)")
    print(f"Cache directory: {cache_dir}")
    print("This may take 10-20 minutes...")

    try:
        from scvi.hub import HubModel

        # This downloads the model and reference adata
        hubmodel = HubModel.pull_from_huggingface_hub(
            "scvi-tools/human-lung-cell-atlas-scanvi",
            cache_dir=cache_dir,
        )

        print(" Downloaded HLCA scANVI model")
        print(f"  Reference cells: {hubmodel.adata.n_obs:,}")
        print(f"  Reference genes: {hubmodel.adata.n_vars:,}")
        print(f"  Model type: {type(hubmodel.model).__name__}")

        return model_marker

    except ImportError:
        print(" Error: scvi-tools not installed")
        print("Install with: pip install scvi-tools")
        raise
    except Exception as e:
        print(f" Failed to download HLCA: {e}")
        raise


def download_luca(output_dir: Path) -> Path:
    """
    Download Lung Cancer Atlas (LuCA).

    Official repository: https://github.com/LungCancerAtlas/

    Returns path to downloaded h5ad file.
    """
    print("\n" + "=" * 60)
    print("Downloading LuCA (Lung Cancer Atlas)")
    print("=" * 60)

    output_dir = Path(output_dir) / "luca"
    output_dir.mkdir(parents=True, exist_ok=True)

    # LuCA LUAD reference (~800MB)
    # Note: Update URL when official LuCA data is released
    # For now, use placeholder or alternative source

    luca_path = output_dir / "luca_luad.h5ad"

    if luca_path.exists():
        print(f" LuCA already exists: {luca_path}")
        return luca_path

    print("  LuCA direct download not yet available")
    print("Options:")
    print("  1. Use HLCA cancer cells as proxy (included in HLCA download)")
    print("  2. Download from GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907")
    print("  3. Contact LuCA authors for access")

    # Alternative: Use HLCA cancer subset
    print("\nUsing HLCA cancer subset as LuCA proxy...")

    # For now, create symlink to HLCA (will filter cancer cells downstream)
    hlca_path = output_dir.parent / "hlca" / "hlca_core.h5ad"
    if hlca_path.exists():
        import os

        os.symlink(hlca_path, luca_path)
        print(f" Created LuCA proxy: {luca_path} -> {hlca_path}")
        print("  (Will filter cancer cells during integration)")
        return luca_path
    else:
        raise FileNotFoundError("HLCA must be downloaded first to create LuCA proxy")


def download_reference_atlases(
    output_dir: Path,
    fetch_hlca: bool = True,
    fetch_luca: bool = True,
) -> dict:
    """
    Download both HLCA and LuCA reference atlases.

    Returns:
        dict with keys 'hlca' and 'luca' pointing to downloaded files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if fetch_hlca:
        results["hlca"] = download_hlca(output_dir)
    else:
        results["hlca"] = None

    if fetch_luca:
        results["luca"] = download_luca(output_dir)
    else:
        results["luca"] = None

    print("\n" + "=" * 60)
    print(" Reference Atlas Download Complete")
    print("=" * 60)
    for key, path in results.items():
        if path:
            print(f"  {key.upper()}: {path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Download HLCA and LuCA reference atlases")

    # Use centralized data directory by default
    from pathlib import Path as PathLib

    default_output = str(PathLib.home() / "data" / "stagebridge" / "processed")

    parser.add_argument(
        "--output_dir", type=str, default=default_output, help="Output directory for references"
    )
    parser.add_argument("--download_hlca", action="store_true", help="Download HLCA")
    parser.add_argument("--download_luca", action="store_true", help="Download LuCA")
    parser.add_argument("--all", action="store_true", help="Download both HLCA and LuCA")

    args = parser.parse_args()

    if args.all:
        args.download_hlca = True
        args.download_luca = True

    if not args.download_hlca and not args.download_luca:
        print("Specify --download_hlca, --download_luca, or --all")
        return

    download_reference_atlases(
        output_dir=args.output_dir,
        fetch_hlca=args.download_hlca,
        fetch_luca=args.download_luca,
    )

    print("\n Done!")


if __name__ == "__main__":
    main()
