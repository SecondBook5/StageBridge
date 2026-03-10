"""Download the LuCA extended atlas into the canonical StageBridge data tree."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from urllib.request import urlopen

from .eamist_common import (
    LUCA_ATLAS_FILENAME,
    LUCA_ATLAS_URL,
    LUCA_MODEL_FILENAME,
    LUCA_MODEL_URL,
    MIN_NONTRIVIAL_H5AD_BYTES,
    MIN_NONTRIVIAL_MODEL_BYTES,
    ensure_dir,
    parse_bool,
    utc_now_iso,
    write_json,
)


def _stream_download(url: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    bytes_written = 0
    with urlopen(url) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            bytes_written += len(chunk)
    tmp_path.replace(target)
    return int(bytes_written)


def _download_or_reuse(url: str, target: Path, *, min_bytes: int) -> tuple[str, int]:
    if target.exists():
        size = int(target.stat().st_size)
        if size >= int(min_bytes):
            return "reused_existing", size
        target.unlink()
    size = _stream_download(url, target)
    if size < int(min_bytes):
        raise ValueError(
            f"Downloaded file {target} is suspiciously small ({size} bytes); expected at least {min_bytes}."
        )
    return "downloaded", size


def _extract_model_archive(archive_path: Path, model_dir: Path) -> list[str]:
    extracted: list[str] = []
    with tarfile.open(archive_path, "r:gz") as handle:
        members = handle.getmembers()
        handle.extractall(model_dir)
        extracted = [str(member.name) for member in members]
    return extracted


def run(root: Path, *, download_model: bool) -> dict[str, object]:
    processed_root = Path(root).resolve() / "processed" / "luca"
    atlas_dir = ensure_dir(processed_root / "atlas")
    model_dir = ensure_dir(processed_root / "model")
    metadata_dir = ensure_dir(processed_root / "metadata")
    tmp_dir = ensure_dir(processed_root / "tmp")

    atlas_path = atlas_dir / LUCA_ATLAS_FILENAME
    atlas_status, atlas_bytes = _download_or_reuse(
        LUCA_ATLAS_URL,
        atlas_path,
        min_bytes=MIN_NONTRIVIAL_H5AD_BYTES,
    )

    model_status = "skipped"
    model_bytes = 0
    extracted_members: list[str] = []
    archive_path = model_dir / LUCA_MODEL_FILENAME
    if download_model:
        model_status, model_bytes = _download_or_reuse(
            LUCA_MODEL_URL,
            archive_path,
            min_bytes=MIN_NONTRIVIAL_MODEL_BYTES,
        )
        extracted_members = _extract_model_archive(archive_path, model_dir)

    manifest = {
        "created_at_utc": utc_now_iso(),
        "root": str(Path(root).resolve()),
        "processed_luca_root": str(processed_root),
        "directories": {
            "atlas": str(atlas_dir),
            "model": str(model_dir),
            "metadata": str(metadata_dir),
            "tmp": str(tmp_dir),
        },
        "atlas": {
            "url": LUCA_ATLAS_URL,
            "path": str(atlas_path),
            "status": atlas_status,
            "bytes": int(atlas_bytes),
        },
        "model": {
            "enabled": bool(download_model),
            "url": LUCA_MODEL_URL,
            "archive_path": str(archive_path),
            "status": model_status,
            "bytes": int(model_bytes),
            "extracted_members": extracted_members[:50],
        },
    }
    write_json(metadata_dir / "download_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="StageBridge data root ($STAGEBRIDGE_DATA_ROOT)")
    parser.add_argument(
        "--download-model",
        type=parse_bool,
        default=False,
        help="Whether to also download the optional LuCA core scANVI model.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = run(args.root, download_model=bool(args.download_model))
    print(f"LuCA atlas: {manifest['atlas']}")
    if manifest["model"]["enabled"]:
        print(f"LuCA model: {manifest['model']}")


if __name__ == "__main__":
    main()
