"""Audit the LuCA atlas schema and export a lightweight metadata table."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata

from .eamist_common import (
    choose_best_embedding,
    infer_useful_obs_columns,
    obs_columns_h5ad,
    obsm_schema_h5ad,
    read_obs_frame_h5ad,
    read_obs_names_h5ad,
    select_luca_columns,
    summarize_obs_columns_h5ad,
    uns_keys_h5ad,
    utc_now_iso,
    var_columns_h5ad,
    write_json,
)


def run(atlas_path: Path, outdir: Path) -> dict[str, object]:
    atlas_path = Path(atlas_path).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Missing LuCA atlas file: {atlas_path}")

    try:
        adata = anndata.read_h5ad(atlas_path, backed="r")
    except Exception as exc:
        raise RuntimeError(f"Failed to open LuCA atlas in backed mode: {atlas_path}") from exc
    try:
        shape = (int(adata.n_obs), int(adata.n_vars))
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

    obs_cols = obs_columns_h5ad(atlas_path)
    var_cols = var_columns_h5ad(atlas_path)
    obsm_schema = obsm_schema_h5ad(atlas_path)
    uns_keys = uns_keys_h5ad(atlas_path)
    candidates = infer_useful_obs_columns(atlas_path)
    selected = select_luca_columns(atlas_path)
    embedding = choose_best_embedding(atlas_path)

    selected_columns = [
        selected.state_column,
        selected.major_celltype_column,
        selected.malignant_column,
        *selected.dataset_columns,
        *selected.sample_columns,
        *selected.patient_columns,
        *selected.epithelial_subtype_columns,
    ]
    selected_columns = [column for column in selected_columns if column is not None]
    selected_columns = list(dict.fromkeys(selected_columns))
    obs_frame = read_obs_frame_h5ad(atlas_path, selected_columns)
    obs_frame.insert(0, "obs_names", read_obs_names_h5ad(atlas_path).astype(str))
    obs_frame.to_parquet(outdir / "luca_obs.parquet", index=False)

    obs_schema = summarize_obs_columns_h5ad(atlas_path)
    write_json(outdir / "luca_obs_schema.json", obs_schema)
    write_json(outdir / "luca_obsm_schema.json", obsm_schema)

    report = {
        "created_at_utc": utc_now_iso(),
        "atlas_path": str(atlas_path),
        "shape": {"n_obs": int(shape[0]), "n_vars": int(shape[1])},
        "obs_columns": obs_cols,
        "var_columns": var_cols,
        "obsm_keys": list(obsm_schema.keys()),
        "uns_keys": uns_keys,
        "selected_columns": {
            "state_column": selected.state_column,
            "major_celltype_column": selected.major_celltype_column,
            "malignant_column": selected.malignant_column,
            "dataset_columns": list(selected.dataset_columns),
            "sample_columns": list(selected.sample_columns),
            "patient_columns": list(selected.patient_columns),
            "epithelial_subtype_columns": list(selected.epithelial_subtype_columns),
        },
        "candidate_columns": candidates,
        "selected_embedding": {
            "key": embedding.key,
            "source": embedding.source,
            "shape": [int(embedding.shape[0]), int(embedding.shape[1])],
        },
        "has_latent_embedding": bool(embedding.source == "obsm"),
        "metadata_export": str(outdir / "luca_obs.parquet"),
    }
    write_json(outdir / "luca_audit_report.json", report)

    print(f"adata shape: {shape}")
    print(f"obs columns ({len(obs_cols)}): {obs_cols}")
    print(f"var columns ({len(var_cols)}): {var_cols}")
    print(f"obsm keys ({len(obsm_schema)}): {list(obsm_schema.keys())}")
    print(f"uns keys ({len(uns_keys)}): {uns_keys}")
    print(f"selected state column: {selected.state_column}")
    print(f"selected embedding: {embedding.key} ({embedding.source})")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True, help="Path to luca_extended_atlas.h5ad")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for LuCA metadata/audit JSON files")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args.atlas, args.outdir)


if __name__ == "__main__":
    main()
