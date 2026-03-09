"""Shared helpers for EA-MIST feature builders and LuCA/HLCA preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import anndata
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from stagebridge.data.luad_evo.stages import CANONICAL_STAGE_ORDER, normalize_stage_label, stage_to_index

LUCA_ATLAS_URL = "https://datasets.cellxgene.cziscience.com/f678fb47-e51b-4dc5-b23f-f9df43a67ee5.h5ad"
LUCA_MODEL_URL = "https://zenodo.org/records/7227571/files/core_atlas_scanvi_model.tar.gz?download=1"
LUCA_ATLAS_FILENAME = "luca_extended_atlas.h5ad"
LUCA_MODEL_FILENAME = "core_atlas_scanvi_model.tar.gz"
MIN_NONTRIVIAL_H5AD_BYTES = 100_000_000
MIN_NONTRIVIAL_MODEL_BYTES = 1_000_000
DEFAULT_LUCA_TOP_K = 5
DEFAULT_HLCA_TOP_K = 5
DEFAULT_RING_EDGES = (0.0, 50.0, 100.0, 150.0, 200.0)
EAMIST_BAG_SCHEMA_VERSION = "eamist_bag_v1_rings4_multitask"
WEAK_STAGE_ORDINAL_SUPERVISION = "weak_stage_ordered_displacement"

TOKEN_LABELS: tuple[str, ...] = (
    "AT2",
    "Basal",
    "Capillary",
    "Ciliated",
    "Fibroblast lineage",
    "Macrophages",
    "Mast cells",
    "Secretory",
    "T cell lineage",
)
TOKEN_LINEAGES: dict[str, tuple[str, ...]] = {
    "epithelial": ("AT2", "Basal", "Ciliated", "Secretory"),
    "immune": ("Macrophages", "Mast cells", "T cell lineage"),
    "stromal_endothelial": ("Capillary", "Fibroblast lineage"),
}
TOKEN_KEYWORD_MAP: dict[str, tuple[str, ...]] = {
    "AT2": ("at2", "alveolar type ii", "type ii", "alveolar"),
    "Basal": ("basal", "squamous", "krt5", "krt17"),
    "Capillary": ("capillary", "endothelial", "vascular", "venous", "arterial", "lymphatic"),
    "Ciliated": ("ciliated",),
    "Fibroblast lineage": ("fibro", "stromal", "mesench", "pericyte", "myofibro", "caf"),
    "Macrophages": ("macrophage", "monocyte", "myeloid", "dendritic", "neutrophil"),
    "Mast cells": ("mast",),
    "Secretory": ("secretory", "club", "goblet", "mucous"),
    "T cell lineage": ("t cell", "lymph", "nk", "b cell", "plasma"),
}
MALIGNANT_KEYWORDS: tuple[str, ...] = (
    "malignant",
    "tumor",
    "tumour",
    "cancer",
    "neoplastic",
    "carcinoma",
    "nsclc",
    "luad",
)
INVASIVE_KEYWORDS: tuple[str, ...] = (
    "invasive",
    "emt",
    "mesench",
    "migration",
    "basal",
    "stress",
    "hypoxia",
    "prolif",
    "dediffer",
)
IMMUNE_KEYWORDS: tuple[str, ...] = (
    "immune",
    "macrophage",
    "myeloid",
    "t cell",
    "b cell",
    "lymph",
    "mast",
    "neutrophil",
    "dendritic",
    "monocyte",
    "nk",
    "plasma",
)
STROMAL_KEYWORDS: tuple[str, ...] = (
    "strom",
    "fibro",
    "mesench",
    "pericyte",
    "smooth muscle",
    "myofibro",
    "endothelial",
    "vascular",
    "capillary",
)
EPITHELIAL_KEYWORDS: tuple[str, ...] = (
    "epithelial",
    "at1",
    "at2",
    "basal",
    "club",
    "ciliated",
    "secretory",
    "alveolar",
)


@dataclass(slots=True, frozen=True)
class SelectedLucaColumns:
    """Resolved LuCA metadata columns for downstream reference construction."""

    state_column: str
    major_celltype_column: str | None
    malignant_column: str | None
    dataset_columns: tuple[str, ...]
    sample_columns: tuple[str, ...]
    patient_columns: tuple[str, ...]
    epithelial_subtype_columns: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SelectedEmbedding:
    """Resolved LuCA embedding choice."""

    key: str
    source: str
    shape: tuple[int, int]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    normalized = str(text).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value from {text!r}.")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def decode_h5_strings(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O"}:
        return np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in arr], dtype=object)
    return arr.astype(object, copy=False)


def obs_columns_h5ad(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return [str(key) for key in handle["obs"].keys() if str(key) != "_index"]


def var_columns_h5ad(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return [str(key) for key in handle["var"].keys() if str(key) != "_index"]


def obsm_schema_h5ad(path: Path) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        group = handle.get("obsm")
        if group is None:
            return schema
        for key in group.keys():
            obj = group[key]
            shape = matrix_shape(obj)
            schema[str(key)] = {
                "encoding_type": str(obj.attrs.get("encoding-type", "unknown")),
                "shape": None if shape is None else [int(shape[0]), int(shape[1])],
                "dtype": str(getattr(obj, "dtype", "unknown")),
            }
    return schema


def uns_keys_h5ad(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        group = handle.get("uns")
        if group is None:
            return []
        return [str(key) for key in group.keys()]


def read_obs_names_h5ad(path: Path) -> pd.Index:
    with h5py.File(path, "r") as handle:
        index_values = decode_h5_strings(handle["obs"]["_index"][()])
    return pd.Index(index_values.astype(str), name="obs_names")


def read_obs_column_h5ad(path: Path, column: str) -> pd.Series:
    with h5py.File(path, "r") as handle:
        obs_group = handle["obs"]
        if column not in obs_group:
            raise KeyError(f"Column '{column}' not found in obs of {path}.")
        obs_names = decode_h5_strings(obs_group["_index"][()]).astype(str)
        obj = obs_group[column]
        if isinstance(obj, h5py.Group):
            encoding = str(obj.attrs.get("encoding-type", ""))
            if encoding == "categorical":
                categories = pd.Index(decode_h5_strings(obj["categories"][()]).astype(str))
                codes = np.asarray(obj["codes"][()], dtype=np.int64)
                safe_codes = np.where(codes < 0, -1, codes)
                values = pd.Categorical.from_codes(
                    safe_codes,
                    categories=categories,
                    ordered=bool(obj.attrs.get("ordered", False)),
                )
            else:
                raise TypeError(f"Unsupported obs group encoding '{encoding}' for column '{column}'.")
        else:
            values = decode_h5_strings(obj[()])
    return pd.Series(values, index=pd.Index(obs_names, name="obs_names"), name=str(column))


def summarize_obs_columns_h5ad(path: Path, *, max_sample_values: int = 12) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        obs_group = handle["obs"]
        n_obs = int(obs_group["_index"].shape[0])
        for column in obs_group.keys():
            if str(column) == "_index":
                continue
            obj = obs_group[str(column)]
            entry: dict[str, Any] = {
                "n_obs": n_obs,
                "encoding_type": str(obj.attrs.get("encoding-type", "unknown")),
            }
            if isinstance(obj, h5py.Group) and str(obj.attrs.get("encoding-type", "")) == "categorical":
                categories = decode_h5_strings(obj["categories"][()]).astype(str)
                entry["dtype"] = "category"
                entry["n_unique"] = int(categories.shape[0])
                entry["sample_values"] = categories[: int(max_sample_values)].tolist()
                try:
                    codes = np.asarray(obj["codes"][()], dtype=np.int64)
                    entry["null_count"] = int(np.count_nonzero(codes < 0))
                except Exception:
                    entry["null_count"] = 0
            else:
                sample = decode_h5_strings(obj[: int(max_sample_values)]).astype(str)
                entry["dtype"] = str(getattr(obj, "dtype", "unknown"))
                entry["n_unique"] = int(pd.Index(sample).nunique())
                entry["sample_values"] = pd.Index(sample).drop_duplicates().tolist()
                entry["null_count"] = 0
            summary[str(column)] = entry
    return summary


def read_obs_frame_h5ad(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    obs_names = read_obs_names_h5ad(path)
    frame = pd.DataFrame(index=obs_names)
    for column in columns:
        frame[str(column)] = read_obs_column_h5ad(path, str(column)).reindex(obs_names)
    return frame


def matrix_shape(obj: h5py.Dataset | h5py.Group) -> tuple[int, int] | None:
    if isinstance(obj, h5py.Dataset):
        if len(obj.shape) == 2:
            return int(obj.shape[0]), int(obj.shape[1])
        return None
    encoding = str(obj.attrs.get("encoding-type", ""))
    if encoding in {"csr_matrix", "csc_matrix"}:
        shape = obj.attrs.get("shape")
        if shape is None:
            return None
        return int(shape[0]), int(shape[1])
    return None


def read_matrix_chunk(obj: h5py.Dataset | h5py.Group, start: int, stop: int) -> np.ndarray:
    if start >= stop:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(obj, h5py.Dataset):
        return np.asarray(obj[start:stop], dtype=np.float32)
    encoding = str(obj.attrs.get("encoding-type", ""))
    if encoding not in {"csr_matrix", "csc_matrix"}:
        raise TypeError(f"Unsupported matrix encoding '{encoding}'.")
    shape = matrix_shape(obj)
    if shape is None:
        raise ValueError("Sparse matrix group was missing shape metadata.")
    if encoding == "csr_matrix":
        indptr = np.asarray(obj["indptr"][start : stop + 1], dtype=np.int64)
        first = int(indptr[0])
        last = int(indptr[-1])
        data = np.asarray(obj["data"][first:last], dtype=np.float32)
        indices = np.asarray(obj["indices"][first:last], dtype=np.int64)
        chunk_indptr = indptr - first
        csr = sp.csr_matrix((data, indices, chunk_indptr), shape=(stop - start, shape[1]))
        return csr.toarray().astype(np.float32, copy=False)
    raise TypeError(f"Unsupported sparse matrix encoding '{encoding}'.")


def choose_best_embedding(path: Path) -> SelectedEmbedding:
    obsm_schema = obsm_schema_h5ad(path)
    ranked: list[tuple[float, str, tuple[int, int]]] = []
    for key, info in obsm_schema.items():
        shape = info.get("shape")
        if shape is None:
            continue
        n_obs, n_dim = int(shape[0]), int(shape[1])
        lower = key.lower()
        score = 0.0
        source = "obsm"
        if any(token in lower for token in ("latent", "scvi", "scanvi", "embedding", "embed")):
            score += 30.0
        if "pca" in lower:
            score += 20.0
        if any(token in lower for token in ("umap", "tsne", "phate")):
            score -= 20.0
        if n_dim >= 8:
            score += 5.0
        if n_dim < 3:
            score -= 25.0
        score += min(n_dim, 256) / 256.0
        ranked.append((score, key, (n_obs, n_dim)))
    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        _score, key, shape = ranked[0]
        return SelectedEmbedding(key=str(key), source="obsm", shape=shape)

    adata = anndata.read_h5ad(path, backed="r")
    try:
        shape = tuple(int(value) for value in adata.shape)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    if len(shape) != 2 or shape[1] <= 1:
        raise ValueError(f"Could not resolve a useful LuCA embedding from {path}.")
    return SelectedEmbedding(key="X", source="X", shape=(int(shape[0]), int(shape[1])))


def infer_useful_obs_columns(path: Path, *, max_sample_values: int = 12) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {
        "state_columns": [],
        "major_celltype_columns": [],
        "malignant_columns": [],
        "dataset_columns": [],
        "sample_columns": [],
        "patient_columns": [],
        "epithelial_subtype_columns": [],
    }
    obs_summary = summarize_obs_columns_h5ad(path, max_sample_values=max_sample_values)
    for column, schema in obs_summary.items():
        unique_count = int(schema.get("n_unique", 0))
        sample_values = [str(value) for value in schema.get("sample_values", [])]
        if not sample_values and unique_count <= 0:
            continue
        lower_name = str(column).lower()
        lower_values = " ".join(sample_values).lower()
        entry = {
            "column": str(column),
            "n_unique": unique_count,
            "sample_values": sample_values,
        }
        state_score = score_state_column(lower_name, lower_values, unique_count)
        if state_score > 0:
            candidates["state_columns"].append({"score": state_score, **entry})
        major_score = score_major_celltype_column(lower_name, lower_values, unique_count)
        if major_score > 0:
            candidates["major_celltype_columns"].append({"score": major_score, **entry})
        malignant_score = score_metadata_kind(lower_name, lower_values, kind="malignant")
        if malignant_score > 0:
            candidates["malignant_columns"].append({"score": malignant_score, **entry})
        dataset_score = score_metadata_kind(lower_name, lower_values, kind="dataset")
        if dataset_score > 0:
            candidates["dataset_columns"].append({"score": dataset_score, **entry})
        sample_score = score_metadata_kind(lower_name, lower_values, kind="sample")
        if sample_score > 0:
            candidates["sample_columns"].append({"score": sample_score, **entry})
        patient_score = score_metadata_kind(lower_name, lower_values, kind="patient")
        if patient_score > 0:
            candidates["patient_columns"].append({"score": patient_score, **entry})
        epithelial_score = score_metadata_kind(lower_name, lower_values, kind="epithelial_subtype")
        if epithelial_score > 0:
            candidates["epithelial_subtype_columns"].append({"score": epithelial_score, **entry})

    for key in candidates:
        candidates[key].sort(key=lambda item: (float(item["score"]), int(item["n_unique"])), reverse=True)
    return candidates


def score_state_column(lower_name: str, lower_values: str, unique_count: int) -> float:
    score = 0.0
    if lower_name == "cell_type_tumor":
        score += 24.0
    if "ann_fine" in lower_name or lower_name.endswith("fine"):
        score += 18.0
    if "tumor" in lower_name or "tumour" in lower_name:
        score += 8.0
    if any(token in lower_name for token in ("state", "subtype", "cell_state", "annotation_level_3", "annotation_level_2")):
        score += 12.0
    if any(token in lower_name for token in ("cell_type", "celltype", "cluster", "lineage", "compartment", "broad", "major")):
        score += 6.0
    if "tumor cells" in lower_values or "malignant" in lower_values:
        score += 6.0
    if any(token in lower_values for token in MALIGNANT_KEYWORDS + IMMUNE_KEYWORDS + STROMAL_KEYWORDS + EPITHELIAL_KEYWORDS):
        score += 4.0
    if 4 <= unique_count <= 500:
        score += 4.0
    elif 2 <= unique_count <= 2000:
        score += 1.0
    return score


def score_major_celltype_column(lower_name: str, lower_values: str, unique_count: int) -> float:
    score = 0.0
    if "ann_coarse" in lower_name or lower_name.endswith("coarse"):
        score += 18.0
    if any(token in lower_name for token in ("major", "broad", "lineage", "compartment", "cell_type", "celltype", "annotation_level_1")):
        score += 10.0
    if any(token in lower_values for token in IMMUNE_KEYWORDS + STROMAL_KEYWORDS + EPITHELIAL_KEYWORDS):
        score += 4.0
    if 3 <= unique_count <= 100:
        score += 3.0
    return score


def score_metadata_kind(lower_name: str, lower_values: str, *, kind: str) -> float:
    keyword_map = {
        "malignant": ("malignant", "malignancy", "tumor_flag", "tumour_flag", "neoplastic", "predicted"),
        "dataset": ("dataset", "study", "cohort", "source", "project", "batch"),
        "sample": ("sample", "specimen", "library", "biosample"),
        "patient": ("patient", "donor", "subject", "case", "individual"),
        "epithelial_subtype": ("epithelial", "alveolar", "basal", "ciliated", "secretory", "club"),
    }
    value_map = {
        "malignant": MALIGNANT_KEYWORDS,
        "dataset": (),
        "sample": (),
        "patient": (),
        "epithelial_subtype": EPITHELIAL_KEYWORDS,
    }
    score = 0.0
    if any(token in lower_name for token in keyword_map[kind]):
        score += 10.0
    if kind == "malignant" and "malignant" in lower_values:
        score += 8.0
    elif value_map[kind] and any(token in lower_values for token in value_map[kind]):
        score += 3.0
    return score


def select_luca_columns(path: Path) -> SelectedLucaColumns:
    candidates = infer_useful_obs_columns(path)
    state_candidates = candidates["state_columns"]
    if not state_candidates:
        raise ValueError("Could not detect a useful LuCA state annotation column.")
    major_candidates = candidates["major_celltype_columns"]
    malignant_candidates = candidates["malignant_columns"]
    dataset_candidates = candidates["dataset_columns"]
    sample_candidates = candidates["sample_columns"]
    patient_candidates = candidates["patient_columns"]
    epithelial_candidates = candidates["epithelial_subtype_columns"]
    state_column = None
    if any(str(entry["column"]) == "cell_type_tumor" for entry in state_candidates):
        state_column = "cell_type_tumor"
    elif any(str(entry["column"]) == "ann_fine" for entry in state_candidates):
        state_column = "ann_fine"
    else:
        state_column = str(state_candidates[0]["column"])
    major_celltype_column = None
    if any(str(entry["column"]) == "ann_coarse" for entry in major_candidates):
        major_celltype_column = "ann_coarse"
    else:
        for entry in major_candidates:
            column = str(entry["column"])
            if column != state_column:
                major_celltype_column = column
                break
    malignant_column = None
    for entry in malignant_candidates:
        column = str(entry["column"])
        if column in {state_column, major_celltype_column}:
            continue
        if "malignant" in column.lower() or "predicted" in column.lower():
            malignant_column = column
            break
    return SelectedLucaColumns(
        state_column=state_column,
        major_celltype_column=major_celltype_column,
        malignant_column=malignant_column,
        dataset_columns=tuple(str(entry["column"]) for entry in dataset_candidates[:3]),
        sample_columns=tuple(str(entry["column"]) for entry in sample_candidates[:3]),
        patient_columns=tuple(str(entry["column"]) for entry in patient_candidates[:3]),
        epithelial_subtype_columns=tuple(str(entry["column"]) for entry in epithelial_candidates[:3]),
    )


def infer_token_profile(*texts: str | None) -> dict[str, float]:
    joined = " ".join(str(text) for text in texts if text is not None).lower()
    weights = {label: 0.0 for label in TOKEN_LABELS}
    for label, keywords in TOKEN_KEYWORD_MAP.items():
        weight = 0.0
        for keyword in keywords:
            if keyword in joined:
                weight += 1.0
        if weight > 0.0:
            weights[label] = weight
    if sum(weights.values()) <= 0.0:
        if any(keyword in joined for keyword in IMMUNE_KEYWORDS):
            weights["T cell lineage"] = 0.5
            weights["Macrophages"] = 0.5
        elif any(keyword in joined for keyword in STROMAL_KEYWORDS):
            weights["Fibroblast lineage"] = 0.5
            weights["Capillary"] = 0.5
        else:
            weights["AT2"] = 0.25
            weights["Basal"] = 0.25
            weights["Ciliated"] = 0.25
            weights["Secretory"] = 0.25
    total = float(sum(weights.values()))
    return {label: float(value / total) for label, value in weights.items()}


def infer_state_grouping(*texts: str | None) -> dict[str, Any]:
    joined = " ".join(str(text) for text in texts if text is not None).lower()
    malignant = any(token in joined for token in MALIGNANT_KEYWORDS)
    immune = any(token in joined for token in IMMUNE_KEYWORDS)
    stromal = any(token in joined for token in STROMAL_KEYWORDS)
    epithelial = any(token in joined for token in EPITHELIAL_KEYWORDS)
    if malignant:
        compartment = "malignant"
    elif immune:
        compartment = "immune"
    elif stromal:
        compartment = "stromal"
    elif epithelial:
        compartment = "epithelial"
    else:
        compartment = "unknown"

    major_lineage = "unknown"
    if any(token in joined for token in ("at2", "alveolar")):
        major_lineage = "AT2_like"
    elif "basal" in joined:
        major_lineage = "Basal_like"
    elif "secretory" in joined or "club" in joined:
        major_lineage = "Secretory_like"
    elif "ciliated" in joined:
        major_lineage = "Ciliated_like"
    elif any(token in joined for token in ("t cell", "b cell", "lymph", "nk", "plasma")):
        major_lineage = "Lymphoid"
    elif any(token in joined for token in ("macrophage", "myeloid", "monocyte", "dendritic", "neutrophil")):
        major_lineage = "Myeloid"
    elif any(token in joined for token in ("fibro", "stromal", "mesench", "pericyte", "myofibro")):
        major_lineage = "Fibro_stromal"
    elif any(token in joined for token in ("capillary", "endothelial", "vascular")):
        major_lineage = "Endothelial"

    epithelial_subtype = None
    for label in ("AT2", "Basal", "Ciliated", "Secretory"):
        if label.lower() in joined:
            epithelial_subtype = label
            break

    return {
        "compartment_group": compartment,
        "major_lineage_tag": major_lineage,
        "malignant_flag": bool(malignant),
        "immune_flag": bool(immune),
        "stromal_flag": bool(stromal),
        "epithelial_flag": bool(epithelial),
        "invasive_like_flag": bool(any(token in joined for token in INVASIVE_KEYWORDS)),
        "epithelial_subtype_label": epithelial_subtype,
    }


def safe_probability_rows(matrix: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    row_sum = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, row_sum, out=np.full_like(arr, np.float32(1.0 / max(arr.shape[1], 1))), where=row_sum > eps)


def entropy_from_rows(matrix: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    probs = safe_probability_rows(matrix, eps=eps)
    return -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1).astype(np.float32, copy=False)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
    denom = np.clip(left_norm * right_norm, eps, None)
    return (left @ right.T) / denom


def choose_niche_token_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], str]:
    smooth_cols = [str(col) for col in frame.columns if str(col).startswith("tok_smooth_")]
    raw_cols = [str(col) for col in frame.columns if str(col).startswith("tok_") and not str(col).startswith("tok_smooth_")]
    if smooth_cols:
        return smooth_cols, [column.removeprefix("tok_smooth_") for column in smooth_cols], "tok_smooth_"
    if raw_cols:
        return raw_cols, [column.removeprefix("tok_") for column in raw_cols], "tok_"
    raise ValueError("Could not detect token columns in niche parquet.")


def normalize_niche_table(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "sample_id" not in df.columns and "lesion_id" in df.columns:
        df["sample_id"] = df["lesion_id"].astype(str)
    if "lesion_id" not in df.columns and "sample_id" in df.columns:
        df["lesion_id"] = df["sample_id"].astype(str)
    if "patient_id" not in df.columns and "donor_id" in df.columns:
        df["patient_id"] = df["donor_id"].astype(str)
    if "donor_id" not in df.columns and "patient_id" in df.columns:
        df["donor_id"] = df["patient_id"].astype(str)
    required = {"lesion_id", "sample_id", "donor_id", "patient_id", "stage"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Niche parquet is missing required columns: {sorted(missing)}")
    if "x" not in df.columns or "y" not in df.columns:
        raise KeyError("Niche parquet is missing spatial coordinates 'x'/'y'.")
    if "spot_id" in df.columns:
        spot_ids = df["spot_id"].astype(str)
        sample_ids = df["sample_id"].astype(str)
        niche_ids = np.asarray(
            [
                spot_id if ":" in spot_id else f"{sample_id}:{spot_id}"
                for sample_id, spot_id in zip(sample_ids.tolist(), spot_ids.tolist(), strict=False)
            ],
            dtype=object,
        )
    else:
        niche_ids = df.index.astype(str)
    df["niche_id"] = niche_ids.astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    df["donor_id"] = df["donor_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["stage"] = df["stage"].astype(str).map(normalize_stage_label)
    if df["niche_id"].duplicated().any():
        raise ValueError("Detected duplicate niche identifiers in niche parquet.")
    return df


def topk_labels_and_scores(similarity: np.ndarray, labels: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    if similarity.ndim != 2:
        raise ValueError(f"Expected a 2D similarity matrix, got shape={similarity.shape}.")
    top_k = min(int(k), similarity.shape[1])
    order = np.argsort(similarity, axis=1)[:, ::-1][:, :top_k]
    top_scores = np.take_along_axis(similarity, order, axis=1).astype(np.float32, copy=False)
    label_array = np.asarray([str(label) for label in labels], dtype=object)
    top_labels = label_array[order]
    return top_scores, top_labels


def numeric_feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if not str(column).startswith(prefix):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(str(column))
    return columns


def align_feature_rows(base: pd.DataFrame, feature_df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    required = {"lesion_id", "niche_id"}
    missing = required.difference(feature_df.columns)
    if missing:
        raise KeyError(f"{source} is missing required key columns: {sorted(missing)}")
    if feature_df.duplicated(["lesion_id", "niche_id"]).any():
        raise ValueError(f"{source} contains duplicate lesion_id/niche_id rows.")
    new_columns = [column for column in feature_df.columns if column not in {"lesion_id", "niche_id"} and column not in base.columns]
    if not new_columns:
        return base.merge(feature_df.loc[:, ["lesion_id", "niche_id"]], on=["lesion_id", "niche_id"], how="left", validate="one_to_one")
    merge_frame = feature_df.loc[:, ["lesion_id", "niche_id", *new_columns]].copy()
    merged = base.merge(merge_frame, on=["lesion_id", "niche_id"], how="left", validate="one_to_one")
    if merged[new_columns].isna().all(axis=1).any():
        missing_rows = merged.loc[merged[new_columns].isna().all(axis=1), ["lesion_id", "niche_id"]].head(5)
        raise ValueError(
            f"Failed to match {source} back to niches for some rows, examples={missing_rows.to_dict(orient='records')}"
        )
    return merged


def stage_index_or_error(stage: str) -> int:
    normalized = normalize_stage_label(stage)
    if normalized not in CANONICAL_STAGE_ORDER:
        raise ValueError(f"Inconsistent stage label '{stage}' (normalized='{normalized}').")
    return int(stage_to_index(normalized))


def default_reports_tables_dir() -> Path:
    return Path("reports/labels/tables")


def default_viability_report_path() -> Path:
    return Path("reports/labels/artifacts/split_viability_report.json")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stage_consistency_or_error(frame: pd.DataFrame, *, key_cols: Sequence[str]) -> None:
    for column in ("stage",):
        if column not in frame.columns:
            continue
        normalized = frame[column].astype(str).map(normalize_stage_label)
        if (normalized != frame[column].astype(str)).any():
            frame[column] = normalized
    if list(key_cols):
        duplicates = frame.duplicated(list(key_cols), keep=False)
        if duplicates.any():
            values = frame.loc[duplicates, list(key_cols)].drop_duplicates().head(5).to_dict(orient="records")
            raise ValueError(f"Detected duplicate keys for {list(key_cols)}: {values}")
