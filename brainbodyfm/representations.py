"""Representation loading, subject alignment, caching, and QC."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


STABLE4_COLUMNS: tuple[str, ...] = (
    "pred_regression_f_total_power_rem_rel_theta",
    "pred_regression_f_fs_chirp_c4m1",
    "pred_regression_f_sigma-9-std",
    "pred_regression_f_sigma_bandpower_kurtosis_c_n1",
)


@dataclass
class Representation:
    name: str
    matrix: np.ndarray
    subject_ids: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.ndim == 1:
            self.matrix = self.matrix.reshape(-1, 1)
        self.subject_ids = np.asarray(self.subject_ids).astype(str).reshape(-1)
        self.feature_names = tuple(str(x) for x in self.feature_names)
        if self.matrix.ndim != 2:
            raise ValueError("representation matrix must be 2-D")
        if len(self.matrix) != len(self.subject_ids):
            raise ValueError("matrix and subject_ids have different row counts")
        if self.matrix.shape[1] != len(self.feature_names):
            raise ValueError("matrix and feature_names have different column counts")
        if pd.Series(self.subject_ids).duplicated().any():
            raise ValueError("representation subject_ids must be unique")

    @property
    def n_subjects(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])

    def aligned_to(self, manifest: pd.DataFrame, *, subject_column: str = "subject_id", require_complete: bool = True) -> "Representation":
        if subject_column not in manifest:
            raise ValueError(f"manifest missing {subject_column!r}")
        target_ids = manifest[subject_column].astype(str).to_numpy()
        positions = {subject: i for i, subject in enumerate(self.subject_ids)}
        missing = [subject for subject in target_ids if subject not in positions]
        if missing and require_complete:
            raise ValueError(f"representation is missing {len(missing)} manifest subjects; examples: {missing[:5]}")
        keep = [i for i, subject in enumerate(target_ids) if subject in positions]
        matrix = np.vstack([self.matrix[positions[target_ids[i]]] for i in keep]) if keep else np.empty((0, self.n_features))
        aligned_ids = target_ids[keep]
        metadata = dict(self.metadata)
        metadata["aligned_to_manifest"] = True
        metadata["dropped_manifest_rows"] = int(len(target_ids) - len(keep))
        return Representation(self.name, matrix, aligned_ids, self.feature_names, metadata)


def infer_semantic_columns(columns: Iterable[object]) -> list[str]:
    """Find the 841 pretrained semantic coordinates in a table."""
    return [str(column) for column in columns if str(column).startswith("pred_")]


def infer_latent_columns(columns: Iterable[object]) -> list[str]:
    names = [str(column) for column in columns if str(column).startswith("lhl_")]
    return sorted(names, key=lambda name: int(name.split("_", 1)[1]) if name.split("_", 1)[1].isdigit() else name)


def _infer_subject_column(frame: pd.DataFrame) -> str:
    for column in ("subject_id", "subject", "BDSPPatientID", "SubjectID", "id"):
        if column in frame:
            return column
    raise ValueError("Could not identify a subject ID column.")


def load_table_representation(
    path: str | Path,
    *,
    name: str | None = None,
    feature_columns: Sequence[str] | None = None,
    subject_column: str | None = None,
    metadata_columns: Sequence[str] | None = None,
) -> Representation:
    """Load a CSV/Parquet representation and infer semantic/latent columns."""
    source = Path(path)
    if source.suffix.lower() == ".parquet" and not source.exists() and source.with_suffix(".csv").exists():
        source = source.with_suffix(".csv")
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix.lower() in {".csv", ".tsv"}:
        frame = pd.read_csv(source, sep="\t" if source.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported representation table: {source.suffix}")
    id_col = subject_column if subject_column in frame.columns else _infer_subject_column(frame)
    if feature_columns is None:
        semantic = infer_semantic_columns(frame.columns)
        latent = infer_latent_columns(frame.columns)
        if semantic and latent:
            raise ValueError("Both semantic and latent columns found; pass feature_columns explicitly.")
        feature_columns = semantic or latent
    features = [str(column) for column in feature_columns]
    missing = [column for column in features if column not in frame]
    if missing:
        raise ValueError(f"Representation columns missing from {source}: {missing[:5]}")
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    metadata = {
        "path": str(source),
        "n_source_columns": int(len(frame.columns)),
        "representation_kind": "semantic" if all(c.startswith("pred_") for c in features) else "latent" if all(c.startswith("lhl_") for c in features) else "custom",
    }
    if metadata_columns:
        for column in metadata_columns:
            if column in frame:
                metadata[column] = frame[column].tolist()
    return Representation(name or source.stem, matrix, frame[id_col].astype(str).to_numpy(), tuple(features), metadata)


def load_npy_representation(
    path: str | Path,
    subject_ids: Sequence[object] | str | Path,
    *,
    name: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> Representation:
    matrix = np.load(path, allow_pickle=False)
    if isinstance(subject_ids, (str, Path)):
        ids_frame = pd.read_csv(subject_ids)
        id_column = _infer_subject_column(ids_frame)
        ids = ids_frame[id_column].astype(str).to_numpy()
    else:
        ids = np.asarray(subject_ids).astype(str)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if feature_names is None:
        feature_names = tuple(f"feature_{i}" for i in range(matrix.shape[1]))
    return Representation(name or Path(path).stem, matrix, ids, tuple(feature_names), {"path": str(path), "representation_kind": "npy"})


def find_subject_ids_sidecar(path: str | Path, *, kind: str | None = None) -> Path | None:
    """Find the explicit subject-ID sidecar for a NumPy cache.

    NumPy arrays do not carry row identifiers.  The BrainBodyFM cache
    convention therefore requires a small CSV sidecar.  Resolution is
    deterministic and deliberately limited to names that cannot be confused
    with another representation table.
    """
    source = Path(path)
    if source.suffix.lower() != ".npy":
        return None
    stem = source.stem
    candidates: list[Path] = [
        source.with_name(f"{stem}_subject_ids.csv"),
        source.with_name(f"{stem}.subject_ids.csv"),
    ]
    if kind:
        normalized = str(kind).strip().lower()
        candidates.extend(
            [
                source.with_name(f"{normalized}_subject_ids.csv"),
                source.with_name(f"{normalized}_ids.csv"),
            ]
        )
    candidates.extend(
        [
            source.with_name("subject_ids.csv"),
            source.with_name("ids.csv"),
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_cached_representation(
    path: str | Path,
    *,
    name: str,
    kind: str,
    feature_columns: Sequence[str] | None = None,
    subject_column: str = "subject_id",
) -> Representation:
    """Load either a tabular cache or an ``.npy`` cache with an ID sidecar.

    The helper is intentionally strict for NumPy inputs: silently assuming
    row order would make cross-cache joins unauditable.
    """
    source = Path(path)
    if source.suffix.lower() != ".npy":
        return load_table_representation(
            source,
            name=name,
            feature_columns=feature_columns,
            subject_column=subject_column,
        )
    ids_path = find_subject_ids_sidecar(source, kind=kind)
    if ids_path is None:
        raise ValueError(
            f"NumPy representation {source} has no subject-ID sidecar; expected "
            f"{source.stem}_subject_ids.csv or subject_ids.csv."
        )
    matrix = np.load(source, allow_pickle=False)
    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        n_features = 1
    elif matrix.ndim == 2:
        n_features = int(matrix.shape[1])
    else:
        raise ValueError(f"NumPy representation must be 1-D or 2-D, got {matrix.shape}.")
    if feature_columns is not None:
        names = tuple(str(column) for column in feature_columns)
        if len(names) != n_features:
            raise ValueError(
                f"feature_columns has {len(names)} names but {source} has {n_features} columns."
            )
    else:
        prefix = "pred_" if str(kind).lower() == "semantic" else "lhl_" if str(kind).lower() == "latent" else "feature_"
        names = tuple(f"{prefix}{i + 1}" for i in range(n_features))
    return load_npy_representation(source, ids_path, name=name, feature_names=names)


def align_representation(
    matrix: object,
    representation_subject_ids: Sequence[object],
    manifest: pd.DataFrame,
    *,
    feature_names: Sequence[str] | None = None,
    name: str = "representation",
    subject_column: str = "subject_id",
    require_complete: bool = True,
) -> Representation:
    representation = Representation(
        name,
        np.asarray(matrix, dtype=float),
        np.asarray(representation_subject_ids).astype(str),
        tuple(feature_names if feature_names is not None else [f"feature_{i}" for i in range(np.asarray(matrix).reshape(len(representation_subject_ids), -1).shape[1])]),
    )
    return representation.aligned_to(manifest, subject_column=subject_column, require_complete=require_complete)


def representation_qc(rep: Representation, *, near_duplicate_decimals: int = 12) -> dict[str, Any]:
    """Return missingness, variance, and duplicate-column diagnostics."""
    finite = np.isfinite(rep.matrix)
    std = np.nanstd(rep.matrix, axis=0)
    duplicate_groups: dict[tuple[float, ...], list[str]] = {}
    for index, column in enumerate(rep.matrix.T):
        key = tuple(np.nan_to_num(column, nan=9.87654321e307).round(near_duplicate_decimals).tolist())
        duplicate_groups.setdefault(key, []).append(rep.feature_names[index])
    duplicates = [names for names in duplicate_groups.values() if len(names) > 1]
    return {
        "name": rep.name,
        "n_subjects": rep.n_subjects,
        "n_features": rep.n_features,
        "finite_fraction_by_feature": finite.mean(axis=0).tolist(),
        "std_by_feature": np.nan_to_num(std, nan=0.0).tolist(),
        "n_constant_features": int(np.count_nonzero(~np.isfinite(std) | (std < 1e-12))),
        "near_duplicate_groups": duplicates,
        "n_near_duplicate_groups": len(duplicates),
    }


def save_representation(rep: Representation, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rep.matrix, columns=list(rep.feature_names))
    frame.insert(0, "subject_id", rep.subject_ids)
    if destination.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(destination, index=False)
        except (ImportError, ValueError, OSError):
            frame.to_csv(destination.with_suffix(".csv"), index=False)
    else:
        frame.to_csv(destination, index=False)


def write_extraction_manifest(
    path: str | Path,
    *,
    model_version: str,
    checkpoint: str,
    inference_commit: str | None,
    signal_channel: str,
    sampling_settings: Mapping[str, Any],
    extraction_date: str,
    input_psgs: Sequence[str],
    representations: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    payload = {
        "model_version": model_version,
        "checkpoint": checkpoint,
        "inference_code_commit": inference_commit,
        "signal_channel": signal_channel,
        "sampling_settings": dict(sampling_settings),
        "extraction_date": extraction_date,
        "input_psgs": list(input_psgs),
        "representations": {key: dict(value) for key, value in (representations or {}).items()},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_aligned_matrix(
    path: str | Path,
    manifest: pd.DataFrame,
    *,
    kind: str = "semantic",
    subject_column: str = "subject_id",
    require_complete: bool = True,
) -> Representation:
    source = Path(path)
    if source.suffix.lower() == ".parquet" and not source.exists() and source.with_suffix(".csv").exists():
        source = source.with_suffix(".csv")
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    columns = infer_semantic_columns(frame.columns) if kind == "semantic" else infer_latent_columns(frame.columns)
    if kind == "physiology":
        raise ValueError("Physiology selection requires a frozen head taxonomy; pass explicit feature_columns.")
    rep = load_table_representation(source, name=kind, feature_columns=columns)
    return rep.aligned_to(manifest, subject_column=subject_column, require_complete=require_complete)
