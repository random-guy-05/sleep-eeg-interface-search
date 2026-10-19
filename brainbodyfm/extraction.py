"""Build the cached semantic/latent interfaces from extracted tables.

Heavy EEG inference remains outside this package.  Extractors write raw model
outputs once; this module performs the deterministic, auditable assembly used by
every downstream experiment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import validate_subject_manifest
from .representations import (
    STABLE4_COLUMNS,
    Representation,
    align_representation,
    infer_latent_columns,
    infer_semantic_columns,
    load_cached_representation,
    load_table_representation,
    representation_qc,
    write_extraction_manifest,
)
from .taxonomy import coerce_bool_series, load_taxonomy


def _metadata_frame(path: str, *, subject_column: str = "subject_id") -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet" and not source.exists() and source.with_suffix(".csv").exists():
        source = source.with_suffix(".csv")
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source, sep="\t" if source.suffix.lower() == ".tsv" else ",")
    if subject_column not in frame:
        for candidate in ("subject", "BDSPPatientID", "SubjectID"):
            if candidate in frame:
                frame = frame.rename(columns={candidate: subject_column})
                break
    if subject_column not in frame:
        raise ValueError(f"{path} has no subject_id column")
    return frame


def assemble_cached_representations(
    manifest: pd.DataFrame,
    *,
    semantic_path: str,
    latent_path: str | None = None,
    taxonomy_path: str | None = None,
    subject_column: str = "subject_id",
    require_complete: bool = True,
    strict_dimensions: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], dict[str, dict[str, object]]]:
    """Align frozen caches and derive benchmark inputs.

    The latent cache is optional under the plan's explicit runtime failure
    rule.  Omitting it yields a semantic-only bundle without weakening any of
    the source-only safeguards.
    """
    validate_subject_manifest(manifest)
    # Tabular caches expose names directly; NumPy caches must carry an
    # explicit subject-ID sidecar and receive deterministic lhl_/pred_ names.
    semantic_frame = None if str(semantic_path).lower().endswith(".npy") else _metadata_frame(semantic_path, subject_column=subject_column)
    latent_frame = (
        None
        if latent_path is None or str(latent_path).lower().endswith(".npy")
        else _metadata_frame(latent_path, subject_column=subject_column)
    )
    semantic_columns = infer_semantic_columns(semantic_frame.columns) if semantic_frame is not None else None
    latent_columns = infer_latent_columns(latent_frame.columns) if latent_frame is not None else None
    if not semantic_columns:
        if semantic_frame is not None:
            raise ValueError("No pred_* semantic columns found in semantic cache.")
    if latent_path is not None and not latent_columns:
        if latent_frame is not None:
            raise ValueError("No lhl_* latent columns found in latent cache.")

    semantic_rep = load_cached_representation(
        semantic_path,
        name="semantic_841",
        kind="semantic",
        feature_columns=semantic_columns,
        subject_column=subject_column,
    )
    latent_rep = (
        load_cached_representation(
            latent_path,
            name="latent_1024",
            kind="latent",
            feature_columns=latent_columns,
            subject_column=subject_column,
        )
        if latent_path is not None
        else None
    )
    semantic_columns = list(semantic_rep.feature_names)
    latent_columns = list(latent_rep.feature_names) if latent_rep is not None else []
    if strict_dimensions and len(semantic_columns) != 841:
        raise ValueError(f"Expected 841 semantic heads, found {len(semantic_columns)}")
    if strict_dimensions and latent_rep is not None and len(latent_columns) != 1024:
        raise ValueError(f"Expected 1024 latent columns, found {len(latent_columns)}")

    taxonomy = load_taxonomy(taxonomy_path) if taxonomy_path else None
    # When a semantic matrix is supplied as a bare 841-column .npy file, the
    # frozen taxonomy is the authoritative column order.  Map generated
    # pred_1…pred_841 names onto those frozen names before selecting families.
    if (
        taxonomy is not None
        and str(semantic_path).lower().endswith(".npy")
        and len(semantic_columns) == len(taxonomy)
    ):
        frozen_names = taxonomy.sort_values("head_index")["head_name"].astype(str).tolist()
        semantic_rep = Representation(
            semantic_rep.name,
            semantic_rep.matrix,
            semantic_rep.subject_ids,
            tuple(frozen_names),
            dict(semantic_rep.metadata, names_from_taxonomy=True),
        )
        semantic_columns = list(semantic_rep.feature_names)
    semantic_rep = semantic_rep.aligned_to(manifest, subject_column=subject_column, require_complete=require_complete)
    if latent_rep is not None:
        latent_rep = latent_rep.aligned_to(manifest, subject_column=subject_column, require_complete=require_complete)
        if not np.array_equal(semantic_rep.subject_ids, latent_rep.subject_ids):
            raise AssertionError("semantic_subject_ids and latent_subject_ids do not align after manifest join")

    if taxonomy is None:
        raise ValueError("A frozen head taxonomy is required to derive physiology-only inputs.")
    allowed = set(taxonomy.loc[coerce_bool_series(taxonomy["include_physiology_only"]), "head_name"].astype(str))
    phys_columns = [column for column in semantic_columns if column in allowed]
    if not phys_columns:
        raise ValueError("Frozen taxonomy selected no semantic physiology columns.")
    indices = [semantic_columns.index(column) for column in phys_columns]
    outputs = {
        "semantic_841": semantic_rep.matrix,
        "physiology": semantic_rep.matrix[:, indices],
    }
    names = {
        "semantic_841": tuple(semantic_columns),
        "physiology": tuple(phys_columns),
    }
    if latent_rep is not None:
        outputs["latent_1024"] = latent_rep.matrix
        names["latent_1024"] = tuple(latent_columns)
    stable_available = [column for column in STABLE4_COLUMNS if column in semantic_columns]
    if len(stable_available) == len(STABLE4_COLUMNS):
        stable_indices = [semantic_columns.index(column) for column in STABLE4_COLUMNS]
        outputs["stable4"] = semantic_rep.matrix[:, stable_indices]
        names["stable4"] = STABLE4_COLUMNS
    qc = {
        "semantic_841": representation_qc(Representation("semantic_841", outputs["semantic_841"], semantic_rep.subject_ids, names["semantic_841"])),
        "physiology": representation_qc(Representation("physiology", outputs["physiology"], semantic_rep.subject_ids, names["physiology"])),
    }
    if latent_rep is not None:
        qc["latent_1024"] = representation_qc(Representation("latent_1024", outputs["latent_1024"], latent_rep.subject_ids, names["latent_1024"]))
    if "stable4" in outputs:
        qc["stable4"] = representation_qc(Representation("stable4", outputs["stable4"], semantic_rep.subject_ids, names["stable4"]))
    return outputs, names, qc


def write_cached_bundle(
    manifest: pd.DataFrame,
    outputs: Mapping[str, np.ndarray],
    names: Mapping[str, Sequence[str]],
    qc: Mapping[str, Mapping[str, object]],
    output_dir: str | Path,
    *,
    model_version: str,
    checkpoint: str,
    inference_commit: str | None,
    signal_channel: str,
    sampling_settings: Mapping[str, object],
    extraction_date: str | None = None,
    subject_column: str = "subject_id",
) -> dict[str, str]:
    """Persist the aligned, raw representation bundle and extraction audit.

    No normalization or imputation is performed here.  Tabular semantic
    interfaces remain human-auditable, while the 1024-D latent interface uses
    the plan's NumPy-plus-ID-sidecar convention.
    """
    frame = validate_subject_manifest(manifest)
    if subject_column not in frame.columns:
        raise ValueError(f"manifest is missing {subject_column!r}")
    ids = frame[subject_column].astype(str).to_numpy()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    required = {"semantic_841", "physiology"}
    missing = sorted(required.difference(outputs))
    if missing:
        raise ValueError(f"representation bundle is missing: {missing}")
    for key, matrix in outputs.items():
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or len(array) != len(ids):
            raise ValueError(
                f"{key} must be a 2-D matrix with one row per manifest subject."
            )
        if key not in names or len(names[key]) != array.shape[1]:
            raise ValueError(f"{key} feature names do not match its matrix columns.")

    files: dict[str, str] = {}

    def write_table(key: str, filename: str) -> None:
        table = pd.DataFrame(np.asarray(outputs[key], dtype=float), columns=list(names[key]))
        table.insert(0, subject_column, ids)
        destination = out / filename
        try:
            table.to_parquet(destination, index=False)
        except (ImportError, ValueError, OSError):
            destination = destination.with_suffix(".csv")
            table.to_csv(destination, index=False)
        files[key] = str(destination)

    write_table("semantic_841", "semantic_841.parquet")
    write_table("physiology", "physiology_semantic.parquet")
    if "stable4" in outputs:
        write_table("stable4", "stable4_heads.parquet")

    if "latent_1024" in outputs:
        latent_path = out / "latent_1024.npy"
        np.save(latent_path, np.asarray(outputs["latent_1024"], dtype=float), allow_pickle=False)
        pd.DataFrame({subject_column: ids}).to_csv(out / "latent_subject_ids.csv", index=False)
        files["latent_1024"] = str(latent_path)
        files["latent_subject_ids"] = str(out / "latent_subject_ids.csv")

    qc_path = out / "representation_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8")
    files["representation_qc"] = str(qc_path)
    extraction_path = out / "extraction_manifest.json"
    representation_metadata = {
        key: {
            "path": files.get(key),
            "n_subjects": int(np.asarray(value).shape[0]),
            "n_features": int(np.asarray(value).shape[1]),
            "raw_before_fold_normalization": True,
        }
        for key, value in outputs.items()
    }
    write_extraction_manifest(
        extraction_path,
        model_version=model_version,
        checkpoint=checkpoint,
        inference_commit=inference_commit,
        signal_channel=signal_channel,
        sampling_settings=sampling_settings,
        extraction_date=extraction_date or datetime.now(timezone.utc).date().isoformat(),
        input_psgs=(frame["record_id"].astype(str).tolist() if "record_id" in frame else ids.tolist()),
        representations=representation_metadata,
    )
    files["extraction_manifest"] = str(extraction_path)
    return files
