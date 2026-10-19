"""Canonical subject manifest and cohort audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_SITES


MANIFEST_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "record_id",
    "site",
    "age",
    "sex",
    "outcome",
    "in_original_338",
    "potential_locked_replication",
    "has_semantic_841",
    "has_latent_1024",
    "has_stable4_heads",
    "included_primary_semantic",
    "included_latent_comparison",
)


def _first_existing(frame: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    if required:
        raise ValueError(f"None of the expected columns are present: {list(candidates)}")
    return None


def _binary_outcome(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(int)
    converted = values.map(
        lambda value: 1 if str(value).strip().lower() in {"1", "true", "yes", "positive"}
        else 0 if str(value).strip().lower() in {"0", "false", "no", "negative"}
        else np.nan
    )
    numeric = pd.to_numeric(values, errors="coerce")
    converted = converted.where(converted.notna(), numeric)
    return converted


def _boolean_flags(values: object) -> pd.Series:
    """Parse persisted boolean flags without treating ``'False'`` as true."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    truthy = {"1", "true", "t", "yes", "y", "on", "include", "included"}
    return series.map(lambda value: str(value).strip().lower() in truthy).astype(bool)


def validate_subject_manifest(
    manifest: pd.DataFrame,
    *,
    sites: Iterable[str] = DEFAULT_SITES,
    require_exact_columns: bool = False,
) -> pd.DataFrame:
    """Validate and return a normalized copy of a one-row-per-subject table."""
    required = {"subject_id", "site", "age", "outcome"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Subject manifest missing required columns: {missing}")
    frame = manifest.copy()
    if frame["subject_id"].isna().any():
        raise ValueError("subject_id must be non-empty and non-null.")
    frame["subject_id"] = frame["subject_id"].astype(str).str.strip()
    if frame["subject_id"].eq("").any() or frame["subject_id"].str.lower().eq("nan").any():
        raise ValueError("subject_id must be non-empty and non-null.")
    if frame["subject_id"].duplicated().any():
        dupes = frame.loc[frame["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"Each subject must contribute exactly one row; duplicates: {dupes[:5]}")
    allowed_sites = tuple(str(site) for site in sites)
    frame["site"] = frame["site"].astype(str).str.strip()
    unknown = sorted(set(frame["site"]).difference(allowed_sites))
    if unknown:
        raise ValueError(f"Unexpected site identifiers: {unknown}")
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    if frame["age"].isna().any():
        raise ValueError("age must be finite for every eligible subject.")
    frame["outcome"] = _binary_outcome(frame["outcome"])
    if frame["outcome"].isna().any() or not frame["outcome"].isin([0, 1]).all():
        raise ValueError("outcome must contain only binary 0/1 values.")
    frame["outcome"] = frame["outcome"].astype(int)
    for flag in (
        "in_original_338",
        "potential_locked_replication",
        "has_semantic_841",
        "has_latent_1024",
        "has_stable4_heads",
        "included_primary_semantic",
        "included_latent_comparison",
    ):
        if flag in frame.columns:
            frame[flag] = _boolean_flags(frame[flag])
    if require_exact_columns:
        missing_exact = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
        if missing_exact:
            raise ValueError(f"Frozen manifest missing columns: {missing_exact}")
    return frame


def _ids(values: Iterable[object] | None) -> set[str]:
    return {str(value) for value in values} if values is not None else set()


def build_subject_manifest(
    phenotype: pd.DataFrame,
    *,
    semantic_subject_ids: Iterable[object] | None = None,
    latent_subject_ids: Iterable[object] | None = None,
    stable4_subject_ids: Iterable[object] | None = None,
    original_subject_ids: Iterable[object] | None = None,
    record_id_column: str | None = None,
    locked_replication_subject_ids: Iterable[object] | None = None,
    sites: Iterable[str] = DEFAULT_SITES,
) -> pd.DataFrame:
    """Create the canonical manifest from a phenotype table.

    Locked-replication eligibility is conservative: it is true only when the
    caller explicitly supplies the audited subject IDs.  Historical ambiguity
    therefore defaults to ``False`` as required by the protocol.
    """
    subject_col = _first_existing(phenotype, ("subject_id", "BDSPPatientID", "SubjectID", "id"))
    record_col = record_id_column or _first_existing(
        phenotype, ("record_id", "file_id", "RecordID"), required=False
    )
    site_col = _first_existing(phenotype, ("site", "site_id", "SiteID", "institution"))
    age_col = _first_existing(phenotype, ("age", "Age"))
    sex_col = _first_existing(phenotype, ("sex", "Sex"), required=False)
    outcome_col = _first_existing(phenotype, ("outcome", "target", "Cognitive_Impairment"))
    if record_col:
        record_ids = phenotype[record_col].astype(str)
    elif "BidsFolder" in phenotype:
        record_ids = phenotype["BidsFolder"].astype(str)
        if "SessionID" in phenotype:
            session = pd.to_numeric(phenotype["SessionID"], errors="coerce")
            if session.isna().any():
                raise ValueError("SessionID must be finite when deriving record_id from BidsFolder")
            record_ids = record_ids + "_ses-" + session.astype(int).astype(str)
    else:
        record_ids = phenotype[subject_col].astype(str)
    base = pd.DataFrame(
        {
            "subject_id": phenotype[subject_col].astype(str),
            "record_id": record_ids,
            "site": phenotype[site_col].astype(str),
            "age": phenotype[age_col],
            "sex": phenotype[sex_col].astype(str) if sex_col else "",
            "outcome": phenotype[outcome_col],
        }
    )
    base = validate_subject_manifest(base, sites=sites)
    original = _ids(original_subject_ids)
    locked = _ids(locked_replication_subject_ids)
    semantic = _ids(semantic_subject_ids)
    latent = _ids(latent_subject_ids)
    stable4 = _ids(stable4_subject_ids)
    base["in_original_338"] = base["subject_id"].isin(original)
    base["potential_locked_replication"] = base["subject_id"].isin(locked) & ~base["in_original_338"]
    base["has_semantic_841"] = base["subject_id"].isin(semantic)
    base["has_latent_1024"] = base["subject_id"].isin(latent)
    base["has_stable4_heads"] = base["subject_id"].isin(stable4)
    base["included_primary_semantic"] = base["has_semantic_841"]
    base["included_latent_comparison"] = base["has_latent_1024"]
    return base.loc[:, MANIFEST_COLUMNS]


def cohort_by_site(
    manifest: pd.DataFrame,
    *,
    sites: Iterable[str] = DEFAULT_SITES,
    original_only: bool = False,
    locked_only: bool = False,
    inclusion_column: str | None = None,
) -> pd.DataFrame:
    """Return the immediately auditable cohort table requested by the plan."""
    frame = validate_subject_manifest(manifest, sites=sites)
    if original_only and "in_original_338" in frame:
        frame = frame[_boolean_flags(frame["in_original_338"])]
    if locked_only and "potential_locked_replication" in frame:
        frame = frame[_boolean_flags(frame["potential_locked_replication"])]
    if inclusion_column is not None:
        if inclusion_column not in frame.columns:
            raise ValueError(f"manifest is missing inclusion flag {inclusion_column!r}")
        frame = frame[_boolean_flags(frame[inclusion_column])]
    rows = []
    for site in sites:
        subset = frame[frame["site"] == str(site)]
        rows.append(
            {
                "Site": str(site),
                "N": int(len(subset)),
                "Positives": int(subset["outcome"].sum()),
                "Positive %": float(100 * subset["outcome"].mean()) if len(subset) else float("nan"),
                "Median age": float(subset["age"].median()) if len(subset) else float("nan"),
                "Female %": float(100 * (subset["sex"].astype(str).str.lower() == "female").mean()) if len(subset) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class LockedReplicationAudit:
    eligible: bool
    subject_ids: tuple[str, ...]
    reason: str
    evidence: Mapping[str, object]


def audit_locked_replication(
    manifest: pd.DataFrame,
    *,
    no_label_influence: bool,
    no_performance_influence: bool,
    no_preprocessing_influence: bool,
    no_inclusion_influence: bool,
    sites: Iterable[str] = DEFAULT_SITES,
) -> LockedReplicationAudit:
    """Apply the strict historical-audit rule for untouched subjects."""
    frame = validate_subject_manifest(manifest, sites=sites)
    conditions = {
        "no_label_influence": bool(no_label_influence),
        "no_performance_influence": bool(no_performance_influence),
        "no_preprocessing_influence": bool(no_preprocessing_influence),
        "no_inclusion_influence": bool(no_inclusion_influence),
    }
    if "potential_locked_replication" in frame:
        candidate_mask = _boolean_flags(frame["potential_locked_replication"])
        if "in_original_338" in frame:
            candidate_mask &= ~_boolean_flags(frame["in_original_338"])
        candidates = frame.loc[candidate_mask, "subject_id"]
    else:
        # Without an explicit audited allowlist, ambiguity defaults to not
        # locked even when the other historical flags are true.
        candidates = frame.iloc[0:0]["subject_id"]
    evidence = {
        **conditions,
        "explicit_candidate_allowlist_present": "potential_locked_replication" in frame,
        "n_candidate_subjects": int(len(candidates)),
    }
    eligible = bool(all(conditions.values()) and len(candidates))
    reason = "all historical audit conditions passed" if eligible else "audit incomplete or ambiguous; not locked"
    return LockedReplicationAudit(eligible, tuple(candidates.astype(str)), reason, evidence)


def write_manifest(manifest: pd.DataFrame, path: str | Path) -> None:
    frame = validate_subject_manifest(manifest, require_exact_columns=True)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, MANIFEST_COLUMNS].to_csv(destination, index=False)


def read_manifest(path: str | Path) -> pd.DataFrame:
    return validate_subject_manifest(pd.read_csv(path), require_exact_columns=True)
