"""Conservative, pre-result taxonomy for semantic output heads."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Sequence

import pandas as pd


CATEGORY_PHYSIOLOGY = "physiology"
CATEGORY_DEMOGRAPHIC = "demographic_general"
CATEGORY_TARGET = "target_proximal_clinical_cognitive"
CATEGORY_AMBIGUOUS = "ambiguous"

PHYSIOLOGY_FAMILIES: tuple[str, ...] = (
    "REM_spectral",
    "NREM_spectral",
    "theta",
    "delta_slow_wave",
    "spindle_morphology",
    "spindle_activity",
    "sigma_dynamics",
    "sleep_architecture",
    "arousal_continuity",
    "respiratory",
    "cardiac_autonomic",
    "signal_morphology",
    "other_sleep_physiology",
)

_TARGET_TERMS = (
    "cog", "cognition", "mmse", "dsst", "casi", "neuro", "dementia", "alzheimer",
    "mci", "diagnos", "mortality", "death", "disease", "impair", "memory", "trail",
    "naming", "processing", "executive", "fluid", "crystallized", "brain_health",
)
_DEMOGRAPHIC_TERMS = (
    "age", "sex", "gender", "bmi", "height", "weight", "anthrop", "demograph", "education",
)
_PHYSIOLOGY_TERMS = (
    "eeg", "spectr", "theta", "delta", "sigma", "gamma", "alpha", "beta", "spindle",
    "sleep", "stage", "rem", "nrem", "slow_wave", "slowosc", "arousal", "continuity",
    "respir", "oxygen", "hypoxia", "t90", "cardiac", "autonomic", "heart", "ecg", "hrv",
    "emg", "signal", "morph", "wave", "bandpower", "power", "chirp", "kurtosis", "plm", "lmi",
)


def coerce_bool_series(values: object) -> pd.Series:
    """Normalize CSV/YAML boolean values without the ``bool('False')`` trap."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    true_values = {"1", "true", "t", "yes", "y", "on", "include", "included"}
    false_values = {"0", "false", "f", "no", "n", "off", "exclude", "excluded", "", "nan", "none", "null"}

    def parse(value: object) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return bool(numeric) if pd.notna(numeric) else False

    return series.map(parse).astype(bool)


def _family_for(name: str) -> str:
    value = name.lower().replace("-", "_")
    if "nrem" in value or "n2" in value or "n3" in value:
        return "NREM_spectral"
    if "rem" in value:
        return "REM_spectral"
    if "spindle" in value or "fs_chirp" in value:
        return "spindle_morphology" if any(t in value for t in ("chirp", "duration", "freq", "morph")) else "spindle_activity"
    if "sigma" in value:
        return "sigma_dynamics"
    if "theta" in value:
        return "theta"
    if "delta" in value or "slow_wave" in value or "slowosc" in value:
        return "delta_slow_wave"
    if any(t in value for t in ("respir", "oxygen", "hypoxia", "t90", "ahi", "apnea")):
        return "respiratory"
    if any(t in value for t in ("ecg", "cardiac", "heart", "hrv", "autonomic")):
        return "cardiac_autonomic"
    if any(t in value for t in ("arousal", "continuity", "fragment")):
        return "arousal_continuity"
    if any(t in value for t in ("kurtosis", "skew", "morph", "wave", "signal")):
        return "signal_morphology"
    if any(t in value for t in ("stage", "sleep", "architecture", "latency", "efficiency")):
        return "sleep_architecture"
    return "other_sleep_physiology"


def classify_head(head_name: str) -> dict[str, object]:
    """Classify one head using fixed conservative keyword rules.

    Target-proximal and demographic terms take precedence over physiology
    terms.  Unknown names are ambiguous and are excluded from the primary
    physiology-only space.
    """
    name = str(head_name)
    value = name.lower().replace("-", "_")
    def has_term(terms: tuple[str, ...]) -> bool:
        return any(re.search(rf"(?:^|_){re.escape(term)}(?:_|$)", value) for term in terms)

    if has_term(_TARGET_TERMS):
        return {
            "category": CATEGORY_TARGET,
            "physiological_family": "",
            "include_physiology_only": False,
            "rationale": "target-proximal clinical/cognitive or outcome-related head",
        }
    if has_term(_DEMOGRAPHIC_TERMS):
        return {
            "category": CATEGORY_DEMOGRAPHIC,
            "physiological_family": "",
            "include_physiology_only": False,
            "rationale": "demographic or anthropometric head",
        }
    if has_term(_PHYSIOLOGY_TERMS):
        family = _family_for(name)
        return {
            "category": CATEGORY_PHYSIOLOGY,
            "physiological_family": family,
            "include_physiology_only": True,
            "rationale": "EEG, sleep, signal, respiratory, or autonomic physiology head",
        }
    return {
        "category": CATEGORY_AMBIGUOUS,
        "physiological_family": "",
        "include_physiology_only": False,
        "rationale": "name is insufficiently specific for conservative inclusion",
    }


def infer_head_names(columns: Iterable[object]) -> list[str]:
    """Return semantic output columns while excluding phenotype metadata."""
    return [
        str(column)
        for column in columns
        if str(column).startswith(("pred_", "head_"))
        and str(column) not in {"predicted", "head_count"}
    ]


def build_head_taxonomy(head_names: Sequence[object]) -> pd.DataFrame:
    rows = []
    for index, raw_name in enumerate(head_names):
        name = str(raw_name)
        classification = classify_head(name)
        rows.append(
            {
                "head_index": int(index),
                "head_name": name,
                "category": classification["category"],
                "physiological_family": classification["physiological_family"],
                "include_physiology_only": bool(classification["include_physiology_only"]),
                "rationale": classification["rationale"],
                "reviewer_1": "",
                "reviewer_2": "",
                "final_decision": "include" if classification["include_physiology_only"] else "exclude",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "head_index", "head_name", "category", "physiological_family",
            "include_physiology_only", "rationale", "reviewer_1", "reviewer_2", "final_decision",
        ],
    )


def build_family_taxonomy(head_taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Summarize the fixed family vocabulary and member counts."""
    required = {"head_name", "category", "physiological_family", "include_physiology_only"}
    missing = sorted(required.difference(head_taxonomy.columns))
    if missing:
        raise ValueError(f"head taxonomy missing columns: {missing}")
    included = head_taxonomy[coerce_bool_series(head_taxonomy["include_physiology_only"])]
    rows = []
    for family in PHYSIOLOGY_FAMILIES:
        members = included.loc[included["physiological_family"] == family, "head_name"].astype(str).tolist()
        rows.append({
            "physiological_family": family,
            "n_heads": len(members),
            "head_names": "|".join(members),
            "include_physiology_only": True,
            "rationale": "prespecified family used for family-level stability analysis",
        })
    return pd.DataFrame(rows)


def write_taxonomy(head_names: Sequence[object], head_path: str | Path, family_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    heads = build_head_taxonomy(head_names)
    families = build_family_taxonomy(heads)
    head_destination = Path(head_path)
    family_destination = Path(family_path)
    head_destination.parent.mkdir(parents=True, exist_ok=True)
    family_destination.parent.mkdir(parents=True, exist_ok=True)
    heads.to_csv(head_destination, index=False)
    families.to_csv(family_destination, index=False)
    return heads, families


def load_taxonomy(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"head_index", "head_name", "category", "physiological_family", "include_physiology_only"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"taxonomy missing columns: {missing}")
    frame["include_physiology_only"] = coerce_bool_series(frame["include_physiology_only"])
    if frame["head_name"].astype(str).duplicated().any() or frame["head_index"].duplicated().any():
        raise ValueError("taxonomy head names and indices must be unique")
    allowed_categories = {
        CATEGORY_PHYSIOLOGY,
        CATEGORY_DEMOGRAPHIC,
        CATEGORY_TARGET,
        CATEGORY_AMBIGUOUS,
    }
    unknown_categories = sorted(set(frame["category"].astype(str)).difference(allowed_categories))
    if unknown_categories:
        raise ValueError(f"taxonomy has unknown categories: {unknown_categories}")
    included = frame[frame["include_physiology_only"]]
    if not included["category"].astype(str).eq(CATEGORY_PHYSIOLOGY).all():
        raise ValueError("only physiology-category heads may enter the primary sparse space")
    if not included["physiological_family"].astype(str).isin(PHYSIOLOGY_FAMILIES).all():
        raise ValueError("included physiology heads must use the frozen family vocabulary")
    return frame
