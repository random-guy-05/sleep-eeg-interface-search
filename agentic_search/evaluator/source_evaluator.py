"""Source-only cross-institution quartet evaluator.

This module intentionally does not import the target evaluator or read any
held-site rows/labels. Each fold file contains only its two source sites.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from brainbodyfm.models import ModelSpec
from brainbodyfm.tuning import tune_source_transfer
from agentic_search.evaluator.cache import SourceCache


PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"
C_GRID = tuple(float(x) for x in np.logspace(-4, 4, 17))
TIE_TOLERANCE = 1e-4
AGE_WINDOW = 2.0


@lru_cache(maxsize=1)
def _catalog() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "heads.csv", dtype=str)
    if frame.head_id.duplicated().any() or frame.head_name.duplicated().any():
        raise ValueError("Head catalog identifiers and names must be unique")
    return frame.set_index("head_id", drop=False)


@lru_cache(maxsize=3)
def _source_frame(held_site: str) -> pd.DataFrame:
    path = DATA / "source_folds" / f"{held_site}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Source-only fold data is missing: {path}")
    frame = pd.read_parquet(path)
    if "outcome" not in frame or held_site in set(frame.site.astype(str)):
        raise ValueError("Source fold contains held-site labels or lacks source outcomes")
    if frame.site.nunique() != 2:
        raise ValueError("Exactly two visible source institutions are required")
    return frame


def canonical_quartet(quartet: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(x) for x in quartet)
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("A candidate must contain exactly four distinct head IDs")
    return tuple(sorted(values))


def evaluate_source(
    held_site: str,
    quartet: Sequence[str],
    *,
    cache_path: str | Path | None = None,
) -> dict[str, object]:
    """Return paired source transfer results for one valid four-head interface."""
    key = canonical_quartet(quartet)
    catalog = _catalog()
    unknown = sorted(set(key).difference(catalog.index))
    if unknown:
        raise ValueError(f"Unknown head IDs: {unknown}")
    quartet_key = "|".join(key)
    cache = SourceCache(cache_path or PACKAGE / "cache/source.sqlite")
    saved = cache.get(str(held_site), quartet_key)
    if saved is not None:
        return {**saved, "cache_hit": True}

    frame = _source_frame(str(held_site))
    sites = sorted(frame.site.astype(str).unique())
    names = catalog.loc[list(key), "head_name"].astype(str).tolist()
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"Source fold is missing catalog heads: {missing}")
    X = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame.outcome, errors="raise").astype(int).to_numpy()
    ages = pd.to_numeric(frame.age, errors="raise").astype(float).to_numpy()
    site_values = frame.site.astype(str).to_numpy()
    ids = frame.subject_id.astype(str).to_numpy()
    spec = ModelSpec(name="prospective_random4_l2", penalty="l2", c_grid=C_GRID,
                     max_iter=10_000, tol=1e-5, random_state=2026090401)
    tuning = tune_source_transfer(
        X, y, ages, site_values, model_spec=spec, feature_names=names,
        subject_ids=ids, age_window=AGE_WINDOW, tie_tolerance=TIE_TOLERANCE,
    )
    # The existing tuner labels directions by the sorted source-site order.
    auc12 = float(tuning.inner_A_to_B_auc)
    auc21 = float(tuning.inner_B_to_A_auc)
    if not np.isfinite([auc12, auc21, tuning.chosen_C]).all():
        raise ValueError(f"Non-finite source transfer result for {quartet_key}")
    payload: dict[str, object] = {
        "held_site": str(held_site),
        "source_site_1": sites[0],
        "source_site_2": sites[1],
        "quartet_key": quartet_key,
        "head_ids": list(key),
        "head_names": names,
        "auc_source1_to_source2": auc12,
        "auc_source2_to_source1": auc21,
        "mean_source_auc": float((auc12 + auc21) / 2.0),
        "min_source_auc": float(min(auc12, auc21)),
        "direction_gap": float(abs(auc12 - auc21)),
        "selected_C": float(tuning.chosen_C),
        "cache_hit": False,
    }
    cache.put(str(held_site), quartet_key, {k: v for k, v in payload.items() if k != "cache_hit"})
    return payload
