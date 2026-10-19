"""Deterministic site-level split helpers used by every analysis."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SiteSplit:
    target_site: str
    source_sites: tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray


def iter_site_loso(
    manifest: pd.DataFrame,
    *,
    site_column: str = "site",
    subject_column: str = "subject_id",
    sites: Iterable[str] | None = None,
) -> Iterator[SiteSplit]:
    """Yield complete site-LOSO folds with subject-level leakage checks."""
    if site_column not in manifest or subject_column not in manifest:
        raise ValueError(f"manifest needs {site_column!r} and {subject_column!r}")
    site_values = manifest[site_column].astype(str).to_numpy()
    subject_values = manifest[subject_column].astype(str).to_numpy()
    if pd.Series(subject_values).duplicated().any():
        raise ValueError("iter_site_loso expects one row per subject")
    order = tuple(str(site) for site in (sites if sites is not None else dict.fromkeys(site_values)))
    for target in order:
        test = np.flatnonzero(site_values == target)
        train = np.flatnonzero(site_values != target)
        if len(test) == 0:
            raise ValueError(f"site {target!r} has no rows")
        if set(subject_values[train]).intersection(subject_values[test]):
            raise AssertionError("subject leakage across a site fold")
        yield SiteSplit(target, tuple(site for site in order if site != target), train, test)


def source_transfer_directions(
    manifest: pd.DataFrame,
    source_sites: Iterable[str],
    *,
    site_column: str = "site",
) -> tuple[tuple[str, str, np.ndarray, np.ndarray], ...]:
    """Return A→B and B→A indices for the two source institutions."""
    source = tuple(sorted(str(site) for site in source_sites))
    if len(source) != 2:
        raise ValueError("source_transfer_directions requires exactly two sites")
    values = manifest[site_column].astype(str).to_numpy()
    a = np.flatnonzero(values == source[0])
    b = np.flatnonzero(values == source[1])
    return ((source[0], source[1], a, b), (source[1], source[0], b, a))


def assert_disjoint_subject_ids(
    train_subject_ids: Iterable[object], test_subject_ids: Iterable[object]
) -> None:
    overlap = set(map(str, train_subject_ids)).intersection(map(str, test_subject_ids))
    if overlap:
        raise AssertionError(f"train/test subject overlap: {sorted(overlap)[:5]}")
