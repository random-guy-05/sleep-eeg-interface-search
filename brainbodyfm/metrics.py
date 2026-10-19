"""Primary age-conditioned metrics and paired site summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _validate_metric_arrays(labels: np.ndarray, scores: np.ndarray, ages: np.ndarray) -> None:
    if not (len(labels) == len(scores) == len(ages)):
        raise ValueError("labels, scores, and ages must have equal length.")
    if np.any(~np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary 0/1.")


def age_conditioned_auroc(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    ages: Sequence[float] | np.ndarray,
    age_window: float = 2.0,
    *,
    delta_years: float | None = None,
) -> float:
    """Compute the Challenge age-window micro-AUROC.

    Every eligible positive-negative pair receives equal weight.  This is
    deliberately not the mean of per-positive AUROCs, which was the source of
    an earlier experiment-level discrepancy.
    """
    if delta_years is not None:
        age_window = float(delta_years)
    if age_window < 0:
        raise ValueError("age_window must be non-negative.")
    y = np.asarray(labels, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    a = np.asarray(ages, dtype=float).reshape(-1)
    _validate_metric_arrays(y, s, a)
    finite = np.isfinite(s) & np.isfinite(a)
    pos = np.flatnonzero((y == 1) & finite)
    neg = np.flatnonzero((y == 0) & finite)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    pairs = 0
    for i in pos:
        eligible = neg[np.abs(a[neg] - a[i]) <= float(age_window)]
        if len(eligible) == 0:
            continue
        difference = s[i] - s[eligible]
        wins += float(np.count_nonzero(difference > 0))
        wins += 0.5 * float(np.count_nonzero(difference == 0))
        pairs += len(eligible)
    return float(wins / pairs) if pairs else float("nan")


# Protocol documents use both AUROC and AUC spellings.  Keep one
# implementation so aliases cannot drift numerically.
age_conditioned_auc = age_conditioned_auroc


def ordinary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(s):
        raise ValueError("labels and scores must have equal length.")
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    difference = pos[:, None] - neg[None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def macro_site_auc(
    labels: Sequence[int] | pd.Series,
    scores: Sequence[float] | pd.Series,
    ages: Sequence[float] | pd.Series,
    sites: Sequence[str] | pd.Series,
    *,
    site_order: Iterable[str] | None = None,
    age_window: float = 2.0,
    strict: bool = False,
) -> float:
    """Unweighted mean of site-level age-conditioned AUROCs."""
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels),
            "score": np.asarray(scores),
            "age": np.asarray(ages),
            "site": np.asarray(sites).astype(str),
        }
    )
    order = list(site_order) if site_order is not None else list(dict.fromkeys(frame["site"]))
    values: list[float] = []
    for site in order:
        subset = frame[frame["site"] == str(site)]
        if subset.empty:
            if strict:
                raise ValueError(f"site {site!r} has no rows.")
            continue
        value = age_conditioned_auroc(
            subset["label"], subset["score"], subset["age"], age_window
        )
        if np.isfinite(value):
            values.append(float(value))
        elif strict:
            raise ValueError(f"site {site!r} has no eligible positive-negative pairs.")
    return float(np.mean(values)) if values else float("nan")


def paired_macro_auc(
    frame: pd.DataFrame,
    score_a: str,
    score_b: str,
    *,
    outcome_column: str = "outcome",
    age_column: str = "age",
    site_column: str = "site",
    site_order: Iterable[str] | None = None,
    age_window: float = 2.0,
) -> dict[str, float]:
    """Return macro AUCs and the paired difference ``a - b``."""
    if score_a not in frame or score_b not in frame:
        raise KeyError(f"Both score columns are required: {score_a}, {score_b}.")
    a = macro_site_auc(
        frame[outcome_column], frame[score_a], frame[age_column], frame[site_column],
        site_order=site_order, age_window=age_window,
    )
    b = macro_site_auc(
        frame[outcome_column], frame[score_b], frame[age_column], frame[site_column],
        site_order=site_order, age_window=age_window,
    )
    return {"auc_a": a, "auc_b": b, "difference": float(a - b)}


def site_auc_table(
    predictions: pd.DataFrame,
    *,
    outcome_column: str = "outcome",
    score_column: str = "score",
    age_column: str = "age",
    site_column: str = "site",
    age_window: float = 2.0,
    site_order: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Machine-readable site metrics used by all benchmark scripts."""
    order = list(site_order) if site_order is not None else list(dict.fromkeys(predictions[site_column]))
    rows = []
    for site in order:
        subset = predictions[predictions[site_column].astype(str) == str(site)]
        rows.append(
            {
                "site": str(site),
                "n": int(len(subset)),
                "n_positive": int(np.sum(subset[outcome_column].to_numpy() == 1)),
                "age_auc": age_conditioned_auroc(
                    subset[outcome_column], subset[score_column], subset[age_column], age_window
                ) if len(subset) else float("nan"),
            }
        )
    return pd.DataFrame(rows)
