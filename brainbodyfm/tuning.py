"""Site-aware source-to-source hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import age_conditioned_auroc
from .models import ModelSpec, FittedProbe, fit_probe


@dataclass
class TuningResult:
    chosen_C: float
    inner_auc: float
    inner_A_to_B_auc: float
    inner_B_to_A_auc: float
    logs: pd.DataFrame
    tie_candidates: tuple[float, ...] = ()

    @property
    def inner_mean_auc(self) -> float:
        """Alias matching the notation used in the frozen analysis plan."""
        return self.inner_auc


def _as_matrix(X: object) -> np.ndarray:
    matrix = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"X must be 2-D, got {matrix.shape}.")
    return matrix


def _site_indices(sites: Sequence[object]) -> dict[str, np.ndarray]:
    values = np.asarray(sites).astype(str).reshape(-1)
    return {site: np.flatnonzero(values == site) for site in sorted(set(values))}


def choose_stronger_regularized(
    scores: Sequence[float],
    c_grid: Sequence[float],
    *,
    tie_tolerance: float = 1e-4,
) -> tuple[float, tuple[float, ...]]:
    """Choose the smallest C among candidates within the score tie band."""
    score_arr = np.asarray(scores, dtype=float)
    c_arr = np.asarray(c_grid, dtype=float)
    if len(score_arr) != len(c_arr):
        raise ValueError("scores and c_grid must have equal length.")
    finite = np.isfinite(score_arr)
    if not np.any(finite):
        raise ValueError("No finite source-transfer tuning scores were available.")
    best = float(np.max(score_arr[finite]))
    tied = c_arr[finite & (score_arr >= best - float(tie_tolerance))]
    if len(tied) == 0:
        raise AssertionError("Internal tuning tie set is empty.")
    tied_sorted = tuple(float(c) for c in sorted(tied.tolist()))
    return tied_sorted[0], tied_sorted


def tune_source_transfer(
    X: object,
    y: Sequence[int],
    ages: Sequence[float],
    sites: Sequence[object],
    *,
    model_spec: ModelSpec,
    feature_names: Sequence[str] | None = None,
    subject_ids: Sequence[object] | None = None,
    age_window: float = 2.0,
    tie_tolerance: float = 1e-4,
) -> TuningResult:
    """Tune C using A→B and B→A only.

    Every direction fits its own imputer/scaler/PCA on the source institution
    and evaluates on the other institution.  No target-site labels or
    statistics are read by the tuner.
    """
    matrix = _as_matrix(X)
    labels = np.asarray(y, dtype=int).reshape(-1)
    age = np.asarray(ages, dtype=float).reshape(-1)
    site_index = _site_indices(sites)
    if not (len(matrix) == len(labels) == len(age)):
        raise ValueError("X, y, ages, and sites must have equal length.")
    if len(site_index) != 2:
        raise ValueError(
            f"Site-transfer tuning expects exactly two sites, got {sorted(site_index)}."
        )
    site_a, site_b = sorted(site_index)
    ids = np.asarray(subject_ids).astype(str) if subject_ids is not None else None
    if ids is not None and len(ids) != len(matrix):
        raise ValueError("subject_ids must have one entry per row of X.")
    names = tuple(str(x) for x in (feature_names if feature_names is not None else [f"feature_{j}" for j in range(matrix.shape[1])]))

    rows: list[dict[str, object]] = []
    for C in model_spec.c_grid:
        direction_values: dict[str, float] = {}
        for source_site, target_site, direction in (
            (site_a, site_b, "A_to_B"),
            (site_b, site_a, "B_to_A"),
        ):
            train_idx = site_index[source_site]
            test_idx = site_index[target_site]
            if ids is not None and set(ids[train_idx]).intersection(ids[test_idx]):
                raise AssertionError("Inner source-transfer subjects overlap across sites.")
            target_rows_in_preprocessing_fit = 0
            convergence_warning = ""
            fit_error = ""
            try:
                probe = fit_probe(
                    matrix[train_idx],
                    labels[train_idx],
                    spec=model_spec,
                    C=float(C),
                    feature_names=names,
                    source_subject_ids=ids[train_idx] if ids is not None else None,
                    target_subject_ids=ids[test_idx] if ids is not None else None,
                )
                scores = probe.decision_function(matrix[test_idx])
                if probe.preprocessor.audit_ is not None:
                    target_rows_in_preprocessing_fit = probe.preprocessor.audit_.target_rows_used
                if target_rows_in_preprocessing_fit:
                    raise AssertionError("Validation subjects entered an inner preprocessing fit.")
                convergence_warning = " | ".join(probe.convergence_warnings)
                value = age_conditioned_auroc(
                    labels[test_idx], scores, age[test_idx], age_window
                )
            except ValueError as exc:
                value = float("nan")
                fit_error = f"{type(exc).__name__}: {exc}"
            direction_values[direction] = float(value)
            rows.append(
                {
                    "C": float(C),
                    "direction": direction,
                    "source_site": source_site,
                    "target_site": target_site,
                    "age_auc": float(value),
                    "target_rows_in_preprocessing_fit": int(target_rows_in_preprocessing_fit),
                    "target_labels_used": 0,
                    "convergence_warning": convergence_warning,
                    "fit_error": fit_error,
                }
            )
        values = [direction_values["A_to_B"], direction_values["B_to_A"]]
        rows.append(
            {
                "C": float(C),
                "direction": "mean",
                "source_site": f"{site_a}+{site_b}",
                "target_site": "paired",
                "age_auc": float(np.mean(values)) if np.all(np.isfinite(values)) else float("nan"),
                "target_rows_in_preprocessing_fit": 0,
                "target_labels_used": 0,
                "convergence_warning": "",
                "fit_error": "",
            }
        )

    logs = pd.DataFrame(rows)
    mean_rows = logs[logs["direction"] == "mean"].sort_values("C")
    chosen_C, ties = choose_stronger_regularized(
        mean_rows["age_auc"].to_numpy(), mean_rows["C"].to_numpy(), tie_tolerance=tie_tolerance
    )
    chosen = mean_rows[np.isclose(mean_rows["C"].to_numpy(), chosen_C)].iloc[0]
    chosen_directions = logs[logs["C"].eq(chosen_C) & logs["direction"].isin(["A_to_B", "B_to_A"])]
    by_direction = {
        str(row.direction): float(row.age_auc) for row in chosen_directions.itertuples()
    }
    return TuningResult(
        chosen_C=float(chosen_C),
        inner_auc=float(chosen.age_auc),
        inner_A_to_B_auc=float(by_direction.get("A_to_B", np.nan)),
        inner_B_to_A_auc=float(by_direction.get("B_to_A", np.nan)),
        logs=logs,
        tie_candidates=ties,
    )


def fit_source_and_score_target(
    X_source: object,
    y_source: Sequence[int],
    X_target: object,
    *,
    C: float,
    model_spec: ModelSpec,
    feature_names: Sequence[str] | None = None,
    source_subject_ids: Sequence[object] | None = None,
    target_subject_ids: Sequence[object] | None = None,
) -> tuple[FittedProbe, np.ndarray]:
    """Fit on a complete source pool and score an untouched target pool."""
    probe = fit_probe(
        X_source,
        y_source,
        spec=model_spec,
        C=float(C),
        feature_names=feature_names,
        source_subject_ids=source_subject_ids,
        target_subject_ids=target_subject_ids,
    )
    return probe, probe.decision_function(X_target)
