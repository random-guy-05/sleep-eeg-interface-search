"""Dimension-matched random and PCA controls plus the sparse path."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import BrainBodyFMConfig, DEFAULT_CONFIG
from .evaluation import NestedEvaluationResult, run_nested_loso
from .models import ModelSpec, fit_probe, make_model_spec
from .preprocess import SourcePreprocessor
from .tuning import tune_source_transfer


def random_feature_subsets(
    n_features: int,
    k: int,
    n_draws: int,
    *,
    seed: int,
) -> list[np.ndarray]:
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if k <= 0 or k > n_features:
        raise ValueError(f"k must be in [1, {n_features}], got {k}")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_features, size=k, replace=False)) for _ in range(n_draws)]


def run_random_k_controls(
    X: object,
    manifest: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    k_values: Iterable[int] = (4, 8, 16, 32),
    n_draws: int = 500,
    seed: int = 2026090401,
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
    max_draws: int | None = None,
    draw_indices: Iterable[int] | None = None,
    retain_results: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[int, int], NestedEvaluationResult]]:
    """Evaluate dimension-matched random physiology subsets.

    The same ridge probe and nested source-transfer machinery are used for each
    draw.  Returning the result objects allows callers to retain predictions for
    the empirical-null tail statistic without serializing huge arrays here.
    """
    matrix = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    names = tuple(str(name) for name in feature_names)
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length does not match X")
    summaries: list[dict[str, object]] = []
    results: dict[tuple[int, int], NestedEvaluationResult] = {}
    draw_limit = max_draws if max_draws is not None else n_draws
    if draw_limit <= 0:
        raise ValueError("max_draws must be positive")
    if draw_indices is None:
        selected_draws = tuple(range(min(int(draw_limit), int(n_draws))))
    else:
        selected_draws = tuple(int(value) for value in draw_indices)
        if max_draws is not None:
            selected_draws = selected_draws[: int(max_draws)]
        if not selected_draws or any(value < 0 or value >= int(n_draws) for value in selected_draws):
            raise ValueError("draw_indices must be non-empty values in [0, n_draws)")
        if len(set(selected_draws)) != len(selected_draws):
            raise ValueError("draw_indices must be unique")
    for k in k_values:
        subsets = random_feature_subsets(matrix.shape[1], int(k), int(n_draws), seed=seed + int(k))
        for draw in selected_draws:
            subset = subsets[draw]
            selected_names = tuple(names[index] for index in subset)
            spec = make_model_spec(
                f"random_{k}_{draw}_ridge",
                c_grid=config.c_grid,
                alpha=config.alpha,
                active_threshold=config.active_threshold,
                random_state=config.random_state + draw,
                max_iter=config.max_iter_ridge,
            )
            result = run_nested_loso(
                matrix[:, subset],
                manifest,
                model_spec=spec,
                representation=f"random_{k}_{draw}",
                feature_names=selected_names,
                config=config,
            )
            if retain_results:
                results[(int(k), int(draw))] = result
            summaries.append({
                "k": int(k),
                "draw": int(draw),
                "seed": int(seed + int(k)),
                "macro_age_auc": result.macro_auc,
                "n_features": int(k),
                "target_site_labels_used": False,
                "evidence_section": "properly_nested_interface",
            })
    return pd.DataFrame(summaries), results


def summarize_random_k_controls(
    draws: pd.DataFrame,
    *,
    observed_auc: float | None = None,
) -> pd.DataFrame:
    """Summarize the prespecified dimension-matched empirical null."""
    required = {"k", "draw", "macro_age_auc"}
    missing = sorted(required.difference(draws.columns))
    if missing:
        raise ValueError(f"random-k draws missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for k, group in draws.groupby("k", sort=True):
        values = pd.to_numeric(group["macro_age_auc"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        row: dict[str, object] = {
            "k": int(k),
            "n_draws": int(len(group)),
            "n_finite": int(len(values)),
            "median_macro_age_auc": float(np.median(values)) if len(values) else float("nan"),
            "q25_macro_age_auc": float(np.quantile(values, 0.25)) if len(values) else float("nan"),
            "q75_macro_age_auc": float(np.quantile(values, 0.75)) if len(values) else float("nan"),
            "p95_macro_age_auc": float(np.quantile(values, 0.95)) if len(values) else float("nan"),
        }
        if observed_auc is not None and np.isfinite(observed_auc) and len(values):
            row["observed_auc"] = float(observed_auc)
            row["empirical_upper_tail"] = float(
                (1 + np.count_nonzero(values >= float(observed_auc))) / (len(values) + 1)
            )
            row["observed_percentile"] = float(100 * np.mean(values <= float(observed_auc)))
        else:
            row["observed_auc"] = float("nan")
            row["empirical_upper_tail"] = float("nan")
            row["observed_percentile"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def run_pca_controls(
    X: object,
    manifest: pd.DataFrame,
    *,
    k_values: Iterable[int] = (4, 8, 16, 32),
    feature_names: Sequence[str] | None = None,
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict[int, NestedEvaluationResult]]:
    """Evaluate unsupervised physiology PCA at prespecified dimensions."""
    matrix = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    names = tuple(str(x) for x in (feature_names if feature_names is not None else [f"feature_{i}" for i in range(matrix.shape[1])]))
    summaries: list[dict[str, object]] = []
    results: dict[int, NestedEvaluationResult] = {}
    for k in k_values:
        spec = make_model_spec(
            f"pca_{k}_ridge",
            c_grid=config.c_grid,
            n_components=int(k),
            random_state=config.random_state,
            max_iter=config.max_iter_ridge,
        )
        result = run_nested_loso(
            matrix,
            manifest,
            model_spec=spec,
            representation=f"pca_{k}",
            feature_names=names,
            config=config,
        )
        results[int(k)] = result
        summaries.append({
            "k": int(k),
            "effective_dimensionality": int(len(result.folds[0].coefficients)) if result.folds else 0,
            "macro_age_auc": result.macro_auc,
            "target_site_labels_used": False,
            "evidence_section": "properly_nested_interface",
        })
    return pd.DataFrame(summaries), results


def sparse_regularization_path(
    X: object,
    manifest: pd.DataFrame,
    *,
    model_spec: ModelSpec,
    feature_names: Sequence[str] | None = None,
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Describe ``(C, k_C, V(C))`` using source validation only."""
    matrix = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    names = tuple(str(x) for x in (feature_names if feature_names is not None else [f"feature_{i}" for i in range(matrix.shape[1])]))
    sites = manifest[config.site_column].astype(str).to_numpy()
    labels = pd.to_numeric(manifest[config.primary_outcome]).astype(int).to_numpy()
    ages = pd.to_numeric(manifest[config.age_column]).to_numpy(dtype=float)
    subjects = manifest[config.subject_column].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for target_site in config.sites:
        target_idx = np.flatnonzero(sites == target_site)
        source_sites = tuple(site for site in config.sites if site != target_site)
        source_idx = np.flatnonzero(np.isin(sites, source_sites))
        for C in model_spec.c_grid:
            one_c_spec = ModelSpec(
                name=model_spec.name,
                penalty=model_spec.penalty,
                alpha=model_spec.alpha,
                c_grid=(float(C),),
                n_components=model_spec.n_components,
                max_iter=model_spec.max_iter,
                tol=model_spec.tol,
                active_threshold=model_spec.active_threshold,
                random_state=model_spec.random_state,
                class_weight=model_spec.class_weight,
            )
            tuning = tune_source_transfer(
                matrix[source_idx], labels[source_idx], ages[source_idx], sites[source_idx],
                model_spec=one_c_spec,
                feature_names=names,
                subject_ids=subjects[source_idx],
                age_window=config.age_window,
                tie_tolerance=config.tie_tolerance,
            )
            # Fit once on all sources to report the natural selected
            # coordinate count.  The descriptive path is source-only and does
            # not even transform the held-out target features.
            probe = fit_probe(
                matrix[source_idx], labels[source_idx], C=float(C),
                spec=model_spec, feature_names=names,
                source_subject_ids=subjects[source_idx],
            )
            rows.append({
                "outer_site": target_site,
                "C": float(C),
                "k_C": int(probe.n_active),
                "V_C": float(tuning.inner_auc),
                "inner_A_to_B_auc": tuning.inner_A_to_B_auc,
                "inner_B_to_A_auc": tuning.inner_B_to_A_auc,
            })
    return pd.DataFrame(rows)
