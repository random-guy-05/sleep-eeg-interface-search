"""Shared nested site-LOSO evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import BrainBodyFMConfig, DEFAULT_CONFIG
from .metrics import age_conditioned_auroc, macro_site_auc
from .models import FittedProbe, ModelSpec
from .tuning import TuningResult, fit_source_and_score_target, tune_source_transfer


def _as_matrix(X: object) -> np.ndarray:
    matrix = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"X must be 2-D, got {matrix.shape}.")
    return matrix


def _validate_inputs(
    X: np.ndarray,
    manifest: pd.DataFrame,
    config: BrainBodyFMConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outcome_column = config.primary_outcome
    if outcome_column not in manifest.columns:
        # Accept the legacy root-pipeline name while emitting the canonical
        # ``outcome`` column in all BrainBodyFM outputs.
        if outcome_column == "outcome" and "target" in manifest.columns:
            outcome_column = "target"
        elif outcome_column == "target" and "outcome" in manifest.columns:
            outcome_column = "outcome"
    required = {
        config.subject_column,
        config.site_column,
        config.age_column,
        outcome_column,
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if len(X) != len(manifest):
        raise ValueError("X and manifest must have equal row counts.")
    subjects = manifest[config.subject_column].astype(str).to_numpy()
    if pd.Series(subjects).duplicated().any():
        duplicates = pd.Series(subjects)[pd.Series(subjects).duplicated()].unique().tolist()
        raise ValueError(f"Manifest must have one row per subject; duplicates include {duplicates[:5]}")
    sites = manifest[config.site_column].astype(str).to_numpy()
    unknown = sorted(set(sites).difference(config.sites))
    if unknown:
        raise ValueError(f"Manifest contains sites outside configured folds: {unknown}")
    absent = sorted(set(config.sites).difference(sites))
    if absent:
        raise ValueError(f"Manifest is missing configured sites: {absent}")
    y = pd.to_numeric(manifest[outcome_column], errors="raise").astype(int).to_numpy()
    if np.any(~np.isin(y, (0, 1))):
        raise ValueError("Outcome column must contain binary 0/1 values.")
    ages = pd.to_numeric(manifest[config.age_column], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(ages)):
        raise ValueError("Age column must be finite for every subject.")
    return subjects, sites, y, ages


@dataclass
class FoldResult:
    outer_site: str
    source_sites: tuple[str, ...]
    chosen_C: float
    inner_A_to_B_auc: float
    inner_B_to_A_auc: float
    inner_auc: float
    outer_auc: float
    n_active: int
    selected_features: tuple[str, ...]
    coefficients: tuple[float, ...]
    convergence_warnings: tuple[str, ...] = ()
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    tuning_logs: pd.DataFrame = field(default_factory=pd.DataFrame)
    preprocessing_fit_subject_ids: tuple[str, ...] = ()
    target_subject_ids: tuple[str, ...] = ()
    target_labels_used_in_hyperparameter_selection: int = 0
    target_labels_used_in_sparse_selection: int = 0

    @property
    def chosen_hyperparameter(self) -> float:
        """Protocol-neutral alias for the selected regularization value."""
        return self.chosen_C

    @property
    def inner_mean_auc(self) -> float:
        """Alias matching the source-transfer notation in the protocol."""
        return self.inner_auc

    @property
    def target_rows_in_preprocessing_fit(self) -> int:
        return len(set(self.preprocessing_fit_subject_ids).intersection(self.target_subject_ids))


@dataclass
class NestedEvaluationResult:
    """All machine-readable outputs for a nested site-transfer benchmark."""

    representation: str
    predictions: pd.DataFrame
    folds: list[FoldResult]
    tuning_logs: pd.DataFrame
    coefficients: pd.DataFrame
    config: BrainBodyFMConfig

    @property
    def macro_auc(self) -> float:
        if self.predictions.empty:
            return float("nan")
        return macro_site_auc(
            self.predictions["outcome"],
            self.predictions["score"],
            self.predictions["age"],
            self.predictions["site"],
            site_order=self.config.sites,
            age_window=self.config.age_window,
            strict=True,
        )

    @property
    def selected_feature_count(self) -> float:
        if not self.folds:
            return float("nan")
        return float(np.mean([fold.n_active for fold in self.folds]))

    def summary(self) -> pd.DataFrame:
        input_dimensionality = int(len(self.folds[0].coefficients)) if self.folds else 0
        is_sparse = "elastic_net" in self.representation or "pure_l1" in self.representation
        effective_dimensionality = self.selected_feature_count if is_sparse else float(input_dimensionality)
        rows = [
            {
                "representation": self.representation,
                "dimensionality": effective_dimensionality,
                "input_dimensionality": input_dimensionality,
                "effective_dimensionality": effective_dimensionality,
                "macro_age_auc": self.macro_auc,
                "mean_n_active": self.selected_feature_count,
                "n_outer_folds": len(self.folds),
            }
        ]
        return pd.DataFrame(rows)

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_table(self.predictions, out / "outer_predictions.parquet")
        _write_table(self.tuning_logs, out / "tuning_logs.parquet")
        _write_table(self.coefficients, out / "elasticnet_coefficients.parquet")
        self.summary().to_csv(out / "outer_summary.csv", index=False)
        fold_rows = []
        for fold in self.folds:
            fold_rows.append(
                {
                    "outer_site": fold.outer_site,
                    "chosen_C": fold.chosen_C,
                    "inner_A_to_B_auc": fold.inner_A_to_B_auc,
                    "inner_B_to_A_auc": fold.inner_B_to_A_auc,
                    "inner_auc": fold.inner_auc,
                    "outer_auc": fold.outer_auc,
                    "n_active": fold.n_active,
                    "target_rows_in_preprocessing_fit": fold.target_rows_in_preprocessing_fit,
                    "target_subjects_in_scaler_fit": fold.target_rows_in_preprocessing_fit,
                    "target_subjects_in_pca_fit": fold.target_rows_in_preprocessing_fit,
                    "target_labels_used_in_hyperparameter_selection": fold.target_labels_used_in_hyperparameter_selection,
                    "target_labels_used_in_sparse_selection": fold.target_labels_used_in_sparse_selection,
                    "convergence_warning": " | ".join(fold.convergence_warnings),
                }
            )
        pd.DataFrame(fold_rows).to_csv(out / "fold_summary.csv", index=False)


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ValueError, OSError):
        # Keep the requested basename available even in a minimal environment.
        frame.to_csv(path.with_suffix(".csv"), index=False)


def run_nested_loso(
    X: object,
    manifest: pd.DataFrame,
    *,
    model_spec: ModelSpec,
    representation: str = "representation",
    feature_names: Sequence[str] | None = None,
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
    hyperparameter_grid: Sequence[float] | None = None,
) -> NestedEvaluationResult:
    """Run A+B→C for every configured target site.

    The inner tuner sees only the two source sites.  Once C is chosen, a fresh
    preprocessor and decoder are fit on all A+B and C is scored exactly once.
    """
    matrix = _as_matrix(X)
    if hyperparameter_grid is not None:
        model_spec = ModelSpec(
            name=model_spec.name,
            penalty=model_spec.penalty,
            alpha=model_spec.alpha,
            c_grid=tuple(float(c) for c in hyperparameter_grid),
            n_components=model_spec.n_components,
            max_iter=model_spec.max_iter,
            tol=model_spec.tol,
            active_threshold=model_spec.active_threshold,
            random_state=model_spec.random_state,
            class_weight=model_spec.class_weight,
        )
    subjects, sites, y, ages = _validate_inputs(matrix, manifest, config)
    names = tuple(str(x) for x in (feature_names if feature_names is not None else [f"feature_{j}" for j in range(matrix.shape[1])]))
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length does not match X.")
    all_predictions: list[pd.DataFrame] = []
    all_tuning: list[pd.DataFrame] = []
    all_coefficients: list[dict[str, object]] = []
    fold_results: list[FoldResult] = []

    for target_site in config.sites:
        target_idx = np.flatnonzero(sites == str(target_site))
        source_sites = tuple(site for site in config.sites if site != target_site)
        source_idx = np.flatnonzero(np.isin(sites, source_sites))
        if len(target_idx) == 0:
            raise ValueError(f"Configured target site {target_site!r} has no rows.")
        if len(source_idx) == 0:
            raise ValueError(f"No source rows available for target site {target_site!r}.")
        source_ids = subjects[source_idx]
        target_ids = subjects[target_idx]
        if set(source_ids).intersection(target_ids):
            raise AssertionError("A subject appears in both source and target outer pools.")

        tuning = tune_source_transfer(
            matrix[source_idx],
            y[source_idx],
            ages[source_idx],
            sites[source_idx],
            model_spec=model_spec,
            feature_names=names,
            subject_ids=source_ids,
            age_window=config.age_window,
            tie_tolerance=config.tie_tolerance,
        )
        tune_log = tuning.logs.copy()
        tune_log.insert(0, "representation", representation)
        tune_log.insert(1, "outer_site", target_site)
        all_tuning.append(tune_log)
        probe, scores = fit_source_and_score_target(
            matrix[source_idx],
            y[source_idx],
            matrix[target_idx],
            C=tuning.chosen_C,
            model_spec=model_spec,
            feature_names=names,
            source_subject_ids=source_ids,
            target_subject_ids=target_ids,
        )
        outer_auc = age_conditioned_auroc(
            y[target_idx], scores, ages[target_idx], config.age_window
        )
        pred = pd.DataFrame(
            {
                "subject_id": target_ids,
                "site": sites[target_idx],
                "age": ages[target_idx],
                "outcome": y[target_idx],
                "representation": representation,
                "score": scores,
                "outer_site": target_site,
            }
        )
        all_predictions.append(pred)
        for row in probe.coefficient_table():
            row.update({"representation": representation, "outer_site": target_site})
            all_coefficients.append(row)
        fold_results.append(
            FoldResult(
                outer_site=str(target_site),
                source_sites=source_sites,
                chosen_C=tuning.chosen_C,
                inner_A_to_B_auc=tuning.inner_A_to_B_auc,
                inner_B_to_A_auc=tuning.inner_B_to_A_auc,
                inner_auc=tuning.inner_auc,
                outer_auc=float(outer_auc),
                n_active=probe.n_active,
                selected_features=probe.active_feature_names,
                coefficients=tuple(float(x) for x in probe.coefficients),
                convergence_warnings=probe.convergence_warnings,
                predictions=pred,
                tuning_logs=tune_log,
                preprocessing_fit_subject_ids=source_ids.tolist(),
                target_subject_ids=target_ids.tolist(),
            )
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    tuning_logs = pd.concat(all_tuning, ignore_index=True) if all_tuning else pd.DataFrame()
    coefficients = pd.DataFrame(all_coefficients)
    return NestedEvaluationResult(
        representation=representation,
        predictions=predictions,
        folds=fold_results,
        tuning_logs=tuning_logs,
        coefficients=coefficients,
        config=config,
    )


def run_fixed_interface_loso(
    X: object,
    manifest: pd.DataFrame,
    *,
    weights: Sequence[float],
    directions: Sequence[float] | None = None,
    representation: str = "stable4_fixed",
    feature_names: Sequence[str] | None = None,
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
    normalization_mean: Sequence[float] | None = None,
    normalization_scale: Sequence[float] | None = None,
    imputation_median: Sequence[float] | None = None,
    normalization_by_outer_site: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    normalization_fit_subject_ids_by_outer_site: Mapping[str, Sequence[object]] | None = None,
) -> NestedEvaluationResult:
    """Evaluate a previously learned, label-frozen linear interface.

    Mean/SD are fit on A+B separately for each outer fold; the supplied
    coefficients, signs, and outcome are never re-estimated.
    """
    matrix = _as_matrix(X)
    subjects, sites, y, ages = _validate_inputs(matrix, manifest, config)
    weights_arr = np.asarray(weights, dtype=float).reshape(-1)
    if len(weights_arr) != matrix.shape[1]:
        raise ValueError("weights length must equal the number of columns in X.")
    if not np.all(np.isfinite(weights_arr)):
        raise ValueError("weights must be finite.")
    direction_arr = np.ones_like(weights_arr) if directions is None else np.asarray(directions, dtype=float).reshape(-1)
    if len(direction_arr) != len(weights_arr):
        raise ValueError("directions length must equal weights length.")
    if not np.all(np.isfinite(direction_arr)):
        raise ValueError("directions must be finite.")
    names = tuple(str(x) for x in (feature_names if feature_names is not None else [f"feature_{j}" for j in range(matrix.shape[1])]))
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length does not match X.")
    frozen_mean = None if normalization_mean is None else np.asarray(normalization_mean, dtype=float).reshape(-1)
    frozen_scale = None if normalization_scale is None else np.asarray(normalization_scale, dtype=float).reshape(-1)
    frozen_median = None if imputation_median is None else np.asarray(imputation_median, dtype=float).reshape(-1)
    if (frozen_mean is None) != (frozen_scale is None):
        raise ValueError("normalization_mean and normalization_scale must be supplied together.")
    if normalization_by_outer_site is not None and frozen_mean is not None:
        raise ValueError("Use either global frozen normalization or per-site normalization, not both.")
    if frozen_mean is not None and (len(frozen_mean) != matrix.shape[1] or len(frozen_scale) != matrix.shape[1]):
        raise ValueError("frozen normalization vectors must match the number of columns in X.")
    if frozen_median is not None and len(frozen_median) != matrix.shape[1]:
        raise ValueError("imputation_median must match the number of columns in X.")
    if frozen_mean is not None:
        if not np.all(np.isfinite(frozen_mean)) or not np.all(np.isfinite(frozen_scale)):
            raise ValueError("frozen normalization vectors must be finite.")
        if np.any(frozen_scale <= 0):
            raise ValueError("frozen normalization scales must be positive.")
    folds: list[FoldResult] = []
    predictions: list[pd.DataFrame] = []
    coeff_rows: list[dict[str, object]] = []
    for target_site in config.sites:
        target_idx = np.flatnonzero(sites == target_site)
        source_idx = np.flatnonzero(sites != target_site)
        if len(target_idx) == 0:
            raise ValueError(f"Configured target site {target_site!r} has no rows.")
        if len(source_idx) == 0:
            raise ValueError(f"No source rows available for target site {target_site!r}.")
        source_ids = subjects[source_idx]
        target_ids = subjects[target_idx]
        if set(source_ids).intersection(target_ids):
            raise AssertionError("A subject appears in both source and target pools.")
        preprocessing_fit_ids: Sequence[object] = source_ids
        if normalization_by_outer_site is not None:
            if target_site not in normalization_by_outer_site:
                raise ValueError(f"No frozen normalization supplied for outer site {target_site!r}.")
            rule = normalization_by_outer_site[target_site]
            try:
                medians = np.asarray(rule["median"], dtype=float).reshape(-1)
                mean = np.asarray(rule["mean"], dtype=float).reshape(-1)
                scale = np.asarray(rule["scale"], dtype=float).reshape(-1)
            except KeyError as exc:
                raise ValueError(f"Frozen normalization for {target_site!r} needs median, mean, and scale.") from exc
            preprocessing_fit_ids = (
                (normalization_fit_subject_ids_by_outer_site or {}).get(target_site, ())
            )
        else:
            finite = np.isfinite(matrix[source_idx])
            medians = np.zeros(matrix.shape[1], dtype=float)
            for j in range(matrix.shape[1]):
                values = matrix[source_idx, j][finite[:, j]]
                medians[j] = np.median(values) if len(values) else 0.0
            if frozen_median is not None:
                medians = frozen_median
            source_filled = np.where(np.isfinite(matrix[source_idx]), matrix[source_idx], medians)
            mean = source_filled.mean(axis=0) if frozen_mean is None else frozen_mean
            scale = source_filled.std(axis=0) if frozen_scale is None else frozen_scale
            if frozen_scale is None:
                scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
        if not (len(medians) == len(mean) == len(scale) == matrix.shape[1]):
            raise ValueError("Frozen normalization vectors do not match X.")
        if not np.all(np.isfinite(medians)) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("Frozen normalization vectors must be finite with positive scales.")
        if set(map(str, preprocessing_fit_ids)).intersection(target_ids):
            raise AssertionError("Target subjects were used to fit frozen-interface normalization.")
        target_filled = np.where(np.isfinite(matrix[target_idx]), matrix[target_idx], medians)
        scores = ((target_filled - mean) / scale) @ (weights_arr * direction_arr)
        outer_auc = age_conditioned_auroc(y[target_idx], scores, ages[target_idx], config.age_window)
        pred = pd.DataFrame({
            "subject_id": target_ids,
            "site": sites[target_idx],
            "age": ages[target_idx],
            "outcome": y[target_idx],
            "representation": representation,
            "score": scores,
            "outer_site": target_site,
        })
        predictions.append(pred)
        active = np.abs(weights_arr) > config.stable4_active_threshold
        for name, weight, sign, is_active in zip(names, weights_arr, direction_arr, active):
            coeff_rows.append({"representation": representation, "outer_site": target_site, "feature": name, "coefficient": float(weight * sign), "intercept": 0.0, "active": bool(is_active)})
        folds.append(FoldResult(
            outer_site=target_site,
            source_sites=tuple(site for site in config.sites if site != target_site),
            chosen_C=float("nan"),
            inner_A_to_B_auc=float("nan"),
            inner_B_to_A_auc=float("nan"),
            inner_auc=float("nan"),
            outer_auc=float(outer_auc),
            n_active=int(np.count_nonzero(active)),
            selected_features=tuple(name for name, flag in zip(names, active) if flag),
            coefficients=tuple(float(x) for x in weights_arr * direction_arr),
            predictions=pred,
            preprocessing_fit_subject_ids=tuple(str(value) for value in preprocessing_fit_ids),
            target_subject_ids=target_ids.tolist(),
        ))
    return NestedEvaluationResult(
        representation=representation,
        predictions=pd.concat(predictions, ignore_index=True),
        folds=folds,
        tuning_logs=pd.DataFrame(),
        coefficients=pd.DataFrame(coeff_rows),
        config=config,
    )


def fit_source_normalization_by_outer_site(
    X: object,
    manifest: pd.DataFrame,
    *,
    config: BrainBodyFMConfig = DEFAULT_CONFIG,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, tuple[str, ...]]]:
    """Freeze imputation/normalization rules from a reference cohort only."""
    matrix = _as_matrix(X)
    if len(matrix) != len(manifest):
        raise ValueError("X and reference manifest must have equal row counts.")
    required = {config.subject_column, config.site_column}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Reference manifest is missing columns: {missing}")
    subjects = manifest[config.subject_column].astype(str).to_numpy()
    if pd.Series(subjects).duplicated().any():
        raise ValueError("Reference normalization requires one row per subject.")
    sites = manifest[config.site_column].astype(str).to_numpy()
    if set(sites) != set(config.sites):
        raise ValueError("Reference normalization cohort must contain every configured site and no others.")
    rules: dict[str, dict[str, np.ndarray]] = {}
    fit_ids: dict[str, tuple[str, ...]] = {}
    for target_site in config.sites:
        source_idx = np.flatnonzero((sites != str(target_site)) & np.isin(sites, config.sites))
        if len(source_idx) == 0:
            raise ValueError(f"No reference source rows for outer site {target_site!r}.")
        source = matrix[source_idx]
        finite = np.isfinite(source)
        medians = np.zeros(source.shape[1], dtype=float)
        for column in range(source.shape[1]):
            values = source[finite[:, column], column]
            medians[column] = np.median(values) if len(values) else 0.0
        filled = np.where(finite, source, medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
        rules[str(target_site)] = {"median": medians, "mean": means, "scale": scales}
        fit_ids[str(target_site)] = tuple(subjects[source_idx])
    return rules, fit_ids
