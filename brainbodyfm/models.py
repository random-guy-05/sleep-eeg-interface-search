"""Frozen-interface downstream probes."""

from __future__ import annotations

import warnings
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import DEFAULT_C_GRID
from .preprocess import SourcePreprocessor

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None


@dataclass(frozen=True)
class ModelSpec:
    """Specification for one representation interface."""

    name: str
    penalty: str = "l2"
    alpha: float = 0.5
    c_grid: tuple[float, ...] = DEFAULT_C_GRID
    n_components: int | None = None
    max_iter: int = 10_000
    tol: float = 1e-5
    active_threshold: float = 1e-8
    random_state: int = 2026090401
    class_weight: str | dict[int, float] | None = None

    def __post_init__(self) -> None:
        if self.penalty not in {"l2", "elasticnet", "l1"}:
            raise ValueError("penalty must be 'l2', 'elasticnet', or 'l1'.")
        if not self.c_grid or any(float(c) <= 0 for c in self.c_grid):
            raise ValueError("c_grid must contain positive values.")
        if self.penalty == "elasticnet" and not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1] for elastic net.")

    @property
    def is_sparse(self) -> bool:
        return self.penalty in {"l1", "elasticnet"}

    def make_estimator(self, C: float):
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn is required for BrainBodyFM probes.")
        kwargs = {
            "C": float(C),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "fit_intercept": True,
            "class_weight": self.class_weight,
            "random_state": int(self.random_state),
        }
        if self.penalty == "elasticnet":
            return LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=float(self.alpha), **kwargs
            )
        if self.penalty == "l1":
            return LogisticRegression(penalty="l1", solver="liblinear", **kwargs)
        return LogisticRegression(penalty="l2", solver="lbfgs", **kwargs)


@dataclass
class FittedProbe:
    """A fitted decoder plus its source-only preprocessing audit."""

    spec: ModelSpec
    chosen_C: float
    preprocessor: SourcePreprocessor
    estimator: object
    feature_names: tuple[str, ...]
    source_subject_ids: tuple[str, ...] = ()
    convergence_warnings: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def coefficients(self) -> np.ndarray:
        return np.asarray(self.estimator.coef_, dtype=float).reshape(-1)

    @property
    def intercept(self) -> float:
        return float(np.asarray(self.estimator.intercept_, dtype=float).reshape(-1)[0])

    @property
    def active_mask(self) -> np.ndarray:
        return np.abs(self.coefficients) > float(self.spec.active_threshold)

    @property
    def active_feature_names(self) -> tuple[str, ...]:
        names = self.preprocessor.output_feature_names_
        if len(names) != len(self.coefficients):
            names = self.feature_names
        return tuple(name for name, active in zip(names, self.active_mask) if active)

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    def decision_function(self, X: object) -> np.ndarray:
        transformed = self.preprocessor.transform(X)
        return np.asarray(self.estimator.decision_function(transformed), dtype=float).reshape(-1)

    def predict_proba(self, X: object) -> np.ndarray:
        transformed = self.preprocessor.transform(X)
        return np.asarray(self.estimator.predict_proba(transformed)[:, 1], dtype=float).reshape(-1)

    def coefficient_table(self) -> list[dict[str, object]]:
        names = self.preprocessor.output_feature_names_
        if len(names) != len(self.coefficients):
            names = self.feature_names
        return [
            {
                "feature": str(name),
                "coefficient": float(coef),
                "intercept": self.intercept,
                "active": bool(abs(coef) > self.spec.active_threshold),
                "chosen_C": float(self.chosen_C),
            }
            for name, coef in zip(names, self.coefficients)
        ]


def fit_probe(
    X_train: object,
    y_train: Sequence[int],
    *,
    spec: ModelSpec,
    C: float,
    feature_names: Sequence[str] | None = None,
    source_subject_ids: Sequence[object] | None = None,
    target_subject_ids: Sequence[object] | None = None,
    preprocessor: SourcePreprocessor | None = None,
) -> FittedProbe:
    """Fit one probe after source-only preprocessing."""
    y = np.asarray(y_train, dtype=int).reshape(-1)
    if len(y) == 0:
        raise ValueError("Cannot fit a probe on zero rows.")
    unique = np.unique(y)
    if len(unique) < 2:
        raise ValueError("A supervised probe requires both outcome classes.")
    matrix = np.asarray(X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if len(matrix) != len(y):
        raise ValueError("X_train and y_train have different row counts.")
    names = tuple(str(x) for x in (feature_names if feature_names is not None else [f"feature_{i}" for i in range(matrix.shape[1])]))
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length does not match X_train.")
    if preprocessor is None:
        preprocessor = SourcePreprocessor(
            n_components=spec.n_components, random_state=spec.random_state
        )
        transformed = preprocessor.fit_transform(
            matrix,
            subject_ids=source_subject_ids,
            target_subject_ids=target_subject_ids,
        )
    else:
        transformed = preprocessor.transform(matrix)
    # Preserve semantic head names for coefficient/selection reports.  PCA
    # components intentionally retain their PC1/PC2 names.
    if preprocessor.pca_ is None:
        preprocessor.output_feature_names_ = names
    estimator = spec.make_estimator(C)
    caught: list[str] = []
    cache_path = None
    cached = False
    warm_started = False
    continuation_iterations = 0
    started = time.monotonic()
    if os.environ.get("BBFM_FIT_CACHE"):
        import joblib
        import sklearn
        # Only training data and estimator settings enter the cache key.
        # Preprocessing is freshly fitted/audited above for every call.
        key = joblib.hash(("probe-v1", sklearn.__version__, transformed, y, estimator.get_params()))
        cache_path = Path(os.environ["BBFM_FIT_CACHE"]) / f"{key}.joblib"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            estimator, caught = joblib.load(cache_path)
            cached = True
        elif spec.penalty == "elasticnet":
            # Same source data, scaling and objective family; only C differs.
            # Reuse the nearest already fitted smaller-C coefficients.
            for previous_C in sorted((v for v in spec.c_grid if v < C), reverse=True):
                candidate = spec.make_estimator(previous_C)
                previous_key = joblib.hash(("probe-v1", sklearn.__version__, transformed, y, candidate.get_params()))
                previous_path = cache_path.parent / f"{previous_key}.joblib"
                if previous_path.exists():
                    previous, _ = joblib.load(previous_path)
                    estimator.coef_ = previous.coef_.copy()
                    estimator.intercept_ = previous.intercept_.copy()
                    estimator.classes_ = previous.classes_.copy()
                    estimator.set_params(warm_start=True)
                    warm_started = True
                    break
    if not cached:
        if os.environ.get("BBFM_PROGRESS"):
            print(f"FIT start {spec.name} n={len(y)} p={transformed.shape[1]} C={C:g}", flush=True)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            estimator.fit(transformed, y)
            caught.extend(str(item.message) for item in records)
    def unconverged(messages):
        return any("converg" in message.lower() and ("max_iter" in message.lower() or "failed" in message.lower()) for message in messages)
    retry = 0
    while unconverged(caught):
        retry += 1
        if retry > 4:
            raise RuntimeError(f"{spec.name} C={C} did not converge after four iteration extensions")
        estimator.set_params(warm_start=True, max_iter=spec.max_iter * (2 ** retry))
        print(f"FIT convergence retry {spec.name} n={len(y)} C={C:g} max_iter={estimator.max_iter}", flush=True)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            estimator.fit(transformed, y)
        continuation_iterations += int(np.max(estimator.n_iter_))
        caught = [str(item.message) for item in records]
        cached = False
    if not cached:
        if cache_path is not None:
            temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
            joblib.dump((estimator, caught), temporary)
            temporary.replace(cache_path)
        if os.environ.get("BBFM_PROGRESS"):
            print(f"FIT done {spec.name} n={len(y)} C={C:g} seconds={time.monotonic()-started:.2f} iterations={estimator.n_iter_.tolist()} warm_start={warm_started} retries={retry} warnings={len(caught)}", flush=True)
    return FittedProbe(
        spec=spec,
        chosen_C=float(C),
        preprocessor=preprocessor,
        estimator=estimator,
        feature_names=names,
        source_subject_ids=tuple(str(x) for x in (source_subject_ids if source_subject_ids is not None else ())),
        convergence_warnings=tuple(caught),
        metadata={"converged": not unconverged(caught), "warm_started": warm_started, "continuation_iterations": continuation_iterations},
    )


def make_model_spec(
    name: str,
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    alpha: float = 0.5,
    active_threshold: float = 1e-8,
    random_state: int = 2026090401,
    n_components: int | None = None,
    max_iter: int | None = None,
) -> ModelSpec:
    """Resolve protocol names to a shared probe specification."""
    normalized = name.strip().lower().replace("-", "_")
    sparse = "elastic" in normalized or normalized in {"l1", "pure_l1"}
    penalty = "l1" if normalized in {"l1", "pure_l1", "pure_l1_sensitivity"} else ("elasticnet" if sparse else "l2")
    return ModelSpec(
        name=name,
        penalty=penalty,
        alpha=float(alpha),
        c_grid=tuple(float(c) for c in c_grid),
        n_components=n_components,
        max_iter=int(max_iter or (20_000 if penalty in {"elasticnet", "l1"} else 10_000)),
        active_threshold=float(active_threshold),
        random_state=int(random_state),
    )
