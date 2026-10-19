"""Secondary linearly decodable institution-information analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .controls import random_feature_subsets

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    LogisticRegression = None


def _cv_balanced_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    folds: int,
    pca_components: int | None = None,
    covariates: np.ndarray | None = None,
) -> float:
    if LogisticRegression is None:
        raise RuntimeError("scikit-learn is required for site decoding.")
    counts = np.bincount(pd.factorize(y)[0])
    n_splits = min(int(folds), int(counts.min())) if len(counts) else 0
    if n_splits < 2:
        return float("nan")
    representation_width = X.shape[1]
    if covariates is not None:
        X = np.column_stack([X, covariates])
    steps: list[object] = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
    if pca_components is not None:
        if int(pca_components) <= 0 or int(pca_components) > representation_width:
            raise ValueError("pca_components must be within the representation dimension")
        steps.append(ColumnTransformer(
            [("representation_pca", PCA(n_components=int(pca_components), random_state=seed), list(range(representation_width)))],
            remainder="passthrough",
        ))
    steps.append(LogisticRegression(max_iter=10_000, solver="lbfgs", random_state=seed))
    estimator = make_pipeline(*steps)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    encoded, _ = pd.factorize(y)
    prediction = cross_val_predict(estimator, X, encoded, cv=splitter)
    return float(balanced_accuracy_score(encoded, prediction))


def site_information_analysis(
    manifest: pd.DataFrame,
    representation: object,
    *,
    feature_names: Sequence[str] | None = None,
    age_column: str = "age",
    sex_column: str = "sex",
    site_column: str = "site",
    folds: int = 5,
    seed: int = 2026090401,
    representation_name: str = "representation",
) -> pd.DataFrame:
    """Compare Age+Sex, representation, and Age+Sex+representation BA."""
    matrix = np.asarray(representation.to_numpy() if hasattr(representation, "to_numpy") else representation, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if len(matrix) != len(manifest):
        raise ValueError("representation and manifest must have equal rows")
    if feature_names is not None and len(feature_names) != matrix.shape[1]:
        raise ValueError("feature_names length does not match representation")
    age = pd.to_numeric(manifest[age_column], errors="coerce").to_numpy(dtype=float)
    sex_values = manifest[sex_column].astype(str).str.lower().map({"female": 1.0, "male": 0.0}).to_numpy(dtype=float)
    covariates = np.column_stack([age, sex_values])
    site = manifest[site_column].astype(str).to_numpy()
    values = {
        "Age+Sex": _cv_balanced_accuracy(covariates, site, seed=seed, folds=folds),
        representation_name: _cv_balanced_accuracy(matrix, site, seed=seed, folds=folds),
        "Age+Sex+" + representation_name: _cv_balanced_accuracy(
            np.column_stack([covariates, matrix]), site, seed=seed, folds=folds
        ),
    }
    baseline = values["Age+Sex"]
    rows = []
    for interface, balanced_accuracy in values.items():
        rows.append({
            "representation": representation_name,
            "interface": interface,
            "balanced_accuracy": balanced_accuracy,
            "delta_site_from_age_sex": float(balanced_accuracy - baseline) if interface.startswith("Age+Sex") else np.nan,
            "n_features": int(covariates.shape[1] if interface == "Age+Sex" else matrix.shape[1] if interface == representation_name else covariates.shape[1] + matrix.shape[1]),
            "folds": int(folds),
        })
    return pd.DataFrame(rows)


def dimension_matched_site_information_analysis(
    manifest: pd.DataFrame,
    physiology: object,
    *,
    feature_names: Sequence[str],
    selected_feature_sets: Mapping[str, Sequence[str]] | None = None,
    k_values: Sequence[int] = (4, 8, 16, 32),
    random_draws: int = 500,
    seed: int = 2026090401,
    folds: int = 5,
    age_column: str = "age",
    sex_column: str = "sex",
    site_column: str = "site",
) -> pd.DataFrame:
    """Decode site from sparse, random-k, and PCA-k physiology interfaces.

    PCA is fitted inside each site-decoding CV training fold.  Random subsets
    use the frozen random-k seed and are recorded draw-by-draw so dimensional
    opportunity is explicit rather than conflated with semantic targeting.
    """
    matrix = np.asarray(physiology.to_numpy() if hasattr(physiology, "to_numpy") else physiology, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    names = tuple(str(name) for name in feature_names)
    if len(matrix) != len(manifest) or len(names) != matrix.shape[1]:
        raise ValueError("physiology, feature_names, and manifest are not aligned")
    if random_draws <= 0:
        raise ValueError("random_draws must be positive")
    site = manifest[site_column].astype(str).to_numpy()
    age = pd.to_numeric(manifest[age_column], errors="coerce").to_numpy(dtype=float)
    sex = manifest[sex_column].astype(str).str.lower().map({"female": 1.0, "male": 0.0}).to_numpy(dtype=float)
    covariates = np.column_stack([age, sex])
    baseline = _cv_balanced_accuracy(covariates, site, seed=seed, folds=folds)
    rows: list[dict[str, object]] = [{
        "interface": "Age+Sex",
        "control_type": "covariate_baseline",
        "outer_site": "",
        "k": 2,
        "draw": -1,
        "balanced_accuracy": baseline,
        "delta_site_from_age_sex": 0.0,
    }]
    index_by_name = {name: index for index, name in enumerate(names)}
    def append_interface(label, control_type, values, k, draw=-1, outer_site="", pca_components=None):
        for adjustment in ("representation_only", "age_sex_plus_representation"):
            value = _cv_balanced_accuracy(
                values, site, seed=seed, folds=folds, pca_components=pca_components,
                covariates=covariates if adjustment == "age_sex_plus_representation" else None,
            )
            rows.append({
                "interface": label, "control_type": control_type, "outer_site": outer_site,
                "k": int(k), "draw": draw, "adjustment": adjustment,
                "balanced_accuracy": value,
                "delta_site_from_age_sex": float(value - baseline) if adjustment == "age_sex_plus_representation" else np.nan,
            })
    rows[0]["adjustment"] = "age_sex_baseline"
    for label, selected in (selected_feature_sets or {}).items():
        selected_names = tuple(str(name) for name in selected if str(name) in index_by_name)
        if not selected_names:
            continue
        indices = [index_by_name[name] for name in selected_names]
        append_interface(str(label), "selected_sparse", matrix[:, indices], len(indices),
                         outer_site=str(label).removeprefix("sparse_outer_") if str(label).startswith("sparse_outer_") else "")
    for k_raw in k_values:
        k = int(k_raw)
        if k <= 0 or k > matrix.shape[1]:
            raise ValueError(f"k={k} is outside the physiology dimension {matrix.shape[1]}")
        append_interface(f"PCA-{k}", "pca_k", matrix, k, pca_components=k)
        subsets = random_feature_subsets(
            matrix.shape[1], k, int(random_draws), seed=seed + k
        )
        for draw, indices in enumerate(subsets):
            append_interface(f"random-{k}", "random_k", matrix[:, indices], k, draw=draw)
    return pd.DataFrame(rows)
